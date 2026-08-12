from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import butter, sosfilt, stft

from .audio import read_audio, read_mono, write_wav
from .config import CorrectionConfig
from .director import local_llm_profile, profile_from_text
from .models import (
    AnalysisReport,
    ChordEvent,
    CreativeBrief,
    InputMode,
    Metric,
    NoteEvent,
    OutputFile,
    TuningControls,
    TuningPlan,
)
from .pitch import (
    estimate_pitch,
    estimate_pitch_quality,
    hz_to_midi,
    midi_to_hz,
    smooth_pitch,
    target_pitch,
)
from .psola import pitch_shift_psola
from .quality import QualityModelError, QualityRunner
from .song import detect_key
from .storage import StudioStore

Progress = Callable[[str, float, str], None]
SUPPORTED_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".aac"}
SEPARATOR_MODEL = "vocals_mel_band_roformer.ckpt"
MIN_HEAVY_MEMORY_PERCENT = 55
NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, capture_output=True, text=True)


def _run_with_progress(command: list[str], progress: Progress) -> None:
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    tick = 0
    try:
        while process.poll() is None:
            time.sleep(0.25)
            tick += 1
            progress(
                "separating",
                min(0.31, 0.16 + tick * 0.002),
                "Separating locally · cancel remains available",
            )
    except BaseException:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
        raise
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command)


def select_separator_stems(candidates: list[Path]) -> tuple[Path | None, Path | None]:
    vocals = [path for path in candidates if "(vocals)" in path.name.lower()]
    if not vocals:
        vocals = [path for path in candidates if "vocal" in path.stem.lower()]
    instrumentals = [
        path
        for path in candidates
        if any(word in path.name.lower() for word in ("(other)", "instrument", "no_vocal"))
    ]
    return (vocals[0] if vocals else None, instrumentals[0] if instrumentals else None)


def memory_free_percent() -> int | None:
    tool = shutil.which("memory_pressure")
    if not tool:
        return None
    try:
        result = _run([tool])
        marker = "System-wide memory free percentage:"
        line = next(line for line in result.stdout.splitlines() if marker in line)
        return int(line.split(marker, 1)[1].strip().rstrip("%"))
    except (OSError, StopIteration, ValueError, subprocess.SubprocessError):
        return None


def low_memory_mode() -> bool:
    return os.getenv("CUTETUNER_LOW_MEMORY", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def decode_audio(source: Path, destination: Path) -> None:
    if source.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported audio format: {source.suffix}")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg is required to decode audio.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-vn",
            "-ar",
            "44100",
            "-c:a",
            "pcm_f32le",
            str(destination),
        ]
    )


def encode_mp3(source: Path, destination: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg is required to export MP3 files.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-map_metadata",
            "-1",
            "-c:a",
            "libmp3lame",
            "-b:a",
            "320k",
            str(destination),
        ]
    )


def _db(value: float) -> float:
    return 20 * math.log10(max(value, 1e-9))


def _technical_metrics(audio: np.ndarray, sr: int) -> dict[str, Metric]:
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    peak = float(np.max(np.abs(mono))) if len(mono) else 0.0
    rms = float(np.sqrt(np.mean(np.square(mono)))) if len(mono) else 0.0
    block = max(1, int(sr * 0.05))
    frame_rms = np.array(
        [
            np.sqrt(np.mean(np.square(mono[index : index + block])))
            for index in range(0, max(1, len(mono) - block + 1), block)
        ]
    )
    noise = float(np.percentile(frame_rms, 15)) if len(frame_rms) else 0.0
    frequencies, _, spectrum = stft(mono, fs=sr, nperseg=2048, noverlap=1536, boundary=None)
    magnitudes = np.abs(spectrum)
    mean_spectrum = magnitudes.mean(axis=1) if magnitudes.size else np.zeros_like(frequencies)
    centroid = float(np.sum(frequencies * mean_spectrum) / max(np.sum(mean_spectrum), 1e-9))
    total_energy = float(np.sum(np.square(mean_spectrum))) + 1e-9
    sibilance = float(
        np.sum(np.square(mean_spectrum[(frequencies >= 5_000) & (frequencies <= 10_000)]))
        / total_energy
    )
    clipping = float(np.mean(np.abs(mono) >= 0.999) * 100) if len(mono) else 0.0
    dynamic = (
        _db(float(np.percentile(frame_rms, 90))) - _db(float(np.percentile(frame_rms, 20)))
        if len(frame_rms)
        else 0.0
    )
    return {
        "peak": Metric(value=round(_db(peak), 2), unit="dBFS", confidence=1, source="DSP"),
        "rms": Metric(value=round(_db(rms), 2), unit="dBFS", confidence=1, source="DSP"),
        "noise_floor": Metric(
            value=round(_db(noise), 2), unit="dBFS", confidence=0.65, source="DSP proxy"
        ),
        "dynamic_range": Metric(value=round(dynamic, 2), unit="dB", confidence=0.8, source="DSP"),
        "spectral_centroid": Metric(
            value=round(centroid), unit="Hz", confidence=0.85, source="DSP"
        ),
        "sibilance_ratio": Metric(value=round(sibilance, 4), confidence=0.65, source="DSP proxy"),
        "clipped_samples": Metric(value=round(clipping, 4), unit="%", confidence=1, source="DSP"),
    }


def _pitch_notes(times: np.ndarray, f0: np.ndarray, confidence: np.ndarray) -> list[NoteEvent]:
    if not len(times):
        return []
    voiced = f0 > 0
    midi = hz_to_midi(np.where(voiced, f0, 440.0))
    notes: list[NoteEvent] = []
    start = 0
    while start < len(times):
        if not voiced[start]:
            start += 1
            continue
        center = round(float(midi[start]))
        end = start + 1
        while end < len(times) and voiced[end] and abs(float(midi[end]) - center) < 0.9:
            end += 1
        duration = float(times[end - 1] - times[start] + 0.01)
        if duration >= 0.07:
            note_midi = float(np.median(midi[start:end]))
            notes.append(
                NoteEvent(
                    start=max(0, round(float(times[start] - 0.023), 3)),
                    end=round(float(times[end - 1] + 0.028), 3),
                    midi=round(note_midi, 2),
                    target_midi=round(note_midi),
                    confidence=round(float(np.mean(confidence[start:end])), 3),
                )
            )
        start = end
    return notes[:2_000]


def _chord_pitch_classes(label: str) -> set[int]:
    root_name = label.removesuffix("m")
    if root_name not in NOTE_NAMES:
        return set()
    root = NOTE_NAMES.index(root_name)
    third = 3 if label.endswith("m") else 4
    return {root, (root + third) % 12, (root + 7) % 12}


def chord_aware_targets(
    notes: list[NoteEvent],
    chords: list[ChordEvent],
    config: CorrectionConfig,
    max_correction_cents: float,
) -> list[float]:
    """Select a coherent note sequence using pitch, harmony, and interval continuity."""
    if not notes:
        return []
    candidate_rows: list[np.ndarray] = []
    local_costs: list[np.ndarray] = []
    for note in notes:
        lower = math.floor(note.midi - max_correction_cents / 100)
        upper = math.ceil(note.midi + max_correction_cents / 100)
        candidates = np.arange(max(12, lower), min(127, upper) + 1, dtype=float)
        scale_penalty = np.array(
            [0 if int(value) % 12 in config.allowed_pcs else 2.2 for value in candidates]
        )
        midpoint = (note.start + note.end) / 2
        active = next((chord for chord in chords if chord.start <= midpoint < chord.end), None)
        chord_pcs = _chord_pitch_classes(active.label) if active else set()
        chord_penalty = np.array(
            [0 if not chord_pcs or int(value) % 12 in chord_pcs else 0.55 for value in candidates]
        )
        movement = np.square(candidates - note.midi) * 1.15
        duration_weight = min(2.0, max(0.65, note.end - note.start))
        candidate_rows.append(candidates)
        local_costs.append((movement + scale_penalty + chord_penalty) * duration_weight)
    accumulated = local_costs[0].copy()
    backpointers: list[np.ndarray] = []
    for index in range(1, len(notes)):
        previous = candidate_rows[index - 1]
        current = candidate_rows[index]
        observed_interval = notes[index].midi - notes[index - 1].midi
        transition = np.abs((current[:, None] - previous[None, :]) - observed_interval) * 0.16
        combined = transition + accumulated[None, :]
        best_previous = np.argmin(combined, axis=1)
        accumulated = local_costs[index] + combined[np.arange(len(current)), best_previous]
        backpointers.append(best_previous)
    selected = [int(np.argmin(accumulated))]
    for pointers in reversed(backpointers):
        selected.append(int(pointers[selected[-1]]))
    selected.reverse()
    return [float(candidate_rows[index][choice]) for index, choice in enumerate(selected)]


def _chroma(audio: np.ndarray, sr: int) -> np.ndarray:
    frequencies, _, spectrum = stft(audio, fs=sr, nperseg=4096, noverlap=3072, boundary=None)
    magnitudes = np.abs(spectrum).mean(axis=1)
    usable = (frequencies >= 55) & (frequencies <= 4_500)
    notes = np.round(69 + 12 * np.log2(np.maximum(frequencies[usable], 1e-7) / 440)).astype(int)
    values = np.zeros(12)
    np.add.at(values, notes % 12, magnitudes[usable])
    return values / max(float(np.linalg.norm(values)), 1e-9)


def _chord_timeline(audio: np.ndarray, sr: int) -> list[ChordEvent]:
    names = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
    candidates = [
        (root, mode, (root, (root + third) % 12, (root + 7) % 12))
        for root in range(12)
        for mode, third in (("", 4), ("m", 3))
    ]
    window = sr * 4
    events: list[ChordEvent] = []
    for start in range(0, len(audio), window):
        segment = audio[start : start + window]
        if len(segment) < sr:
            continue
        chroma = _chroma(segment, sr)
        scores = [
            (float(sum(chroma[pitch] for pitch in chord)), root, mode)
            for root, mode, chord in candidates
        ]
        scores.sort(reverse=True)
        best, second = scores[0], scores[1]
        events.append(
            ChordEvent(
                start=round(start / sr, 3),
                end=round(min(len(audio), start + window) / sr, 3),
                label=f"{names[best[1]]}{best[2]}",
                confidence=round(max(0.0, (best[0] - second[0]) / max(best[0], 1e-9)), 3),
            )
        )
    return events


def _semantic_descriptors(technical: dict[str, Metric], vocal: dict[str, Metric]) -> list[str]:
    descriptors: list[str] = []
    centroid = float(technical["spectral_centroid"].value or 0)
    dynamic = float(technical["dynamic_range"].value or 0)
    pitch_span = float(vocal["pitch_span"].value or 0)
    descriptors.append(
        "dark and rounded"
        if centroid < 1_600
        else "clear and present"
        if centroid < 2_800
        else "bright and airy"
    )
    descriptors.append("intimate dynamics" if dynamic < 12 else "expressive dynamics")
    descriptors.append("focused melodic range" if pitch_span < 8 else "wide melodic range")
    return descriptors


def _naturalness_proxy(technical: dict[str, Metric], pitch_confidence: float) -> Metric:
    clipping_penalty = min(1.0, float(technical["clipped_samples"].value or 0) / 2)
    noise_penalty = max(
        0.0, min(1.0, (_db(0.01) - float(technical["noise_floor"].value or -80)) / 40)
    )
    score = 2.5 + 1.8 * pitch_confidence - 0.8 * clipping_penalty - 0.4 * noise_penalty
    return Metric(
        value=round(float(np.clip(score, 1, 5)), 2),
        unit="/5",
        confidence=0.35,
        source="local heuristic; SingMOS not loaded",
    )


def _vocal_behaviour(
    times: np.ndarray,
    f0: np.ndarray,
    confidence: np.ndarray,
) -> dict[str, Metric]:
    voiced = f0 > 0
    if voiced.sum() < 5:
        return {
            "vibrato_depth": Metric(value=0, unit="cents", confidence=0, source="pitch contour"),
            "slide_activity": Metric(value=0, unit="%", confidence=0, source="pitch contour"),
            "pitch_drift": Metric(value=0, unit="cents/s", confidence=0, source="phrase proxy"),
            "breath_activity": Metric(
                value=0, unit="%", confidence=0.2, source="periodicity proxy"
            ),
        }
    indices = np.flatnonzero(voiced)
    midi_track = np.interp(np.arange(len(f0)), indices, hz_to_midi(f0[voiced]))
    trend = median_filter(midi_track, size=21, mode="nearest")
    residual_cents = (midi_track - trend) * 100
    step_cents = np.abs(np.diff(midi_track)) * 100
    hop = float(np.median(np.diff(times))) if len(times) > 1 else 0.01
    drift = np.median(np.diff(trend)[voiced[1:]]) * 100 / max(hop, 1e-6)
    breath_like = (~voiced) & (confidence > 0.03)
    return {
        "vibrato_depth": Metric(
            value=round(float(np.percentile(np.abs(residual_cents[voiced]), 80)), 1),
            unit="cents",
            confidence=0.62,
            source="detrended pitch contour",
        ),
        "slide_activity": Metric(
            value=round(float(np.mean(step_cents > 9) * 100), 1),
            unit="%",
            confidence=0.58,
            source="pitch-motion proxy",
        ),
        "pitch_drift": Metric(
            value=round(float(drift), 1),
            unit="cents/s",
            confidence=0.5,
            source="phrase-trend proxy",
        ),
        "breath_activity": Metric(
            value=round(float(np.mean(breath_like) * 100), 1),
            unit="%",
            confidence=0.38,
            source="low-periodicity proxy",
        ),
    }


def _what_i_hear(
    technical: dict[str, Metric], vocal: dict[str, Metric], descriptors: list[str]
) -> list[str]:
    lines = [f"The vocal reads as {descriptors[0]} with {descriptors[1]}."]
    correction = float(vocal["median_note_error"].value or 0)
    if correction < 18:
        lines.append(
            "Most sustained notes already sit close to a musical centre; correction should stay subtle."
        )
    elif correction < 45:
        lines.append(
            "Several notes can benefit from gentle centring while keeping their approaches and vibrato."
        )
    else:
        lines.append(
            "Pitch centres vary enough that harmonic context should be reviewed before a strong render."
        )
    if float(technical["clipped_samples"].value or 0) > 0.05:
        lines.append("The recording contains clipping that tuning cannot repair cleanly.")
    if float(technical["sibilance_ratio"].value or 0) > 0.12:
        lines.append(
            "Upper consonants are prominent, so de-essing should remain restrained and breath-aware."
        )
    return lines


class StudioEngine:
    def __init__(self, store: StudioStore) -> None:
        self.store = store
        self.quality = QualityRunner(store.root)

    def model_status(self) -> dict[str, object]:
        memory = memory_free_percent()
        low_memory = low_memory_mode()
        bundled_separator = self.store.root / "runtimes" / "separation" / "bin" / "audio-separator"
        return {
            "ffmpeg": bool(shutil.which("ffmpeg")),
            "rubberband": bool(shutil.which("rubberband")),
            "llama_server": shutil.which("llama-server"),
            "audio_separator": bool(shutil.which("audio-separator") or bundled_separator.is_file()),
            "clap_checkpoint": self.quality.clap_ready,
            "singmos_checkpoint": self.quality.singmos_ready,
            "memory_free_percent": memory,
            "low_memory_mode": low_memory,
            "heavy_tasks_ready": not low_memory
            and (memory is None or memory >= MIN_HEAVY_MEMORY_PERCENT),
            "quality_install": "./scripts/setup_quality.sh",
        }

    def _track_pitch(
        self,
        audio: np.ndarray,
        sr: int,
        config: CorrectionConfig,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        if low_memory_mode():
            times, f0, confidence = estimate_pitch(audio, sr, config)
            return times, f0, confidence, "built-in fallback; low-memory mode enabled"
        free_memory = memory_free_percent()
        if free_memory is None or free_memory >= MIN_HEAVY_MEMORY_PERCENT:
            return estimate_pitch_quality(audio, sr, config)
        times, f0, confidence = estimate_pitch(audio, sr, config)
        return (
            times,
            f0,
            confidence,
            f"built-in fallback; neural tracker paused at {free_memory}% free memory",
        )

    def _prepare_stems(self, project_id: str, progress: Progress) -> tuple[Path, Path | None]:
        project = self.store.get_project(project_id)
        paths = self.store.paths(project_id)
        directory = self.store.project_dir(project_id)
        working = directory / "working"
        vocal = working / "vocal.wav"
        accompaniment = working / "accompaniment.wav"
        working.mkdir(exist_ok=True)
        if project.mode == InputMode.FULL_MIX:
            bundled = self.store.root / "runtimes" / "separation" / "bin" / "audio-separator"
            separator = os.getenv("CUTETUNER_AUDIO_SEPARATOR_BIN") or shutil.which(
                "audio-separator"
            )
            if not separator and bundled.is_file():
                separator = str(bundled)
            if not separator:
                raise RuntimeError(
                    "Full-mix mode needs the separation sidecar. Run: ./scripts/setup_quality.sh"
                )
            if low_memory_mode():
                raise RuntimeError(
                    "Stem separation is paused because CUTETUNER_LOW_MEMORY is enabled."
                )
            free_memory = memory_free_percent()
            if free_memory is not None and free_memory < MIN_HEAVY_MEMORY_PERCENT:
                raise RuntimeError(
                    f"Stem separation paused: only {free_memory}% memory headroom is free. "
                    "Close memory-heavy apps, then try again."
                )
            progress("separating", 0.16, "Separating the lead vocal from the song")
            separated = working / "separated"
            separated.mkdir(exist_ok=True)
            separation_input = working / "separation-input.wav"
            decode_audio(paths["primary_path"], separation_input)
            mix_audio, mix_sr = read_audio(separation_input)
            original_frames = len(mix_audio)
            minimum_frames = round(mix_sr * 11)
            if original_frames < minimum_frames:
                mix_audio = np.pad(
                    mix_audio,
                    ((0, minimum_frames - original_frames), (0, 0)),
                )
                write_wav(separation_input, mix_audio, mix_sr)
            model_directory = self.store.models_root / "audio-separator"
            model_directory.mkdir(exist_ok=True)
            _run_with_progress(
                [
                    separator,
                    str(separation_input),
                    "--model_filename",
                    SEPARATOR_MODEL,
                    "--model_file_dir",
                    str(model_directory),
                    "--output_dir",
                    str(separated),
                    "--output_format",
                    "WAV",
                ],
                progress,
            )
            candidates = list(separated.glob("*.wav"))
            vocals, instrumental = select_separator_stems(candidates)
            if vocals is None or instrumental is None:
                raise RuntimeError(
                    "The separator did not produce both vocal and instrumental stems."
                )
            vocal_audio, vocal_sr = read_audio(vocals)
            accompaniment_audio, accompaniment_sr = read_audio(instrumental)
            if vocal_sr != mix_sr or accompaniment_sr != mix_sr:
                raise RuntimeError("The separator returned an unexpected sample rate.")
            write_wav(vocal, vocal_audio[:original_frames], mix_sr)
            write_wav(accompaniment, accompaniment_audio[:original_frames], mix_sr)
        elif project.mode == InputMode.VOCAL_BACKING:
            progress("decoding", 0.14, "Preparing the dry vocal and backing track")
            decode_audio(paths["primary_path"], vocal)
            if paths["backing_path"] is None:
                raise RuntimeError("This project mode requires a backing track.")
            decode_audio(paths["backing_path"], accompaniment)
            vocal_audio, vocal_sr = read_mono(vocal)
            backing_audio, backing_sr = read_mono(accompaniment)
            if (
                vocal_sr != backing_sr
                or abs(len(vocal_audio) - len(backing_audio)) / vocal_sr > 0.05
            ):
                raise RuntimeError(
                    "Vocal and backing must begin at the same timestamp and have matching duration."
                )
        else:
            progress("decoding", 0.14, "Preparing the dry vocal")
            decode_audio(paths["primary_path"], vocal)
            accompaniment = None
        self.store.update(
            project_id,
            vocal_path=str(vocal),
            accompaniment_path=str(accompaniment) if accompaniment else None,
        )
        return vocal, accompaniment

    def analyze(self, project_id: str, progress: Progress) -> AnalysisReport:
        self.store.update(project_id, status="analyzing")
        progress("ingest", 0.04, "Checking the recording")
        vocal_path, accompaniment_path = self._prepare_stems(project_id, progress)
        progress("pitch", 0.35, "Following notes, slides, and vibrato")
        vocal_audio, sr = read_mono(vocal_path)
        raw_audio, _ = read_audio(vocal_path)
        base = CorrectionConfig()
        times, f0, confidence, tracker = self._track_pitch(vocal_audio, sr, base)
        np.savez_compressed(
            self.store.project_dir(project_id) / "working" / "pitch-analysis.npz",
            times=times,
            f0=f0,
            confidence=confidence,
            tracker=np.asarray(tracker),
        )
        voiced = f0 > 0
        midi = hz_to_midi(f0[voiced]) if voiced.any() else np.array([])
        nearest = np.round(midi) if len(midi) else np.array([])
        technical = _technical_metrics(raw_audio, sr)
        vocal_metrics = {
            "voiced_duration": Metric(
                value=round(float(voiced.sum() * base.hop_ms / 1_000), 2),
                unit="s",
                confidence=0.9,
                source="pitch tracker",
            ),
            "median_pitch": Metric(
                value=round(float(np.median(f0[voiced])), 1) if voiced.any() else None,
                unit="Hz",
                confidence=round(float(np.mean(confidence[voiced])), 3) if voiced.any() else 0,
                source="pitch tracker",
            ),
            "pitch_span": Metric(
                value=round(float(np.ptp(midi)), 2) if len(midi) else 0,
                unit="semitones",
                confidence=0.8 if voiced.any() else 0,
                source="pitch tracker",
            ),
            "median_note_error": Metric(
                value=round(float(np.median(np.abs(midi - nearest)) * 100), 1) if len(midi) else 0,
                unit="cents",
                confidence=0.75 if voiced.any() else 0,
                source="nearest-note proxy",
            ),
            "pitch_confidence": Metric(
                value=round(float(np.mean(confidence[voiced])), 3) if voiced.any() else 0,
                confidence=0.9,
                source="pitch tracker",
            ),
        }
        vocal_metrics.update(_vocal_behaviour(times, f0, confidence))
        progress("harmony", 0.58, "Reading harmonic context")
        key = scale = None
        key_confidence = 0.0
        key_alternatives: list[str] = []
        chords: list[ChordEvent] = []
        if accompaniment_path:
            key_estimate = detect_key(accompaniment_path)
            key, scale, key_confidence = (
                key_estimate.key,
                key_estimate.scale,
                key_estimate.confidence,
            )
            key_alternatives = list(key_estimate.alternatives)
            accompaniment, accompaniment_sr = read_mono(accompaniment_path)
            chords = _chord_timeline(accompaniment, accompaniment_sr)
        descriptors = _semantic_descriptors(technical, vocal_metrics)
        progress("perception", 0.78, "Estimating character and naturalness")
        pitch_confidence = float(vocal_metrics["pitch_confidence"].value or 0)
        naturalness = _naturalness_proxy(technical, pitch_confidence)
        semantic_status = "local descriptor fallback; CLAP is not installed"
        naturalness_status = "heuristic fallback; SingMOS-Pro is not installed"
        warnings: list[str] = []
        free_memory = memory_free_percent()
        low_memory = low_memory_mode()
        if (
            self.quality.clap_ready
            and not low_memory
            and (free_memory is None or free_memory >= MIN_HEAVY_MEMORY_PERCENT)
        ):
            try:
                progress("perception", 0.8, "Comparing vocal character with music-aware CLAP")
                clap = self.quality.clap(vocal_path)
                descriptors = [str(value) for value in clap.get("descriptors", descriptors)][:3]
                semantic_status = (
                    f"{clap.get('source', 'CLAP')} · confidence {clap.get('confidence', 'unknown')}"
                )
            except QualityModelError as error:
                semantic_status = f"CLAP failed; local fallback used: {error}"
                warnings.append(
                    "The CLAP perception model failed; semantic labels use DSP proxies."
                )
        elif self.quality.clap_ready:
            semantic_status = (
                "CLAP paused by low-memory mode; local fallback used"
                if low_memory
                else f"CLAP paused at {free_memory}% free memory; local fallback used"
            )
        free_memory = memory_free_percent()
        if (
            self.quality.singmos_ready
            and not low_memory
            and (free_memory is None or free_memory >= MIN_HEAVY_MEMORY_PERCENT)
        ):
            try:
                progress("perception", 0.88, "Predicting singing naturalness with SingMOS-Pro")
                singmos = self.quality.singmos(vocal_path)
                naturalness = Metric(
                    value=float(singmos["score"]),
                    unit="/5",
                    confidence=float(singmos.get("confidence", 0.68)),
                    source=str(singmos.get("source", "SingMOS-Pro prediction")),
                )
                naturalness_status = (
                    f"SingMOS-Pro v1.1.2 · {singmos.get('windows', 1)} representative window(s)"
                )
            except (QualityModelError, KeyError, TypeError, ValueError) as error:
                naturalness_status = f"SingMOS-Pro failed; heuristic fallback used: {error}"
                warnings.append(
                    "The SingMOS-Pro model failed; naturalness is shown as a low-confidence proxy."
                )
        elif self.quality.singmos_ready:
            naturalness_status = (
                "SingMOS-Pro paused by low-memory mode; heuristic fallback used"
                if low_memory
                else f"SingMOS-Pro paused at {free_memory}% free memory; heuristic fallback used"
            )
        if not voiced.any() or float(vocal_metrics["voiced_duration"].value or 0) < 0.5:
            warnings.append("Insufficient voiced audio for confident tuning.")
        if accompaniment_path and key_confidence < 0.035:
            warnings.append("Harmonic confidence is low; review the key before rendering.")
        if not accompaniment_path:
            warnings.append(
                "No harmonic context was supplied; choose a key or use chromatic correction."
            )
        if float(technical["clipped_samples"].value or 0) > 0.05:
            warnings.append("Clipping is present in the vocal source.")
        report = AnalysisReport(
            duration_seconds=round(len(vocal_audio) / sr, 3),
            sample_rate=sr,
            channels=raw_audio.shape[1],
            key=key,
            scale=scale,
            key_confidence=key_confidence,
            key_alternatives=key_alternatives,
            technical=technical,
            vocal=vocal_metrics,
            semantic_descriptors=descriptors,
            naturalness=naturalness,
            notes=_pitch_notes(times, f0, confidence),
            chords=chords,
            what_i_hear=_what_i_hear(technical, vocal_metrics, descriptors),
            warnings=warnings,
            analyzer_status={
                "pitch": tracker,
                "semantic": semantic_status,
                "naturalness": naturalness_status,
                "separation": "not needed"
                if self.store.get_project(project_id).mode != InputMode.FULL_MIX
                else SEPARATOR_MODEL,
            },
        )
        self.store.update(project_id, status="analyzed", report_json=report.model_dump_json())
        progress("complete", 1.0, "Analysis complete")
        return report

    def create_plan(
        self, project_id: str, brief: CreativeBrief, controls: TuningControls | None = None
    ) -> TuningPlan:
        project = self.store.get_project(project_id)
        if not project.report:
            raise RuntimeError("Analyze the project before creating a tuning plan.")
        defaults = {
            "transparent": (0.52, 120.0, 145.0),
            "balanced": (0.72, 72.0, 220.0),
            "obvious": (0.94, 24.0, 400.0),
        }
        strength, retune, max_cents = defaults[brief.tuning_intent]
        facts = {key: metric.value for key, metric in project.report.vocal.items()}
        settings = self.store.settings()
        llm = local_llm_profile(
            brief.vibe,
            facts,
            endpoint=settings.get("llm_endpoint"),
            model=settings.get("llm_model"),
        )
        vibe_profile = llm or profile_from_text(brief.vibe)
        strength = (strength + vibe_profile.strength) / 2
        retune = (retune + vibe_profile.retune_ms) / 2
        selected = controls or TuningControls(
            key=project.report.key,
            scale=project.report.scale or "chromatic",
            strength=round(strength, 3),
            retune_ms=round(retune, 1),
            max_correction_cents=max_cents,
        )
        if not selected.key and selected.scale != "chromatic":
            raise ValueError("Choose a key for major or minor correction.")
        config = CorrectionConfig(
            key=selected.key or "C", scale=selected.scale or "chromatic", strength=1
        )
        sequence = chord_aware_targets(
            project.report.notes,
            project.report.chords,
            config,
            selected.max_correction_cents,
        )
        note_events = [
            note.model_copy(
                update={
                    "target_midi": round(
                        note.midi + selected.strength * (target - note.midi),
                        2,
                    )
                }
            )
            for note, target in zip(project.report.notes, sequence, strict=True)
        ]
        rationale = [
            f"The {brief.tuning_intent} intent sets {round(selected.strength * 100)}% note centring.",
            f"The '{vibe_profile.name}' interpretation keeps retune transitions near {round(selected.retune_ms)} ms.",
            "Slides, breaths, and micro-vibrato remain protected unless advanced controls reduce protection.",
        ]
        paths = self.store.paths(project_id)
        vocal_path = paths.get("vocal_path")
        free_memory = memory_free_percent()
        if (
            vocal_path
            and self.quality.clap_ready
            and not low_memory_mode()
            and (free_memory is None or free_memory >= MIN_HEAVY_MEMORY_PERCENT)
        ):
            try:
                vibe_result = self.quality.clap(vocal_path, brief.vibe)
                similarity = float(vibe_result.get("vibe_similarity", 0))
                rationale.append(
                    f"Music-aware CLAP found {round(similarity * 100)}% audio-to-vibe alignment; "
                    "this is semantic evidence, not an artistic verdict."
                )
            except (QualityModelError, TypeError, ValueError):
                rationale.append(
                    "CLAP vibe comparison was unavailable; bounded rules remained in control."
                )
        plan = TuningPlan(
            rationale=rationale,
            controls=selected,
            notes=note_events,
            chords=project.report.chords,
            protected_ranges=brief.protected_ranges,
            source="local LLM + bounded rules" if llm else "bounded local rules",
        )
        self.store.update(
            project_id,
            status="planned",
            brief_json=brief.model_dump_json(),
            plan_json=plan.model_dump_json(),
        )
        return plan

    def _rubberband_render(
        self,
        audio: np.ndarray,
        sr: int,
        times: np.ndarray,
        f0: np.ndarray,
        target: np.ndarray,
    ) -> np.ndarray:
        executable = shutil.which("rubberband")
        if not executable:
            raise FileNotFoundError("rubberband")
        voiced = (f0 > 0) & (target > 0)
        if not voiced.any():
            return audio.copy()
        shifts = np.zeros(len(f0), dtype=np.float64)
        shifts[voiced] = 12 * np.log2(target[voiced] / f0[voiced])
        shifts = np.nan_to_num(shifts, nan=0.0, posinf=0.0, neginf=0.0)
        if float(np.max(np.abs(shifts))) < 0.01:
            return audio.copy()
        with tempfile.TemporaryDirectory(prefix="cutetuner-rubberband-") as temporary:
            scratch = Path(temporary)
            source = scratch / "source.wav"
            destination = scratch / "shifted.wav"
            pitch_map = scratch / "pitch-map.txt"
            write_wav(source, audio, sr)
            point_map = {0: 0.0}
            for time, shift in zip(times, shifts, strict=True):
                frame = min(len(audio) - 1, max(0, round(float(time) * sr)))
                point_map[frame] = float(shift)
            point_map[len(audio) - 1] = float(shifts[-1])
            pitch_map.write_text(
                "\n".join(f"{frame} {point_map[frame]:.7f}" for frame in sorted(point_map)),
                encoding="utf-8",
            )
            _run(
                [
                    executable,
                    "-q",
                    "-F",
                    "-p",
                    "0",
                    "--pitchmap",
                    str(pitch_map),
                    str(source),
                    str(destination),
                ]
            )
            shifted, shifted_sr = read_mono(destination)
            if shifted_sr != sr:
                raise RuntimeError("Rubber Band changed the sample rate")
            if len(shifted) != len(audio):
                shifted = np.interp(
                    np.linspace(0, max(0, len(shifted) - 1), len(audio)),
                    np.arange(len(shifted)),
                    shifted,
                )
            return shifted.astype(np.float64)

    def _correct(
        self,
        vocal_path: Path,
        output_path: Path,
        controls: TuningControls,
        protected: list[tuple[float, float]],
        planned_notes: list[NoteEvent] | None = None,
        pitch_track: tuple[np.ndarray, np.ndarray, np.ndarray, str] | None = None,
    ) -> dict[str, object]:
        audio, sr = read_mono(vocal_path)
        config = CorrectionConfig(
            key=controls.key or "C",
            scale=controls.scale or "chromatic",
            strength=controls.strength,
            retune_ms=controls.retune_ms,
        )
        if pitch_track is None:
            times, f0, confidence, tracker = self._track_pitch(audio, sr, config)
        else:
            times, f0, confidence, tracker = pitch_track
        target = target_pitch(f0, config)
        voiced = f0 > 0
        for note in planned_notes or []:
            if note.target_midi is None:
                continue
            note_frames = (times >= note.start) & (times <= note.end) & voiced
            target[note_frames] = midi_to_hz(note.target_midi)
        if voiced.any():
            cents = np.zeros_like(f0)
            cents[voiced] = 1200 * np.log2(target[voiced] / f0[voiced])
            cents = np.clip(cents, -controls.max_correction_cents, controls.max_correction_cents)
            target[voiced] = f0[voiced] * np.exp2(cents[voiced] / 1200)
            protection = controls.breath_protection * (1 - np.clip(confidence[voiced], 0, 1))
            target_midi = hz_to_midi(target[voiced])
            observed_midi = hz_to_midi(f0[voiced])
            target[voiced] = midi_to_hz(
                observed_midi + (target_midi - observed_midi) * (1 - protection)
            )
        for start, end in protected:
            protected_frames = (times >= start) & (times <= end)
            target[protected_frames] = f0[protected_frames]
        target = smooth_pitch(target, times, config)
        if voiced.sum() >= 5 and controls.vibrato_preservation > 0:
            indices = np.flatnonzero(voiced)
            observed_track = np.interp(np.arange(len(f0)), indices, hz_to_midi(f0[voiced]))
            vibrato = observed_track - median_filter(observed_track, size=11, mode="nearest")
            target_midi = hz_to_midi(np.where(target > 0, target, 440))
            preserve = controls.strength * controls.vibrato_preservation
            target[voiced] = midi_to_hz(target_midi[voiced] + preserve * vibrato[voiced])
        renderer = "PSOLA emergency fallback"
        if shutil.which("rubberband"):
            try:
                corrected = self._rubberband_render(audio, sr, times, f0, target)
                renderer = "Rubber Band continuous pitch map with formant preservation"
            except (OSError, RuntimeError, subprocess.SubprocessError, ValueError):
                corrected = pitch_shift_psola(audio, sr, times, f0, target)
        else:
            corrected = pitch_shift_psola(audio, sr, times, f0, target)
        write_wav(output_path, corrected, sr)
        return {
            "tracker": tracker,
            "renderer": renderer,
            "mean_confidence": round(float(np.mean(confidence[voiced])), 3) if voiced.any() else 0,
        }

    def _cached_pitch(
        self, project_id: str, start: float = 0, end: float | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, str] | None:
        cache = self.store.project_dir(project_id) / "working" / "pitch-analysis.npz"
        if not cache.is_file():
            return None
        with np.load(cache, allow_pickle=False) as payload:
            times = payload["times"].astype(np.float64)
            f0 = payload["f0"].astype(np.float64)
            confidence = payload["confidence"].astype(np.float64)
            tracker = str(payload["tracker"].item()) + " · reused analysis contour"
        if end is None:
            return times, f0, confidence, tracker
        selected = (times >= start) & (times < end)
        return times[selected] - start, f0[selected], confidence[selected], tracker

    def _polish(self, source: Path, destination: Path, controls: TuningControls) -> None:
        audio, sr = read_mono(source)
        processed = audio.astype(np.float64)
        if controls.formant_shift > 0:
            emphasis = np.concatenate([[processed[0]], np.diff(processed)])
            processed += emphasis * min(0.35, controls.formant_shift * 0.12)
        elif controls.formant_shift < 0:
            lowpass = butter(1, 3_200, btype="lowpass", fs=sr, output="sos")
            softened = sosfilt(lowpass, processed)
            blend = min(0.5, abs(controls.formant_shift) * 0.18)
            processed = processed * (1 - blend) + softened * blend
        if controls.eq_amount > 0:
            highpass = butter(2, 75, btype="highpass", fs=sr, output="sos")
            filtered = sosfilt(highpass, processed)
            processed = processed * (1 - controls.eq_amount) + filtered * controls.eq_amount
        if controls.deesser_amount > 0:
            band = butter(
                2, (5_000, min(10_000, sr / 2 - 200)), btype="bandpass", fs=sr, output="sos"
            )
            sibilance = sosfilt(band, processed)
            processed -= sibilance * controls.deesser_amount * 0.34
        if controls.compression_amount > 0:
            threshold = 10 ** (-18 / 20)
            magnitude = np.abs(processed)
            ratio = 1 + controls.compression_amount * 2.5
            compressed = np.where(
                magnitude > threshold, threshold + (magnitude - threshold) / ratio, magnitude
            )
            processed = np.sign(processed) * compressed
        if controls.ambience_amount > 0:
            delay = int(sr * 0.075)
            room = np.zeros_like(processed)
            room[delay:] = processed[:-delay]
            processed += room * controls.ambience_amount * 0.22
        processed *= 10 ** (controls.vocal_gain_db / 20)
        peak = float(np.max(np.abs(processed))) if len(processed) else 0
        if peak > 0.94:
            processed *= 0.94 / peak
        write_wav(destination, processed, sr)

    def _mix(self, vocal_path: Path, accompaniment_path: Path, destination: Path) -> None:
        vocal, sr = read_mono(vocal_path)
        accompaniment, _ = read_audio(accompaniment_path, sr)
        length = max(len(vocal), len(accompaniment))
        mixed = np.zeros((length, max(2, accompaniment.shape[1])), dtype=np.float32)
        mixed[: len(accompaniment), : accompaniment.shape[1]] = accompaniment
        vocal_padded = np.zeros(length, dtype=np.float32)
        vocal_padded[: len(vocal)] = vocal
        mixed += vocal_padded[:, None]
        peak = float(np.max(np.abs(mixed))) if mixed.size else 0
        if peak > 0.94:
            mixed *= 0.94 / peak
        write_wav(destination, mixed, sr)

    def render_previews(self, project_id: str, progress: Progress) -> list[OutputFile]:
        project = self.store.get_project(project_id)
        if not project.plan:
            raise RuntimeError("Create the tuning plan before rendering previews.")
        paths = self.store.paths(project_id)
        vocal_path = paths["vocal_path"]
        if vocal_path is None:
            raise RuntimeError("The analyzed vocal stem is missing.")
        directory = self.store.project_dir(project_id) / "previews"
        directory.mkdir(exist_ok=True)
        audio, sr = read_mono(vocal_path)
        start = max(0, int((project.report.duration_seconds / 2 - 6) * sr)) if project.report else 0
        sample = audio[start : start + sr * 12]
        offset = start / sr
        preview_notes = [
            note.model_copy(update={"start": max(0, note.start - offset), "end": note.end - offset})
            for note in project.plan.notes
            if note.end > offset and note.start < offset + len(sample) / sr
        ]
        sample_path = directory / "source.wav"
        write_wav(sample_path, sample, sr)
        variants = {
            "preview_natural": replace_controls(
                project.plan.controls,
                strength=max(0.25, project.plan.controls.strength - 0.18),
                retune_ms=min(220, project.plan.controls.retune_ms + 45),
            ),
            "preview_recommended": project.plan.controls,
            "preview_tighter": replace_controls(
                project.plan.controls,
                strength=min(1, project.plan.controls.strength + 0.18),
                retune_ms=max(8, project.plan.controls.retune_ms - 35),
            ),
        }
        cached_pitch = self._cached_pitch(project_id, offset, offset + len(sample) / sr)
        outputs: list[OutputFile] = []
        for index, (kind, controls) in enumerate(variants.items()):
            progress(
                "preview", 0.1 + index * 0.27, f"Rendering {kind.replace('preview_', '')} preview"
            )
            raw = directory / f"{kind}.wav"
            polished = directory / f"{kind}-polished.wav"
            mp3 = directory / f"{kind}.mp3"
            strength_ratio = controls.strength / max(project.plan.controls.strength, 1e-6)
            variant_notes = [
                note.model_copy(
                    update={
                        "target_midi": note.midi
                        + (float(note.target_midi or note.midi) - note.midi) * strength_ratio
                    }
                )
                for note in preview_notes
            ]
            self._correct(
                sample_path,
                raw,
                controls,
                [],
                variant_notes,
                pitch_track=cached_pitch,
            )
            self._polish(raw, polished, controls)
            encode_mp3(polished, mp3)
            outputs.append(OutputFile(kind=kind, filename=mp3.name))
        existing = [output for output in project.outputs if not output.kind.startswith("preview_")]
        self.store.update(
            project_id,
            status="previewed",
            outputs_json=json.dumps([item.model_dump() for item in existing + outputs]),
        )
        progress("complete", 1, "Three previews are ready")
        return outputs

    def render(
        self, project_id: str, progress: Progress, controls: TuningControls | None = None
    ) -> list[OutputFile]:
        project = self.store.get_project(project_id)
        if not project.plan:
            raise RuntimeError("Create the tuning plan before rendering.")
        selected = controls or project.plan.controls
        if controls:
            self.create_plan(project_id, project.brief, controls)
            project = self.store.get_project(project_id)
        paths = self.store.paths(project_id)
        vocal_path = paths["vocal_path"]
        if vocal_path is None:
            raise RuntimeError("The analyzed vocal stem is missing.")
        output_dir = self.store.project_dir(project_id) / "outputs"
        output_dir.mkdir(exist_ok=True)
        tuned = output_dir / "corrected-vocal-master.wav"
        polished = output_dir / "corrected-vocal-polished.wav"
        vocal_mp3 = output_dir / "corrected-vocal.mp3"
        progress("tuning", 0.12, "Correcting note centres while preserving phrasing")
        protected = [(item.start, item.end) for item in project.plan.protected_ranges]
        self._correct(
            vocal_path,
            tuned,
            selected,
            protected,
            project.plan.notes,
            pitch_track=self._cached_pitch(project_id),
        )
        progress("polish", 0.52, "Applying restrained vocal polish")
        self._polish(tuned, polished, selected)
        progress("encoding", 0.72, "Encoding corrected vocal MP3")
        encode_mp3(polished, vocal_mp3)
        outputs = [OutputFile(kind="corrected_vocal", filename=vocal_mp3.name)]
        accompaniment = paths["accompaniment_path"]
        if accompaniment:
            mix_wav = output_dir / "finished-mix-master.wav"
            mix_mp3 = output_dir / "finished-mix.mp3"
            progress("mixing", 0.82, "Remixing the corrected vocal with the accompaniment")
            self._mix(polished, accompaniment, mix_wav)
            encode_mp3(mix_wav, mix_mp3)
            outputs.append(OutputFile(kind="finished_mix", filename=mix_mp3.name))
        if not bool(self.store.settings().get("keep_masters", True)):
            for master in (tuned, polished, output_dir / "finished-mix-master.wav"):
                master.unlink(missing_ok=True)
        previews = [output for output in project.outputs if output.kind.startswith("preview_")]
        self.store.update(
            project_id,
            status="complete",
            outputs_json=json.dumps([item.model_dump() for item in previews + outputs]),
        )
        progress("complete", 1, "Final exports are ready")
        return outputs


def replace_controls(controls: TuningControls, **changes: float) -> TuningControls:
    values = controls.model_dump()
    values.update(changes)
    return TuningControls.model_validate(values)
