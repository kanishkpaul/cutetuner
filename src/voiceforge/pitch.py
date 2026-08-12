from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import correlate

from .config import CorrectionConfig

_CREPE_CENTS_BASE = 1997.3794084376191
_CREPE_CENTS_PER_BIN = 20.0
_CREPE_BINS = 360


def _viterbi_path(probabilities: np.ndarray) -> np.ndarray:
    """Decode CREPE bins with its 23-bin transition band without a dense 360² matrix."""
    if not len(probabilities):
        return np.array([], dtype=np.int64)
    states = probabilities.shape[1]
    offsets = np.arange(-11, 12, dtype=np.int64)[:, None]
    current = np.arange(states, dtype=np.int64)[None, :]
    previous = current - offsets
    valid = (previous >= 0) & (previous < states)
    previous = np.clip(previous, 0, states - 1)
    weights = np.broadcast_to(12 - np.abs(offsets), previous.shape).astype(np.float64)
    row_sums = np.array(
        [
            sum(max(12 - abs(target - source), 0) for target in range(states))
            for source in range(states)
        ],
        dtype=np.float64,
    )
    transitions = np.full(previous.shape, -np.inf, dtype=np.float64)
    transitions[valid] = np.log(weights[valid] / row_sums[previous[valid]])
    emissions = np.log(np.clip(probabilities, 1e-12, 1.0))
    scores = emissions[0] - np.log(states)
    backpointers = np.zeros((len(emissions), states), dtype=np.int16)
    columns = np.arange(states)
    for frame in range(1, len(emissions)):
        candidates = scores[previous] + transitions
        best = np.argmax(candidates, axis=0)
        backpointers[frame] = previous[best, columns]
        scores = candidates[best, columns] + emissions[frame]
    path = np.zeros(len(emissions), dtype=np.int64)
    path[-1] = int(np.argmax(scores))
    for frame in range(len(emissions) - 1, 0, -1):
        path[frame - 1] = backpointers[frame, path[frame]]
    return path


def _frequency_to_bin(frequency: float, *, ceil: bool = False) -> int:
    cents = 1200.0 * np.log2(max(frequency, 1e-10) / 10.0)
    value = (cents - _CREPE_CENTS_BASE) / _CREPE_CENTS_PER_BIN
    quantized = np.ceil(value) if ceil else np.floor(value)
    return int(np.clip(quantized, 0, _CREPE_BINS))


def _bins_to_frequency(bins: np.ndarray) -> np.ndarray:
    cents = _CREPE_CENTS_PER_BIN * bins + _CREPE_CENTS_BASE
    return 10.0 * np.exp2(cents / 1200.0)


def _torchcrepe_activations(audio: np.ndarray, sr: int, hop: int) -> np.ndarray:
    """Infer CREPE activations in low-priority child processes with bounded memory."""
    chunk_samples = max(hop, int(sr * 30.0))
    parts: list[np.ndarray] = []
    worker_env = os.environ.copy()
    worker_env.update(
        {
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
            "VECLIB_MAXIMUM_THREADS": "2",
        }
    )
    with tempfile.TemporaryDirectory(prefix="cutetuner-pitch-") as temporary:
        root = Path(temporary)
        for chunk_index, start in enumerate(range(0, len(audio), chunk_samples)):
            chunk = audio[start : start + chunk_samples].astype(np.float32, copy=False)
            input_path = root / f"input-{chunk_index}.npy"
            output_path = root / f"activations-{chunk_index}.npy"
            np.save(input_path, chunk, allow_pickle=False)
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "voiceforge.pitch_worker",
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--sample-rate",
                    str(sr),
                    "--hop-length",
                    str(hop),
                ],
                check=True,
                capture_output=True,
                env=worker_env,
                timeout=300,
            )
            activations = np.load(output_path, allow_pickle=False)
            if chunk_index:
                activations = activations[1:]
            parts.append(activations)
    return np.concatenate(parts, axis=0)


def hz_to_midi(hz: np.ndarray | float) -> np.ndarray:
    values = np.asarray(hz, dtype=np.float64)
    return 69.0 + 12.0 * np.log2(np.maximum(values, 1e-10) / 440.0)


def midi_to_hz(midi: np.ndarray | float) -> np.ndarray:
    return 440.0 * np.exp2((np.asarray(midi, dtype=np.float64) - 69.0) / 12.0)


def estimate_pitch(
    audio: np.ndarray, sr: int, config: CorrectionConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate monophonic F0 with normalized autocorrelation.

    It is dependency-light and inspectable. It works best on a close-mic, dry lead
    vocal; swap in a neural estimator only after measuring it on your own takes.
    """
    frame = max(256, int(sr * config.frame_ms / 1000))
    hop = max(64, int(sr * config.hop_ms / 1000))
    frame += frame % 2
    if len(audio) < frame:
        audio = np.pad(audio, (0, frame - len(audio)))
    starts = np.arange(0, len(audio) - frame + 1, hop)
    window = np.hanning(frame)
    min_lag = max(1, int(sr / config.max_hz))
    max_lag = min(frame - 2, int(sr / config.min_hz))
    f0 = np.zeros(len(starts), dtype=np.float64)
    confidence = np.zeros_like(f0)

    for index, start in enumerate(starts):
        sample = audio[start : start + frame].astype(np.float64)
        sample = (sample - sample.mean()) * window
        if np.linalg.norm(sample) < 1e-4:
            continue
        ac = correlate(sample, sample, mode="full")[frame - 1 :]
        ac = ac / (ac[0] + 1e-12)
        region = ac[min_lag : max_lag + 1]
        lag = int(np.argmax(region)) + min_lag
        peak = float(ac[lag])
        if 1 <= lag < len(ac) - 1:
            left, center, right = ac[lag - 1 : lag + 2]
            denominator = left - 2 * center + right
            adjustment = 0.5 * (left - right) / denominator if abs(denominator) > 1e-12 else 0.0
        else:
            adjustment = 0.0
        if peak >= config.voiced_threshold:
            f0[index] = sr / (lag + adjustment)
            confidence[index] = peak

    voiced = f0 > 0
    if voiced.any():
        midi = hz_to_midi(f0[voiced])
        f0[voiced] = midi_to_hz(median_filter(midi, size=3, mode="nearest"))
    centers = (starts + frame // 2) / sr
    return centers, f0, confidence


def estimate_pitch_quality(
    audio: np.ndarray,
    sr: int,
    config: CorrectionConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Use full torchcrepe with Viterbi when installed, otherwise use autocorrelation.

    Imports stay lazy so the normal studio remains small and a model is never loaded merely
    by opening the UI. Callers are responsible for applying the memory-headroom policy.
    """
    if importlib.util.find_spec("torchcrepe") is None or importlib.util.find_spec("torch") is None:
        times, f0, confidence = estimate_pitch(audio, sr, config)
        return times, f0, confidence, "built-in autocorrelation fallback"
    try:
        effective_hop_ms = max(config.hop_ms, 20.0) if len(audio) / sr > 30 else config.hop_ms
        hop = max(64, int(sr * effective_hop_ms / 1_000))
        activations = _torchcrepe_activations(audio, sr, hop)
        minimum = _frequency_to_bin(config.min_hz)
        maximum = _frequency_to_bin(config.max_hz, ceil=True)
        masked = activations.copy()
        masked[:, :minimum] = -np.inf
        masked[:, maximum:] = -np.inf
        masked -= np.max(masked, axis=1, keepdims=True)
        probabilities = np.exp(masked)
        probabilities /= np.sum(probabilities, axis=1, keepdims=True)
        bins = _viterbi_path(probabilities)
        f0 = _bins_to_frequency(bins).astype(np.float64)
        confidence = activations[np.arange(len(bins)), bins].astype(np.float64)
        times = np.arange(len(f0), dtype=np.float64) * hop / sr
        return_source = (
            f"torchcrepe full + global Viterbi · {effective_hop_ms:g} ms contour · "
            "30 s isolated workers, 32-frame batches"
        )
        f0[confidence < max(0.18, config.voiced_threshold * 0.7)] = 0
        return times, f0, confidence, return_source
    except (
        ImportError,
        RuntimeError,
        TypeError,
        ValueError,
        AttributeError,
        OSError,
        subprocess.SubprocessError,
    ):
        times, f0, confidence = estimate_pitch(audio, sr, config)
        return times, f0, confidence, "built-in autocorrelation fallback (torchcrepe unavailable)"


def target_pitch(f0: np.ndarray, config: CorrectionConfig) -> np.ndarray:
    """Map voiced F0 to the nearest scale note, retaining motion at low strength."""
    result = f0.astype(np.float64).copy()
    voiced = f0 > 0
    if not voiced.any():
        return result
    observed = hz_to_midi(f0[voiced])
    candidates = np.array([n for n in range(12, 128) if n % 12 in config.allowed_pcs], dtype=float)
    snapped = candidates[np.abs(observed[:, None] - candidates[None, :]).argmin(axis=1)]
    result[voiced] = midi_to_hz(observed + config.strength * (snapped - observed))
    return result


def smooth_pitch(target_f0: np.ndarray, times: np.ndarray, config: CorrectionConfig) -> np.ndarray:
    """One-pole retune control; lower retune_ms makes transitions more obvious."""
    output = target_f0.copy()
    if len(output) < 2:
        return output
    alpha = 1.0 - np.exp(-(np.median(np.diff(times)) * 1000) / config.retune_ms)
    previous = 0.0
    for index, value in enumerate(output):
        if value <= 0:
            previous = 0.0
        else:
            previous = value if previous == 0 else previous + alpha * (value - previous)
            output[index] = previous
    return output
