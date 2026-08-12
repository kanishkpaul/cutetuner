from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np

from .audio import read_mono, write_wav
from .config import CorrectionConfig
from .director import local_llm_profile, profile_from_text
from .pitch import estimate_pitch, smooth_pitch, target_pitch
from .psola import pitch_shift_psola
from .song import detect_key, parse_key


def correct_file(
    input_path: str | Path, output_path: str | Path, config: CorrectionConfig
) -> dict[str, object]:
    audio, sr = read_mono(input_path)
    times, f0, confidence = estimate_pitch(audio, sr, config)
    target = smooth_pitch(target_pitch(f0, config), times, config)
    processed = pitch_shift_psola(audio, sr, times, f0, target)
    processed *= 0.95 / max(0.95, float(np.max(np.abs(processed))))
    write_wav(output_path, processed, sr)
    voiced = f0 > 0
    cents = 1200 * np.log2(target[voiced] / f0[voiced]) if voiced.any() else np.array([])
    return {
        "input": str(input_path),
        "output": str(output_path),
        "sample_rate": sr,
        "seconds": round(len(audio) / sr, 3),
        "voiced_frames": int(voiced.sum()),
        "mean_abs_correction_cents": round(float(np.mean(np.abs(cents))), 1) if len(cents) else 0.0,
        "mean_confidence": round(float(np.mean(confidence[voiced])), 3) if voiced.any() else 0.0,
        "config": asdict(config),
    }


def auto_tune_file(
    vocal_path: str | Path,
    song_path: str | Path,
    output_path: str | Path,
    vibe: str,
    key_override: str | None = None,
    use_local_llm: bool = False,
) -> dict[str, object]:
    """Tune a vocal from song harmony plus a conservative production-vibe policy."""
    key_estimate = detect_key(song_path)
    if key_override:
        key, scale = parse_key(key_override)
        key_source = "user override"
    else:
        key, scale = key_estimate.key, key_estimate.scale
        key_source = "detected from song"
    vocal, sr = read_mono(vocal_path)
    probe = CorrectionConfig(key=key, scale=scale)
    _, f0, confidence = estimate_pitch(vocal, sr, probe)
    voiced = f0 > 0
    facts = {
        "voiced_seconds": round(float(voiced.sum() * probe.hop_ms / 1000), 2),
        "median_hz": round(float(np.median(f0[voiced])), 1) if voiced.any() else None,
        "pitch_span_semitones": round(float(np.ptp(12 * np.log2(f0[voiced] / 440))), 1)
        if voiced.any()
        else 0.0,
        "mean_pitch_confidence": round(float(np.mean(confidence[voiced])), 3)
        if voiced.any()
        else 0.0,
    }
    llm_profile = local_llm_profile(vibe, facts) if use_local_llm else None
    profile = llm_profile or profile_from_text(vibe)
    result = correct_file(
        vocal_path,
        output_path,
        CorrectionConfig(
            key=key, scale=scale, strength=profile.strength, retune_ms=profile.retune_ms
        ),
    )
    result["auto"] = {
        "song": str(song_path),
        "key_source": key_source,
        "key_estimate": key_estimate.to_dict(),
        "profile": asdict(profile),
        "profile_source": "local LLM" if llm_profile else "built-in vibe map",
        "vocal_facts": facts,
        "warning": (
            "Low key confidence: use --key-override after checking the chord progression."
            if not key_override and key_estimate.confidence < 0.035
            else None
        ),
    }
    return result
