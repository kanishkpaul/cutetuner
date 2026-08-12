from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def read_mono(path: str | Path, target_sr: int = 44_100) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, always_2d=True, dtype="float32")
    mono = audio.mean(axis=1)
    if sr != target_sr:
        gcd = int(np.gcd(sr, target_sr))
        mono = resample_poly(mono, target_sr // gcd, sr // gcd).astype(np.float32)
        sr = target_sr
    peak = float(np.max(np.abs(mono))) if len(mono) else 0.0
    return (mono / peak if peak > 1.0 else mono), sr


def read_audio(path: str | Path, target_sr: int = 44_100) -> tuple[np.ndarray, int]:
    """Read audio as frames x channels and preserve stereo for final remixing."""
    audio, sr = sf.read(path, always_2d=True, dtype="float32")
    if sr != target_sr:
        gcd = int(np.gcd(sr, target_sr))
        audio = np.stack(
            [
                resample_poly(audio[:, channel], target_sr // gcd, sr // gcd)
                for channel in range(audio.shape[1])
            ],
            axis=1,
        ).astype(np.float32)
        sr = target_sr
    return audio, sr


def write_wav(path: str | Path, audio: np.ndarray, sr: int) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    safe = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    sf.write(path, np.clip(safe, -0.999, 0.999), sr, subtype="PCM_24")
