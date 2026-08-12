"""Music-context analysis that stays local and explainable."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.signal import stft

from .audio import read_mono
from .config import NOTE_TO_PC

MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.6, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
PC_TO_NAME = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


@dataclass(frozen=True)
class KeyEstimate:
    key: str
    scale: str
    confidence: float
    alternatives: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _chroma(audio: np.ndarray, sr: int) -> np.ndarray:
    frequencies, _, spectrum = stft(audio, fs=sr, nperseg=4096, noverlap=3072, boundary=None)
    energy = np.abs(spectrum).mean(axis=1)
    usable = (frequencies >= 55) & (frequencies <= 5_000)
    frequencies, energy = frequencies[usable], energy[usable]
    midi = np.round(69 + 12 * np.log2(np.maximum(frequencies, 1e-8) / 440)).astype(int)
    chroma = np.zeros(12, dtype=float)
    np.add.at(chroma, midi % 12, energy)
    return chroma / max(float(np.linalg.norm(chroma)), 1e-12)


def detect_key(path: str | Path) -> KeyEstimate:
    """Estimate the tonic and major/minor mode from a backing track or instrumental.

    It deliberately reports confidence and alternatives because modulating songs and
    sparse intros cannot be represented honestly by a single global key.
    """
    audio, sr = read_mono(path)
    chroma = _chroma(audio, sr)
    candidates: list[tuple[float, int, str]] = []
    for tonic in range(12):
        for mode, profile in (("major", MAJOR_PROFILE), ("minor", MINOR_PROFILE)):
            normalized = profile / np.linalg.norm(profile)
            candidates.append((float(chroma @ np.roll(normalized, tonic)), tonic, mode))
    candidates.sort(reverse=True)
    best, second = candidates[0], candidates[1]
    margin = max(0.0, best[0] - second[0])
    alternatives = tuple(f"{PC_TO_NAME[t]} {mode}" for _, t, mode in candidates[1:4])
    return KeyEstimate(
        key=PC_TO_NAME[best[1]],
        scale=best[2],
        confidence=round(float(margin / max(abs(best[0]), 1e-9)), 3),
        alternatives=alternatives,
    )


def parse_key(value: str) -> tuple[str, str]:
    """Parse a human override such as `F# minor` or `Bb major`."""
    parts = value.strip().upper().split()
    if len(parts) == 2:
        parts[1] = {"MIN": "MINOR", "MAJ": "MAJOR"}.get(parts[1], parts[1])
    if len(parts) != 2 or parts[0] not in NOTE_TO_PC or parts[1].lower() not in {"major", "minor"}:
        raise ValueError("Key override must look like `F# minor` or `Bb major`.")
    return parts[0], parts[1].lower()
