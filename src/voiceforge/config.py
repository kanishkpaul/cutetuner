from __future__ import annotations

from dataclasses import dataclass

SCALES: dict[str, tuple[int, ...]] = {
    "major": (0, 2, 4, 5, 7, 9, 11),
    "minor": (0, 2, 3, 5, 7, 8, 10),
    "chromatic": tuple(range(12)),
}
NOTE_TO_PC = {
    "C": 0,
    "C#": 1,
    "DB": 1,
    "D": 2,
    "D#": 3,
    "EB": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "GB": 6,
    "G": 7,
    "G#": 8,
    "AB": 8,
    "A": 9,
    "A#": 10,
    "BB": 10,
    "B": 11,
}


@dataclass(frozen=True)
class CorrectionConfig:
    key: str = "G"
    scale: str = "minor"
    strength: float = 0.72
    retune_ms: float = 85.0
    min_hz: float = 70.0
    max_hz: float = 600.0
    frame_ms: float = 46.0
    hop_ms: float = 10.0
    voiced_threshold: float = 0.35

    def __post_init__(self) -> None:
        if self.key.upper() not in NOTE_TO_PC:
            raise ValueError(f"Unknown key: {self.key}. Try G, F#, Bb, etc.")
        if self.scale.lower() not in SCALES:
            raise ValueError(f"Unknown scale: {self.scale}. Try major, minor, or chromatic.")
        if not 0 <= self.strength <= 1:
            raise ValueError("strength must be between 0 and 1")
        if self.retune_ms <= 0:
            raise ValueError("retune_ms must be positive")

    @property
    def key_pc(self) -> int:
        return NOTE_TO_PC[self.key.upper()]

    @property
    def allowed_pcs(self) -> tuple[int, ...]:
        return tuple((self.key_pc + degree) % 12 for degree in SCALES[self.scale.lower()])
