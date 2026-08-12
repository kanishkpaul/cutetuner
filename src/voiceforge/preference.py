from __future__ import annotations

import json
from pathlib import Path

import numpy as np

LABELS = ("original", "natural", "recommended", "tighter")
FEATURES = (
    "strength",
    "retune_ms",
    "max_correction_cents",
    "vibrato_preservation",
    "deesser_amount",
    "compression_amount",
    "ambience_amount",
    "pitch_span",
    "note_error",
)


class PreferenceRanker:
    """Small-data ridge classifier for private preview choices.

    This intentionally starts with a stable linear model rather than a neural network;
    a few dozen personal comparisons are not enough data to justify an MLP.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def train(self, ratings: list[dict[str, object]]) -> bool:
        usable = [rating for rating in ratings if rating.get("winner") in LABELS]
        if len(usable) < 20:
            return False
        matrix = np.array(
            [
                [float(rating.get("context", {}).get(feature, 0) or 0) for feature in FEATURES]
                for rating in usable
            ],
            dtype=float,
        )
        mean = matrix.mean(axis=0)
        scale = matrix.std(axis=0)
        scale[scale < 1e-6] = 1
        normalized = (matrix - mean) / scale
        design = np.column_stack([np.ones(len(normalized)), normalized])
        targets = np.zeros((len(usable), len(LABELS)))
        for index, rating in enumerate(usable):
            targets[index, LABELS.index(str(rating["winner"]))] = 1
        regularizer = np.eye(design.shape[1]) * 1.5
        regularizer[0, 0] = 0
        weights = np.linalg.solve(design.T @ design + regularizer, design.T @ targets)
        payload = {
            "version": 1,
            "labels": LABELS,
            "features": FEATURES,
            "mean": mean.tolist(),
            "scale": scale.tolist(),
            "weights": weights.tolist(),
            "comparisons": len(usable),
        }
        self.path.write_text(json.dumps(payload, indent=2))
        return True

    def recommend(self, context: dict[str, object]) -> str | None:
        if not self.path.is_file():
            return None
        payload = json.loads(self.path.read_text())
        vector = np.array([float(context.get(feature, 0) or 0) for feature in FEATURES])
        normalized = (vector - np.array(payload["mean"])) / np.array(payload["scale"])
        scores = np.concatenate([[1.0], normalized]) @ np.array(payload["weights"])
        return str(payload["labels"][int(np.argmax(scores))])
