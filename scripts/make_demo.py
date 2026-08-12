"""Create a short synthetic vocal-like sweep for an end-to-end smoke test."""

from pathlib import Path

import numpy as np

from voiceforge.audio import write_wav

sr = 44_100
seconds = 2.0
t = np.arange(int(sr * seconds)) / sr
f0 = 450 + 4 * np.sin(2 * np.pi * 0.7 * t)
phase = 2 * np.pi * np.cumsum(f0) / sr
audio = 0.48 * np.sin(phase) + 0.12 * np.sin(2 * phase) + 0.05 * np.sin(3 * phase)
write_wav(Path("renders") / "demo-raw.wav", audio * np.hanning(len(audio)), sr)

# A simple C-major backing chord, sufficient to exercise automatic key detection.
chord = sum(0.16 * np.sin(2 * np.pi * hz * t) for hz in (130.81, 164.81, 196.0, 261.63))
write_wav(Path("renders") / "demo-song-c-major.wav", chord * np.hanning(len(chord)), sr)
