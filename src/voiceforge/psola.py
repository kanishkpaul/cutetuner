from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks


def _at_times(values: np.ndarray, times: np.ndarray, positions: np.ndarray, sr: int) -> np.ndarray:
    voiced = values > 0
    if not voiced.any():
        return np.zeros_like(positions, dtype=np.float64)
    return np.interp(positions / sr, times[voiced], values[voiced], left=0.0, right=0.0)


def pitch_shift_psola(
    audio: np.ndarray, sr: int, times: np.ndarray, source_f0: np.ndarray, target_f0: np.ndarray
) -> np.ndarray:
    """Compact TD-PSOLA resynthesis for voiced lead vocal, with no cloud dependency.

    This is an editable learning-grade core, not a claim to equal commercial tuning.
    """
    n = len(audio)
    result = np.zeros(n, dtype=np.float64)
    weight = np.zeros(n, dtype=np.float64)
    positions = np.arange(n, dtype=np.float64)
    f0 = _at_times(source_f0, times, positions, sr)
    desired = _at_times(target_f0, times, positions, sr)
    valid = (f0 > 0) & (desired > 0)
    if valid.sum() < 5:
        return audio.copy()

    first = int(np.flatnonzero(valid)[0])
    marks: list[int] = []
    cursor = float(first)
    while cursor < n - 4:
        index = int(np.clip(round(cursor), 0, n - 1))
        if not valid[index]:
            cursor += max(1.0, sr / 180.0)
            continue
        period = float(np.clip(sr / f0[index], 8, sr / 55))
        radius = max(2, int(period * 0.45))
        left, right = max(0, index - radius), min(n, index + radius + 1)
        peaks, _ = find_peaks(np.abs(audio[left:right]))
        peak = left + int(peaks[np.argmin(np.abs(left + peaks - index))]) if len(peaks) else index
        if not marks or peak - marks[-1] >= max(4, int(period * 0.45)):
            marks.append(peak)
        cursor = max(cursor + period, float(peak) + period * 0.6)
    if len(marks) < 3:
        return audio.copy()

    synthesis = [float(marks[0])]
    for mark in marks[1:]:
        desired_hz = desired[min(mark, n - 1)] or f0[min(mark, n - 1)]
        synthesis.append(synthesis[-1] + sr / max(desired_hz, 1.0))

    for mark, destination in zip(marks, synthesis, strict=True):
        output_mark = round(destination)
        if output_mark >= n:
            break
        period = int(np.clip(sr / max(f0[mark], 1.0), 12, sr / 55))
        radius = int(period * 1.35)
        input_left, input_right = max(0, mark - radius), min(n, mark + radius)
        output_left, output_right = max(0, output_mark - radius), min(n, output_mark + radius)
        length = min(input_right - input_left, output_right - output_left)
        if length < 8:
            continue
        window = np.hanning(length)
        result[output_left : output_left + length] += (
            audio[input_left : input_left + length] * window
        )
        weight[output_left : output_left + length] += window
    return np.where(weight > 1e-5, result / np.maximum(weight, 1e-5), audio).astype(np.float32)
