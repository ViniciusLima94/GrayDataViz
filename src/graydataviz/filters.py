"""Zero-phase bandpass filtering for LFP traces, and the dataset's standard bands.

The production pipeline (`phase_coupling_analysis/phasedifferences.py`) filters
with `mne.filter.filter_data(..., method="iir")`, an IIR (Butterworth-family)
filter applied forward-backward for zero phase distortion. Here the same
zero-phase IIR behavior is reproduced directly with `scipy.signal` so the GUI
doesn't need to pull in `mne` as a dependency just to preview a filtered trace.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt

#: Per-monkey frequency bands (Hz), ported from phase_coupling_analysis/config.py.
DEFAULT_BANDS: dict[str, list[tuple[float, float]]] = {
    "lucy": [(0, 6), (6, 14), (14, 26), (26, 43), (43, 80)],
    "ethyl": [(0, 8), (8, 21), (21, 32), (32, 80)],
}


def bandpass_filter(
    x: np.ndarray,
    fsample: float,
    f_low: float,
    f_high: float,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth filter of `x` (last axis = time) to `[f_low, f_high]` Hz.

    `f_low <= 0` is treated as a low-pass filter up to `f_high` (matching bands
    such as lucy's `(0, 6)`, which describe the low end of the spectrum rather
    than a literal band edge at 0 Hz).
    """
    nyquist = fsample / 2
    if f_high >= nyquist:
        raise ValueError(f"f_high={f_high} must be below the Nyquist frequency ({nyquist})")

    if f_low <= 0:
        b, a = butter(order, f_high / nyquist, btype="lowpass", output="ba")  # type: ignore[misc]
    else:
        b, a = butter(  # type: ignore[misc]
            order, [f_low / nyquist, f_high / nyquist], btype="bandpass", output="ba"
        )
    return filtfilt(b, a, x, axis=-1)


def band_presets(monkey: str) -> list[tuple[float, float]]:
    """Standard frequency bands for a monkey, or an empty list if unknown."""
    return DEFAULT_BANDS.get(monkey, [])
