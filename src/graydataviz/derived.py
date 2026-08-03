"""Loading derived (already-computed) NetCDF products: power, PEC, crackle
co-occurrence, and burst probability.

These functions are a cleaned-up port of `GDa.loader.loader`: same filename
conventions and directory layout (`results_root/<monkey>/<date>/session01/...`),
but each raises `DerivedDataNotFoundError` naming the exact path it looked for
(and, if the parent directory exists, what files *are* there) instead of letting
a bare `FileNotFoundError` surface from `xr.load_dataarray`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from .config import DataConfig, default_config
from .exceptions import DerivedDataNotFoundError


def _load_dataarray(path: Path) -> xr.DataArray:
    if not path.is_file():
        hint = ""
        if path.parent.is_dir():
            siblings = sorted(p.name for p in path.parent.iterdir())
            hint = f" Files present in {path.parent}: {siblings}"
        raise DerivedDataNotFoundError(f"expected derived data file not found: {path}.{hint}")
    return xr.load_dataarray(path)


def load_power(
    monkey: str,
    date: str,
    session: int = 1,
    trial_type: int = 1,
    behavioral_response: int = 1,
    aligned_at: str = "cue",
    decim: int = 20,
    mode: str = "hilbert",
    channel_numbers: bool = False,
    config: DataConfig | None = None,
) -> xr.DataArray:
    """Load a session's power time series, transposed to `("roi", "freqs", "trials", "times")`."""
    config = config or default_config()
    filename = f"power_tt_{trial_type}_br_{behavioral_response}_at_{aligned_at}_decim_{decim}_{mode}.nc"
    power = _load_dataarray(config.results_dir(monkey, date, session) / filename)
    power = power.transpose("roi", "freqs", "trials", "times")

    if channel_numbers:
        roi = [
            f"{r} ({ch})" for r, ch in zip(power.roi.data, power.attrs["channels_labels"])
        ]
        power = power.assign_coords({"roi": roi})
    return power


def load_pec_strength(
    monkey: str,
    date: str,
    session: int = 1,
    metric: str = "pec",
    aligned_at: str = "cue",
    decim: int = 10,
    config: DataConfig | None = None,
) -> xr.DataArray:
    """Load a session's PEC/degree time series, transposed to `("roi", "freqs", "trials", "times")`."""
    config = config or default_config()
    filename = f"{metric}_degree_at_{aligned_at}_decim_{decim}.nc"
    pecst = _load_dataarray(config.results_dir(monkey, date, session) / "network" / filename)
    return pecst.transpose("roi", "freqs", "trials", "times")


def load_crackle_cooccurrence(
    monkey: str,
    session_date: str,
    trial_type: int = 1,
    strength: bool = False,
    thr: int = 90,
    incorrect: bool = False,
    rectf: int | None = None,
    surrogate: bool = False,
    drop_roi: str | None = None,
    config: DataConfig | None = None,
) -> xr.DataArray:
    """Load the crackle co-occurrence (Kij) matrix, or per-ROI strength if `strength=True`.

    Note this product is aggregated per-monkey (not per session-date directory
    like power/PEC), matching `Results/<monkey>/crk_stats/...` in the original layout.
    """
    config = config or default_config()
    task_or_fix = "task" if trial_type == 1 else "fix"
    prefix = f"kij_{task_or_fix}_incorrect" if incorrect else f"kij_{task_or_fix}"

    if surrogate:
        filename = f"{prefix}_surr_{session_date}_q_{thr}.nc"
    elif rectf is not None:
        filename = f"{prefix}_{session_date}_q_{thr}_rectf_{rectf}.nc"
    else:
        filename = f"{prefix}_{session_date}_q_{thr}.nc"

    path = config.results_root / monkey / "crk_stats" / filename
    kij = _load_dataarray(path)

    if drop_roi is not None:
        if drop_roi not in ("same", "diff"):
            raise ValueError(f'drop_roi must be "same", "diff" or None, got {drop_roi!r}')
        masked = kij.copy()
        for roi in np.unique(kij.sources.data):
            idx = kij.sources.data == roi
            if drop_roi == "diff":
                masked[:, idx, ~idx, :] = 0
            else:
                masked[:, idx, idx, :] = 0
        kij = masked

    dims = ("sources", "targets", "freqs", "times", "boot") if surrogate else (
        "sources", "targets", "freqs", "times"
    )
    kij = kij.transpose(*dims)

    if strength:
        kij = kij.mean("targets").rename({"sources": "roi"})
    return kij


def load_burst_probability(
    monkey: str,
    session_date: str,
    trial_type: int = 1,
    aligned_at: str = "cue",
    thr: int = 90,
    conditional: bool = False,
    config: DataConfig | None = None,
) -> xr.DataArray:
    """Load burst probability time series, aggregated per-monkey like crackle co-occurrence."""
    config = config or default_config()
    task_or_fix = "task" if trial_type == 1 else "fix"
    if conditional:
        filename = f"P_b_{task_or_fix}_stim_{session_date}_at_{aligned_at}_q_{thr}.nc"
    else:
        filename = f"P_b_{task_or_fix}_{session_date}_at_{aligned_at}_q_{thr}.nc"

    path = config.results_root / monkey / "rate_modulations" / filename
    p_b = _load_dataarray(path)

    dims = ("roi", "freqs", "boot", "stim", "times") if conditional else (
        "roi", "freqs", "boot", "times"
    )
    return p_b.transpose(*dims)
