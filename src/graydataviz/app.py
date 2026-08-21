"""Panel GUI for exploring raw LFP recordings.

Lets you pick a monkey/date/session/trial and up to two channels, see the
trial type and behavioral response, optionally overlay the spike raster on
the LFP trace, optionally overlay a bandpass-filtered version of the trace on
top of the raw signal, and see each trace's power spectrum (multitaper,
matching phase_coupling_analysis's xr_psd_array_multitaper params) below it.
When two channels are selected, their coherence (multitaper, matching
phase_coupling_analysis's conn_spec_average params) is shown between the two
power spectra instead, with an optional non-parametric spectral Granger
causality panel (pyGC, matching phase_coupling_analysis's conn_gc_average
params) next to it. A Hilbert decomposition panel (envelope and/or
instantaneous phase, matching phase_coupling_analysis's hilbert_decomposition:
bandpass filter then scipy.signal.hilbert) can be shown below the LFP trace,
along with spike-triggered average, phase-amplitude coupling (Tort modulation
index), and spike-phase locking panels. The LFP trace itself is an interactive
Bokeh chart (pan/zoom with the mouse, or type a start/end time) rather than a
static image -- unlike every other plot here, its underlying sample values are
therefore inspectable via the browser's dev tools.

Run with:
    python -m graydataviz.app
or, after `pip install -e ".[gui]"`:
    graydataviz-gui
"""

from __future__ import annotations

import colorsys

import matplotlib

matplotlib.use("Agg")
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import panel as pn
from bokeh.models import ColumnDataSource, FixedTicker, Span
from bokeh.plotting import figure as bokeh_figure
from mne.time_frequency import psd_array_multitaper
from scipy import stats as scipy_stats
from scipy.signal import hilbert as scipy_hilbert
from scipy.signal import butter, sosfiltfilt

try:
    # Sibling package, not on PyPI (the "pygc" name there belongs to an
    # unrelated project) -- install with `pip install -e ../pyGC`, matching
    # how phase_coupling_analysis itself depends on it. Optional: the rest of
    # the app works without it, just without the Granger causality panel.
    from pygc import spectral_granger_causality

    _PYGC_AVAILABLE = True
except ImportError:
    granger_causality = None
    _PYGC_AVAILABLE = False

from .config import DataConfig, default_config
from .discovery import list_dates, list_monkeys, list_sessions
from .exceptions import GrayDataVizError
from .filters import band_presets, bandpass_filter
from .metadata import SessionMetadata, load_session_metadata
from .session import load_session
from .trials import BehavioralResponse, TrialType

pn.extension()

# Preferred initial monkey/date on launch, when available on disk -- both
# `Select` widgets otherwise default to the first option alphabetically
# (`list_monkeys`/`list_dates` both sort), which put "ethyl" first even
# though most of the reference screenshots/figures here use lucy/141017.
# Falls back to that alphabetical-first behavior if either isn't present.
_DEFAULT_MONKEY = "lucy"
_DEFAULT_DATE = "141017"

# Validated categorical palette (see the dataviz skill's reference palette):
# slot 1 blue / slot 2 orange, so identity stays consistent between the
# time-domain trace, its spike raster, and the power spectrum below it.
_COLOR_RAW = "#2a78d6"
_COLOR_FILTERED = "#eb6834"

# Raw recordings are in volts; phase_coupling_analysis/util.py's
# load_session_data multiplies by this same factor before anything is
# plotted or analyzed, so we match that convention here too.
_VOLTS_TO_MICROVOLTS = 1e6

# Multitaper PSD params, matching phase_coupling_analysis/src/metrics/spectral.py's
# xr_psd_array_multitaper defaults.
_PSD_BANDWIDTH = 1.0
_PSD_FMIN = 0.1
_PSD_FMAX = 80.0

# Multitaper coherence params, matching the bandwidth/fmin/fmax that
# savecoherence.py actually passes to conn_spec_average (a wider bandwidth
# than the PSD's, for a smoother cross-spectral estimate).
_COH_BANDWIDTH = 5.0
_COH_FMIN = 0.1
_COH_FMAX = 80.0

# Granger causality reuses the exact same bandwidth/fmin/fmax -- savegc.py
# passes conn_gc_average the identical values savecoherence.py passes
# conn_spec_average, so the two panels are directly comparable.

_PAC_N_BINS = 18
_SPIKE_PHASE_N_BINS = 18
_STA_HALF_WINDOW = 0.25  # seconds

# Power-product quantile regions & phase-difference circular plot, matching
# plot_phase_coupling_analysis/notebooks/Phase_Analysis-Method_Sumary.ipynb:
# quartiles of the pooled (all trials, all times) element-wise product of two
# channels' Hilbert power (envelope squared) -- the same quantity
# save_burst_trains.py thresholds for burst detection.
_QUANTILE_EDGES = np.array([0.0, 0.25, 0.50, 0.75, 1.0])
_QUANTILE_LABELS = ["Q1 (0-25%)", "Q2 (25-50%)", "Q3 (50-75%)", "Q4 (75-100%)"]
# Palette slots 3/4/6/7 (aqua/yellow/green/violet) -- slots 1/2/8
# (blue/orange/red) are already channel-1/channel-2/spikes identity colors.
_QUANTILE_COLORS = ["#1baf7a", "#eda100", "#008300", "#4a3aa7"]
_CIRCULAR_HIST_BINS = 16

# Full 8-slot categorical order, assigned in fixed order to stimulus groups
# when 2+ stimuli are selected in the trial-subset filter (PSD/coherence/STA
# overlay one line per stimulus instead of channel identity in that case).
_STIMULUS_COLORS = [
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#4a3aa7", "#e34948",
]

# All-channels montage tab: slot 8 (red) for the two eye traces, so they
# read as a different signal type than the LFP channels below them (which
# get one distinct hue each -- see _distinct_colors).
_COLOR_EYE = "#e34948"
_MONTAGE_DEFAULT_HEIGHT = 1000  # px -- initial value of the height slider
_MONTAGE_DEFAULT_SPACING = 12  # initial value of the spacing slider (see _update_montage)
# Row baselines always sit exactly 1.0 apart, regardless of the spacing
# slider -- trace amplitude (in the same data units) is what the slider
# actually controls, as 1 / spacing. Keeping amplitude *relative to* a fixed
# baseline unit (rather than both in absolute, height-dependent units) is
# what makes the spacing slider's effect independent of the height slider:
# with both expressed in absolute data-units instead, a small height + large
# spacing combination could shrink every trace to sub-pixel amplitude
# (technically still "correct" data, but visually indistinguishable from
# empty) -- discovered by testing that exact combination and confirming
# server- and browser-side data were fine even though the render looked
# blank.
_MONTAGE_ROW_UNIT = 1.0


def _distinct_colors(n: int) -> list[str]:
    """`n` hues spread evenly around the color wheel (fixed saturation/
    lightness) as hex colors -- the all-channels montage needs one
    distinguishable color per channel, and the channel count isn't known
    ahead of time (varies by monkey/session/filter toggles), so a fixed
    8-slot categorical palette would start repeating past 8 channels."""
    return [
        "#{:02x}{:02x}{:02x}".format(*(round(c * 255) for c in colorsys.hls_to_rgb(i / n, 0.5, 0.55)))
        for i in range(n)
    ]


def _psd_all_trials(x: np.ndarray, fsample: float) -> tuple[np.ndarray, np.ndarray]:
    """Multitaper PSD of `x` (`(n_times,)` or `(n_trials, n_times)`), averaged
    over trials -- matching xr_psd_array_multitaper's own trial averaging
    (mean of per-trial, already taper-averaged, power estimates)."""
    psd, freqs = psd_array_multitaper(
        x,
        sfreq=fsample,
        fmin=_PSD_FMIN,
        fmax=min(_PSD_FMAX, fsample / 2),
        bandwidth=_PSD_BANDWIDTH,
        n_jobs=1,
        verbose=False,
    )
    if psd.ndim > 1:
        psd = psd.mean(axis=0)
    return freqs, psd


def _coherence_all_trials(
    x: np.ndarray, y: np.ndarray, fsample: float
) -> tuple[np.ndarray, np.ndarray]:
    """Multitaper coherence between `x` and `y` (`(n_trials, n_times)`),
    averaged over trials and tapers -- matching `_coh`'s
    `(w * conj(w)).mean((trials, tapers))` pattern in
    phase_coupling_analysis/src/metrics/spectral.py.
    """
    kw = dict(
        sfreq=fsample,
        fmin=_COH_FMIN,
        fmax=min(_COH_FMAX, fsample / 2),
        bandwidth=_COH_BANDWIDTH,
        output="complex",
        n_jobs=1,
        verbose=False,
    )
    wx, freqs, _ = psd_array_multitaper(x, **kw)
    wy, _, _ = psd_array_multitaper(y, **kw)
    axes = (0, 1) if wx.ndim == 3 else 0  # (trials, tapers, freqs) vs (tapers, freqs)
    sxx = (wx * np.conj(wx)).mean(axis=axes).real
    syy = (wy * np.conj(wy)).mean(axis=axes).real
    sxy = (wy * np.conj(wx)).mean(axis=axes)
    coh = np.abs(sxy) ** 2 / (sxx * syy)
    return freqs, coh


def _gc_all_trials(
    x: np.ndarray, y: np.ndarray, fsample: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pairwise non-parametric spectral Granger causality between `x` and `y`
    (`(n_trials, n_times)`), matching `conn_gc_average`'s pyGC call in
    phase_coupling_analysis/src/metrics/granger.py: one multitaper
    cross-spectral density + Wilson (1972) spectral factorization per pair,
    which returns both directions from a single factorization. As in
    `conn_gc_average`, `fmin`/`fmax` are applied only *after* factorization
    (Wilson factorization needs the cross-spectral density sampled uniformly
    over the full 0-Nyquist band to correctly reconstruct the causal transfer
    function) -- unlike `_coherence_all_trials`, which can pass them straight
    into the spectral estimator.

    Returns `(freqs, gc_x_to_y, gc_y_to_x)`.
    """
    pair = np.stack([x, y], axis=1)  # (trials, 2, times)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        ds_gc = spectral_granger_causality(
            pair,
            fsample,
            spectral_method="multitaper",
            backend="numpy",
            verbose=False,
            spectral_params={"bandwidth": _COH_BANDWIDTH, "n_jobs": 1},
        )
    ix2y, iy2x, freqs = ds_gc["x2y"][0], ds_gc["y2x"][0],ds_gc["freq"]
    freqs = np.asarray(freqs)
    fmask = (freqs >= _COH_FMIN) & (freqs <= min(_COH_FMAX, fsample / 2))
    return freqs[fmask], np.asarray(ix2y)[fmask], np.asarray(iy2x)[fmask]


def _hilbert_analytic(x: np.ndarray, fsample: float, low: float, high: float) -> np.ndarray:
    """Analytic signal for envelope/phase, matching
    phase_coupling_analysis/src/metrics/phase.py's hilbert_decomposition: that
    function applies scipy.signal.hilbert to data that's already been
    band-filtered upstream (in phasedifferences.py), so we filter first too --
    instantaneous phase/envelope of a broadband signal isn't physically
    meaningful. Works on `(n_times,)` or `(n_trials, n_times)` -- both
    bandpass_filter and scipy.signal.hilbert operate independently along the
    last axis, so trials never bleed into each other.
    """
    filtered = bandpass_filter(x, fsample, low, high)
    return scipy_hilbert(filtered, axis=-1)


def _spike_triggered_average(
    lfp_all_trials: np.ndarray,
    spikes_all_trials: np.ndarray,
    fsample: float,
    half_window: float = _STA_HALF_WINDOW,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Average LFP waveform (same units as `lfp_all_trials`) in a window
    around every spike, pooled across all trials. Spikes too close to a
    trial's edge to fit the full window are skipped."""
    n_half = int(round(half_window * fsample))
    n_trials, n_times = lfp_all_trials.shape
    segments = []
    for tr in range(n_trials):
        for si in np.where(spikes_all_trials[tr])[0]:
            start, end = si - n_half, si + n_half
            if start < 0 or end > n_times:
                continue
            segments.append(lfp_all_trials[tr, start:end])
    if not segments:
        return None, None
    sta = np.mean(segments, axis=0)
    t_rel = np.arange(-n_half, n_half) / fsample
    return t_rel, sta


def _phase_amplitude_bins(
    phase: np.ndarray, values: np.ndarray, n_bins: int, reduce: str
) -> tuple[np.ndarray, np.ndarray]:
    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_idx = np.clip(np.digitize(phase, bin_edges) - 1, 0, n_bins - 1)
    out = np.zeros(n_bins)
    for b in range(n_bins):
        mask = bin_idx == b
        if not mask.any():
            continue
        out[b] = values[mask].mean() if reduce == "mean" else mask.sum()
    return bin_centers, out


def _pac_modulation_index(
    x_all_trials: np.ndarray,
    fsample: float,
    phase_band: tuple[float, float],
    amp_band: tuple[float, float],
    n_bins: int = _PAC_N_BINS,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Tort et al. (2010) modulation index: bin the amplitude envelope of
    `amp_band` by the phase of `phase_band` (same channel, both derived from
    `x_all_trials`, pooled across all trials), then MI = (log(n_bins) -
    Shannon entropy of the resulting distribution) / log(n_bins).
    """
    phase = np.angle(_hilbert_analytic(x_all_trials, fsample, *phase_band)).ravel()
    amplitude = np.abs(_hilbert_analytic(x_all_trials, fsample, *amp_band)).ravel()

    bin_centers, mean_amp = _phase_amplitude_bins(phase, amplitude, n_bins, reduce="mean")
    total = mean_amp.sum()
    p = mean_amp / total if total > 0 else np.full(n_bins, 1.0 / n_bins)
    p_safe = np.where(p > 0, p, 1.0)  # log(1) = 0, contributes nothing -- avoids log(0)
    entropy = -np.sum(p * np.log(p_safe))
    mi = (np.log(n_bins) - entropy) / np.log(n_bins)
    return bin_centers, mean_amp, float(mi)


def _spike_phase_locking(
    x_all_trials: np.ndarray,
    spikes_all_trials: np.ndarray,
    fsample: float,
    band: tuple[float, float],
    n_bins: int = _SPIKE_PHASE_N_BINS,
) -> tuple[np.ndarray, np.ndarray, float] | tuple[None, None, None]:
    """Phase (in `band`) at every spike, pooled across all trials: returns a
    normalized spike-phase histogram plus the mean resultant length (vector
    strength `R`, 0 = no locking, 1 = perfect locking).
    """
    phase_all = np.angle(_hilbert_analytic(x_all_trials, fsample, *band))
    spike_phases = phase_all[spikes_all_trials.astype(bool)]
    if spike_phases.size == 0:
        return None, None, None
    resultant = float(np.abs(np.mean(np.exp(1j * spike_phases))))
    bin_centers, counts = _phase_amplitude_bins(
        spike_phases, np.ones_like(spike_phases), n_bins, reduce="count"
    )
    density = counts / counts.sum() if counts.sum() > 0 else counts
    return bin_centers, density, resultant


def _zscore(x: np.ndarray) -> np.ndarray:
    """`(x - mean) / std`, NaN-aware (the eye traces can have a NaN tail --
    see `load_session`'s `load_eye` docstring) and NaN-propagating (Bokeh
    skips NaN samples when drawing a line, i.e. it renders as a gap)."""
    std = np.nanstd(x)
    return (x - np.nanmean(x)) / std if std > 0 else np.zeros_like(x)


def _rank_percentile(x: np.ndarray) -> np.ndarray:
    """Percentile rank (0-1) of every value in `x`, pooled across the whole
    array (not row-wise) -- `scipy.stats.rankdata` handles ties."""
    flat = x.ravel()
    return (scipy_stats.rankdata(flat) / flat.size).reshape(x.shape)


def _quantile_thresholds(x: np.ndarray, edges: np.ndarray = _QUANTILE_EDGES) -> np.ndarray:
    return np.quantile(x.ravel(), edges)


def _phase_difference_all_trials(analytic1: np.ndarray, analytic2: np.ndarray) -> np.ndarray:
    diff = np.angle(analytic1) - np.angle(analytic2)
    return (diff + np.pi) % (2 * np.pi) - np.pi


def _circular_hist(
    ax, x: np.ndarray, bins: int = _CIRCULAR_HIST_BINS, color: str = "b", alpha: float = 0.6
) -> None:
    """Area-true circular histogram: bin *radius* is set so bin *area* (not
    radius) is proportional to frequency, which is what makes it read
    correctly as a density -- a naive radius-proportional histogram visually
    over-emphasizes high-count bins. Ported from
    plot_phase_coupling_analysis/notebooks/plot_.py's circular_hist.
    """
    x = (x + np.pi) % (2 * np.pi) - np.pi
    bin_edges = np.linspace(-np.pi, np.pi, bins + 1)
    counts, bin_edges = np.histogram(x, bins=bin_edges)
    widths = np.diff(bin_edges)
    area = counts / x.size if x.size else counts.astype(float)
    radius = np.sqrt(area / np.pi)
    ax.bar(
        bin_edges[:-1], radius, width=widths, align="edge",
        edgecolor="k", fill=True, linewidth=1, alpha=alpha, color=color,
    )
    ax.set_yticks([])


@pn.cache
def _cached_metadata(config: DataConfig, monkey: str, date: str, session: int) -> SessionMetadata:
    return load_session_metadata(monkey, date, session, config=config)


def _detect_microsaccades(
    eye,
    fsample,
    velocity_threshold=10.0,
    lowpass_hz=40.0,
    min_duration_ms=10.0,
    min_separation_ms=50.0,
):
    """
    Detect microsaccades from calibrated eye position.

    Parameters
    ----------
    eye : np.ndarray
        Eye position with shape (n_trials, 2, n_times), where axis 1 is
        [horizontal, vertical] and position is in degrees of visual angle.
    fsample : float
        Eye sampling frequency in Hz.
    velocity_threshold : float
        Microsaccade velocity threshold in dva/s.
    lowpass_hz : float
        Low-pass cutoff frequency for eye position, in Hz.
    min_duration_ms : float
        Minimum duration of a microsaccade in milliseconds.
    min_separation_ms : float
        Minimum separation between microsaccades in milliseconds.

    Returns
    -------
    result : dict
        Dictionary containing:

        "filtered_eye"
            Low-pass filtered eye position, same shape as `eye`.

        "velocity"
            2-D eye velocity magnitude in dva/s, shape
            (n_trials, n_times).

        "is_microsaccade"
            Boolean mask of detected microsaccade samples, shape
            (n_trials, n_times).

        "events"
            List of lists. Each trial contains dictionaries with:
            start_idx, end_idx, peak_idx, start_time, end_time,
            peak_time, duration_ms, peak_velocity.
    """
    eye = np.asarray(eye, dtype=float)

    if eye.ndim != 3:
        raise ValueError(
            "eye must have shape (n_trials, 2, n_times)"
        )

    if eye.shape[1] != 2:
        raise ValueError(
            "eye axis 1 must contain [horizontal, vertical]"
        )

    n_trials, _, n_times = eye.shape

    # ------------------------------------------------------------------
    # 1. Low-pass filter eye position at 40 Hz
    # ------------------------------------------------------------------
    if lowpass_hz >= fsample / 2:
        raise ValueError("lowpass_hz must be below the Nyquist frequency")

    sos = butter(
        N=4,
        Wn=lowpass_hz,
        btype="lowpass",
        fs=fsample,
        output="sos",
    )

    filtered_eye = np.full_like(eye, np.nan)

    for trial in range(n_trials):
        for axis in range(2):
            x = eye[trial, axis]

            valid = np.isfinite(x)

            # Avoid filtering NaNs.
            if valid.sum() >= 10:
                filtered_eye[trial, axis, valid] = sosfiltfilt(
                    sos,
                    x[valid],
                )

    # ------------------------------------------------------------------
    # 2. Differentiate position -> velocity
    # ------------------------------------------------------------------
    # np.gradient gives units of dva/s because position is dva and
    # spacing is seconds.
    vx = np.gradient(filtered_eye[:, 0], 1.0 / fsample, axis=-1)
    vy = np.gradient(filtered_eye[:, 1], 1.0 / fsample, axis=-1)

    velocity = np.sqrt(vx**2 + vy**2)

    # ------------------------------------------------------------------
    # 3. Threshold velocity
    # ------------------------------------------------------------------
    supra_threshold = velocity >= velocity_threshold

    # Convert temporal criteria from ms to samples.
    min_duration_samples = max(
        1,
        int(np.ceil(min_duration_ms / 1000.0 * fsample)),
    )

    min_separation_samples = max(
        1,
        int(np.ceil(min_separation_ms / 1000.0 * fsample)),
    )

    # ------------------------------------------------------------------
    # 4. Detect contiguous supra-threshold periods
    # ------------------------------------------------------------------
    is_microsaccade = np.zeros(
        (n_trials, n_times),
        dtype=bool,
    )

    events = []

    for trial in range(n_trials):
        mask = supra_threshold[trial]

        # Find contiguous True regions.
        padded = np.pad(mask.astype(int), (1, 1))
        changes = np.diff(padded)

        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1) - 1

        candidates = []

        for start, end in zip(starts, ends):
            duration = end - start + 1

            if duration >= min_duration_samples:
                peak_idx = start + np.argmax(
                    velocity[trial, start:end + 1]
                )

                candidates.append(
                    {
                        "start_idx": int(start),
                        "end_idx": int(end),
                        "peak_idx": int(peak_idx),
                        "peak_velocity": float(
                            velocity[trial, peak_idx]
                        ),
                    }
                )

        # --------------------------------------------------------------
        # 5. Enforce minimum 50 ms separation
        # --------------------------------------------------------------
        accepted = []

        for candidate in candidates:
            if not accepted:
                accepted.append(candidate)
                continue

            previous = accepted[-1]

            separation = (
                candidate["start_idx"]
                - previous["end_idx"]
                - 1
            )

            if separation >= min_separation_samples:
                accepted.append(candidate)
            else:
                # If two events are too close, retain the one with the
                # larger peak velocity.
                if (
                    candidate["peak_velocity"]
                    > previous["peak_velocity"]
                ):
                    accepted[-1] = candidate

        # Mark accepted events.
        trial_events = []

        for event in accepted:
            start = event["start_idx"]
            end = event["end_idx"]
            peak = event["peak_idx"]

            is_microsaccade[trial, start:end + 1] = True

            event["start_time"] = start / fsample
            event["end_time"] = end / fsample
            event["peak_time"] = peak / fsample
            event["duration_ms"] = (
                (end - start + 1) / fsample * 1000
            )

            trial_events.append(event)

        events.append(trial_events)

    return {
        "filtered_eye": filtered_eye,
        "velocity": velocity,
        "is_microsaccade": is_microsaccade,
        "events": events,
    }

@pn.cache
def _cached_session(
    config: DataConfig,
    monkey: str,
    date: str,
    session: int,
    align_to: str,
    exclude_slvr_msmod: bool,
    only_unique_recordings: bool,
):
    return load_session(
        monkey,
        date,
        session,
        align_to=align_to,
        exclude_slvr_msmod=False,
        only_unique_recordings=False,
        load_spike_times=True,
        load_eye=True,
        config=config,
    )


def _trial_label(trial_info: pd.DataFrame, trial_index: int) -> str:
    # `trial_index` is the row's *position* in the raw (unfiltered) trial_info
    # table -- see session.py, which builds `ds.trials` from exactly that.
    row = trial_info.iloc[trial_index]
    ttype = TrialType(int(row["trial_type"])).name
    resp = row.get("behavioral_response")
    resp_label = BehavioralResponse(int(resp)).name if pd.notna(resp) else "N/A"
    return f"Trial {trial_index} — {ttype}, {resp_label}"


def _legend_above(ax, n_entries: int, fontsize: int = 8, handles=None, labels=None) -> None:
    """Legend as a horizontal strip above the axes (never inside the plot
    area, where it covers data) -- combine with `pad=` on `ax.set_title` (if
    any) so the title sits above this strip rather than overlapping it.
    `handles`/`labels` let a twin-axes plot (e.g. `_dual_psd_figure`) combine
    both axes' entries into one legend instead of `ax`'s own alone."""
    kw = {} if handles is None else dict(handles=handles, labels=labels)
    ax.legend(
        **kw,
        loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=min(max(n_entries, 1), 4),
        fontsize=fontsize, frameon=False, handlelength=1.5, columnspacing=1.2,
        borderaxespad=0,
    )


def _psd_figure(
    series: list[tuple[np.ndarray, np.ndarray, str, str]],
    figsize: tuple[float, float],
    title: str | None = None,
) -> Figure:
    fig = Figure(figsize=figsize)
    ax = fig.add_subplot(111)
    for freqs, psd, color, label in series:
        ax.plot(freqs, psd, color=color, lw=1.5, label=label)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power spectral density (µV²/Hz)")
    if title:
        ax.set_title(title, fontsize=9, pad=26)
    _legend_above(ax, len(series))
    return fig


def _dual_psd_figure(
    series1: list[tuple[np.ndarray, np.ndarray, str, str]],
    series2: list[tuple[np.ndarray, np.ndarray, str, str]],
    label1: str,
    label2: str,
    figsize: tuple[float, float],
) -> Figure:
    """Both channels' power spectra on one plot, `label1` on the left y-axis
    and `label2` on the right (`ax.twinx()`), sharing the x-axis. Channel
    identity is both color (`series1`/`series2` are pre-colored by the
    caller -- `_COLOR_RAW`/`_COLOR_FILTERED` when not split by stimulus,
    matching the two-channel LFP trace's own fixed channel-1/channel-2
    colors) and linestyle (solid = `label1`, dashed = `label2`), the latter
    still needed on its own when split by stimulus, where color instead
    varies per stimulus and is shared between both channels.
    """
    fig = Figure(figsize=figsize)
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    for freqs, psd, color, label in series1:
        ax1.plot(freqs, psd, color=color, lw=1.5, linestyle="-", label=f"{label1}: {label}")
    for freqs, psd, color, label in series2:
        ax2.plot(freqs, psd, color=color, lw=1.5, linestyle="--", label=f"{label2}: {label}")
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel(f"PSD (µV²/Hz) — {label1} (solid)")
    ax2.set_ylabel(f"PSD (µV²/Hz) — {label2} (dashed)")
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    _legend_above(ax1, len(handles1) + len(handles2), handles=handles1 + handles2, labels=labels1 + labels2)
    return fig


def _make_linked_time_series_figure(x_range, y_axis_label: str, height: int = 180):
    """A small Bokeh line-chart factory for panels that should pan/zoom in
    lockstep with the main LFP chart: sharing `x_range` (the same Bokeh Range
    object, not a copy) means dragging/scrolling on any one of them moves all
    of them together.
    """
    fig = bokeh_figure(
        height=height,
        sizing_mode="stretch_width",
        x_range=x_range,
        tools="pan,box_zoom,wheel_zoom,reset,save",
        active_drag="box_zoom",
        active_scroll="wheel_zoom",
        x_axis_label="Time (s)",
        y_axis_label=y_axis_label,
    )
    fig.toolbar.logo = None
    source_1 = ColumnDataSource(data=dict(x=[], y=[]))
    source_2 = ColumnDataSource(data=dict(x=[], y=[]))
    line_1 = fig.line("x", "y", source=source_1, color=_COLOR_RAW, line_width=1.2, legend_label="Channel 1")
    line_2 = fig.line("x", "y", source=source_2, color=_COLOR_FILTERED, line_width=1.2, legend_label="Channel 2")
    fig.legend.visible = False
    fig.legend.click_policy = "hide"
    fig.legend.background_fill_alpha = 0
    fig.legend.border_line_alpha = 0
    fig.legend.orientation = "horizontal"
    fig.legend.location = "top_left"
    return fig, (source_1, source_2), (line_1, line_2)


def _coherence_figure(
    series: list[tuple[np.ndarray, np.ndarray, str, str]],
    label1: str,
    label2: str,
    figsize: tuple[float, float],
) -> Figure:
    fig = Figure(figsize=figsize)
    ax = fig.add_subplot(111)
    for freqs, coh, color, label in series:
        ax.plot(freqs, coh, color=color, lw=1.5, label=label)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Coherence")
    ax.set_title(f"Coherence: {label1} vs {label2}", fontsize=9, pad=26)
    _legend_above(ax, len(series))
    return fig


def _gc_figure(
    series: list[tuple[np.ndarray, np.ndarray, np.ndarray, str, str, str]],
    label1: str,
    label2: str,
    figsize: tuple[float, float],
) -> Figure:
    """`series`: `(freqs, gc_x_to_y, gc_y_to_x, color_xy, color_yx, label)`
    -- one entry per stimulus group (or a single ungrouped entry), each
    contributing two lines: solid `label1`->`label2` in `color_xy` (the
    color of the channel the arrow originates from), dashed `label2`->
    `label1` in `color_yx`. When not split by stimulus, `color_xy`/
    `color_yx` are `_COLOR_RAW`/`_COLOR_FILTERED` -- the same fixed identity
    colors as `label1`/`label2`'s own PSD lines and the LFP trace -- so a
    direction's line always matches its source channel's color elsewhere on
    the page; when split by stimulus the two are equal (that stimulus's
    color) and only linestyle tells the directions apart."""
    fig = Figure(figsize=figsize)
    ax = fig.add_subplot(111)
    for freqs, gc_xy, gc_yx, color_xy, color_yx, glabel in series:
        prefix = f"{glabel} " if glabel else ""
        ax.plot(freqs, gc_xy, color=color_xy, lw=1.5, linestyle="-", label=f"{prefix}{label1}→{label2}")
        ax.plot(freqs, gc_yx, color=color_yx, lw=1.5, linestyle="--", label=f"{prefix}{label2}→{label1}")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Granger causality")
    ax.set_title(f"Spectral GC: {label1} vs {label2}", fontsize=9, pad=26)
    _legend_above(ax, len(series) * 2)
    return fig


def _gc_unavailable_figure(figsize: tuple[float, float]) -> Figure:
    fig = Figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.text(
        0.5, 0.5, "pyGC not installed\n(pip install -e ../pyGC)",
        ha="center", va="center", fontsize=8, color="0.5", transform=ax.transAxes,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return fig


def _sta_figure(
    series: list[tuple[np.ndarray, np.ndarray, str, str, str]],
    figsize: tuple[float, float],
    title: str = "Spike-triggered average",
) -> Figure:
    """`series`: (t_rel, sta_wave_uv, color, label, linestyle)."""
    fig = Figure(figsize=figsize)
    ax = fig.add_subplot(111)
    for t_rel, sta, color, label, linestyle in series:
        ax.plot(t_rel, sta, color=color, lw=1.5, label=label, linestyle=linestyle)
    ax.axvline(0, color="0.6", lw=0.8, linestyle="--")
    ax.set_xlabel("Time from spike (s)")
    ax.set_ylabel("LFP (µV)")
    ax.set_title(title, fontsize=9, pad=26)
    _legend_above(ax, len(series))
    return fig


def _phase_bar_figure(
    results: list[dict],
    xlabel: str,
    ylabel: str,
    title: str,
    figsize: tuple[float, float],
) -> Figure:
    """`results`: dicts with `bin_centers`, `values`, `metric_label`, `color`, `label`."""
    fig = Figure(figsize=figsize)
    ax = fig.add_subplot(111)
    n = max(len(results), 1)
    full_width = (2 * np.pi / len(results[0]["bin_centers"])) * 0.85 if results else 0.3
    bar_width = full_width / n
    for i, r in enumerate(results):
        offset = (i - (n - 1) / 2) * bar_width
        ax.bar(
            r["bin_centers"] + offset, r["values"], width=bar_width,
            color=r["color"], alpha=0.85, label=f'{r["label"]} ({r["metric_label"]})',
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9, pad=26)
    ax.set_xticks([-np.pi, 0, np.pi])
    ax.set_xticklabels(["-π", "0", "π"])
    _legend_above(ax, len(results), fontsize=7)
    return fig


def _phase_diff_circular_figure(
    bin_results: list[dict], figsize: tuple[float, float], suptitle: str | None = None
) -> Figure:
    """`bin_results`: dicts with `label`, `phases` (1D array, radians), `color`.
    One polar subplot per bin, side by side, each an area-true circular
    histogram of the phase difference with its circular mean/std in the
    title. `suptitle` states the sign convention (phase(A) - phase(B)) used
    for every subplot -- circular std is sign-symmetric, but the mean/the
    histogram's orientation flips if A and B are swapped, so this is the
    only thing that tells you which way round it was computed.
    """
    n = max(len(bin_results), 1)
    fig = Figure(figsize=figsize)
    if suptitle:
        fig.suptitle(suptitle, fontsize=8)
    for i, r in enumerate(bin_results):
        ax = fig.add_subplot(1, n, i + 1, projection="polar")
        phases = r["phases"]
        if phases.size:
            _circular_hist(ax, phases, color=r["color"])
            mu_deg = np.degrees(scipy_stats.circmean(phases, high=np.pi, low=-np.pi))
            std_deg = np.degrees(scipy_stats.circstd(phases, nan_policy="omit"))
            ax.set_title(f'{r["label"]}\nμ={mu_deg:.1f}°, σ={std_deg:.1f}°', fontsize=9)
        else:
            ax.set_title(f'{r["label"]}\n(no samples)', fontsize=9)
    return fig

def _label(roi, ch, slvr, ms_mod):
    flags = [f for f, on in (("slvr", slvr), ("ms_mod", ms_mod)) if on]
    suffix = f"; {' '.join(flags)}" if flags else ""
    return f"{roi} (ch {ch}{suffix})"

def build_app(config: DataConfig | None = None) -> pn.viewable.Viewable:
    config = config or default_config()

    monkeys = list_monkeys(config)
    default_monkey = _DEFAULT_MONKEY if _DEFAULT_MONKEY in monkeys else (monkeys[0] if monkeys else None)
    monkey_select = pn.widgets.Select(label="Monkey", options=monkeys, value=default_monkey)
    date_select = pn.widgets.Select(label="Date", options=[])
    session_select = pn.widgets.Select(label="Session", options=[1])
    align_select = pn.widgets.RadioButtonGroup(
        label="Align to", options=["cue", "match"], value="cue"
    )
    trial_select = pn.widgets.Select(label="Trial", options={})
    channel_select = pn.widgets.MultiSelect(
        label="Channel(s) — select up to 2 (2nd shows coherence)", options={}, size=8
    )
    clear_channels_button = pn.widgets.Button(label="Clear selection")
    include_flagged_channels = pn.widgets.Checkbox(
        label="Include slvr/ms_mod-flagged channels", value=True
    )
    unique_recordings_only = pn.widgets.Checkbox(
        label="Unique recordings only", value=True
    )
    
    # --- Raw data plot: everything that changes what the LFP trace (and its
    # direct spectral view -- PSD/coherence) shows for the selected trial.
    zoom_start = pn.widgets.FloatInput(label="Zoom start (s)", value=0.0)
    zoom_end = pn.widgets.FloatInput(label="Zoom end (s)", value=0.0)
    reset_zoom_button = pn.widgets.Button(label="Reset zoom")
    show_spikes = pn.widgets.Checkbox(label="Overlay spikes", value=True)
    show_events = pn.widgets.Checkbox(label="Show cue onset/offset & match onset", value=True)
    filter_enabled = pn.widgets.Checkbox(
        label="Apply bandpass filter (trace, PSD, coherence & GC)", value=False
    )
    raw_band_select = pn.widgets.Select(label="Band preset", options=["custom"])
    raw_custom_low = pn.widgets.FloatInput(label="Low (Hz)", value=8.0, start=0.0)
    raw_custom_high = pn.widgets.FloatInput(label="High (Hz)", value=12.0, start=0.1)
    show_gc = pn.widgets.Checkbox(
        label="Granger causality spectrum (2 channels)",
        value=False,
        disabled=not _PYGC_AVAILABLE,
    )

    # --- Spike-triggered average: its own filter/band, independent of the
    # raw trace's -- so the STA waveform doesn't silently change just because
    # the raw trace's display filter was toggled for an unrelated reason.
    show_sta = pn.widgets.Checkbox(label="Spike-triggered average (µV)", value=False)
    show_cross_sta = pn.widgets.Checkbox(
        label="Include cross-channel STA (2 channels: A on B's spikes, B on A's spikes)",
        value=False,
    )
    sta_filter_enabled = pn.widgets.Checkbox(label="Apply bandpass filter to STA", value=False)
    sta_band_select = pn.widgets.Select(label="Band preset", options=["custom"])
    sta_custom_low = pn.widgets.FloatInput(label="Low (Hz)", value=8.0, start=0.0)
    sta_custom_high = pn.widgets.FloatInput(label="High (Hz)", value=12.0, start=0.1)

    # --- Phase-amplitude coupling: fully self-contained (its own phase/amp
    # bands), nothing else here to decouple.
    show_pac = pn.widgets.Checkbox(label="Phase-amplitude coupling (Tort MI)", value=False)
    pac_phase_low = pn.widgets.FloatInput(label="PAC phase band low (Hz)", value=4.0, start=0.0)
    pac_phase_high = pn.widgets.FloatInput(label="PAC phase band high (Hz)", value=8.0, start=0.1)
    pac_amp_low = pn.widgets.FloatInput(label="PAC amplitude band low (Hz)", value=30.0, start=0.0)
    pac_amp_high = pn.widgets.FloatInput(label="PAC amplitude band high (Hz)", value=80.0, start=0.1)

    # --- Phase coupling: Hilbert decomposition, spike-phase locking, the
    # power-product quantile regions, and the phase-difference circular plot
    # all derive from the same analytic signal, so they share one band here.
    hilbert_mode = pn.widgets.Select(
        label="Hilbert decomposition (uses the band below)",
        options=["None", "Envelope", "Phase", "Envelope + Phase"],
        value="None",
    )
    phase_band_select = pn.widgets.Select(label="Band preset", options=["custom"])
    phase_custom_low = pn.widgets.FloatInput(label="Low (Hz)", value=8.0, start=0.0)
    phase_custom_high = pn.widgets.FloatInput(label="High (Hz)", value=12.0, start=0.1)
    show_spike_phase = pn.widgets.Checkbox(
        label="Spike-phase locking (uses the band above)", value=False
    )
    show_quantile_regions = pn.widgets.Checkbox(
        label="Power-product quantile regions (2 channels, uses the band above)", value=False
    )
    phase_diff_bins = pn.widgets.MultiSelect(
        label="Phase-difference circular plot: quantile bin(s)",
        options=_QUANTILE_LABELS, value=[], size=4,
    )

    trial_type_filter = pn.widgets.MultiSelect(
        label="Trial type (none = all)",
        options=[t.name for t in TrialType], value=[], size=3,
    )
    behavioral_response_filter = pn.widgets.MultiSelect(
        label="Behavioral response (none = all)",
        options=[r.name for r in BehavioralResponse], value=[], size=2,
    )
    stimulus_filter = pn.widgets.MultiSelect(
        label="Stimulus label (none = all)", options=[], value=[], size=5,
    )
    clear_trial_subset_button = pn.widgets.Button(label="Clear selection")
    subset_info_pane = pn.pane.Markdown("", styles={"font-size": "0.85em", "color": "#666"})

    info_pane = pn.pane.Markdown("")

    lfp_bokeh = bokeh_figure(
        height=350,
        sizing_mode="stretch_width",
        tools="pan,box_zoom,wheel_zoom,reset,save",
        active_drag="box_zoom",
        active_scroll="wheel_zoom",
        x_axis_label="Time (s)",
        y_axis_label="LFP (µV)",
    )
    lfp_bokeh.toolbar.logo = None
    lfp_source_1 = ColumnDataSource(data=dict(x=[], y=[]))
    lfp_source_2 = ColumnDataSource(data=dict(x=[], y=[]))
    lfp_line_1 = lfp_bokeh.line("x", "y", source=lfp_source_1, color=_COLOR_RAW, line_width=1.2, legend_label="Channel 1")
    lfp_line_2 = lfp_bokeh.line("x", "y", source=lfp_source_2, color=_COLOR_FILTERED, line_width=1.2, legend_label="Channel 2")
    spike_source_1 = ColumnDataSource(data=dict(x=[], y=[]))
    spike_source_2 = ColumnDataSource(data=dict(x=[], y=[]))
    # Each channel's spikes are colored to match that channel's own LFP line
    # (same convention as the old matplotlib rendering) -- both used to be
    # hardcoded to the same red, which made channel 2's row (only ~5% of the
    # y-range above channel 1's) very easy to miss, especially when spike
    # times correlate between the two channels.
    spike_scatter_1 = lfp_bokeh.scatter(
        "x", "y", source=spike_source_1, marker="dash", size=14, angle=1.5708,
        color=_COLOR_RAW, legend_label="Ch1 spikes",
    )
    spike_scatter_2 = lfp_bokeh.scatter(
        "x", "y", source=spike_source_2, marker="dash", size=14, angle=1.5708,
        color=_COLOR_FILTERED, legend_label="Ch2 spikes",
    )
    event_spans = {
        "cue_on": Span(location=0, dimension="height", line_color="gray", line_dash="solid", line_width=1),
        "cue_off": Span(location=0, dimension="height", line_color="gray", line_dash="dashed", line_width=1),
        "match_on": Span(location=0, dimension="height", line_color="gray", line_dash="dotted", line_width=1),
    }
    for span in event_spans.values():
        span.visible = False
        lfp_bokeh.add_layout(span)
    lfp_bokeh.legend.visible = False
    lfp_bokeh.legend.click_policy = "hide"
    lfp_bokeh.legend.background_fill_alpha = 0
    lfp_bokeh.legend.border_line_alpha = 0
    lfp_bokeh.legend.orientation = "horizontal"
    lfp_bokeh.legend.location = "top_left"
    lfp_pane = pn.pane.Bokeh(lfp_bokeh)

    psd_pane = pn.pane.Matplotlib(Figure(figsize=(4, 3)), tight=True, sizing_mode="stretch_width")
    coherence_pane = pn.pane.Matplotlib(
        Figure(figsize=(4, 3)), tight=True, sizing_mode="stretch_width"
    )
    gc_pane = pn.pane.Matplotlib(
        Figure(figsize=(4, 3)), tight=True, sizing_mode="stretch_width"
    )
    envelope_bokeh, envelope_sources, envelope_lines = _make_linked_time_series_figure(
        lfp_bokeh.x_range, "Envelope (µV)"
    )
    envelope_pane = pn.pane.Bokeh(envelope_bokeh)
    phase_bokeh, phase_sources, phase_lines = _make_linked_time_series_figure(
        lfp_bokeh.x_range, "Phase (rad)", height=150
    )
    phase_bokeh.y_range.start, phase_bokeh.y_range.end = -np.pi, np.pi
    phase_pane = pn.pane.Bokeh(phase_bokeh)

    quantile_bokeh = bokeh_figure(
        height=180,
        sizing_mode="stretch_width",
        x_range=lfp_bokeh.x_range,
        tools="pan,box_zoom,wheel_zoom,reset,save",
        active_drag="box_zoom",
        active_scroll="wheel_zoom",
        x_axis_label="Time (s)",
        y_axis_label="Power-product percentile",
    )
    quantile_bokeh.toolbar.logo = None
    percentile_source = ColumnDataSource(data=dict(x=[], y=[]))
    quantile_bokeh.line(
        "x", "y", source=percentile_source, color="#666666", line_width=1.2, legend_label="percentile"
    )
    quantile_band_sources = [ColumnDataSource(data=dict(x=[], y=[])) for _ in range(4)]
    for i in range(4):
        quantile_bokeh.line(
            "x", "y", source=quantile_band_sources[i], color=_QUANTILE_COLORS[i],
            line_width=5, legend_label=_QUANTILE_LABELS[i],
        )
    quantile_bokeh.legend.click_policy = "hide"
    quantile_bokeh.legend.background_fill_alpha = 0
    quantile_bokeh.legend.border_line_alpha = 0
    quantile_bokeh.legend.orientation = "horizontal"
    quantile_bokeh.legend.location = "top_left"
    quantile_bokeh.legend.label_text_font_size = "8pt"
    quantile_pane = pn.pane.Bokeh(quantile_bokeh)

    # All-channels montage (second tab): every currently-loaded LFP channel
    # plus both eye traces, z-scored and stacked with a fixed vertical offset
    # per trace -- one shared x-range with the main LFP chart, so zooming
    # either one zooms both.
    montage_bokeh = bokeh_figure(
        height=1000,
        sizing_mode="stretch_width",
        x_range=lfp_bokeh.x_range,
        tools="pan,box_zoom,wheel_zoom,reset,save",
        active_drag="box_zoom",
        active_scroll="wheel_zoom",
        x_axis_label="Time (s)",
    )
    montage_bokeh.toolbar.logo = None
    montage_bokeh.yaxis.axis_label = None
    montage_source = ColumnDataSource(data=dict(xs=[], ys=[], color=[]))
    montage_bokeh.multi_line("xs", "ys", source=montage_source, line_color="color", line_width=1)
    montage_event_spans = {
        "cue_on": Span(location=0, dimension="height", line_color="gray", line_dash="solid", line_width=1),
        "cue_off": Span(location=0, dimension="height", line_color="gray", line_dash="dashed", line_width=1),
        "match_on": Span(location=0, dimension="height", line_color="gray", line_dash="dotted", line_width=1),
    }
    for span in montage_event_spans.values():
        span.visible = False
        montage_bokeh.add_layout(span)
    montage_pane = pn.pane.Bokeh(montage_bokeh)
    montage_caption = pn.pane.Markdown(
        styles={"font-size": "0.85em", "color": "#666"},
    )
    montage_channel_select = pn.widgets.MultiSelect(
        label="Channels to display", options={}, size=8
    )
    montage_clear_channels_button = pn.widgets.Button(label="Clear selection")
    montage_height_slider = pn.widgets.IntSlider(
        label="Plot height (px)", start=300, end=3000, step=50, value=_MONTAGE_DEFAULT_HEIGHT
    )
    montage_spacing_slider = pn.widgets.IntSlider(
        label="Trace spacing", start=2, end=40, step=1, value=_MONTAGE_DEFAULT_SPACING
    )

    sta_pane = pn.pane.Matplotlib(Figure(figsize=(4, 3)), tight=True, sizing_mode="stretch_width")
    cross_sta_pane = pn.pane.Matplotlib(Figure(figsize=(4, 3)), tight=True, sizing_mode="stretch_width")
    pac_pane = pn.pane.Matplotlib(Figure(figsize=(4, 3)), tight=True, sizing_mode="stretch_width")
    spike_phase_pane = pn.pane.Matplotlib(Figure(figsize=(4, 3)), tight=True, sizing_mode="stretch_width")
    phase_diff_pane = pn.pane.Matplotlib(Figure(figsize=(4, 3)), tight=True, sizing_mode="stretch_width")

    state: dict = {"metadata": None, "ds": None, "spectral_cache": {}, "full_range": (0.0, 0.0)}

    def _update_dates(_event=None):
        dates = list_dates(monkey_select.value, config) if monkey_select.value else []
        date_select.options = dates
        date_select.value = _DEFAULT_DATE if _DEFAULT_DATE in dates else (dates[0] if dates else None)

    def _update_sessions(_event=None):
        sessions = (
            list_sessions(monkey_select.value, date_select.value, config)
            if monkey_select.value and date_select.value
            else []
        ) or [1]
        session_select.options = sessions
        session_select.value = sessions[0]

    def _update_band_options(_event=None):
        bands = band_presets(monkey_select.value) if monkey_select.value else []
        labels = [f"{lo:g}-{hi:g} Hz" for lo, hi in bands] + ["custom"]
        for select in (raw_band_select, sta_band_select, phase_band_select):
            select.options = labels
            select.value = labels[0]

    def _resolve_band(
        select: pn.widgets.Select, low: pn.widgets.FloatInput, high: pn.widgets.FloatInput
    ) -> tuple[float, float]:
        if select.value == "custom":
            return low.value, high.value
        low_str, high_str = select.value.replace(" Hz", "").split("-")
        return float(low_str), float(high_str)

    def _limit_channel_selection(event):
        if len(event.new) > 2:
            channel_select.value = event.new[:2]

    def _clear_channel_selection(_event=None):
        channel_select.value = []

    def _clear_montage_channel_selection(_event=None):
        montage_channel_select.value = []

    def _clear_trial_subset(_event=None):
        trial_type_filter.value = []
        behavioral_response_filter.value = []
        stimulus_filter.value = []

    def _reset_zoom(_event=None):
        start, end = state["full_range"]
        lfp_bokeh.x_range.start = start
        lfp_bokeh.x_range.end = end
        zoom_start.value = start
        zoom_end.value = end

    def _apply_zoom(_event=None):
        if zoom_end.value > zoom_start.value:
            lfp_bokeh.x_range.start = zoom_start.value
            lfp_bokeh.x_range.end = zoom_end.value

    def _reload_session(_event=None):
        if not (monkey_select.value and date_select.value and session_select.value):
            return
        try:
            metadata = _cached_metadata(
                config, monkey_select.value, date_select.value, session_select.value
            )
            ds = _cached_session(
                config,
                monkey_select.value,
                date_select.value,
                session_select.value,
                align_select.value,
                not include_flagged_channels.value,
                unique_recordings_only.value,
            )
        except GrayDataVizError as exc:
            info_pane.object = f"**Error loading session:** {exc}"
            return

        state["metadata"] = metadata
        state["ds"] = ds
        state["spectral_cache"] = {}

        trial_options = {
            _trial_label(metadata.trial_info, int(t)): int(t) for t in ds.trials.values
        }
        trial_select.options = trial_options
        trial_select.value = next(iter(trial_options.values()))

        channel_options = {
            _label(roi, ch, slvr, ms_mod): i
            for i, (roi, ch, slvr, ms_mod) in enumerate(
                zip(ds.roi.values, ds.attrs["channels_labels"], ds.attrs["slvr"], ds.attrs["ms_mod"])
            )
        }

        channel_select.options = channel_options
        channel_select.value = [next(iter(channel_options.values()))]
        montage_channel_select.options = channel_options
        montage_channel_select.value = list(channel_options.values())

        stim_values = sorted(
            int(v) for v in pd.unique(metadata.trial_info["sample_image"]) if pd.notna(v)
        )
        stimulus_filter.options = [str(v) for v in stim_values]
        stimulus_filter.value = []

        state["full_range"] = (float(ds.time.values[0]), float(ds.time.values[-1]))
        _update_plot()
        _update_montage()
        _reset_zoom()

    def _event_times_for_trial(ds, trial_index, fsample):
        # Shared by the main LFP chart and the all-channels montage.
        if not show_events.value:
            return {}
        pos = int(np.where(ds.trials.values == trial_index)[0][0])
        align_to = ds.attrs["align_to"]
        t0 = (ds.attrs["t_cue_on"] if align_to == "cue" else ds.attrs["t_match_on"])[pos]
        return {
            "cue_on": (ds.attrs["t_cue_on"][pos] - t0) / fsample,
            "cue_off": (ds.attrs["t_cue_off"][pos] - t0) / fsample,
            "match_on": (ds.attrs["t_match_on"][pos] - t0) / fsample,
        }

    def _update_montage(_event=None):
        # All-channels montage tab: intentionally watches only trial_select/
        # show_events (plus session reload) -- not every other-analysis
        # toggle -- so it doesn't get recomputed every time an unrelated
        # checkbox flips.
        ds = state["ds"]
        if ds is None or trial_select.value is None:
            return
        trial_index = trial_select.value
        time = ds.time.values
        fsample = float(ds.attrs["fsample"])
        selected_channels = sorted(montage_channel_select.value)
        n_channels = len(selected_channels)
        has_eye = "eye" in ds

        # amplitude_scale is data-units per z-score std, relative to the
        # fixed 1.0 row unit -- see _MONTAGE_ROW_UNIT's comment for why this
        # (rather than an absolute spacing value) is what the slider drives.
        amplitude_scale = _MONTAGE_ROW_UNIT / float(montage_spacing_slider.value)
        n_eye_rows = 2 if has_eye else 0
        n_microsaccade_rows = 1 if has_eye else 0
        total_rows = n_channels + n_eye_rows + n_microsaccade_rows + (1 if has_eye else 0)  # +1 gap row
        pos = total_rows

        xs, ys, colors, ticks, labels = [], [], [], [], []
        channel_colors = _distinct_colors(n_channels)

        for color_i, i in enumerate(selected_channels):
            raw = ds.lfp.sel(trials=trial_index).isel(roi=i).values * _VOLTS_TO_MICROVOLTS
            y = _zscore(raw) * amplitude_scale + pos * _MONTAGE_ROW_UNIT
            xs.append(time)
            ys.append(y)
            colors.append(channel_colors[color_i])
            ticks.append(pos * _MONTAGE_ROW_UNIT)
            labels.append(f"{ds.roi.values[i]} (ch {ds.attrs['channels_labels'][i]})")
            pos -= 1

        if has_eye:
            pos -= 1  # gap row between LFP channels and eye traces
            eye_trial = ds.eye.sel(trials=trial_index).values  # (eye_axis, time)

            for axis_i, axis_label in enumerate(("Eye H", "Eye V")):
                y = _zscore(eye_trial[axis_i]) * amplitude_scale + pos * _MONTAGE_ROW_UNIT
                xs.append(time)
                ys.append(y)
                colors.append(_COLOR_EYE)
                ticks.append(pos * _MONTAGE_ROW_UNIT)
                labels.append(axis_label)
                pos -= 1

            # _detect_microsaccades expects (n_trials, 2, n_times); eye_trial is
            # alre{ady sliced to this one trial (2, n_times), so add back a dummy
            # trials axis and unwrap the (1, n_times) result with [0]. Use the
            # local `fsample` (ds.attrs["fsample"]) -- `ds.fsample` isn't a real
            # attribute and raises AttributeError.
            result = _detect_microsaccades(
                eye_trial[None, :, :], fsample=fsample
            )
            microsaccades = result["is_microsaccade"][0]
            print(f"microsaccades detected: {microsaccades.sum()} / {len(microsaccades)} samples, "
                f"max velocity: {np.nanmax(result['velocity']):.2f} dva/s, "
                f"eye value range: {np.nanmin(eye_trial):.3f} to {np.nanmax(eye_trial):.3f}")
            y = microsaccades.astype(float) * amplitude_scale + pos * _MONTAGE_ROW_UNIT
            xs.append(time)
            ys.append(y)
            colors.append("#000000")
            ticks.append(pos * _MONTAGE_ROW_UNIT)
            labels.append("microsaccades")
            pos -= 1

        montage_source.data = dict(xs=xs, ys=ys, color=colors)
        montage_bokeh.yaxis.ticker = FixedTicker(ticks=ticks)
        montage_bokeh.yaxis.major_label_overrides = {t: lbl for t, lbl in zip(ticks, labels)}
        montage_bokeh.y_range.start = -_MONTAGE_ROW_UNIT
        montage_bokeh.y_range.end = (total_rows + 1) * _MONTAGE_ROW_UNIT
        montage_bokeh.height = montage_height_slider.value

        event_times = _event_times_for_trial(ds, trial_index, fsample)
        match_on = event_times.get("match_on")
        state["full_range"] = (
            (-0.5, match_on + 0.5)
            if match_on is not None and not np.isnan(match_on)
            else (float(ds.time.values[0]), float(ds.time.values[-1]))
        )  
        for key, sp in montage_event_spans.items():
            t = event_times.get(key)
            if t is not None:
                sp.location = t
                sp.visible = True
            else:
                sp.visible = False

    def _update_plot(_event=None):
        ds = state["ds"]
        metadata = state["metadata"]
        selected = channel_select.value[:2]
        if ds is None or trial_select.value is None or not selected:
            return

        trial_index = trial_select.value
        time = ds.time.values
        fsample = float(ds.attrs["fsample"])
        index_to_label = {i: label for label, i in channel_select.options.items()}
        channel_colors = {selected[0]: _COLOR_RAW}
        if len(selected) > 1:
            channel_colors[selected[1]] = _COLOR_FILTERED

        raw_low = raw_high = None
        raw_band_label = None
        if filter_enabled.value:
            raw_low, raw_high = _resolve_band(raw_band_select, raw_custom_low, raw_custom_high)
            raw_band_label = f"{raw_low:g}-{raw_high:g} Hz"

        sta_low = sta_high = None
        sta_band_label = None
        if sta_filter_enabled.value:
            sta_low, sta_high = _resolve_band(sta_band_select, sta_custom_low, sta_custom_high)
            sta_band_label = f"{sta_low:g}-{sta_high:g} Hz"

        needs_phase_band = (
            hilbert_mode.value != "None"
            or show_spike_phase.value
            or show_quantile_regions.value
            or bool(phase_diff_bins.value)
        )
        phase_low = phase_high = None
        phase_band_label = None
        if needs_phase_band:
            phase_low, phase_high = _resolve_band(
                phase_band_select, phase_custom_low, phase_custom_high
            )
            phase_band_label = f"{phase_low:g}-{phase_high:g} Hz"

        hilbert_modes = []
        if hilbert_mode.value in ("Envelope", "Envelope + Phase"):
            hilbert_modes.append("Envelope")
        if hilbert_mode.value in ("Phase", "Envelope + Phase"):
            hilbert_modes.append("Phase")

        def _cached_spectral(key, compute_fn):
            cache = state["spectral_cache"]
            if key not in cache:
                cache[key] = compute_fn()
            return cache[key]

        def _trial_position():
            return int(np.where(ds.trials.values == trial_index)[0][0])

        def _all_trials_raw(ch_idx):
            return ds.lfp.isel(roi=ch_idx).values * _VOLTS_TO_MICROVOLTS  # (n_trials, n_times)

        def _all_trials_filtered(ch_idx):
            return bandpass_filter(_all_trials_raw(ch_idx), fsample, phase_low, phase_high)

        def _all_trials_spikes(ch_idx):
            return ds.spikes.isel(roi=ch_idx).values.astype(bool) if "spikes" in ds else None

        def _burst_window_mask() -> np.ndarray:
            # Matches phase_coupling_analysis/save_burst_trains.py's
            # trials_length_mask exactly: for the power-product quantile
            # thresholds and the phase-difference circular histogram (NOT
            # PSD/coherence/STA/PAC/spike-phase, which pool the full window),
            # only samples from -0.5s through that trial's own match onset
            # are valid -- everything before that, or the padding after each
            # trial's actual behavioral event, is excluded. Without this the
            # quantile thresholds/phase-difference std come out systematically
            # different from the reference pipeline's saved outputs.
            align_to = ds.attrs["align_to"]
            t0 = ds.attrs["t_cue_on"] if align_to == "cue" else ds.attrs["t_match_on"]
            match_on_rel = (ds.attrs["t_match_on"] - t0) / fsample  # (n_trials,)
            return (time[None, :] >= -0.5) & (time[None, :] <= match_on_rel[:, None])

        def _trial_type_behavior_mask() -> np.ndarray:
            rows = metadata.trial_info.iloc[ds.trials.values]
            mask = np.ones(len(rows), dtype=bool)
            if trial_type_filter.value:
                codes = [TrialType[t].value for t in trial_type_filter.value]
                mask &= rows["trial_type"].isin(codes).to_numpy()
            if behavioral_response_filter.value:
                codes = [BehavioralResponse[r].value for r in behavioral_response_filter.value]
                mask &= rows["behavioral_response"].isin(codes).to_numpy()
            return mask

        def _trial_subset_mask() -> np.ndarray:
            if not (trial_type_filter.value or behavioral_response_filter.value or stimulus_filter.value):
                return np.ones(len(ds.trials), dtype=bool)
            mask = _trial_type_behavior_mask()
            if stimulus_filter.value:
                rows = metadata.trial_info.iloc[ds.trials.values]
                codes = [int(s) for s in stimulus_filter.value]
                mask &= rows["sample_image"].isin(codes).to_numpy()
            return mask

        subset_mask = _trial_subset_mask()
        n_subset = int(subset_mask.sum())
        subset_note = ""
        if n_subset == 0:
            subset_mask = np.ones(len(ds.trials), dtype=bool)
            n_subset = len(ds.trials)
            subset_note = " (filter matched 0 trials — showing all trials instead)"
        subset_signature = (
            tuple(sorted(trial_type_filter.value)),
            tuple(sorted(behavioral_response_filter.value)),
            tuple(sorted(stimulus_filter.value)),
        )

        def _stimulus_groups() -> list[tuple[str | None, np.ndarray]]:
            # When 2+ stimuli are selected, split into one (label, mask) pair
            # per stimulus (each still respecting the trial type/behavioral
            # response filters) so PSD/coherence/STA/phase-difference can
            # overlay one curve per stimulus instead of pooling them together.
            # With 0 or 1 stimuli selected, this is just the ordinary single
            # combined subset (label=None, same as before this feature).
            if len(stimulus_filter.value) < 2:
                return [(None, subset_mask)]
            base_mask = _trial_type_behavior_mask()
            rows = metadata.trial_info.iloc[ds.trials.values]
            groups = []
            for stim in sorted(stimulus_filter.value, key=int):
                stim_mask = base_mask & rows["sample_image"].isin([int(stim)]).to_numpy()
                if stim_mask.sum() > 0:
                    groups.append((f"Stim {stim}", stim_mask))
            return groups or [(None, subset_mask)]

        stim_groups = _stimulus_groups()
        split_by_stimulus = len(stim_groups) > 1

        if split_by_stimulus:
            counts = ", ".join(f"{label}: {int(mask.sum())}" for label, mask in stim_groups)
            subset_info_pane.object = (
                f"*PSD, coherence, STA, and phase-difference are split by stimulus "
                f"({counts} trials). PAC, spike-phase locking, and the quantile panel "
                f"still pool all {n_subset}/{len(ds.trials)} matching trials together.*"
            )
        else:
            subset_info_pane.object = (
                f"*Pooled analyses (PSD, coherence, STA, PAC, spike-phase, quantile/phase-diff) "
                f"use {n_subset}/{len(ds.trials)} trials.{subset_note}*"
            )

        def _pooled_trials_raw_for(ch_idx, mask):
            return _all_trials_raw(ch_idx)[mask]

        def _pooled_trials_filtered_for(ch_idx, mask, low, high):
            return bandpass_filter(_pooled_trials_raw_for(ch_idx, mask), fsample, low, high)

        def _pooled_trials_spikes_for(ch_idx, mask):
            spikes = _all_trials_spikes(ch_idx)
            return spikes[mask] if spikes is not None else None

        def _pooled_trials_raw(ch_idx):
            return _pooled_trials_raw_for(ch_idx, subset_mask)

        def _pooled_trials_spikes(ch_idx):
            return _pooled_trials_spikes_for(ch_idx, subset_mask)

        def _trial_signal(ch_idx):
            # The one selected trial's raw/filtered/spike traces, for the
            # time-domain plot only -- PSD/coherence/STA/PAC use every trial.
            lfp = ds.lfp.sel(trials=trial_index).isel(roi=ch_idx).values * _VOLTS_TO_MICROVOLTS
            filtered = None
            if filter_enabled.value:
                try:
                    filtered = bandpass_filter(lfp, fsample, raw_low, raw_high)
                except ValueError as exc:
                    info_pane.object = f"**Filter error:** {exc}"
            spikes = None
            if show_spikes.value and "spikes" in ds:
                spikes = ds.spikes.sel(trials=trial_index).isel(roi=ch_idx).values.astype(bool)
            return lfp, filtered, spikes

        def _channel_psd_series(ch_idx, color_override=None):
            # Filtered replaces raw (not layered on top of it) once a filter is
            # applied. One line per stimulus group when split_by_stimulus,
            # otherwise the same single raw/filtered line as before -- unless
            # `color_override` is given (the two-channel view passes each
            # channel's own fixed identity color, `_COLOR_RAW`/`_COLOR_FILTERED`,
            # so raw vs filtered no longer overrides it and both channels stay
            # distinguishable regardless of filter state).
            series = []
            for i, (glabel, gmask) in enumerate(stim_groups):
                color = _STIMULUS_COLORS[i % len(_STIMULUS_COLORS)] if glabel else color_override
                added = False
                if filter_enabled.value:
                    try:
                        freqs, psd = _cached_spectral(
                            ("psd", ch_idx, "filt", raw_low, raw_high, glabel, subset_signature),
                            lambda m=gmask: _psd_all_trials(
                                _pooled_trials_filtered_for(ch_idx, m, raw_low, raw_high), fsample
                            ),
                        )
                        label = (
                            f"{glabel} filtered {raw_band_label}" if glabel else f"filtered {raw_band_label}"
                        )
                        series.append((freqs, psd, color or _COLOR_FILTERED, label))
                        added = True
                    except ValueError as exc:
                        info_pane.object = f"**Filter error:** {exc}"
                if not added:
                    freqs, psd = _cached_spectral(
                        ("psd", ch_idx, "raw", glabel, subset_signature),
                        lambda m=gmask: _psd_all_trials(_pooled_trials_raw_for(ch_idx, m), fsample),
                    )
                    series.append((freqs, psd, color or _COLOR_RAW, glabel or "raw"))
            return series

        def _channel_coherence_series(ch1_idx, ch2_idx):
            series = []
            for i, (glabel, gmask) in enumerate(stim_groups):
                color = _STIMULUS_COLORS[i % len(_STIMULUS_COLORS)] if glabel else None
                added = False
                if filter_enabled.value:
                    try:
                        freqs, coh = _cached_spectral(
                            ("coh", ch1_idx, ch2_idx, "filt", raw_low, raw_high, glabel, subset_signature),
                            lambda m=gmask: _coherence_all_trials(
                                _pooled_trials_filtered_for(ch1_idx, m, raw_low, raw_high),
                                _pooled_trials_filtered_for(ch2_idx, m, raw_low, raw_high),
                                fsample,
                            ),
                        )
                        label = (
                            f"{glabel} filtered {raw_band_label}" if glabel else f"filtered {raw_band_label}"
                        )
                        series.append((freqs, coh, color or _COLOR_FILTERED, label))
                        added = True
                    except ValueError as exc:
                        info_pane.object = f"**Filter error:** {exc}"
                if not added:
                    freqs, coh = _cached_spectral(
                        ("coh", ch1_idx, ch2_idx, "raw", glabel, subset_signature),
                        lambda m=gmask: _coherence_all_trials(
                            _pooled_trials_raw_for(ch1_idx, m), _pooled_trials_raw_for(ch2_idx, m), fsample
                        ),
                    )
                    series.append((freqs, coh, color or _COLOR_RAW, glabel or "raw"))
            return series

        def _channel_gc_series(ch1_idx, ch2_idx):
            # Each direction gets the color of the channel it originates from
            # (ch1->ch2 = _COLOR_RAW, ch2->ch1 = _COLOR_FILTERED -- same
            # fixed identity colors as the two-channel LFP trace and PSD
            # panel), unless split by stimulus, where both directions share
            # that stimulus's color and only linestyle (see `_gc_figure`)
            # tells the two directions apart.
            series = []
            for i, (glabel, gmask) in enumerate(stim_groups):
                stim_color = _STIMULUS_COLORS[i % len(_STIMULUS_COLORS)] if glabel else None
                color_xy = stim_color or _COLOR_RAW
                color_yx = stim_color or _COLOR_FILTERED
                added = False
                if filter_enabled.value:
                    try:
                        freqs, gc_xy, gc_yx = _cached_spectral(
                            ("gc", ch1_idx, ch2_idx, "filt", raw_low, raw_high, glabel, subset_signature),
                            lambda m=gmask: _gc_all_trials(
                                _pooled_trials_filtered_for(ch1_idx, m, raw_low, raw_high),
                                _pooled_trials_filtered_for(ch2_idx, m, raw_low, raw_high),
                                fsample,
                            ),
                        )
                        label = (
                            f"{glabel} filtered {raw_band_label}" if glabel else f"filtered {raw_band_label}"
                        )
                        series.append((freqs, gc_xy, gc_yx, color_xy, color_yx, label))
                        added = True
                    except ValueError as exc:
                        info_pane.object = f"**Filter error:** {exc}"
                if not added:
                    freqs, gc_xy, gc_yx = _cached_spectral(
                        ("gc", ch1_idx, ch2_idx, "raw", glabel, subset_signature),
                        lambda m=gmask: _gc_all_trials(
                            _pooled_trials_raw_for(ch1_idx, m), _pooled_trials_raw_for(ch2_idx, m), fsample
                        ),
                    )
                    series.append((freqs, gc_xy, gc_yx, color_xy, color_yx, glabel or "raw"))
            return series

        def _update_gc_pane(ch1_idx, ch2_idx, label1, label2) -> bool:
            if not (show_gc.value and _PYGC_AVAILABLE):
                return False
            gc_pane.object = _gc_figure(
                _channel_gc_series(ch1_idx, ch2_idx), label1, label2, figsize=(4, 3)
            )
            return True

        def _update_hilbert_panes(channel_specs) -> list:
            # channel_specs: list of up to 2 raw signals (µV). Returns the list
            # of panel(s) to show (0, 1, or 2 of envelope_pane/phase_pane,
            # matching hilbert_mode) -- both are linked in x-range to the LFP
            # chart, so zooming either one zooms all of them together.
            if not hilbert_modes:
                return []
            try:
                analytics = [_hilbert_analytic(raw, fsample, phase_low, phase_high) for raw in channel_specs]
            except ValueError as exc:
                info_pane.object = f"**Hilbert error:** {exc}"
                return []

            def _fill(sources, lines, values):
                for i in range(2):
                    if i < len(values):
                        sources[i].data = dict(x=time, y=values[i])
                        lines[i].visible = True
                    else:
                        sources[i].data = dict(x=[], y=[])
                        lines[i].visible = False

            panes = []
            if "Envelope" in hilbert_modes:
                _fill(envelope_sources, envelope_lines, [np.abs(a) for a in analytics])
                envelope_bokeh.legend.visible = len(analytics) > 1
                panes.append(envelope_pane)
            if "Phase" in hilbert_modes:
                _fill(phase_sources, phase_lines, [np.angle(a) for a in analytics])
                phase_bokeh.legend.visible = len(analytics) > 1
                panes.append(phase_pane)
            return panes

        def _update_sta_pane() -> list:
            if not show_sta.value:
                return []
            # (lfp_ch, spike_ch) pairs: self-STA for every selected channel,
            # plus (if enabled, and exactly 2 channels selected) the two
            # cross-channel pairs -- channel A's LFP triggered on channel B's
            # spikes and vice versa. Cross terms go in their own separate
            # figure (cross_sta_pane): overlaid with the self-terms they were
            # easy to miss (much smaller amplitude, sharing axes with the
            # self-STA's own deflection).
            pairs = [(ch_idx, ch_idx) for ch_idx in selected]
            if show_cross_sta.value and len(selected) == 2:
                ch_a, ch_b = selected
                pairs += [(ch_a, ch_b), (ch_b, ch_a)]

            def _linestyle(lfp_ch, spike_ch):
                if lfp_ch == spike_ch:
                    return "-" if (not split_by_stimulus or lfp_ch == selected[0]) else "--"
                return ":" if lfp_ch == selected[0] else "-."

            self_series, cross_series = [], []
            for lfp_ch, spike_ch in pairs:
                is_cross = lfp_ch != spike_ch
                linestyle = _linestyle(lfp_ch, spike_ch)
                for i, (glabel, gmask) in enumerate(stim_groups):
                    spikes_all = _pooled_trials_spikes_for(spike_ch, gmask)
                    if spikes_all is None or spikes_all.size == 0:
                        continue
                    # Filtered replaces raw here too, once a filter is applied --
                    # STA has its own filter toggle/band, independent of the raw
                    # trace's (sta_filter_enabled, not filter_enabled).
                    if sta_filter_enabled.value:
                        try:
                            lfp_all = _pooled_trials_filtered_for(lfp_ch, gmask, sta_low, sta_high)
                        except ValueError as exc:
                            info_pane.object = f"**STA error:** {exc}"
                            continue
                        cache_key = (
                            "sta", lfp_ch, spike_ch, "filt", sta_low, sta_high, glabel, subset_signature
                        )
                        suffix = f" filtered {sta_band_label}"
                    else:
                        lfp_all = _pooled_trials_raw_for(lfp_ch, gmask)
                        cache_key = ("sta", lfp_ch, spike_ch, "raw", glabel, subset_signature)
                        suffix = ""
                    t_rel, sta = _cached_spectral(
                        cache_key,
                        lambda a=lfp_all, s=spikes_all: _spike_triggered_average(a, s, fsample),
                    )
                    if sta is None:
                        continue
                    label_core = (
                        f"{index_to_label[lfp_ch]} on {index_to_label[spike_ch]} spikes"
                        if is_cross
                        else index_to_label[lfp_ch]
                    )
                    if split_by_stimulus:
                        color = _STIMULUS_COLORS[i % len(_STIMULUS_COLORS)]
                        label = f"{label_core} — {glabel}{suffix}"
                    else:
                        color = channel_colors[lfp_ch]
                        label = f"{label_core}{suffix}"
                    entry = (t_rel, sta, color, label, linestyle)
                    (cross_series if is_cross else self_series).append(entry)

            panes = []
            if self_series:
                sta_pane.object = _sta_figure(self_series, figsize=(4, 3))
                panes.append(sta_pane)
            if cross_series:
                cross_sta_pane.object = _sta_figure(
                    cross_series, figsize=(4, 3), title="Cross-channel STA"
                )
                panes.append(cross_sta_pane)
            return panes

        def _update_pac_pane() -> bool:
            if not show_pac.value:
                return False
            phase_band = (pac_phase_low.value, pac_phase_high.value)
            amp_band = (pac_amp_low.value, pac_amp_high.value)
            results = []
            for ch_idx in selected:
                try:
                    bin_centers, mean_amp, mi = _cached_spectral(
                        ("pac", ch_idx, phase_band, amp_band, subset_signature),
                        lambda ch=ch_idx: _pac_modulation_index(
                            _pooled_trials_raw(ch), fsample, phase_band, amp_band
                        ),
                    )
                except ValueError as exc:
                    info_pane.object = f"**PAC error:** {exc}"
                    return False
                results.append({
                    "bin_centers": bin_centers, "values": mean_amp,
                    "metric_label": f"MI={mi:.3f}",
                    "color": channel_colors[ch_idx], "label": index_to_label[ch_idx],
                })
            pac_pane.object = _phase_bar_figure(
                results, "Phase (rad)", "Mean amplitude (µV)",
                f"PAC: phase {phase_band[0]:g}-{phase_band[1]:g} Hz / amp {amp_band[0]:g}-{amp_band[1]:g} Hz",
                figsize=(4, 3),
            )
            return True

        def _update_spike_phase_pane() -> bool:
            if not show_spike_phase.value:
                return False
            results = []
            for ch_idx in selected:
                spikes_all = _pooled_trials_spikes(ch_idx)
                if spikes_all is None:
                    continue
                try:
                    bin_centers, density, r = _cached_spectral(
                        ("spike_phase", ch_idx, phase_low, phase_high, subset_signature),
                        lambda ch=ch_idx, s=spikes_all: _spike_phase_locking(
                            _pooled_trials_raw(ch), s, fsample, (phase_low, phase_high)
                        ),
                    )
                except ValueError as exc:
                    info_pane.object = f"**Spike-phase error:** {exc}"
                    return False
                if bin_centers is None:
                    continue
                results.append({
                    "bin_centers": bin_centers, "values": density,
                    "metric_label": f"R={r:.3f}",
                    "color": channel_colors[ch_idx], "label": index_to_label[ch_idx],
                })
            if not results:
                return False
            spike_phase_pane.object = _phase_bar_figure(
                results, "Spike phase (rad)", "Spike probability",
                f"Spike-phase locking ({phase_band_label})", figsize=(4, 3),
            )
            return True

        def _quantile_data(ch1_idx, ch2_idx):
            # Power product of both channels' Hilbert power (envelope
            # squared) for EVERY trial (same quantity save_burst_trains.py
            # thresholds for burst detection), but the quartile thresholds and
            # percentile rank are computed only from samples that are both in
            # the trial-subset filter AND inside the -0.5s-to-match-onset
            # burst window (_burst_window_mask) -- so the currently selected
            # trial (which may or may not be in that subset/window) can still
            # be displayed against them. Returns (product, percentile-rank,
            # quartile thresholds): product and percentile are (n_trials,
            # n_times) over ALL trials, thresholds are 5 edge values.
            def _compute():
                p1 = np.abs(_hilbert_analytic(_all_trials_raw(ch1_idx), fsample, phase_low, phase_high)) ** 2
                p2 = np.abs(_hilbert_analytic(_all_trials_raw(ch2_idx), fsample, phase_low, phase_high)) ** 2
                product = p1 * p2
                valid_subset = subset_mask[:, None] & _burst_window_mask()
                thrs = _quantile_thresholds(product[valid_subset])
                sorted_subset = np.sort(product[valid_subset].ravel())
                percentile = (
                    np.searchsorted(sorted_subset, product.ravel(), side="right")
                    / sorted_subset.size
                ).reshape(product.shape)
                return product, percentile, thrs
            return _cached_spectral(
                ("quantile", ch1_idx, ch2_idx, phase_low, phase_high, subset_signature), _compute
            )

        def _phase_diff_all(ch1_idx, ch2_idx):
            def _compute():
                a1 = _hilbert_analytic(_all_trials_raw(ch1_idx), fsample, phase_low, phase_high)
                a2 = _hilbert_analytic(_all_trials_raw(ch2_idx), fsample, phase_low, phase_high)
                return _phase_difference_all_trials(a1, a2)
            return _cached_spectral(
                ("phase_diff_all", ch1_idx, ch2_idx, phase_low, phase_high), _compute
            )

        def _update_quantile_pane(ch1_idx, ch2_idx) -> bool:
            if not show_quantile_regions.value:
                return False
            try:
                product, percentile, thrs = _quantile_data(ch1_idx, ch2_idx)
            except ValueError as exc:
                info_pane.object = f"**Quantile error:** {exc}"
                return False
            pos = _trial_position()
            percentile_source.data = dict(x=time, y=percentile[pos])
            trial_product = product[pos]
            for i in range(4):
                lo, hi = thrs[i], thrs[i + 1]
                mask = (trial_product >= lo) & (trial_product <= hi if i == 3 else trial_product < hi)
                y = np.where(mask, 1.05 + i * 0.08, np.nan)
                quantile_band_sources[i].data = dict(x=time, y=y)
            return True

        def _roi_channel_key(ch_idx) -> str:
            return f"{ds.roi.values[ch_idx]}_{ds.attrs['channels_labels'][ch_idx]}"

        def _canonical_phase_diff_pair(ch1_idx, ch2_idx):
            # phase_coupling_analysis's pairwise phase-difference (the
            # reference this panel is meant to match) always orders each
            # channel pair alphabetically by "{roi}_{channel}" and computes
            # phase(first) - phase(second) -- not by which channel the user
            # happened to pick first here. Circular std doesn't care about
            # sign, but the mean/the histogram's orientation flips if this
            # isn't matched, which is why it needs to be pinned down rather
            # than just using (ch1_idx, ch2_idx) as selected.
            if _roi_channel_key(ch1_idx) <= _roi_channel_key(ch2_idx):
                return ch1_idx, ch2_idx
            return ch2_idx, ch1_idx

        def _update_phase_diff_pane(ch1_idx, ch2_idx) -> bool:
            if not phase_diff_bins.value:
                return False
            pd_ch1, pd_ch2 = _canonical_phase_diff_pair(ch1_idx, ch2_idx)
            try:
                product, _unused, thrs = _quantile_data(ch1_idx, ch2_idx)
                phase_diff_all = _phase_diff_all(pd_ch1, pd_ch2)
            except ValueError as exc:
                info_pane.object = f"**Phase-difference error:** {exc}"
                return False
            # Quartile thresholds are always computed from the full (possibly
            # stimulus-combined) trial subset -- see _quantile_data -- so Q1-Q4
            # mean the same power-product regime across stimulus groups. When
            # split_by_stimulus, one polar subplot per (stimulus, quantile
            # bin) combination is added, all side by side in the same row
            # (_phase_diff_circular_figure lays out a flat list that way).
            window_mask = _burst_window_mask()
            results = []
            for glabel, gmask in stim_groups:
                product_g = product[gmask]
                phase_diff_g = phase_diff_all[gmask]
                window_g = window_mask[gmask]
                for label in phase_diff_bins.value:
                    i = _QUANTILE_LABELS.index(label)
                    lo, hi = thrs[i], thrs[i + 1]
                    mask = (
                        (product_g >= lo) & (product_g <= hi if i == 3 else product_g < hi) & window_g
                    )
                    entry_label = f"{glabel} · {label}" if glabel else label
                    results.append(
                        {"label": entry_label, "phases": phase_diff_g[mask], "color": _QUANTILE_COLORS[i]}
                    )
            phase_diff_pane.object = _phase_diff_circular_figure(
                results, figsize=(2.6 * len(results), 3),
                suptitle=f"Δφ = phase({index_to_label[pd_ch1]}) − phase({index_to_label[pd_ch2]})",
            )
            return True

        def _update_lfp_bokeh(series, spike_channels, event_times):
            # series: list of up to 2 dicts {y, label}. spike_channels: same
            # length list of (bool array or None).
            sources = (lfp_source_1, lfp_source_2)
            lines = (lfp_line_1, lfp_line_2)
            spike_sources = (spike_source_1, spike_source_2)
            spike_scatters = (spike_scatter_1, spike_scatter_2)

            all_y = []
            for i in range(2):
                if i < len(series):
                    y = series[i]["y"]
                    sources[i].data = dict(x=time, y=y)
                    lines[i].visible = True
                    all_y.append(y)
                else:
                    sources[i].data = dict(x=[], y=[])
                    lines[i].visible = False

            y_top = y_step = 0.0
            if all_y:
                combined = np.concatenate(all_y)
                span = np.ptp(combined) if len(combined) else 1.0
                y_top = combined.max() + 0.08 * span
                y_step = 0.08 * span

            for i in range(2):
                spikes = spike_channels[i] if i < len(spike_channels) else None
                if spikes is not None and spikes.any():
                    spike_times = time[spikes]
                    spike_sources[i].data = dict(x=spike_times, y=np.full_like(spike_times, y_top + i * y_step))
                    spike_scatters[i].visible = True
                else:
                    spike_sources[i].data = dict(x=[], y=[])
                    spike_scatters[i].visible = False

            for key, sp in event_spans.items():
                t = event_times.get(key)
                if t is not None:
                    sp.location = t
                    sp.visible = True
                else:
                    sp.visible = False

            lfp_bokeh.legend.visible = len(series) > 1

        if len(selected) == 1:
            ch_idx = selected[0]
            lfp, filtered, spikes = _trial_signal(ch_idx)

            # Filtered replaces raw (not layered on top of it) once a filter is applied.
            trace = filtered if filtered is not None else lfp
            _update_lfp_bokeh(
                [{"y": trace, "label": "signal"}], [spikes],
                _event_times_for_trial(ds, trial_index, fsample),
            )

            psd_pane.object = _psd_figure(_channel_psd_series(ch_idx), figsize=(4, 3))

            hilbert_panes = _update_hilbert_panes([lfp])
            extras = []
            extras.extend(_update_sta_pane())
            if _update_pac_pane():
                extras.append(pac_pane)
            if _update_spike_phase_pane():
                extras.append(spike_phase_pane)

            main.objects = (
                [info_pane, lfp_pane]
                + hilbert_panes
                + ([pn.Row(*extras, sizing_mode="stretch_width")] if extras else [])
                + [psd_pane]
            )
        else:
            ch1_idx, ch2_idx = selected
            label1, label2 = index_to_label[ch1_idx], index_to_label[ch2_idx]
            lfp1, filt1, spk1 = _trial_signal(ch1_idx)
            lfp2, filt2, spk2 = _trial_signal(ch2_idx)

            trace1, trace2 = (filt1, filt2) if filter_enabled.value else (lfp1, lfp2)
            trace1 = lfp1 if trace1 is None else trace1  # filter error fallback
            trace2 = lfp2 if trace2 is None else trace2
            _update_lfp_bokeh(
                [{"y": trace1, "label": label1}, {"y": trace2, "label": label2}],
                [spk1, spk2],
                _event_times_for_trial(ds, trial_index, fsample),
            )

            psd_pane.object = _dual_psd_figure(
                _channel_psd_series(ch1_idx, color_override=_COLOR_RAW),
                _channel_psd_series(ch2_idx, color_override=_COLOR_FILTERED),
                label1, label2, figsize=(4, 3),
            )
            coherence_pane.object = _coherence_figure(
                _channel_coherence_series(ch1_idx, ch2_idx), label1, label2, figsize=(4, 3)
            )
            spectral_panes = [psd_pane, coherence_pane]
            if _update_gc_pane(ch1_idx, ch2_idx, label1, label2):
                spectral_panes.append(gc_pane)
            elif show_gc.value and not _PYGC_AVAILABLE:
                gc_pane.object = _gc_unavailable_figure(figsize=(4, 3))
                spectral_panes.append(gc_pane)

            hilbert_panes = _update_hilbert_panes([lfp1, lfp2])
            if _update_quantile_pane(ch1_idx, ch2_idx):
                hilbert_panes = hilbert_panes + [quantile_pane]

            extras = []
            extras.extend(_update_sta_pane())
            if _update_pac_pane():
                extras.append(pac_pane)
            if _update_spike_phase_pane():
                extras.append(spike_phase_pane)
            if _update_phase_diff_pane(ch1_idx, ch2_idx):
                extras.append(phase_diff_pane)

            main.objects = (
                [info_pane, lfp_pane]
                + hilbert_panes
                + ([pn.Row(*extras, sizing_mode="stretch_width")] if extras else [])
                + [pn.Row(*spectral_panes, sizing_mode="stretch_width")]
            )

        row = metadata.trial_info.iloc[trial_index]
        ttype = TrialType(int(row["trial_type"])).name
        resp = row.get("behavioral_response")
        resp_label = BehavioralResponse(int(resp)).name if pd.notna(resp) else "N/A"
        stim_id = row.get("sample_image")
        stim_label = str(int(stim_id)) if pd.notna(stim_id) else "N/A"
        info_pane.object = (
            f"### Trial {trial_index}\n"
            f"**Type:** {ttype} &nbsp;&nbsp; **Behavioral response:** {resp_label} "
            f"&nbsp;&nbsp; **Stimulus:** {stim_label}"
        )

    sidebar = pn.Column(
        "## Session",
        monkey_select,
        date_select,
        session_select,
        align_select,
        pn.layout.Divider(),
        "## Raw data plot",
        trial_select,
        channel_select,
        clear_channels_button,
        #include_flagged_channels,
        #unique_recordings_only,
        #zoom_start,
        #zoom_end,
        #reset_zoom_button,
        #show_spikes,
        #show_events,
        #pn.layout.Divider(),
        "### Trial subset (pooled analyses)",
        trial_type_filter,
        behavioral_response_filter,
        stimulus_filter,
        clear_trial_subset_button,
        subset_info_pane,
        pn.layout.Divider(),
        "## Spectral analysis",
        filter_enabled,
        raw_band_select,
        raw_custom_low,
        raw_custom_high,
        show_gc,
        pn.layout.Divider(),
        "## Phase coupling",
        hilbert_mode,
        phase_band_select,
        phase_custom_low,
        phase_custom_high,
        show_spike_phase,
        show_quantile_regions,
        phase_diff_bins,
        "## Phase-amplitude coupling (PAC)",
        show_pac,
        pac_phase_low,
        pac_phase_high,
        pac_amp_low,
        pac_amp_high,
        pn.layout.Divider(),
        "## Spike-triggered average",
        show_sta,
        show_cross_sta,
        sta_filter_enabled,
        sta_band_select,
        sta_custom_low,
        sta_custom_high,
        pn.layout.Divider(),
    )
    main = pn.Column(info_pane, lfp_pane, psd_pane)
    montage_tab = pn.Column(
        montage_caption,
        pn.Row(montage_channel_select, montage_clear_channels_button),
        montage_pane,
        pn.Row(montage_height_slider, montage_spacing_slider, sizing_mode="stretch_width"),
    )

    monkey_select.param.watch(lambda e: (_update_dates(), _update_band_options()), "value")
    trial_select.param.watch(_update_montage, "value")
    show_events.param.watch(_update_montage, "value")
    montage_channel_select.param.watch(_update_montage, "value")
    montage_clear_channels_button.on_click(_clear_montage_channel_selection)
    montage_height_slider.param.watch(_update_montage, "value_throttled")
    montage_spacing_slider.param.watch(_update_montage, "value_throttled")
    date_select.param.watch(_update_sessions, "value")
    clear_channels_button.on_click(_clear_channel_selection)
    clear_trial_subset_button.on_click(_clear_trial_subset)
    channel_select.param.watch(_limit_channel_selection, "value")
    reset_zoom_button.on_click(_reset_zoom)
    for widget in (zoom_start, zoom_end):
        widget.param.watch(_apply_zoom, "value")
    for widget in (
        monkey_select,
        date_select,
        session_select,
        align_select,
        include_flagged_channels,
        unique_recordings_only,
    ):
        widget.param.watch(_reload_session, "value")
    for widget in (
        trial_select,
        channel_select,
        show_spikes,
        show_events,
        filter_enabled,
        raw_band_select,
        raw_custom_low,
        raw_custom_high,
        show_gc,
        hilbert_mode,
        phase_band_select,
        phase_custom_low,
        phase_custom_high,
        show_sta,
        show_cross_sta,
        sta_filter_enabled,
        sta_band_select,
        sta_custom_low,
        sta_custom_high,
        show_pac,
        pac_phase_low,
        pac_phase_high,
        pac_amp_low,
        pac_amp_high,
        show_spike_phase,
        show_quantile_regions,
        phase_diff_bins,
        trial_type_filter,
        behavioral_response_filter,
        stimulus_filter,
    ):
        widget.param.watch(_update_plot, "value")

    _update_dates()
    _update_band_options()
    _update_sessions()
    _reload_session()

    tabs = pn.Tabs(("Trace / spectral analysis", main), ("Multi-channel view", montage_tab))
    return pn.template.FastListTemplate(
        title="GrayDataViz — LFP Explorer", sidebar=[sidebar], main=[tabs]
    )


def main() -> None:
    pn.serve(build_app, show=True)


if __name__ == "__main__":
    main()
