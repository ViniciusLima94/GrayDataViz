"""Panel GUI for exploring raw LFP recordings.

Lets you pick a monkey/date/session/trial and up to two channels, see the
trial type and behavioral response, optionally overlay the spike raster on
the LFP trace, optionally overlay a bandpass-filtered version of the trace on
top of the raw signal, and see each trace's power spectrum (multitaper,
matching phase_coupling_analysis's xr_psd_array_multitaper params) below it.
When two channels are selected, their coherence (multitaper, matching
phase_coupling_analysis's conn_spec_average params) is shown between the two
power spectra instead.

Run with:
    python -m graydataviz.app
or, after `pip install -e ".[gui]"`:
    graydataviz-gui
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import panel as pn
from mne.time_frequency import psd_array_multitaper

from .config import DataConfig, default_config
from .discovery import list_dates, list_monkeys, list_sessions
from .exceptions import GrayDataVizError
from .filters import band_presets, bandpass_filter
from .metadata import SessionMetadata, load_session_metadata
from .session import load_session
from .trials import BehavioralResponse, TrialType

pn.extension()

# Validated categorical palette (see the dataviz skill's reference palette):
# slot 1 blue / slot 2 orange / slot 8 red, so identity stays consistent
# between the time-domain trace and the power spectrum below it.
_COLOR_RAW = "#2a78d6"
_COLOR_FILTERED = "#eb6834"
_COLOR_SPIKES = "#e34948"

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


@pn.cache
def _cached_metadata(config: DataConfig, monkey: str, date: str, session: int) -> SessionMetadata:
    return load_session_metadata(monkey, date, session, config=config)


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
        exclude_slvr_msmod=exclude_slvr_msmod,
        only_unique_recordings=only_unique_recordings,
        load_spike_times=True,
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


def _lfp_trace_figure(
    time: np.ndarray,
    series: list[dict],
    spike_series: list[dict],
    figsize: tuple[float, float],
    title: str | None = None,
) -> Figure:
    """`series`: dicts with `y`, `color`, `linestyle`, `label`.
    `spike_series`: dicts with `spikes` (bool array), `color`, `label`.

    All spike rows sit above the combined range of every plotted trace (not
    each channel's own range) since they share one y-axis -- otherwise a
    lower-amplitude channel's spikes would land low enough to sit inside, or
    below, a higher-amplitude channel's trace. Each row gets its own offset
    so multiple channels' spikes stack instead of overlapping.
    """
    fig = Figure(figsize=figsize)
    ax = fig.add_subplot(111)
    for s in series:
        ax.plot(time, s["y"], color=s["color"], lw=1 if s["linestyle"] == "-" else 1.2,
                 linestyle=s["linestyle"], label=s["label"])
    if spike_series and series:
        all_y = np.concatenate([s["y"] for s in series])
        y_top = all_y.max() + 0.05 * np.ptp(all_y)
        y_step = 0.05 * np.ptp(all_y)
        for row, s in enumerate(spike_series):
            if not s["spikes"].any():
                continue
            spike_times = time[s["spikes"]]
            y = np.full_like(spike_times, y_top + row * y_step)
            ax.scatter(spike_times, y, marker="|", color=s["color"], label=s["label"])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("LFP")
    if title:
        ax.set_title(title, fontsize=9)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles, labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=len(labels),
            frameon=False,
            fontsize=8,
        )
    return fig


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
    ax.set_ylabel("Power spectral density")
    if title:
        ax.set_title(title, fontsize=9)
    ax.legend(loc="upper right", fontsize=8)
    return fig


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
    ax.set_title(f"Coherence: {label1} vs {label2}", fontsize=9)
    ax.legend(loc="upper right", fontsize=8)
    return fig


def build_app(config: DataConfig | None = None) -> pn.viewable.Viewable:
    config = config or default_config()

    monkeys = list_monkeys(config)
    monkey_select = pn.widgets.Select(label="Monkey", options=monkeys)
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
        label="Include slvr/ms_mod-flagged channels", value=False
    )
    unique_recordings_only = pn.widgets.Checkbox(
        label="Unique recordings only", value=False
    )
    show_spikes = pn.widgets.Checkbox(label="Overlay spikes", value=False)
    filter_enabled = pn.widgets.Checkbox(label="Apply bandpass filter", value=False)
    band_select = pn.widgets.Select(label="Band preset", options=["custom"])
    custom_low = pn.widgets.FloatInput(label="Low (Hz)", value=8.0, start=0.0)
    custom_high = pn.widgets.FloatInput(label="High (Hz)", value=12.0, start=0.1)

    info_pane = pn.pane.Markdown("")
    lfp_pane = pn.pane.Matplotlib(Figure(), tight=True, sizing_mode="stretch_width")
    psd_pane = pn.pane.Matplotlib(Figure(figsize=(4, 3)), tight=True, sizing_mode="stretch_width")
    psd_pane_2 = pn.pane.Matplotlib(Figure(figsize=(4, 3)), tight=True, sizing_mode="stretch_width")
    coherence_pane = pn.pane.Matplotlib(
        Figure(figsize=(4, 3)), tight=True, sizing_mode="stretch_width"
    )

    state: dict = {"metadata": None, "ds": None, "spectral_cache": {}}

    def _update_dates(_event=None):
        dates = list_dates(monkey_select.value, config) if monkey_select.value else []
        date_select.options = dates
        date_select.value = dates[0] if dates else None

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
        band_select.options = labels
        band_select.value = labels[0]

    def _resolve_band() -> tuple[float, float]:
        if band_select.value == "custom":
            return custom_low.value, custom_high.value
        low_str, high_str = band_select.value.replace(" Hz", "").split("-")
        return float(low_str), float(high_str)

    def _limit_channel_selection(event):
        if len(event.new) > 2:
            channel_select.value = event.new[:2]

    def _clear_channel_selection(_event=None):
        channel_select.value = []

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
            f"{roi} (ch {ch})": i
            for i, (roi, ch) in enumerate(zip(ds.roi.values, ds.attrs["channels_labels"]))
        }
        channel_select.options = channel_options
        channel_select.value = [next(iter(channel_options.values()))]

        _update_plot()

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

        band_label = None
        low = high = None
        if filter_enabled.value:
            low, high = _resolve_band()
            band_label = f"{low:g}-{high:g} Hz"

        def _cached_spectral(key, compute_fn):
            cache = state["spectral_cache"]
            if key not in cache:
                cache[key] = compute_fn()
            return cache[key]

        def _all_trials_raw(ch_idx):
            return ds.lfp.isel(roi=ch_idx).values  # (n_trials, n_times)

        def _all_trials_filtered(ch_idx):
            return bandpass_filter(_all_trials_raw(ch_idx), fsample, low, high)

        def _trial_signal(ch_idx):
            # The one selected trial's raw/filtered/spike traces, for the
            # time-domain plot only -- PSD/coherence use every trial (below).
            lfp = ds.lfp.sel(trials=trial_index).isel(roi=ch_idx).values
            filtered = None
            if filter_enabled.value:
                try:
                    filtered = bandpass_filter(lfp, fsample, low, high)
                except ValueError as exc:
                    info_pane.object = f"**Filter error:** {exc}"
            spikes = None
            if show_spikes.value and "spikes" in ds:
                spikes = ds.spikes.sel(trials=trial_index).isel(roi=ch_idx).values.astype(bool)
            return lfp, filtered, spikes

        def _channel_psd_series(ch_idx):
            # Filtered replaces raw (not layered on top of it) once a filter is applied.
            if filter_enabled.value:
                try:
                    freqs, psd = _cached_spectral(
                        ("psd", ch_idx, "filt", low, high),
                        lambda: _psd_all_trials(_all_trials_filtered(ch_idx), fsample),
                    )
                    return [(freqs, psd, _COLOR_FILTERED, f"filtered {band_label}")]
                except ValueError as exc:
                    info_pane.object = f"**Filter error:** {exc}"
            freqs, psd = _cached_spectral(
                ("psd", ch_idx, "raw"), lambda: _psd_all_trials(_all_trials_raw(ch_idx), fsample)
            )
            return [(freqs, psd, _COLOR_RAW, "raw")]

        def _channel_coherence_series(ch1_idx, ch2_idx):
            if filter_enabled.value:
                try:
                    freqs, coh = _cached_spectral(
                        ("coh", ch1_idx, ch2_idx, "filt", low, high),
                        lambda: _coherence_all_trials(
                            _all_trials_filtered(ch1_idx), _all_trials_filtered(ch2_idx), fsample
                        ),
                    )
                    return [(freqs, coh, _COLOR_FILTERED, f"filtered {band_label}")]
                except ValueError as exc:
                    info_pane.object = f"**Filter error:** {exc}"
            freqs, coh = _cached_spectral(
                ("coh", ch1_idx, ch2_idx, "raw"),
                lambda: _coherence_all_trials(
                    _all_trials_raw(ch1_idx), _all_trials_raw(ch2_idx), fsample
                ),
            )
            return [(freqs, coh, _COLOR_RAW, "raw")]

        if len(selected) == 1:
            ch_idx = selected[0]
            lfp, filtered, spikes = _trial_signal(ch_idx)

            # Filtered replaces raw (not layered on top of it) once a filter is applied.
            if filtered is not None:
                trace, trace_label = filtered, f"filtered {band_label}"
            else:
                trace, trace_label = lfp, "raw"
            series = [{"y": trace, "color": _COLOR_RAW, "linestyle": "-", "label": trace_label}]
            spike_series = (
                [{"spikes": spikes, "color": _COLOR_SPIKES, "label": "spikes"}]
                if spikes is not None
                else []
            )
            lfp_pane.object = _lfp_trace_figure(time, series, spike_series, figsize=(8, 3))

            psd_pane.object = _psd_figure(_channel_psd_series(ch_idx), figsize=(4, 3))
            main.objects = [info_pane, lfp_pane, psd_pane]
        else:
            ch1_idx, ch2_idx = selected
            label1, label2 = index_to_label[ch1_idx], index_to_label[ch2_idx]
            lfp1, filt1, spk1 = _trial_signal(ch1_idx)
            lfp2, filt2, spk2 = _trial_signal(ch2_idx)

            trace1, trace2 = (filt1, filt2) if filter_enabled.value else (lfp1, lfp2)
            trace1 = lfp1 if trace1 is None else trace1  # filter error fallback
            trace2 = lfp2 if trace2 is None else trace2
            suffix = f"filtered {band_label}" if filter_enabled.value else "raw"
            series = [
                {"y": trace1, "color": _COLOR_RAW, "linestyle": "-", "label": f"{label1} {suffix}"},
                {"y": trace2, "color": _COLOR_FILTERED, "linestyle": "-", "label": f"{label2} {suffix}"},
            ]
            spike_series = []
            if spk1 is not None:
                spike_series.append(
                    {"spikes": spk1, "color": _COLOR_RAW, "label": f"{label1} spikes"}
                )
            if spk2 is not None:
                spike_series.append(
                    {"spikes": spk2, "color": _COLOR_FILTERED, "label": f"{label2} spikes"}
                )
            lfp_pane.object = _lfp_trace_figure(time, series, spike_series, figsize=(9, 3))

            psd_pane.object = _psd_figure(_channel_psd_series(ch1_idx), figsize=(4, 3), title=label1)
            psd_pane_2.object = _psd_figure(_channel_psd_series(ch2_idx), figsize=(4, 3), title=label2)
            coherence_pane.object = _coherence_figure(
                _channel_coherence_series(ch1_idx, ch2_idx), label1, label2, figsize=(4, 3)
            )

            main.objects = [
                info_pane,
                lfp_pane,
                pn.Row(psd_pane, psd_pane_2, coherence_pane, sizing_mode="stretch_width"),
            ]

        row = metadata.trial_info.iloc[trial_index]
        ttype = TrialType(int(row["trial_type"])).name
        resp = row.get("behavioral_response")
        resp_label = BehavioralResponse(int(resp)).name if pd.notna(resp) else "N/A"
        info_pane.object = (
            f"### Trial {trial_index}\n"
            f"**Type:** {ttype} &nbsp;&nbsp; **Behavioral response:** {resp_label}"
        )

    sidebar = pn.Column(
        "## Session",
        monkey_select,
        date_select,
        session_select,
        align_select,
        "## Trial / channel",
        trial_select,
        channel_select,
        clear_channels_button,
        include_flagged_channels,
        unique_recordings_only,
        pn.layout.Divider(),
        "## Overlays",
        show_spikes,
        filter_enabled,
        band_select,
        custom_low,
        custom_high,
    )
    main = pn.Column(info_pane, lfp_pane, psd_pane)

    monkey_select.param.watch(lambda e: (_update_dates(), _update_band_options()), "value")
    date_select.param.watch(_update_sessions, "value")
    clear_channels_button.on_click(_clear_channel_selection)
    channel_select.param.watch(_limit_channel_selection, "value")
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
        filter_enabled,
        band_select,
        custom_low,
        custom_high,
    ):
        widget.param.watch(_update_plot, "value")

    _update_dates()
    _update_band_options()
    _update_sessions()
    _reload_session()

    return pn.template.FastListTemplate(
        title="GrayDataViz — LFP Explorer", sidebar=[sidebar], main=[main]
    )


def main() -> None:
    pn.serve(build_app, show=True)


if __name__ == "__main__":
    main()
