"""Panel GUI for exploring raw LFP recordings.

Lets you pick a monkey/date/session/trial/channel, see the trial type and
behavioral response, see the cue/match/non-match stimulus images embedded in
that session's recording_info, optionally overlay the spike raster on the LFP
trace, and optionally overlay a bandpass-filtered version of the trace on top
of the raw signal.

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

from .config import DataConfig, default_config
from .discovery import list_dates, list_monkeys, list_sessions
from .exceptions import GrayDataVizError
from .filters import band_presets, bandpass_filter
from .metadata import SessionMetadata, load_session_metadata
from .session import load_session
from .stimuli import get_stimulus_image, get_stimulus_name
from .trials import BehavioralResponse, TrialType

pn.extension()

_STIMULUS_FIELDS = (("sample_image", "Cue"), ("match_image", "Match"), ("nonmatch_image", "Non-match"))


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


def _empty_image_figure(message: str) -> Figure:
    fig = Figure(figsize=(2.2, 2.2))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True, fontsize=9)
    return fig


def _image_figure(image: np.ndarray, title: str) -> Figure:
    fig = Figure(figsize=(2.2, 2.2))
    ax = fig.add_subplot(111)
    ax.imshow(image)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
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
    channel_select = pn.widgets.Select(label="Channel", options={})
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
    cue_pane = pn.pane.Matplotlib(_empty_image_figure("—"), tight=True)
    match_pane = pn.pane.Matplotlib(_empty_image_figure("—"), tight=True)
    nonmatch_pane = pn.pane.Matplotlib(_empty_image_figure("—"), tight=True)
    image_panes = {"sample_image": cue_pane, "match_image": match_pane, "nonmatch_image": nonmatch_pane}

    state: dict = {"metadata": None, "ds": None}

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
        channel_select.value = next(iter(channel_options.values()))

        _update_plot()

    def _update_plot(_event=None):
        ds = state["ds"]
        metadata = state["metadata"]
        if ds is None or trial_select.value is None or channel_select.value is None:
            return

        trial_index = trial_select.value
        ch_idx = channel_select.value
        lfp = ds.lfp.sel(trials=trial_index).isel(roi=ch_idx).values
        time = ds.time.values

        fig = Figure(figsize=(8, 3))
        ax = fig.add_subplot(111)
        ax.plot(time, lfp, color="0.4", lw=1, label="raw")

        if filter_enabled.value:
            low, high = _resolve_band()
            try:
                filtered = bandpass_filter(lfp, float(ds.attrs["fsample"]), low, high)
                ax.plot(time, filtered, color="C0", lw=1.2, label=f"filtered {low:g}-{high:g} Hz")
            except ValueError as exc:
                info_pane.object = f"**Filter error:** {exc}"

        if show_spikes.value and "spikes" in ds:
            spike_train = ds.spikes.sel(trials=trial_index).isel(roi=ch_idx).values.astype(bool)
            if spike_train.any():
                spike_times = time[spike_train]
                y = np.full_like(spike_times, lfp.max() + 0.05 * np.ptp(lfp) if len(lfp) else 1.0)
                ax.scatter(spike_times, y, marker="|", color="crimson", label="spikes")

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("LFP")
        ax.legend(loc="upper right", fontsize=8)
        lfp_pane.object = fig

        row = metadata.trial_info.iloc[trial_index]
        ttype = TrialType(int(row["trial_type"])).name
        resp = row.get("behavioral_response")
        resp_label = BehavioralResponse(int(resp)).name if pd.notna(resp) else "N/A"
        info_pane.object = (
            f"### Trial {trial_index}\n"
            f"**Type:** {ttype} &nbsp;&nbsp; **Behavioral response:** {resp_label}"
        )

        for field, pane in image_panes.items():
            image_id = row.get(field)
            image = get_stimulus_image(metadata.recording_info, image_id)
            label = next(name for key, name in _STIMULUS_FIELDS if key == field)
            if image is None:
                pane.object = _empty_image_figure(f"{label}\n(no image)")
            else:
                name = get_stimulus_name(metadata.recording_info, image_id)
                pane.object = _image_figure(image, f"{label}: {name or int(image_id)}")

    monkey_select.param.watch(lambda e: (_update_dates(), _update_band_options()), "value")
    date_select.param.watch(_update_sessions, "value")
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

    sidebar = pn.Column(
        "## Session",
        monkey_select,
        date_select,
        session_select,
        align_select,
        "## Trial / channel",
        trial_select,
        channel_select,
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
    main = pn.Column(
        info_pane,
        lfp_pane,
        pn.Row(cue_pane, match_pane, nonmatch_pane),
    )
    return pn.template.FastListTemplate(
        title="GrayDataViz — LFP Explorer", sidebar=[sidebar], main=[main]
    )


def main() -> None:
    pn.serve(build_app, show=True)


if __name__ == "__main__":
    main()
