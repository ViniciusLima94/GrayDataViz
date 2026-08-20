from graydataviz.app import build_app

from .conftest import MONKEY, N_TRIALS

_CHANNEL_LABEL = "Channel(s) — select up to 2 (2nd shows coherence)"


def _collect_widgets(app):
    widgets = {}

    def _collect(obj):
        label = getattr(obj, "label", None)
        if label:
            widgets.setdefault(label, obj)
        for child in getattr(obj, "objects", []):
            _collect(child)

    for root in list(app.sidebar) + list(app.main):
        _collect(root)
    return widgets


def test_build_app_smoke(data_config):
    app = build_app(config=data_config)
    assert app is not None

    widgets = _collect_widgets(app)

    assert widgets["Monkey"].value == MONKEY
    assert len(widgets["Trial"].options) == N_TRIALS
    assert len(widgets[_CHANNEL_LABEL].options) == 2  # one channel excluded (slvr) by default
    assert widgets[_CHANNEL_LABEL].value  # a default channel is preselected

    first_trial = next(iter(widgets["Trial"].options.values()))
    other_trials = [v for v in widgets["Trial"].options.values() if v != first_trial]
    widgets["Trial"].value = other_trials[0]

    widgets["Overlay spikes"].value = True
    widgets["Apply bandpass filter (trace, PSD, coherence & GC)"].value = True
    widgets["Band preset"].value = widgets["Band preset"].options[0]

    # Spike-triggered average, PAC, and phase coupling each have their own
    # independent filter/band controls, decoupled from the raw trace's.
    widgets["Spike-triggered average (µV)"].value = True
    widgets["Apply bandpass filter to STA"].value = True
    widgets["Phase-amplitude coupling (Tort MI)"].value = True
    widgets["Hilbert decomposition (uses the band below)"].value = "Envelope + Phase"


def test_two_channel_analyses_smoke(data_config):
    app = build_app(config=data_config)
    widgets = _collect_widgets(app)

    channel_select = widgets[_CHANNEL_LABEL]
    channel_select.value = list(channel_select.options.values())[:2]

    widgets["Spike-triggered average (µV)"].value = True
    widgets["Include cross-channel STA (2 channels: A on B's spikes, B on A's spikes)"].value = True
    widgets["Spike-phase locking (uses the band above)"].value = True
    widgets["Power-product quantile regions (2 channels, uses the band above)"].value = True
    widgets["Phase-difference circular plot: quantile bin(s)"].value = ["Q1 (0-25%)", "Q4 (75-100%)"]

    assert app.main is not None


def test_include_flagged_channels_toggle(data_config):
    app = build_app(config=data_config)
    widgets = _collect_widgets(app)

    assert len(widgets[_CHANNEL_LABEL].options) == 2
    widgets["Include slvr/ms_mod-flagged channels"].value = True
    assert len(widgets[_CHANNEL_LABEL].options) == 3


def test_selecting_two_channels_shows_coherence_layout(data_config):
    app = build_app(config=data_config)
    widgets = _collect_widgets(app)

    channel_select = widgets[_CHANNEL_LABEL]
    all_channels = list(channel_select.options.values())
    assert len(all_channels) >= 2

    channel_select.value = all_channels[:2]
    # Two channels selected -> the inner main Column (first of the two tabs)
    # is rebuilt as [info, lfp row, psd+coherence row].
    tabs = app.main.objects[0]
    main_column = tabs.objects[0]
    assert len(main_column.objects) == 3
    bottom_row = main_column.objects[2]
    assert len(bottom_row.objects) == 2  # dual-axis psd (both channels), coherence


def test_gc_checkbox_adds_pane_for_two_channels(data_config):
    app = build_app(config=data_config)
    widgets = _collect_widgets(app)

    channel_select = widgets[_CHANNEL_LABEL]
    channel_select.value = list(channel_select.options.values())[:2]

    tabs = app.main.objects[0]
    main_column = tabs.objects[0]
    bottom_row = main_column.objects[2]
    assert len(bottom_row.objects) == 2  # GC off by default: dual-axis psd, coherence

    widgets["Granger causality spectrum (2 channels, pyGC — slower, off by default)"].value = True
    bottom_row = main_column.objects[2]
    assert len(bottom_row.objects) == 3  # dual-axis psd, coherence, GC


def test_selecting_more_than_two_channels_is_truncated(data_config):
    app = build_app(config=data_config)
    widgets = _collect_widgets(app)

    widgets["Include slvr/ms_mod-flagged channels"].value = True
    channel_select = widgets[_CHANNEL_LABEL]
    all_channels = list(channel_select.options.values())
    assert len(all_channels) == 3

    channel_select.value = all_channels
    assert len(channel_select.value) == 2


def test_multi_channel_view_tab(data_config):
    app = build_app(config=data_config)
    widgets = _collect_widgets(app)

    tabs = app.main.objects[0]
    assert tabs._names[1] == "Multi-channel view"
    montage_column = tabs.objects[1]
    montage_channel_row = montage_column.objects[1]
    montage_channel_select, montage_clear_button = montage_channel_row.objects
    montage_bokeh_pane = montage_column.objects[2]

    # Defaults to every currently-loaded LFP channel (2, by default -- one
    # slvr-flagged channel excluded) plus the two eye traces (H/V).
    assert len(montage_channel_select.value) == 2
    multi_line_source = montage_bokeh_pane.object.renderers[0].data_source
    assert len(multi_line_source.data["xs"]) == 2 + 2

    # Switching trial recomputes the montage without needing any of the
    # other-analyses toggles touched.
    first_trial = next(iter(widgets["Trial"].options.values()))
    other_trials = [v for v in widgets["Trial"].options.values() if v != first_trial]
    widgets["Trial"].value = other_trials[0]
    assert len(multi_line_source.data["xs"]) == 2 + 2

    # Narrowing the channel selection narrows the montage independently of
    # the (up to 2) channels picked for the trace/spectral view.
    one_channel = list(montage_channel_select.options.values())[:1]
    montage_channel_select.value = one_channel
    assert len(multi_line_source.data["xs"]) == 1 + 2

    montage_clear_button.clicks += 1
    assert montage_channel_select.value == []
    assert len(multi_line_source.data["xs"]) == 0 + 2

    # Including the flagged channel adds one more available LFP channel,
    # auto-selected as part of the new "select all" default.
    widgets["Include slvr/ms_mod-flagged channels"].value = True
    assert len(montage_channel_select.value) == 3
    multi_line_source = montage_bokeh_pane.object.renderers[0].data_source
    assert len(multi_line_source.data["xs"]) == 3 + 2


def test_clear_selection_button(data_config):
    app = build_app(config=data_config)
    widgets = _collect_widgets(app)

    channel_select = widgets[_CHANNEL_LABEL]
    clear_button = widgets["Clear selection"]
    assert channel_select.value

    clear_button.clicks += 1
    assert channel_select.value == []
    # main falls back to its single-channel layout objects (unchanged, no crash)
    assert app.main is not None
