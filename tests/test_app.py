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
    widgets["Apply bandpass filter"].value = True
    widgets["Band preset"].value = widgets["Band preset"].options[0]


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
    # Two channels selected -> the inner main Column is rebuilt as
    # [info, lfp row, psd+coherence row].
    main_column = app.main.objects[0]
    assert len(main_column.objects) == 3
    bottom_row = main_column.objects[2]
    assert len(bottom_row.objects) == 3  # psd, coherence, psd


def test_selecting_more_than_two_channels_is_truncated(data_config):
    app = build_app(config=data_config)
    widgets = _collect_widgets(app)

    widgets["Include slvr/ms_mod-flagged channels"].value = True
    channel_select = widgets[_CHANNEL_LABEL]
    all_channels = list(channel_select.options.values())
    assert len(all_channels) == 3

    channel_select.value = all_channels
    assert len(channel_select.value) == 2


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
