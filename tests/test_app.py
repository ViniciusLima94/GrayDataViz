from graydataviz.app import build_app

from .conftest import MONKEY, N_TRIALS


def test_build_app_smoke(data_config):
    app = build_app(config=data_config)
    assert app is not None

    # Walk the returned template to find the widgets driving the app, and
    # exercise the reactive callbacks by changing a couple of them, to make
    # sure the whole load -> plot pipeline runs without raising.
    widgets = {}

    def _collect(obj):
        label = getattr(obj, "label", None)
        if label:
            widgets.setdefault(label, obj)
        for child in getattr(obj, "objects", []):
            _collect(child)

    for root in list(app.sidebar) + list(app.main):
        _collect(root)

    assert widgets["Monkey"].value == MONKEY
    assert len(widgets["Trial"].options) == N_TRIALS
    assert len(widgets["Channel"].options) == 2  # one channel excluded (slvr)

    first_trial = next(iter(widgets["Trial"].options.values()))
    other_trials = [v for v in widgets["Trial"].options.values() if v != first_trial]
    widgets["Trial"].value = other_trials[0]

    widgets["Overlay spikes"].value = True
    widgets["Apply bandpass filter"].value = True
    widgets["Band preset"].value = widgets["Band preset"].options[0]
