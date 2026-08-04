from graydataviz.windows import DEFAULT_EVT_DT, default_evt_dt


def test_default_evt_dt_differs_per_monkey():
    # This is the exact bug this module fixes: using lucy's cue window on
    # ethyl's data sliced past the end of real trial recordings.
    assert default_evt_dt("lucy", "cue") != default_evt_dt("ethyl", "cue")


def test_default_evt_dt_matches_table():
    for (monkey, align_to), window in DEFAULT_EVT_DT.items():
        assert default_evt_dt(monkey, align_to) == window


def test_default_evt_dt_unknown_monkey_falls_back_to_lucy():
    assert default_evt_dt("unknown_monkey", "cue") == DEFAULT_EVT_DT[("lucy", "cue")]
