import numpy as np
import pytest

from graydataviz.exceptions import RawDataNotFoundError
from graydataviz.session import load_session

from .conftest import DATE, FSAMPLE, MONKEY, N_TRIALS, SESSION

EVT_DT = (-0.65, 3.00)
EXPECTED_N_TIMES = int(FSAMPLE * (EVT_DT[1] - EVT_DT[0]))


def test_load_session_default_excludes_slvr_channel(data_config):
    ds = load_session(MONKEY, DATE, SESSION, evt_dt=EVT_DT, config=data_config)

    assert ds.sizes["trials"] == N_TRIALS
    assert ds.sizes["roi"] == 2  # channel index 2 is flagged slvr, excluded by default
    assert ds.sizes["time"] == EXPECTED_N_TIMES
    assert list(ds.roi.values) == ["F1", "F1"]
    assert "spikes" not in ds
    assert ds.attrs["fsample"] == FSAMPLE
    assert ds.attrs["monkey"] == MONKEY


def test_load_session_include_slvr_channel(data_config):
    ds = load_session(
        MONKEY, DATE, SESSION, evt_dt=EVT_DT, exclude_slvr_msmod=False, config=data_config
    )
    assert ds.sizes["roi"] == 3


def test_load_session_with_spike_times(data_config):
    ds = load_session(
        MONKEY, DATE, SESSION, evt_dt=EVT_DT, load_spike_times=True, config=data_config
    )
    assert "spikes" in ds
    assert ds.spikes.dims == ds.lfp.dims
    assert ds.spikes.dtype.kind in ("i", "u")
    assert set(np.unique(ds.spikes.values)) <= {0, 1}


def test_invalid_align_to(data_config):
    with pytest.raises(ValueError):
        load_session(MONKEY, DATE, SESSION, align_to="banana", config=data_config)


def test_missing_session_raises(data_config):
    with pytest.raises(RawDataNotFoundError):
        load_session(MONKEY, "999999", SESSION, config=data_config)
