import pytest

from graydataviz.derived import load_power
from graydataviz.exceptions import DerivedDataNotFoundError

from .conftest import DATE, MONKEY, SESSION


def test_load_power(power_file):
    power = load_power(MONKEY, DATE, SESSION, config=power_file)
    assert power.dims == ("roi", "freqs", "trials", "times")


def test_load_power_missing_file_raises_with_hint(data_config):
    with pytest.raises(DerivedDataNotFoundError) as exc_info:
        load_power(MONKEY, DATE, SESSION, config=data_config)
    assert "power_tt_1_br_1_at_cue_decim_20_hilbert.nc" in str(exc_info.value)
