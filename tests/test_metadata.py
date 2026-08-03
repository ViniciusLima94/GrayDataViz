import pytest

from graydataviz.exceptions import InvalidMonkeyError, RawDataNotFoundError
from graydataviz.metadata import load_session_metadata

from .conftest import (
    AREA,
    CHANNEL_NUMBERS,
    DATE,
    FSAMPLE,
    MONKEY,
    N_CHANNELS_TOTAL,
    N_TRIALS,
    SESSION,
)


def test_load_session_metadata(data_config):
    metadata = load_session_metadata(MONKEY, DATE, SESSION, config=data_config)

    assert metadata.n_channels == N_CHANNELS_TOTAL
    assert metadata.sampling_rate == FSAMPLE
    assert list(metadata.recording_info["channel_numbers"].astype(int)) == list(
        CHANNEL_NUMBERS
    )
    assert list(metadata.recording_info["area"]) == list(AREA)
    assert len(metadata.trial_info) == N_TRIALS
    assert set(metadata.trial_info.columns) >= {
        "trial_type",
        "behavioral_response",
        "sample_on",
        "sample_off",
        "match_on",
        "sample_image",
    }


def test_invalid_monkey(data_config):
    with pytest.raises(InvalidMonkeyError):
        load_session_metadata("bob", DATE, SESSION, config=data_config)


def test_missing_session(data_config):
    with pytest.raises(RawDataNotFoundError):
        load_session_metadata(MONKEY, "999999", SESSION, config=data_config)
