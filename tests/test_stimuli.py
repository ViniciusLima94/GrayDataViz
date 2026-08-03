import numpy as np

from graydataviz.metadata import load_session_metadata
from graydataviz.stimuli import get_stimulus_image, get_stimulus_name

from .conftest import DATE, IMAGE_NAMES, MONKEY, SESSION


def test_get_stimulus_image_and_name(data_config):
    metadata = load_session_metadata(MONKEY, DATE, SESSION, config=data_config)

    image = get_stimulus_image(metadata.recording_info, 1.0)
    assert image is not None
    assert image.shape == (4, 4, 3)

    name = get_stimulus_name(metadata.recording_info, 1.0)
    assert name == IMAGE_NAMES[0]

    image2 = get_stimulus_image(metadata.recording_info, 2.0)
    assert not np.array_equal(image, image2)


def test_get_stimulus_image_missing_id_returns_none(data_config):
    metadata = load_session_metadata(MONKEY, DATE, SESSION, config=data_config)

    assert get_stimulus_image(metadata.recording_info, np.nan) is None
    assert get_stimulus_image(metadata.recording_info, None) is None


def test_get_stimulus_image_out_of_range_returns_none(data_config):
    metadata = load_session_metadata(MONKEY, DATE, SESSION, config=data_config)

    assert get_stimulus_image(metadata.recording_info, 99) is None
