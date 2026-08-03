import numpy as np

from graydataviz.filters import DEFAULT_BANDS, band_presets, bandpass_filter


def test_bandpass_filter_attenuates_out_of_band_signal():
    fsample = 1000.0
    t = np.arange(0, 2, 1 / fsample)
    in_band = np.sin(2 * np.pi * 10 * t)  # inside 6-14 Hz
    out_of_band = np.sin(2 * np.pi * 60 * t)  # outside 6-14 Hz
    x = in_band + out_of_band

    filtered = bandpass_filter(x, fsample, 6, 14)

    # The filtered signal should look much more like the in-band component
    # than the raw mixture did.
    err_filtered = np.std(filtered - in_band)
    err_raw = np.std(x - in_band)
    assert err_filtered < 0.2 * err_raw


def test_bandpass_filter_lowpass_when_low_is_zero():
    fsample = 1000.0
    t = np.arange(0, 2, 1 / fsample)
    low_freq = np.sin(2 * np.pi * 2 * t)
    high_freq = np.sin(2 * np.pi * 100 * t)
    x = low_freq + high_freq

    filtered = bandpass_filter(x, fsample, 0, 6)

    assert np.std(filtered - low_freq) < 0.2 * np.std(x - low_freq)


def test_bandpass_filter_rejects_high_above_nyquist():
    import pytest

    with pytest.raises(ValueError):
        bandpass_filter(np.zeros(100), fsample=100.0, f_low=1, f_high=60)


def test_band_presets():
    assert band_presets("lucy") == DEFAULT_BANDS["lucy"]
    assert band_presets("unknown_monkey") == []
