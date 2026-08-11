"""Synthetic fixtures standing in for the real (cluster-only) Gray Lab dataset.

Builds small but format-faithful `recording_info.mat` (legacy MAT struct),
`trial_info.mat` (MAT v7.3 / HDF5 group of arrays), and per-trial LFP+spike
HDF5 files, plus a couple of derived NetCDF products, so the test suite
exercises the real file-format parsing without needing the actual cluster
mount.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest
import scipy.io as scio
import xarray as xr

from graydataviz.config import DataConfig

MONKEY = "lucy"
DATE = "150128"
SESSION = 1
FSAMPLE = 1000.0
N_TRIAL_SAMPLES = 6000  # length of each raw per-trial recording, in samples

# channel 2 (0-indexed) is flagged slvr and excluded by default
CHANNEL_NUMBERS = np.array([1, 2, 3])
AREA = np.array(["F1", "F1", "V4"])
DEPTH = np.array([1.0, 1.0, 2.0])
SLVR = np.array([0, 0, 1])
MS_MOD = np.array([0, 0, 0])

TRIAL_TYPE = np.array([1.0, 1.0, 2.0, 3.0])
BEHAVIORAL_RESPONSE = np.array([1.0, 0.0, 1.0, 1.0])
# Task trials (0, 1) reference real embedded images; fixation trials (2, 3) show none.
SAMPLE_IMAGE = np.array([1.0, 2.0, np.nan, np.nan])
MATCH_IMAGE = np.array([2.0, 1.0, np.nan, np.nan])
NONMATCH_IMAGE = np.array([1.0, 2.0, np.nan, np.nan])
SAMPLE_ON = np.array([1000, 1000, 1000, 1000])
SAMPLE_OFF = SAMPLE_ON + 200
MATCH_ON = SAMPLE_ON + np.array([1500, 1600, 1500, 1500])

N_TRIALS = len(TRIAL_TYPE)
N_CHANNELS_TOTAL = len(CHANNEL_NUMBERS)

N_IMAGES = 2
IMAGE_NAMES = ["apple", "banana"]
IMAGE_LOCATIONS = np.array([[1.0, 2.0], [3.0, 4.0]])


def _make_image_data(rng: np.random.Generator) -> np.ndarray:
    images = np.empty(N_IMAGES, dtype=object)
    for i in range(N_IMAGES):
        images[i] = rng.integers(0, 255, size=(4, 4, 3)).astype(np.uint8)
    return images


def _write_recording_info(path, rng: np.random.Generator):
    recording_info = {
        "channel_count": float(N_CHANNELS_TOTAL),
        "lfp_sampling_rate": FSAMPLE,
        "channel_numbers": CHANNEL_NUMBERS.astype(float),
        "area": AREA,
        "depth": DEPTH,
        "slvr": SLVR.astype(float),
        "ms_mod": MS_MOD.astype(float),
        "image_data": _make_image_data(rng),
        "image_names": np.array(IMAGE_NAMES, dtype=object),
        "image_locations": IMAGE_LOCATIONS,
    }
    scio.savemat(path, {"recording_info": recording_info})


def _write_trial_info(path):
    with h5py.File(path, "w") as f:
        grp = f.create_group("trial_info")
        grp.create_dataset("trial_type", data=TRIAL_TYPE)
        grp.create_dataset("behavioral_response", data=BEHAVIORAL_RESPONSE)
        grp.create_dataset("sample_image", data=SAMPLE_IMAGE)
        grp.create_dataset("match_image", data=MATCH_IMAGE)
        grp.create_dataset("nonmatch_image", data=NONMATCH_IMAGE)
        grp.create_dataset("sample_on", data=SAMPLE_ON.astype(float))
        grp.create_dataset("sample_off", data=SAMPLE_OFF.astype(float))
        grp.create_dataset("match_on", data=MATCH_ON.astype(float))


def _write_trial_recording(path, rng: np.random.Generator):
    """One per-trial raw recording: LFP for all channels + spike time references
    + calibrated eye position (shorter than the LFP window, matching the real
    per-trial files where eye_data/calib_horz+calib_vert end before lfp_data
    does)."""
    lfp = rng.standard_normal((N_TRIAL_SAMPLES, N_CHANNELS_TOTAL))
    n_eye_samples = N_TRIAL_SAMPLES - 200
    with h5py.File(path, "w") as f:
        f.create_dataset("lfp_data", data=lfp)

        ref_dtype = h5py.special_dtype(ref=h5py.Reference)
        refs = []
        for ch in range(N_CHANNELS_TOTAL):
            n_spikes = 5 + ch
            spike_samples = rng.integers(0, N_TRIAL_SAMPLES, size=n_spikes).astype(float)
            ds = f.create_dataset(f"_spikes_ch{ch}", data=np.sort(spike_samples))
            refs.append(ds.ref)
        ref_ds = f.create_dataset("spike_times", (1, N_CHANNELS_TOTAL), dtype=ref_dtype)
        ref_ds[0, :] = refs

        eye_grp = f.create_group("eye_data")
        eye_grp.create_dataset(
            "calib_horz", data=rng.standard_normal((n_eye_samples, 1))
        )
        eye_grp.create_dataset(
            "calib_vert", data=rng.standard_normal((n_eye_samples, 1))
        )
        eye_grp.create_dataset(
            "raw_horz", data=rng.standard_normal((n_eye_samples * 30, 1))
        )
        eye_grp.create_dataset(
            "raw_vert", data=rng.standard_normal((n_eye_samples * 30, 1))
        )


@pytest.fixture
def data_config(tmp_path) -> DataConfig:
    """A DataConfig pointing at a freshly-built synthetic raw dataset under tmp_path."""
    raw_root = tmp_path / "GrayLab"
    results_root = tmp_path / "Results"
    session_dir = raw_root / MONKEY / DATE / f"session{SESSION:02d}"
    session_dir.mkdir(parents=True)

    rng = np.random.default_rng(0)
    _write_recording_info(session_dir / "recording_info.mat", rng)
    _write_trial_info(session_dir / "trial_info.mat")

    for i in range(N_TRIALS):
        _write_trial_recording(session_dir / f"{DATE}_trial_{i:03d}.mat", rng)

    results_root.mkdir(parents=True)
    return DataConfig(raw_root=raw_root, results_root=results_root)


@pytest.fixture
def power_file(data_config) -> DataConfig:
    """Adds a synthetic power NetCDF product matching `load_power`'s naming convention."""
    session_results_dir = data_config.results_dir(MONKEY, DATE, SESSION)
    session_results_dir.mkdir(parents=True, exist_ok=True)

    power = xr.DataArray(
        np.random.default_rng(1).standard_normal((2, 3, 4, 5)),
        dims=("freqs", "trials", "times", "roi"),
        coords={
            "freqs": [4.0, 8.0],
            "trials": np.arange(3),
            "times": np.arange(4),
            "roi": ["F1", "F1", "V4", "V4", "V4"][:5],
        },
        attrs={"channels_labels": np.array([1, 2, 3, 4, 5])},
    )
    power.to_netcdf(session_results_dir / "power_tt_1_br_1_at_cue_decim_20_hilbert.nc")
    return data_config
