"""Loading raw LFP (and optionally spike-time) recordings for a session.

This is the standalone reimplementation of `GDa.session.session.read_from_mat` /
`src.session.session.read_from_mat`. Differences from that implementation:

- LFP and spike times are returned together as a single `xr.Dataset` (data
  variables `"lfp"` and, if requested, `"spikes"`) sharing one `attrs` dict,
  instead of two separate `xr.DataArray` objects with the metadata dict
  duplicated between them.
- Errors (missing session directory, no trial files found) raise
  `RawDataNotFoundError` with a message naming the missing path, instead of
  letting a bare `FileNotFoundError`/`IndexError` propagate from deep inside
  the read loop.
"""

from __future__ import annotations

import glob
import os
from typing import Literal

import numpy as np
import xarray as xr
from tqdm import tqdm

from .config import DataConfig, default_config
from .exceptions import RawDataNotFoundError
from .io import read_hdf5_mat
from .metadata import load_session_metadata
from .trials import TrialType
from .windows import default_evt_dt

AlignTo = Literal["cue", "match"]


def load_session(
    monkey: str,
    date: str,
    session: int = 1,
    align_to: AlignTo = "cue",
    evt_dt: tuple[float, float] | None = None,
    exclude_slvr_msmod: bool = True,
    only_unique_recordings: bool = False,
    load_spike_times: bool = False,
    load_eye: bool = False,
    config: DataConfig | None = None,
    show_progress: bool = False,
) -> xr.Dataset:
    """Load raw LFP (and optionally binary spike-time rasters, and/or eye
    position) for one session.

    Parameters
    ----------
    monkey, date, session:
        Identify the recording, as accepted by `DataConfig.session_dir`.
    align_to:
        Whether trial windows are cut relative to cue onset or match onset.
    evt_dt:
        `(t_start, t_end)` window (seconds) around the alignment event to keep.
        Defaults to the monkey/align_to-specific window from
        `graydataviz.windows.DEFAULT_EVT_DT` -- using another monkey's window
        (e.g. lucy's on ethyl's data) can slice past the end of a trial's
        actual recording.
    exclude_slvr_msmod:
        Drop channels flagged with short-latency visual response or
        microsaccade modulation. Matches the default behavior of the previous
        `session(slvr_msmod=False)` implementations.
    only_unique_recordings:
        Restrict to channels marked non-redundant in `<monkey>/unique_recordings.nc`.
    load_spike_times:
        Also build a binary spike raster aligned to the same window.
    load_eye:
        Also load calibrated horizontal/vertical eye position, aligned to the
        same window. Each per-trial file's `eye_data/calib_horz`+`calib_vert`
        share the same sample clock as `lfp_data` (verified empirically:
        their peak deflection lines up with `match_on + reaction_time`, i.e.
        the saccade to the match target, across many trials) but the
        recording can end before the LFP window does, in which case the
        tail is left as NaN.
    show_progress:
        Display a tqdm progress bar while reading per-trial files.

    Returns
    -------
    xr.Dataset
        Data variables:
            - `"lfp"`: dims `("trials", "roi", "time")`
            - `"spikes"` (only if `load_spike_times=True`): same dims, binary
            - `"eye"` (only if `load_eye=True`): dims `("trials", "eye_axis",
              "time")`, `eye_axis` coord `["horizontal", "vertical"]`
        Shared attrs: `nC`, `fsample`, `channels_labels`, `stim`, `indch`,
        `t_cue_on`, `t_cue_off`, `t_match_on`, `monkey`, `date`, `session`,
        `align_to`.
    """
    if align_to not in ("cue", "match"):
        raise ValueError(f'align_to must be "cue" or "match", got {align_to!r}')
    if evt_dt is None:
        evt_dt = default_evt_dt(monkey, align_to)

    config = config or default_config()
    metadata = load_session_metadata(monkey, date, session, config=config)
    session_dir = config.session_dir(monkey, date, session)

    trial_info = metadata.trial_info
    trial_info = trial_info[trial_info["trial_type"].isin([t.value for t in TrialType])]
    trial_info = trial_info.rename_axis("trial_index").reset_index()

    files = sorted(glob.glob(os.path.join(session_dir, date + "*")))
    if not files:
        raise RawDataNotFoundError(f"no per-trial recording files found in {session_dir}")

    recording_info = metadata.recording_info
    fsample = float(recording_info["lfp_sampling_rate"])

    t_cue_on = trial_info["sample_on"].values
    t_cue_off = trial_info["sample_off"].values
    t_match_on = trial_info["match_on"].values
    t0 = t_cue_on if align_to == "cue" else t_match_on

    indch = np.arange(int(recording_info["channel_count"]))
    if exclude_slvr_msmod:
        keep = (recording_info["slvr"] == 0) & (recording_info["ms_mod"] == 0)
        indch = indch[keep]

    n_trials = len(trial_info)
    n_times = int(fsample * (evt_dt[1] - evt_dt[0]))
    n_channels = len(indch)
    time = np.arange(evt_dt[0], evt_dt[1], 1 / fsample)[:n_times]

    lfp = np.empty((n_trials, n_channels, n_times))
    spikes = np.zeros((n_trials, n_channels, n_times), dtype=int) if load_spike_times else None
    eye = np.full((n_trials, 2, n_times), np.nan) if load_eye else None

    trial_positions = trial_info["trial_index"].values
    iterator = tqdm(range(n_trials)) if show_progress else range(n_trials)
    for i in iterator:
        file_index = trial_positions[i]
        if file_index >= len(files):
            raise RawDataNotFoundError(
                f"trial_index {file_index} has no corresponding file in {session_dir} "
                f"(found {len(files)} files)"
            )
        with read_hdf5_mat(files[file_index]) as f:
            trial_lfp = np.transpose(f["lfp_data"])
            indb = int(t0[i] + fsample * evt_dt[0])
            inde = indb + n_times
            lfp[i] = trial_lfp[indch, indb:inde]

            if load_spike_times:
                refs = f["spike_times"][0][indch]
                for ch, ref in enumerate(refs):
                    spike_frames = np.asarray(f[ref]) - t0[i]
                    in_window = (spike_frames >= fsample * evt_dt[0]) & (
                        spike_frames < fsample * evt_dt[1]
                    )
                    # Re-base to the window start so index 0 aligns with evt_dt[0];
                    # spike_frames can otherwise be negative, wrapping via numpy indexing.
                    frames = (spike_frames[in_window] - fsample * evt_dt[0]).astype(int)
                    if len(frames) > 0:
                        spikes[i, ch, frames] = 1

            if load_eye and "eye_data" in f:
                eye_data = f["eye_data"]
                for axis, mat_key in enumerate(("calib_horz", "calib_vert")):
                    trace = np.asarray(eye_data[mat_key]).ravel()
                    # The calibrated eye trace can end before the LFP window
                    # does (see the load_eye docstring) -- whatever's past its
                    # end is left as NaN rather than wrapping/erroring.
                    seg = trace[indb:inde]
                    eye[i, axis, : len(seg)] = seg

    stimulus = trial_info["sample_image"].values
    channels_labels = recording_info["channel_numbers"][indch]
    roi = np.array(recording_info["area"][indch], dtype="<U13")

    if only_unique_recordings:
        indch, roi, channels_labels, lfp, spikes = _restrict_to_unique_recordings(
            monkey, date, config, indch, roi, channels_labels, lfp, spikes
        )

    coords = {"trials": trial_positions, "roi": roi, "time": time}
    data_vars = {"lfp": (("trials", "roi", "time"), lfp)}
    if load_spike_times:
        data_vars["spikes"] = (("trials", "roi", "time"), spikes)
    if load_eye:
        coords["eye_axis"] = ["horizontal", "vertical"]
        data_vars["eye"] = (("trials", "eye_axis", "time"), eye)

    ds = xr.Dataset(data_vars, coords=coords)
    ds.attrs = {
        "nC": n_channels,
        "fsample": fsample,
        "channels_labels": channels_labels.astype(np.int64),
        "stim": stimulus,
        "indch": indch,
        "t_cue_on": t_cue_on,
        "t_cue_off": t_cue_off,
        "t_match_on": t_match_on,
        "monkey": monkey,
        "date": date,
        "session": session,
        "align_to": align_to,
    }
    return ds


def _restrict_to_unique_recordings(monkey, date, config, indch, roi, channels_labels, lfp, spikes):
    unique_recordings_path = config.raw_root / monkey / "unique_recordings.nc"
    if not unique_recordings_path.is_file():
        raise RawDataNotFoundError(
            f"only_unique_recordings=True but {unique_recordings_path} does not exist"
        )
    unique_recordings = xr.load_dataarray(unique_recordings_path)
    unique_channels = np.where(unique_recordings.sel(dates=date).data == 1)[0] + 1
    keep = np.hstack([i for i in range(len(channels_labels)) if channels_labels[i] in unique_channels])

    indch = indch[keep]
    roi = roi[keep]
    channels_labels = channels_labels[keep]
    lfp = lfp[:, keep, :]
    if spikes is not None:
        spikes = spikes[:, keep, :]
    return indch, roi, channels_labels, lfp, spikes
