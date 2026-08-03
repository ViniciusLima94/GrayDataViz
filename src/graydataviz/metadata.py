"""Reading per-session recording and trial metadata.

Every session directory contains two files describing what was recorded:

- `recording_info.mat` (legacy MAT format): per-electrode metadata — brain area,
  depth, channel number, sampling rate, and slvr/ms_mod exclusion flags.
- `trial_info.mat` (MAT v7.3 / HDF5): per-trial metadata — trial type, behavioral
  response, stimulus identity, and cue/match onset times.

`load_session_metadata` reads both and normalizes them into a `SessionMetadata`
with the electrode info as a plain dict of arrays and the trial info as a
`pandas.DataFrame`, ready to be filtered/joined against by downstream code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import DataConfig, KNOWN_MONKEYS, default_config
from .exceptions import InvalidMonkeyError, RawDataNotFoundError
from .io import read_hdf5_mat, read_legacy_mat

RECORDING_INFO_FILE = "recording_info.mat"
TRIAL_INFO_FILE = "trial_info.mat"


@dataclass
class SessionMetadata:
    """Recording and trial metadata for a single session."""

    monkey: str
    date: str
    session: int
    recording_info: dict[str, Any] = field(repr=False)
    trial_info: pd.DataFrame = field(repr=False)

    @property
    def n_channels(self) -> int:
        return int(self.recording_info["channel_count"])

    @property
    def sampling_rate(self) -> float:
        return float(self.recording_info["lfp_sampling_rate"])


def load_session_metadata(
    monkey: str,
    date: str,
    session: int = 1,
    config: DataConfig | None = None,
) -> SessionMetadata:
    """Load recording and trial metadata for one session.

    Raises
    ------
    InvalidMonkeyError
        If `monkey` is not one of `graydataviz.config.KNOWN_MONKEYS`.
    RawDataNotFoundError
        If the session directory or its metadata files are missing.
    """
    if monkey not in KNOWN_MONKEYS:
        raise InvalidMonkeyError(
            f"monkey={monkey!r} is not one of the known monkeys {KNOWN_MONKEYS!r}"
        )

    config = config or default_config()
    session_dir = config.session_dir(monkey, date, session)
    recording_info_path = session_dir / RECORDING_INFO_FILE
    trial_info_path = session_dir / TRIAL_INFO_FILE

    missing = [p for p in (recording_info_path, trial_info_path) if not p.is_file()]
    if missing:
        raise RawDataNotFoundError(
            f"missing metadata file(s) for {monkey}/{date}/session{session:02d}: "
            f"{[str(p) for p in missing]}"
        )

    recording_info = _read_recording_info(recording_info_path)
    trial_info = _read_trial_info(trial_info_path)

    return SessionMetadata(
        monkey=monkey,
        date=date,
        session=session,
        recording_info=recording_info,
        trial_info=trial_info,
    )


def _read_recording_info(path: Path) -> dict[str, Any]:
    raw = read_legacy_mat(path)["recording_info"]
    return {key: np.squeeze(getattr(raw, key)) for key in raw._fieldnames}


def _read_trial_info(path: Path) -> pd.DataFrame:
    with read_hdf5_mat(path) as f:
        raw = f["trial_info"]
        data = {key: np.squeeze(raw[key][()]) for key in raw.keys()}
    return pd.DataFrame.from_dict(data, orient="columns")
