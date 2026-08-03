"""Trial-type/behavioral-response vocabulary and trial filtering.

The raw `trial_info` table encodes trial type and behavioral outcome as small
integer codes. `TrialType` and `BehavioralResponse` give those codes names so
calling code reads as `TrialType.TASK` rather than a bare `1`.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np
import pandas as pd


class TrialType(IntEnum):
    TASK = 1
    FIXATION_INTERLEAVED = 2
    FIXATION_BLOCKED = 3


class BehavioralResponse(IntEnum):
    INCORRECT = 0
    CORRECT = 1


def filter_trial_indexes(
    trial_info: pd.DataFrame,
    trial_type: int | list[int] | None = None,
    behavioral_response: int | list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Select trials matching the given trial type(s) and/or behavioral response(s).

    Parameters
    ----------
    trial_info:
        DataFrame as returned in `SessionMetadata.trial_info`, indexed by the
        default RangeIndex with a `trial_index` column and `trial_type` /
        `behavioral_response` columns.
    trial_type:
        One or more `TrialType` values (or their raw ints) to keep. `None` means
        no filtering on trial type.
    behavioral_response:
        One or more `BehavioralResponse` values (or their raw ints) to keep.
        `None` means no filtering on behavioral response.

    Returns
    -------
    filtered_trials:
        `trial_index` values of the matching trials (matches the `trials`
        coordinate used by `load_session`).
    filtered_trials_idx:
        Positional (DataFrame index) locations of the matching trials.
    """
    mask = pd.Series(True, index=trial_info.index)
    if trial_type is not None:
        mask &= trial_info["trial_type"].isin(np.atleast_1d(trial_type))
    if behavioral_response is not None:
        mask &= trial_info["behavioral_response"].isin(np.atleast_1d(behavioral_response))

    selected = trial_info[mask]
    return selected["trial_index"].values, selected.index.values
