"""Access to the stimulus images embedded in a session's recording_info.

Contrary to what the rest of the pipeline (GDa, phase_coupling_analysis) assumes,
`sample_image` / `match_image` / `nonmatch_image` in `trial_info` are not opaque
category labels with images living elsewhere: `recording_info.mat` embeds the
actual stimuli for the session directly, as `image_data` (a cell array of RGB
matrices), `image_names`, and `image_locations` (x/y position in degrees). Each
`*_image` id in `trial_info` is a 1-based index into these arrays (fixation
trials, where no stimulus was shown, have a NaN id).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _is_missing(image_id: Any) -> bool:
    return image_id is None or (isinstance(image_id, float) and np.isnan(image_id))


def get_stimulus_image(recording_info: dict[str, Any], image_id: Any) -> np.ndarray | None:
    """Return the RGB matrix for a 1-based stimulus image id.

    Returns None if `image_id` is missing/NaN, the session has no embedded
    images, or the id is out of range.
    """
    if _is_missing(image_id):
        return None
    image_data = recording_info.get("image_data")
    if image_data is None:
        return None
    idx = int(image_id) - 1
    if idx < 0 or idx >= len(image_data):
        return None
    return np.asarray(image_data[idx])


def get_stimulus_name(recording_info: dict[str, Any], image_id: Any) -> str | None:
    """Return the stimulus name for a 1-based image id, if `image_names` is present."""
    if _is_missing(image_id):
        return None
    image_names = recording_info.get("image_names")
    if image_names is None:
        return None
    idx = int(image_id) - 1
    if idx < 0 or idx >= len(image_names):
        return None
    return str(image_names[idx])
