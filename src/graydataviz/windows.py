"""Default trial time windows, per monkey and alignment event.

Ported from `phase_coupling_analysis/config.py`'s `return_evt_dt`. These aren't
arbitrary: each monkey's raw recordings only reliably contain enough samples
around the cue/match event for *that monkey's* window — using lucy's cue
window for ethyl (or vice versa) can slice past the end of a trial's actual
recording.
"""

from __future__ import annotations

from typing import Literal

#: (monkey, align_to) -> (t_start, t_end) in seconds relative to the alignment event.
DEFAULT_EVT_DT: dict[tuple[str, str], tuple[float, float]] = {
    ("lucy", "cue"): (-0.65, 3.00),
    ("lucy", "match"): (-2.2, 0.65),
    ("ethyl", "cue"): (-0.5, 2.7),
    ("ethyl", "match"): (-2.2, 0.65),
}


def default_evt_dt(monkey: str, align_to: Literal["cue", "match"]) -> tuple[float, float]:
    """The standard trial window for this monkey/alignment, or a same-shape
    fallback if the monkey isn't in `DEFAULT_EVT_DT` (e.g. an unlisted monkey
    in the raw data) -- callers should treat that fallback as a starting
    guess, not a validated window."""
    try:
        return DEFAULT_EVT_DT[(monkey, align_to)]
    except KeyError:
        return DEFAULT_EVT_DT[("lucy", align_to)]
