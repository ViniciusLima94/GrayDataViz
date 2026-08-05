"""Default trial time windows, per monkey and alignment event.

The `cue` values are ported from `phase_coupling_analysis/util.py`'s
`return_evt_dt(monkey)` — NOT `config.py`'s same-named function, which is
never actually imported by any pipeline script there (`savepower.py`,
`savecoherence*.py`, `phasedifferences.py`, `save_burst_trains.py` all go
through `util.py`'s `load_session_data`, which calls `util.py`'s
`return_evt_dt`). That function ignores alignment entirely and only takes
`monkey`, and every pipeline invocation in `run.sh` uses `align="cue"` —
`match` alignment is never actually exercised in that codebase, so there's
no reference value to port for it; the numbers here are a reasonable
same-shape guess, not a verified match.

These values aren't arbitrary either way: each monkey's raw recordings only
reliably contain enough samples around the cue/match event for *that
monkey's* window — using lucy's cue window for ethyl (or vice versa) can
slice past the end of a trial's actual recording.
"""

from __future__ import annotations

from typing import Literal

#: (monkey, align_to) -> (t_start, t_end) in seconds relative to the alignment event.
DEFAULT_EVT_DT: dict[tuple[str, str], tuple[float, float]] = {
    ("lucy", "cue"): (-0.65, 2.00),
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
