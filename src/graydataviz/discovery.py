"""Filesystem-based discovery of available monkeys, recording dates, and sessions.

GrayData-Analysis and phase_coupling_analysis both hard-code the list of recording
dates per monkey (`get_dates()`), duplicated between the two repos and requiring a
manual edit whenever a new session is added. Here the same information is derived
by scanning `DataConfig.raw_root` for directories that actually contain a
`recording_info.mat` / `trial_info.mat` pair, so newly added sessions show up
automatically and typos/removed sessions can't silently drift out of sync.
"""

from __future__ import annotations

import re

from .config import DataConfig, KNOWN_MONKEYS, default_config

_DATE_RE = re.compile(r"^\d{6}$")
_SESSION_RE = re.compile(r"^session(\d{2})$")


def list_monkeys(config: DataConfig | None = None) -> list[str]:
    """List monkey subdirectories present under the raw data root.

    Only names in `KNOWN_MONKEYS` are returned, in case the raw root also
    contains unrelated directories (scratch files, other datasets, ...).
    """
    config = config or default_config()
    if not config.raw_root.is_dir():
        return []
    present = {p.name for p in config.raw_root.iterdir() if p.is_dir()}
    return sorted(present & set(KNOWN_MONKEYS))


def list_dates(monkey: str, config: DataConfig | None = None) -> list[str]:
    """List recording dates (YYMMDD) available for a monkey.

    A date is included only if it has at least one session directory containing
    both `recording_info.mat` and `trial_info.mat` — a bare or partially-populated
    date directory does not count as an available recording.
    """
    config = config or default_config()
    monkey_dir = config.raw_root / monkey
    if not monkey_dir.is_dir():
        return []

    dates = []
    for date_dir in monkey_dir.iterdir():
        if not date_dir.is_dir() or not _DATE_RE.match(date_dir.name):
            continue
        if _has_any_session(date_dir):
            dates.append(date_dir.name)
    return sorted(dates)


def list_sessions(monkey: str, date: str, config: DataConfig | None = None) -> list[int]:
    """List session numbers available for a given monkey/date."""
    config = config or default_config()
    date_dir = config.raw_root / monkey / date
    if not date_dir.is_dir():
        return []

    sessions = []
    for session_dir in date_dir.iterdir():
        match = _SESSION_RE.match(session_dir.name) if session_dir.is_dir() else None
        if match and _is_populated_session_dir(session_dir):
            sessions.append(int(match.group(1)))
    return sorted(sessions)


def _has_any_session(date_dir) -> bool:
    return any(
        _SESSION_RE.match(p.name) and _is_populated_session_dir(p)
        for p in date_dir.iterdir()
        if p.is_dir()
    )


def _is_populated_session_dir(session_dir) -> bool:
    return (session_dir / "recording_info.mat").is_file() and (
        session_dir / "trial_info.mat"
    ).is_file()
