"""Central configuration for where raw and derived Gray Lab data live on disk.

Both of this dataset's other consumers (GrayData-Analysis, phase_coupling_analysis)
hard-code the filesystem root and the list of available recording dates per monkey.
Here the root is configurable (constructor arg or environment variable) and the
available dates/sessions are discovered from disk on demand (see `discovery.py`)
instead of being duplicated as literal lists.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

#: Monkeys present in the dataset. Used only for validation, not for path building.
KNOWN_MONKEYS = ("lucy", "ethyl")

_DEFAULT_RAW_ROOT = os.environ.get("GRAYDATAVIZ_RAW_ROOT", "~/funcog/gda/GrayLab")
_DEFAULT_RESULTS_ROOT = os.environ.get("GRAYDATAVIZ_RESULTS_ROOT", "~/funcog/gda/Results")


@dataclass(frozen=True)
class DataConfig:
    """Filesystem layout for the dataset.

    Pass an explicit config to any loading function to point it at a different
    location (e.g. a local copy of a few sessions used in tests) without touching
    environment variables or global state.
    """

    raw_root: Path = field(default_factory=lambda: Path(_DEFAULT_RAW_ROOT).expanduser())
    results_root: Path = field(
        default_factory=lambda: Path(_DEFAULT_RESULTS_ROOT).expanduser()
    )

    def session_dir(self, monkey: str, date: str, session: int = 1) -> Path:
        """Directory containing the raw recording for one session."""
        return self.raw_root / monkey / date / f"session{session:02d}"

    def results_dir(self, monkey: str, date: str, session: int = 1) -> Path:
        """Directory containing derived NetCDF products for one session."""
        return self.results_root / monkey / date / f"session{session:02d}"


def default_config() -> DataConfig:
    return DataConfig()
