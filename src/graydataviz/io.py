"""Low-level readers for the two raw-data formats used by the dataset.

`recording_info.mat` is written in the legacy (pre-7.3) MAT format and is read with
`scipy.io.loadmat`. `trial_info.mat` and the per-trial LFP/spike files are saved as
MAT v7.3, which is really HDF5 under the hood, and are read with `h5py`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import h5py
import scipy.io as scio


def read_legacy_mat(path: str | Path) -> dict[str, Any]:
    """Read an old-format (pre-7.3) .mat file into a dict of arrays/structs."""
    return scio.loadmat(os.fspath(path), squeeze_me=True, struct_as_record=False)


def read_hdf5_mat(path: str | Path) -> h5py.File:
    """Open a MAT v7.3 (HDF5) file for reading.

    Returns an open `h5py.File`; the caller is responsible for closing it
    (or using it as a context manager) once done with any referenced datasets.
    """
    return h5py.File(os.fspath(path), "r")


def save_mat(path: str | Path, data: dict[str, Any]) -> None:
    """Write a dict of arrays to a legacy-format .mat file."""
    scio.savemat(os.fspath(path), data)
