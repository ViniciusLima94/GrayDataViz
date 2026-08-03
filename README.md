# GrayDataViz

Data loading (and, eventually, visualization) toolkit for the Gray Lab primate
LFP dataset, replacing the loading code duplicated across `GrayData-Analysis`
(`GDa/`) and `phase_coupling_analysis` (`src/`).

This is phase 1: a standalone, tested loading API. Visualization comes later.

## What's different from `GDa` / `src`

- **Filesystem auto-discovery** instead of hard-coded per-monkey date lists:
  `list_monkeys()`, `list_dates(monkey)`, `list_sessions(monkey, date)` scan the
  raw data root and only report sessions that actually have both
  `recording_info.mat` and `trial_info.mat` present.
- **Configurable root paths** via `DataConfig` (constructor args or the
  `GRAYDATAVIZ_RAW_ROOT` / `GRAYDATAVIZ_RESULTS_ROOT` env vars) instead of
  paths baked into module-level globals.
- **One `xr.Dataset`** (`load_session`) with `"lfp"` and optional `"spikes"`
  data variables sharing a single `attrs` dict, instead of two `DataArray`s
  with the metadata dict duplicated between them.
- **Typed trial vocabulary** (`TrialType`, `BehavioralResponse` enums) instead
  of bare integer codes.
- **Descriptive errors**: missing raw/derived files raise
  `RawDataNotFoundError` / `DerivedDataNotFoundError` naming the exact path
  (and, for derived products, what files *are* present in that directory)
  instead of a bare `FileNotFoundError` from deep inside `xarray`/`h5py`.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

```python
from graydataviz import DataConfig, list_dates, load_session, load_power, TrialType

config = DataConfig()  # or DataConfig(raw_root=..., results_root=...)

dates = list_dates("lucy", config=config)

ds = load_session("lucy", dates[0], align_to="cue", config=config)
task_trials = ds.sel(trials=ds.lfp.trials)  # or use filter_trial_indexes()

power = load_power("lucy", dates[0], trial_type=TrialType.TASK, config=config)
```

## Layout

```
src/graydataviz/
├── config.py     # DataConfig: raw_root / results_root, path conventions
├── discovery.py  # list_monkeys / list_dates / list_sessions
├── io.py         # low-level .mat (legacy + HDF5) readers
├── metadata.py   # SessionMetadata: recording_info + trial_info
├── trials.py     # TrialType / BehavioralResponse, filter_trial_indexes
├── session.py    # load_session: raw LFP + spikes -> xr.Dataset
├── derived.py    # load_power / load_pec_strength / load_crackle_cooccurrence / load_burst_probability
└── exceptions.py
```

## Tests

Tests build small synthetic `.mat`/HDF5/NetCDF fixtures on the fly (see
`tests/conftest.py`) so the suite runs without access to the real dataset:

```bash
pytest
```
