# GrayDataViz

Data loading and visualization toolkit for the Gray Lab primate LFP dataset,
replacing the loading code duplicated across `GrayData-Analysis` (`GDa/`) and
`phase_coupling_analysis` (`src/`).

Phase 1 was a standalone, tested loading API. Phase 2 (this update) adds a
Panel-based GUI for browsing raw LFP recordings: pick a monkey/date/session/
trial/channel, see the trial type and behavioral response, see the cue/match/
non-match stimulus images (these are embedded directly in each session's
`recording_info.mat` as `image_data`/`image_names` — not separate files),
overlay the spike raster, and overlay a bandpass-filtered version of the trace
on top of the raw signal.

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
pip install -e ".[dev]"       # loading API + tests
pip install -e ".[gui]"       # + panel/matplotlib, to also run the GUI
```

## Usage: loading API

```python
from graydataviz import DataConfig, list_dates, load_session, load_power, TrialType

config = DataConfig()  # or DataConfig(raw_root=..., results_root=...)

dates = list_dates("lucy", config=config)

ds = load_session("lucy", dates[0], align_to="cue", config=config)
task_trials = ds.sel(trials=ds.lfp.trials)  # or use filter_trial_indexes()

power = load_power("lucy", dates[0], trial_type=TrialType.TASK, config=config)
```

## Usage: GUI

```bash
graydataviz-gui                # after `pip install -e ".[gui]"`
# or
python -m graydataviz.app
```

By default it points at `DataConfig()` (i.e. `GRAYDATAVIZ_RAW_ROOT` /
`GRAYDATAVIZ_RESULTS_ROOT`, or `~/funcog/gda/GrayLab` / `~/funcog/gda/Results`).
Set those env vars to point at wherever the raw data actually lives before
launching.

The sidebar lets you pick monkey → date → session → alignment → trial →
channel; the main panel shows the LFP trace (with optional bandpass-filtered
overlay and spike-raster overlay) plus the cue/match/non-match stimulus images
for the selected trial, alongside its trial type and behavioral response.

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
├── stimuli.py    # get_stimulus_image / get_stimulus_name (from embedded image_data)
├── filters.py    # bandpass_filter (zero-phase Butterworth), per-monkey DEFAULT_BANDS
├── app.py        # Panel GUI (build_app / main)
└── exceptions.py
```

## Tests

Tests build small synthetic `.mat`/HDF5/NetCDF fixtures on the fly (see
`tests/conftest.py`) so the suite runs without access to the real dataset:

```bash
pytest
```
