# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Dependency management uses `uv` (see `uv.lock`); the package installs as `labchart2cfd` via the `[project.scripts]` entry in `pyproject.toml`.

```bash
# Install in editable mode with dev extras
uv pip install -e ".[dev]"

# Run the full test suite (pytest config lives in pyproject.toml)
pytest

# Run a single test or class
pytest tests/test_processing.py::TestDriftCorrection
pytest tests/test_processing.py -k drift

# CLI entry points (Typer)
labchart2cfd process <file.mat> <subject> --start S --end E [--workflow standard|cpap|phase-contrast|ct]
labchart2cfd validate <file.mat> [--verbose]
labchart2cfd visualize <file.mat> --row R --column C
labchart2cfd detect-steps <file.mat>          # CT trigger discovery
labchart2cfd                                  # bare command launches the PyQt5 GUI
labchart2cfd gui                              # explicit form (used by the macOS relaunch — keep it)

# The package is also runnable as a module
python -m labchart2cfd <subcommand>
```

No linter/formatter is wired into `pyproject.toml`; correctness is enforced purely by the pytest suite.

## Architecture

The package converts LabChart `.mat` exports into Star-CCM+ boundary-condition CSVs. Data moves through four layers; changes almost always touch more than one of them.

### 1. IO layer — `src/labchart2cfd/io/`
- `labchart.py` ports the MATLAB `LabChart.m` reader. `LabChartData` stores **1-indexed** `[channels][blocks]` cell arrays of (time, data) pairs, mirroring MATLAB conventions (`get_data(row, column)`, `get_time(row, column)`, `is_block_empty`). Row/column arguments throughout the CLI, GUI, and workflows are 1-indexed for this reason — do not silently switch to 0-indexed.
- `csv_export.py` writes the `{subject}FlowProfile.csv` / `{subject}PressureProfile.csv` Star-CCM+ outputs.

### 2. Processing primitives — `src/labchart2cfd/processing/`
Pure functions over NumPy arrays, each in its own module: `drift_correction`, `resampling`, `smoothing`, `unit_conversion` (includes `AIR_DENSITY_KG_M3`, `CMH2O_TO_PA`), `step_detection` (CT trigger finder), `rearrangement` (cyclic shift to a landmark time + image-index math for 4D CT).

These are the units under test in `tests/test_processing.py` — many tests exist to pin Python behaviour to the original MATLAB output, so changes here need reference comparisons, not just "looks fine".

### 3. Workflow layer — `src/labchart2cfd/workflows/`
`BaseWorkflow` (ABC) defines the pipeline contract and returns a `WorkflowResult` dataclass (`time`, `mass_flow`, optional `pressure`, `drift_error`, `sample_rate`, `metadata`). Concrete workflows differ in *where* drift correction runs and *what* the signal represents:

- `StandardOSAMRIWorkflow` — drift correction on the full signal, then window.
- `CPAPWorkflow` — drift correction on the already-windowed signal.
- `PhaseContrastWorkflow` — xenon density (5.761 kg/m³) with voltage calibration; supports `process_with_bag_config(data, row, col, bag_id)` for Bag1/Bag2/Bag4 presets; defaults target sample rate to 1000 Hz.
- `CTWorkflow` — step-trigger driven. Caller passes `inhale_start_time`; signal is rearranged at that landmark *before* drift correction (positive sign). `metadata` carries the image-index mapping computed from `temporal_resolution`.

When adding a workflow, update both `workflows/__init__.py` and the `WorkflowType` enum + dispatch in `cli/main.py:process`.

### 4. Surfaces — `cli/main.py`, `gui/app.py`, `visualization/plots.py`
- CLI is a single Typer `app` with `process`, `validate`, `visualize`, `detect-steps`, `gui`, `version`. `process` is the dispatcher that instantiates the right workflow; CT also calls `detect_steps` here when `--step-number` is given.
- GUI is PyQt5 (`matplotlib.use("QtAgg")`). The `gui` CLI command has a macOS-specific relaunch: on Darwin, it re-execs itself through `open python.app --args ...` guarded by `LABCHART2CFD_GUI=1` to get real keyboard focus. Do not remove that branch — launching the GUI directly on macOS loses keyboard input (see commit `1669bb3`).

### Tests
`tests/test_io.py` and `tests/test_processing.py` cover the two lower layers. `tests/data/` and `tests/reference/` hold fixture `.mat` inputs and expected-output arrays used to validate parity with the MATLAB originals — treat reference files as authoritative when tuning algorithms.
