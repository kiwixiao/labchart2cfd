# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Project memory lives inside the project folder so it travels with it:

@.claude/memory/MEMORY.md

`PROGRESS.md` at the repo root is the source of truth for where work stopped.

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
labchart2cfd convert-adicht <file.adicht> --out <dir>   # native LabChart file to a .mat like LabChart's export
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
- `adicht.py` reads native LabChart recordings (`.adicht`). The format has no public description; the layout was decoded from real recordings and is written down in the FORMAT NOTES at the bottom of that file. It was checked against LabChart's own exports bit for bit, and it raises `AdichtFormatError` on anything it was not checked on instead of guessing. Keep that property: never relax one of its checks to make a file open. Real recordings are patient data and are not in the repository, so its two real file tests are skipped unless `LABCHART2CFD_ADICHT_FILE` and `LABCHART2CFD_ADICHT_MAT` name a recording and LabChart's export of it. Run them after any change to this file.
- `recording.py` is the one entry point for opening a recording: `load_recording(path, notes=None)` picks the reader by extension and returns the same `LabChartData` either way. Native values are rounded to single precision first, as LabChart's export stores them, so both routes give identical numbers. The GUI uses it; the CLI `process` command still reads `.mat` only (`convert-adicht` turns a native file into one).

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
- CLI is a single Typer `app` with `process`, `validate`, `visualize`, `detect-steps`, `convert-adicht`, `gui`, `version`. `convert-adicht` only hands its options to `io/adicht.py`'s own `main`, so the two entry points cannot drift apart. `process` is the dispatcher that instantiates the right workflow; CT also calls `detect_steps` here when `--step-number` is given.
- GUI is PyQt5 (`matplotlib.use("QtAgg")`). The `gui` CLI command has a macOS-specific relaunch: on Darwin, it re-execs itself through `open python.app --args ...` guarded by `LABCHART2CFD_GUI=1` to get real keyboard focus. Do not remove that branch — launching the GUI directly on macOS loses keyboard input (see commit `1669bb3`).
- GUI layout (`gui/app.py`, `FlowProfileApp`): a sidebar of numbered sections beside the plot (1 Recording, 2 Study type, 3 Time window, then 4 Triggers, or for CT 4 CT scan and 5 Build the breath). Subject, output folder and the single Export button are pinned below the sections. `gui/widgets.py` holds the reusable widgets (`ChannelCombo`, `CollapsibleSection`, `VerticalScrollArea`), `gui/dialogs.py` the preview, result and guide dialogs, `gui/theme.py` the colours and style sheet.
- Three rules hold the GUI together. `_click_mode()` is the single answer to what a plot click means; the click handler dispatches on it and the banner above the plot displays it, so add a new click mode there and in `_BANNERS`, nowhere else. `_ct_export_inputs()` lets only the CT method on screen feed an export. `_export()` is the only exit, and it reports through `_show_export_result()`. `_export_blockers()` is both the text of the line above Export and the gate inside `_export()`: a condition the export would refuse belongs in that list, never only in an inner check, or the line says Ready while Export refuses (or names a problem while Export writes).
- Plotted signals are reduced for display only (`gui/decimate.py`): each line draws the minimum and maximum of each group of samples at the current zoom (`EnvelopeLine`, refreshed by `_on_view_changed`), and zooming in ends at the raw samples. Never plot a full channel directly and never thin by keeping every Nth sample: a real recording has 92 million samples per channel (9.5 s per redraw when drawn in full), and a stride hid short scan steps on the overview. Clicks, zoom limits and exports read the full arrays, not the drawn lines.
- The trigger scale is never fitted to the visible window. It starts at the whole range of the trigger channel in the block and changes only through the Trigger Y controls or the wheel over its axis (`_scale_y`, `_fit_y`, `_on_scroll`). Fitted to a scan step it showed 2 mV of noise on a real recording and hid the step. `_zoom_to_window` fits the flow scale only.
- LabChart comments live on `LabChartData.comments` (`Comment`: block and channel 1-indexed, channel None for all channels, time in seconds from the start of the block). Both readers fill it. The `.mat` reader drops all comments, with a note, when the table cannot be placed in time; it never guesses. `_draw_comments` adds them as plain artists in axes height coordinates, so they cannot move a scale.
- Nothing the user placed on the plot is removed without a question. `_ask_before_dropping()` is the one place that asks, and every control that redraws goes through it: the pickers (`_on_selection_changed`, which also does nothing when the plot would not change), All blocks and Selected block (`_user_show_overview`, `_user_show_block`), Select Step (`_user_select_step`), the study type, an overview tile of another block, and opening a recording. Only the Clear buttons do not ask. Changing the CT method removes the marks of the method left, after asking, so one method's marks are on the plot at a time. A new control that redraws must be wired to a `_user_...` wrapper, not to `_plot_selected` or `_go_to_overview` directly. An independent review found marks dropped silently twice, each time through a route the previous fix had not covered.
- `_plot_selected` checks the choice and calls `_draw_detail`. If drawing fails it goes back to the overview and says why, so the window is never left half drawn. Text that comes from a file (comments, channel titles, the file name) is drawn with `parse_math=False`: a dollar sign pair is a formula to matplotlib and a broken one raises while drawing.
- Comment lines are one `LineCollection` per plot. Labels are made on demand for the comments in view and withheld while more than `_COMMENT_LABELS_MAX` are in view. `_layout_comment_labels` and `_layout_step_labels` keep labels from printing over each other at the current zoom.
- Every dropdown, number box and slider in the sidebar is covered by `guard_wheel` (`gui/widgets.py`): the wheel scrolls the sidebar and never steps a control. A control added to the sidebar after `_build_sidebar` returns must be given the guard too.
- Channel titles are used to flag, never to choose. `_channel_name_warnings()` reports a role whose channel is named like something else (the default rows put Flow on an esophageal pressure channel for the ENT recordings), and the export asks before going on. Nothing selects a channel for the user.
- Widget attribute names such as `flow_row_spin` and `col_spin` date from when they were spin boxes. They are dropdowns now (`ChannelCombo` keeps `value()` and `setValue()`), and the names stay because handlers and tests use them.

### Tests
`tests/test_gui_*.py` drive the real window with Qt offscreen (`QT_QPA_PLATFORM=offscreen`, set in `tests/conftest.py`, which also provides the `win`, `mat_file` and `no_modal_dialogs` fixtures). A modal `QMessageBox` or `exec_()` hangs an offscreen run, so new GUI code shows results without blocking and tests use `no_modal_dialogs`. After any visual change, also run `python scripts/gui_screenshots.py <dir>` and look at the PNGs: a sidebar that clipped its controls passed every test and was only visible in a screenshot. Then run `python scripts/gui_native_check.py <dir>` on a machine with a display: it walks every study type under the real platform plugin, fonts and pixel ratio, lists controls whose text does not fit, and saves a picture of each state. Its first run found three defects the offscreen pictures had not shown (toolbar squeezed at the smallest window size, plot title cut off, marker times printed into each other).

`tests/test_io.py` and `tests/test_processing.py` cover the two lower layers. `tests/data/` and `tests/reference/` hold fixture `.mat` inputs and expected-output arrays used to validate parity with the MATLAB originals — treat reference files as authoritative when tuning algorithms.
