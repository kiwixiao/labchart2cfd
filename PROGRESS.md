# PROGRESS

## NEXT STEP
Click through `feature/gui-pro-redesign` by hand on one real MRI recording and one real CT recording, then merge it if it holds (the user merges). The branch is verified by tests and by offscreen screenshots of a synthetic recording only; no person has driven it on real data yet.

Still open from before, neither blocking:
1. **Confirm Star-CCM+ accepts the CSV data format.** The data column is scientific notation (`3.001234560e-04`) because the old `%.5f` quantised mass flow (~1e-4 kg/s) to about 61 distinct values. Nobody has loaded one of the new files into Star-CCM+. If its table reader rejects it, switch to a high-precision fixed format in `io/csv_export.py` rather than reverting to `%.5f`.
2. **MATLAB parity on a real subject.** Still unmeasured. The drift-correction fix deliberately diverges from the MATLAB original, which carries the same `abs()` bug, so old outputs for positive-drift recordings will legitimately differ.

## LAST SESSION
2026-11-20. GUI redesign on `feature/gui-pro-redesign`, built test first in four units. 265 tests (was 143). Engine untouched: no file under `processing/`, `workflows/` or `io/` changed.

What changed for the user:
- **Layout.** Controls moved from a tab strip under the plot to a sidebar of numbered sections beside it (1 Recording, 2 Study type, 3 Time window, then 4 Triggers, or for CT 4 CT scan and 5 Build the breath). Subject, output folder and Export are pinned below.
- **Recording.** Opening a file loads it and shows the overview in one action (also by drag and drop, or Ctrl+O). Channels are picked by the names stored in the file. Clicking an overview tile opens that block. Changing a picker redraws the view.
- **Click modes.** A banner above the plot always says what a click will do, and turns red while the zoom or pan tool swallows clicks.
- **CT.** The three methods (reorder parts, landmarks, cycle frames) sit behind one chooser that shows one at a time. Manual mode follows the method instead of a separate enable switch. The export reads only the method on screen.
- **Export.** One Export button. The synthetic preview can be cancelled. Existing files are replaced only after a question. A result panel lists the files, the numbers and the sanity plot, with a missing pressure profile as a warning at the top.

Behaviour changes worth knowing when testing by hand:
- Cycle labels no longer override landmarks. Whichever CT method is chosen is the one exported.
- In CT with the parts method, plain clicks never set Start/End (that was already so once manual mode was ticked; it is now so from the moment the method is chosen). Start/End come from Select Step, the offsets, or typing.
- A hand picked output folder now survives typing the Subject ID. It resets only when another recording is opened.
- The study type list shows "CPAP" and "Phase contrast (xenon)". The handlers still receive `cpap` and `phase-contrast`.

**First run on a real recording** (the user's LabChart export of ENT005: 5 channels, 92,144,000 samples per channel at 40 kHz in block 2). It loaded in about 3 s and found 7 scan steps whose times agree with the recording's comments. It also showed three weaknesses that synthetic data had hidden, all fixed:
- Every sample was drawn, three times, on every redraw: 21.5 s to open the block and 9.5 s per redraw at full range. Lines now draw the minimum and maximum of each group of samples at the current zoom (`gui/decimate.py`): 0.6 s and 0.1 s on the same file. Zooming in ends at the raw samples. Only the picture is reduced.
- The overview kept every Nth sample and showed 4 trigger spikes where 7 steps exist. It uses the same minimum and maximum reduction now.
- The default rows put Flow on "2: ESOPH PRESS" and Pressure on "4: VOL" for this recording. A warning under the pickers now names the mismatch, and the export asks before going on. It flags and never chooses.

**Native LabChart files.** The `.adicht` format has no public description. It was decoded from real recordings (`io/adicht.py`, layout written in its FORMAT NOTES) and checked against LabChart's own exports:
- ENT005, against the export the user made by hand for this purpose: all 13 shared variables identical, 461,780,000 samples with zero differing bits, all five channels including the computed VOL. Checked with a script that uses none of the reader's code.
- 72 recordings that have a LabChart export (LabChart 6.1.2 and 8.1.11 to 8.1.30): 1,140 channel records, 5,829,009,000 samples, zero differing. 1,129 exact (844 recorded, 285 Integral channels recomputed). 11 channel records in 2 recordings carry another LabChart calculation and are left empty and named. That run used the reader revision just before its last edit, which its author states changed comments only. On the committed file (SHA-256 `471bff64…`) an independent comparison script found every shared variable identical for ENT 002, 003, 004 and 005: 1,561,820,000 samples, zero differing bits. A rerun of all 72 pairs on the committed file was requested; its result is not recorded here yet.
- Opening the native file through `load_recording` gives arrays identical to opening the export, for every channel and block, data and time (the acceptance test, run on ENT005).
- The GUI opens `.adicht` directly (dialog, drop, Ctrl+O) and shows what was recomputed or left out under the file name. `labchart2cfd convert-adicht` writes the `.mat`.
- Limits: it refuses a layout it was not checked on and does not write `rangemin`/`rangemax` (nor `comtext` when there are no comments). Of 198 native files on the laptop, 158 open and 40 are refused: 19 saved from a selection, 7 with deleted records, 8 with display or units flag values never seen in a checked file, 3 where a record lacks a channel, 2 imports from other systems, 1 with a channel shorter than its record. When a new kind of recording appears, export one by hand and run the two real file tests (see README).
- Known gap: only Integral calculations had an export to compare with. A channel with another LabChart calculation is left empty when its marker is found. That the marker covers every kind of calculation is an inference; a calculation stored without it would come back as the raw input, unnoticed. The opposite error was seen once (a recorded channel left out by a false alarm), which is the safe direction.
- The decoding scripts and their result files are kept in `.claude/adicht_decoding/` (git ignored, because they name recordings). The session's scratch folder does not survive a reboot.

Peak memory on that file is about 10 GB and comes from the loader (`io/labchart.py` expands every channel to 64 bit and builds a time array per channel). That is engine code and was not touched.

Two defects were found by looking at screenshots, not by tests: the sidebar clipped its widest controls behind the scroll bar in CT mode, and choosing CT left the CT sections below the fold. Both are fixed and pinned by tests. `scripts/gui_screenshots.py` renders the window offscreen for this kind of check.

Also this session: project memory moved into the project folder (`.claude/memory/`, loaded from `CLAUDE.md`), per the standing rule. `.claude/` is git ignored here, so it travels by rsync, not by clone.

Not verified: a person clicking through it on a real display, any real MRI recording, an export from real data compared with a known good result, Windows, and the PyInstaller freeze with the new GUI modules.

Before that (2026-08-19): deep review of the whole codebase and a fix batch, merged to main as `9755710..5d83081`.

## STATUS

| Work item | Stage |
|---|---|
| GUI redesign (sidebar, named pickers, click banner, CT method chooser, single Export, result panel, theme) | done on `feature/gui-pro-redesign`, 287 tests, independent review findings fixed; needs a manual pass on real data, then merge |
| Read native LabChart `.adicht` files without LabChart (`io/adicht.py`, `io/recording.py`, GUI open, `convert-adicht` command) | done on `feature/gui-pro-redesign`; equals LabChart's export bit for bit on ENT005 (user's manual export) and on 72 recording pairs; refuses what it was not checked on |
| PhaseContrast calibration (peak of cumulative integral) | done, on main |
| CT image counting unified on slices | done, on main |
| Landmark cut snapped to image start | done, on main |
| Temporal Res / image count decoupled; grid owns the export | done, on main |
| Reorder map in absolute recording time | done, on main |
| Auto-zoom to working window; numeric boxes commit on Enter | done, on main |
| CLI explicit `--trigger-row` and `--pressure-row` | done, on main |
| 2026-08-19 review fixes (drift sign, CSV precision, resampler, step threshold + declared polarity, CT rearrange, reader validation, GUI worker signals + state hygiene, bidirectional Start Shift) | done, on main |
| Loader memory: about 10 GB for a 1.85 GB export (`io/labchart.py` makes every channel 64 bit plus a time array each) | open; engine code, not touched by the GUI work |
| Export logic still inside the Execute worker closure in `gui/app.py` | open; the CLI does not write the trigger info file or the sanity plot, so GUI and CLI outputs differ |
| Xenon / phase-contrast calibration questions | deferred by user ("leave xenon, no worries for now") |
| MATLAB parity check on real data | not started, blocked on a known-good reference; drift fix intentionally diverges |
| Star-CCM+ acceptance of the new scientific-notation CSV | not confirmed; see NEXT STEP |
| `tests/data/` + `tests/reference/` fixtures | empty; CLAUDE.md describes them as parity fixtures |
| Windows packaging / CI | not started; see notes on PyInstaller freeze |

## SESSION LOG

- 2026-11-20: independent review of the redesigned GUI (a second agent, experiments run against `8eef0cf`, outputs kept in the session scratch folder). It proved eight defects and one clean result: all 11 standard export paths write files byte identical to `main`. Fixed test first in `tests/test_gui_review_fixes.py` (22 tests, 287 in total): a picker change no longer drops CT markers silently (same entry and Pressure change nothing, a real change asks first); the mouse wheel over a sidebar control scrolls the sidebar and no longer changes the control (`WheelGuard`); the line above Export says "whole window as recorded" when no landmark or cycle frame is marked; changing the CT method asks, then removes the marks of the method left, so only one method's marks are on the plot; Export on an empty flow block is blocked and explained (it ended the program), and `launch_gui` installs an exception hook that shows unexpected errors; a long file name is shortened in the middle (`ElidedLabel`) and cannot widen the sidebar; a new recording resets Start, End, image count and offsets; the overwrite question leaves out a pressure file that would not be written; an extreme Temporal Res no longer draws 300,000 grid lines.

- 2026-11-20: GUI redesign on `feature/gui-pro-redesign` in four tested units (layout, click modes and CT methods, export, polish). 143 to 265 tests. Memory migrated into the project folder. First run on a real recording led to drawing signal envelopes (21.5 s to 0.6 s) and a channel name check. Native `.adicht` reader added and checked bit for bit against LabChart exports; the GUI opens `.adicht` directly.
- 2026-08-19: full-codebase review + fix batch on `fix/review-findings` (15 commits, 141 tests). Fifteen defects in the fixes themselves were caught by five rounds of verification and fixed, the worst being three failed attempts to auto-detect trigger polarity (now a declared parameter). All four subsystems got an adversarial pass.
- 2026-08-18: CLI trigger-row bug found by user question, fixed on a feature branch with tests driving the real CLI on a 3-channel file (trigger ch1, decoy ch2, flow ch3). 97 tests. Created this file.
- 2026-08-17: Confirmed main merged and pushed (`9755710`), workspace clean.
- 2026-08-13: Reorder map absolute time (`9755710`); grid offset bidirectional (`03b7c82`); dt/image-count decoupled (`4ac6941`); keyboard tracking (`0a112ed`). Merged the 18-commit feature branch to main and pushed.
- Earlier: PhaseContrast calibration fix, CT slice model (stages 1 and 2), auto-zoom, workflow factory, dead `config/` removal, dependency floors, PHI ignores, LICENSE. Tests grew 32 to 97.

## NOTES

- Trigger channel: the GUI always used the exact row you set. Only the CLI guessed. Channel layouts are not standard across recordings, so nothing should infer a channel silently. The redesign keeps this: the pickers show names but never choose a channel for you.
- Two clocks in the exports, deliberately: `FlowProfile.csv` time starts at 0 (the synthetic breath); `ImageReorderMap.csv` `original_time_s` is absolute recording time (where the image sits in the scan).
- CT trigger channel carries only the step (scan on/off). There are no per-image triggers; the 200 ms comes from the reconstructed image series, so the image grid must be computed, not measured.
- Two grids are still drawn in the CT parts method, on purpose: grey numbers along the top are the parts grid and move with the grid offset, blue numbers along the bottom are the step's own intervals and stay fixed. The displacement between the rows is the offset.
- Unresolved and unreproduced: a report that the plot zoom resets to full range after Add Part. Not reproducible across seven scenarios; may have been cured incidentally by the keyboard-tracking change.
