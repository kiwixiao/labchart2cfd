# PROGRESS

## NEXT STEP
User reviews and merges branch `fix/review-findings` (9 commits: the full bug-fix batch from the 2026-08-19 deep review). After merge: the MATLAB parity comparison remains the one outstanding correctness check. NOTE: the drift-correction fix deliberately breaks parity with the MATLAB original (which carries the same abs() bug), so old outputs for positive-drift recordings will differ, correctly.

## LAST SESSION
2026-08-19. Deep review of the whole codebase (3 parallel reviewers, findings independently verified), then fixed everything except the xenon calibration (deferred by user). On branch `fix/review-findings`, stacked on the unmerged trigger-row branch. **14 commits, 130 tests (was 97).**

Engine fixes: drift correction doubled the drift for one polarity (abs() inherited from MATLAB, now signed, deliberately diverging from the original); CSV %.5f quantized mass flow to a ~61-step staircase (now %.9e, plus length/finite validation); resampler delivered 99.75 Hz while reporting 100 and lost samples to int() truncation; step detection swallowed the whole recording on DC-offset triggers and missed negative-going ones; CT rearrangement was silently skipped in the CLI's documented invocation; the .mat reader trusted the file's index tables (truncation clamped silently, a bad sentinel returned another channel's data); pressure was masked with the flow channel's time array and crashed on mixed-rate recordings.

Surface fixes: the GUI Execute worker's completion callbacks never fired (QTimer from a plain thread, now pyqtSignal), plus failed-load state hygiene, pre-Execute validation, exclusive click modes; CLI gained --pressure-row and guarded loads.

**Four rounds of self- and adversarial verification found six defects in my own fixes**, all fixed: a median polarity heuristic that inverted >50%-duty steps; a first-sample anchor that then inverted recordings starting mid-step; pressure trimming that silently shortened the exported flow; `validate --verbose` crashing on the corrupt-block dicts the reader newly emits; `pressure_note` reaching no operator; tab 3/tab 4 dt truncation. Note that 3 of 4 verifier agents died on network errors, so IO, GUI and math have had less independent scrutiny than the completeness pass.

Before that (2026-08-18): CLI --trigger-row fix on `fix/cli-explicit-trigger-row` (now part of the same line). Before that (2026-08-13, merged): CT synthetic-profile work, user-validated on real data.

## STATUS

| Work item | Stage |
|---|---|
| PhaseContrast calibration (peak of cumulative integral) | done, on main |
| CT image counting unified on slices | done, on main |
| Landmark cut snapped to image start | done, on main |
| Temporal Res / image count decoupled; grid owns the export | done, on main |
| Reorder map in absolute recording time | done, on main |
| Auto-zoom to working window; numeric boxes commit on Enter | done, on main |
| CLI explicit `--trigger-row` | done, in the `fix/review-findings` line, needs merge |
| 2026-08-19 review fixes (drift sign, CSV precision, resampler, step threshold, CT rearrange, reader validation, GUI worker signals + state hygiene, CLI --pressure-row) | done, branch `fix/review-findings`, needs merge |
| Xenon / phase-contrast calibration questions | deferred by user ("leave xenon, no worries for now") |
| MATLAB parity check on real data | not started, blocked on a known-good reference; drift fix intentionally diverges |
| `tests/data/` + `tests/reference/` fixtures | empty; CLAUDE.md describes them as parity fixtures |
| `gui/app.py` decomposition (~2400 lines) | deferred by choice; needs manual click-through testing |
| Windows packaging / CI | not started; see notes on PyInstaller freeze |

## SESSION LOG

- 2026-08-19: full-codebase review + fix batch on `fix/review-findings` (14 commits, 130 tests). Six defects in the fixes themselves were caught by verification and fixed. IO/GUI/math verifier agents died on network errors; those subsystems rest on my own checks plus the suite.
- 2026-08-18: CLI trigger-row bug found by user question, fixed on a feature branch with tests driving the real CLI on a 3-channel file (trigger ch1, decoy ch2, flow ch3). 97 tests. Created this file.
- 2026-08-17: Confirmed main merged and pushed (`9755710`), workspace clean.
- 2026-08-13: Reorder map absolute time (`9755710`); grid offset bidirectional (`03b7c82`); dt/image-count decoupled (`4ac6941`); keyboard tracking (`0a112ed`). Merged the 18-commit feature branch to main and pushed.
- Earlier: PhaseContrast calibration fix, CT slice model (stages 1 and 2), auto-zoom, workflow factory, dead `config/` removal, dependency floors, PHI ignores, LICENSE. Tests grew 32 to 97.

## NOTES

- Trigger channel: the GUI always used the exact row you set. Only the CLI guessed. Channel layouts are not standard across recordings, so nothing should infer a channel silently.
- Two clocks in the exports, deliberately: `FlowProfile.csv` time starts at 0 (the synthetic breath); `ImageReorderMap.csv` `original_time_s` is absolute recording time (where the image sits in the scan).
- CT trigger channel carries only the step (scan on/off). There are no per-image triggers; the 200 ms comes from the reconstructed image series, so the image grid must be computed, not measured.
- Unresolved and unreproduced: a report that the plot zoom resets to full range after Add Part. Not reproducible across seven scenarios; may have been cured incidentally by the keyboard-tracking change.
