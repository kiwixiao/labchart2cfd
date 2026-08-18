# PROGRESS

## NEXT STEP
Compare one exported `{Subject}FlowProfile.csv` against a known-good MATLAB-era result for the same subject; this is the only outstanding correctness check and it needs a subject with an old result on hand.

## LAST SESSION
2026-08-18. Found and fixed a CLI bug: `process` and `detect-steps` derived the trigger channel as `flow_row - 1` instead of accepting one, so with non-adjacent channel layouts they silently read the wrong channel. Both now take `--trigger-row` and use it verbatim; when omitted the fallback guess is printed rather than applied silently. On branch `fix/cli-explicit-trigger-row`, awaiting your merge.

Before that (2026-08-13, merged to main): the CT synthetic-profile work. Temporal Res and image count decoupled, the grid defines the exported range, reorder map records absolute recording time, grid offset made bidirectional. User tested the GUI end to end on real data and confirmed it works.

## STATUS

| Work item | Stage |
|---|---|
| PhaseContrast calibration (peak of cumulative integral) | done, on main |
| CT image counting unified on slices | done, on main |
| Landmark cut snapped to image start | done, on main |
| Temporal Res / image count decoupled; grid owns the export | done, on main |
| Reorder map in absolute recording time | done, on main |
| Auto-zoom to working window; numeric boxes commit on Enter | done, on main |
| CLI explicit `--trigger-row` | done, branch `fix/cli-explicit-trigger-row`, needs merge |
| MATLAB parity check on real data | not started, blocked on a known-good reference |
| `tests/data/` + `tests/reference/` fixtures | empty; CLAUDE.md describes them as parity fixtures |
| `gui/app.py` decomposition (~2400 lines) | deferred by choice; needs manual click-through testing |
| Windows packaging / CI | not started; see notes on PyInstaller freeze |

## SESSION LOG

- 2026-08-18: CLI trigger-row bug found by user question, fixed on a feature branch with tests driving the real CLI on a 3-channel file (trigger ch1, decoy ch2, flow ch3). 97 tests. Created this file.
- 2026-08-17: Confirmed main merged and pushed (`9755710`), workspace clean.
- 2026-08-13: Reorder map absolute time (`9755710`); grid offset bidirectional (`03b7c82`); dt/image-count decoupled (`4ac6941`); keyboard tracking (`0a112ed`). Merged the 18-commit feature branch to main and pushed.
- Earlier: PhaseContrast calibration fix, CT slice model (stages 1 and 2), auto-zoom, workflow factory, dead `config/` removal, dependency floors, PHI ignores, LICENSE. Tests grew 32 to 97.

## NOTES

- Trigger channel: the GUI always used the exact row you set. Only the CLI guessed. Channel layouts are not standard across recordings, so nothing should infer a channel silently.
- Two clocks in the exports, deliberately: `FlowProfile.csv` time starts at 0 (the synthetic breath); `ImageReorderMap.csv` `original_time_s` is absolute recording time (where the image sits in the scan).
- CT trigger channel carries only the step (scan on/off). There are no per-image triggers; the 200 ms comes from the reconstructed image series, so the image grid must be computed, not measured.
- Unresolved and unreproduced: a report that the plot zoom resets to full range after Add Part. Not reproducible across seven scenarios; may have been cured incidentally by the keyboard-tracking change.
