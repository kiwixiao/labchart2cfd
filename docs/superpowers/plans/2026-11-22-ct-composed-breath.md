# CT composed breath: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a CT export keep recorded flow after a frame, export the time of every frame, measure drift on a breath picked by hand, force in = out by scaling the composed half, and show a labeled preview before writing.

**Architecture:** Engine first (`processing/`, `workflows/ct.py`, `io/`), each piece tested on synthetic arrays. Then Cycle frames moves onto the Reorder parts route (`_assemble_and_export`), so every new feature exists once. GUI tasks wire the engine to section 3, the parts and cycle pages, and the preview dialog.

**Tech Stack:** Python 3.10, NumPy, SciPy, PyQt5, matplotlib; pytest with Qt offscreen (conda env `labchart`).

**Spec:** `docs/superpowers/specs/2026-11-22-ct-composed-breath-design.md` (approved by the user 2026-11-22).

## Global Constraints

- Branch `feature/ct-composed-breath`. Every task ends with a commit that includes a PROGRESS.md update (NEXT STEP, STATUS row, SESSION LOG line). No attribution line in commit messages.
- Run commands in the conda env: `conda run -n labchart python -m pytest ...`. Long runs in the background.
- Temporal Res is used exactly as typed. Each image boundary is the sample nearest to grid start + k x dt, rounded half up, on its own (spec 4.1a).
- MRI, CPAP and phase contrast are untouched.
- `_export_blockers()` is both the line above Export and the gate. Nothing the user placed is removed without `_ask_before_dropping()`. New sidebar controls get `guard_wheel`. GUI tests use `no_modal_dialogs`.
- On screen text: plain, short, complete sentences, no dashes, no hyphenated words.
- Real recordings never go into the repository or a fixture.

## Review Focus

1. Temporal Res that is not a whole number of 100 Hz samples (275 ms): export must succeed and the frame table must name the samples used. Test in Task 3 and Task 5.
2. Extra flow that runs past the end of the recording: named before Export, never a crash. Test in Task 7.
3. A composed half with no volume (all samples of one sign): the export says so instead of dividing by zero. Test in Task 4 and Task 8.
4. Changing step, block or study type with a drift breath or extra flow placed: asks first. Test in Task 6 and Task 7.
5. No breath picked and extra flow set: drift is measured over the images only, so the level does not move when extra flow is added. Test in Task 5.

---

### Task 1: Whole image window (spec 4.1, G1)

**Files:** Modify `src/labchart2cfd/gui/app.py` (`__init__` state, `_select_ct_step`, `_draw_ct_intervals`, `_keep_user_image_count`, `_on_ct_window_edge_changed`, `_apply_window_offset`, hand window clicks in `_on_click`). Test: create `tests/test_gui_composed_breath.py`.

**Interfaces:**
- Produces: `self._ct_raw_span_s: Optional[float]`, the window length the step or the user gave before rounding to whole images. `_ct_auto_count() -> int` = `compute_image_count(self._ct_raw_span_s, dt)`. `_fit_end_to_count()` replaces `_keep_user_image_count()`: in CT with a grid, End = Start + count x dt, always (not only for a typed count).

Rules: Select step sets the span to the step length. A typed or clicked End sets the span to End minus Start and clears `_ct_images_user_set`. Start changes and Window offset keep the span. Temporal Res changes recount from the span (unless the count is typed). A typed count wins as today.

- [ ] **Step 1: Failing tests** in `tests/test_gui_composed_breath.py` (mat_file step is 2 s to 5 s):
  - `test_a_selected_step_holds_whole_images`: CT, cycle method, Select step, Temporal Res 400 -> count 8, End 5.2.
  - `test_temporal_res_recounts_from_the_step_not_from_the_rounded_window`: then Temporal Res 300 -> count 10, End 5.0 (from the 3.2 s rounded window it would be 11).
  - `test_a_typed_end_rounds_to_whole_images`: at 200 ms type End 4.1 -> count 11, End 4.2.
  - `test_a_typed_start_moves_the_whole_window`: Start 2.5 -> count 15, End 5.5.
  - `test_a_hand_window_holds_whole_images`: trigger None, clicks at 2.0 and 4.1 -> End 4.2.
- [ ] **Step 2:** Run `conda run -n labchart python -m pytest tests/test_gui_composed_breath.py -q`. Expected: 5 failures.
- [ ] **Step 3:** Implement the rules above.
- [ ] **Step 4:** Run the new tests (pass), then the full suite in the background. Tests that pinned a fractional window: update each one to the spec's rule and name it in the commit message.
- [ ] **Step 5:** Commit `feat(gui): the CT window always holds whole images`.

### Task 2: Drift range in the CT engine (spec 4.4 engine)

**Files:** Modify `src/labchart2cfd/workflows/ct.py` (`CTWorkflow.process`). Test: `tests/test_workflows.py`.

**Interfaces:**
- Produces: `CTWorkflow.process(..., drift_range: Optional[Tuple[float, float]] = None)`. When given, the drift error is `calculate_drift_error` over the raw flow that `_extract_time_window(full_time, full_flow, t0, t1)` returns, and it is subtracted from the processed (and, with landmarks, rearranged) flow. `result.metadata["drift_range"]` is `(t0, t1)` or None. Raises ValueError when t0 >= t1 or the range is outside the recording (the existing named error of `_extract_time_window`).

- [ ] **Step 1: Failing tests:**
  - `test_ct_drift_range_removes_the_offset_measured_on_that_range`: `_single_channel_data(duration_s=10)` plus 0.05; process window 1.0 to 1.5 s with `drift_range=(2.0, 6.0)` (two whole 0.5 Hz cycles) -> `result.drift_error == pytest.approx(0.05, abs=1e-4)`.
  - `test_ct_without_drift_range_measures_the_window`: the same call without the argument gives `drift_error == calculate_drift_error(t_window, f_window)` exactly, on the window `_extract_time_window` returns (today's behaviour).
  - `test_ct_drift_range_equal_to_the_window_changes_nothing`: `drift_range=(start, end)` gives a `mass_flow` array equal (`np.array_equal`) to the call without it.
  - `test_ct_drift_range_outside_the_recording_raises`.
- [ ] **Step 2:** Run, expect failures. **Step 3:** Implement. **Step 4:** Run `tests/test_workflows.py` and `tests/test_processing.py`, pass. **Step 5:** Commit `feat(ct): measure drift on a chosen range`.

### Task 3: Fractional Temporal Res, extra flow and frame times in assembly (spec 4.1a, 4.2, 4.3)

**Files:** Modify `src/labchart2cfd/processing/profile_assembly.py`, `src/labchart2cfd/io/csv_export.py`. Tests: `tests/test_profile_assembly.py`, `tests/test_io.py`.

**Interfaces:**
- Produces: `ProfilePart.extra_until_s: Optional[float] = None` (absolute recording time).
- `@dataclass FrameRow: new_frame: int, original_image: int, original_time_s: float, profile_start_s: float, profile_end_s: float, extra_flow_s: float, extra_flow_until_s: Optional[float]`.
- `AssemblyResult.frames: List[FrameRow]`; `reorder_map` stays, built from the first three fields.
- `AssemblyResult.extra_spans: List[Tuple[float, float]]`, profile time ranges holding extra flow (for the preview).
- Boundary rule: line k is sample `off + int(np.floor(k * dt * sample_rate + 0.5 + 1e-9))`, `off` from `grid_offset_s` as today. Every slice must hold at least one sample. The whole sample dt check is removed.
- Extra flow after a part ends at sample `int(np.floor((extra_until_s - window_start_s) * sample_rate + 0.5 + 1e-9))`; it must be after the part's last boundary and inside `flow`; it is refused when the next part starts at the next image.
- `export_frame_table_csv(filepath, frames: List[FrameRow]) -> Path`, header `new_frame,original_image,original_time_s,profile_start_s,profile_end_s,extra_flow_s,extra_flow_until_s`, times `%.6f`, empty last field when None, CRLF like the existing writer.

- [ ] **Step 1: Failing tests:**
  - `test_fractional_dt_rounds_each_boundary_on_its_own`: dt 0.275, total 4, flow ramp 0..200 -> chunk boundaries at samples 0, 28, 55, 83, 110 (assert via `frames[i].profile_start_s * 100`).
  - `test_extra_flow_follows_its_part`: ramp 0..300, dt 0.2, parts (4..6, extra_until_s 1.5), (1..3), window_start 0 -> flow is samples 60..119, 120..149, 0..59; frame 3 has `profile_end_s` 0.6, `extra_flow_s` 0.3, `extra_flow_until_s` 1.5; frame 4 has `profile_start_s` 0.9; `extra_spans == [(0.6, 0.9)]`.
  - `test_extra_flow_before_the_next_image_is_refused`: parts (1..3, extra 0.8), (4..6) -> ValueError containing "next image".
  - `test_extra_flow_must_end_after_its_image_and_inside_the_flow`: two ValueError cases.
  - `test_frame_times_name_the_samples_used`: dt 0.275 -> every `original_time_s == window_start + boundary / 100`.
  - `test_export_frame_table_csv_writes_seven_columns` in `tests/test_io.py`.
- [ ] **Step 2:** Run, expect failures. **Step 3:** Implement. **Step 4:** Run `tests/test_profile_assembly.py tests/test_io.py`, all pass, the existing assembly tests unchanged. **Step 5:** Commit `feat(assembly): extra flow, frame times, boundaries for any Temporal Res`.

### Task 4: In equals out (spec 4.7 engine)

**Files:** Create `src/labchart2cfd/processing/volume_balance.py`. Test: create `tests/test_volume_balance.py`.

**Interfaces:**
- Produces: `@dataclass BalanceResult: flow: NDArray, factor: float, kept_volume: float, scaled_volume_before: float` (volumes as trapezoid integrals in flow units x s, signed).
- `balance_halves(flow, sample_rate, keep: str) -> BalanceResult`, `keep` is "positive" or "negative". The samples of the other sign are multiplied by `factor = -trapz(kept) / trapz(other)`. Raises ValueError("the half to scale holds no volume") when trapz(other) is 0.
- `mass_volume_to_ml(volume_kg: float, density: float = 1.2) -> float` = volume_kg / (density * 1e-3) * 1000.

- [ ] **Step 1: Failing tests:** `test_net_volume_is_zero_after_balancing` (trapz of result below 1e-12 relative); `test_kept_half_is_unchanged_sample_by_sample`; `test_no_sample_changes_sign`; `test_factor_is_the_volume_ratio`; `test_a_half_with_no_volume_raises`.
- [ ] **Step 2 to 4:** Run, implement, run. **Step 5:** Commit `feat(processing): balance the composed half so in equals out`.

### Task 5: Cycle frames on the assembly route, details file (spec approach A, 4.3)

**Files:** Modify `src/labchart2cfd/gui/app.py` (`_uses_assembly`, `_assemble_and_export`, `_synth_grid`, `_export_targets`), create `src/labchart2cfd/io/ct_details.py`. Tests: `tests/test_gui_composed_breath.py`, golden files in `tests/reference/`.

**Interfaces:**
- Produces: `_uses_assembly()` true for CT with method parts or cycle. `_assembly_grid() -> Optional[Tuple[float, float, int]]`: cycle `_ct_grid()`, parts `_synth_grid()` (dt as typed now, no rounding to 10 ms). `_export_parts() -> List[ProfilePart]`: parts `self._parts`; cycle splits `self._ct_cycle_frames` into runs of consecutive images, all images in order when there are no labels.
- `write_ct_details(filepath, entries: List[Tuple[str, str]]) -> Path` writes `key: value` lines.
- `_ct_details_entries(...) -> List[Tuple[str, str]]`: every key the current trigger info file has (cycle keys included) plus the spec 4.3 keys that exist by this task (method, Temporal Res as typed, image count, grid start, grid offset, start shift, step, window, frame order, profile duration, frame table file name).
- The processed range is from the grid start to the latest of the grid end and every extra flow end (plus one sample when the last boundary rounds up). The drift range passed to `process` is always the image grid when no breath is picked (Task 6 adds the breath), so extra flow never moves the level.
- Files: parts writes Flow, Pressure, ImageReorderMap (seven columns), trigger info. Cycle writes Flow, Pressure, ImageReorderMap, trigger info, sanity check plot (same drawing as today, moved into a helper used by both routes).

- [ ] **Step 1: Golden files first, from the current code:** a test helper writes the cycle export (labels images 4 to 15 then 1 to 4 on mat_file step 2 to 5 s) at 200 ms and 100 ms, and the parts export, and the step saves their Flow and Pressure CSVs as `tests/reference/composed_*` (synthetic data only). Commit them before changing the route.
- [ ] **Step 2: Failing tests:** `test_cycle_export_flow_is_byte_identical_to_the_old_route` (both rates); `test_parts_export_flow_is_byte_identical`; `test_cycle_export_writes_the_frame_table_and_keeps_its_old_files`; `test_parts_export_writes_the_details_file`; `test_parts_export_at_275_ms_succeeds_and_the_table_names_the_samples_used`; `test_extra_flow_does_not_move_the_drift_level_without_a_breath` (engine level: same `drift_error` with and without an extra flow part); `test_export_targets_lists_the_files_written` for both methods.
- [ ] **Step 3:** Implement. Remove `test_dt_is_quantized_to_whole_flow_samples` (it pinned the old rule; its regression is covered by the 275 ms test) and say so in the commit.
- [ ] **Step 4:** Full suite in the background, pass. **Step 5:** Commit `feat(gui): cycle frames export through the assembly route`.

### Task 6: Drift correction breath on screen (spec 4.4)

**Files:** Modify `src/labchart2cfd/gui/app.py` (section 3 build, `_BANNERS`, `_click_mode`, `_on_click`, `_marks_at_risk`, `_export_cautions`, both export routes, details entries). Test: `tests/test_gui_composed_breath.py`.

**Interfaces:**
- Produces: `self._drift_breath: Optional[Tuple[float, float]]`; widgets `drift_from_spin`, `drift_to_spin`, `drift_pick_btn`, `drift_clear_btn` (visible for CT only); click modes `"drift_first"`, `"drift_second"` with banners; band drawn on both plots, kept in `self._drift_artists`. Details keys `drift_source`, `drift_range_start`, `drift_range_end`, `drift_removed_L_per_s`.

- [ ] **Step 1: Failing tests:** `test_pick_on_plot_sets_the_drift_breath_with_two_clicks`; `test_typed_from_and_to_set_the_drift_breath`; `test_the_drift_breath_is_drawn_on_both_plots`; `test_no_breath_gives_an_amber_caution_naming_the_range`; `test_a_drift_breath_reaches_the_engine` (monkeypatch `CTWorkflow.process` to capture `drift_range`, for landmarks and cycle); `test_selecting_another_step_asks_before_dropping_the_breath`; `test_drift_controls_are_hidden_outside_ct`.
- [ ] **Step 2 to 4:** Run, implement, run full suite. **Step 5:** Commit `feat(gui): pick the drift correction breath`.

### Task 7: Extra flow on screen (spec 4.2)

**Files:** Modify `src/labchart2cfd/gui/app.py` (parts page, cycle page, click mode `"extra_until"`, drawing, `_export_blockers`, `_marks_at_risk`). Test: `tests/test_gui_composed_breath.py`.

**Interfaces:**
- Produces: parts: the selected part's `extra_until_s`, set by `extra_until_spin` with `extra_pick_btn` and `extra_clear_btn` on the parts page. Cycle: `self._ct_cycle_extra: Dict[int, float]` keyed by frame number; `extra_frame_spin` chooses the frame. `_extra_flow_problems() -> List[str]` mirrors the engine rules (after its image, inside the recording, not before the next image) and feeds `_export_blockers()`. Hatched band from the image end to the chosen time, labeled "extra after frame N".

- [ ] **Step 1: Failing tests:** `test_extra_flow_after_frame_12_is_set_by_typing` (cycle labels 4..15, 1..4 on a 100 ms grid); `test_extra_flow_is_picked_on_the_plot`; `test_extra_flow_before_the_next_image_is_named_above_export`; `test_extra_flow_past_the_recording_is_named_above_export`; `test_extra_flow_reaches_the_export` (frame table row has `extra_flow_until_s`); `test_clearing_labels_asks_before_dropping_extra_flow`.
- [ ] **Step 2 to 4:** Run, implement, run full suite. **Step 5:** Commit `feat(gui): extra flow after a frame`.

### Task 8: Labeled preview, saved picture, in equals out, read back (spec 4.5, 4.6, 4.7)

**Files:** Create `src/labchart2cfd/gui/frame_order_plot.py`; modify `src/labchart2cfd/gui/dialogs.py` (`AssemblyPreviewDialog`), `src/labchart2cfd/gui/app.py` (`_assemble_and_export`, result report). Test: `tests/test_gui_composed_breath.py`, `tests/test_gui_export.py` (existing preview tests updated to the new signature).

**Interfaces:**
- Produces: `draw_frame_order(fig, time, flow, frames, extra_spans, notes: List[str]) -> None`: frame boundary lines, new frame number above each frame, original image number below, shaded extra flow labeled with its length and "no image", labels skipped when a frame is narrower than its text.
- `AssemblyPreviewDialog(asm, balances: Dict[str, BalanceResult], default_keep: str, notes, parent=None)`; radio buttons "above zero" and "below zero" choose the complete half (the default is the half the extra flow is not in, by the sign of the mean flow over the extra spans; "above zero" when there is no extra flow) **(designer's call: the halves are named by their side of zero, because which sign is inhale differs between setups)**; `chosen_keep` attribute; the curve redraws for the choice. Amber note when the factor is outside 0.9 to 1.1.
- After Export: the flow written is the chosen `BalanceResult.flow`; the figure is saved as `{subject}_frame_order.png`; `{subject}FlowProfile.csv` is read back and in, out and net (mL, `mass_volume_to_ml`) go into the result summary; net above 0.01 mL is a red warning. Details keys from spec 4.7.

- [ ] **Step 1: Failing tests:** `test_preview_labels_new_and_original_numbers` (texts in the figure); `test_preview_shades_extra_flow`; `test_the_written_flow_has_zero_net_volume` (read the CSV back, trapz below 0.01 mL); `test_the_complete_half_is_written_unchanged`; `test_the_frame_order_picture_is_saved`; `test_a_factor_far_from_one_is_cautioned`; `test_a_half_with_no_volume_is_named_not_crashed`.
- [ ] **Step 2 to 4:** Run, implement, run full suite, then `scripts/gui_screenshots.py` and `scripts/gui_native_check.py` and look at the pictures. **Step 5:** Commit `feat(gui): labeled preview with in equals out`.

### Task 9: Real recording check and hand over

- [ ] Run the user's ENT005 case offscreen from a scratchpad script (block 2, step 2, 100 ms, 15 images, labels 4 to 15 then 1 to 4, extra flow after frame 12 until 972.4 s): report the frame table rows 12 and 13, the factor, in, out and net read back. Nothing from the recording is saved in the repository.
- [ ] Update CLAUDE.md (one paragraph: the shared assembly route, the boundary rule, in equals out) and PROGRESS.md, commit, relaunch the app for the user.
