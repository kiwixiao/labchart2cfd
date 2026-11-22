# CT composed breath: design

Date: 2026-11-22. Branch: `feature/ct-composed-breath`, taken from `feature/gui-pro-redesign` at `1bcd9bd`.
Decided with the user in conversation on 2026-11-22. Calls made by the designer and not by the user are marked **(designer's call)**.

Two answers from the user after the builder check (2026-11-22):
- **Grid offset stays in Reorder parts.**
- **Export all the details.** Downstream use comes later. The files must record what happened, so registration can align the image times with the breathing profile.

## 1. Why

The user builds a CFD breathing profile from a 4D CT scan. The scan gives N images, each lasting one Temporal Res (dt). The user knows N and dt from the image series. They reorder the images so the breath starts at inhale, and the flow profile is cut and joined in the same order.

A real case shows what is missing (ENT005, block 2, step 2, 15 images at 100 ms, Start about 970.37 s). The user labels the frames 4 to 15, then 1 to 4: frame 1 is image 4 (inhale onset), frame 12 is image 15, frame 13 is image 1, frame 16 is image 4 again. The 15 images cover 1.5 s, but the breath is longer. After image 15 ends (about 971.87 s) the exhale continues in the recording until about 972.4 s. Today frame 13 follows frame 12 directly, so every frame lasts dt and the profile is 16 x 100 ms = 1.6 s. The real exhale tail is lost.

The user asked for four things:

1. After a frame, keep the recorded flow until a time they choose, then continue with the next labeled frame. In the example: after frame 12, keep the flow until 972.4 s, then frame 13. The standard join (frames one dt apart) stays the default.
2. Export the time of every frame in the profile, so image registration and interpolation know the exact gap between frames.
3. Measure drift on one complete breath that they pick by hand, related to this scan. Over a complete breath the volume in equals the volume out, so the mean flow left over is sensor drift.
4. Before writing files, show the composed flow profile labeled with the new frame order and the original image numbers, so the image stack can be reordered correctly.

The user also agreed (2026-11-22) that the CT window is confusing when the step is not a whole number of images. On step 2 at 200 ms the window is 1.866 s = 9.33 images: 9 images end at 972.327 s and 66 ms are left before End. Reorder parts drops those 66 ms, Landmarks exports them (a 1.866 s profile that reports 9 images), and Cycle frames leaves them out of the file but includes them in the drift range (measured on ENT005 and read from the code, 2026-11-22).

## 2. Goals, non goals, anti goals

Goals:

- **G1** In CT the window always holds whole images: End = Start + count x Temporal Res.
- **G2** Extra flow after a frame, in Cycle frames and in Reorder parts.
- **G3** The time of every frame in the profile, in `{subject}ImageReorderMap.csv`, for both methods.
- **G4** A drift correction breath picked by hand, used by all three CT methods.
- **G5** A labeled preview of the composed profile before any file is written, saved with the export as a picture.
- **G6** The net volume of the exported breath is shown, so the user can judge the extra flow and the picked breath together.
- **G7** The exported breath has in = out: the half the user names as complete is kept as recorded, and the composed half is made to match its volume (decided with the user, 2026-11-22; section 4.7).

Non goals:

- MRI, CPAP and phase contrast are unchanged: drift rules, files and screens.
- No interpolation of images or geometry. The CFD and registration side does that, using the frame times.
- No automatic breath detection. The user picks the breath.
- Landmarks gets G1 and G4 only. It has no frame list to join, so G2, G3 and G5 do not apply to it.

Anti goals (a failure even if the feature works):

- The plot, the preview, the frame time file and the flow file disagree about which flow belongs to a frame, or where a frame sits in the profile.
- An export that does not use the new options changes silently. See section 6 for the exact identity checks.
- A mark the user placed (a picked breath, an extra flow time) is removed without a question.
- A choice that shapes the exported profile (shift, offset, drift breath, extra flow, frame order) is missing from the exported files.

## 3. Approaches

- **A. One assembly route for both methods (chosen by the designer; the user handed over the how).** Cycle frames labels split into runs of consecutive images, which are the same thing as Reorder parts. Both methods go through `assemble_parts` (`processing/profile_assembly.py`). Extra flow, frame times, the drift breath and the preview are built once. Cost: the Cycle frames export leaves its own route inside `_execute_processing` and its set of files changes (section 4.3).
- **B. Add each feature to each route separately.** Smaller first steps, but the extra flow rule, the frame times and the preview would each exist twice. Two copies of one rule is how this project got its worst bug (two image counts, fixed in `1bcd9bd`). Rejected.
- **C. A new third CT method for gaps.** Adds a method to learn and duplicates A. Rejected.

## 4. Design

### 4.1 Whole image window (G1)

In CT, End is always Start + count x Temporal Res.

- Select step: the count comes from the step as today (step length / dt, rounded half up by `compute_image_count`), then End is set to Start + count x dt. The step itself stays drawn on the plot, so the step edge remains visible for reference.
- A typed count: as today (`1bcd9bd`), End moves to fit it, and the count is kept.
- A typed End, or an End clicked for a window set by hand: the count is recomputed from it, and End moves to the nearest whole image. The last thing set wins, as today.
- Start or Temporal Res changes: the count is kept and End moves.
- With this rule a count can never be "the same number the box already shows but not in force", so typing it again is never needed.

Effect: the Start to End range is exactly the image grid, so `_ct_grid()` and the window agree. Landmarks exports exactly count x dt. Temporal Res here is the value as typed, never a rounded one (see 4.1a).

Grid offset (user's decision, 2026-11-22): it stays in Reorder parts. There the cut grid starts at Start + Grid offset, as today, and the existing amber note says when it differs from the scan window. Cycle frames and Landmarks always use the grid from Start (`_ct_grid()`); the Grid offset box sits on the parts page only and never affects them. Window offset in the parts page stays as it is. In Reorder parts, extra flow, frame times and the drift fallback range (4.4) follow the parts grid, offset included.

### 4.1a Cutting images when Temporal Res is not a whole number of samples (designer's call)

Temporal Res itself does not change (the user, 2026-11-22): it shows the standard value until the user types one, and any value typed is allowed and used as typed. This section is only about where the program places each cut on its internal samples.

The export resamples to 100 Hz. A Temporal Res such as 275 ms or 66.67 ms is not a whole number of samples. Today Reorder parts snaps it to 10 ms steps (`_synth_grid()`: 275 ms becomes 280 ms), so its cuts drift from the real images by 5 ms per image (75 ms by image 15). Cycle frames cuts by time masks on the typed value. The user's rule is that Temporal Res is used exactly as typed.

New rule, for both methods on the shared route: the boundary between image k and image k+1 is the sample nearest to Start + k x dt, with dt as typed. Each boundary is rounded on its own, so the error is at most half a sample (5 ms) and never accumulates. Frames may differ in length by one sample. The frame times in the map (4.3) are the times of the samples actually used, so the map and the flow file always name the same instants. This replaces the "dt must be whole samples" check in `assemble_parts`, whose purpose (the map naming times the chunks do not start at) the new rule meets by construction. With 100 ms and 200 ms nothing changes, because every boundary is already a whole sample.

### 4.2 Extra flow after a frame (G2)

Engine. `ProfilePart` gets an optional `extra_until_s` (absolute recording time). After the part's last image, `assemble_parts` appends the recorded flow (and pressure) from the end of that image to `extra_until_s`, then continues with the next part. The processed range must cover every extra flow, so the caller processes from the start of the grid to the later of the grid end and the latest `extra_until_s`.

Rules (checked in the engine and named in `_export_blockers()` so the line above Export and the export agree):

- `extra_until_s` must be later than the end of the part's last image, and inside the recording.
- Extra flow is refused after a part when the next part starts at the next image (it would repeat that image's own flow). It is allowed after the last part of the profile.
- The extra flow is the recorded flow, whatever images that time also belongs to. The preview shows it.
- One extra flow per part at most.

Cycle frames. The labels split into runs of consecutive images (the same rule the current export uses). A run is a part. Extra flow is attached to the frame that ends a run. In the example the runs are images 4 to 15 and images 1 to 4, and the extra flow is after frame 12. With no labels, Cycle frames exports all images in order as one part, which is what it exports today **(designer's call)**.

Screen. In both methods, the user selects the part (Reorder parts) or the frame that ends a run (Cycle frames) and sets "Extra flow until (s)" by typing or with Pick on plot, with a Clear button. The extra flow is drawn on the plot as a hatched band from the image end to the chosen time, labeled with the frame it follows. **(designer's call)** for the exact widgets and colors.

### 4.3 Frame times in the export (G3)

The user's rule (2026-11-22): export all the details, so it is clear what happened and registration can align the image times with the breathing profile. No export drops a file it writes today.

**Frame table.** `{subject}ImageReorderMap.csv` gets four columns at the end (example rows from the ENT005 case, times rounded from a screenshot):

```
new_frame,original_image,original_time_s,profile_start_s,profile_end_s,extra_flow_s,extra_flow_until_s
1,4,970.670000,0.000000,0.100000,0.000000,
12,15,971.770000,1.100000,1.200000,0.530000,972.400000
13,1,970.370000,1.730000,1.830000,0.000000,
```

- `original_time_s`: where the image starts in the recording (unchanged meaning and format).
- `profile_start_s` and `profile_end_s`: where the frame's flow sits in the exported profile, whose time starts at 0.
- `extra_flow_s`: the length of recorded flow kept after this frame, before the next frame (0 when none). `extra_flow_until_s`: where that extra flow ends in the recording (empty when none). The gap is therefore stated directly and can also be checked from the profile times.
- The first three columns keep their position and format, so a script reading them still works.
- Every time in the table is the time of a sample actually used in the flow file (4.1a).

**Details file.** `{subject}_trigger_info.txt` is written by every CT export, including Reorder parts, which does not write it today. It keeps every key it has today and adds, in the same `key: value` form **(designer's call for the key names)**:

- `ct_method` (reorder parts, cycle frames, landmarks)
- `temporal_resolution_s` as typed, `image_count`, `grid_start_time` (where image 1 starts), `grid_offset_s` (Reorder parts), `start_shift_ms`
- `step_start_time`, `step_end_time` (the detected step, if one was selected), `window_start_time`, `window_end_time`
- `drift_source` (your breath, or the image grid), `drift_range_start`, `drift_range_end`, `drift_removed_L_per_s`
- one `extra_flow` line per extra flow: the frame and image it follows, its recording start and end, its length
- `frame_order_new_to_original`, `profile_duration_s`, `profile_net_volume_mL`
- the frame table file name, so the two files point at each other

**Files per method after this change:**

- Reorder parts: FlowProfile, PressureProfile (when a pressure channel has data), ImageReorderMap, and new: the details file and the preview picture `{subject}_frame_order.png`.
- Cycle frames: everything it writes today (FlowProfile, PressureProfile, `{subject}_trigger_info.txt`, `{subject}_sanity_check.png`), and new: ImageReorderMap and `{subject}_frame_order.png`.
- Landmarks: the files it writes today; the details file gains the new keys that apply (drift and window).

`_export_targets()` must list exactly these files, so the overwrite question names what is written. The result panel lists them.

### 4.4 Drift correction breath (G4)

Screen. Section 3 Time window gets a "Drift correction breath" pair, From (s) and To (s), with Pick on plot and Clear, shown only for CT. Pick on plot takes two clicks, through the click mode system (`_click_mode()` and `_BANNERS`). The breath is drawn on both plots as a band in its own color, labeled "drift breath". It may reach outside Start and End and outside the step.

Engine. `CTWorkflow.process` gets an optional `drift_range=(t0, t1)`. When given, the drift error is `calculate_drift_error` over the raw recorded flow between t0 and t1 (the same raw signal and the same formula as today, only the range differs), and it is subtracted from the processed flow. When not given, the drift is measured over the processed range, as today. Pressure is not drift corrected, as today.

Rules:

- From < To, both inside the recording, at least 2 samples between them.
- No breath picked: the export runs with today's range, and the line above Export shows an amber caution naming the range used. It does not block (decided by the user, 2026-11-22, the same way Temporal Res is cautioned).
- Today's range means the image grid the method cuts, never including extra flow **(designer's call)**: Start to Start + count x dt for Cycle frames and Landmarks, and the parts grid (with its Grid offset) for Reorder parts. Setting an extra flow then never moves the level of the whole profile; only a picked breath does.
- The breath is a mark the user placed: it is cleared only through `_ask_before_dropping()` (new recording, other block, other step, other study type), or by its Clear button.

### 4.5 Labeled preview before writing (G5)

Both Cycle frames and Reorder parts show the preview before any file is written (Reorder parts already has `AssemblyPreviewDialog` with Export and Cancel; it is extended, and Cycle frames gains it).

```
 frame   1    2    3  ...  12 |               | 13   14   15   16
 image   4    5    6  ...  15 |  extra flow   |  1    2    3    4
        ~~~~~~~ flow curve ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        0.0                1.2             1.73               2.13   profile time (s)
 Drift: your breath, 970.72 to 973.05 s, 0.012 L/s removed.
 Complete half: (o) inhale  ( ) exhale         [Fit extra flow]
 In 412 mL   Out 395 mL before scaling   exhale x 1.043   Net 0.0 mL
                                                        [Export] [Cancel]
```

(Numbers in the sketch are placeholders, not measurements.)

- The curve is the exact array that will be written.
- Thin lines at every frame boundary. Above each frame: its new number. Below it: its original image number.
- Extra flow is shaded and labeled with its length and "no image".
- Labels thin out when frames are too narrow to print, so they never print over each other.
- Below the plot: where the drift was measured and how much was removed, and the net volume of the breath.
- The same figure is saved as `{subject}_frame_order.png` with the export.

### 4.6 Net volume (G6)

The net volume of the exported profile in mL: the time integral of the exported flow converted back to L/s. Shown in the preview and in the result panel, next to the drift source and the removed drift.

### 4.7 In equals out (G7)

Why: the CFD repeats the profile breath after breath, so any net volume left accumulates every cycle. A composed breath has one complete half and one composed half. In the ENT005 case the inhale is complete (inside images 4 to 15); the exhale is composed (the end of image 15, the extra flow, images 1 to 3). The user accepts the limit this sets: the composed half is matched to the complete half, not measured.

Order of operations, on the composed profile:

1. Drift is removed with the picked breath (4.4), or the fallback range. This sets where zero flow is, so it decides what counts as in and what counts as out.
2. The user names the complete half in the preview: inhale (default) or exhale.
3. **Fit extra flow** (a button in the preview, optional): moves the end of the extra flow in the composed half until its volume is closest to the complete half's volume, using recorded flow only. The user can accept the suggested time or keep their own. Available only when the composed half contains extra flow.
4. **Scale** (always): the samples of the composed half's sign are multiplied by one factor k so that the net volume is exactly zero. Write the profile as q = p + n, where p holds the inhale direction samples and n the exhale direction samples (each with the other direction set to 0; directions as in the recorded flow, before the export's sign inversion). With the volume taken as the trapezoid integral and the exhale as the composed half, net(k) = trapz(p) + k trapz(n), which is linear in k, so k = -trapz(p) / trapz(n) gives zero exactly. With the inhale as the composed half the roles swap. The trapezoid integral is the volume a solver sees when it interpolates the table linearly between samples. The sign of every sample is unchanged, so each zero crossing stays between the same two samples.

How it is made sure:

- By construction: k is computed from the same array that is written, after every other step (resampling, smoothing, assembly, unit conversion are all done before it, and unit conversion is linear).
- On the written file: after writing, the export reads `{subject}FlowProfile.csv` back, integrates it, and shows in, out and net in the result panel. A net volume above a rounding tolerance (0.01 mL) is shown in red **(designer's call for the tolerance)**. This guards against any later change to the array or the writer.
- Shown: the preview and the details file show the complete half, in, out before scaling, k, and the fitted or chosen extra flow end. A k outside 0.9 to 1.1 gets an amber caution, not a block **(designer's call)**: it means the composed half differed by more than 10 percent before scaling, so the extra flow or the picked breath is probably off.
- Tests: engine tests on synthetic flows with known volumes (net zero after scaling, complete half unchanged sample by sample, zero crossings between the same samples); the ENT005 case by hand.

Details file keys **(designer's call for the names)**: `complete_half`, `volume_in_mL_before_scaling`, `volume_out_mL_before_scaling`, `composed_half_scale_factor`, `extra_flow_fitted` (yes or no), and `volume_in_mL`, `volume_out_mL`, `net_volume_mL` as read back from the written file.

## 5. Constraints

- The project rules in `CLAUDE.md` hold: `_ct_grid()` is the one grid; `_export_blockers()` is both the line above Export and the gate; nothing the user placed is removed without a question; new sidebar controls get `guard_wheel`; GUI tests use `no_modal_dialogs` (a modal dialog hangs an offscreen run, so the preview must be testable without blocking).
- `assemble_parts` slices on whole samples, with each image boundary rounded on its own (4.1a). Extra flow ends are also rounded to whole samples. The frame times in the map name the samples actually used.
- Real recordings never go into the repository or a fixture.
- Writing style of on screen text: plain, short, complete sentences, no dashes.

## 6. Testing

Test first for every unit (red, green, refactor), small units, each committed with PROGRESS.md.

- Engine: `assemble_parts` with extra flow (lengths, frame times, refused cases); `CTWorkflow.process` with `drift_range` (known offset removed exactly; no range gives today's numbers).
- GUI: whole image window; extra flow set, drawn, blocked, cleared with a question; drift breath picked, drawn, cautioned when missing; preview shows the labels and is the written array; files listed by `_export_targets()` are the files written.
- Identity checks (anti goal 2), with no extra flow and no breath picked:
  - Reorder parts: FlowProfile and PressureProfile byte identical to `1bcd9bd`; ImageReorderMap identical in its first three columns.
  - Cycle frames: FlowProfile and PressureProfile compared with the old route on a whole image window at 100 ms and 200 ms. Byte identity is expected; any difference is explained sample by sample and reported before merge.
  - MRI, CPAP, phase contrast and Landmarks: byte identical to `1bcd9bd` when the window is already whole images.
- `scripts/gui_screenshots.py` and `scripts/gui_native_check.py` after visual changes.
- ENT005 by hand with the user's labels (4 to 15, 1 to 4, extra flow after frame 12 until 972.4 s): check the frame times, the preview and the net volume.

## 7. Order of work

1. Whole image window (G1).
2. Engine: drift range (G4 engine).
3. Engine: extra flow and frame times in `assemble_parts` (G2, G3 engine).
4. Cycle frames on the assembly route, with the identity check.
5. Drift correction breath on screen (G4).
6. Extra flow on screen (G2).
7. Labeled preview, saved picture, net volume (G5, G6).
8. In equals out: scaling in the engine, then the complete half choice, Fit extra flow and the read back check (G7).

## 8. Left to the builder

Widget placement inside the sections, colors and hatching, the label thinning rule, internal names, and the exact wording of hints and cautions (in the project's plain style).
