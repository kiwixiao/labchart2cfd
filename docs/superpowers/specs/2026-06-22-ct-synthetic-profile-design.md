# CT Synthetic Profile Assembly — Design Spec

- **Date:** 2026-06-22
- **Status:** Approved design (pre-implementation)
- **Workflow:** CT (4D imaging)
- **Phase:** Phase 1 deliverable — primary first use case

## 1. Problem

In 4D CT of the airway, the scanner acquires images over time but **cannot control which breathing phase** the patient is in when each image is captured. In natural acquisition order the images are therefore *not* a clean inhale→exhale cycle — they are scattered across phases. Flow is recorded simultaneously, so every image timepoint has a known flow value (and thus a known breathing phase).

The clinician needs to **manually cut the recorded flow into ordered pieces and stitch them into one synthetic breathing cycle**, and to obtain the corresponding **image reorder map** so the CT image stack can be re-aligned to that synthetic cycle.

A real example file (`CFD ENT 003`) confirmed two facts that shape this design:
- The `TRIGGER` channel is **flat** (no usable step triggers) → auto step-detection cannot drive this; a **full manual mode** is required.
- The core CT processing pipeline already produces correct output on real flow (verified: 12 s window → 1200 samples @ 100 Hz, image map 61 fenceposts, well-formed CSV on macOS).

## 2. Goal & deliverables

Given a manually defined window and an ordered list of labeled parts, produce:

- 🅐 **Synthetic flow profile** — the CFD boundary condition (continuous, 100 Hz).
- 🅑 **Image reorder map** — one row per image, giving the original image number in the new assembled order, so the image stack can be re-aligned.

(Plus an assembled **pressure** profile when a pressure channel is selected.)

## 3. Requirements (settled during brainstorming)

1. **Full manual mode**: the clinician sets the total profile window `[start, end]` by hand (auto-detection optional / not required). The trigger channel becomes optional for the CT workflow.
2. **Image grid**: within the window, images are at a fixed temporal resolution (default 200 ms), numbered `1…N` in **ascending time order** (the natural acquisition order). `N = round(window ÷ dt) + 1` (fencepost). Window and N stay mutually consistent.
3. **Parts model**: the clinician builds an **ordered list of parts**. Each part is a **contiguous range of image markers** `(start_image, end_image)` with an optional descriptive `label`. A part may span across the inhale/exhale boundary (labels are descriptive only; they do not drive the math).
4. **Selection mechanic**: a part is defined by clicking two endpoints on the trace; clicks **snap to the nearest numbered image marker**. The part is defined by its x-axis time range; the image numbers travel with it for tracking.
5. **Assembly**: parts are **concatenated in list order** into the synthetic profile. Order of the list = assembly order.
6. **Reorder map**: the original image numbers spanned by each part, in assembled order, form 🅑.
7. **Reuse/overlap allowed**: an image may appear in more than one part; 🅑 is therefore a **sequence**, not a permutation.

## 4. Architecture (Approach 1 — tested engine, thin GUI)

| File | Change |
|---|---|
| `src/labchart2cfd/processing/profile_assembly.py` | **NEW** — pure, Qt-free assembly engine |
| `src/labchart2cfd/io/csv_export.py` | add `export_reorder_map_csv(...)`; fix CRLF (`newline=""`) on all writers |
| `src/labchart2cfd/workflows/ct.py` | allow `process()` to return the processed window unrearranged for the engine to splice |
| `src/labchart2cfd/gui/app.py` | replace inline splice (`~1429-1477`) with engine calls; add full-manual window + parts list/preview |
| `tests/test_profile_assembly.py` | **NEW** — unit tests for the engine |
| `tests/test_io.py` | strengthen CRLF assertion; add reorder-map export test |

Rationale: the riskiest existing code (untested inline splice flagged in the 2026-06 review) is replaced by a small, pure, tested function. Matches the repo architecture (pure primitives + thin surfaces) and enables future CLI reuse.

## 5. Data model

```python
@dataclass
class ProfilePart:
    start_image: int      # 1-based image number, inclusive
    end_image: int        # 1-based image number, inclusive (>= start_image)
    label: str = ""       # optional, descriptive only

@dataclass
class AssemblyResult:
    time: np.ndarray                  # synthetic, starts at 0, at target rate (100 Hz)
    flow: np.ndarray                  # 🅐 assembled mass-flow
    pressure: Optional[np.ndarray]    # assembled pressure, or None
    reorder_map: list[tuple[int, int, float]]  # 🅑 (new_frame, original_image, original_time_s)
```

The GUI's labeling state is exactly an ordered `list[ProfilePart]` plus the window and temporal resolution.

## 6. Assembly semantics — **process-then-assemble**

1. `CTWorkflow` processes the **whole manual window once**: drift-correct (sign +1) → L/s→kg/s → resample to 100 Hz → smooth. **No rearrangement.**
2. The engine **splices the already-processed window** per part. Part `(a, b)` maps to the processed-time span and those samples are taken in order; parts are concatenated.
3. The reorder map collects images `a…b` from each part, in order, with their original acquisition time.

Rejected alternative (assemble raw, then process): would let smoothing/drift bleed **across artificial part boundaries**, mixing non-adjacent breaths — physically wrong for a reassembled cycle.

**Fencepost boundary convention (the off-by-one the old code got wrong).** Image `i` sits at processed time `t_i = (i-1)·dt`.
- *Flow (🅐):* a part `(a, b)` contributes the processed samples in the half-open span `[t_a, t_b)`; the **final part in the list** uses the closed span `[t_a, t_b]`, so the assembled profile spans end-to-end while internal joins never double-count the shared instant.
- *Reorder map (🅑):* a part `(a, b)` lists images `a, a+1, …, b` (**inclusive**, `b-a+1` entries). Concatenated across parts in order, this is the per-image map.
- Flow and map **intentionally live at different granularities** (continuous 100 Hz vs. one row per image); they are not equal length. Both conventions are **pinned by dedicated unit tests**.

Drift correction runs over the manual window (existing CT behavior). A future toggle to disable it for manual mode is out of scope for v1.

## 7. GUI flow

Panel (generalized from the existing CT controls):

```
Mode:  ( ) Auto (triggers)   (•) Full manual
Channels:  Flow [3 FLOW ▾]  Trigger [none ▾]  Pressure [5 MASK ▾]
Window:  start [190.0] s  end [202.0] s  [Set from clicks]
Temporal res [200] ms  →  Images: 1 … 61  (auto from window)
Parts (top = first in assembled profile):
  # | label          | images | dur(s) | [↑][↓][edit][x]
  1 | inhale+½exhale  |  4–9   |  1.00  |
  2 | exhale tail     |  1–3   |  0.40  |
[ + Add Part ]   [ Assemble & Preview ]   [ Export Profile + Map ]
```

1. Pick Flow (and optional Pressure). Trigger optional/ignored in manual mode.
2. Set window (type or "Set from clicks"); window shades and the numbered fencepost grid draws. Window↔N kept consistent.
3. "Add Part" → click two endpoint frames (snap to nearest image); part appended to the ordered list and highlighted on the plot with its order number.
4. Reorder / edit-label / delete parts; list order = assembly order.
5. "Assemble & Preview" → engine runs; preview pane shows synthetic flow + a small reorder-map table.
6. "Export Profile + Map" → choose output folder; write CSVs.

The GUI holds only the parts list + window/res and calls the engine. **State resets cleanly on window/mode change** (fixes the cycle-frame leak and shift-desync findings).

## 8. Outputs

To a user-chosen folder:
- `{subject}FlowProfile.csv` — 🅐 synthetic mass-flow, 100 Hz, CRLF (with `newline=""`).
- `{subject}PressureProfile.csv` — only when a pressure channel is set.
- `{subject}ImageReorderMap.csv` — 🅑 header `new_frame,original_image,original_time_s`.

## 9. Error handling (no silent failures)

- No parts → block export with a clear message.
- Part image outside `1…N` → rejected at add time.
- Window contains no samples → explicit `ValueError` (fixes the opaque `IndexError`).
- Reused image across parts → allowed, with a non-blocking warning (so it is intentional).

## 10. Testing

- `assemble_parts`: single-image part; multi-part ordering; half-open boundary (no double-count at joins); reorder-map correctness; overlap/reuse; out-of-range rejection; empty input.
- Export: reorder-map format; **CRLF assertion that rejects `\r\r\n`** (the current test cannot).
- One synthetic end-to-end: in-memory `LabChartData` → window → 2 parts → assert assembled flow + map.

## 11. Scope boundaries (YAGNI — explicitly NOT in v1)

- ❌ No DICOM/image-file renaming — we emit the map CSV; the clinician applies it in their image tool.
- ❌ No changes to auto trigger-detection (manual mode stands alone).
- ❌ No multi-cycle stitching beyond the ordered-parts model.
- ❌ No drift-correction on/off toggle (keeps existing CT behavior).
- ✅ Pressure handled only if a pressure channel is chosen.

## 12. Review findings closed by this work

- **CRLF blocker** (`csv_export.py`) — fixed by `newline=""` across writers + strengthened test.
- **CT cycle-frame state leak** (`gui/app.py`) — eliminated; state is one list, reset on mode/window change.
- **Untested CT splice** — replaced by the pure, tested `profile_assembly` engine.
- **Opaque empty-window `IndexError`** — replaced by an explicit error.

(Out of scope here, tracked separately: PhaseContrast calibration bug, pydantic-v1 config dead code, Windows packaging.)
