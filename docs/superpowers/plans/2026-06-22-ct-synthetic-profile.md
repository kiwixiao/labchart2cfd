# CT Synthetic Profile Assembly — Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Let a clinician manually define a window, label ordered parts of the flow, and assemble a synthetic breathing profile plus an image reorder map.

**Architecture:** A pure, tested engine (`processing/profile_assembly.py`) does the splice; `CTWorkflow` processes the window unrearranged (process-then-assemble); the GUI is a thin controller; a new CSV exporter writes the reorder map. Spec: `docs/superpowers/specs/2026-06-22-ct-synthetic-profile-design.md`.

**Tech Stack:** Python, NumPy, SciPy, PyQt5, pytest. Env: conda `labchart` (`/Users/xiaz9n/anaconda3/envs/labchart/bin/python`).

## Global Constraints

- Row/column indices are **1-indexed** everywhere (LabChart convention).
- CSV writers must use `open(..., "w", newline="")` (Windows CRLF correctness).
- No `pydantic`/`config` involvement (dead code, out of scope).
- Engine module must NOT import PyQt5 or matplotlib (stays unit-testable, headless).
- Run tests with: `/Users/xiaz9n/anaconda3/envs/labchart/bin/python -m pytest`.

---

### Task 1: Assembly engine — data model + `assemble_parts`

**Files:**
- Create: `src/labchart2cfd/processing/profile_assembly.py`
- Test: `tests/test_profile_assembly.py`

**Interfaces:**
- Produces: `ProfilePart(start_image:int, end_image:int, label:str="")`; `AssemblyResult(time, flow, pressure, reorder_map)`; `assemble_parts(flow:np.ndarray, parts:list[ProfilePart], dt:float, sample_rate:float, total_images:int, pressure:Optional[np.ndarray]=None) -> AssemblyResult`. `reorder_map: list[tuple[int,int,float]]` = `(new_frame, original_image, original_time_s)`.

Semantics: `spi = round(dt*sample_rate)`. Part `(a,b)` flow span = samples `[(a-1)*spi, (b-1)*spi)` (half-open), except the **last** part `[(a-1)*spi, (b-1)*spi]` (closed). Reorder map lists images `a..b` inclusive per part. Validate `1 <= start < end <= total_images`; empty parts → `ValueError`.

- [ ] **Step 1:** Write failing tests in `tests/test_profile_assembly.py`:
```python
import numpy as np, pytest
from labchart2cfd.processing.profile_assembly import ProfilePart, assemble_parts

def _flow(n): return np.arange(n, dtype=float)  # ramp 0..n-1, 100 Hz

def test_single_part_reorders_and_lengths():
    # N=11 images, dt=0.2, 100 Hz -> spi=20, window len = 201 samples
    flow = _flow(201)
    r = assemble_parts(flow, [ProfilePart(4, 9)], dt=0.2, sample_rate=100.0, total_images=11)
    # last (only) part is closed: samples [60 .. 160] inclusive -> 101 samples
    assert len(r.flow) == 101
    assert r.flow[0] == 60.0 and r.flow[-1] == 160.0
    # reorder map lists images 4..9 inclusive
    assert [m[1] for m in r.reorder_map] == [4, 5, 6, 7, 8, 9]
    assert r.reorder_map[0] == (1, 4, pytest.approx(0.6))

def test_two_parts_join_no_double_count():
    flow = _flow(201)
    r = assemble_parts(flow, [ProfilePart(4, 9), ProfilePart(1, 3)], dt=0.2, sample_rate=100.0, total_images=11)
    # part1 non-last half-open [60,160) = 100 samples; part2 last closed [0,40] = 41
    assert len(r.flow) == 141
    assert r.flow[0] == 60.0 and r.flow[99] == 159.0 and r.flow[100] == 0.0 and r.flow[-1] == 40.0
    assert [m[1] for m in r.reorder_map] == [4, 5, 6, 7, 8, 9, 1, 2, 3]

def test_pressure_assembled_parallel():
    flow = _flow(201); pres = _flow(201) * 10
    r = assemble_parts(flow, [ProfilePart(2, 4)], dt=0.2, sample_rate=100.0, total_images=11, pressure=pres)
    assert r.pressure is not None and len(r.pressure) == len(r.flow)
    assert r.pressure[0] == 200.0  # image2 -> sample 20 -> 20*10

def test_empty_parts_raises():
    with pytest.raises(ValueError):
        assemble_parts(_flow(201), [], dt=0.2, sample_rate=100.0, total_images=11)

@pytest.mark.parametrize("a,b", [(0, 3), (3, 3), (5, 2), (9, 12)])
def test_invalid_parts_raise(a, b):
    with pytest.raises(ValueError):
        assemble_parts(_flow(201), [ProfilePart(a, b)], dt=0.2, sample_rate=100.0, total_images=11)
```
- [ ] **Step 2:** Run `... -m pytest tests/test_profile_assembly.py -q` → expect FAIL (module missing).
- [ ] **Step 3:** Implement `profile_assembly.py` (dataclasses + `assemble_parts` per semantics above; clamp slice indices to `len(flow)`).
- [ ] **Step 4:** Run the tests → expect PASS.
- [ ] **Step 5:** Commit: `git add src/labchart2cfd/processing/profile_assembly.py tests/test_profile_assembly.py && git commit -m "feat: add profile_assembly engine for CT synthetic profiles"`.

---

### Task 2: CSV export — CRLF fix + reorder-map writer

**Files:**
- Modify: `src/labchart2cfd/io/csv_export.py`
- Test: `tests/test_io.py`

**Interfaces:**
- Produces: `export_reorder_map_csv(filepath, reorder_map) -> Path` writing header `new_frame,original_image,original_time_s`.

- [ ] **Step 1:** Add failing tests to `tests/test_io.py`: assert exported flow CSV bytes contain `b"\r\n"` and NOT `b"\r\r\n"`; assert `export_reorder_map_csv` writes the header + rows `1,4,0.60000`.
- [ ] **Step 2:** Run → FAIL (no `export_reorder_map_csv`).
- [ ] **Step 3:** Add `newline=""` to the three existing `open(...)` calls (lines ~41, 72, 108); add `export_reorder_map_csv`.
- [ ] **Step 4:** Run `tests/test_io.py` → PASS.
- [ ] **Step 5:** Commit: `git commit -am "fix: CRLF newline on CSV writers; add reorder-map export"`.

---

### Task 3: Pin CTWorkflow window processing (no-rearrange path)

**Files:** Test: `tests/test_workflows.py` (new)

- [ ] **Step 1:** Write a test building a tiny in-memory `LabChartData` (follow `tests/test_io.py` construction), run `CTWorkflow().process(..., inhale_start_time=None)`, assert: `len(result.time)` matches window·rate, `result.metadata["total_images"]` equals `round(dur/dt)+1`, `result.metadata` has no rearrangement (cut_image_index == 0).
- [ ] **Step 2:** Run → PASS (behavior already exists) or fix if it surfaces a bug.
- [ ] **Step 3:** Commit: `git commit -am "test: pin CTWorkflow no-rearrange window processing"`.

---

### Task 4: GUI full-manual mode + parts list + engine wiring

**Files:** Modify: `src/labchart2cfd/gui/app.py`

Sub-steps (manual GUI verification; keep changes minimal and reversible):
- [ ] **4a:** Add Mode radio (Auto / Full manual). In Full manual, window start/end come from spinboxes or "Set from clicks"; `total_images` derives from window & temporal-res and stays consistent (connect the spinbox `valueChanged` signals — fixes the no-signal finding).
- [ ] **4b:** Add a Parts table widget (`# | label | images | dur | ↑ ↓ edit ✕`). State = `self._parts: list[ProfilePart]`. "Add Part" arms a two-click capture that reuses `_snap_to_fencepost`; append part, draw its span + order number.
- [ ] **4c:** Replace the inline splice block (`app.py:1429-1477`) with: `from labchart2cfd.processing.profile_assembly import assemble_parts`; call it with `result.mass_flow`, `self._parts`, `dt`, `result.sample_rate`, `total_images`, `result.pressure`; set `result.time/mass_flow/pressure` from the `AssemblyResult`; store `reorder_map` in `result.metadata`.
- [ ] **4d:** "Export Profile + Map" writes flow CSV (+ pressure if set) + `export_reorder_map_csv`.
- [ ] **4e:** Reset `self._parts` + artists on mode/window change (fixes cycle-frame leak); ensure `_select_ct_step` and `_clear_all_markers` clear the new state too.
- [ ] **Commit:** `git commit -am "feat: GUI full-manual CT synthetic-profile mode"`.

---

### Task 5: Real-data end-to-end verification

**Files:** `/tmp/labchart_ct_test/verify_assembly.py` (throwaway, not committed)

- [ ] Build parts against a manual window on the cached decimated flow (or the real file), call `assemble_parts`, export all three CSVs to `/tmp`, assert: flow CSV well-formed (no `\r\r\n`), reorder map matches expected order, lengths consistent. Print a summary.

---

### Task 6: Full suite + branch tidy

- [ ] Run full `pytest` → all green.
- [ ] `git status` clean (only intended files); plan + spec committed.

## Self-Review

- **Spec coverage:** window/grid (T4a), parts model+assembly (T1), snap (T4b), reorder map (T1/T2), outputs (T2/T4d), errors (T1 validation, empty-window covered by CTWorkflow), testing (T1/T2/T3/T5), CRLF (T2), cycle-leak fix (T4e). ✓
- **Placeholders:** none — engine + export + tests have full code; GUI steps reference exact insertion points.
- **Type consistency:** `assemble_parts` signature identical in T1 and T4c; `reorder_map` tuple shape identical in T1 and T2.
