"""Tests for the CT composed breath (design 2026-11-22-ct-composed-breath).

The fixture recording has a scan step from 2 s to 5 s on the trigger channel.
"""
from pathlib import Path

import numpy as np
import pytest
from PyQt5.QtWidgets import QDialog
from scipy.integrate import trapezoid

from labchart2cfd.gui import app as gui_app
from tests.gui_helpers import ClickEvent, choose_ct_method, open_block

pytestmark = pytest.mark.usefixtures("no_modal_dialogs")


def _step_selected(win, mat_file, method="cycle"):
    win.workflow_combo.setCurrentText("CT")
    choose_ct_method(win, method)
    open_block(win, mat_file)
    win._select_ct_step()


# ------------------------------------ 4.1: the window always holds whole images

def test_a_selected_step_holds_whole_images(win, mat_file):
    """A 3 s step at 400 ms is 7.5 images. The window is 8 whole images."""
    _step_selected(win, mat_file)

    win.ct_temporal_res_spin.setValue(400)

    assert win.ct_total_images_spin.value() == 8
    assert win.start_spin.value() == pytest.approx(2.0)
    assert win.end_spin.value() == pytest.approx(5.2)


def test_temporal_res_recounts_from_the_step_not_from_the_rounded_window(win, mat_file):
    """Counted from the rounded 3.2 s window, 300 ms would give 11 images."""
    _step_selected(win, mat_file)
    win.ct_temporal_res_spin.setValue(400)
    assert win.end_spin.value() == pytest.approx(5.2)   # the rounded window

    win.ct_temporal_res_spin.setValue(300)

    assert win.ct_total_images_spin.value() == 10
    assert win.end_spin.value() == pytest.approx(5.0)


def test_a_typed_end_rounds_to_whole_images(win, mat_file):
    """2.1 s at 200 ms is 10.5 images, rounded to 11."""
    _step_selected(win, mat_file)

    win.end_spin.setValue(4.1)

    assert win.ct_total_images_spin.value() == 11
    assert win.end_spin.value() == pytest.approx(4.2)


def test_a_typed_start_moves_the_whole_window(win, mat_file):
    """The count stays at the step's 15 images and End follows Start."""
    _step_selected(win, mat_file)

    win.start_spin.setValue(2.5)

    assert win.ct_total_images_spin.value() == 15
    assert win.end_spin.value() == pytest.approx(5.5)


def test_a_hand_window_holds_whole_images(win, mat_file):
    """No trigger: the window is set by two clicks and still rounds."""
    win.workflow_combo.setCurrentText("CT")
    win.trigger_row_spin.setValue(0)
    open_block(win, mat_file)

    win._on_click(ClickEvent(win.ax_bottom, 2.0))
    win._on_click(ClickEvent(win.ax_bottom, 4.1))

    assert win.start_spin.value() == pytest.approx(2.0)
    assert win.end_spin.value() == pytest.approx(4.2)
    assert win.ct_total_images_spin.value() == 11


# ------------------------------------ 4.3 and approach A: one export route

# The user's order on ENT005: images 4 to 15, then 1 to 4.
_USER_ORDER = list(range(4, 16)) + list(range(1, 5))
_REFERENCE = Path(__file__).parent / "reference"


def _wait_for_report(win, qapp, seconds=30):
    """The old cycle route wrote its files on a worker thread."""
    import time
    deadline = time.time() + seconds
    while win._last_export_report is None and time.time() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    assert win._last_export_report is not None, win.status_bar.currentMessage()


def _label_cycle(win, mat_file, dt_ms=100.0, order=_USER_ORDER):
    """Label the cycle on the fixture step (2 s to 5 s), clicking mid image."""
    _step_selected(win, mat_file, "cycle")
    win.ct_temporal_res_spin.setValue(dt_ms)
    win.cycle_mode_btn.setChecked(True)
    dt = dt_ms / 1000.0
    for image in order:
        win._on_click(ClickEvent(win.ax_top, 2.0 + (image - 0.5) * dt))
    assert win._ct_cycle_frames == order


def _cycle_export(win, qapp, mat_file, out_dir, dt_ms, order=_USER_ORDER):
    """Label the cycle on the fixture step (2 s to 5 s) and export it."""
    _label_cycle(win, mat_file, dt_ms, order)
    win.subject_edit.setText("S1")
    win._set_output_dir(out_dir)
    win._export()
    _wait_for_report(win, qapp)
    return win._last_export_report


def _parts_export(win, qapp, mat_file, out_dir, dt_ms=200.0, parts=((6, 15), (1, 5))):
    """Reorder parts on the fixture step, accepted in the preview."""
    _step_selected(win, mat_file, "parts")
    win.ct_temporal_res_spin.setValue(dt_ms)
    for first, last in parts:
        win.part_from_spin.setValue(first)
        win.part_to_spin.setValue(last)
        win._add_part_from_spins()
    win.subject_edit.setText("S1")
    win._set_output_dir(out_dir)
    win._export()
    _wait_for_report(win, qapp)
    return win._last_export_report


def _info(path):
    """The details file as a dict of key: value."""
    pairs = (line.split(": ", 1) for line in Path(path).read_text().splitlines() if ": " in line)
    return {k: v for k, v in pairs}


def _values(path):
    """The value column of a profile CSV, as written (exact text)."""
    return [line.split(",")[1] for line in Path(path).read_text().splitlines()[1:]]


def _times(path):
    """The time column of a profile CSV, as written (exact text)."""
    return [line.split(",")[0] for line in Path(path).read_text().splitlines()[1:]]


def _fmt(values):
    """Values as the profile writer prints them."""
    return ["%.9e" % v for v in values]


def _capture_preview(monkeypatch, choose=None, answer=QDialog.Accepted):
    """Stand in for the user at the preview: keep the dialog, optionally
    choose the complete half, then press Export (or Cancel)."""
    shown = []

    def fake_exec(dialog):
        shown.append(dialog)
        if choose is not None:
            dialog.keep_buttons[choose].click()
        return answer
    monkeypatch.setattr("labchart2cfd.gui.dialogs.AssemblyPreviewDialog.exec_", fake_exec)
    return shown


def _chosen_flow(dialog):
    return dialog.balances[dialog.chosen_keep].flow


@pytest.mark.parametrize("dt_ms", [200.0, 100.0])
def test_cycle_export_matches_the_old_route_sample_by_sample(
        win, qapp, mat_file, tmp_path, dt_ms, monkeypatch):
    """The old route cut each run of images with a floating point time mask
    and took one sample wrong at a run edge. At 200 ms it kept the first
    sample of image 5 after the last run (0.8 s fell just under the bound).
    At 100 ms it also dropped the first sample of image 4, because 3 x 0.1 is
    0.30000000000000004 in floating point. The new route cuts whole slices
    by sample index, as Reorder parts always did. Every other sample is the
    same text, value for value, in flow and pressure.

    The flow compared is the composed array before in equals out scales a
    half (4.7); the written flow is that array with one half scaled."""
    shown = _capture_preview(monkeypatch)
    _cycle_export(win, qapp, mat_file, tmp_path, dt_ms)
    composed = _fmt(shown[0].asm.flow)

    for name, new in (("flow", composed), ("pressure", _values(tmp_path / "S1PressureProfile.csv"))):
        old = _values(_REFERENCE / f"composed_cycle_{dt_ms:.0f}ms_{name}.csv")
        if dt_ms == 200.0:
            assert len(new) == 16 * 20 and new == old[:-1]
        else:
            assert len(new) == 16 * 10
            assert new[1:120] == old[:119] and new[120:] == old[119:-1]
    assert _values(tmp_path / "S1FlowProfile.csv") == _fmt(_chosen_flow(shown[0]))


def test_parts_export_flow_is_byte_identical(win, qapp, mat_file, tmp_path, monkeypatch):
    """Pressure is byte identical. The composed flow before in equals out is
    the old route's flow, value for value; the written flow is it with the
    named half scaled (4.7, the scale always runs)."""
    shown = _capture_preview(monkeypatch)
    _parts_export(win, qapp, mat_file, tmp_path)

    written = (tmp_path / "S1PressureProfile.csv").read_bytes()
    assert written == (_REFERENCE / "composed_parts_200ms_pressure.csv").read_bytes()
    golden_flow = _REFERENCE / "composed_parts_200ms_flow.csv"
    assert _fmt(shown[0].asm.flow) == _values(golden_flow)
    assert _times(tmp_path / "S1FlowProfile.csv") == _times(golden_flow)
    assert _values(tmp_path / "S1FlowProfile.csv") == _fmt(_chosen_flow(shown[0]))
    old = (_REFERENCE / "composed_parts_200ms_reorder_map.csv").read_text().splitlines()
    new = (tmp_path / "S1ImageReorderMap.csv").read_text().splitlines()
    assert [",".join(line.split(",")[:3]) for line in new] == old


_NEW_DETAIL_KEYS = {
    "ct_method", "image_count", "grid_start_time", "grid_offset_s", "start_shift_ms",
    "window_start_time", "window_end_time", "drift_source", "drift_range_start",
    "drift_range_end", "drift_removed_L_per_s", "frame_order_new_to_original",
    "profile_duration_s", "frame_table_file",
}


def test_cycle_export_writes_the_frame_table_and_keeps_its_old_files(win, qapp, mat_file, tmp_path):
    report = _cycle_export(win, qapp, mat_file, tmp_path, 200.0)

    names = {p.name for p in report["files"]}
    assert names == {"S1FlowProfile.csv", "S1PressureProfile.csv", "S1ImageReorderMap.csv",
                     "S1_trigger_info.txt", "S1_frame_order.png", "S1_sanity_check.png"}
    assert all(p.exists() for p in report["files"])
    header = (tmp_path / "S1ImageReorderMap.csv").read_text().splitlines()[0]
    assert header.endswith("profile_start_s,profile_end_s,extra_flow_s,extra_flow_until_s")
    info = _info(tmp_path / "S1_trigger_info.txt")
    old_keys = set(_info(_REFERENCE / "composed_cycle_200ms_trigger_info.txt"))
    assert old_keys <= set(info)
    assert _NEW_DETAIL_KEYS <= set(info)
    assert info["ct_method"] == "cycle frames"
    assert info["cycle_original_indices"] == ",".join(str(i) for i in _USER_ORDER)
    assert info["frame_table_file"] == "S1ImageReorderMap.csv"
    assert info["drift_source"] == "image grid"


def test_parts_export_writes_the_details_file(win, qapp, mat_file, tmp_path):
    report = _parts_export(win, qapp, mat_file, tmp_path)

    assert "S1_trigger_info.txt" in {p.name for p in report["files"]}
    info = _info(tmp_path / "S1_trigger_info.txt")
    assert _NEW_DETAIL_KEYS <= set(info)
    assert info["ct_method"] == "reorder parts"
    assert info["frame_order_new_to_original"].startswith("1->6;2->7;")
    assert info["image_count"] == "15"


def test_parts_export_at_275_ms_succeeds_and_the_table_names_the_samples_used(
        win, qapp, mat_file, tmp_path):
    """275 ms is 27.5 samples at 100 Hz. It used to be rounded to 280 ms."""
    _parts_export(win, qapp, mat_file, tmp_path, dt_ms=275.0, parts=((6, 11), (1, 5)))

    assert win._synth_grid()[1] == pytest.approx(0.275)
    rows = [line.split(",") for line in (tmp_path / "S1ImageReorderMap.csv").read_text().splitlines()[1:]]
    for row in rows:
        image = int(row[1])
        first_sample = int((image - 1) * 27.5 + 0.5)
        assert float(row[2]) == pytest.approx(2.0 + first_sample / 100, abs=1e-6)
    assert [int(r[1]) for r in rows] == [6, 7, 8, 9, 10, 11, 1, 2, 3, 4, 5]


def test_extra_flow_does_not_move_the_drift_level_without_a_breath(
        win, qapp, mat_file, tmp_path, monkeypatch):
    """Without a picked breath the drift is measured on the image grid, so
    keeping extra flow after a part leaves the level of the profile alone."""
    from labchart2cfd.workflows.ct import CTWorkflow
    calls = []
    original = CTWorkflow.process

    def spy(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        calls.append((args[4], kwargs.get("drift_range"), result.drift_error))
        return result
    monkeypatch.setattr(CTWorkflow, "process", spy)

    _parts_export(win, qapp, mat_file, tmp_path / "a")
    win._parts[-1].extra_until_s = 7.0
    win._last_export_report = None
    win._set_output_dir(tmp_path / "b")
    win._export()
    _wait_for_report(win, qapp)

    (end_a, range_a, drift_a), (end_b, range_b, drift_b) = calls
    assert range_a == range_b == (pytest.approx(2.0), pytest.approx(5.0))
    assert drift_a == drift_b
    assert end_b >= 7.0 > end_a


@pytest.mark.parametrize("method", ["cycle", "parts"])
def test_export_targets_lists_the_files_written(win, qapp, mat_file, tmp_path, method):
    if method == "cycle":
        report = _cycle_export(win, qapp, mat_file, tmp_path, 200.0)
    else:
        report = _parts_export(win, qapp, mat_file, tmp_path)

    assert {p.name for p in win._export_targets()} == {p.name for p in report["files"]}


# ------------------------------------------- 4.4: the drift correction breath

def _type_breath(win, start, end):
    win.drift_from_spin.setValue(start)
    win.drift_to_spin.setValue(end)


def test_pick_on_plot_sets_the_drift_breath_with_two_clicks(win, mat_file):
    _step_selected(win, mat_file, "landmarks")

    win.drift_pick_btn.click()
    assert win._click_mode() == "drift_first"
    win._on_click(ClickEvent(win.ax_bottom, 6.0))
    assert win._click_mode() == "drift_second"
    win._on_click(ClickEvent(win.ax_bottom, 2.0))     # backward: still one breath

    assert win._drift_breath() == (pytest.approx(2.0), pytest.approx(6.0))
    assert win.drift_from_spin.value() == pytest.approx(2.0)
    assert win.drift_to_spin.value() == pytest.approx(6.0)
    assert win._click_mode() == "ct_landmarks_idle"


def test_right_click_cancels_picking_the_drift_breath(win, mat_file):
    _step_selected(win, mat_file, "landmarks")
    win.drift_pick_btn.click()
    win._on_click(ClickEvent(win.ax_bottom, 6.0))

    win._on_click(ClickEvent(win.ax_bottom, 2.0, button=3))

    assert win._drift_breath() is None
    assert win._click_mode() == "ct_landmarks_idle"


def test_typed_from_and_to_set_the_drift_breath(win, mat_file):
    _step_selected(win, mat_file, "landmarks")

    _type_breath(win, 1.0, 5.0)

    assert win._drift_breath() == (pytest.approx(1.0), pytest.approx(5.0))


def test_the_drift_breath_is_drawn_on_both_plots(win, mat_file):
    _step_selected(win, mat_file, "landmarks")

    _type_breath(win, 1.0, 5.0)

    assert {a.axes for a in win._drift_artists} >= {win.ax_top, win.ax_bottom}
    assert any(getattr(a, "get_text", lambda: "")().strip() == "drift breath"
               for a in win._drift_artists)
    win.drift_clear_btn.click()
    assert win._drift_artists == [] and win._drift_breath() is None


def test_no_breath_gives_an_amber_caution_naming_the_range(win, mat_file):
    _step_selected(win, mat_file, "landmarks")
    win.subject_edit.setText("S1")
    win._ct_dt_confirmed = True
    win._update_export_hint()

    hint = win.export_hint.text()
    assert win.export_hint.property("state") == "caution"
    assert hint.startswith("Ready to export")
    assert "no drift correction breath" in hint and "2.000 s to 5.000 s" in hint

    _type_breath(win, 1.0, 5.0)

    assert "drift correction breath" not in win.export_hint.text()


def test_a_breath_outside_the_recording_holds_the_export_back(win, mat_file):
    _step_selected(win, mat_file, "landmarks")
    win.subject_edit.setText("S1")

    _type_breath(win, 8.0, 12.0)                      # the block ends near 10 s

    hint = win.export_hint.text()
    assert hint.startswith("Before export") and "drift correction breath" in hint


@pytest.mark.parametrize("method", ["landmarks", "cycle"])
def test_a_drift_breath_reaches_the_engine(win, qapp, mat_file, tmp_path, monkeypatch, method):
    from labchart2cfd.workflows.ct import CTWorkflow
    ranges = []
    original = CTWorkflow.process

    def spy(self, *args, **kwargs):
        ranges.append(kwargs.get("drift_range"))
        return original(self, *args, **kwargs)
    monkeypatch.setattr(CTWorkflow, "process", spy)
    _step_selected(win, mat_file, method)
    _type_breath(win, 2.0, 6.0)
    win.subject_edit.setText("S1")
    win._set_output_dir(tmp_path)

    win._export()
    _wait_for_report(win, qapp)

    assert ranges == [(pytest.approx(2.0), pytest.approx(6.0))]
    info = _info(tmp_path / "S1_trigger_info.txt")
    assert info["drift_source"] == "your breath"
    assert info["drift_range_start"] == "2.000000" and info["drift_range_end"] == "6.000000"


def test_landmarks_details_name_the_drift_without_a_breath(win, qapp, mat_file, tmp_path):
    _step_selected(win, mat_file, "landmarks")
    win.subject_edit.setText("S1")
    win._set_output_dir(tmp_path)

    win._export()
    _wait_for_report(win, qapp)

    info = _info(tmp_path / "S1_trigger_info.txt")
    assert info["drift_source"] == "image grid"
    assert (info["drift_range_start"], info["drift_range_end"]) == ("2.000000", "5.000000")
    assert {"window_start_time", "window_end_time", "drift_removed_L_per_s"} <= set(info)
    assert info["ct_method"] == "landmarks"


def test_selecting_another_step_asks_before_dropping_the_breath(win, mat_file, monkeypatch):
    _step_selected(win, mat_file, "landmarks")
    _type_breath(win, 1.0, 5.0)
    asked = []
    monkeypatch.setattr(
        "labchart2cfd.gui.app.QMessageBox.question",
        staticmethod(lambda _p, _title, text, *a, **k: asked.append(text) or gui_app.QMessageBox.Yes),
    )

    win._user_select_step()

    assert len(asked) == 1 and "the drift breath" in asked[0]
    assert win._drift_breath() is None


def test_drift_controls_are_hidden_outside_ct(win):
    win.workflow_combo.setCurrentText("CT")
    assert not win.drift_box.isHidden()

    win.workflow_combo.setCurrentText("MRI")

    assert win.drift_box.isHidden()


# ------------------------------------------------ 4.2: extra flow after a frame
# At 100 ms, frame 12 of the user's order is image 15, the last image of the
# run 4 to 15. It ends at 2.0 + 15 x 0.1 = 3.5 s. The recording ends at 9.999 s.

def _ask_recorder(monkeypatch, answer):
    asked = []
    monkeypatch.setattr(
        "labchart2cfd.gui.app.QMessageBox.question",
        staticmethod(lambda _p, _title, text, *a, **k: asked.append(text) or answer),
    )
    return asked


def test_extra_flow_after_frame_12_is_set_by_typing(win, mat_file):
    _label_cycle(win, mat_file)

    win.extra_frame_spin.setValue(12)
    win.extra_until_spin.setValue(3.8)

    assert win._ct_cycle_extra == {12: pytest.approx(3.8)}
    assert [(p.start_image, p.end_image, p.extra_until_s) for p in win._export_parts()] == [
        (4, 15, pytest.approx(3.8)), (1, 4, None),
    ]


def test_extra_flow_is_drawn_as_a_band_labelled_with_its_frame(win, mat_file):
    _label_cycle(win, mat_file)
    win.extra_frame_spin.setValue(12)

    win.extra_until_spin.setValue(3.8)

    bands = [a for a in win._extra_artists if hasattr(a, "get_hatch")]
    assert {a.axes for a in bands} >= {win.ax_top, win.ax_bottom}
    assert all(a.get_hatch() for a in bands)
    labels = [a.get_text() for a in win._extra_artists if hasattr(a, "get_text")]
    assert any("extra after frame 12" in text for text in labels)
    win.extra_clear_btn.click()
    assert win._ct_cycle_extra == {} and win._extra_artists == []


def test_extra_flow_is_picked_on_the_plot(win, mat_file):
    _label_cycle(win, mat_file)
    win.extra_frame_spin.setValue(12)

    win.extra_pick_btn.click()
    assert win._click_mode() == "extra_until"
    win._on_click(ClickEvent(win.ax_bottom, 3.8))

    assert win._ct_cycle_extra == {12: pytest.approx(3.8)}
    assert win.extra_until_spin.value() == pytest.approx(3.8)
    assert win._click_mode() != "extra_until"
    assert win._ct_cycle_frames == _USER_ORDER          # the click labelled nothing


def test_right_click_cancels_picking_extra_flow(win, mat_file):
    _label_cycle(win, mat_file)
    win.extra_frame_spin.setValue(12)
    win.extra_pick_btn.click()

    win._on_click(ClickEvent(win.ax_bottom, 3.8, button=3))

    assert win._ct_cycle_extra == {} and win._ct_cycle_frames == _USER_ORDER
    assert win._click_mode() != "extra_until"


def _ready_to_export(win):
    win.subject_edit.setText("S1")
    win._ct_dt_confirmed = True
    win._update_export_hint()
    return win.export_hint.text()


def test_extra_flow_before_the_next_image_is_named_above_export(win, mat_file):
    """Parts 1 to 5, then 6 to 10: image 6 already follows image 5."""
    _step_selected(win, mat_file, "parts")
    for first, last in ((1, 5), (6, 10)):
        win.part_from_spin.setValue(first)
        win.part_to_spin.setValue(last)
        win._add_part_from_spins()
    win.parts_list.setCurrentRow(0)

    win.extra_until_spin.setValue(3.2)                  # image 5 ends at 3.0 s

    assert win._parts[0].extra_until_s == pytest.approx(3.2)
    hint = _ready_to_export(win)
    assert hint.startswith("Before export")
    assert "extra flow after frame 5" in hint and "next image, 6" in hint


def test_extra_flow_before_its_image_ends_is_named_above_export(win, mat_file):
    _label_cycle(win, mat_file)
    win.extra_frame_spin.setValue(12)

    win.extra_until_spin.setValue(3.4)

    hint = _ready_to_export(win)
    assert hint.startswith("Before export")
    assert "extra flow after frame 12" in hint and "ends at 3.500 s" in hint


def test_extra_flow_past_the_recording_is_named_above_export(win, mat_file):
    _label_cycle(win, mat_file)
    win.extra_frame_spin.setValue(16)                   # the last frame: allowed

    win.extra_until_spin.setValue(10.5)

    hint = _ready_to_export(win)
    assert hint.startswith("Before export")
    assert "extra flow after frame 16" in hint and "9.989 s" in hint


def test_extra_flow_on_a_frame_inside_a_run_is_named_above_export(win, mat_file):
    """Frame 5 is image 8, inside the run 4 to 15: the next frame is image 9."""
    _label_cycle(win, mat_file)
    win.extra_frame_spin.setValue(5)

    win.extra_until_spin.setValue(3.0)

    hint = _ready_to_export(win)
    assert hint.startswith("Before export")
    assert "extra flow after frame 5" in hint and "next image, 9" in hint


def test_parts_extra_flow_is_set_on_the_selected_part(win, mat_file):
    _step_selected(win, mat_file, "parts")
    for first, last in ((6, 15), (1, 5)):
        win.part_from_spin.setValue(first)
        win.part_to_spin.setValue(last)
        win._add_part_from_spins()
    win.parts_list.setCurrentRow(0)

    win.extra_until_spin.setValue(5.3)                  # image 15 ends at 5.0 s

    assert [p.extra_until_s for p in win._parts] == [pytest.approx(5.3), None]
    assert "extra flow to 5.300 s" in win.parts_list.item(0).text()
    assert win.parts_list.currentRow() == 0
    win.parts_list.setCurrentRow(1)
    assert win.extra_until_spin.value() == 0.0           # the box shows that part's own


def test_extra_flow_reaches_the_export(win, qapp, mat_file, tmp_path):
    _label_cycle(win, mat_file)
    win.extra_frame_spin.setValue(12)
    win.extra_until_spin.setValue(3.8)
    win.subject_edit.setText("S1")
    win._set_output_dir(tmp_path)

    win._export()
    _wait_for_report(win, qapp)

    rows = (tmp_path / "S1ImageReorderMap.csv").read_text().splitlines()
    frame_12 = dict(zip(rows[0].split(","), rows[12].split(",")))
    assert frame_12["new_frame"] == "12" and frame_12["original_image"] == "15"
    assert frame_12["extra_flow_s"] == "0.300000"
    assert frame_12["extra_flow_until_s"] == "3.800000"


def test_clearing_labels_asks_before_dropping_extra_flow(win, mat_file, monkeypatch):
    _label_cycle(win, mat_file)
    win.extra_frame_spin.setValue(12)
    win.extra_until_spin.setValue(3.8)
    asked = _ask_recorder(monkeypatch, gui_app.QMessageBox.No)

    win.clear_cycle_btn.click()

    assert len(asked) == 1 and "extra flow after frame 12" in asked[0]
    assert win._ct_cycle_frames == _USER_ORDER and win._ct_cycle_extra

    asked = _ask_recorder(monkeypatch, gui_app.QMessageBox.Yes)
    win.clear_cycle_btn.click()

    assert win._ct_cycle_frames == [] and win._ct_cycle_extra == {}


def test_deleting_a_part_asks_before_dropping_its_extra_flow(win, mat_file, monkeypatch):
    _step_selected(win, mat_file, "parts")
    for first, last in ((6, 15), (1, 5)):
        win.part_from_spin.setValue(first)
        win.part_to_spin.setValue(last)
        win._add_part_from_spins()
    win.parts_list.setCurrentRow(0)
    win.extra_until_spin.setValue(5.3)
    asked = _ask_recorder(monkeypatch, gui_app.QMessageBox.No)

    win._delete_part()

    assert len(asked) == 1 and "extra flow after frame 10" in asked[0]
    assert len(win._parts) == 2


def test_a_redraw_names_the_extra_flow(win, mat_file):
    _label_cycle(win, mat_file)
    win.extra_frame_spin.setValue(12)
    win.extra_until_spin.setValue(3.8)

    assert "the extra flow after frame 12" in win._marks_at_risk(include_parts=True)


# ----------------------------------- 4.5 to 4.7: preview, picture, in equals out

def _texts(dialog, gid):
    return [t.get_text() for t in dialog.figure.axes[0].texts if t.get_gid() == gid]


def test_preview_labels_new_and_original_numbers(win, qapp, mat_file, tmp_path, monkeypatch):
    shown = _capture_preview(monkeypatch)
    _cycle_export(win, qapp, mat_file, tmp_path, 100.0)

    assert {"1", "12", "13", "16"} <= set(_texts(shown[0], "frame"))
    assert {"4", "15", "1"} <= set(_texts(shown[0], "image"))


def test_the_preview_curve_is_the_written_array(win, qapp, mat_file, tmp_path, monkeypatch):
    shown = _capture_preview(monkeypatch)
    _parts_export(win, qapp, mat_file, tmp_path)

    (curve,) = [line for line in shown[0].figure.axes[0].lines if line.get_gid() == "flow"]
    assert _fmt(curve.get_ydata()) == _values(tmp_path / "S1FlowProfile.csv")


def test_preview_shades_extra_flow(win, qapp, mat_file, tmp_path, monkeypatch):
    shown = _capture_preview(monkeypatch)
    _label_cycle(win, mat_file)
    win.extra_frame_spin.setValue(12)
    win.extra_until_spin.setValue(3.8)
    win.subject_edit.setText("S1")
    win._set_output_dir(tmp_path)

    win._export()
    _wait_for_report(win, qapp)

    axes = shown[0].figure.axes[0]
    assert [p for p in axes.patches if p.get_gid() == "extra"]
    (label,) = _texts(shown[0], "extra_label")
    assert "0.300 s" in label and "no image" in label


def test_the_written_flow_has_zero_net_volume(win, qapp, mat_file, tmp_path):
    report = _parts_export(win, qapp, mat_file, tmp_path)

    data = np.loadtxt(tmp_path / "S1FlowProfile.csv", delimiter=",", skiprows=1)
    net_ml = trapezoid(data[:, 1], data[:, 0]) / 1.2 * 1e6
    assert abs(net_ml) < 0.01
    summary = dict(report["summary"])
    assert summary["Net volume"].startswith("0.000 mL")
    info = _info(tmp_path / "S1_trigger_info.txt")
    assert {"complete_half", "volume_above_zero_mL_before_scaling",
            "volume_below_zero_mL_before_scaling", "composed_half_scale_factor",
            "volume_above_zero_mL", "volume_below_zero_mL", "net_volume_mL"} <= set(info)
    assert info["complete_half"] == "above zero"
    assert abs(float(info["net_volume_mL"])) < 0.01


@pytest.mark.parametrize("keep", ["positive", "negative"])
def test_the_complete_half_is_written_unchanged(win, qapp, mat_file, tmp_path, monkeypatch, keep):
    shown = _capture_preview(monkeypatch, choose=keep)
    _parts_export(win, qapp, mat_file, tmp_path)

    composed = shown[0].asm.flow
    kept = composed > 0 if keep == "positive" else composed < 0
    written = np.array(_values(tmp_path / "S1FlowProfile.csv"))
    assert list(written[kept]) == _fmt(composed[kept])
    assert list(written[~kept & (composed != 0)]) != _fmt(composed[~kept & (composed != 0)])
    side = "above zero" if keep == "positive" else "below zero"
    assert _info(tmp_path / "S1_trigger_info.txt")["complete_half"] == side


def test_the_frame_order_picture_is_saved(win, qapp, mat_file, tmp_path):
    report = _parts_export(win, qapp, mat_file, tmp_path)

    picture = tmp_path / "S1_frame_order.png"
    assert picture in report["files"]
    assert picture.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert report["plot"] == picture


def test_a_read_back_that_is_not_zero_is_red(win, qapp, mat_file, tmp_path, monkeypatch):
    """A writer that changed the values after the balance is caught, and named
    in red: the file is wrong, which is more than a caution."""
    from labchart2cfd.io import csv_export
    real = csv_export.export_flow_csv
    monkeypatch.setattr(csv_export, "export_flow_csv",
                        lambda path, time, flow, *a, **k: real(path, time, flow + 1e-5, *a, **k))

    report = _parts_export(win, qapp, mat_file, tmp_path)

    assert any("net volume" in e and "read back" in e for e in report["errors"])
    shown = win._result_dialog.error_label
    assert shown.objectName() == "errorBox" and not shown.isHidden()
    assert "net volume" in shown.text()


def _composed(offset):
    """A 2 s breath on a 200 ms grid, shifted by offset (kg/s)."""
    from labchart2cfd.processing.profile_assembly import ProfilePart, assemble_parts
    t = np.arange(0, 2.0, 0.01)
    flow = 4e-4 * np.sin(2 * np.pi * 0.5 * t) + offset
    return assemble_parts(flow, [ProfilePart(1, 10)], dt=0.2, sample_rate=100.0, total_images=10)


def test_a_factor_far_from_one_is_cautioned(qapp):
    from labchart2cfd.gui.dialogs import AssemblyPreviewDialog
    from labchart2cfd.processing.volume_balance import balance_choices
    near, far = _composed(0.0), _composed(1e-4)

    calm = AssemblyPreviewDialog(near, balance_choices(near.flow, 100.0), "positive", [])
    loud = AssemblyPreviewDialog(far, balance_choices(far.flow, 100.0), "positive", [])

    assert calm.caution_label.isHidden()
    assert not loud.caution_label.isHidden()
    assert "10 percent" in loud.caution_label.text()


def test_a_half_with_no_volume_is_named_not_crashed(qapp):
    from labchart2cfd.gui.dialogs import AssemblyPreviewDialog
    from labchart2cfd.processing.volume_balance import balance_choices
    one_sided = _composed(1e-3)                       # every sample above zero

    dialog = AssemblyPreviewDialog(one_sided, balance_choices(one_sided.flow, 100.0),
                                   "positive", [])

    assert not dialog.export_button.isEnabled()
    assert "one side of zero" in dialog.caution_label.text()
    assert not dialog.caution_label.isHidden()


def test_choosing_the_other_half_redraws_the_curve(qapp):
    from labchart2cfd.gui.dialogs import AssemblyPreviewDialog
    from labchart2cfd.processing.volume_balance import balance_choices
    asm = _composed(1e-5)
    choices = balance_choices(asm.flow, 100.0)
    dialog = AssemblyPreviewDialog(asm, choices, "positive", [])

    dialog.keep_buttons["negative"].click()

    assert dialog.chosen_keep == "negative"
    (curve,) = [line for line in dialog.figure.axes[0].lines if line.get_gid() == "flow"]
    assert np.array_equal(curve.get_ydata(), choices["negative"].flow)
