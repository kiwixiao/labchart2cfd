"""Tests for the CT composed breath (design 2026-11-22-ct-composed-breath).

The fixture recording has a scan step from 2 s to 5 s on the trigger channel.
"""
from pathlib import Path

import pytest

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


def _cycle_export(win, qapp, mat_file, out_dir, dt_ms, order=_USER_ORDER):
    """Label the cycle on the fixture step (2 s to 5 s) and export it."""
    _step_selected(win, mat_file, "cycle")
    win.ct_temporal_res_spin.setValue(dt_ms)
    win.cycle_mode_btn.setChecked(True)
    dt = dt_ms / 1000.0
    for image in order:
        win._on_click(ClickEvent(win.ax_top, 2.0 + (image - 0.5) * dt))
    assert win._ct_cycle_frames == order
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
