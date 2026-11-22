"""Tests for the CT composed breath (design 2026-11-22-ct-composed-breath).

The fixture recording has a scan step from 2 s to 5 s on the trigger channel.
"""
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
