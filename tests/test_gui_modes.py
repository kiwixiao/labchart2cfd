"""Tests for click modes and CT methods.

A left click on the plot can mean five different things. These tests pin that
the banner above the plot always says which one applies, that the click then
does that thing, and that the three CT methods cannot leak into one another.
"""
import pytest

from tests.gui_helpers import ClickEvent, choose_ct_method, open_block

pytestmark = pytest.mark.usefixtures("no_modal_dialogs")


def _banner(win) -> str:
    return win.mode_banner.text().lower()


# -------------------------------------------------------------------- banner

def test_banner_asks_for_a_recording_when_none_is_open(win):
    assert "open" in _banner(win)
    assert win.mode_banner.property("kind") == "idle"


def test_banner_says_overview_tiles_can_be_clicked(win, mat_file):
    win.open_recording(mat_file)

    assert "click a block" in _banner(win)


def test_banner_says_a_plain_click_sets_start_then_end(win, mat_file):
    open_block(win, mat_file)

    assert "start" in _banner(win)
    assert "end" in _banner(win)
    assert win.mode_banner.property("kind") == "idle"


def test_banner_follows_an_armed_landmark_and_returns_after_the_click(win, mat_file):
    win.workflow_combo.setCurrentText("CT")
    choose_ct_method(win, "landmarks")
    open_block(win, mat_file)

    win._mark_inhale_start()
    assert "inhale" in _banner(win)
    assert win.mode_banner.property("kind") == "active"

    win._on_click(ClickEvent(win.ax_bottom, 3.0))

    assert "inhale" not in _banner(win)          # the single click is spent
    assert win.ct_inhale_label.text() == "3.000 s"


def test_an_armed_landmark_click_does_not_move_the_window(win, mat_file):
    """The click does what the banner said, and only that."""
    win.workflow_combo.setCurrentText("CT")
    choose_ct_method(win, "landmarks")
    open_block(win, mat_file)
    win.start_spin.setValue(2.0)
    win.end_spin.setValue(5.0)

    win._mark_exhale_end()
    win._on_click(ClickEvent(win.ax_bottom, 4.0))

    assert win.start_spin.value() == pytest.approx(2.0)
    assert win.end_spin.value() == pytest.approx(5.0)
    assert win.ct_exhale_label.text() == "4.000 s"


def test_banner_announces_cycle_labelling(win, mat_file):
    win.workflow_combo.setCurrentText("CT")
    choose_ct_method(win, "cycle")
    open_block(win, mat_file)

    win.cycle_mode_btn.setChecked(True)
    assert "cycle" in _banner(win)
    assert win.mode_banner.property("kind") == "active"

    win.cycle_mode_btn.setChecked(False)
    assert "cycle" not in _banner(win)


def test_banner_explains_why_plain_clicks_do_nothing_in_parts_method(win, mat_file):
    win.workflow_combo.setCurrentText("CT")
    open_block(win, mat_file)
    assert "select step" in _banner(win)          # the first action in CT

    win._select_ct_step()

    assert "pick on plot" in _banner(win)


def test_banner_walks_through_picking_a_part(win, mat_file):
    win.workflow_combo.setCurrentText("CT")
    open_block(win, mat_file)
    win._select_ct_step()                          # Start 2.0, End 5.0, 15 images

    win._arm_add_part()
    assert "first image" in _banner(win)
    assert win.mode_banner.property("kind") == "active"

    win._on_click(ClickEvent(win.ax_top, 2.5))
    assert "last image" in _banner(win)

    win._on_click(ClickEvent(win.ax_top, 3.5))
    assert "pick on plot" in _banner(win)        # capture finished


def test_banner_warns_while_the_zoom_tool_swallows_clicks(win, mat_file):
    """Clicks were dropped silently while zoom or pan was active."""
    open_block(win, mat_file)

    win.toolbar.zoom()
    assert "ignored" in _banner(win)
    assert win.mode_banner.property("kind") == "blocked"

    win.toolbar.zoom()   # toggles the tool off again
    assert "ignored" not in _banner(win)
    assert win.mode_banner.property("kind") == "idle"


# ---------------------------------------------------------------- CT methods

def test_reorder_parts_method_is_what_turns_manual_mode_on(win):
    """No separate enable switch: the mode follows the chosen method."""
    win.workflow_combo.setCurrentText("CT")
    assert win._synth_mode

    choose_ct_method(win, "landmarks")
    assert not win._synth_mode

    choose_ct_method(win, "parts")
    assert win._synth_mode

    win.workflow_combo.setCurrentText("MRI")
    assert not win._synth_mode


def test_choosing_ct_with_a_block_open_looks_for_scan_steps(win, mat_file):
    """Steps are found when a block is plotted. Choosing CT afterwards used to
    leave none, and Select Step then asked for a detection first."""
    open_block(win, mat_file)                # opened as MRI: pulse detection
    assert win._ct_steps == []

    win.workflow_combo.setCurrentText("CT")

    assert len(win._ct_steps) == 1           # the 2 s to 5 s step
    assert win._plot_mode == "selected"


def test_selecting_a_step_seeds_the_parts_image_count_once(win, mat_file):
    """Manual mode is now on before a step exists, so the seed that used to
    happen on ticking the switch has to happen when the step is selected."""
    win.workflow_combo.setCurrentText("CT")
    open_block(win, mat_file)

    win._select_ct_step()            # the 2 s to 5 s step at 200 ms is 15 images
    assert win.synth_images_spin.value() == 15

    win.synth_images_spin.setValue(12)
    win._select_ct_step()
    assert win.synth_images_spin.value() == 12   # yours now; never rewritten


def test_switching_method_disarms_the_previous_methods_click_mode(win, mat_file):
    win.workflow_combo.setCurrentText("CT")
    choose_ct_method(win, "cycle")
    open_block(win, mat_file)
    win.cycle_mode_btn.setChecked(True)

    choose_ct_method(win, "landmarks")
    assert not win._ct_cycle_mode

    win._mark_inhale_start()
    choose_ct_method(win, "parts")
    assert win._ct_landmark_mode is None


def test_export_reads_only_the_chosen_ct_method(win):
    """Cycle labels used to override landmarks silently, whichever was on screen."""
    win.workflow_combo.setCurrentText("CT")
    win._ct_cycle_frames = [3, 4, 5]
    win._ct_inhale_start = 1.0
    win._ct_exhale_end = 2.0

    choose_ct_method(win, "landmarks")
    assert win._ct_export_inputs() == ([], 1.0, 2.0)

    choose_ct_method(win, "cycle")
    assert win._ct_export_inputs() == ([3, 4, 5], None, None)

    choose_ct_method(win, "parts")
    assert win._ct_export_inputs() == ([], None, None)
