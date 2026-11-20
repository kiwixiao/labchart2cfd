"""Tests for the usability review of the redesigned window (first time user).

These are not crashes. Each is a place where the screen led a new user to the
wrong action, or left them without the fact they needed.
"""
import pytest

from labchart2cfd.gui import app as gui_app
from tests.gui_helpers import ClickEvent, choose_ct_method, open_block

pytestmark = pytest.mark.usefixtures("no_modal_dialogs")


def _ct_step_selected(win, mat_file, method):
    win.workflow_combo.setCurrentText("CT")
    choose_ct_method(win, method)
    open_block(win, mat_file)
    win._select_ct_step()


# --------------------------------- a plain click must not move the scan window

@pytest.mark.parametrize("method", ["landmarks", "cycle"])
def test_a_plain_click_leaves_the_selected_scan_step_alone(win, mat_file, method):
    """The banner said 'Click the plot to set Start', so the natural first
    click in these methods moved the scan window away from the step, with
    nothing to warn or undo."""
    _ct_step_selected(win, mat_file, method)
    assert (win.start_spin.value(), win.end_spin.value()) == (2.0, 5.0)

    win._on_click(ClickEvent(win.ax_bottom, 3.5))
    win._on_click(ClickEvent(win.ax_bottom, 4.0))

    assert (win.start_spin.value(), win.end_spin.value()) == (2.0, 5.0)
    assert win._time_markers == []


@pytest.mark.parametrize("method, button", [
    ("landmarks", "Mark Inhale Start"), ("cycle", "Label Cycle Frames"),
])
def test_the_banner_names_the_button_to_press_next(win, mat_file, method, button):
    _ct_step_selected(win, mat_file, method)

    banner = win.mode_banner.text()

    assert button in banner
    assert "set Start" not in banner


def test_before_a_step_is_selected_clicks_still_set_the_window(win, mat_file):
    """A recording without a usable trigger has no step to select."""
    win.workflow_combo.setCurrentText("CT")
    choose_ct_method(win, "landmarks")
    open_block(win, mat_file)

    win._on_click(ClickEvent(win.ax_bottom, 3.0))
    win._on_click(ClickEvent(win.ax_bottom, 4.0))

    assert (win.start_spin.value(), win.end_spin.value()) == (3.0, 4.0)


def test_selecting_a_step_removes_window_markers_that_no_longer_apply(win, mat_file):
    """A green marker stayed at a time that was no longer Start."""
    win.workflow_combo.setCurrentText("CT")
    choose_ct_method(win, "landmarks")
    open_block(win, mat_file)
    win._on_click(ClickEvent(win.ax_bottom, 3.0))
    win._on_click(ClickEvent(win.ax_bottom, 4.0))

    win._select_ct_step()

    assert win._time_markers == []
    assert (win.start_spin.value(), win.end_spin.value()) == (2.0, 5.0)


# ------------------------------------- the status line fits the method on screen

@pytest.mark.parametrize("method, wanted, unwanted", [
    ("parts", "Add Part", "Inhale"),
    ("landmarks", "Mark Inhale Start", "Label Cycle"),
    ("cycle", "Label Cycle Frames", "Inhale"),
])
def test_select_step_points_at_controls_that_are_on_screen(
        win, mat_file, method, wanted, unwanted):
    """It said 'Use buttons to mark Inhale Start and Exhale End' in every
    method, also where those buttons are not shown."""
    _ct_step_selected(win, mat_file, method)

    status = win.status_bar.currentMessage()

    assert wanted in status and unwanted not in status


def test_changing_the_method_replaces_a_status_line_about_the_old_one(win, mat_file):
    """'Added part: images 1-5' stayed on screen in the other methods."""
    _ct_step_selected(win, mat_file, "parts")
    win.part_from_spin.setValue(1)
    win.part_to_spin.setValue(3)
    win._add_part_from_spins()
    assert "part" in win.status_bar.currentMessage().lower()
    win._clear_parts()

    choose_ct_method(win, "landmarks")

    assert "part" not in win.status_bar.currentMessage().lower()
    assert "landmark" in win.status_bar.currentMessage().lower()


# --------------------------------------------------------- one word for a thing

def test_no_builder_word_reaches_the_user(win, mat_file):
    """'Fencepost' was the developers' word for a grid line."""
    texts = [text for _kind, text in gui_app._BANNERS.values()]
    texts.append(win.cycle_mode_btn.toolTip())
    _ct_step_selected(win, mat_file, "cycle")
    win.cycle_mode_btn.setChecked(True)
    texts += [win.status_bar.currentMessage(), win.mode_banner.text()]
    win._on_click(ClickEvent(win.ax_top, 3.3))
    texts.append(win.status_bar.currentMessage())

    assert not [t for t in texts if "fencepost" in t.lower()]


def test_a_cycle_click_says_which_image_it_took(win, mat_file):
    """A click takes the image that starts at the nearest grid line, which is
    not always the number under the pointer. The screen has to say which."""
    _ct_step_selected(win, mat_file, "cycle")
    win.cycle_mode_btn.setChecked(True)

    # 3.35 s lies in the slice numbered 7 (3.2 to 3.4 s), nearer its right
    # edge, where image 8 starts.
    win._on_click(ClickEvent(win.ax_top, 3.35))

    status = win.status_bar.currentMessage()
    assert win._ct_cycle_frames == [8]
    assert "image 8" in status and "grid line" in status
    assert "grid line" in win.mode_banner.text()
