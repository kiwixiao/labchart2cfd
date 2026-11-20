"""What the detail view shows: the y scales and LabChart's comments.

Two requests from the first try on a real recording. The trigger axis had
been fitted to the few millivolts of noise inside the selected step, which
hid the step itself. And the comments typed in LabChart were not shown.
"""
import numpy as np
import pytest
from matplotlib.text import Text
from scipy.io import loadmat, savemat

from tests.gui_helpers import ClickEvent, open_block

pytestmark = pytest.mark.usefixtures("no_modal_dialogs")


class ScrollEvent:
    """Minimal stand-in for a matplotlib scroll event (pixel position)."""

    def __init__(self, x, y, step):
        self.x, self.y, self.step = x, y, step


def _span(limits):
    return limits[1] - limits[0]


def _strip_point(win, ax, side):
    """A pixel beside an axes: on its left or right tick labels."""
    box = ax.bbox
    x = box.x0 - 15 if side == "left" else box.x1 + 15
    return x, (box.y0 + box.y1) / 2


# ------------------------------------------------------- the trigger scale

def test_selecting_a_step_keeps_the_whole_trigger_range_in_view(win, mat_file):
    """The step runs from 0 to 5. Fitted to the step window the axis showed
    4.5 to 5.5, a flat line, and on a real recording 2 mV of noise."""
    win.workflow_combo.setCurrentText("CT")
    open_block(win, mat_file)

    win._select_ct_step()                          # zooms to the step

    lo, hi = win.ax_top_trig.get_ylim()
    assert lo < 0.0 and hi > 5.0


def test_zoom_to_window_leaves_the_trigger_scale_alone(win, mat_file):
    open_block(win, mat_file)
    before = win.ax_top_trig.get_ylim()
    win.start_spin.setValue(6.0)                   # trigger is flat 0 here
    win.end_spin.setValue(8.0)

    win._zoom_to_window()

    assert win.ax_top_trig.get_ylim() == before
    assert before[0] < 0.0 and before[1] > 5.0


def test_zoom_to_window_still_fits_the_flow_scale(win, mat_file):
    open_block(win, mat_file)
    win.start_spin.setValue(2.2)                   # flow rises 0.64 to 1 to 0.81
    win.end_spin.setValue(2.8)

    win._zoom_to_window()

    lo, hi = win.ax_top.get_ylim()
    assert 0.5 < lo < 0.65 and 1.0 < hi < 1.1


# ------------------------------------------------------- y scale controls

def test_trigger_scale_buttons_zoom_about_the_middle(win, mat_file):
    open_block(win, mat_file)
    full = win.ax_top_trig.get_ylim()
    middle = sum(full) / 2

    win.trig_y_in_btn.click()
    zoomed = win.ax_top_trig.get_ylim()
    assert _span(zoomed) == pytest.approx(_span(full) / 2)
    assert sum(zoomed) / 2 == pytest.approx(middle)

    win.trig_y_out_btn.click()
    win.trig_y_out_btn.click()
    assert _span(win.ax_top_trig.get_ylim()) == pytest.approx(_span(full) * 2)

    win.trig_y_fit_btn.click()
    assert win.ax_top_trig.get_ylim() == pytest.approx(full)


def test_flow_scale_buttons_move_both_flow_plots_together(win, mat_file):
    open_block(win, mat_file)
    full = win.ax_top.get_ylim()

    win.flow_y_out_btn.click()

    assert _span(win.ax_top.get_ylim()) == pytest.approx(_span(full) * 2)
    assert win.ax_bottom.get_ylim() == pytest.approx(win.ax_top.get_ylim())


def test_flow_fit_uses_the_data_in_view(win, mat_file):
    open_block(win, mat_file)
    win.ax_top.set_xlim(2.2, 2.8)
    win.flow_y_out_btn.click()

    win.flow_y_fit_btn.click()

    lo, hi = win.ax_top.get_ylim()
    assert 0.5 < lo < 0.65 and 1.0 < hi < 1.1
    assert win.ax_bottom.get_ylim() == pytest.approx((lo, hi))


def test_scale_buttons_are_off_until_a_block_is_open(win, mat_file):
    buttons = (win.flow_y_in_btn, win.flow_y_out_btn, win.flow_y_fit_btn,
               win.trig_y_in_btn, win.trig_y_out_btn, win.trig_y_fit_btn)
    assert not any(b.isEnabled() for b in buttons)

    win.open_recording(mat_file)                   # overview
    assert not any(b.isEnabled() for b in buttons)

    win._on_overview_click(ClickEvent(win._overview_axes[(2, 1)]))
    assert all(b.isEnabled() for b in buttons)


def test_trigger_scale_buttons_are_off_without_a_trigger(win, mat_file):
    win.trigger_row_spin.setValue(0)
    open_block(win, mat_file)

    assert win.flow_y_in_btn.isEnabled()
    assert not win.trig_y_in_btn.isEnabled()
    win._scale_y("trigger", 0.5)                   # and asking anyway is harmless


def test_the_wheel_beside_the_plot_zooms_that_scale(win, mat_file):
    """Over the numbers of a y axis, the wheel stretches that axis, as in
    LabChart. Left of the plot is flow, right of the top plot is trigger."""
    open_block(win, mat_file)
    win.canvas.draw()
    flow, trig = win.ax_top.get_ylim(), win.ax_top_trig.get_ylim()

    win._on_scroll(ScrollEvent(*_strip_point(win, win.ax_top, "right"), step=1))
    assert _span(win.ax_top_trig.get_ylim()) < _span(trig)
    assert win.ax_top.get_ylim() == flow

    win._on_scroll(ScrollEvent(*_strip_point(win, win.ax_bottom, "left"), step=-1))
    assert _span(win.ax_top.get_ylim()) > _span(flow)
    assert win.ax_bottom.get_ylim() == pytest.approx(win.ax_top.get_ylim())


def test_the_wheel_zooms_about_the_value_under_the_pointer(win, mat_file):
    open_block(win, mat_file)
    win.canvas.draw()
    x, y = _strip_point(win, win.ax_top, "right")
    value = win.ax_top_trig.transData.inverted().transform((x, y))[1]

    win._on_scroll(ScrollEvent(x, y, step=3))

    assert win.ax_top_trig.transData.inverted().transform((x, y))[1] == pytest.approx(value)


def test_the_wheel_inside_the_plot_changes_no_scale(win, mat_file):
    open_block(win, mat_file)
    win.canvas.draw()
    box = win.ax_top.bbox
    before = (win.ax_top.get_ylim(), win.ax_top_trig.get_ylim(), win.ax_top.get_xlim())

    win._on_scroll(ScrollEvent((box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2, step=2))

    assert (win.ax_top.get_ylim(), win.ax_top_trig.get_ylim(), win.ax_top.get_xlim()) == before


def test_the_wheel_on_the_overview_does_nothing(win, mat_file):
    win.open_recording(mat_file)

    win._on_scroll(ScrollEvent(5, 200, step=1))    # must not raise


# ------------------------------------------------------------- comments

@pytest.fixture
def commented(mat_file, tmp_path):
    """The fixture recording with three LabChart comments."""
    mat = {k: v for k, v in loadmat(str(mat_file)).items() if not k.startswith("__")}
    mat["com"] = np.array([
        [-1, 1, 2000, 1, 1],
        [4, 1, 5000, 1, 2],                        # on Pressure, not plotted
        [2, 1, 7000, 1, 3],                        # on Flow, plotted
        [-1, 2, 1000, 1, 1],
    ], dtype=float)
    mat["comtext"] = np.array(["CT scan start", "mask moved   ", "cough        "])
    mat["tickrate"] = np.full((1, 3), 1000.0)
    path = tmp_path / "commented.mat"
    savemat(str(path), mat)
    return path


def _comment_texts(win):
    return [a.get_text().strip() for a in win._comment_artists if isinstance(a, Text)]


def _comment_times(win):
    return sorted({float(a.get_xdata()[0]) for a in win._comment_artists
                   if not isinstance(a, Text)})


def test_the_comments_of_the_open_block_are_drawn(win, commented):
    open_block(win, commented)

    assert _comment_times(win) == [2.0, 5.0, 7.0]
    texts = _comment_texts(win)
    assert texts[0] == "CT scan start"
    assert texts[2] == "cough"                     # on a plotted channel: as typed


def test_a_comment_on_a_channel_not_plotted_names_that_channel(win, commented):
    open_block(win, commented)

    assert _comment_texts(win)[1].startswith("mask moved")
    assert "Pressure" in _comment_texts(win)[1]


def test_another_block_shows_its_own_comments(win, commented):
    open_block(win, commented, col=2)

    assert _comment_times(win) == [1.0]
    assert _comment_texts(win) == ["CT scan start"]


def test_a_recording_without_comments_draws_none(win, mat_file):
    open_block(win, mat_file)

    assert win._comment_artists == []
    assert not win.comments_action.isEnabled()


def test_comments_can_be_hidden_and_stay_hidden(win, commented):
    open_block(win, commented)
    assert win.comments_action.isChecked() and win.comments_action.isEnabled()
    assert all(a.get_visible() for a in win._comment_artists)

    win.comments_action.trigger()
    assert not any(a.get_visible() for a in win._comment_artists)

    win._plot_selected()                           # a redraw keeps the choice
    assert win._comment_artists and not any(a.get_visible() for a in win._comment_artists)

    win.comments_action.trigger()
    assert all(a.get_visible() for a in win._comment_artists)


def test_comments_do_not_change_the_scales(win, commented, mat_file):
    def _scales():
        return (*win.ax_top.get_xlim(), *win.ax_top.get_ylim(),
                *win.ax_top_trig.get_ylim())

    open_block(win, commented)
    with_comments = _scales()

    open_block(win, mat_file)

    assert _scales() == pytest.approx(with_comments)


def test_the_status_line_counts_the_comments(win, commented):
    open_block(win, commented)

    assert "3 comments" in win.status_bar.currentMessage()


def test_a_note_about_unreadable_comments_is_shown_under_the_file_name(
        win, mat_file, tmp_path):
    mat = {k: v for k, v in loadmat(str(mat_file)).items() if not k.startswith("__")}
    mat["com"] = np.array([[-1.0, 1, 2000, 1, 1]])     # no tick rate, no text
    path = tmp_path / "broken_comments.mat"
    savemat(str(path), mat)

    win.open_recording(path)

    assert "comment" in win.load_notes.text().lower()
    assert win.data is not None
