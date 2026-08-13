"""Tests for CT synthetic-profile (tab 4) interaction rules.

Qt runs offscreen so no window appears. These pin two contracts that were
previously easy to break by hand:

1. In full-manual mode an unarmed plot click must not silently redefine the
   Start/End window, because the image grid is computed live from it.
2. Clearing must never abort part-way on a stale matplotlib artist, and the
   tab's "Clear All" must clear everything drawn, not just the parts list.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication

from labchart2cfd.gui.app import FlowProfileApp
from labchart2cfd.io.labchart import LabChartData
from labchart2cfd.processing.profile_assembly import ProfilePart


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def win(qapp):
    w = FlowProfileApp()
    # Put the window in the state a user is in right after "Plot Selected".
    w._plot_mode = "selected"
    w.ax_top = w.fig.add_subplot(211)
    w.ax_bottom = w.fig.add_subplot(212, sharex=w.ax_top)
    yield w
    w.close()


class _Event:
    """Minimal stand-in for a matplotlib MouseEvent."""

    def __init__(self, inaxes, xdata, button=1):
        self.inaxes = inaxes
        self.xdata = xdata
        self.button = button


def test_unarmed_click_in_manual_mode_is_ignored(win):
    """A stray click must not move the window the image grid is built from."""
    win._synth_mode = True
    win._capturing_part = False
    win.start_spin.setValue(10.0)
    win.end_spin.setValue(20.0)

    win._on_click(_Event(win.ax_top, 12.34))

    assert win._time_markers == []
    assert win.start_spin.value() == 10.0
    assert win.end_spin.value() == 20.0


def test_click_outside_manual_mode_still_marks(win):
    """Guard: normal time-marking behavior is unchanged elsewhere."""
    win._synth_mode = False
    win._capturing_part = False

    win._on_click(_Event(win.ax_top, 5.0))

    assert len(win._time_markers) == 1


def test_clear_time_markers_survives_stale_artist(win):
    """One detached artist must not abort the whole clear."""
    win._synth_mode = False
    win._on_click(_Event(win.ax_top, 1.0))
    win._on_click(_Event(win.ax_top, 2.0))
    assert len(win._time_markers) == 2

    # Simulate what fig.clear() leaves behind: an already-detached artist.
    win._time_markers[0][0].remove()

    win._clear_time_markers()

    assert win._time_markers == []


def _set_window(win, start, end, dt_ms=200, n_images=20):
    win.start_spin.setValue(start)
    win.end_spin.setValue(end)
    win.synth_dt_spin.setValue(float(dt_ms))
    win.synth_images_spin.setValue(n_images)   # independent input now
    win.synth_grid_offset_spin.setValue(0)


def test_grid_is_exactly_dt_times_image_count(win):
    """dt and the image count are independent inputs; the grid is their product."""
    _set_window(win, 0.0, 4.0)
    win.synth_images_spin.setValue(15)

    origin, dt, n_images = win._synth_grid()

    assert origin == 0.0
    assert dt == pytest.approx(0.2)
    assert n_images == 15
    assert n_images * dt == pytest.approx(3.0)   # not the 4 s window


def test_changing_image_count_leaves_dt_alone(win):
    _set_window(win, 0.0, 4.0)
    win.synth_dt_spin.setValue(200.0)

    win.synth_images_spin.setValue(15)
    assert win.synth_dt_spin.value() == pytest.approx(200.0)

    win.synth_images_spin.setValue(14)
    assert win.synth_dt_spin.value() == pytest.approx(200.0)


def test_changing_dt_leaves_image_count_alone(win):
    _set_window(win, 0.0, 4.0)
    win.synth_images_spin.setValue(15)

    win.synth_dt_spin.setValue(400.0)

    assert win.synth_images_spin.value() == 15
    assert win._synth_grid()[1] == pytest.approx(0.4)


def test_grid_may_extend_past_the_step_end(win):
    """End is reference only; the grid is allowed to run past it."""
    _set_window(win, 0.0, 1.0)          # 1 s step
    win.synth_dt_spin.setValue(200.0)
    win.synth_images_spin.setValue(15)  # 3 s of grid

    origin, dt, n_images = win._synth_grid()

    assert n_images == 15
    assert origin + n_images * dt == pytest.approx(3.0)


def test_grid_offset_moves_the_anchor_without_changing_the_count(win):
    _set_window(win, 0.0, 4.0)
    win.synth_images_spin.setValue(15)
    win.synth_grid_offset_spin.setValue(500)  # 0.5 s

    origin, dt, n_images = win._synth_grid()

    assert origin == pytest.approx(0.5)
    assert n_images == 15                      # unchanged by the offset
    assert origin + n_images * dt == pytest.approx(3.5)


def test_grid_offset_can_slide_the_grid_earlier(win):
    """The detected step edge can sit after the true first image, so the grid
    has to shift left as well as right."""
    _set_window(win, 5.0, 9.0)
    win.synth_images_spin.setValue(15)

    win.synth_grid_offset_spin.setValue(-300)   # 0.3 s earlier

    origin, dt, n_images = win._synth_grid()
    assert origin == pytest.approx(4.7)
    assert n_images == 15                        # count unaffected
    assert origin + n_images * dt == pytest.approx(7.7)


def test_dt_is_quantized_to_whole_flow_samples(win):
    """Slicing happens in whole samples, so the grid must use the same dt.

    Regression for 'Part 9..15 needs samples up to 300 but the window has 297',
    which came from drawing at a fractional dt and slicing at a rounded one.
    """
    _set_window(win, 0.0, 4.0)
    win.synth_dt_spin.setValue(198.6)   # not a whole number of 100 Hz samples

    _origin, dt, _n = win._synth_grid()

    assert dt == pytest.approx(0.20)    # snapped to 20 samples


def test_click_snaps_to_the_slice_it_landed_in(win):
    _set_window(win, 0.0, 4.0)

    # 0.75 s lies inside slice 4 = [0.6, 0.8), not slice 5.
    assert win._snap_synth(0.75) == 4
    assert win._snap_synth(0.60) == 4
    assert win._snap_synth(0.59) == 3


def test_window_offset_slides_start_and_end(win):
    _set_window(win, 1.0, 5.0)
    win.synth_window_offset_spin.setValue(250)

    win._apply_window_offset()

    assert win.start_spin.value() == pytest.approx(1.25)
    assert win.end_spin.value() == pytest.approx(5.25)
    assert win.synth_window_offset_spin.value() == 0  # consumed


def test_typing_a_number_recomputes_only_once_on_enter(win):
    """Typing "15" must not first recompute for 1 image.

    Qt's default keyboard tracking emits valueChanged per keystroke, which made
    the grid churn through intermediate values while the user was still typing.
    """
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest

    win.show()
    # The synth controls live on a tab that is disabled until CT is chosen,
    # and keystrokes only reach a realized, enabled widget.
    win.workflow_combo.setCurrentText("CT")
    QApplication.processEvents()

    spin = win.synth_images_spin
    seen = []
    spin.valueChanged.connect(seen.append)

    spin.setFocus()
    spin.lineEdit().selectAll()
    QTest.keyClicks(spin, "15")
    assert seen == []            # nothing committed while typing

    QTest.keyClick(spin, Qt.Key_Return)
    assert seen == [15]          # exactly one recompute, with the final value


def test_arrow_stepping_still_reacts_immediately(win):
    """Keyboard tracking off must not make the up/down arrows feel dead."""
    spin = win.synth_images_spin
    spin.setValue(5)
    seen = []
    spin.valueChanged.connect(seen.append)

    spin.stepBy(1)

    assert seen == [6]


def _load_fake_signal(win):
    """10 s recording whose flow is quiet only inside 4..6 s.

    Full-signal amplitude is 1.0 but only 0.25 inside the window, so a y-zoom
    that ignores the window is easy to tell apart from one that respects it.
    """
    t = np.linspace(0.0, 10.0, 10001)
    inside = (t >= 4.0) & (t <= 6.0)
    # Full amplitude everywhere outside, including right up against the window
    # edges, so a y-range taken over the padded span would be visibly wrong.
    flow = np.where(inside, 0.25 * np.sin(2 * np.pi * 0.5 * t), 1.0)
    trigger = np.zeros_like(t)
    trigger[inside] = 5.0

    win.data = LabChartData(
        data_cell=[[flow], [trigger]],
        time_cell=[[t], [t]],
        num_channels=2,
        num_blocks=1,
    )
    win.flow_row_spin.setValue(1)
    win.trigger_row_spin.setValue(2)
    win.col_spin.setValue(1)
    win.ax_top_trig = win.ax_top.twinx()


def test_zoom_to_window_pads_x_on_both_ends(win):
    _load_fake_signal(win)

    win._zoom_to_window(4.0, 6.0)

    lo, hi = win.ax_top.get_xlim()
    assert lo == pytest.approx(3.9)   # 5% of the 2 s span, left
    assert hi == pytest.approx(6.1)   # and right


def test_zoom_to_window_scales_y_to_data_inside_the_window(win):
    _load_fake_signal(win)

    win._zoom_to_window(4.0, 6.0)

    ylo, yhi = win.ax_top.get_ylim()
    # Inside the window |flow| <= 0.25, so the view must be far tighter than
    # the full-signal range of +/-1.0, and must still clear the peak.
    assert 0.25 < yhi < 0.6
    assert -0.6 < ylo < -0.25
    # The bottom flow axis follows the same limits.
    assert win.ax_bottom.get_ylim() == pytest.approx((ylo, yhi))


def test_zoom_to_window_scales_the_trigger_twin_axis_separately(win):
    _load_fake_signal(win)

    win._zoom_to_window(4.0, 6.0)

    tlo, thi = win.ax_top_trig.get_ylim()
    # The trigger is held high for the whole step, so its in-window range is
    # flat at 5.0; the flat-signal fallback keeps the axis from collapsing.
    assert tlo < 5.0 < thi
    assert thi < 7.0          # not left at some huge default
    # Scaled independently of the flow axis, which sits near zero.
    assert win.ax_top.get_ylim()[1] < 1.0


def test_zoom_to_window_ignores_an_empty_or_inverted_window(win):
    _load_fake_signal(win)
    win.ax_top.set_xlim(0.0, 10.0)

    win._zoom_to_window(6.0, 4.0)   # end before start

    assert win.ax_top.get_xlim() == pytest.approx((0.0, 10.0))


def _select_fake_step(win, start=0.0, end=2.0, dt_ms=200):
    win._ct_selected_step = {
        "index": 1, "start_time": start, "end_time": end, "duration": end - start,
    }
    win._ct_detected_start, win._ct_detected_end = start, end
    win.ct_temporal_res_spin.setValue(dt_ms)


def test_step_selection_computes_image_count_from_the_step(win):
    """Regression: the count spinbox's leftover value must not be taken as the
    image count, which silently pinned a freshly selected step to 1 image."""
    _select_fake_step(win, 0.0, 2.0, 200)
    assert win.ct_total_images_spin.value() == 1  # stale default

    win._draw_ct_intervals(update_total_images_field=True)

    assert win.ct_total_images_spin.value() == 10  # 2.0 s / 200 ms slices


def test_manual_frame_override_still_applies_during_a_shift_preview(win):
    _select_fake_step(win, 0.0, 2.0, 200)
    win.ct_total_images_spin.setValue(7)

    win._draw_ct_intervals(update_total_images_field=False)

    assert win.ct_total_images_spin.value() == 7  # untouched by the preview


def test_landmark_reports_where_the_cut_snaps(win):
    _select_fake_step(win, 0.0, 2.0, 200)

    text = win._landmark_snap_text(1.27)

    assert "image 7" in text     # 1-based; index 6 in the workflow metadata
    assert "1.200s" in text
    assert "70 ms" in text


def test_clear_synth_all_clears_parts_markers_and_grid(win):
    win._synth_mode = False
    win._on_click(_Event(win.ax_top, 3.0))
    win.start_spin.setValue(0.0)
    win.end_spin.setValue(2.0)
    win.ct_temporal_res_spin.setValue(200)
    win._draw_synth_grid()
    win._parts.append(ProfilePart(start_image=1, end_image=3, label=""))
    win._draw_parts_overlay()
    assert win._time_markers and win._grid_artists and win._parts

    win._clear_synth_all()

    assert win._parts == []
    assert win._time_markers == []
    assert win._grid_artists == []
    assert win._capture_marker is None
