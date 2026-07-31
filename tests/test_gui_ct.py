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

import pytest
from PyQt5.QtWidgets import QApplication

from labchart2cfd.gui.app import FlowProfileApp
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


def _set_window(win, start, end, dt_ms=200):
    win.start_spin.setValue(start)
    win.end_spin.setValue(end)
    win.synth_dt_spin.setValue(float(dt_ms))
    win.synth_grid_offset_spin.setValue(0)


def test_grid_counts_slices_not_lines(win):
    """A 4 s window at 200 ms holds 20 images, not 21 (the old fencepost count)."""
    _set_window(win, 0.0, 4.0)

    origin, dt, n_images = win._synth_grid()

    assert origin == 0.0
    assert dt == pytest.approx(0.2)
    assert n_images == 20


def test_grid_offset_moves_origin_and_shrinks_count(win):
    _set_window(win, 0.0, 4.0)
    win.synth_grid_offset_spin.setValue(500)  # 0.5 s

    origin, _dt, n_images = win._synth_grid()

    assert origin == pytest.approx(0.5)
    assert n_images == 17  # 3.5 s / 0.2


def test_editing_image_count_derives_dt(win):
    """Image count is ground truth, so dt follows from it."""
    _set_window(win, 0.0, 4.0)

    win.synth_images_spin.setValue(16)

    assert win.synth_dt_spin.value() == pytest.approx(250.0)
    assert win._synth_grid()[2] == 16  # round-trips back to the count you set


def test_editing_dt_derives_image_count(win):
    _set_window(win, 0.0, 4.0)

    win.synth_dt_spin.setValue(400.0)

    assert win._synth_grid()[2] == 10


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
