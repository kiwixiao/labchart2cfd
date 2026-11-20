"""Tests for the display reduction in ``labchart2cfd.gui.decimate``.

Real recordings hold tens of millions of samples per channel. The plot draws
an envelope (the minimum and maximum of each group of samples) instead of
every sample, so these tests pin the one promise that makes that safe: what is
drawn never hides a peak and never invents a value.
"""
import numpy as np
import pytest

from labchart2cfd.gui.decimate import minmax_envelope


def _signal(n=1_000_000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.arange(n) / 1000.0
    y = np.sin(2 * np.pi * 0.3 * x) + 0.05 * rng.standard_normal(n)
    return x, y


def test_a_short_signal_is_drawn_as_it_is():
    x = np.linspace(0.0, 1.0, 500)
    y = np.sin(x)

    xd, yd = minmax_envelope(x, y, 0.0, 1.0, max_bins=1000)

    assert np.array_equal(xd, x)
    assert np.array_equal(yd, y)


def test_a_long_signal_is_reduced_to_a_bounded_number_of_points():
    x, y = _signal()

    xd, yd = minmax_envelope(x, y, x[0], x[-1], max_bins=2000)

    assert len(xd) == len(yd)
    assert len(xd) <= 2 * 2000 + 4      # two per group, plus the two ends


def test_the_extremes_of_the_visible_range_survive():
    x, y = _signal()

    _xd, yd = minmax_envelope(x, y, x[0], x[-1], max_bins=2000)

    assert yd.max() == y.max()
    assert yd.min() == y.min()


def test_a_single_sample_spike_is_never_lost():
    """Keeping every Nth sample dropped short scan steps from the overview."""
    x = np.arange(5_000_000) / 40000.0
    y = np.zeros_like(x)
    y[3_141_592] = 5.0                   # one sample wide

    xd, yd = minmax_envelope(x, y, x[0], x[-1], max_bins=250)

    assert yd.max() == 5.0
    assert xd[np.argmax(yd)] == x[3_141_592]


def test_every_drawn_point_is_a_real_sample_in_time_order():
    x, y = _signal(200_000)

    xd, yd = minmax_envelope(x, y, x[0], x[-1], max_bins=500)

    assert np.all(np.diff(xd) > 0)
    index = np.searchsorted(x, xd)
    assert np.array_equal(x[index], xd)
    assert np.array_equal(y[index], yd)


def test_the_drawn_line_reaches_both_ends_of_the_signal():
    x, y = _signal(200_000)

    xd, _yd = minmax_envelope(x, y, x[0], x[-1], max_bins=500)

    assert xd[0] == x[0]
    assert xd[-1] == x[-1]


def test_only_the_visible_range_is_reduced():
    """Zooming in must bring back detail, down to the raw samples."""
    x, y = _signal()

    xd, yd = minmax_envelope(x, y, 100.0, 101.0, max_bins=2000)   # 1000 samples

    assert xd[0] <= 100.0 and xd[-1] >= 101.0     # one sample of margin each side
    inside = (x >= xd[0]) & (x <= xd[-1])
    assert np.array_equal(xd, x[inside])          # raw, nothing dropped
    assert np.array_equal(yd, y[inside])


def test_a_view_outside_the_signal_draws_nothing_and_does_not_fail():
    x, y = _signal(10_000)

    xd, yd = minmax_envelope(x, y, 500.0, 600.0, max_bins=100)

    assert len(xd) == len(yd)
    assert len(xd) <= 2


def test_an_empty_signal_is_handled():
    xd, yd = minmax_envelope(np.array([]), np.array([]), 0.0, 1.0)

    assert len(xd) == 0 and len(yd) == 0


@pytest.mark.parametrize("n", [4001, 4003, 9999, 100_003])
def test_lengths_that_do_not_divide_evenly_keep_their_tail(n):
    x = np.arange(n, dtype=float)
    y = np.zeros(n)
    y[-1] = 7.0                          # the peak sits in the leftover tail

    xd, yd = minmax_envelope(x, y, x[0], x[-1], max_bins=1000)

    assert yd.max() == 7.0
    assert xd[-1] == x[-1]


# ------------------------------------------------------------ in the window

pytestmark_gui = pytest.mark.usefixtures("no_modal_dialogs")


def _load_long_recording(win, n=2_000_000, fs=10000.0):
    """200 s at 10 kHz: a flow sine and a trigger with one 20 sample step."""
    from labchart2cfd.io.labchart import LabChartData

    t = np.arange(n) / fs
    flow = np.sin(2 * np.pi * 0.25 * t)
    trigger = np.zeros(n)
    trigger[1_234_500:1_234_520] = 5.0          # 2 ms wide
    win.data = LabChartData(
        data_cell=[[trigger], [flow]], time_cell=[[t], [t]],
        num_channels=2, num_blocks=1, titles=["Trigger", "Flow"],
    )
    win.flow_row_spin.setValue(2)
    win.trigger_row_spin.setValue(1)
    win.pressure_row_spin.setValue(0)
    win.col_spin.setValue(1)
    return t, flow, trigger


@pytestmark_gui
def test_detail_view_draws_a_bounded_number_of_points(win):
    """Every sample used to be drawn, three times, on every redraw."""
    _load_long_recording(win)

    win._plot_selected()

    drawn = [len(line.get_xdata()) for ax in (win.ax_top, win.ax_bottom, win.ax_top_trig)
             for line in ax.get_lines()]
    assert max(drawn) <= 2 * 4000 + 4
    assert len(win._envelopes) == 3              # flow twice, trigger once


@pytestmark_gui
def test_the_full_view_still_shows_the_true_extremes(win):
    _t, flow, trigger = _load_long_recording(win)

    win._plot_selected()

    flow_line, trigger_line = win._envelopes[0].line, win._envelopes[1].line
    assert flow_line.get_ydata().max() == flow.max()
    assert flow_line.get_ydata().min() == flow.min()
    assert trigger_line.get_ydata().max() == trigger.max()    # the 2 ms step


@pytestmark_gui
def test_zooming_in_brings_back_the_raw_samples_on_every_axis(win):
    t, flow, _trigger = _load_long_recording(win)
    win._plot_selected()

    win.ax_bottom.set_xlim(50.0, 50.1)           # 1000 samples; x is shared

    for envelope in win._envelopes:
        xd = envelope.line.get_xdata()
        assert 1000 <= len(xd) <= 1004
        assert xd[0] <= 50.0 and xd[-1] >= 50.1
    inside = (t >= win._envelopes[0].line.get_xdata()[0]) & (
        t <= win._envelopes[0].line.get_xdata()[-1])
    assert np.array_equal(win._envelopes[0].line.get_ydata(), flow[inside])


@pytestmark_gui
def test_zooming_back_out_reduces_again(win):
    _load_long_recording(win)
    win._plot_selected()
    win.ax_top.set_xlim(50.0, 50.1)

    win.ax_top.set_xlim(0.0, 200.0)

    assert len(win._envelopes[0].line.get_xdata()) <= 2 * 4000 + 4


@pytestmark_gui
def test_overview_thumbnail_keeps_a_short_trigger_step(win):
    """Every Nth sample hid short scan steps on the overview of a real file."""
    _t, _flow, trigger = _load_long_recording(win)

    win._show_overview()

    tile = win._overview_axes[(1, 1)]
    assert tile.get_lines()[0].get_ydata().max() == trigger.max()
    assert len(tile.get_lines()[0].get_xdata()) <= 2 * 250 + 4
