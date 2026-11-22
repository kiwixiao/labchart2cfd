"""The frame order drawing shared by the preview and the saved picture."""

import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from labchart2cfd.gui.frame_order_plot import draw_frame_order
from labchart2cfd.processing.profile_assembly import FrameRow


def _breath(n_frames=16, dt=0.1, rate=100, amplitude=0.003):
    time = np.arange(int(n_frames * dt * rate)) / rate
    flow = amplitude * np.sin(2 * np.pi * time / time[-1])
    frames = [FrameRow(k + 1, (k + 3) % 15 + 1, 2.0 + k * dt, k * dt, (k + 1) * dt, 0.0, None)
              for k in range(n_frames)]
    return time, flow, frames


def _texts(fig, renderer):
    """Every text drawn: notes, title, axis labels, the labels in the axes
    and the tick labels in view."""
    ax = fig.axes[0]
    texts = list(fig.texts) + list(ax.texts) + [ax.title, ax.xaxis.label, ax.yaxis.label]
    low, high = sorted(ax.get_ylim())
    texts += [label for value, label in zip(ax.get_yticks(), ax.get_yticklabels())
              if low <= value <= high]
    left, right = sorted(ax.get_xlim())
    texts += [label for value, label in zip(ax.get_xticks(), ax.get_xticklabels())
              if left <= value <= right]
    return [(t.get_text(), t.get_window_extent(renderer))
            for t in texts if t.get_visible() and t.get_text()]


@pytest.mark.parametrize("size, dpi, n_notes, amplitude", [
    ((8, 4.4), 100, 4, 0.003), ((10, 5.5), 150, 4, 0.003), ((8, 4.4), 100, 8, 0.003),
    ((8, 4.4), 100, 4, 0.0004), ((10, 5.5), 150, 4, 0.0004)])
def test_no_text_in_the_frame_order_picture_overlaps_another(size, dpi, n_notes, amplitude):
    """The row names 'frame' and 'image' were printed over the top and bottom
    numbers of the flow scale, and the flow axis name fell off the preview.
    Sizes: the preview and the saved picture; 0.0004 kg/s is an ENT005 flow."""
    time, flow, frames = _breath(amplitude=amplitude)
    fig = Figure(figsize=size, dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    notes = [f"Note line {i}: drift, extra flow or balance text." for i in range(n_notes)]

    draw_frame_order(fig, time, flow, frames, [], notes)
    canvas.draw()

    boxes = _texts(fig, canvas.get_renderer())
    overlaps = [(a, b) for i, (a, box_a) in enumerate(boxes)
                for b, box_b in boxes[i + 1:] if box_a.overlaps(box_b)]
    assert overlaps == []
    # Nothing falls off the figure.
    off = [t for t, box in boxes if box.x0 < 0 or box.y0 < 0
           or box.x1 > fig.bbox.width or box.y1 > fig.bbox.height]
    assert off == []


@pytest.mark.parametrize("offset", [0.0, -0.002, 0.004])
def test_the_curve_stays_between_the_two_rows_of_numbers(offset):
    """The flow ran through the image numbers at the bottom of the plot."""
    time, flow, frames = _breath()
    flow = flow + offset
    fig = Figure(figsize=(8, 4.4), dpi=100)
    FigureCanvasAgg(fig)

    draw_frame_order(fig, time, flow, frames, [], [])

    low, high = fig.axes[0].get_ylim()
    span = high - low
    assert (flow.min() - low) / span >= 0.1 and (high - flow.max()) / span >= 0.1
