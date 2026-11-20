"""Display reduction for long signals.

A LabChart recording can hold tens of millions of samples per channel, and a
plot is about a thousand pixels wide. Drawing every sample is slow and shows
nothing more than the vertical extent of each pixel column. Keeping every Nth
sample is fast but can skip a short pulse entirely.

``minmax_envelope`` keeps the minimum and the maximum of each group of
samples instead. The drawn line then looks like the full signal, no peak can
vanish, and every drawn point is a real sample. Only the picture is reduced:
clicks, zoom limits and exports always read the full arrays.
"""

import numpy as np

# Groups per view. Each contributes at most two points, so a line holds at
# most about twice this many, several per pixel column of a wide plot.
DEFAULT_MAX_BINS = 4000


def minmax_envelope(x, y, x_lo, x_hi, max_bins=DEFAULT_MAX_BINS):
    """Samples to draw for ``y`` over the view ``[x_lo, x_hi]``.

    Args:
        x: Sample positions, increasing.
        y: Sample values, same length as ``x``.
        x_lo, x_hi: The visible range.
        max_bins: Number of groups the visible samples are split into.

    Returns:
        ``(xd, yd)``, a subset of the samples in time order. When the view
        holds no more than ``2 * max_bins`` samples they are all returned, so
        zooming in ends at the raw data. Otherwise each group contributes its
        minimum and its maximum, and the first and last visible samples are
        kept so the line reaches both ends.
    """
    n_all = len(x)
    if n_all == 0:
        return x[:0], y[:0]

    # One sample of margin on each side, so the line runs off the plot edges
    # instead of stopping short of them.
    i0 = max(int(np.searchsorted(x, x_lo, side="left")) - 1, 0)
    i1 = min(int(np.searchsorted(x, x_hi, side="right")) + 1, n_all)
    n = i1 - i0
    if n <= 2 * max_bins:
        return x[i0:i1], y[i0:i1]

    size = -(-n // max_bins)                 # samples per group, rounded up
    full = (n // size) * size                # samples covered by whole groups
    body = y[i0:i0 + full].reshape(-1, size)
    base = i0 + np.arange(body.shape[0]) * size
    picks = [
        np.array([i0, i1 - 1]),              # both ends of the view
        base + body.argmin(axis=1),
        base + body.argmax(axis=1),
    ]
    if full < n:                             # the leftover tail is a group too
        tail = y[i0 + full:i1]
        picks.append(i0 + full + np.array([int(tail.argmin()), int(tail.argmax())]))

    # Sorted and without repeats: a group's minimum and maximum come out in
    # the order they occur, which is what a line through them needs.
    index = np.unique(np.concatenate(picks))
    return x[index], y[index]


class EnvelopeLine:
    """One plotted line that shows a long signal's envelope at the current zoom.

    It keeps the full arrays and redraws its matplotlib line from them when
    ``update`` is given a new visible range.
    """

    def __init__(self, ax, x, y, *plot_args, max_bins=DEFAULT_MAX_BINS,
                 **plot_kwargs) -> None:
        self.x = x
        self.y = y
        self.max_bins = max_bins
        lo, hi = (float(x[0]), float(x[-1])) if len(x) else (0.0, 0.0)
        xd, yd = minmax_envelope(x, y, lo, hi, max_bins)
        (self.line,) = ax.plot(xd, yd, *plot_args, **plot_kwargs)
        self._span = (lo, hi)

    def update(self, x_lo: float, x_hi: float) -> bool:
        """Redraw for a new visible range. Returns False if nothing changed."""
        if len(self.x) == 0:
            return False
        # A view wider than the signal shows the same thing as the signal's
        # own range, so both count as one span and are computed once.
        span = (max(float(x_lo), float(self.x[0])), min(float(x_hi), float(self.x[-1])))
        if span == self._span:
            return False
        self.line.set_data(*minmax_envelope(self.x, self.y, x_lo, x_hi, self.max_bins))
        self._span = span
        return True
