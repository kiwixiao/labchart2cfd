"""The composed breath with its frame order (design 2026-11-22, 4.5).

One drawing for the preview and for the picture saved with the export, so
the two cannot differ. Each frame is marked by thin lines at its edges, with
its new number above and its original image number below. Extra flow is
shaded and labelled with its length and "no image". Labels that would print
over each other are left out, the frame lines stay.
"""

from typing import List, Sequence, Tuple

import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy.integrate import trapezoid

from labchart2cfd.processing.profile_assembly import FrameRow
from labchart2cfd.processing.volume_balance import BalanceResult, mass_volume_to_ml

# Axes box: fixed before anything is drawn, so label widths can be measured
# in pixels without a renderer. The left margin is in inches: as a fraction of
# the width it held the flow scale's numbers but not the axis name on the
# 8 inch preview.
_LEFT_IN = 1.0
_AXES_BOTTOM, _AXES_HEIGHT, _AXES_RIGHT = 0.34, 0.54, 0.97
_LABEL_SIZE = 8
_NOTE_SIZE = 8
_EXTRA_COLOR = "#b9770e"

SIDE_NAMES = {"positive": "above zero", "negative": "below zero"}


def _text_px(text: str, fontsize: float, dpi: float) -> float:
    """Width of a short label in pixels, estimated from its font size."""
    return len(text) * fontsize * 0.62 * dpi / 72.0


def _place_row(ax, centers: Sequence[float], texts: Sequence[str], y: float,
               gid: str, px_per_s: float, dpi: float) -> None:
    """Write labels left to right, leaving out any that would touch the last."""
    last_right = -np.inf
    for x, text in zip(centers, texts):
        half = _text_px(text, _LABEL_SIZE, dpi) / 2.0 / px_per_s
        gap = 3.0 / px_per_s
        if x - half < last_right + gap:
            continue
        ax.text(x, y, text, transform=ax.get_xaxis_transform(), ha="center",
                va="center", fontsize=_LABEL_SIZE, gid=gid, parse_math=False)
        last_right = x + half


def draw_frame_order(fig: Figure, time: NDArray[np.float64], flow: NDArray[np.float64],
                     frames: List[FrameRow], extra_spans: List[Tuple[float, float]],
                     notes: List[str]) -> None:
    """Draw the composed breath, its frames and its notes on ``fig``.

    Args:
        fig: The figure to draw on; it is cleared first.
        time: Profile time of each sample (s).
        flow: The flow exactly as it will be written.
        frames: The frame table rows, in profile order.
        extra_spans: (start, end) in profile time of each extra flow.
        notes: Lines printed below the plot (drift, volumes, scale).
    """
    fig.clear()
    left = min(0.3, _LEFT_IN / fig.get_figwidth())
    ax = fig.add_axes((left, _AXES_BOTTOM, _AXES_RIGHT - left, _AXES_HEIGHT))
    end = max(float(time[-1]) if len(time) else 0.0,
              max((f.profile_end_s for f in frames), default=0.0),
              max((b for _a, b in extra_spans), default=0.0))
    ax.set_xlim(0.0, end if end > 0 else 1.0)
    px_per_s = ((_AXES_RIGHT - left) * fig.get_figwidth() * fig.dpi) / (ax.get_xlim()[1] - ax.get_xlim()[0])

    for a, b in extra_spans:
        ax.axvspan(a, b, color=_EXTRA_COLOR, alpha=0.22, linewidth=0, gid="extra")
        ax.text((a + b) / 2.0, 0.5, f"extra flow\n{b - a:.3f} s, no image",
                transform=ax.get_xaxis_transform(), ha="center", va="center",
                fontsize=_LABEL_SIZE, color=_EXTRA_COLOR, gid="extra_label",
                parse_math=False)
    edges = sorted({f.profile_start_s for f in frames} | {f.profile_end_s for f in frames})
    for x in edges:
        ax.axvline(x, color="0.55", linewidth=0.5, zorder=1)
    ax.plot(time, flow, color="#1f4e9c", linewidth=0.9, gid="flow", zorder=3)
    ax.axhline(0.0, color="0.4", linestyle="--", linewidth=0.5)
    # Headroom: the curve and zero stay between the two rows of numbers
    # (axes 0.05 and 0.95); the flow ran through the image numbers.
    if len(flow):
        low = min(float(np.min(flow)), 0.0)
        high = max(float(np.max(flow)), 0.0)
        pad = (high - low) * 0.12 / 0.76 if high > low else 1e-6
        ax.set_ylim(low - pad, high + pad)

    centers = [(f.profile_start_s + f.profile_end_s) / 2.0 for f in frames]
    _place_row(ax, centers, [str(f.new_frame) for f in frames], 0.95, "frame", px_per_s, fig.dpi)
    _place_row(ax, centers, [str(f.original_image) for f in frames], 0.05, "image", px_per_s, fig.dpi)

    ax.set_xlabel("Profile time (s)")
    ax.set_ylabel("Mass flow (kg/s)")
    # The rows are named in the title: names in the left margin were printed
    # over the top and bottom numbers of the flow scale.
    ax.set_title(f"Composed breath: {len(frames)} frames, {len(flow)} samples\n"
                 "Top row: frame number in the composed breath. "
                 "Bottom row: image number in the scan.", fontsize=9)
    ax.grid(True, alpha=0.25)
    # Below the axis label: the lines share the space left, at most 0.045 apart.
    step = min(0.045, 0.19 / max(1, len(notes)))
    for i, line in enumerate(notes):
        fig.text(0.02, 0.21 - i * step, line, fontsize=_NOTE_SIZE, ha="left",
                 va="center", parse_math=False)


def balance_notes(balance: BalanceResult, keep: str, time: NDArray[np.float64]) -> List[str]:
    """The lines that say which half was kept and how the other was scaled."""
    other = "negative" if keep == "positive" else "positive"
    kept_ml = mass_volume_to_ml(balance.kept_volume)
    scaled_ml = mass_volume_to_ml(balance.scaled_volume_before)
    above, below = (kept_ml, scaled_ml) if keep == "positive" else (scaled_ml, kept_ml)
    # Rounded first, then + 0.0, so a rounding residue prints as 0.000, not -0.000.
    net_ml = round(mass_volume_to_ml(float(trapezoid(balance.flow, time))), 3) + 0.0
    return [
        f"Complete half: {SIDE_NAMES[keep]}, kept as recorded. Before scaling: "
        f"above zero {abs(above):.1f} mL, below zero {abs(below):.1f} mL.",
        f"The half {SIDE_NAMES[other]} is scaled by {balance.factor:.4f}. "
        f"Net volume after scaling: {net_ml:.3f} mL.",
    ]
