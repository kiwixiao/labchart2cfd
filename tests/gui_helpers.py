"""Small helpers shared by the GUI tests (plain functions, not fixtures)."""


class ClickEvent:
    """Minimal stand-in for a matplotlib MouseEvent."""

    def __init__(self, inaxes, xdata=0.0, button=1):
        self.inaxes = inaxes
        self.xdata = xdata
        self.button = button


def choose_ct_method(win, key):
    """Pick a CT method in the chooser, as a user would: parts, landmarks, cycle."""
    win.ct_method_combo.setCurrentIndex(win.ct_method_combo.findData(key))


def open_block(win, mat_file, row=2, col=1):
    """Open a recording and click one overview tile, landing on the detail view."""
    win.open_recording(mat_file)
    win._on_overview_click(ClickEvent(win._overview_axes[(row, col)]))


def pick_drift_breath(win, start=2.0, end=4.0):
    """Type a drift correction breath into section 3 (one 2 s period of the
    fixture's 0.5 Hz sine by default). Without one, every CT export line
    carries an amber caution naming the image grid (design 2026-11-22, 4.4)."""
    win.drift_from_spin.setValue(start)
    win.drift_to_spin.setValue(end)
