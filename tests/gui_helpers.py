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
