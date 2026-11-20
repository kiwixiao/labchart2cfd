"""Walk the GUI on the real display and audit it.

The offscreen screenshots (gui_screenshots.py) use Qt's own fonts and a 1x
screen. This run uses the machine's platform plugin, fonts, style metrics and
pixel ratio, where a control can clip that fitted offscreen. The first run
found three defects no offscreen run had shown: the plot toolbar squeezed at
the smallest window size, the plot title cut off, and marker times printed
into each other.

It walks every study type, saves a picture of each state, and lists controls
whose text needs more room than they have. Synthetic recording only, never
patient data. A window is on screen for a few seconds.

Usage:
    python scripts/gui_native_check.py [output_dir]
"""
import os
import sys
import time
from pathlib import Path

import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix="labchart2cfd_native_"))
out.mkdir(parents=True, exist_ok=True)

from PyQt5.QtWidgets import (QAbstractButton, QApplication, QComboBox, QLabel,
                             QLineEdit, QMessageBox)

import labchart2cfd.gui.app as gui_app
# gui_screenshots asks for the offscreen platform when none is named. This
# check is about the real one, so that default is taken back.
_platform_named = "QT_QPA_PLATFORM" in os.environ
from gui_screenshots import _Click, write_synthetic_recording  # noqa: E402
if not _platform_named:
    os.environ.pop("QT_QPA_PLATFORM", None)
from labchart2cfd.gui.app import FlowProfileApp, install_exception_hook
from labchart2cfd.gui.theme import apply_theme

for kind in ("warning", "critical", "information", "question"):
    setattr(gui_app.QMessageBox, kind, staticmethod(
        lambda _p, title, text, *a, **k: (print(f"  [dialog] {title}: {text[:90]!r}"), QMessageBox.Yes)[1]))

app = QApplication(sys.argv[:1])
install_exception_hook()
apply_theme(app)
if app.platformName() == "offscreen":
    sys.exit("This check needs a real display. For offscreen pictures use gui_screenshots.py.")
w = FlowProfileApp()
w.show()
screen = app.primaryScreen()
print(f"platform {app.platformName()} | style {app.style().objectName()} | pixel ratio {w.devicePixelRatioF()} | "
      f"screen {screen.availableGeometry().width()}x{screen.availableGeometry().height()} | font {app.font().family()} {app.font().pointSizeF()}")


def settle(n=6):
    for _ in range(n):
        app.processEvents()
        time.sleep(0.02)


def audit(state):
    """Controls whose text needs more room than they have, in this state."""
    settle()
    clipped = []
    for widget in w.findChildren((QLabel, QAbstractButton, QComboBox, QLineEdit)):
        if not widget.isVisibleTo(w) or widget.window() is not w:
            continue
        if isinstance(widget, QLabel) and widget.wordWrap():
            need = widget.heightForWidth(widget.width())
            if need > widget.height() + 1:
                clipped.append(f"wrapped label needs {need}px high, has {widget.height()}: {widget.text()[:50]!r}")
            continue
        if type(widget).__name__ == "ElidedLabel":
            continue  # shortens its own text by design
        if widget.minimumWidth() == widget.maximumWidth():
            continue  # a fixed width was chosen on purpose (the small scale buttons)
        need = widget.minimumSizeHint().width()
        if need > widget.width() + 1:
            text = widget.currentText() if isinstance(widget, QComboBox) else widget.text()
            clipped.append(f"{type(widget).__name__} needs {need}px wide, has {widget.width()}: {text[:50]!r}")
    bar = w.sidebar_scroll.verticalScrollBar()
    print(f"[{state}] window {w.width()}x{w.height()} | sidebar {w.sidebar.width()}px | plot canvas {w.canvas.width()}x{w.canvas.height()} "
          f"| sidebar scroll range {bar.maximum()} | clipped controls: {len(clipped)}")
    for line in clipped:
        print("     ", line)
    w.canvas.draw()
    settle(2)
    w.grab().save(str(out / f"native_{state}.png"))


rec = out / "synthetic_demo.mat"
write_synthetic_recording(rec)
audit("1_empty")
w.open_recording(rec)
audit("2_overview")
w._on_overview_click(_Click(w._overview_axes[(2, 1)]))
w._on_click(_Click(w.ax_bottom, 21.0))
w._on_click(_Click(w.ax_bottom, 24.3))
w.subject_edit.setText("DEMO001")
audit("3_mri")
for study in ("CPAP", "Phase contrast (xenon)"):
    w.workflow_combo.setCurrentText(study)
    audit("4_" + study.split()[0].lower())
w.workflow_combo.setCurrentText("CT")
settle(10)
w._select_ct_step()
audit("5_ct_parts")
for key in ("landmarks", "cycle"):
    w.ct_method_combo.setCurrentIndex(w.ct_method_combo.findData(key))
    audit("6_ct_" + key)
# the smallest window the application allows
w.resize(w.minimumSize())
audit("7_minimum_size")
tb = w.toolbar
print(f"toolbar width {tb.width()} (hint {tb.sizeHint().width()}) | y buttons right edge {w.trig_y_fit_btn.mapTo(w, w.trig_y_fit_btn.rect().topRight()).x()} of window {w.width()}")
w._show_quick_guide()
settle()
g = w._guide_dialog
print(f"guide dialog {g.width()}x{g.height()} | text needs {g.guide_label.heightForWidth(g.guide_label.width())} | viewport {g.guide_scroll.viewport().height()} | scroll {g.guide_scroll.verticalScrollBar().maximum()}")
g.grab().save(str(out / "native_8_guide.png"))
g.close()
w.close()
print("pictures in", out)
