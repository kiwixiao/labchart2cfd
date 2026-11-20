"""Render the GUI offscreen in its main states and save PNG screenshots.

A way to look at the window without a display, for checking a layout change.
Tests cannot see a clipped or crowded control; a picture can. Uses a synthetic
recording only, never patient data.

Usage:
    python scripts/gui_screenshots.py [output_dir]

Without an output directory the pictures go to a new temporary directory,
whose path is printed.
"""
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5.QtWidgets import QApplication
from scipy.io import savemat

from labchart2cfd.gui.app import FlowProfileApp
from labchart2cfd.gui.theme import apply_theme


class _Click:
    """Minimal stand-in for a matplotlib MouseEvent."""

    def __init__(self, inaxes, xdata=0.0, button=1):
        self.inaxes = inaxes
        self.xdata = xdata
        self.button = button


def write_synthetic_recording(path: Path) -> None:
    """A 4 channel, 3 block LabChart export: 60 s per block at 1000 Hz."""
    fs = 1000.0
    t = np.arange(0.0, 60.0, 1 / fs)
    rng = np.random.default_rng(0)
    flow = 0.4 * np.sin(2 * np.pi * 0.3 * t) + 0.01 * rng.standard_normal(t.size)
    trigger = np.zeros_like(t)
    trigger[(t >= 20.0) & (t < 24.0)] = 5.0
    trigger[(t >= 40.0) & (t < 44.0)] = 5.0
    spare = 0.05 * rng.standard_normal(t.size)
    pressure = 3.0 * np.cos(2 * np.pi * 0.3 * t)
    grid = [
        [trigger, trigger, None],
        [flow, 0.6 * flow, None],
        [None, spare, spare],
        [pressure, 0.8 * pressure, None],
    ]

    chunks = []
    start = np.full((4, 3), -1.0)
    end = np.full((4, 3), -1.0)
    position = 1  # LabChart indices are 1-based and inclusive
    for ch, row in enumerate(grid):
        for bl, signal in enumerate(row):
            if signal is None:
                continue
            chunks.append(signal)
            start[ch, bl] = position
            end[ch, bl] = position + len(signal) - 1
            position += len(signal)
    savemat(str(path), {
        "data": np.concatenate(chunks),
        "datastart": start,
        "dataend": end,
        "samplerate": np.full((4, 3), fs),
        "titles": np.array(["Trigger ", "Flow    ", "        ", "Pressure"]),
    })


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix="labchart2cfd_gui_"))
    out.mkdir(parents=True, exist_ok=True)
    recording = out / "synthetic_demo.mat"
    write_synthetic_recording(recording)

    app = QApplication.instance() or QApplication([])
    apply_theme(app)
    window = FlowProfileApp()
    window.resize(1400, 860)
    window.show()

    def settle() -> None:
        for _ in range(5):
            app.processEvents()

    def shot(name: str, widget=None) -> None:
        settle()
        window.canvas.draw()
        settle()
        (widget or window).grab().save(str(out / f"{name}.png"))
        print("saved", out / f"{name}.png")

    shot("1_empty")

    window.open_recording(recording)
    shot("2_overview")

    window._on_overview_click(_Click(window._overview_axes[(2, 1)]))
    window._on_click(_Click(window.ax_bottom, 21.0))
    window._on_click(_Click(window.ax_bottom, 24.3))
    window.subject_edit.setText("DEMO001")
    shot("3_detail_mri")

    window.toolbar.zoom()
    shot("4_zoom_tool_on")
    window.toolbar.zoom()

    # A real export, to capture the result panel.
    window._set_output_dir(out / "export_demo")
    window._export()
    deadline = time.time() + 30
    while not window.execute_btn.isEnabled() and time.time() < deadline:
        app.processEvents()
        time.sleep(0.01)
    settle()
    if window._result_dialog is not None:
        shot("5_export_result", window._result_dialog)
        window._result_dialog.close()

    window.workflow_combo.setCurrentText("CT")
    window._plot_selected()
    window._select_ct_step()
    for first, last in ((6, 20), (1, 5)):
        window.part_from_spin.setValue(first)
        window.part_to_spin.setValue(last)
        window._add_part_from_spins()
    shot("6_ct_parts")

    for name, key in (("7_ct_landmarks", "landmarks"), ("8_ct_cycle", "cycle")):
        window.ct_method_combo.setCurrentIndex(window.ct_method_combo.findData(key))
        shot(name)

    window.close()
    print("screenshots in", out)


if __name__ == "__main__":
    main()
