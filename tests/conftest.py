"""Shared fixtures for the GUI tests. Qt runs offscreen so no window appears."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from scipy.io import savemat


@pytest.fixture(scope="session")
def qapp():
    from PyQt5.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture
def win(qapp):
    """A freshly opened main window with no recording loaded."""
    from labchart2cfd.gui.app import FlowProfileApp

    window = FlowProfileApp()
    yield window
    window.close()


@pytest.fixture
def no_modal_dialogs(monkeypatch):
    """Fail on any message box. A modal dialog would otherwise hang the run."""
    def _refuse(kind):
        def _raise(*args, **_kwargs):
            text = " | ".join(str(a) for a in args[1:3])
            raise AssertionError(f"unexpected {kind} dialog: {text}")
        return staticmethod(_raise)

    for kind in ("warning", "critical", "information", "question"):
        monkeypatch.setattr(f"labchart2cfd.gui.app.QMessageBox.{kind}", _refuse(kind))


@pytest.fixture
def mat_file(tmp_path):
    """A 4 channel, 3 block LabChart export, 10 s per block at 1000 Hz.

    Channel 1 "Trigger" holds one step from 2 s to 5 s. Channel 2 "Flow" is a
    0.5 Hz sine. Channel 3 has a blank title. Channel 4 is "Pressure". Block 3
    holds only channel 3, so the flow channel is empty there.
    """
    fs = 1000.0
    t = np.arange(0.0, 10.0, 1 / fs)
    trigger = np.zeros_like(t)
    trigger[(t >= 2.0) & (t < 5.0)] = 5.0
    flow = np.sin(2 * np.pi * 0.5 * t)
    spare = np.full_like(t, 0.1)
    pressure = 2.0 * np.cos(2 * np.pi * 0.5 * t)
    grid = [
        [trigger, trigger, None],
        [flow, 0.5 * flow, None],
        [None, spare, spare],
        [pressure, pressure, None],
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

    path = tmp_path / "rec.mat"
    savemat(str(path), {
        "data": np.concatenate(chunks),
        "datastart": start,
        "dataend": end,
        "samplerate": np.full((4, 3), fs),
        "titles": np.array(["Trigger ", "Flow    ", "        ", "Pressure"]),
    })
    return path
