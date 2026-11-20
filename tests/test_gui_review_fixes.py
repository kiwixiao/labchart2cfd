"""Tests for the defects an independent review of the redesigned GUI proved.

Each one came with a reproduction against the running window. Most share one
shape: the screen said one thing and the program did another, or work was
dropped without a word.
"""
import shutil
import sys
import warnings

import pytest
from PyQt5.QtCore import QPoint, QPointF, Qt
from PyQt5.QtGui import QWheelEvent
from PyQt5.QtWidgets import QApplication, QMessageBox

from labchart2cfd.processing.profile_assembly import ProfilePart
from tests.gui_helpers import ClickEvent, choose_ct_method, open_block

pytestmark = pytest.mark.usefixtures("no_modal_dialogs")


def _pick(picker, row):
    """Choose a row the way a user does (setValue alone is a silent restore)."""
    picker.setValue(row)
    picker.activated.emit(picker.currentIndex())


def _answer(monkeypatch, answer, asked):
    def _question(_parent, title, text, *_args, **_kwargs):
        asked.append(f"{title}: {text}")
        return answer
    monkeypatch.setattr(
        "labchart2cfd.gui.app.QMessageBox.question", staticmethod(_question)
    )


def _landmarks_marked(win, mat_file):
    win.workflow_combo.setCurrentText("CT")
    choose_ct_method(win, "landmarks")
    open_block(win, mat_file)
    win._select_ct_step()
    win._mark_inhale_start()
    win._on_click(ClickEvent(win.ax_bottom, 4.0))
    win._mark_exhale_end()
    win._on_click(ClickEvent(win.ax_bottom, 3.0))
    win.subject_edit.setText("S1")


def _cycle_labelled(win, mat_file, times=(3.3, 3.5, 2.1)):
    win.workflow_combo.setCurrentText("CT")
    choose_ct_method(win, "cycle")
    open_block(win, mat_file)
    win._select_ct_step()
    win.cycle_mode_btn.setChecked(True)
    for t in times:
        win._on_click(ClickEvent(win.ax_top, t))


def _hint(win) -> str:
    return win.export_hint.text().lower()


# --------------------------------------------- pickers must not drop work

def test_changing_the_pressure_channel_keeps_every_marker(win, mat_file):
    """Pressure is not on the plot, so nothing has to be redrawn or dropped."""
    _landmarks_marked(win, mat_file)

    _pick(win.pressure_row_spin, 0)

    assert win._ct_inhale_start == 4.0 and win._ct_exhale_end == 3.0
    assert win._ct_selected_step is not None
    assert win.ct_inhale_label.text() == "4.000 s"


def test_picking_the_same_entry_again_changes_nothing(win, mat_file):
    _cycle_labelled(win, mat_file)
    frames = list(win._ct_cycle_frames)

    _pick(win.flow_row_spin, win.flow_row_spin.value())

    assert win._ct_cycle_frames == frames and frames


def test_changing_the_flow_channel_asks_before_dropping_landmarks(win, mat_file, monkeypatch):
    """They used to vanish while the line above Export still said Ready, and
    the export then wrote a different profile."""
    _landmarks_marked(win, mat_file)
    asked = []

    _answer(monkeypatch, QMessageBox.No, asked)
    _pick(win.flow_row_spin, 4)
    assert "landmark" in asked[0].lower()
    assert win.flow_row_spin.value() == 2          # the choice is taken back
    assert win._ct_inhale_start == 4.0             # and nothing is lost

    _answer(monkeypatch, QMessageBox.Yes, asked)
    _pick(win.flow_row_spin, 4)
    assert win.flow_row_spin.value() == 4
    assert win._ct_inhale_start is None


def test_changing_the_block_asks_before_dropping_cycle_labels_and_parts(
        win, mat_file, monkeypatch):
    _cycle_labelled(win, mat_file)
    win._parts.append(ProfilePart(start_image=1, end_image=3, label=""))
    asked = []

    _answer(monkeypatch, QMessageBox.No, asked)
    _pick(win.col_spin, 2)
    assert win.col_spin.value() == 1
    assert win._ct_cycle_frames and win._parts

    _answer(monkeypatch, QMessageBox.Yes, asked)
    _pick(win.col_spin, 2)
    assert win.col_spin.value() == 2
    assert win._ct_cycle_frames == [] and win._parts == []   # another scan


def test_changing_a_picker_with_nothing_marked_asks_nothing(win, mat_file):
    open_block(win, mat_file)

    _pick(win.col_spin, 2)       # the no_modal_dialogs fixture fails on a question

    assert win.col_spin.value() == 2


def test_cycle_sequence_readout_is_cleared_with_the_labels(win, mat_file):
    """The count went to 0 while the sequence line kept the old frames."""
    _cycle_labelled(win, mat_file)
    assert win.cycle_seq_label.text() != "seq: none"

    win._plot_selected()

    assert win._ct_cycle_frames == []
    assert win.cycle_seq_label.text() == "seq: none"


# ----------------------------------------------------- the mouse wheel

def _wheel(widget, notches=-1):
    center = QPointF(widget.rect().center())
    event = QWheelEvent(
        center, QPointF(widget.mapToGlobal(widget.rect().center())),
        QPoint(0, 0), QPoint(0, 120 * notches),
        Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False,
    )
    QApplication.sendEvent(widget, event)


def test_the_wheel_never_changes_a_sidebar_control(win, qapp, mat_file):
    """Scrolling the sidebar with the pointer over a control changed it: one
    notch turned MRI into CPAP, another set Trigger to None."""
    win.show()
    open_block(win, mat_file)
    win.start_spin.setValue(2.0)
    qapp.processEvents()
    before = (win.workflow_combo.currentText(), win.trigger_row_spin.value(),
              win.flow_row_spin.value(), win.start_spin.value(),
              win.ct_temporal_res_spin.value(), win.ct_shift_slider.value())

    for control in (win.workflow_combo, win.trigger_row_spin, win.flow_row_spin,
                    win.start_spin, win.ct_temporal_res_spin, win.ct_shift_slider):
        _wheel(control, -1)
        _wheel(control, +1)
        _wheel(control, -3)

    after = (win.workflow_combo.currentText(), win.trigger_row_spin.value(),
             win.flow_row_spin.value(), win.start_spin.value(),
             win.ct_temporal_res_spin.value(), win.ct_shift_slider.value())
    assert after == before


def test_the_wheel_over_a_control_still_scrolls_the_sidebar(win, qapp):
    win.resize(1400, 520)
    win.show()
    qapp.processEvents()
    bar = win.sidebar_scroll.verticalScrollBar()
    assert bar.maximum() > 0 and bar.value() == 0

    _wheel(win.workflow_combo, -3)

    assert bar.value() > 0


# ------------------------------------------- the line above Export is honest

def test_hint_says_the_whole_window_when_no_landmark_is_marked(win, mat_file):
    """It said 'landmarks method' while a plain window was exported."""
    win.workflow_combo.setCurrentText("CT")
    choose_ct_method(win, "landmarks")
    open_block(win, mat_file)
    win._select_ct_step()
    win.subject_edit.setText("S1")

    assert "no landmark" in _hint(win)
    assert "whole" in _hint(win)


def test_hint_says_the_whole_window_when_no_cycle_frame_is_labelled(win, mat_file):
    win.workflow_combo.setCurrentText("CT")
    choose_ct_method(win, "cycle")
    open_block(win, mat_file)
    win._select_ct_step()
    win.subject_edit.setText("S1")

    assert "no cycle frame" in _hint(win)
    assert "whole" in _hint(win)


def test_hint_counts_what_is_marked(win, mat_file):
    _landmarks_marked(win, mat_file)
    assert "landmarks method" in _hint(win) and "no landmark" not in _hint(win)

    win._clear_ct_landmarks()
    assert "no landmark" in _hint(win)             # follows the plot at once


def test_hint_counts_cycle_frames(win, mat_file):
    _cycle_labelled(win, mat_file)
    win.subject_edit.setText("S1")

    assert "3 frames" in _hint(win)


def test_switching_method_asks_before_dropping_that_methods_marks(win, mat_file, monkeypatch):
    """The other method's markers stayed on the plot although the export
    ignored them. Now one method's marks exist at a time."""
    _landmarks_marked(win, mat_file)
    asked = []

    _answer(monkeypatch, QMessageBox.No, asked)
    choose_ct_method(win, "cycle")
    assert win._ct_method() == "landmarks"         # the chooser goes back
    assert win._ct_inhale_start == 4.0
    assert "landmark" in asked[0].lower()

    _answer(monkeypatch, QMessageBox.Yes, asked)
    choose_ct_method(win, "cycle")
    assert win._ct_method() == "cycle"
    assert win._ct_inhale_start is None and win._ct_landmark_artists == []


def test_switching_method_with_nothing_marked_asks_nothing(win):
    win.workflow_combo.setCurrentText("CT")

    choose_ct_method(win, "landmarks")
    choose_ct_method(win, "cycle")
    choose_ct_method(win, "parts")

    assert win._ct_method() == "parts"


# --------------------------------------------------------------- no crash

def test_export_from_an_empty_flow_block_explains_instead_of_crashing(
        win, mat_file, monkeypatch):
    """Export in the parts method aborted the program when the chosen block
    had no flow data (the default Block 3 of a 3 block recording)."""
    win.workflow_combo.setCurrentText("CT")
    win.open_recording(mat_file)                   # Block 3: flow is empty
    win._parts.append(ProfilePart(start_image=1, end_image=1, label=""))
    win._refresh_parts_list()
    win.subject_edit.setText("S1")
    shown = []
    for kind in ("warning", "critical"):
        monkeypatch.setattr(
            f"labchart2cfd.gui.app.QMessageBox.{kind}",
            staticmethod(lambda _p, title, text, *a, **k: shown.append(f"{title}: {text}")),
        )

    assert "block" in _hint(win) and not _hint(win).startswith("ready")
    win._export()                                  # must not raise

    assert shown and "empty" in shown[0].lower()


def test_an_unexpected_error_is_shown_not_fatal(monkeypatch):
    """PyQt ends the process on an error inside a button handler unless an
    exception hook takes it."""
    from labchart2cfd.gui.app import install_exception_hook, report_unhandled_exception

    shown = []
    monkeypatch.setattr(
        "labchart2cfd.gui.app.QMessageBox.critical",
        staticmethod(lambda _p, title, text, *a, **k: shown.append(text)),
    )
    try:
        raise ValueError("Block [2, 3] is empty")
    except ValueError:
        report_unhandled_exception(*sys.exc_info())
    assert "Block [2, 3] is empty" in shown[0]

    previous = sys.excepthook
    try:
        install_exception_hook()
        assert sys.excepthook is report_unhandled_exception
    finally:
        sys.excepthook = previous


# ------------------------------------------------------------------ layout

def test_a_long_file_name_does_not_widen_the_sidebar(win, qapp, mat_file, tmp_path):
    """A 100 character name stretched the sidebar to 900 pixels."""
    long_name = tmp_path / ("recording_with_a_very_long_name_" * 4 + ".mat")
    shutil.copy(mat_file, long_name)
    win.show()
    qapp.processEvents()
    needed_before = win.sidebar_scroll.widget().minimumSizeHint().width()

    win.open_recording(long_name)
    qapp.processEvents()

    assert win.sidebar_scroll.widget().minimumSizeHint().width() == needed_before
    assert win.file_label.text() == long_name.name      # the full name is kept
    assert win.file_label.toolTip() == str(long_name)


def test_an_extreme_temporal_res_draws_no_runaway_grid(win, mat_file):
    """0.01 ms asks for 300,000 images in a 3 s step, and drawing a line for
    each froze the window."""
    win.workflow_combo.setCurrentText("CT")
    open_block(win, mat_file)
    win._select_ct_step()

    win.ct_temporal_res_spin.setValue(0.01)

    assert len(win._ct_interval_markers) < 10_000
    assert "too many" in win.status_bar.currentMessage().lower()


def test_zooming_to_a_few_samples_raises_no_warning(win, mat_file):
    open_block(win, mat_file)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        win.ax_top.set_xlim(8.499, 8.503)
        win.canvas.draw()


# ------------------------------------------------------------ smaller ones

def test_overwrite_question_leaves_out_a_pressure_file_that_cannot_be_written(
        win, mat_file, tmp_path, monkeypatch):
    open_block(win, mat_file)
    win.subject_edit.setText("S1")
    win._set_output_dir(tmp_path)
    win.start_spin.setValue(2.0)
    win.end_spin.setValue(6.0)
    win.pressure_row_spin.setValue(3)              # empty in block 1
    (tmp_path / "S1PressureProfile.csv").write_text("an earlier export")
    asked, calls = [], []
    _answer(monkeypatch, QMessageBox.Yes, asked)
    monkeypatch.setattr(win, "_execute_processing", lambda: calls.append("execute"))

    win._export()

    assert calls == ["execute"]
    assert not any("Replace" in question for question in asked)


def test_a_new_recording_starts_with_a_clean_window_and_grid(win, mat_file, tmp_path):
    """Start, End, the image count and the offsets of the previous recording
    carried over and described a scan that was no longer open."""
    win.workflow_combo.setCurrentText("CT")
    open_block(win, mat_file)
    win._select_ct_step()
    win.synth_grid_offset_spin.setValue(-200)
    win.ct_shift_spin.setValue(50)
    assert win.synth_images_spin.value() == 15 and win.start_spin.value() == 2.0
    other = tmp_path / "other.mat"
    shutil.copy(mat_file, other)

    win.open_recording(other)

    assert win.start_spin.value() == 0.0 and win.end_spin.value() == 0.0
    assert win.synth_images_spin.value() == 1      # seeded again by the next step
    assert win.synth_grid_offset_spin.value() == 0
    assert win.ct_shift_spin.value() == 0


def test_reopening_the_same_recording_keeps_the_window(win, mat_file):
    win.workflow_combo.setCurrentText("CT")
    open_block(win, mat_file)
    win._select_ct_step()

    win.open_recording(mat_file)

    assert win.start_spin.value() == 2.0 and win.end_spin.value() == 5.0
