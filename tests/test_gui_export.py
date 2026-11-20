"""Tests for the export flow of the main window.

One Export button, an output folder that stays where the user put it, a
question before existing files are replaced, a preview that can be cancelled,
and a result panel that lists what was written.
"""
import time

import numpy as np
import pytest
from PyQt5.QtWidgets import QDialog, QMessageBox, QPushButton

from labchart2cfd.gui.dialogs import AssemblyPreviewDialog, ExportResultDialog
from tests.gui_helpers import choose_ct_method, open_block

pytestmark = pytest.mark.usefixtures("no_modal_dialogs")


def _wait_for_export(win, qapp, seconds=15.0):
    deadline = time.time() + seconds
    while not win.execute_btn.isEnabled() and time.time() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    assert win.execute_btn.isEnabled(), "the export never finished"
    qapp.processEvents()


def _answer_question(monkeypatch, answer, asked):
    def _question(_parent, _title, text, *_args, **_kwargs):
        asked.append(text)
        return answer
    monkeypatch.setattr(
        "labchart2cfd.gui.app.QMessageBox.question", staticmethod(_question)
    )


# ------------------------------------------------------------- output folder

def test_output_folder_follows_the_subject_until_one_is_picked(win, tmp_path):
    win.mat_filepath = tmp_path / "rec.mat"

    win.subject_edit.setText("S1")

    assert win.outdir_edit.text() == str(tmp_path / "FlowResults_S1")


def test_a_hand_picked_output_folder_survives_typing_the_subject(win, tmp_path):
    """Typing the Subject ID used to replace a folder chosen with the browser."""
    win.mat_filepath = tmp_path / "rec.mat"
    win._set_output_dir(tmp_path / "custom")

    win.subject_edit.setText("S1")

    assert win.outdir_edit.text() == str(tmp_path / "custom")


def test_typing_in_the_folder_box_counts_as_picking_it(win, tmp_path):
    win.mat_filepath = tmp_path / "rec.mat"
    win.outdir_edit.setText(str(tmp_path / "typed"))
    win.outdir_edit.textEdited.emit(str(tmp_path / "typed"))   # a user edit

    win.subject_edit.setText("S1")

    assert win.outdir_edit.text() == str(tmp_path / "typed")


def test_opening_another_recording_returns_to_the_default_folder(win, mat_file, tmp_path):
    win._set_output_dir(tmp_path / "custom")

    win.open_recording(mat_file)

    assert win.outdir_edit.text() == str(mat_file.parent / "FlowResults")


# ------------------------------------------------------------- one Export

def test_there_is_exactly_one_export_button(win):
    """Execute on one tab and Assemble & Export on another were two exits."""
    labels = [b.text() for b in win.sidebar.findChildren(QPushButton)]

    assert [text for text in labels if "export" in text.lower()] == ["Export"]


def test_export_runs_the_path_of_the_chosen_method(win, monkeypatch):
    calls = []
    monkeypatch.setattr(win, "_assemble_and_export", lambda: calls.append("assemble"))
    monkeypatch.setattr(win, "_execute_processing", lambda: calls.append("execute"))

    win.workflow_combo.setCurrentText("CT")      # the parts method by default
    win.execute_btn.click()
    choose_ct_method(win, "landmarks")
    win.execute_btn.click()
    win.workflow_combo.setCurrentText("MRI")
    win.execute_btn.click()

    assert calls == ["assemble", "execute", "execute"]


# ------------------------------------------------------- replacing old files

def _ready_mri_export(win, mat_file, out_dir, subject="S1"):
    open_block(win, mat_file)
    win.subject_edit.setText(subject)
    win._set_output_dir(out_dir)
    win.start_spin.setValue(2.0)
    win.end_spin.setValue(6.0)


def test_export_asks_before_replacing_existing_files(win, mat_file, tmp_path, monkeypatch):
    _ready_mri_export(win, mat_file, tmp_path)
    (tmp_path / "S1FlowProfile.csv").write_text("an earlier export")
    calls, asked = [], []
    monkeypatch.setattr(win, "_execute_processing", lambda: calls.append("execute"))

    _answer_question(monkeypatch, QMessageBox.No, asked)
    win._export()
    assert calls == []
    assert "S1FlowProfile.csv" in asked[0]

    _answer_question(monkeypatch, QMessageBox.Yes, asked)
    win._export()
    assert calls == ["execute"]


def test_export_does_not_ask_when_nothing_would_be_replaced(win, mat_file, tmp_path, monkeypatch):
    _ready_mri_export(win, mat_file, tmp_path)
    calls = []
    monkeypatch.setattr(win, "_execute_processing", lambda: calls.append("execute"))

    win._export()       # the no_modal_dialogs fixture fails on any question

    assert calls == ["execute"]


def test_export_is_ignored_while_one_is_running(win, monkeypatch):
    """The menu item and its shortcut must not start a second export."""
    calls = []
    monkeypatch.setattr(win, "_execute_processing", lambda: calls.append("execute"))
    win.execute_btn.setEnabled(False)        # what a running export does

    win.export_action.trigger()

    assert calls == []


def test_overwrite_question_covers_only_files_this_export_writes(
        win, mat_file, tmp_path, monkeypatch):
    _ready_mri_export(win, mat_file, tmp_path)
    win.pressure_row_spin.setValue(0)        # no pressure asked for
    (tmp_path / "S1PressureProfile.csv").write_text("an earlier export")
    calls = []
    monkeypatch.setattr(win, "_execute_processing", lambda: calls.append("execute"))

    win._export()       # no question: this export writes no pressure file

    assert calls == ["execute"]


def test_a_stale_pressure_file_is_called_out_after_the_export(qapp, win, mat_file, tmp_path):
    """An old pressure profile beside a new flow profile looks like a pair."""
    _ready_mri_export(win, mat_file, tmp_path)
    win.pressure_row_spin.setValue(0)
    (tmp_path / "S1PressureProfile.csv").write_text("an earlier export")

    win._export()
    _wait_for_export(win, qapp)

    warnings = win._last_export_report["warnings"]
    assert any("older" in w.lower() and "S1PressureProfile.csv" in w for w in warnings)


# ------------------------------------------------------------- result report

def test_a_finished_export_lists_the_files_it_wrote(qapp, win, mat_file, tmp_path):
    """The outcome used to be one long status bar line."""
    _ready_mri_export(win, mat_file, tmp_path)

    win._export()
    _wait_for_export(win, qapp)

    report = win._last_export_report
    assert [p.name for p in report["files"]] == [
        "S1FlowProfile.csv", "S1PressureProfile.csv",
        "S1_trigger_info.txt", "S1_sanity_check.png",
    ]
    assert all(p.exists() for p in report["files"])
    assert report["warnings"] == []
    assert win._result_dialog is not None


def test_a_missing_pressure_profile_is_reported_as_a_warning(qapp, win, mat_file, tmp_path):
    """Pressure was asked for but the channel is empty in this block."""
    _ready_mri_export(win, mat_file, tmp_path)
    win.pressure_row_spin.setValue(3)        # channel 3 is empty in block 1

    win._export()
    _wait_for_export(win, qapp)

    report = win._last_export_report
    assert "S1PressureProfile.csv" not in [p.name for p in report["files"]]
    assert any("pressure" in w.lower() for w in report["warnings"])


def test_result_dialog_lists_files_and_shows_warnings(qapp, tmp_path):
    report = {
        "folder": tmp_path,
        "files": [tmp_path / "S1FlowProfile.csv", tmp_path / "S1_trigger_info.txt"],
        "summary": [("Duration", "4.00 s"), ("Sample rate", "100 Hz")],
        "warnings": ["No pressure profile was produced."],
        "plot": None,
    }

    dialog = ExportResultDialog(report)

    assert dialog.file_list.count() == 2
    assert dialog.file_list.item(0).text() == "S1FlowProfile.csv"
    assert not dialog.warning_label.isHidden()
    assert "pressure" in dialog.warning_label.text().lower()


def test_result_dialog_hides_the_warning_box_when_there_are_none(qapp, tmp_path):
    report = {"folder": tmp_path, "files": [], "summary": [], "warnings": [],
              "plot": None}

    dialog = ExportResultDialog(report)

    assert dialog.warning_label.isHidden()


# ---------------------------------------------------------------- readiness

def _hint(win) -> str:
    return win.export_hint.text().lower()


def test_export_hint_names_what_is_still_missing(win, mat_file):
    assert "recording" in _hint(win)
    assert "subject" in _hint(win)

    open_block(win, mat_file)
    assert "recording" not in _hint(win)
    assert "subject" in _hint(win)
    assert "start" in _hint(win)

    win.subject_edit.setText("S1")
    win.start_spin.setValue(2.0)
    win.end_spin.setValue(6.0)
    assert _hint(win).startswith("ready")


def test_export_hint_names_the_ct_method_that_will_be_exported(win, mat_file):
    win.workflow_combo.setCurrentText("CT")
    choose_ct_method(win, "landmarks")
    open_block(win, mat_file)
    win.subject_edit.setText("S1")
    win.start_spin.setValue(2.0)
    win.end_spin.setValue(5.0)

    assert "landmarks" in _hint(win)


def test_export_hint_asks_for_a_part_in_the_parts_method(win, mat_file):
    win.workflow_combo.setCurrentText("CT")
    open_block(win, mat_file)
    win.subject_edit.setText("S1")
    win._select_ct_step()
    assert "part" in _hint(win)
    assert not _hint(win).startswith("ready")

    win.part_from_spin.setValue(1)
    win.part_to_spin.setValue(5)
    win._add_part_from_spins()
    assert _hint(win).startswith("ready")


# ------------------------------------------------------------------ preview

def test_preview_offers_export_and_cancel(qapp):
    """The preview had only Close, and the export ran after it regardless."""
    t = np.linspace(0.0, 1.0, 101)

    accepted = AssemblyPreviewDialog(t, np.sin(t), 5)
    accepted.export_button.click()
    assert accepted.result() == QDialog.Accepted

    rejected = AssemblyPreviewDialog(t, np.sin(t), 5)
    rejected.cancel_button.click()
    assert rejected.result() == QDialog.Rejected


def _ready_parts_export(win, mat_file, out_dir):
    win.workflow_combo.setCurrentText("CT")
    open_block(win, mat_file)
    win._select_ct_step()                     # 2 s to 5 s, 15 images of 200 ms
    for first, last in ((6, 15), (1, 5)):
        win.part_from_spin.setValue(first)
        win.part_to_spin.setValue(last)
        win._add_part_from_spins()
    win.subject_edit.setText("S1")
    win._set_output_dir(out_dir)


def test_cancelling_the_preview_writes_nothing(win, mat_file, tmp_path, monkeypatch):
    out_dir = tmp_path / "out"          # apart from the recording itself
    _ready_parts_export(win, mat_file, out_dir)
    monkeypatch.setattr(AssemblyPreviewDialog, "exec_", lambda self: QDialog.Rejected)

    win._export()

    assert not out_dir.exists()
    assert win._last_export_report is None


def test_accepting_the_preview_writes_the_profile_and_the_reorder_map(
        win, mat_file, tmp_path, monkeypatch):
    _ready_parts_export(win, mat_file, tmp_path)
    monkeypatch.setattr(AssemblyPreviewDialog, "exec_", lambda self: QDialog.Accepted)

    win._export()

    report = win._last_export_report
    assert [p.name for p in report["files"]] == [
        "S1FlowProfile.csv", "S1PressureProfile.csv", "S1ImageReorderMap.csv",
    ]
    assert all(p.exists() for p in report["files"])
    assert win._result_dialog is not None
