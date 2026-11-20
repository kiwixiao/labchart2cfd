"""What the window remembers between sessions: the last folder and its size.

Nothing was remembered, so every session started in the working directory
and at the default size. Channel roles are deliberately not remembered.
"""
import pytest

from labchart2cfd.gui.app import FlowProfileApp

pytestmark = pytest.mark.usefixtures("no_modal_dialogs")


def _open_dialog_folder(win, monkeypatch):
    """The folder the Open dialog would start in."""
    seen = []
    monkeypatch.setattr(
        "labchart2cfd.gui.app.QFileDialog.getOpenFileName",
        staticmethod(lambda _p, _title, folder, _filter: (seen.append(folder), ("", ""))[1]),
    )
    win._browse_file()
    return seen[0]


def test_the_open_dialog_starts_where_the_last_recording_was(win, qapp, mat_file, monkeypatch):
    assert _open_dialog_folder(win, monkeypatch) == ""

    win.open_recording(mat_file)
    assert _open_dialog_folder(win, monkeypatch) == str(mat_file.parent)

    win.close()
    again = FlowProfileApp()                       # a new session
    try:
        assert _open_dialog_folder(again, monkeypatch) == str(mat_file.parent)
    finally:
        again.close()


def test_a_folder_that_no_longer_exists_is_not_offered(win, qapp, mat_file, monkeypatch, tmp_path):
    import shutil
    gone = tmp_path / "gone"
    gone.mkdir()
    moved = gone / "rec.mat"
    shutil.copy(mat_file, moved)
    win.open_recording(moved)
    shutil.rmtree(gone)

    assert _open_dialog_folder(win, monkeypatch) == ""


def test_the_window_reopens_at_the_size_it_was_closed_with(win, qapp):
    win.show()
    win.resize(1111, 777)
    qapp.processEvents()
    win.close()

    again = FlowProfileApp()
    try:
        assert (again.width(), again.height()) == (1111, 777)
    finally:
        again.close()


def test_a_first_session_opens_at_the_default_size(win):
    assert (win.width(), win.height()) == (1400, 860)


def test_channel_choices_are_not_remembered(win, qapp, mat_file):
    """Restoring them would choose channels without a click."""
    win.open_recording(mat_file)
    win.flow_row_spin.setValue(4)
    win.close()

    again = FlowProfileApp()
    try:
        assert again.flow_row_spin.value() == 2    # the plain default
    finally:
        again.close()
