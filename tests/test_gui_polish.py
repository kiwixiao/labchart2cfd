"""Tests for the conveniences around the main flow: menus, shortcuts, drag and
drop, the window readout, the quick guide and the visual theme."""
import pytest
from PyQt5.QtCore import QMimeData, QPointF, Qt, QUrl
from PyQt5.QtGui import QDropEvent

from tests.gui_helpers import open_block

pytestmark = pytest.mark.usefixtures("no_modal_dialogs")


# ------------------------------------------------------- menus and shortcuts

def test_open_and_export_have_the_usual_shortcuts(win):
    assert win.open_action.shortcut().toString() == "Ctrl+O"
    assert win.export_action.shortcut().toString() == "Ctrl+E"


def test_export_menu_item_runs_the_same_export_as_the_button(win, monkeypatch):
    calls = []
    monkeypatch.setattr(win, "_export", lambda: calls.append("export"))

    win.export_action.trigger()

    assert calls == ["export"]


def test_view_menu_returns_to_all_blocks(win, mat_file):
    open_block(win, mat_file)
    assert win._plot_mode == "selected"

    win.overview_action.trigger()

    assert win._plot_mode == "overview"


def test_view_menu_reopens_the_selected_block(win, mat_file):
    open_block(win, mat_file)
    win.overview_action.trigger()

    win.detail_action.trigger()

    assert win._plot_mode == "selected"


# ------------------------------------------------------ bringing CT into view

def test_choosing_ct_scrolls_the_ct_sections_into_view(win, qapp):
    """The old window jumped to the CT tab. The sidebar must not leave the CT
    controls below the fold, out of sight."""
    win.resize(1400, 700)
    win.show()
    qapp.processEvents()
    assert win.sidebar_scroll.verticalScrollBar().value() == 0

    win.workflow_combo.setCurrentText("CT")
    for _ in range(5):
        qapp.processEvents()

    viewport = win.sidebar_scroll.viewport()
    top = win.ct_scan_section.mapTo(viewport, win.ct_scan_section.rect().topLeft()).y()
    assert 0 <= top < viewport.height()


# ------------------------------------------------------------- drag and drop

def _mime_for(path) -> QMimeData:
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(path))])
    return mime


def test_only_a_mat_file_is_accepted_as_a_drop(win, mat_file, tmp_path):
    other = tmp_path / "notes.txt"
    other.write_text("not a recording")

    assert win._dropped_recording(_mime_for(mat_file)) == mat_file
    assert win._dropped_recording(_mime_for(other)) is None


def test_dropping_a_recording_on_the_window_opens_it(win, mat_file):
    mime = _mime_for(mat_file)       # kept alive for the life of the event
    event = QDropEvent(
        QPointF(10, 10), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier
    )

    win.dropEvent(event)

    assert win.data is not None
    assert win._plot_mode == "overview"


# ------------------------------------------------------------ window readout

def test_window_readout_shows_the_duration(win):
    win.start_spin.setValue(2.0)
    win.end_spin.setValue(5.5)

    assert "3.500 s" in win.window_readout.text()


def test_window_readout_flags_an_end_before_the_start(win):
    win.start_spin.setValue(5.0)
    win.end_spin.setValue(2.0)

    assert "after" in win.window_readout.text().lower()


# --------------------------------------------------------------- quick guide

def test_help_menu_opens_a_quick_guide_without_blocking(win):
    win.guide_action.trigger()

    assert win._guide_dialog is not None
    assert "Open recording" in win._guide_dialog.guide_label.text()
    assert not win._guide_dialog.isModal()


def test_the_quick_guide_names_every_section_and_ct_method_on_screen(win):
    """The guide still described CT as one step, "4. CT only", after the
    window gained 4 CT scan, 5 Build the breath, the drift correction
    breath, extra flow and a preview where the complete half is chosen.
    The names come from the window, so a new section or method the guide
    leaves out fails here."""
    from labchart2cfd.gui.app import _CT_METHODS
    from labchart2cfd.gui.widgets import CollapsibleSection

    win.guide_action.trigger()
    text = " ".join(win._guide_dialog.guide_label.text().split())
    sections = [" ".join(s.header.text().split()[1:])
                for s in win.findChildren(CollapsibleSection)]

    assert len(sections) == 6
    for name in sections + [shown for shown, _key, _tip in _CT_METHODS]:
        assert name in text, name
    for words in ("Drift correction breath", "Extra flow", "Complete half"):
        assert words in text, words


# --------------------------------------------------------------------- theme

def test_the_window_carries_the_theme_style_sheet(win):
    assert "#modeBanner" in win.styleSheet()
    assert "#primaryButton" in win.styleSheet()


def test_each_banner_kind_has_its_own_colour():
    from labchart2cfd.gui.theme import BANNER_COLORS

    assert set(BANNER_COLORS) == {"idle", "active", "blocked"}
    backgrounds = {colors["background"] for colors in BANNER_COLORS.values()}
    assert len(backgrounds) == 3


# ------------------------------------------------- native .adicht recordings

def _fake_recording():
    import numpy as np
    from labchart2cfd.io.labchart import LabChartData

    t = np.arange(0.0, 10.0, 0.001)
    return LabChartData(
        data_cell=[[np.zeros_like(t)], [np.sin(t)]], time_cell=[[t], [t]],
        num_channels=2, num_blocks=1, titles=["TRIGGER", "FLOW"],
    )


def test_open_dialog_offers_native_recordings(win):
    assert "*.adicht" in win._OPEN_FILTER
    assert "*.mat" in win._OPEN_FILTER


def test_a_dropped_adicht_file_is_accepted(win, tmp_path):
    native = tmp_path / "rec.adicht"
    native.write_bytes(b"")

    assert win._dropped_recording(_mime_for(native)) == native


def test_opening_an_adicht_shows_its_channels(win, tmp_path, monkeypatch):
    """The native file opens like an export: no manual conversion step."""
    native = tmp_path / "rec.adicht"
    native.write_bytes(b"")
    monkeypatch.setattr(
        "labchart2cfd.gui.app.load_recording", lambda path, notes=None, progress=None: _fake_recording()
    )

    win.open_recording(native)

    assert win._plot_mode == "overview"
    assert win.flow_row_spin.itemText(1) == "2: FLOW"
    assert "rec.adicht" in win.file_label.text()


def test_what_the_reader_recomputed_or_left_out_is_shown(win, tmp_path, monkeypatch):
    """A channel recomputed or left out must be said on screen, not only in a console."""
    native = tmp_path / "rec.adicht"
    native.write_bytes(b"")

    def _load(path, notes=None, progress=None):
        notes.append("LEFT OUT: channel 4 (VOL) carries a calculation.")
        return _fake_recording()

    monkeypatch.setattr("labchart2cfd.gui.app.load_recording", _load)

    win.open_recording(native)

    assert not win.load_notes.isHidden()
    assert "LEFT OUT: channel 4 (VOL)" in win.load_notes.text()


def test_load_notes_are_cleared_by_the_next_recording(win, tmp_path, monkeypatch, mat_file):
    native = tmp_path / "rec.adicht"
    native.write_bytes(b"")

    def _load(path, notes=None, progress=None):
        notes.append("LEFT OUT: channel 4 (VOL) carries a calculation.")
        return _fake_recording()

    monkeypatch.setattr("labchart2cfd.gui.app.load_recording", _load)
    win.open_recording(native)
    monkeypatch.undo()

    win.open_recording(mat_file)

    assert win.load_notes.isHidden()


def test_a_refused_adicht_reports_the_readers_reason(win, tmp_path, monkeypatch):
    bad = tmp_path / "rec.adicht"
    bad.write_bytes(b"not a recording " * 100)
    shown = []
    monkeypatch.setattr(
        "labchart2cfd.gui.app.QMessageBox.critical",
        staticmethod(lambda _parent, _title, text, *a, **k: shown.append(text)),
    )

    win.open_recording(bad)

    assert win.data is None
    assert "ADInstruments" in shown[0]
    assert "Could not open" in win.file_label.text()


def test_the_wait_cursor_is_released_after_loading(win, mat_file, tmp_path, monkeypatch):
    from PyQt5.QtWidgets import QApplication

    win.open_recording(mat_file)
    assert QApplication.overrideCursor() is None

    bad = tmp_path / "rec.adicht"
    bad.write_bytes(b"not a recording " * 100)
    monkeypatch.setattr(
        "labchart2cfd.gui.app.QMessageBox.critical", staticmethod(lambda *a, **k: None)
    )
    win.open_recording(bad)
    assert QApplication.overrideCursor() is None      # also after a failed load


def test_the_controls_are_locked_while_a_recording_loads(win, tmp_path, monkeypatch):
    """The window keeps processing events during a long decode. A click then
    must not export the previous recording under the new file's name."""
    native = tmp_path / "rec.adicht"
    native.write_bytes(b"")
    seen = {}

    def _load(path, notes=None, progress=None):
        seen["sidebar"] = win.sidebar.isEnabled()
        seen["menus"] = win.menuBar().isEnabled()
        return _fake_recording()

    monkeypatch.setattr("labchart2cfd.gui.app.load_recording", _load)

    win.open_recording(native)

    assert seen == {"sidebar": False, "menus": False}
    assert win.sidebar.isEnabled() and win.menuBar().isEnabled()


def test_the_controls_are_unlocked_after_a_failed_load(win, tmp_path, monkeypatch):
    bad = tmp_path / "rec.adicht"
    bad.write_bytes(b"not a recording " * 100)
    monkeypatch.setattr(
        "labchart2cfd.gui.app.QMessageBox.critical", staticmethod(lambda *a, **k: None)
    )

    win.open_recording(bad)

    assert win.sidebar.isEnabled() and win.menuBar().isEnabled()


def test_decoding_progress_reaches_the_status_bar(win, tmp_path, monkeypatch):
    native = tmp_path / "rec.adicht"
    native.write_bytes(b"")
    seen = []

    def _load(path, notes=None, progress=None):
        progress(3, 10, "channel 2 FLOW, record 1")
        seen.append(win.status_bar.currentMessage())
        return _fake_recording()

    monkeypatch.setattr("labchart2cfd.gui.app.load_recording", _load)

    win.open_recording(native)

    assert "3 of 10" in seen[0]
    assert "channel 2 FLOW, record 1" in seen[0]
