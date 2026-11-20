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


# --------------------------------------------------------------------- theme

def test_the_window_carries_the_theme_style_sheet(win):
    assert "#modeBanner" in win.styleSheet()
    assert "#primaryButton" in win.styleSheet()


def test_each_banner_kind_has_its_own_colour():
    from labchart2cfd.gui.theme import BANNER_COLORS

    assert set(BANNER_COLORS) == {"idle", "active", "blocked"}
    backgrounds = {colors["background"] for colors in BANNER_COLORS.values()}
    assert len(backgrounds) == 3
