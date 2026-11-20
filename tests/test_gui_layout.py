"""Tests for the main window layout and its guided flow.

Qt runs offscreen so no window appears. These pin what the user sees and does:
the controls live in a sidebar beside the plot, a recording opens in one
action, channels are picked by name, and only the controls that belong to the
chosen study type are on screen.
"""
import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QSplitter

from tests.gui_helpers import ClickEvent as _Event
from tests.gui_helpers import choose_ct_method as _choose_ct_method

pytestmark = pytest.mark.usefixtures("no_modal_dialogs")


# ------------------------------------------------------------------- layout

def test_controls_sit_in_a_sidebar_beside_the_plot(win):
    """The plot keeps the full window height; controls no longer sit under it."""
    splitter = win.centralWidget()

    assert isinstance(splitter, QSplitter)
    assert splitter.orientation() == Qt.Horizontal
    assert splitter.widget(0).isAncestorOf(win.flow_row_spin)
    assert splitter.widget(0).isAncestorOf(win.execute_btn)
    assert splitter.widget(1).isAncestorOf(win.canvas)


def test_no_sidebar_control_is_clipped_behind_the_scroll_bar(win, qapp):
    """The CT sections are the widest. They must still fit the sidebar."""
    win.show()
    win.workflow_combo.setCurrentText("CT")
    qapp.processEvents()

    content = win.sidebar_scroll.widget()

    assert win.sidebar_scroll.viewport().width() >= content.minimumSizeHint().width()


def test_only_one_image_interval_box_is_shown(win):
    """Two mirrored Temporal Res boxes showed one quantity twice."""
    assert win.synth_dt_spin.isHidden()
    assert not win.ct_temporal_res_spin.isHidden()

    win.ct_temporal_res_spin.setValue(150.0)

    assert win.synth_dt_spin.value() == pytest.approx(150.0)  # still mirrored


# ------------------------------------------------------- opening a recording

def test_opening_a_recording_loads_it_and_shows_the_overview(win, mat_file):
    """One action replaces Browse, then Load Overview."""
    win.open_recording(mat_file)

    assert win.data is not None
    assert win._plot_mode == "overview"
    assert mat_file.name in win.file_label.text()


def test_channel_pickers_show_the_names_from_the_file(win, mat_file):
    win.open_recording(mat_file)

    flow = [win.flow_row_spin.itemText(i) for i in range(win.flow_row_spin.count())]
    assert flow == ["1: Trigger", "2: Flow", "Channel 3", "4: Pressure"]
    assert win.trigger_row_spin.itemText(0) == "None"
    assert win.col_spin.count() == 3


def test_clicking_an_overview_tile_opens_that_block(win, mat_file):
    win.open_recording(mat_file)

    win._on_overview_click(_Event(win._overview_axes[(2, 1)]))

    assert win._plot_mode == "selected"
    assert win.col_spin.value() == 1


def test_clicking_a_block_without_flow_data_stays_on_the_overview(win, mat_file):
    win.open_recording(mat_file)

    win._on_overview_click(_Event(win._overview_axes[(3, 3)]))

    assert win._plot_mode == "overview"
    assert "empty" in win.status_bar.currentMessage().lower()


def test_changing_the_block_dropdown_refreshes_the_detail_view(win, mat_file):
    """No separate Plot Selected click is needed after changing a picker."""
    win.open_recording(mat_file)
    win._on_overview_click(_Event(win._overview_axes[(2, 1)]))

    win.col_spin.setValue(2)
    win.col_spin.activated.emit(win.col_spin.currentIndex())

    assert win._plot_mode == "selected"
    assert "block 2" in win.ax_top.get_title().lower()


def test_changing_a_channel_on_the_overview_keeps_the_overview(win, mat_file):
    win.open_recording(mat_file)

    win.flow_row_spin.setValue(4)
    win.flow_row_spin.activated.emit(win.flow_row_spin.currentIndex())

    assert win._plot_mode == "overview"


# ---------------------------------------------------------------- study type

def test_ct_sections_appear_only_for_ct(win):
    assert win.ct_scan_section.isHidden()
    assert win.ct_method_section.isHidden()
    assert not win.trigger_section.isHidden()

    win.workflow_combo.setCurrentText("CT")

    assert not win.ct_scan_section.isHidden()
    assert not win.ct_method_section.isHidden()
    assert win.trigger_section.isHidden()

    win.workflow_combo.setCurrentText("MRI")

    assert win.ct_scan_section.isHidden()
    assert not win.trigger_section.isHidden()


@pytest.mark.parametrize("shown, engine", [
    ("MRI", "MRI"),
    ("CPAP", "cpap"),
    ("Phase contrast (xenon)", "phase-contrast"),
    ("CT", "CT"),
])
def test_study_type_maps_to_the_engine_workflow(win, shown, engine):
    win.workflow_combo.setCurrentText(shown)

    assert win._workflow_name() == engine


# ----------------------------------------------------------------- CT method

def test_ct_method_chooser_shows_one_method_at_a_time(win):
    """Three ways to pick images used to share the screen across two tabs."""
    win.workflow_combo.setCurrentText("CT")
    assert win.ct_method_stack.currentWidget() is win.parts_page

    _choose_ct_method(win, "landmarks")
    assert win.ct_method_stack.currentWidget() is win.landmarks_page

    _choose_ct_method(win, "cycle")
    assert win.ct_method_stack.currentWidget() is win.cycle_page


# -------------------------------------------------------- channel name check

def _pick(picker, row):
    """Choose a row the way a user does, which also redraws the view."""
    picker.setValue(row)
    picker.activated.emit(picker.currentIndex())


def test_no_channel_warning_when_names_fit_the_roles(win, mat_file):
    win.open_recording(mat_file)

    assert win.channel_check.isHidden()


def test_a_role_on_a_differently_named_channel_is_flagged(win, mat_file):
    """On a real recording the default rows put Flow on a pressure channel.
    Exported like that, pressure would have been written as flow."""
    win.open_recording(mat_file)

    _pick(win.flow_row_spin, 4)             # channel 4 is named Pressure

    assert not win.channel_check.isHidden()
    text = win.channel_check.text()
    assert "Flow is set to 4: Pressure" in text
    assert "2: Flow" in text                # the channel that is named like flow


def test_the_name_check_never_changes_a_choice(win, mat_file):
    """Channels are flagged, not inferred: the user decides."""
    win.open_recording(mat_file)

    _pick(win.flow_row_spin, 4)

    assert win.flow_row_spin.value() == 4


def test_fixing_the_channel_clears_the_warning(win, mat_file):
    win.open_recording(mat_file)
    _pick(win.flow_row_spin, 4)

    _pick(win.flow_row_spin, 2)

    assert win.channel_check.isHidden()


def test_untitled_channels_raise_no_warning(win):
    import numpy as np
    from labchart2cfd.io.labchart import LabChartData

    t = np.linspace(0.0, 1.0, 1001)
    win.data = LabChartData(
        data_cell=[[t], [t], [t], [t]], time_cell=[[t], [t], [t], [t]],
        num_channels=4, num_blocks=1,
    )

    win._show_overview()

    assert win.channel_check.isHidden()
