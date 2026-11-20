"""Tests for the usability review of the redesigned window (first time user).

These are not crashes. Each is a place where the screen led a new user to the
wrong action, or left them without the fact they needed.
"""
import pytest

from labchart2cfd.gui import app as gui_app
from tests.gui_helpers import ClickEvent, choose_ct_method, open_block

pytestmark = pytest.mark.usefixtures("no_modal_dialogs")


def _ct_step_selected(win, mat_file, method):
    win.workflow_combo.setCurrentText("CT")
    choose_ct_method(win, method)
    open_block(win, mat_file)
    win._select_ct_step()


# --------------------------------- a plain click must not move the scan window

@pytest.mark.parametrize("method", ["landmarks", "cycle"])
def test_a_plain_click_leaves_the_selected_scan_step_alone(win, mat_file, method):
    """The banner said 'Click the plot to set Start', so the natural first
    click in these methods moved the scan window away from the step, with
    nothing to warn or undo."""
    _ct_step_selected(win, mat_file, method)
    assert (win.start_spin.value(), win.end_spin.value()) == (2.0, 5.0)

    win._on_click(ClickEvent(win.ax_bottom, 3.5))
    win._on_click(ClickEvent(win.ax_bottom, 4.0))

    assert (win.start_spin.value(), win.end_spin.value()) == (2.0, 5.0)
    assert win._time_markers == []


@pytest.mark.parametrize("method, button", [
    ("landmarks", "Mark Inhale Start"), ("cycle", "Label Cycle Frames"),
])
def test_the_banner_names_the_button_to_press_next(win, mat_file, method, button):
    _ct_step_selected(win, mat_file, method)

    banner = win.mode_banner.text()

    assert button in banner
    assert "set Start" not in banner


def test_before_a_step_is_selected_clicks_still_set_the_window(win, mat_file):
    """A recording without a usable trigger has no step to select."""
    win.workflow_combo.setCurrentText("CT")
    choose_ct_method(win, "landmarks")
    open_block(win, mat_file)

    win._on_click(ClickEvent(win.ax_bottom, 3.0))
    win._on_click(ClickEvent(win.ax_bottom, 4.0))

    assert (win.start_spin.value(), win.end_spin.value()) == (3.0, 4.0)


def test_selecting_a_step_removes_window_markers_that_no_longer_apply(win, mat_file):
    """A green marker stayed at a time that was no longer Start."""
    win.workflow_combo.setCurrentText("CT")
    choose_ct_method(win, "landmarks")
    open_block(win, mat_file)
    win._on_click(ClickEvent(win.ax_bottom, 3.0))
    win._on_click(ClickEvent(win.ax_bottom, 4.0))

    win._select_ct_step()

    assert win._time_markers == []
    assert (win.start_spin.value(), win.end_spin.value()) == (2.0, 5.0)


# ------------------------------------- the status line fits the method on screen

@pytest.mark.parametrize("method, wanted, unwanted", [
    ("parts", "Add Part", "Inhale"),
    ("landmarks", "Mark Inhale Start", "Label Cycle"),
    ("cycle", "Label Cycle Frames", "Inhale"),
])
def test_select_step_points_at_controls_that_are_on_screen(
        win, mat_file, method, wanted, unwanted):
    """It said 'Use buttons to mark Inhale Start and Exhale End' in every
    method, also where those buttons are not shown."""
    _ct_step_selected(win, mat_file, method)

    status = win.status_bar.currentMessage()

    assert wanted in status and unwanted not in status


def test_changing_the_method_replaces_a_status_line_about_the_old_one(win, mat_file):
    """'Added part: images 1-5' stayed on screen in the other methods."""
    _ct_step_selected(win, mat_file, "parts")
    win.part_from_spin.setValue(1)
    win.part_to_spin.setValue(3)
    win._add_part_from_spins()
    assert "part" in win.status_bar.currentMessage().lower()
    win._clear_parts()

    choose_ct_method(win, "landmarks")

    assert "part" not in win.status_bar.currentMessage().lower()
    assert "landmark" in win.status_bar.currentMessage().lower()


# --------------------------------------------------------- one word for a thing

def test_no_builder_word_reaches_the_user(win, mat_file):
    """'Fencepost' was the developers' word for a grid line."""
    texts = [text for _kind, text in gui_app._BANNERS.values()]
    texts.append(win.cycle_mode_btn.toolTip())
    _ct_step_selected(win, mat_file, "cycle")
    win.cycle_mode_btn.setChecked(True)
    texts += [win.status_bar.currentMessage(), win.mode_banner.text()]
    win._on_click(ClickEvent(win.ax_top, 3.3))
    texts.append(win.status_bar.currentMessage())

    assert not [t for t in texts if "fencepost" in t.lower()]


def test_a_cycle_click_says_which_image_it_took(win, mat_file):
    """A click takes the image that starts at the nearest grid line, which is
    not always the number under the pointer. The screen has to say which."""
    _ct_step_selected(win, mat_file, "cycle")
    win.cycle_mode_btn.setChecked(True)

    # 3.35 s lies in the slice numbered 7 (3.2 to 3.4 s), nearer its right
    # edge, where image 8 starts.
    win._on_click(ClickEvent(win.ax_top, 3.35))

    status = win.status_bar.currentMessage()
    assert win._ct_cycle_frames == [8]
    assert "image 8" in status and "grid line" in status
    assert "grid line" in win.mode_banner.text()


# ----------------------------------------- CT controls that explain themselves

def _visible_text(section):
    """Every label and button text of a sidebar section, as one string."""
    from PyQt5.QtWidgets import QAbstractButton, QLabel
    parts = [w.text() for w in section.findChildren(QLabel)]
    parts += [w.text() for w in section.findChildren(QAbstractButton)]
    return "\n".join(parts)


def test_the_ct_scan_section_says_what_its_controls_do(win):
    """Step #, Temporal Res, Total Images, Start Shift, a slider with no
    label and 'Apply Correction' (of what?) explained nothing on screen.
    Tooltips existed, but a new user does not know to hover."""
    win.workflow_combo.setCurrentText("CT")
    text = _visible_text(win.ct_scan_section)

    assert "time one image covers" in text.lower()          # Temporal Res
    assert "Images in step" in text                         # not a second 'Images'
    assert "Apply Correction" not in text
    assert "Apply shift and image count" in text
    assert "earlier" in text.lower() and "later" in text.lower()   # what the shift does


def test_the_two_image_counts_have_different_names(win):
    win.workflow_combo.setCurrentText("CT")

    assert "Images in step" in _visible_text(win.ct_scan_section)
    assert "Images in grid" in _visible_text(win.parts_page)


def test_the_parts_page_says_which_numbers_on_the_plot_are_which(win):
    """Two rows of image numbers were drawn, grey above the plot and blue
    inside it, with nothing to tell them apart."""
    win.workflow_combo.setCurrentText("CT")
    text = _visible_text(win.parts_page).lower()

    assert "grey numbers above the plot" in text and "exported" in text
    assert "blue numbers inside the plot" in text and "scan step" in text


def test_picking_a_part_speaks_of_images_in_plain_sentences(win, mat_file):
    _ct_step_selected(win, mat_file, "parts")
    win._arm_add_part()
    first = win.capture_hint.text()
    win._on_click(ClickEvent(win.ax_top, 2.1))
    second = win.capture_hint.text()
    win._on_click(ClickEvent(win.ax_top, 2.9))
    third = win.capture_hint.text()

    for text in (first, second, third):
        assert text and "—" not in text and "frame" not in text.lower()
    assert "image" in first.lower() and "Add Part" in third


# ------------------------------------------------ what was exported, and how

def _ready(win, mat_file, study="MRI"):
    win.workflow_combo.setCurrentText(study)
    open_block(win, mat_file)
    win.subject_edit.setText("S1")
    win.start_spin.setValue(2.0)
    win.end_spin.setValue(6.0)


def test_the_export_line_names_the_block_and_the_channels(win, mat_file):
    """'Ready to export' named the study and the window, never the block or
    the channels, which is where a wrong export comes from."""
    _ready(win, mat_file)

    hint = win.export_hint.text()

    assert hint.startswith("Ready to export: MRI, 2.000 s to 6.000 s.")
    assert "Block 1" in hint and "2: Flow" in hint and "4: Pressure" in hint


def test_the_export_line_says_when_no_pressure_is_written(win, mat_file):
    _ready(win, mat_file)
    win.pressure_row_spin.setValue(0)
    win._update_export_hint()
    assert "no pressure" in win.export_hint.text()

    win.workflow_combo.setCurrentText("Phase contrast (xenon)")
    assert "no pressure" in win.export_hint.text()


def test_the_result_records_where_the_numbers_came_from(win, mat_file):
    """The result panel listed files and sample counts, but not the
    recording, block, channels or the constants used. A CFD user needs
    exactly those before trusting a boundary condition."""
    _ready(win, mat_file)

    facts = dict(win._export_provenance())

    assert facts["Recording"] == "rec.mat"
    assert facts["Study type"] == "MRI"
    assert facts["Block"] == "1"
    assert facts["Flow channel"] == "2: Flow"
    assert facts["Pressure channel"] == "4: Pressure"
    assert "1.2 kg/m3" in facts["Mass flow"] and "sign" in facts["Mass flow"]
    assert "98.0665" in facts["Pressure"]


def test_the_result_states_the_xenon_density_for_phase_contrast(win, mat_file):
    _ready(win, mat_file, study="Phase contrast (xenon)")

    facts = dict(win._export_provenance())

    assert "5.761 kg/m3" in facts["Mass flow"]
    assert "Pressure channel" not in facts and "no pressure" in facts["Pressure"].lower()


def test_a_finished_export_shows_those_facts(win, qapp, mat_file, tmp_path):
    import time
    _ready(win, mat_file)
    win._set_output_dir(tmp_path / "out")

    win._export()
    deadline = time.time() + 30
    while win._last_export_report is None and time.time() < deadline:
        qapp.processEvents()
        time.sleep(0.01)

    labels = [label for label, _value in win._last_export_report["summary"]]
    assert labels[:4] == ["Recording", "Study type", "Block", "Flow channel"]
    if win._result_dialog is not None:
        win._result_dialog.close()


def test_each_study_type_states_its_gas_density(win):
    descriptions = {shown: text for shown, _engine, text in gui_app._STUDY_TYPES}

    assert "1.2 kg/m3" in descriptions["MRI"] and "1.2 kg/m3" in descriptions["CPAP"]
    assert "1.2 kg/m3" in descriptions["CT"]
    assert "5.761 kg/m3" in descriptions["Phase contrast (xenon)"]


def test_the_plot_names_its_channels_not_amplitude_and_info(win, mat_file):
    """The left axis said 'Amplitude' and the legend 'flow info' and
    'trigger info'."""
    open_block(win, mat_file)

    assert "Flow" in win.ax_top.get_ylabel() and "Amplitude" not in win.ax_top.get_ylabel()
    assert "Trigger" in win.ax_top_trig.get_ylabel()
    legend = [t.get_text() for t in win.ax_top.get_legend().get_texts()]
    assert not [t for t in legend if "info" in t]


# ----------------------------------------------------------- smaller points

def test_the_plot_toolbar_has_no_tool_that_can_wreck_the_layout(win):
    """Matplotlib's 'Configure subplots' and 'Edit axis' were on offer."""
    names = [action.text() for action in win.toolbar.actions()]

    assert "Subplots" not in names and "Customize" not in names
    assert "Pan" in names and "Zoom" in names and "Save" in names


def test_the_window_title_names_the_open_recording(win, mat_file):
    assert "rec.mat" not in win.windowTitle()

    win.open_recording(mat_file)

    assert "rec.mat" in win.windowTitle()


def test_the_pickers_are_off_until_a_recording_is_open(win, mat_file):
    """They read 'Channel 2', 'Channel 1', 'Block 3' as if something had
    been chosen."""
    pickers = (win.flow_row_spin, win.trigger_row_spin, win.pressure_row_spin, win.col_spin)
    assert not any(p.isEnabled() for p in pickers)

    win.open_recording(mat_file)

    assert all(p.isEnabled() for p in pickers)


def test_phase_contrast_switches_the_pressure_picker_off(win, mat_file):
    """It stayed on, and every export then warned that no pressure profile
    was produced."""
    win.open_recording(mat_file)

    win.workflow_combo.setCurrentText("Phase contrast (xenon)")
    assert not win.pressure_row_spin.isEnabled()
    assert "no pressure" in win.pressure_row_spin.toolTip().lower()

    win.workflow_combo.setCurrentText("MRI")
    assert win.pressure_row_spin.isEnabled()


def test_a_missing_pressure_profile_gives_one_reason_not_two(win, tmp_path):
    """'...is empty in this block, or this study type has no pressure
    output' left the user to guess which."""
    warnings = win._pressure_warnings(True, None, tmp_path / "none.csv")

    assert len(warnings) == 1 and " or " not in warnings[0]
    assert "empty" in warnings[0]


def test_the_triggers_section_says_what_it_is_for(win):
    text = _visible_text(win.trigger_section).lower()

    assert "optional" in text and "start and end" in text


def test_an_accepted_channel_warning_is_not_asked_again(win, mat_file, tmp_path, monkeypatch):
    """Every export asked 'Check Channels' again, with no way to say the
    choice is deliberate."""
    from PyQt5.QtWidgets import QMessageBox
    _ready(win, mat_file)
    win._set_output_dir(tmp_path / "out")
    win.pressure_row_spin.setValue(1)              # 'Trigger' used as pressure
    asked, ran = [], []
    monkeypatch.setattr(
        "labchart2cfd.gui.app.QMessageBox.question",
        staticmethod(lambda _p, title, *a, **k: (asked.append(title), QMessageBox.Yes)[1]),
    )
    monkeypatch.setattr(win, "_execute_processing", lambda: ran.append(1))

    win._export()
    win._export()
    assert asked.count("Check Channels") == 1 and len(ran) == 2

    win.pressure_row_spin.setValue(3)              # a different mismatch asks again
    win._update_channel_check()
    win._export()
    assert asked.count("Check Channels") == (2 if win._channel_name_warnings() else 1)


def test_the_help_menu_states_the_version(win, monkeypatch):
    import labchart2cfd
    shown = []
    monkeypatch.setattr(
        "labchart2cfd.gui.app.QMessageBox.about",
        staticmethod(lambda _p, title, text: shown.append(text)),
    )

    win.about_action.trigger()

    assert labchart2cfd.__version__ in shown[0]
