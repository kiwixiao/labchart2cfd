"""Tests for the usability review of the redesigned window (first time user).

These are not crashes. Each is a place where the screen led a new user to the
wrong action, or left them without the fact they needed.
"""
import pytest

from labchart2cfd.gui import app as gui_app
from tests.gui_helpers import ClickEvent, choose_ct_method, open_block, pick_drift_breath

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
    ("landmarks", "Mark inhale start"), ("cycle", "Label cycle frames"),
])
def test_the_banner_names_the_button_to_press_next(win, mat_file, method, button):
    _ct_step_selected(win, mat_file, method)

    banner = win.mode_banner.text()

    assert button in banner
    assert "set Start" not in banner


def test_selecting_a_step_removes_window_markers_that_no_longer_apply(win, mat_file):
    """A green marker stayed at a time that was no longer Start."""
    win.workflow_combo.setCurrentText("MRI")       # markers placed as an MRI window
    open_block(win, mat_file)
    win._on_click(ClickEvent(win.ax_bottom, 3.0))
    win._on_click(ClickEvent(win.ax_bottom, 4.0))
    win.workflow_combo.setCurrentText("CT")
    choose_ct_method(win, "landmarks")
    win._add_time_marker(3.0)                      # as if left over on the plot

    win._select_ct_step()

    assert win._time_markers == []
    assert (win.start_spin.value(), win.end_spin.value()) == (2.0, 5.0)


# ------------------------------------- the status line fits the method on screen

@pytest.mark.parametrize("method, wanted, unwanted", [
    ("parts", "Add part", "Inhale"),
    ("landmarks", "Mark inhale start", "Label Cycle"),
    ("cycle", "Label cycle frames", "Inhale"),
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


def test_a_cycle_click_takes_the_image_under_the_pointer(win, mat_file):
    """The owner's decision, 2026-11-20. A click used to take the image that
    starts at the nearest grid line, so the right half of the slice numbered
    7 recorded image 8, although the export cuts whole slices. The reorder
    parts method already took the slice under the pointer."""
    _ct_step_selected(win, mat_file, "cycle")       # step 2 to 5 s, 200 ms images
    win.cycle_mode_btn.setChecked(True)

    win._on_click(ClickEvent(win.ax_top, 3.21))    # image 7 runs from 3.2 to 3.4 s
    win._on_click(ClickEvent(win.ax_top, 3.39))
    win._on_click(ClickEvent(win.ax_top, 3.41))    # image 8

    assert win._ct_cycle_frames == [7, 7, 8]
    assert "image 8" in win.status_bar.currentMessage()
    assert "under the pointer" in win.mode_banner.text()


def test_cycle_clicks_at_the_edges_of_the_scan(win, mat_file):
    _ct_step_selected(win, mat_file, "cycle")
    win.cycle_mode_btn.setChecked(True)

    win._on_click(ClickEvent(win.ax_top, 2.0))     # the first instant of image 1
    win._on_click(ClickEvent(win.ax_top, 4.999))   # the last image, 15
    win._on_click(ClickEvent(win.ax_top, 1.5))     # before the scan
    win._on_click(ClickEvent(win.ax_top, 6.0))     # after the scan

    assert win._ct_cycle_frames == [1, 15]
    assert "outside the scan" in win.status_bar.currentMessage()


def test_a_cycle_label_sits_on_its_image_and_shades_it(win, mat_file):
    """The green circle was drawn on the line between two numbers, so the
    picture did not say which image it meant."""
    _ct_step_selected(win, mat_file, "cycle")
    win.cycle_mode_btn.setChecked(True)

    win._on_click(ClickEvent(win.ax_top, 3.3))     # image 7, 3.2 to 3.4 s

    shade, circle = win._ct_cycle_artists[-1]
    assert circle.xy[0] == pytest.approx(3.3)      # the middle of the image
    assert shade.get_x() == pytest.approx(3.2) and shade.get_width() == pytest.approx(0.2)


def test_removing_a_cycle_label_removes_its_shading_too(win, mat_file):
    _ct_step_selected(win, mat_file, "cycle")
    win.cycle_mode_btn.setChecked(True)
    win._on_click(ClickEvent(win.ax_top, 3.3))
    win._on_click(ClickEvent(win.ax_top, 3.5))
    shade, circle = win._ct_cycle_artists[-1]

    win._on_click(ClickEvent(win.ax_top, 3.0, button=3))

    assert win._ct_cycle_frames == [7] and len(win._ct_cycle_artists) == 1
    assert shade.axes is None and circle.axes is None


def test_clicking_backward_starts_a_new_run_of_images(win, qapp, mat_file, tmp_path):
    """What the method is for: click a breath in order, also backward, and
    the runs of images are cut out and joined in click order."""
    import time
    _ct_step_selected(win, mat_file, "cycle")
    win.subject_edit.setText("S1")
    win._set_output_dir(tmp_path / "out")
    win.cycle_mode_btn.setChecked(True)
    for t in (3.3, 3.5, 3.7, 2.1, 2.3):            # images 7, 8, 9, then back to 1, 2
        win._on_click(ClickEvent(win.ax_top, t))
    assert win._ct_cycle_frames == [7, 8, 9, 1, 2]

    win._export()
    deadline = time.time() + 30
    while win._last_export_report is None and time.time() < deadline:
        qapp.processEvents()
        time.sleep(0.01)

    facts = dict(win._last_export_report["summary"])
    assert facts["Images selected"] == "5 (7,8,9,1,2)"
    assert float(facts["Duration"].split()[0]) == pytest.approx(1.0, abs=0.02)   # 5 x 200 ms
    if win._result_dialog is not None:
        win._result_dialog.close()


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
    assert "Apply shift" in text and "Apply shift and image count" not in text
    assert "earlier" in text.lower() and "later" in text.lower()   # what the shift does


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
    assert "image" in first.lower() and "Add part" in third


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


# -------------------------------------- CT cannot be exported half set up

def test_ct_is_not_ready_until_a_scan_step_is_selected(win, mat_file):
    """The steps were drawn and 'Ready to export' appeared without one being
    selected. Following the banner then built a part on a grid of one image
    at 0 s, and the export wrote 0.2 s from the start of the recording."""
    win.workflow_combo.setCurrentText("CT")
    choose_ct_method(win, "landmarks")
    open_block(win, mat_file)
    win.subject_edit.setText("S1")
    win.start_spin.setValue(2.0)
    win.end_spin.setValue(5.0)

    hint = win.export_hint.text()
    assert hint.startswith("Before export") and "Select step" in hint

    win._select_ct_step()
    assert win.export_hint.text().startswith("Ready to export")


@pytest.mark.parametrize("method", ["parts", "landmarks", "cycle"])
def test_the_banner_says_to_select_the_step_first(win, mat_file, method):
    """In the parts method the banner explained why clicks do nothing, not
    what to do. The first action in every CT method is Select step."""
    win.workflow_combo.setCurrentText("CT")
    choose_ct_method(win, method)
    open_block(win, mat_file)

    assert "Select step" in win.mode_banner.text()

    win._on_click(ClickEvent(win.ax_bottom, 3.0))  # and a click moves nothing
    assert win._time_markers == []


def test_ct_without_any_step_can_still_take_a_window_by_hand(win, mat_file):
    """A recording whose trigger shows no step must not be a dead end."""
    win.workflow_combo.setCurrentText("CT")
    choose_ct_method(win, "landmarks")
    win.trigger_row_spin.setValue(0)
    open_block(win, mat_file)
    win.subject_edit.setText("S1")

    win._on_click(ClickEvent(win.ax_bottom, 3.0))
    win._on_click(ClickEvent(win.ax_bottom, 4.0))

    assert (win.start_spin.value(), win.end_spin.value()) == (3.0, 4.0)
    assert win.export_hint.text().startswith("Ready to export")


def test_a_shift_shown_but_not_applied_holds_the_export_back(win, mat_file):
    """The slider moved the grid on the plot. The export used the window
    without the shift and counted its images on a grid no longer drawn."""
    _ct_step_selected(win, mat_file, "landmarks")
    win.subject_edit.setText("S1")
    assert win.export_hint.text().startswith("Ready to export")

    win.ct_shift_slider.setValue(300)

    hint = win.export_hint.text()
    assert hint.startswith("Before export") and "press Apply shift (4 CT scan)" in hint

    win._apply_ct_correction()
    assert win.export_hint.text().startswith("Ready to export")
    assert win.start_spin.value() == pytest.approx(2.3)


def test_changing_temporal_res_is_not_a_pending_correction(win, mat_file):
    """The image count follows Temporal Res by itself. That is not an edit
    waiting for Apply."""
    _ct_step_selected(win, mat_file, "landmarks")
    win.subject_edit.setText("S1")

    win.ct_temporal_res_spin.setValue(150.0)

    assert win.ct_total_images_spin.value() == 20
    assert win.export_hint.text().startswith("Ready to export")


def test_the_export_line_states_the_temporal_res_it_will_use(win, mat_file):
    """200 ms sat in the box like a measured value. It is the user's to
    supply, and every image number follows from it."""
    _ct_step_selected(win, mat_file, "landmarks")
    win.subject_edit.setText("S1")

    assert "Temporal Res 200 ms" in win.export_hint.text()
    assert "15 images" in win.export_hint.text()


def test_the_export_line_names_the_window_rounded_to_whole_images(win, mat_file):
    """3 s at 180 ms is 16.67 images. The line used to say so while the
    export cut 17. The window now grows to the 17 whole images it holds
    (design 2026-11-22, 4.1), and the line names that window."""
    _ct_step_selected(win, mat_file, "landmarks")
    win.subject_edit.setText("S1")

    win.ct_temporal_res_spin.setValue(180.0)

    hint = win.export_hint.text()
    assert "2.000 s to 5.060 s" in hint
    assert "17 images" in hint
    assert "images long" not in hint


def test_the_ct_section_says_temporal_res_is_the_users_to_supply(win):
    win.workflow_combo.setCurrentText("CT")

    assert "cannot measure" in _visible_text(win.ct_scan_section).lower()


def test_a_ct_result_lists_the_images_it_selected(win, qapp, mat_file, tmp_path):
    """Inhale and exhale marked on the same image gave a one image result,
    and the panel showed nothing about which images were taken."""
    import time
    _ct_step_selected(win, mat_file, "landmarks")
    win.subject_edit.setText("S1")
    win._set_output_dir(tmp_path / "out")
    win._mark_inhale_start()
    win._on_click(ClickEvent(win.ax_bottom, 4.0))
    win._mark_exhale_end()
    win._on_click(ClickEvent(win.ax_bottom, 3.0))

    win._export()
    deadline = time.time() + 30
    while win._last_export_report is None and time.time() < deadline:
        qapp.processEvents()
        time.sleep(0.01)

    facts = dict(win._last_export_report["summary"])
    assert facts["Images in the scan"] == "15"
    assert facts["Inhale starts at image"] == "11"
    assert facts["Exhale ends at image"] == "6"
    assert facts["Images selected"].startswith("11 (")
    if win._result_dialog is not None:
        win._result_dialog.close()


# ------------------------------------ the plot and the numbers show one window

def _span(win, ax):
    """(left, right) of the shaded window on an axes, or None when hidden."""
    patch = win._window_spans[ax]
    if not patch.get_visible():
        return None
    return (round(patch.get_x(), 6), round(patch.get_x() + patch.get_width(), 6))


def test_the_window_is_shaded_wherever_start_and_end_are(win, mat_file):
    """Typing Start left the green marker at the old time, and clearing the
    markers left a window in force with nothing on the plot to show it."""
    open_block(win, mat_file)
    assert _span(win, win.ax_bottom) is None       # no window yet

    win.start_spin.setValue(3.0)
    win.end_spin.setValue(6.0)
    assert _span(win, win.ax_bottom) == (3.0, 6.0)
    assert _span(win, win.ax_top) == (3.0, 6.0)

    win.start_spin.setValue(3.5)
    assert _span(win, win.ax_bottom) == (3.5, 6.0)

    win.end_spin.setValue(1.0)                     # not a window
    assert _span(win, win.ax_bottom) is None


def test_the_shading_follows_clicks_and_survives_clear_markers(win, mat_file):
    open_block(win, mat_file)
    win._on_click(ClickEvent(win.ax_bottom, 2.0))
    win._on_click(ClickEvent(win.ax_bottom, 6.0))
    assert _span(win, win.ax_bottom) == (2.0, 6.0)

    win._clear_time_markers()

    assert (win.start_spin.value(), win.end_spin.value()) == (2.0, 6.0)
    assert _span(win, win.ax_bottom) == (2.0, 6.0)  # the window is still in force


def test_a_typed_time_removes_a_marker_that_no_longer_matches(win, mat_file):
    open_block(win, mat_file)
    win._on_click(ClickEvent(win.ax_bottom, 2.0))
    win._on_click(ClickEvent(win.ax_bottom, 6.0))
    assert len(win._time_markers) == 2

    win.start_spin.setValue(3.5)

    assert win._time_markers == []


def test_the_shading_is_redrawn_with_the_block(win, mat_file):
    open_block(win, mat_file)
    win.start_spin.setValue(3.0)
    win.end_spin.setValue(6.0)

    win._plot_selected()

    assert _span(win, win.ax_bottom) == (3.0, 6.0)


def test_the_shading_does_not_move_a_scale(win, mat_file):
    open_block(win, mat_file)
    before = (win.ax_top.get_xlim(), win.ax_top.get_ylim())

    win.start_spin.setValue(3.0)
    win.end_spin.setValue(600.0)                   # far past the recording

    assert (win.ax_top.get_xlim(), win.ax_top.get_ylim()) == before


def test_clear_markers_asks_before_it_removes_ct_marks(win, mat_file, monkeypatch):
    """It sits under Time window and is named for markers, but it also
    removed landmarks and cycle labels, the one control that did not ask."""
    from PyQt5.QtWidgets import QMessageBox
    _ct_step_selected(win, mat_file, "landmarks")
    win._mark_inhale_start()
    win._on_click(ClickEvent(win.ax_bottom, 4.0))
    asked = []
    monkeypatch.setattr(
        "labchart2cfd.gui.app.QMessageBox.question",
        staticmethod(lambda _p, title, text, *a, **k: (asked.append(text), QMessageBox.No)[1]),
    )

    win.clear_markers_btn.click()

    assert asked and "landmark" in asked[0]
    assert win._ct_inhale_start == 4.0


# ------------------------------------------------------ moving along the time axis

def test_the_wheel_over_the_plot_zooms_in_time_about_the_pointer(win, mat_file):
    """A long block opens as a solid band, and the only way in was to switch
    the zoom tool on and off. LabChart users expect the wheel to zoom."""
    from tests.test_gui_view import ScrollEvent
    open_block(win, mat_file)
    win.canvas.draw()
    box = win.ax_bottom.bbox
    x, y = box.x0 + 0.25 * box.width, (box.y0 + box.y1) / 2
    under_pointer = win.ax_bottom.transData.inverted().transform((x, y))[0]
    before = win.ax_bottom.get_xlim()

    win._on_scroll(ScrollEvent(x, y, step=2))

    after = win.ax_bottom.get_xlim()
    assert (after[1] - after[0]) < (before[1] - before[0])
    assert win.ax_bottom.transData.inverted().transform((x, y))[0] == pytest.approx(under_pointer)
    assert win.ax_top.get_xlim() == pytest.approx(after)       # both plots together


def test_the_wheel_cannot_zoom_out_past_the_recording(win, mat_file):
    from tests.test_gui_view import ScrollEvent
    open_block(win, mat_file)
    win.canvas.draw()
    box = win.ax_bottom.bbox
    full = win.ax_bottom.get_xlim()

    for _ in range(30):
        win._on_scroll(ScrollEvent((box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2, step=-5))

    assert win.ax_bottom.get_xlim() == pytest.approx(full)


def test_the_zoom_tool_keeps_the_wheel_to_itself(win, mat_file):
    from tests.test_gui_view import ScrollEvent
    open_block(win, mat_file)
    win.canvas.draw()
    box = win.ax_bottom.bbox
    before = win.ax_bottom.get_xlim()
    win.toolbar.zoom()

    win._on_scroll(ScrollEvent((box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2, step=2))

    assert win.ax_bottom.get_xlim() == before
    win.toolbar.zoom()


# ------------------------------------------------- ready looks different from not

def test_the_export_line_looks_different_when_ready(win, mat_file):
    """The same grey text said 'Before export' and 'Ready to export'."""
    assert win.export_hint.property("state") == "blocked"

    _ready(win, mat_file)

    assert win.export_hint.property("state") == "ready"


# ------------------- the retest: a green line must mean the plot is what is exported

def _grid_starts(win):
    """Times of the step grid lines drawn on the top plot."""
    lines = [a for a in win._ct_interval_markers if hasattr(a, "get_xdata") and a.axes is win.ax_top]
    return sorted(round(float(a.get_xdata()[0]), 6) for a in lines)


def test_the_step_grid_follows_the_window_after_an_applied_shift(win, mat_file):
    """Apply a shift, then change Temporal Res: the grid was redrawn from the
    step edge while the export used the shifted window, so a landmark sat on
    image 5 on the plot and image 4 in the file, under a green line."""
    _ct_step_selected(win, mat_file, "landmarks")
    win.ct_shift_slider.setValue(300)
    win._apply_ct_correction()
    assert win.start_spin.value() == pytest.approx(2.3)

    win.ct_temporal_res_spin.setValue(250.0)

    grid = _grid_starts(win)
    assert grid[0] == pytest.approx(2.3)                       # from Start, not the step edge
    assert len(grid) == win.ct_total_images_spin.value() + 1   # N images, N + 1 lines
    assert win.ct_total_images_spin.value() == 12              # 3 s at 250 ms
    assert "Correction applied" not in win.status_bar.currentMessage()


def test_the_step_grid_follows_a_typed_start(win, mat_file):
    """Typing Start moved the window. The plot kept 15 images from 2.0 s
    while the export wrote the images of the new window."""
    _ct_step_selected(win, mat_file, "landmarks")
    win.subject_edit.setText("S1")

    win.start_spin.setValue(3.0)

    # A new Start keeps the count and moves End (design 2026-11-22, 4.1).
    grid = _grid_starts(win)
    assert grid[0] == pytest.approx(3.0) and grid[-1] == pytest.approx(6.0)
    assert win.end_spin.value() == pytest.approx(6.0)
    assert win.ct_total_images_spin.value() == 15
    assert "15 images" in win.export_hint.text()


def test_a_step_number_that_is_not_the_selected_step_holds_the_export_back(win, tmp_path, mat_file):
    """Step # was changed to 2 without pressing Select step, and step 1 was
    exported under a box that said 2."""
    from tests.test_gui_view import _recording_with_steps
    two_steps = _recording_with_steps(mat_file, tmp_path, [(1.0, 2.0), (6.0, 8.0)])
    _ct_step_selected(win, two_steps, "landmarks")
    win.subject_edit.setText("S1")
    assert win.export_hint.text().startswith("Ready to export")

    win.ct_step_spin.setValue(2)

    hint = win.export_hint.text()
    assert hint.startswith("Before export") and "Select step" in hint and "step 1" in hint

    win._select_ct_step()
    assert win.export_hint.text().startswith("Ready to export")


def test_a_previewed_shift_holds_a_parts_export_back_too(win, mat_file):
    from labchart2cfd.processing.profile_assembly import ProfilePart
    _ct_step_selected(win, mat_file, "parts")
    win.subject_edit.setText("S1")
    win._parts.append(ProfilePart(start_image=1, end_image=3, label=""))
    win._refresh_parts_list()
    assert win.export_hint.text().startswith("Ready to export")

    win.ct_shift_slider.setValue(300)

    assert "press Apply shift (4 CT scan)" in win.export_hint.text()


def test_the_pending_blocker_says_what_is_in_force_and_where_the_button_is(win, mat_file):
    """'Put the value back' did not say what the value was."""
    _ct_step_selected(win, mat_file, "landmarks")
    win.subject_edit.setText("S1")

    win.ct_shift_slider.setValue(300)

    hint = win.export_hint.text()
    assert "300 ms on screen, 0 ms in force" in hint and "4 CT scan" in hint


# ------------------------------------------------ warnings are not shown in green

def test_a_new_temporal_res_leaves_no_partial_image_to_caution_about(win, mat_file):
    """'(the window is 26.67 images long)' was printed in the colour that
    means go. The window now grows to whole images (design 2026-11-22, 4.1),
    so there is no partial image left to caution about."""
    _ct_step_selected(win, mat_file, "landmarks")
    pick_drift_breath(win)   # so only this test's own caution can turn the line amber
    win.subject_edit.setText("S1")
    win._ct_dt_confirmed = True
    win._mark_inhale_start()
    win._on_click(ClickEvent(win.ax_bottom, 4.0))
    win._update_export_hint()
    assert win.export_hint.property("state") == "ready"

    win.ct_temporal_res_spin.setValue(180.0)

    assert win.end_spin.value() == pytest.approx(5.06)              # 17 whole images
    assert win.export_hint.property("state") == "ready"
    assert "images long" not in win.export_hint.text()


def test_a_ct_export_with_nothing_marked_is_amber(win, mat_file):
    _ct_step_selected(win, mat_file, "landmarks")
    win.subject_edit.setText("S1")
    win._ct_dt_confirmed = True
    win._update_export_hint()

    assert "no landmark marked" in win.export_hint.text()
    assert win.export_hint.property("state") == "caution"


def test_temporal_res_is_unconfirmed_until_the_user_touches_it(win, mat_file):
    """200 ms is the program's default, and a wrong value that divides the
    step evenly passed in green with nothing to notice."""
    _ct_step_selected(win, mat_file, "landmarks")
    pick_drift_breath(win)   # so only this test's own caution can turn the line amber
    win.subject_edit.setText("S1")
    win._mark_inhale_start()
    win._on_click(ClickEvent(win.ax_bottom, 4.0))

    hint = win.export_hint.text()
    assert win.export_hint.property("state") == "caution"
    assert "default" in hint and "confirm" in hint

    win.ct_temporal_res_spin.lineEdit().returnPressed.emit()   # Enter in the box

    assert win.export_hint.property("state") == "ready"
    assert "default" not in win.export_hint.text()


def test_another_recording_needs_its_temporal_res_confirmed_again(win, mat_file, tmp_path):
    import shutil
    _ct_step_selected(win, mat_file, "landmarks")
    win.ct_temporal_res_spin.lineEdit().returnPressed.emit()
    assert win._ct_dt_confirmed
    other = tmp_path / "other.mat"
    shutil.copy(mat_file, other)

    win.open_recording(other)

    assert not win._ct_dt_confirmed


# ------------------------------------------------------ the screen agrees with itself

def test_before_a_step_is_selected_no_ct_click_mode_can_be_armed(win, mat_file):
    """The banner said clicks do nothing, yet Pick on plot and Mark inhale
    start took clicks, and a part was built on a grid of one image."""
    win.workflow_combo.setCurrentText("CT")
    open_block(win, mat_file)

    win._arm_add_part()
    assert not win._capturing_part
    choose_ct_method(win, "landmarks")
    win._mark_inhale_start()
    assert win._ct_landmark_mode is None
    choose_ct_method(win, "cycle")
    win.cycle_mode_btn.setChecked(True)
    assert not win._ct_cycle_mode and not win.cycle_mode_btn.isChecked()
    assert "Select step" in win.status_bar.currentMessage()


def test_a_click_on_a_scan_step_selects_it(win, tmp_path, mat_file):
    """The step could only be chosen by number in a box."""
    from tests.test_gui_view import _recording_with_steps
    two_steps = _recording_with_steps(mat_file, tmp_path, [(1.0, 2.0), (6.0, 8.0)])
    win.workflow_combo.setCurrentText("CT")
    choose_ct_method(win, "landmarks")
    open_block(win, two_steps)
    assert "click" in win.mode_banner.text().lower()

    win._on_click(ClickEvent(win.ax_top, 7.0))     # inside the second step

    assert win.ct_step_spin.value() == 2
    assert (win.start_spin.value(), win.end_spin.value()) == (6.0, 8.0)

def test_a_click_outside_every_step_selects_nothing(win, mat_file):
    win.workflow_combo.setCurrentText("CT")
    open_block(win, mat_file)

    win._on_click(ClickEvent(win.ax_top, 8.0))

    assert win._ct_selected_step is None


def test_the_time_window_line_does_not_contradict_the_banner_in_ct(win, mat_file):
    """Banner: 'Clicks on the plot do nothing until then.' Time window:
    'Not set yet. Click the plot twice'."""
    win.workflow_combo.setCurrentText("CT")
    open_block(win, mat_file)

    assert "Click the plot twice" not in win.window_readout.text()
    assert "scan step" in win.window_readout.text().lower()


def test_a_right_click_cancels_an_armed_landmark(win, mat_file):
    """It answered 'Time markers cleared' and the banner stayed on 'Click
    the plot where inhale starts'."""
    _ct_step_selected(win, mat_file, "landmarks")
    win._mark_inhale_start()

    win._on_click(ClickEvent(win.ax_bottom, 3.0, button=3))

    assert win._ct_landmark_mode is None and win._ct_inhale_start is None
    assert "inhale starts" not in win.mode_banner.text()


def test_clear_markers_keeps_the_selected_step_and_its_grid(win, mat_file):
    """It removed the image grid and the step band without a question, while
    the banner still said a step was selected."""
    _ct_step_selected(win, mat_file, "landmarks")
    grid = len(win._ct_interval_markers)
    assert grid > 0

    win.clear_markers_btn.click()

    assert len(win._ct_interval_markers) == grid and win._ct_step_markers


def test_leaving_the_parts_method_removes_its_grid_from_the_plot(win, mat_file):
    _ct_step_selected(win, mat_file, "parts")
    win._draw_synth_grid(quiet=True)
    assert win._grid_artists

    choose_ct_method(win, "landmarks")

    assert win._grid_artists == []


def test_a_hand_window_removes_the_trigger_range_band(win, mat_file):
    """The green 'selected range' band stayed beside a window clicked later."""
    open_block(win, mat_file)
    win._mark_trigger_range()
    assert win._trigger_markers

    win._on_click(ClickEvent(win.ax_bottom, 6.0))

    assert win._trigger_markers == []


def test_pickers_that_are_off_look_off(win):
    """The palette gave disabled pickers the same text colour as live ones."""
    from labchart2cfd.gui.theme import STYLE_SHEET

    assert "QComboBox:disabled" in STYLE_SHEET


def test_the_result_rows_are_short_enough_to_read_whole(win, mat_file):
    """In the CT and phase contrast panels the conversion row was cut off."""
    win.workflow_combo.setCurrentText("Phase contrast (xenon)")
    open_block(win, mat_file)
    assert max(len(value) for _label, value in win._export_provenance()) <= 60

    win.workflow_combo.setCurrentText("MRI")
    assert max(len(value) for _label, value in win._export_provenance()) <= 60


# ---------------- second retest: one grid, and every mark is drawn from it

def _cycle_shades(win):
    """(start, width) of the shading of each cycle label, in label order."""
    return [(round(drawn[0].get_x(), 6), round(drawn[0].get_width(), 6))
            for drawn in win._ct_cycle_artists if drawn]   # empty: image no longer exists


def _two_cycle_labels(win, mat_file):
    """Images 5 and 6 of a step from 2 to 5 s at 200 ms: 2.8 to 3.0, 3.0 to 3.2."""
    _ct_step_selected(win, mat_file, "cycle")
    win.subject_edit.setText("S1")
    win.cycle_mode_btn.setChecked(True)
    win._on_click(ClickEvent(win.ax_top, 2.9))
    win._on_click(ClickEvent(win.ax_top, 3.1))
    win.cycle_mode_btn.setChecked(False)
    assert win._ct_cycle_frames == [5, 6]
    assert _cycle_shades(win) == [(2.8, 0.2), (3.0, 0.2)]


def test_cycle_labels_are_redrawn_when_temporal_res_changes(win, mat_file):
    """Labels are image numbers. After Temporal Res changed, the shading
    stayed where it was while the export cut the same numbers on the new
    grid: the plot shaded 2.8 to 3.2 s and the file held 2.4 to 2.6 s."""
    _two_cycle_labels(win, mat_file)

    win.ct_temporal_res_spin.setValue(100.0)

    assert win._ct_cycle_frames == [5, 6]
    assert _cycle_shades(win) == [(2.4, 0.1), (2.5, 0.1)]
    assert "cycle label" in win.status_bar.currentMessage().lower()


def test_cycle_labels_are_redrawn_when_a_shift_is_applied(win, mat_file):
    _two_cycle_labels(win, mat_file)

    win.ct_shift_slider.setValue(300)
    win._apply_ct_correction()

    assert _cycle_shades(win) == [(3.1, 0.2), (3.3, 0.2)]


def test_a_cycle_label_on_an_image_that_no_longer_exists_holds_the_export_back(win, mat_file):
    """Images 14 and 15 were labelled, then the scan was cut to 12 images.
    The line stayed green."""
    _ct_step_selected(win, mat_file, "cycle")
    win.subject_edit.setText("S1")
    win.cycle_mode_btn.setChecked(True)
    win._on_click(ClickEvent(win.ax_top, 4.7))     # image 14
    win._on_click(ClickEvent(win.ax_top, 4.9))     # image 15
    win.cycle_mode_btn.setChecked(False)

    win.ct_total_images_spin.setValue(12)
    win._apply_ct_correction()

    hint = win.export_hint.text()
    assert hint.startswith("Before export") and "14" in hint and "12 images" in hint
    assert len(win._ct_cycle_artists) == 2          # still two labels to remove
    assert _cycle_shades(win) == []                  # but nothing false is shaded


def test_a_cycle_click_counts_from_the_window_in_force(win, mat_file):
    """With a typed window the grid started at Start, but the click and its
    shading were counted from the step edge, so a click in the drawn image 1
    was recorded as image 2 and shaded across two drawn images."""
    _ct_step_selected(win, mat_file, "cycle")
    win.start_spin.setValue(2.09)
    win.end_spin.setValue(5.09)
    win.cycle_mode_btn.setChecked(True)

    win._on_click(ClickEvent(win.ax_top, 2.25))

    assert win._ct_cycle_frames == [1]
    assert _cycle_shades(win) == [(2.09, 0.2)]


@pytest.mark.parametrize("touch", [
    lambda win: win._on_shift_entry_changed(),                         # box entered and left
    lambda win: (win.ct_shift_slider.setValue(200), win.ct_shift_slider.setValue(0)),
], ids=["shift box left unchanged", "slider moved and moved back"])
def test_touching_the_shift_without_changing_it_keeps_the_grid_in_force(win, mat_file, touch):
    """Either one redrew the grid from the step edge under a green line."""
    _ct_step_selected(win, mat_file, "landmarks")
    win.start_spin.setValue(2.1)
    win.end_spin.setValue(5.1)
    assert _grid_starts(win)[0] == pytest.approx(2.1)

    touch(win)

    assert _grid_starts(win)[0] == pytest.approx(2.1)


def test_the_landmark_status_names_the_image_the_export_will_cut(win, mat_file):
    """After an applied shift of 100 ms the status said image 7 while the
    plot and the result panel said image 6: it counted from the step edge."""
    _ct_step_selected(win, mat_file, "landmarks")
    win.ct_shift_slider.setValue(100)
    win._apply_ct_correction()

    win._mark_inhale_start()
    win._on_click(ClickEvent(win.ax_bottom, 3.12))

    assert "image 6 at 3.100" in win.status_bar.currentMessage()


def test_a_ready_cycle_export_is_not_refused_afterwards(win, qapp, mat_file, tmp_path, monkeypatch):
    """With Start typed after labelling, the line said Ready and Export then
    answered 'Grid Out of Sync', naming a button that no longer exists."""
    import time
    _two_cycle_labels(win, mat_file)
    win._set_output_dir(tmp_path / "out")
    win.start_spin.setValue(2.4)                   # labels follow: images 5, 6 from 3.2 s
    refused = []
    monkeypatch.setattr(
        "labchart2cfd.gui.app.QMessageBox.critical",
        staticmethod(lambda _p, title, text, *a, **k: refused.append(title)),
    )
    assert win.export_hint.text().startswith("Ready to export")

    win._export()
    deadline = time.time() + 30
    while win._last_export_report is None and not refused and time.time() < deadline:
        qapp.processEvents()
        time.sleep(0.01)

    assert refused == [] and win._last_export_report is not None
    if win._result_dialog is not None:
        win._result_dialog.close()


# ------------------------------------------------------ cautions that were missing

def test_reorder_parts_also_warns_about_a_default_temporal_res(win, mat_file):
    """Nothing in this method warned about Temporal Res at all."""
    from labchart2cfd.processing.profile_assembly import ProfilePart
    _ct_step_selected(win, mat_file, "parts")
    pick_drift_breath(win)   # so only this test's own caution can turn the line amber
    win.subject_edit.setText("S1")
    win._parts.append(ProfilePart(start_image=1, end_image=3, label=""))
    win._refresh_parts_list()

    assert win.export_hint.property("state") == "caution"
    assert "default" in win.export_hint.text()

    win.ct_temporal_res_spin.lineEdit().returnPressed.emit()
    assert win.export_hint.property("state") == "ready"


def test_focus_leaving_the_box_does_not_confirm_temporal_res(win, mat_file):
    """Tabbing through section 4, or clicking the box to read it, cleared
    the warning."""
    _ct_step_selected(win, mat_file, "landmarks")

    win.ct_temporal_res_spin.editingFinished.emit()            # focus left, value untouched

    assert not win._ct_dt_confirmed


def test_a_parts_grid_that_is_not_the_scan_is_amber(win, mat_file):
    """The grid may sit off the scan on purpose (Grid offset). But a grid
    that did not cover the scan passed in green."""
    from labchart2cfd.processing.profile_assembly import ProfilePart
    _ct_step_selected(win, mat_file, "parts")
    pick_drift_breath(win)   # so only this test's own caution can turn the line amber
    win.subject_edit.setText("S1")
    win.ct_temporal_res_spin.lineEdit().returnPressed.emit()
    win._parts.append(ProfilePart(start_image=1, end_image=3, label=""))
    win._refresh_parts_list()
    assert win.export_hint.property("state") == "ready"

    win.synth_grid_offset_spin.setValue(200)       # the grid 0.2 s later than the scan

    hint = win.export_hint.text()
    assert win.export_hint.property("state") == "caution"
    assert "2.200 s to 5.200 s" in hint and "2.000 s to 5.000 s" in hint


def test_selecting_another_step_counts_its_images_for_the_parts_grid(win, mat_file, tmp_path):
    """The grid's count was set from the first step selected and never
    again, so a second, shorter scan was cut with the first one's count and
    the export ran past its end under a green line."""
    from tests.test_gui_view import _recording_with_steps
    two_steps = _recording_with_steps(mat_file, tmp_path, [(1.0, 3.0), (6.0, 7.0)])
    _ct_step_selected(win, two_steps, "parts")
    assert win.synth_images_spin.value() == 10

    win.ct_step_spin.setValue(2)
    win._select_ct_step()

    assert win.synth_images_spin.value() == 5


def test_landmarks_on_the_same_image_are_amber_and_say_so(win, mat_file):
    """Inhale and exhale marked at the same zero crossing select one image
    out of 15, under a green line."""
    _ct_step_selected(win, mat_file, "landmarks")
    win.subject_edit.setText("S1")
    win.ct_temporal_res_spin.lineEdit().returnPressed.emit()
    win._mark_inhale_start()
    win._on_click(ClickEvent(win.ax_bottom, 4.02))
    win._mark_exhale_end()
    win._on_click(ClickEvent(win.ax_bottom, 4.04))

    assert win.export_hint.property("state") == "caution"
    assert "same image" in win.export_hint.text()


def test_refusing_and_cautioning_do_not_share_a_colour():
    """Both were the same amber and differed only in their first words."""
    from labchart2cfd.gui.theme import EXPORT_HINT_COLORS

    assert len(set(EXPORT_HINT_COLORS.values())) == 3
    assert set(EXPORT_HINT_COLORS) == {"blocked", "caution", "ready"}


# ------------------------------------------------------------ layout leftovers

def test_the_result_panel_shows_every_row_whole(qapp, tmp_path):
    """In the reorder parts panel the image order stopped after '6,' and
    another row was cut mid sentence: wrapped labels in a form lose height."""
    from labchart2cfd.gui.dialogs import ExportResultDialog
    order = ",".join(str(i) for i in list(range(6, 21)) + list(range(1, 6)))
    dialog = ExportResultDialog({
        "folder": tmp_path, "files": [tmp_path / "S1FlowProfile.csv"],
        "summary": [("Image order (original numbers)", order),
                    ("Time", "starts at 0 in the profile; the reorder map keeps recording time"),
                    ("Odd <text> & more", "a < b")],
        "warnings": [], "plot": None,
    })
    dialog.show()
    qapp.processEvents()

    text = dialog.summary_label.text()
    assert order in text and "keeps recording time" in text
    assert "a &lt; b" in text                      # file text is shown, not read as markup
    needed = dialog.summary_label.heightForWidth(dialog.summary_label.width())
    assert dialog.summary_label.height() >= needed
    dialog.close()


def test_a_short_method_page_does_not_keep_the_height_of_the_long_one(win, qapp):
    """Section 5 kept 800 px for the landmarks page, the height of the parts
    page, so blank white followed its three buttons."""
    win.show()
    win.workflow_combo.setCurrentText("CT")
    qapp.processEvents()
    parts_height = win.ct_method_stack.height()

    choose_ct_method(win, "landmarks")
    qapp.processEvents()

    assert win.ct_method_stack.height() < parts_height * 0.6
    assert win.ct_method_stack.currentWidget() is win.landmarks_page


def test_the_banner_says_how_to_use_a_window_that_is_not_a_step(win, mat_file):
    """Steps were found, the user wanted another window, and nothing said how."""
    win.workflow_combo.setCurrentText("CT")
    open_block(win, mat_file)

    assert "Trigger to None" in win.mode_banner.text()


# -------- the user's report: the image count of a detected step is theirs to set

def _type_images(win, n):
    """Type a count into Images in step, as a user does (not a program)."""
    win.ct_total_images_spin.setValue(n)


def test_a_typed_image_count_survives_a_temporal_res_change(win, mat_file):
    """The count was recomputed from the window whenever the grid was drawn,
    so a lower number snapped back up to the one that covers the step. The
    user: 'I do want to be free to put those numbers even in the auto detect
    step setting.' Temporal Res and the count are independent inputs."""
    _ct_step_selected(win, mat_file, "landmarks")   # 3.0 s at 200 ms: 15 images
    _type_images(win, 9)

    win.ct_temporal_res_spin.setValue(190.0)

    assert win.ct_total_images_spin.value() == 9


def test_a_typed_image_count_survives_a_typed_start(win, mat_file):
    _ct_step_selected(win, mat_file, "landmarks")
    _type_images(win, 9)

    win.start_spin.setValue(2.05)

    assert win.ct_total_images_spin.value() == 9


def test_an_applied_image_count_keeps_its_number_when_temporal_res_changes(win, mat_file):
    """After Apply the window is N images long. Changing Temporal Res turned
    9 into 11. Now the count stays and the window follows: N times the new
    Temporal Res."""
    _ct_step_selected(win, mat_file, "landmarks")
    win.subject_edit.setText("S1")
    _type_images(win, 9)
    win._apply_ct_correction()
    assert win.end_spin.value() == pytest.approx(3.8)

    win.ct_temporal_res_spin.setValue(150.0)

    assert win.ct_total_images_spin.value() == 9
    assert win.end_spin.value() == pytest.approx(2.0 + 9 * 0.15)
    assert len(_grid_starts(win)) == 10                          # 9 images, 10 lines
    assert "Apply shift" not in win.export_hint.text()           # nothing is pending
    assert "9 images" in win.export_hint.text()


def test_an_applied_image_count_moves_with_a_typed_start(win, mat_file):
    _ct_step_selected(win, mat_file, "landmarks")
    _type_images(win, 9)
    win._apply_ct_correction()

    win.start_spin.setValue(2.5)

    assert win.ct_total_images_spin.value() == 9
    assert win.end_spin.value() == pytest.approx(2.5 + 9 * 0.2)


def test_the_count_may_be_longer_than_the_step(win, mat_file):
    _ct_step_selected(win, mat_file, "landmarks")
    _type_images(win, 20)                           # 4.0 s, the step is 3.0 s

    win._apply_ct_correction()

    assert win.ct_total_images_spin.value() == 20
    assert win.end_spin.value() == pytest.approx(6.0)


def test_typing_end_hands_the_count_back_to_the_window(win, mat_file):
    """The last thing the user set wins. A typed End defines the window, and
    the count shows what that window holds."""
    _ct_step_selected(win, mat_file, "landmarks")
    _type_images(win, 9)
    win._apply_ct_correction()

    win.end_spin.setValue(5.0)

    assert win.ct_total_images_spin.value() == 15


def test_selecting_the_step_again_counts_its_images_afresh(win, mat_file):
    _ct_step_selected(win, mat_file, "landmarks")
    _type_images(win, 9)
    win._apply_ct_correction()

    win._select_ct_step()

    assert win.ct_total_images_spin.value() == 15
    assert win.end_spin.value() == pytest.approx(5.0)

    win.ct_temporal_res_spin.setValue(150.0)       # and the count is the program's again
    assert win.ct_total_images_spin.value() == 20


def test_the_section_says_the_count_is_yours_to_set(win):
    win.workflow_combo.setCurrentText("CT")

    text = _visible_text(win.ct_scan_section).lower()
    assert "images in step" in text and "your own" in text


# ------------------------ third retest: the narrow cases left under a green line

def test_a_landmark_left_outside_the_window_holds_the_export_back(win, mat_file):
    """Inhale was marked, then Start was typed past it. The mark was off
    screen and the file said inhale_image 1, under a green line. Cycle
    labels were refused in the same situation."""
    _ct_step_selected(win, mat_file, "landmarks")
    win.subject_edit.setText("S1")
    win._mark_inhale_start()
    win._on_click(ClickEvent(win.ax_bottom, 2.42))

    win.start_spin.setValue(3.0)

    hint = win.export_hint.text()
    assert hint.startswith("Before export") and "inhale" in hint and "2.420 s" in hint

    win._clear_ct_landmarks()
    win._update_export_hint()
    assert win.export_hint.text().startswith("Ready to export")


def test_an_exhale_landmark_past_a_shortened_scan_holds_the_export_back(win, mat_file):
    _ct_step_selected(win, mat_file, "landmarks")
    win.subject_edit.setText("S1")
    win._mark_exhale_end()
    win._on_click(ClickEvent(win.ax_bottom, 4.9))

    win.ct_total_images_spin.setValue(12)          # the scan now ends at 4.4 s
    win._apply_ct_correction()

    assert "exhale" in win.export_hint.text() and win.export_hint.text().startswith("Before export")


def _ct_hand_window(win, mat_file, method):
    """CT on a recording with no usable trigger: the window is set by hand."""
    win.workflow_combo.setCurrentText("CT")
    choose_ct_method(win, method)
    win.trigger_row_spin.setValue(0)
    open_block(win, mat_file)
    win.subject_edit.setText("S1")
    win.start_spin.setValue(2.0)
    win.end_spin.setValue(5.0)


def test_a_hand_window_gets_an_image_grid_too(win, mat_file):
    """The banner sends users to Trigger None when no step fits. On that
    route no grid was drawn, yet the file named image numbers."""
    _ct_hand_window(win, mat_file, "landmarks")

    grid = _grid_starts(win)
    assert grid[0] == pytest.approx(2.0) and len(grid) == 16      # 15 images, 16 lines
    assert win.ct_total_images_spin.value() == 15

    win._mark_inhale_start()
    win._on_click(ClickEvent(win.ax_bottom, 3.12))
    assert "image 7 at 3.200" in win.status_bar.currentMessage()


def test_cycle_frames_work_on_a_hand_window(win, mat_file):
    """It answered 'No Step: Select a CT step' where there was no step to
    select, with the button left pressed."""
    _ct_hand_window(win, mat_file, "cycle")
    win.cycle_mode_btn.setChecked(True)

    win._on_click(ClickEvent(win.ax_top, 2.9))

    assert win._ct_cycle_frames == [5]
    assert _cycle_shades(win) == [(2.8, 0.2)]


def test_reorder_parts_on_a_hand_window_takes_clicks_for_the_window(win, mat_file):
    """With no step and no window the plot took no clicks at all, and the
    grid held one image until a count was typed."""
    win.workflow_combo.setCurrentText("CT")
    win.trigger_row_spin.setValue(0)
    open_block(win, mat_file)

    win._on_click(ClickEvent(win.ax_bottom, 2.0))
    win._on_click(ClickEvent(win.ax_bottom, 5.0))

    assert (win.start_spin.value(), win.end_spin.value()) == (2.0, 5.0)
    assert win.synth_images_spin.value() == 15


def test_selecting_another_step_asks_before_dropping_the_parts_of_this_one(
        win, mat_file, tmp_path, monkeypatch):
    """Parts built on one scan were carried to the next without a question,
    where cycle labels and landmarks ask."""
    from PyQt5.QtWidgets import QMessageBox
    from tests.test_gui_view import _recording_with_steps
    two_steps = _recording_with_steps(mat_file, tmp_path, [(1.0, 3.0), (6.0, 7.0)])
    _ct_step_selected(win, two_steps, "parts")
    win.part_from_spin.setValue(1)
    win.part_to_spin.setValue(10)
    win._add_part_from_spins()
    asked = []

    def _answer(reply):
        monkeypatch.setattr(
            "labchart2cfd.gui.app.QMessageBox.question",
            staticmethod(lambda _p, title, text, *a, **k: (asked.append(text), reply)[1]),
        )

    win.ct_step_spin.setValue(2)
    _answer(QMessageBox.No)
    win.select_step_btn.click()
    assert "1 part" in asked[0] and len(win._parts) == 1
    assert win._ct_selected_step["index"] == 1

    _answer(QMessageBox.Yes)
    win.select_step_btn.click()
    assert win._parts == [] and win._ct_selected_step["index"] == 2


def test_a_part_that_names_images_past_the_grid_holds_the_export_back(win, mat_file):
    """The line said Ready and Export answered 'Assembly Failed: Each part
    must satisfy ... <= 18; got 1..20.'"""
    from labchart2cfd.processing.profile_assembly import ProfilePart
    _ct_step_selected(win, mat_file, "parts")      # 15 images in the grid
    win.subject_edit.setText("S1")
    win._parts.append(ProfilePart(start_image=1, end_image=20, label=""))
    win._refresh_parts_list()
    win._update_export_hint()

    hint = win.export_hint.text()
    assert hint.startswith("Before export") and "15 images" in hint and "1 to 20" in hint


def test_the_export_line_follows_the_grid_offset(win, mat_file):
    """The line stayed as it was until some other edit refreshed it."""
    from labchart2cfd.processing.profile_assembly import ProfilePart
    _ct_step_selected(win, mat_file, "parts")
    pick_drift_breath(win)   # so only this test's own caution can turn the line amber
    win.subject_edit.setText("S1")
    win.ct_temporal_res_spin.lineEdit().returnPressed.emit()
    win._parts.append(ProfilePart(start_image=1, end_image=3, label=""))
    win._refresh_parts_list()
    win._update_export_hint()
    assert win.export_hint.property("state") == "ready"

    win.synth_grid_offset_spin.setValue(100)

    assert win.export_hint.property("state") == "caution"
    assert "2.100 s to 5.100 s" in win.export_hint.text()


def test_a_window_outside_the_recording_is_named_before_export(win, mat_file):
    """'Window Outside Recording' came only after a line that began Ready."""
    open_block(win, mat_file)                      # the block is 10 s long
    win.subject_edit.setText("S1")
    win.start_spin.setValue(8.0)

    win.end_spin.setValue(64.0)

    hint = win.export_hint.text()
    assert hint.startswith("Before export") and "recording" in hint and "9.999" in hint


def test_a_parts_grid_outside_the_recording_is_named_before_export(win, mat_file):
    from labchart2cfd.processing.profile_assembly import ProfilePart
    _ct_step_selected(win, mat_file, "parts")
    win.subject_edit.setText("S1")
    win._parts.append(ProfilePart(start_image=1, end_image=3, label=""))
    win._refresh_parts_list()

    win.synth_images_spin.setValue(250)            # 50 s of grid in a 10 s block

    hint = win.export_hint.text()
    assert hint.startswith("Before export") and "recording" in hint


def test_no_cycle_label_can_be_placed_on_a_grid_that_is_only_previewed(win, mat_file):
    """With a shift on screen and not applied, the numbers drawn were the
    preview while the click took an image of the grid in force."""
    _ct_step_selected(win, mat_file, "cycle")
    win.cycle_mode_btn.setChecked(True)
    win.ct_shift_slider.setValue(300)

    win._on_click(ClickEvent(win.ax_top, 2.95))

    assert win._ct_cycle_frames == []
    assert "Press Apply shift (4 CT scan)" in win.status_bar.currentMessage()


def test_the_blocker_about_lost_cycle_labels_names_the_way_out(win, mat_file):
    _ct_step_selected(win, mat_file, "cycle")
    win.subject_edit.setText("S1")
    win.cycle_mode_btn.setChecked(True)
    win._on_click(ClickEvent(win.ax_top, 4.9))
    win.cycle_mode_btn.setChecked(False)

    win.end_spin.setValue(4.0)

    assert "Clear cycle labels" in win.export_hint.text()


def test_moved_cycle_labels_are_announced_after_a_shift_and_a_typed_start(win, mat_file):
    """Only a Temporal Res change said 'check them'."""
    _two_cycle_labels(win, mat_file)
    win.ct_shift_slider.setValue(300)
    win._apply_ct_correction()
    assert "cycle label" in win.status_bar.currentMessage().lower()

    win.start_spin.setValue(2.5)
    assert "cycle label" in win.status_bar.currentMessage().lower()


def test_a_long_result_summary_is_given_the_height_it_needs(qapp, tmp_path):
    """The single summary label was 368 px for text that needs 384, in every
    study type, so the last row was sliced through."""
    from labchart2cfd.gui.dialogs import ExportResultDialog
    rows = [("Recording", "rec.mat"), ("Study type", "CT"), ("Block", "1"),
            ("Flow channel", "2: Flow"), ("Pressure channel", "4: Pressure"),
            ("Mass flow", "L/s x 0.001 x 1.2 kg/m3 (air), sign inverted"),
            ("Pressure", "cmH2O x 98.0665 = Pa"), ("Images in the scan", "15"),
            ("Inhale starts at image", "11"), ("Exhale ends at image", "6"),
            ("Images selected", "11 (11,12,13,14,15,1,2,3,4,5,6)"),
            ("Duration", "3.00 s"), ("Sample rate", "100 Hz"), ("Samples", "301"),
            ("Drift error", "0.214545"), ("Time", "starts at 0, as STAR-CCM+ expects")]
    dialog = ExportResultDialog({"folder": tmp_path, "files": [tmp_path / "a.csv"] * 4,
                                 "summary": rows, "warnings": [], "plot": None})
    dialog.show()
    qapp.processEvents()

    label = dialog.summary_label
    assert label.height() >= label.heightForWidth(label.width())
    assert label.minimumHeight() >= label.heightForWidth(dialog.PLOT_WIDTH)
    dialog.close()


# -------- the user's choice (2026-11-21): one image count, in force at once

def _hand_window(win, mat_file):
    """A recording read with no trigger: the window is set by two clicks."""
    win.workflow_combo.setCurrentText("CT")
    win.trigger_row_spin.setValue(0)
    open_block(win, mat_file)
    win._on_click(ClickEvent(win.ax_bottom, 2.0))
    win._on_click(ClickEvent(win.ax_bottom, 5.0))


def test_a_typed_image_count_takes_effect_at_once(win, mat_file):
    """The user typed 6 into a step of 9 and the plot kept 9. The count was
    only pending until Apply shift and image count, a button out of view."""
    _ct_step_selected(win, mat_file, "landmarks")   # 2.0 s to 5.0 s at 200 ms: 15 images
    win.subject_edit.setText("S1")

    _type_images(win, 9)

    assert win.end_spin.value() == pytest.approx(3.8)
    assert len(_grid_starts(win)) == 10                 # 9 images drawn, 10 lines
    hint = win.export_hint.text()
    assert hint.startswith("Ready to export") and "Apply" not in hint


def test_the_parts_grid_holds_the_same_count(win, mat_file):
    """Reorder parts cut its own count, set once from the step. After 6 was
    applied in section 4 the export still processed and cut 9 images."""
    _ct_step_selected(win, mat_file, "parts")

    _type_images(win, 9)

    assert win.synth_images_spin.value() == 9
    assert win._synth_export_window() == pytest.approx((2.0, 3.8))


def test_a_count_typed_under_build_the_breath_is_the_same_count(win, mat_file):
    _ct_step_selected(win, mat_file, "parts")

    win.synth_images_spin.setValue(6)

    assert win.ct_total_images_spin.value() == 6
    assert win.end_spin.value() == pytest.approx(3.2)
    assert len(_grid_starts(win)) == 7


def test_both_boxes_carry_one_name_and_one_range(win):
    win.workflow_combo.setCurrentText("CT")

    assert "Images in grid" not in _visible_text(win.parts_page)
    assert "Images in step" in _visible_text(win.parts_page)
    assert (win.synth_images_spin.minimum(), win.synth_images_spin.maximum()) == (
        win.ct_total_images_spin.minimum(), win.ct_total_images_spin.maximum())


def test_selecting_a_step_counts_both_boxes_afresh(win, mat_file, tmp_path):
    from tests.test_gui_view import _recording_with_steps
    two_steps = _recording_with_steps(mat_file, tmp_path, [(1.0, 3.0), (6.0, 7.0)])
    _ct_step_selected(win, two_steps, "parts")
    win.synth_images_spin.setValue(8)

    win.ct_step_spin.setValue(2)
    win._select_ct_step()

    assert win.ct_total_images_spin.value() == 5
    assert win.synth_images_spin.value() == 5
    assert (win.start_spin.value(), win.end_spin.value()) == (6.0, 7.0)


def test_a_typed_count_sets_the_length_of_a_hand_window(win, mat_file):
    """With no step on the trigger the window is set by hand. It has the
    same one count: typed, End moves to fit it."""
    _hand_window(win, mat_file)
    assert win.ct_total_images_spin.value() == 15

    _type_images(win, 9)

    assert win.end_spin.value() == pytest.approx(3.8)
    assert win.synth_images_spin.value() == 9
    assert len(_grid_starts(win)) == 10


def test_a_typed_end_hands_the_count_of_a_hand_window_back(win, mat_file):
    _hand_window(win, mat_file)
    _type_images(win, 9)

    win.end_spin.setValue(5.0)

    assert win.ct_total_images_spin.value() == 15
    assert win.synth_images_spin.value() == 15
