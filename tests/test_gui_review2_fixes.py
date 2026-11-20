"""Tests for what the second independent review pass proved.

Most of it was in code one day old: the y scale controls and the comments.
The rest is one rule applied everywhere: nothing the user placed on the plot
is removed without a question, whichever control triggers it.
"""
import shutil
import warnings

import numpy as np
import pytest
from matplotlib.text import Text
from PyQt5.QtWidgets import QMessageBox
from scipy.io import loadmat, savemat

from labchart2cfd.processing.profile_assembly import ProfilePart
from tests.gui_helpers import ClickEvent, choose_ct_method, open_block
from tests.test_gui_view import ScrollEvent, _strip_point

pytestmark = pytest.mark.usefixtures("no_modal_dialogs")


def _answer(monkeypatch, answer, asked):
    def _question(_parent, title, text, *_args, **_kwargs):
        asked.append(f"{title}: {text}")
        return answer
    monkeypatch.setattr(
        "labchart2cfd.gui.app.QMessageBox.question", staticmethod(_question)
    )


def _changed(mat_file, tmp_path, name="changed.mat", channel=1, block=1, edit=None, **extra):
    """A copy of the fixture export with one channel block edited in place."""
    mat = {k: v for k, v in loadmat(str(mat_file)).items() if not k.startswith("__")}
    if edit is not None:
        data = mat["data"].ravel().copy()
        first = int(mat["datastart"][channel - 1, block - 1]) - 1
        stop = int(mat["dataend"][channel - 1, block - 1])
        data[first:stop] = edit(data[first:stop])
        mat["data"] = data
    mat.update(extra)
    path = tmp_path / name
    savemat(str(path), mat)
    return path


def _with_comments(mat_file, tmp_path, times, texts):
    width = max(len(t) for t in texts)
    return _changed(
        mat_file, tmp_path, name="comments.mat",
        com=np.array([[-1, 1, round(t * 1000), 1, i + 1] for i, t in enumerate(times)],
                     dtype=float),
        comtext=np.array([t.ljust(width) for t in texts]),
        tickrate=np.full((1, 3), 1000.0),
    )


def _landmarks_marked(win, mat_file):
    win.workflow_combo.setCurrentText("CT")
    choose_ct_method(win, "landmarks")
    open_block(win, mat_file)
    win._select_ct_step()
    win._mark_inhale_start()
    win._on_click(ClickEvent(win.ax_bottom, 4.0))
    win.subject_edit.setText("S1")


# ------------------------------------------------ signals with NaN or infinity

def _poke(value, at=3000):
    def _edit(samples):
        samples = samples.copy()
        samples[at] = value
        return samples
    return _edit


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_a_trigger_with_a_bad_sample_still_opens(win, mat_file, tmp_path, value):
    """One NaN in the trigger made the block fail to open, where the old
    window opened it."""
    open_block(win, _changed(mat_file, tmp_path, edit=_poke(value)))

    lo, hi = win.ax_top_trig.get_ylim()
    assert win._plot_mode == "selected" and win._click_cid is not None
    assert np.isfinite([lo, hi]).all() and lo < 0.0 and hi > 5.0


def test_a_trigger_of_only_nan_still_opens(win, mat_file, tmp_path):
    open_block(win, _changed(mat_file, tmp_path, edit=lambda s: np.full_like(s, np.nan)))

    assert win._plot_mode == "selected"
    assert np.isfinite(win.ax_top_trig.get_ylim()).all()


def test_zoom_to_window_with_a_nan_in_the_flow(win, mat_file, tmp_path):
    open_block(win, _changed(mat_file, tmp_path, channel=2, edit=_poke(np.nan, at=2500)))
    win.start_spin.setValue(2.2)
    win.end_spin.setValue(2.8)

    win._zoom_to_window()

    lo, hi = win.ax_top.get_ylim()
    assert 0.5 < lo < 0.65 and 1.0 < hi < 1.1


def test_a_block_that_cannot_be_drawn_falls_back_to_all_blocks(win, mat_file, monkeypatch):
    """A failure half way through left a view that took no clicks under a
    banner that described the overview."""
    win.open_recording(mat_file)
    shown = []
    monkeypatch.setattr(
        "labchart2cfd.gui.app.QMessageBox.critical",
        staticmethod(lambda _p, title, text, *a, **k: shown.append(f"{title}: {text}")),
    )
    monkeypatch.setattr(win, "_draw_comments",
                        lambda: (_ for _ in ()).throw(RuntimeError("no ink")))

    win._on_overview_click(ClickEvent(win._overview_axes[(2, 1)]))

    assert win._plot_mode == "overview" and win._overview_axes
    assert "no ink" in shown[0]


# ----------------------------------------------------- y scale at the extremes

def test_zooming_out_stops_before_the_scale_breaks(win, mat_file):
    """6000 presses took the limits to 1e308 and drawing raised."""
    open_block(win, mat_file)

    for _ in range(400):
        win._scale_y("flow", 2.0)
        win._scale_y("trigger", 2.0)

    assert max(np.abs(win.ax_top.get_ylim())) < 1e12
    assert max(np.abs(win.ax_top_trig.get_ylim())) < 1e12
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        win.canvas.draw()
    assert "scale" in win.status_bar.currentMessage().lower()


def test_zooming_in_stops_while_the_axis_can_still_be_labelled(win, mat_file):
    open_block(win, mat_file)

    for _ in range(400):
        win._scale_y("trigger", 0.5)

    lo, hi = win.ax_top_trig.get_ylim()
    assert hi - lo > 1e-10
    win.canvas.draw()


def test_an_absurd_wheel_step_is_harmless(win, mat_file):
    open_block(win, mat_file)
    win.canvas.draw()
    point = _strip_point(win, win.ax_top, "left")

    win._on_scroll(ScrollEvent(*point, step=-5000))
    win._on_scroll(ScrollEvent(*point, step=5000))

    assert np.isfinite(win.ax_top.get_ylim()).all()
    win.canvas.draw()


# ------------------------------------------------------------------- comments

def _visible_labels(win):
    return [a.get_text() for a in win._comment_artists
            if isinstance(a, Text) and a.get_visible()]


@pytest.mark.parametrize("text", [r"a $\frac$ b", "dose 5$ then 10$", "$x_", "50% O2_{et} ^ ok"])
def test_a_comment_is_drawn_as_typed_not_as_a_formula(win, mat_file, tmp_path, text):
    """A dollar sign pair is a formula to matplotlib, and a broken one raised
    while drawing."""
    open_block(win, _with_comments(mat_file, tmp_path, [5.0], [text]))

    win.canvas.draw()

    assert _visible_labels(win) == [text]


def test_a_channel_title_with_dollar_signs_is_drawn_as_typed(win, mat_file, tmp_path):
    path = _changed(mat_file, tmp_path,
                    titles=np.array(["Trigger   ", r"$\frac$ in", "          ", "Pressure  "]))

    open_block(win, path)
    win.canvas.draw()

    assert r"$\frac$ in" in win.ax_top.get_title()


def test_many_comments_are_lines_until_few_enough_are_in_view(win, mat_file, tmp_path):
    """500 comments drew 1500 artists and every redraw took 0.3 s. Unreadable
    labels are left out until zooming in leaves few enough to read."""
    times = list(np.linspace(0.5, 9.5, 300))
    open_block(win, _with_comments(mat_file, tmp_path, times, [f"c{i}" for i in range(300)]))

    assert _visible_labels(win) == []
    assert len(win._comment_artists) <= 4          # one set of lines per plot
    assert "zoom in" in win.status_bar.currentMessage().lower()

    win.ax_top.set_xlim(times[10] - 0.001, times[13] + 0.001)

    assert _visible_labels(win) == ["c10", "c11", "c12", "c13"]


def test_hidden_comments_stay_hidden_while_zooming(win, mat_file, tmp_path):
    open_block(win, _with_comments(mat_file, tmp_path, [4.0, 6.0], ["a", "b"]))
    win.comments_action.trigger()

    win.ax_top.set_xlim(3.0, 7.0)

    assert _visible_labels(win) == []
    assert not any(a.get_visible() for a in win._comment_artists)


# ---------------------------------------- nothing placed is removed unasked

@pytest.mark.parametrize("action", [
    lambda win: win.overview_btn.click(),
    lambda win: win.overview_action.trigger(),
    lambda win: win.detail_btn.click(),
    lambda win: win.detail_action.trigger(),
    lambda win: win.select_step_btn.click(),
    lambda win: win.workflow_combo.setCurrentText("MRI"),
], ids=["all blocks button", "all blocks menu", "selected block button",
        "selected block menu", "select step again", "study type"])
def test_every_way_of_redrawing_asks_before_dropping_landmarks(
        win, mat_file, monkeypatch, action):
    """These removed the landmarks without a word. Only the pickers asked."""
    _landmarks_marked(win, mat_file)
    asked = []

    _answer(monkeypatch, QMessageBox.No, asked)
    action(win)
    assert len(asked) == 1 and "landmark" in asked[0].lower()
    assert win._ct_inhale_start == 4.0 and win._plot_mode == "selected"
    assert win.workflow_combo.currentText() == "CT"
    assert win.ct_inhale_label.text() == "4.000 s"

    _answer(monkeypatch, QMessageBox.Yes, asked)
    action(win)
    assert win._ct_inhale_start is None


def test_opening_a_recording_asks_before_dropping_marks(win, mat_file, tmp_path, monkeypatch):
    _landmarks_marked(win, mat_file)
    other = tmp_path / "other.mat"
    shutil.copy(mat_file, other)
    asked = []

    _answer(monkeypatch, QMessageBox.No, asked)
    win.open_recording(other)
    assert asked and win.mat_filepath == mat_file
    assert win.file_label.text() == mat_file.name
    assert win._ct_inhale_start == 4.0

    _answer(monkeypatch, QMessageBox.Yes, asked)
    win.open_recording(other)
    assert win.mat_filepath == other and win._ct_inhale_start is None


def test_opening_another_recording_asks_before_dropping_parts(
        win, mat_file, tmp_path, monkeypatch):
    win.workflow_combo.setCurrentText("CT")
    open_block(win, mat_file)
    win._parts.append(ProfilePart(start_image=1, end_image=3, label=""))
    other = tmp_path / "other.mat"
    shutil.copy(mat_file, other)
    asked = []

    _answer(monkeypatch, QMessageBox.No, asked)
    win.open_recording(other)

    assert "1 part" in asked[0] and win._parts and win.mat_filepath == mat_file


def test_a_tile_of_another_block_asks_before_dropping_parts(win, mat_file, monkeypatch):
    """Through the overview the parts of block 1 followed to block 2, and the
    export cut block 2 by block 1's image ranges."""
    win.workflow_combo.setCurrentText("CT")
    open_block(win, mat_file)
    win._parts.append(ProfilePart(start_image=1, end_image=3, label=""))
    win._go_to_overview()
    asked = []

    _answer(monkeypatch, QMessageBox.No, asked)
    win._on_overview_click(ClickEvent(win._overview_axes[(2, 2)]))
    assert "1 part" in asked[0]
    assert win._plot_mode == "overview" and win.col_spin.value() == 1 and win._parts

    _answer(monkeypatch, QMessageBox.Yes, asked)
    win._on_overview_click(ClickEvent(win._overview_axes[(2, 2)]))
    assert win._plot_mode == "selected" and win.col_spin.value() == 2
    assert win._parts == []


def test_the_block_picker_on_the_overview_asks_before_dropping_parts(
        win, mat_file, monkeypatch):
    win.workflow_combo.setCurrentText("CT")
    open_block(win, mat_file)
    win._parts.append(ProfilePart(start_image=1, end_image=3, label=""))
    win._go_to_overview()
    asked = []
    _answer(monkeypatch, QMessageBox.No, asked)

    win.col_spin.setValue(2)
    win.col_spin.activated.emit(win.col_spin.currentIndex())

    assert asked and win.col_spin.value() == 1 and win._parts


def test_the_same_block_tile_keeps_the_parts_and_asks_nothing(win, mat_file):
    win.workflow_combo.setCurrentText("CT")
    open_block(win, mat_file)
    win._parts.append(ProfilePart(start_image=1, end_image=3, label=""))
    win._go_to_overview()

    win._on_overview_click(ClickEvent(win._overview_axes[(2, 1)]))

    assert win._plot_mode == "selected" and len(win._parts) == 1


def test_redrawing_with_nothing_marked_asks_nothing(win, mat_file):
    win.workflow_combo.setCurrentText("CT")
    open_block(win, mat_file)
    win._select_ct_step()

    win.detail_btn.click()
    win.select_step_btn.click()
    win.overview_btn.click()
    win.workflow_combo.setCurrentText("MRI")

    assert win._plot_mode == "overview"


# ------------------------------------------------------- an absurd image count

def test_export_is_held_back_while_temporal_res_gives_too_many_images(win, mat_file):
    """At 0.01 ms the grid was refused but Export still wrote 300,000 images
    while the box on screen said 15."""
    _landmarks_marked(win, mat_file)

    win.ct_temporal_res_spin.setValue(0.01)

    hint = win.export_hint.text().lower()
    assert hint.startswith("before export") and "temporal res" in hint
    assert "300000" in hint.replace(",", "")
