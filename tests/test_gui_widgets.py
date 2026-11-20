"""Tests for the reusable GUI widgets in ``labchart2cfd.gui.widgets``.

Qt runs offscreen so no window appears.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5.QtWidgets import QApplication, QLabel

from labchart2cfd.gui.widgets import ChannelCombo, CollapsibleSection


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


# ---------------------------------------------------------------- ChannelCombo

def test_channel_combo_reports_the_one_indexed_row(qapp):
    """Handlers read the row with value(), exactly as they did from a spin box."""
    combo = ChannelCombo(noun="Channel")

    combo.setValue(3)

    assert combo.value() == 3
    assert combo.currentText() == "Channel 3"


def test_channel_combo_shows_the_channel_names_from_the_file(qapp):
    combo = ChannelCombo(noun="Channel")

    combo.set_items(["trig", "", "flow"])

    texts = [combo.itemText(i) for i in range(combo.count())]
    # A blank title keeps its own row number rather than borrowing a neighbour's.
    assert texts == ["1: trig", "Channel 2", "3: flow"]


def test_set_items_keeps_the_current_row_when_it_still_exists(qapp):
    combo = ChannelCombo(noun="Channel")
    combo.setValue(2)

    combo.set_items(["trig", "flow", "pressure"])

    assert combo.value() == 2


def test_set_items_clamps_a_row_beyond_the_new_range(qapp):
    """Same rule the spin boxes had: a row past the last channel moves to it."""
    combo = ChannelCombo(noun="Channel")
    combo.setValue(7)

    combo.set_items(["trig", "flow", "pressure"])

    assert combo.value() == 3


def test_set_value_clamps_into_the_available_rows(qapp):
    combo = ChannelCombo(noun="Block")
    combo.set_items(["", ""])

    combo.setValue(9)
    assert combo.value() == 2

    combo.setValue(-4)
    assert combo.value() == 1


def test_optional_combo_offers_none_and_reports_zero(qapp):
    """Trigger and pressure are optional; 0 has always meant 'no channel'."""
    combo = ChannelCombo(noun="Channel", allow_none=True)

    combo.setValue(0)

    assert combo.value() == 0
    assert combo.currentText() == "None"


def test_none_survives_repopulating(qapp):
    combo = ChannelCombo(noun="Channel", allow_none=True)
    combo.setValue(0)

    combo.set_items(["trig", "flow"])

    assert combo.value() == 0
    assert combo.itemText(0) == "None"
    assert combo.count() == 3


# ---------------------------------------------------------- CollapsibleSection

def test_collapsible_section_starts_expanded(qapp):
    section = CollapsibleSection("1  Recording")
    section.add_widget(QLabel("body"))

    assert section.is_expanded()
    assert not section.body.isHidden()


def test_collapsing_a_section_hides_its_body_only(qapp):
    section = CollapsibleSection("1  Recording")
    section.add_widget(QLabel("body"))

    section.set_expanded(False)

    assert not section.is_expanded()
    assert section.body.isHidden()
    assert not section.header.isHidden()   # the title stays so it can reopen


def test_clicking_the_header_toggles_the_section(qapp):
    section = CollapsibleSection("1  Recording")
    section.add_widget(QLabel("body"))

    section.header.click()
    assert not section.is_expanded()

    section.header.click()
    assert section.is_expanded()
