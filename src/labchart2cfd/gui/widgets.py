"""Reusable widgets for the labchart2cfd GUI.

Kept apart from ``app.py`` so each widget can be tested on its own, without
building the whole main window.
"""

from typing import Sequence

from PyQt5.QtCore import QEvent, QObject, QSize, Qt
from PyQt5.QtGui import QPainter
from PyQt5.QtWidgets import (
    QAbstractScrollArea, QAbstractSlider, QAbstractSpinBox, QApplication,
    QComboBox, QFrame, QLabel, QLayout, QScrollArea, QScrollBar, QSizePolicy,
    QToolButton, QVBoxLayout, QWidget,
)


def channel_label(index: int, title: str, noun: str = "Channel") -> str:
    """Text for one channel or block entry.

    A blank title falls back to "<noun> <index>", so every entry still names
    its own 1-indexed position and none is mistaken for a neighbour.
    """
    title = (title or "").strip()
    return f"{index}: {title}" if title else f"{noun} {index}"


class ChannelCombo(QComboBox):
    """Dropdown that picks a 1-indexed channel (or block) by name.

    It answers ``value()`` and ``setValue()`` like the QSpinBox it replaces, so
    the handlers that read a row number did not have to change. With
    ``allow_none`` the first entry is "None" and reports 0, the value that has
    always meant "no trigger" or "no pressure" in this application.
    """

    # Entries offered before a recording is loaded. The old spin boxes allowed
    # rows 1 to 20 at that stage, and callers set a row before loading.
    DEFAULT_COUNT = 20
    NONE_TEXT = "None"

    def __init__(self, noun: str = "Channel", allow_none: bool = False,
                 parent=None) -> None:
        super().__init__(parent)
        self._noun = noun
        self._allow_none = allow_none
        self.set_items([""] * self.DEFAULT_COUNT)

    def minimum(self) -> int:
        return 0 if self._allow_none else 1

    def maximum(self) -> int:
        return self.count() - (1 if self._allow_none else 0)

    def value(self) -> int:
        data = self.currentData()
        return int(data) if data is not None else self.minimum()

    def setValue(self, value: int) -> None:  # noqa: N802 (QSpinBox spelling)
        """Select a row, clamped into the rows on offer as a spin box would."""
        value = max(self.minimum(), min(int(value), self.maximum()))
        index = self.findData(value)
        if index >= 0:
            self.setCurrentIndex(index)

    def set_items(self, titles: Sequence[str]) -> None:
        """Offer one entry per title, keeping the current row when it survives.

        A current row past the new last entry moves to the last entry, the
        same clamp the spin boxes applied when a file had fewer channels.
        """
        previous = self.value() if self.count() else None
        self.blockSignals(True)
        try:
            self.clear()
            if self._allow_none:
                self.addItem(self.NONE_TEXT, 0)
            for index, title in enumerate(titles, start=1):
                self.addItem(channel_label(index, title, self._noun), index)
        finally:
            self.blockSignals(False)
        if previous is not None:
            self.setValue(previous)


class WheelGuard(QObject):
    """Stops the mouse wheel from changing a control it happens to be over.

    Inside a scrolling panel the wheel means "scroll". Qt also lets it step a
    dropdown, a number box or a slider under the pointer, so scrolling the
    sidebar silently changed the study type, a channel or a time. The guard
    takes the wheel away from those controls and hands it to the scroll bar of
    the panel around them, the same way a scroll area forwards its own wheel
    events, so the panel scrolls as if the control were not there.
    """

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 (Qt spelling)
        if event.type() != QEvent.Wheel:
            return False
        area = watched.parentWidget()
        while area is not None and not isinstance(area, QAbstractScrollArea):
            area = area.parentWidget()
        if area is not None:
            QApplication.sendEvent(area.verticalScrollBar(), event)
        # Accepted either way, so Qt does not offer the same wheel step to the
        # parents again and scroll twice.
        event.accept()
        return True


def guard_wheel(root: QWidget) -> WheelGuard:
    """Protect every dropdown, number box and slider under ``root``.

    Scroll bars are sliders too and are left alone: they are what the wheel
    is meant to move.
    """
    guard = WheelGuard(root)
    for kind in (QComboBox, QAbstractSpinBox, QAbstractSlider):
        for control in root.findChildren(kind):
            if isinstance(control, QScrollBar):
                continue
            control.installEventFilter(guard)
            # Without wheel focus, hovering can never make a control active.
            control.setFocusPolicy(Qt.StrongFocus)
    return guard


class ElidedLabel(QLabel):
    """A one line label that shortens long text in the middle.

    A plain label is as wide as its text. A long file name then widened the
    whole sidebar and squeezed the plot. ``text()`` still returns everything.
    """

    _MIN_WIDTH = 40

    def minimumSizeHint(self) -> QSize:  # noqa: N802 (Qt spelling)
        return QSize(self._MIN_WIDTH, super().minimumSizeHint().height())

    def sizeHint(self) -> QSize:  # noqa: N802 (Qt spelling)
        return QSize(self._MIN_WIDTH, super().sizeHint().height())

    def paintEvent(self, _event) -> None:  # noqa: N802 (Qt spelling)
        painter = QPainter(self)
        area = self.contentsRect()
        shown = self.fontMetrics().elidedText(self.text(), Qt.ElideMiddle, area.width())
        painter.setPen(self.palette().color(self.foregroundRole()))
        painter.drawText(area, int(self.alignment()) | Qt.TextSingleLine, shown)


class PageSwitcher(QWidget):
    """Shows one of several pages, and is only as tall as that page.

    A QStackedWidget is as tall as its tallest page, so a short page was
    followed by the blank height of a long one. Here the other pages are
    hidden, and a hidden widget takes no room.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._pages = []
        self._index = -1
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)

    def addWidget(self, page: QWidget) -> int:  # noqa: N802 (QStackedWidget spelling)
        self._pages.append(page)
        self._layout.addWidget(page)
        if self._index < 0:
            self._index = 0
        page.setVisible(len(self._pages) - 1 == self._index)
        return len(self._pages) - 1

    def count(self) -> int:
        return len(self._pages)

    def widget(self, index: int) -> QWidget:
        return self._pages[index]

    def currentIndex(self) -> int:  # noqa: N802 (QStackedWidget spelling)
        return self._index

    def currentWidget(self) -> QWidget:  # noqa: N802 (QStackedWidget spelling)
        return self._pages[self._index]

    def setCurrentIndex(self, index: int) -> None:  # noqa: N802 (QStackedWidget spelling)
        if not 0 <= index < len(self._pages):
            return
        self._index = index
        for i, page in enumerate(self._pages):
            page.setVisible(i == index)


class VerticalScrollArea(QScrollArea):
    """A scroll area that scrolls up and down only.

    With the horizontal bar off, content wider than the viewport is simply cut
    off at the right edge. This area therefore keeps itself at least as wide
    as its content needs, plus room for the vertical scroll bar.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

    def setWidget(self, widget: QWidget) -> None:  # noqa: N802 (Qt spelling)
        super().setWidget(widget)
        widget.installEventFilter(self)
        self._fit_width_to_content()

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 (Qt spelling)
        # The content's needs change when sections appear, fold or restyle.
        if watched is self.widget() and event.type() in (
            QEvent.LayoutRequest, QEvent.Resize, QEvent.Polish,
        ):
            self._fit_width_to_content()
        return super().eventFilter(watched, event)

    def _fit_width_to_content(self) -> None:
        content = self.widget()
        if content is None:
            return
        needed = (content.minimumSizeHint().width()
                  + self.verticalScrollBar().sizeHint().width())
        if needed != self.minimumWidth():
            self.setMinimumWidth(needed)


class CollapsibleSection(QWidget):
    """A titled sidebar section whose body folds away when its header is clicked."""

    def __init__(self, title: str, parent=None) -> None:
        super().__init__(parent)
        self.header = QToolButton(self)
        self.header.setObjectName("sectionHeader")
        self.header.setText(title)
        self.header.setCheckable(True)
        self.header.setChecked(True)
        self.header.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.header.setArrowType(Qt.DownArrow)
        self.header.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.body = QWidget(self)
        self.body.setObjectName("sectionBody")
        self.body_layout = QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(10, 6, 10, 10)
        self.body_layout.setSpacing(6)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self.header)
        outer.addWidget(self.body)

        self.header.toggled.connect(self._on_toggled)

    def _on_toggled(self, expanded: bool) -> None:
        self.body.setVisible(expanded)
        self.header.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)

    def is_expanded(self) -> bool:
        return self.header.isChecked()

    def set_expanded(self, expanded: bool) -> None:
        self.header.setChecked(expanded)

    def add_widget(self, widget: QWidget) -> None:
        self.body_layout.addWidget(widget)

    def add_layout(self, layout: QLayout) -> None:
        self.body_layout.addLayout(layout)
