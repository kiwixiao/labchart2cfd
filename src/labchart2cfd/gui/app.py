"""PyQt5 GUI for interactive flow profile selection and processing.

The window is a sidebar of numbered sections beside a plot. The sections read
in the order the work is done:

1. Recording: open a LabChart recording (a .mat export, or the native
   .adicht file directly) and name the Flow, Trigger and
   Pressure channels. An overview shows every channel and block; clicking a
   block opens it.
2. Study type: MRI, CPAP, phase contrast or CT. Only that type's controls show.
3. Time window: click the plot for Start and End, or type them.
4. Triggers (pulse triggers for 4D registration), or for CT the scan step and
   one of three ways to build the breath from the images.

Subject, output folder and the single Export button stay pinned below.

Three rules keep the window honest:

- ``_click_mode()`` is the one answer to what a plot click means. The click
  handler dispatches on it and the banner above the plot displays it.
- ``_ct_export_inputs()`` lets only the CT method on screen feed an export.
- ``_export()`` is the only exit, and its result is shown as a report.

Reusable widgets live in ``widgets.py``, dialogs in ``dialogs.py`` and the
colours in ``theme.py``.
"""

import sys
import traceback
from pathlib import Path
from threading import Thread
from typing import Optional, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.transforms import Bbox, TransformedBbox
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from scipy.signal import find_peaks

from PyQt5.QtCore import QEvent, QEventLoop, Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QComboBox,
    QSpinBox, QDoubleSpinBox, QSlider, QFileDialog, QMessageBox,
    QStatusBar, QGridLayout, QSizePolicy, QFrame,
    QCheckBox, QListWidget, QListWidgetItem, QInputDialog, QDialog,
    QSplitter, QStackedWidget, QFormLayout, QAction, QBoxLayout,
)
from PyQt5.QtGui import QKeySequence

from labchart2cfd.gui.decimate import EnvelopeLine, minmax_envelope
from labchart2cfd.gui.dialogs import (
    AssemblyPreviewDialog, ExportResultDialog, QuickGuideDialog,
)
from labchart2cfd.gui.theme import STYLE_SHEET, apply_theme
from labchart2cfd.gui.widgets import (
    ChannelCombo, CollapsibleSection, ElidedLabel, VerticalScrollArea,
    channel_label, guard_wheel,
)
from labchart2cfd.io.labchart import LabChartData
from labchart2cfd.io.recording import RECORDING_SUFFIXES, load_recording
from labchart2cfd.processing.profile_assembly import ProfilePart, assemble_parts
from labchart2cfd.processing.rearrangement import (
    compute_image_count,
    image_index_to_time,
    time_to_image_index,
)


def _safe_remove(artist) -> None:
    """Detach a matplotlib artist, tolerating one that is already detached.

    ``fig.clear()`` invalidates artists while our lists still hold references
    to them. An unguarded ``artist.remove()`` then raises and aborts the rest
    of the loop, leaving some markers on screen and the list un-emptied.
    """
    try:
        artist.remove()
    except (ValueError, AttributeError):
        pass


def _remove_all(artists) -> None:
    """Detach every artist in an iterable, skipping any already detached."""
    for artist in artists:
        _safe_remove(artist)


# Study types: (name shown, workflow name the handlers branch on, description).
# The second item is what `_workflow_name()` returns, and its spellings are the
# ones the processing code has always compared against.
_STUDY_TYPES = (
    ("MRI", "MRI",
     "Standard MRI study. Drift is corrected on the whole signal, then the "
     "window is cut out."),
    ("CPAP", "cpap",
     "CPAP study. Drift is corrected inside the chosen window."),
    ("Phase contrast (xenon)", "phase-contrast",
     "Xenon phase contrast study. Uses xenon density and voltage calibration."),
    ("CT", "CT",
     "4D CT study. The trigger channel marks the scan, and the images are "
     "placed on a time grid across it."),
)

# CT methods: (name shown, key, description). One method is on screen at a time.
_CT_METHODS = (
    ("Reorder image parts", "parts",
     "Pick ranges of images and put them in breathing order. Exports the "
     "profile and an image reorder map."),
    ("Inhale and exhale landmarks", "landmarks",
     "Mark where inhale starts and exhale ends. The profile is rotated to "
     "start at the inhale image when needed, and the trigger info file lists "
     "the images from inhale to exhale."),
    ("Label cycle frames", "cycle",
     "Click frames in breathing order. Runs of consecutive frames are cut out "
     "and joined in the order you clicked."),
)

# Roles a channel can play: (name shown, picker attribute, word fragments a
# channel title of that kind contains). Used only to flag a choice that
# contradicts the titles, never to make the choice.
_ROLE_TITLE_WORDS = (
    ("Flow", "flow_row_spin", ("flow",)),
    ("Trigger", "trigger_row_spin", ("trig",)),
    ("Pressure", "pressure_row_spin", ("press",)),
)

# Line colours for the role a channel plays, used on the overview.
_ROLE_COLORS = {"flow": "#1f77b4", "trigger": "#d62728", "pressure": "#2ca02c"}

# What a click on the plot does in each click mode: (kind, text for the banner).
# "idle" is the resting state, "active" means a tool is armed and waiting for a
# click, and "blocked" means clicks are being swallowed.
_BANNERS = {
    "empty": ("idle",
              "Open a LabChart recording to begin (.mat export or native "
              ".adicht): use Open recording on the left, or drop the file "
              "here."),
    "overview": ("idle",
                 "All blocks of the recording. Click a block to open it."),
    "window": ("idle",
               "Click the plot to set Start, then click again to set End. "
               "Right click clears the markers."),
    "toolbar": ("blocked",
                "The zoom or pan tool is on, so clicks on the plot are "
                "ignored. Turn the tool off in the toolbar to place markers."),
    "inhale_start": ("active", "Click the plot where inhale starts."),
    "exhale_end": ("active", "Click the plot where exhale ends."),
    "cycle": ("active",
              "Labelling cycle frames: click the images in breathing order. "
              "A click takes the image that starts at the nearest grid line. "
              "Right click removes the last label."),
    "ct_landmarks_idle": ("idle",
                          "Scan step selected. Press Mark Inhale Start or "
                          "Mark Exhale End, then click the plot. Plain clicks "
                          "do nothing here, so the scan window cannot move by "
                          "accident."),
    "ct_cycle_idle": ("idle",
                      "Scan step selected. Press Label Cycle Frames, then "
                      "click the images in breathing order. Plain clicks do "
                      "nothing here, so the scan window cannot move by "
                      "accident."),
    "parts_idle": ("idle",
                   "Reorder parts: plain clicks do nothing here, so the image "
                   "grid cannot move by accident. Press Pick on Plot to "
                   "choose images."),
    "part_first": ("active",
                   "Picking a part: click the first image of the range. "
                   "Right click cancels."),
    "part_second": ("active",
                    "Picking a part: now click the last image of the range. "
                    "Right click cancels."),
}


# What to do next once a scan step is selected, by CT method. The status
# line used to point at the landmark buttons in every method, also where they
# are not on screen.
_CT_NEXT_STEP = {
    "parts": "Reorder parts: choose image ranges under 5 Build the breath, then Add Part",
    "landmarks": "Landmarks: press Mark Inhale Start, then click the plot",
    "cycle": "Cycle frames: press Label Cycle Frames, then click the images in breathing order",
}


class ModeAwareToolbar(NavigationToolbar2QT):
    """Navigation toolbar that announces when its zoom or pan tool toggles.

    While either tool is on it owns every click on the plot. The window needs
    to hear about the change to tell the user why markers stopped working.
    """

    mode_changed = pyqtSignal()

    def pan(self, *args) -> None:
        super().pan(*args)
        self.mode_changed.emit()

    def zoom(self, *args) -> None:
        super().zoom(*args)
        self.mode_changed.emit()


class FlowProfileApp(QMainWindow):
    """Main GUI application for flow profile processing."""

    # Cross-thread completion signals for the Execute worker. QTimer.singleShot
    # from a plain threading.Thread never fires (no Qt event loop in that
    # thread), which left the button dead and error dialogs unshown; signal
    # emission from a foreign thread is queued to the GUI thread correctly.
    _processing_done = pyqtSignal(str)
    _processing_failed = pyqtSignal(str)
    # Carries the report dict of a finished export (files, summary, warnings).
    _processing_report = pyqtSignal(object)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("CFD Flow Profile Processor")
        self.resize(1400, 860)
        self.setMinimumSize(900, 600)

        # State
        self.data: Optional[LabChartData] = None
        self.mat_filepath: Optional[Path] = None
        self._plot_mode = "empty"  # "empty", "overview", "selected"
        # Overview tiles by (row, column), both 1-indexed, so a click on a
        # tile can be traced back to the block it shows.
        self._overview_axes: dict = {}
        # The signal lines of the detail view. Each draws an envelope of its
        # full data and is refreshed whenever the visible range changes.
        self._envelopes: List[EnvelopeLine] = []

        # Remarks from the reader of the open recording.
        self._load_notes: List[str] = []
        # The LabChart comments of the open block and what is drawn for them:
        # one set of lines per plot, and a label for each comment that has
        # been in view (by its position in the list), made on demand.
        self._comments_drawn: list = []
        self._comment_artists: list = []
        self._comment_labels: dict = {}
        self._comment_layout_key = None
        # True while more comments are in view than can be labelled readably.
        self._comment_labels_withheld = False
        # Whole block range of the channel behind each y scale, read once.
        self._y_reference: dict = {}
        # The plot title of the open block, from whole to shortest.
        self._title_forms: List[str] = []
        # (flow row, trigger row, block) the current view was drawn from, so a
        # picker change can tell whether anything on the plot changed.
        self._shown_selection: Optional[Tuple[int, int, int]] = None
        # The study type the window is laid out for, to go back to when a
        # change is declined.
        self._workflow_index_shown = 0
        # The CT method whose marks are on the plot. Only one method's marks
        # exist at a time, so the plot always shows what will be exported.
        self._active_ct_method = "parts"

        # Export state
        self._outdir_user_set = False   # True once the user picks the folder
        self._last_export_report: Optional[dict] = None
        self._result_dialog: Optional[ExportResultDialog] = None
        self._guide_dialog: Optional[QuickGuideDialog] = None

        # Set on the window, not only on the application, so every way of
        # creating the window (launch, tests, screenshots) looks the same.
        self.setStyleSheet(STYLE_SHEET)
        self.setAcceptDrops(True)

        # Axes references (set during _plot_selected)
        self.ax_top = None
        self.ax_bottom = None
        self.ax_top_trig = None  # twin y-axis for trigger (separate scale from flow)

        # Trigger detection state
        self._trigger_times: np.ndarray = np.array([])
        self._trigger_count: int = 0
        self._trigger_markers: list = []  # artists for trigger range markers

        # Data picker state (time markers on bottom plot)
        self._time_markers: List[Tuple] = []  # (line, annotation) pairs
        self._click_cid = None  # canvas click event connection ID

        # CT workflow state
        self._ct_steps: list = []  # detected step triggers
        self._ct_selected_step: Optional[dict] = None  # currently selected step
        self._ct_step_markers: list = []  # artists for step span visualization
        # (step, label) for each detected step, in time order, so labels that
        # would print over each other can be merged at the current zoom.
        self._ct_step_labels: list = []
        self._ct_landmark_mode: Optional[str] = None  # "inhale_start" or "exhale_end"
        self._ct_inhale_start: Optional[float] = None
        self._ct_exhale_end: Optional[float] = None
        self._ct_landmark_artists: list = []  # artists for landmark markers
        self._ct_interval_markers: list = []  # artists for temporal interval lines
        self._ct_detected_start: Optional[float] = None  # original auto-detected start
        self._ct_detected_end: Optional[float] = None    # original auto-detected end

        # CT cycle-labeling state (independent alternative to inhale/exhale trim)
        self._ct_cycle_mode: bool = False
        self._ct_cycle_frames: list = []  # original image indices in click order
        self._ct_cycle_artists: list = []  # artists for green numbered labels

        # Full-manual synthetic-profile state (window + ordered parts)
        self._synth_mode: bool = False
        self._syncing_grid: bool = False         # guards the dt <-> images linkage
        self._parts: List[ProfilePart] = []      # ordered parts = assembly order
        self._part_artists: list = []            # span + number artists per part
        self._grid_artists: list = []            # fencepost grid lines + numbers
        self._capturing_part: bool = False       # collecting 2 clicks for a part
        self._part_click_buffer: List[int] = []  # snapped image #s during capture
        self._capture_marker = None              # red line at the start frame

        # Internal flag to suppress slider callback during programmatic changes
        self._suppress_shift_slider = False

        self._build_ui()

    # Width the sidebar opens at. The splitter lets the user drag it wider.
    SIDEBAR_WIDTH = 380

    def _build_ui(self) -> None:
        """Build the window: guided controls on the left, the plot on the right."""
        # Created first, because handlers post to it while the rest is built.
        self.status_bar = QStatusBar()
        self.status_bar.showMessage("Ready")
        self.setStatusBar(self.status_bar)

        plot_panel = self._build_plot_panel()
        sidebar = self._build_sidebar()

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(sidebar)
        splitter.addWidget(plot_panel)
        splitter.setStretchFactor(0, 0)  # extra width goes to the plot
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([self.SIDEBAR_WIDTH, 10 * self.SIDEBAR_WIDTH])
        self.setCentralWidget(splitter)

        self._build_menus()
        self._connect_live_updates()
        # Show only the sections that belong to the current study type.
        self._on_workflow_changed()
        self._refresh_banner()
        self._update_window_readout()

        # Worker-completion signals land on the GUI thread here.
        self._processing_done.connect(self._on_processing_done)
        self._processing_failed.connect(self._on_processing_failed)
        self._processing_report.connect(self._show_export_result)

    def _build_menus(self) -> None:
        """File, View and Help menus, with the usual shortcuts."""
        file_menu = self.menuBar().addMenu("&File")
        self.open_action = QAction("&Open recording...", self)
        self.open_action.setShortcut(QKeySequence("Ctrl+O"))
        self.open_action.triggered.connect(lambda: self._browse_file())
        file_menu.addAction(self.open_action)
        self.export_action = QAction("&Export", self)
        self.export_action.setShortcut(QKeySequence("Ctrl+E"))
        self.export_action.triggered.connect(lambda: self._export())
        file_menu.addAction(self.export_action)
        file_menu.addSeparator()
        quit_action = QAction("&Quit", self)
        quit_action.setShortcut(QKeySequence("Ctrl+Q"))
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        view_menu = self.menuBar().addMenu("&View")
        self.overview_action = QAction("&All blocks", self)
        self.overview_action.setShortcut(QKeySequence("Ctrl+1"))
        self.overview_action.triggered.connect(lambda: self._user_show_overview())
        view_menu.addAction(self.overview_action)
        self.detail_action = QAction("&Selected block", self)
        self.detail_action.setShortcut(QKeySequence("Ctrl+2"))
        self.detail_action.triggered.connect(lambda: self._user_show_block())
        view_menu.addAction(self.detail_action)
        self.zoom_action = QAction("&Zoom to window", self)
        self.zoom_action.setShortcut(QKeySequence("Ctrl+0"))
        self.zoom_action.triggered.connect(lambda: self._zoom_to_window())
        view_menu.addAction(self.zoom_action)
        view_menu.addSeparator()
        self.comments_action = QAction("Show &comments", self)
        self.comments_action.setShortcut(QKeySequence("Ctrl+3"))
        self.comments_action.setCheckable(True)
        self.comments_action.setChecked(True)
        self.comments_action.setEnabled(False)
        self.comments_action.toggled.connect(self._set_comments_visible)
        view_menu.addAction(self.comments_action)

        help_menu = self.menuBar().addMenu("&Help")
        self.guide_action = QAction("&Quick guide", self)
        self.guide_action.triggered.connect(lambda: self._show_quick_guide())
        help_menu.addAction(self.guide_action)

    def _show_quick_guide(self) -> None:
        """Open the one page guide beside the window, without blocking it."""
        if self._guide_dialog is None:
            self._guide_dialog = QuickGuideDialog(self)
        self._guide_dialog.show()
        self._guide_dialog.raise_()

    # ------------------------------------------------------------------
    # Drag and drop
    # ------------------------------------------------------------------

    @staticmethod
    def _dropped_recording(mime) -> Optional[Path]:
        """The first local recording (.mat or .adicht) in a drag, or None."""
        if not mime.hasUrls():
            return None
        for url in mime.urls():
            if url.isLocalFile():
                path = Path(url.toLocalFile())
                if path.suffix.lower() in RECORDING_SUFFIXES:
                    return path
        return None

    def dragEnterEvent(self, event) -> None:  # noqa: N802 (Qt spelling)
        if self._dropped_recording(event.mimeData()) is not None:
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802 (Qt spelling)
        path = self._dropped_recording(event.mimeData())
        if path is not None:
            event.acceptProposedAction()
            self.open_recording(path)

    # ------------------------------------------------------------------
    # Layout helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _form() -> QFormLayout:
        """A label-and-field form whose fields fill the sidebar width."""
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(6)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        return form

    @staticmethod
    def _hint(text: str) -> QLabel:
        """A wrapped explanatory line, styled quieter than a control label."""
        label = QLabel(text)
        label.setObjectName("hintLabel")
        label.setWordWrap(True)
        return label

    @staticmethod
    def _readout(text: str) -> QLabel:
        """A framed read-only value, such as a count or a marked time."""
        label = QLabel(text)
        label.setObjectName("readout")
        label.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        label.setAlignment(Qt.AlignCenter)
        label.setMinimumWidth(70)
        return label

    def _build_plot_panel(self) -> QWidget:
        """The matplotlib canvas with its navigation toolbar."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        self.fig = Figure(figsize=(12, 6), dpi=100, constrained_layout=True)
        # Placeholder text instead of pre-created axes.
        self.fig.text(
            0.5, 0.5, "Open a LabChart recording to begin\n(.mat export or native .adicht)",
            ha="center", va="center", fontsize=14, color="gray",
        )

        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Matplotlib navigation toolbar (zoom, pan, home, save).
        self.toolbar = ModeAwareToolbar(self.canvas, panel)
        self.toolbar.mode_changed.connect(self._refresh_banner)

        # One line that always says what a click on the plot will do. The
        # click modes are otherwise invisible state.
        self.mode_banner = QLabel()
        self.mode_banner.setObjectName("modeBanner")
        self.mode_banner.setWordWrap(True)
        self.mode_banner.setProperty("kind", "idle")

        # The plot tools and the y scale buttons share a row while both fit.
        # In a narrow window the buttons move to a row of their own: squeezed
        # into one, the toolbar hid pan, zoom and save behind an overflow arrow.
        self.y_scale_strip = QWidget()
        strip_row = QHBoxLayout(self.y_scale_strip)
        strip_row.setContentsMargins(0, 0, 0, 0)
        strip_row.setSpacing(4)
        self._build_y_scale_controls(strip_row)
        tools = QBoxLayout(QBoxLayout.LeftToRight)
        tools.setContentsMargins(0, 0, 0, 0)
        tools.setSpacing(4)
        tools.addWidget(self.toolbar, 1)
        tools.addWidget(self.y_scale_strip, 0, Qt.AlignRight)
        self._tools_row = tools
        self.plot_panel = panel
        panel.installEventFilter(self)

        # Over the numbers of a y axis, the wheel stretches that axis.
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.canvas.mpl_connect("resize_event", self._on_canvas_resized)

        layout.addWidget(self.mode_banner)
        layout.addLayout(tools)
        layout.addWidget(self.canvas, 1)
        return panel

    # Room kept beside the toolbar buttons for its time and value readout.
    _TOOLBAR_READOUT_PX = 180

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 (Qt spelling)
        if watched is getattr(self, "plot_panel", None) and event.type() == QEvent.Resize:
            self._arrange_plot_tools()
        return super().eventFilter(watched, event)

    def _arrange_plot_tools(self) -> None:
        """One row for the plot tools while they fit, two when they do not."""
        margins = self.plot_panel.layout().contentsMargins()
        room = self.plot_panel.width() - margins.left() - margins.right()
        needed = (self.toolbar.sizeHint().width() + self.y_scale_strip.sizeHint().width()
                  + self._tools_row.spacing() + self._TOOLBAR_READOUT_PX)
        direction = QBoxLayout.LeftToRight if room >= needed else QBoxLayout.TopToBottom
        if self._tools_row.direction() != direction:
            self._tools_row.setDirection(direction)

    # ------------------------------------------------------------------
    # Y scales
    # ------------------------------------------------------------------

    # Scale change of one button press, and of one wheel notch.
    _Y_BUTTON_FACTOR = 0.5
    _Y_WHEEL_FACTOR = 0.8
    _Y_WHEEL_MAX_STEP = 20

    def _build_y_scale_controls(self, row: QHBoxLayout) -> None:
        """Zoom in, zoom out and fit for the flow and the trigger scale.

        The two scales are independent (flow on the left, trigger on the
        right), and each can be set by hand, so a small signal can be
        enlarged or a large one brought back into view.
        """
        def _button(text: str, tip: str, action) -> QPushButton:
            button = QPushButton(text)
            button.setObjectName("scaleButton")
            button.setToolTip(tip)
            button.setFocusPolicy(Qt.NoFocus)
            button.setEnabled(False)
            # Compact, so the strip leaves the toolbar its coordinate readout.
            button.setFixedWidth(34 if len(text) == 1 else 50)
            button.clicked.connect(action)
            row.addWidget(button)
            return button

        wheel = " The mouse wheel over the numbers of the axis does the same."
        row.addWidget(QLabel("Flow Y"))
        self.flow_y_out_btn = _button(
            "−", "Show a larger range on the flow scale." + wheel,
            lambda: self._scale_y("flow", 1 / self._Y_BUTTON_FACTOR))
        self.flow_y_in_btn = _button(
            "+", "Enlarge the flow signal." + wheel,
            lambda: self._scale_y("flow", self._Y_BUTTON_FACTOR))
        self.flow_y_fit_btn = _button(
            "Fit", "Fit the flow scale to the flow in view.",
            lambda: self._fit_y("flow"))
        row.addSpacing(10)
        row.addWidget(QLabel("Trigger Y"))
        self.trig_y_out_btn = _button(
            "−", "Show a larger range on the trigger scale." + wheel,
            lambda: self._scale_y("trigger", 1 / self._Y_BUTTON_FACTOR))
        self.trig_y_in_btn = _button(
            "+", "Enlarge the trigger signal." + wheel,
            lambda: self._scale_y("trigger", self._Y_BUTTON_FACTOR))
        self.trig_y_fit_btn = _button(
            "Full", "Show the whole range of the trigger channel in this block.",
            lambda: self._fit_y("trigger"))

    def _y_axes(self, which: str) -> list:
        """The axes that carry one scale: 'flow' (both plots) or 'trigger'."""
        if self._plot_mode != "selected":
            return []
        if which == "trigger":
            return [self.ax_top_trig] if self.ax_top_trig is not None else []
        return [ax for ax in (self.ax_top, self.ax_bottom) if ax is not None]

    def _update_y_controls(self) -> None:
        """Enable each scale's buttons only while that scale is on screen."""
        for which, buttons in (
            ("flow", (self.flow_y_out_btn, self.flow_y_in_btn, self.flow_y_fit_btn)),
            ("trigger", (self.trig_y_out_btn, self.trig_y_in_btn, self.trig_y_fit_btn)),
        ):
            shown = bool(self._y_axes(which))
            for button in buttons:
                button.setEnabled(shown)

    def _scale_y(self, which: str, factor: float,
                 anchor: Optional[float] = None) -> None:
        """Stretch one y scale by ``factor`` (below 1 enlarges the signal).

        ``anchor`` is the value that keeps its place on screen. Without one
        the middle of the scale stays put.
        """
        axes = self._y_axes(which)
        if not axes:
            return
        lo, hi = axes[0].get_ylim()
        if anchor is None:
            anchor = (lo + hi) / 2
        new = (anchor + (lo - anchor) * factor, anchor + (hi - anchor) * factor)
        # Kept within a wide band around the signal's own range. Far outside
        # it the axis can no longer be ticked and drawing fails.
        reference = self._y_reference_span(which)
        span = new[1] - new[0]
        if (not np.all(np.isfinite(new))
                or not reference / self._Y_SPAN_BAND <= span <= reference * self._Y_SPAN_BAND):
            self.status_bar.showMessage(
                f"The {which} scale cannot be "
                f"{'enlarged' if factor < 1 else 'widened'} further."
            )
            return
        for ax in axes:
            ax.set_ylim(*new)
        self.canvas.draw_idle()

    # How far a scale may leave the signal's own range, as a factor each way.
    _Y_SPAN_BAND = 1e6

    def _y_reference_span(self, which: str) -> float:
        """The range of the channel behind a scale over the whole block.

        Read once per drawn block: on a real recording it is a pass over 92
        million samples.
        """
        if which not in self._y_reference:
            row = (self.trigger_row_spin.value() if which == "trigger"
                   else self.flow_row_spin.value())
            limits = self._channel_y_limits(row, self.col_spin.value(), -np.inf, np.inf)
            span = (limits[1] - limits[0]) if limits is not None else 0.0
            self._y_reference[which] = span if span > 0 else 1.0
        return self._y_reference[which]

    def _fit_y(self, which: str) -> None:
        """Flow: fit to the flow in view. Trigger: the whole block.

        The trigger scale is never fitted to the view. Inside a scan step the
        trigger is flat, and fitting to it showed a few millivolts of noise
        in place of the step.
        """
        axes = self._y_axes(which)
        if not axes:
            return
        col = self.col_spin.value()
        if which == "trigger":
            limits = self._channel_y_limits(
                self.trigger_row_spin.value(), col, -np.inf, np.inf)
        else:
            limits = self._channel_y_limits(
                self.flow_row_spin.value(), col, *self.ax_top.get_xlim())
        if limits is None:
            return
        for ax in axes:
            ax.set_ylim(*limits)
        self.canvas.draw_idle()

    def _on_scroll(self, event) -> None:
        """The wheel beside a plot, over its y axis numbers, zooms that scale.

        Left of either plot is the flow scale, right of the top plot is the
        trigger scale. Inside a plot the wheel does nothing, so a scale
        cannot change by accident while reading the signal.
        """
        if self._plot_mode != "selected" or not event.step:
            return
        for ax, right in ((self.ax_top, "trigger"), (self.ax_bottom, None)):
            if ax is None:
                continue
            box = ax.bbox
            if not box.y0 <= event.y <= box.y1:
                continue
            if event.x < box.x0:
                which, scale_ax = "flow", ax
            elif event.x > box.x1 and right and self.ax_top_trig is not None:
                which, scale_ax = right, self.ax_top_trig
            else:
                return
            anchor = scale_ax.transData.inverted().transform((event.x, event.y))[1]
            # One notch up enlarges the signal, as zooming in does elsewhere.
            # A device can report a burst as one huge step; that is capped.
            step = max(-self._Y_WHEEL_MAX_STEP, min(event.step, self._Y_WHEEL_MAX_STEP))
            self._scale_y(which, self._Y_WHEEL_FACTOR ** step, anchor)
            return

    # ------------------------------------------------------------------
    # Click modes
    # ------------------------------------------------------------------

    def _click_mode(self) -> str:
        """The one thing a click on the plot means right now (a _BANNERS key).

        Both the click handler and the banner read this, so what the banner
        promises and what a click does cannot drift apart.
        """
        if self._plot_mode == "empty":
            return "empty"
        if self.toolbar.mode:
            return "toolbar"
        if self._plot_mode == "overview":
            return "overview"
        if self._capturing_part:
            return "part_second" if self._part_click_buffer else "part_first"
        if self._ct_cycle_mode:
            return "cycle"
        if self._ct_landmark_mode is not None:
            return self._ct_landmark_mode
        if self._synth_mode:
            return "parts_idle"
        if self._workflow_name() == "CT" and self._ct_selected_step is not None:
            # The window is the scan step now. A plain click that moved it
            # would cut the wrong part of the recording, with nothing to
            # show it. Before a step is selected, clicks still set the window.
            return "ct_cycle_idle" if self._ct_method() == "cycle" else "ct_landmarks_idle"
        return "window"

    def _refresh_banner(self) -> None:
        """Make the banner describe the current click mode."""
        kind, text = _BANNERS[self._click_mode()]
        self.mode_banner.setText(text)
        # Every view change passes through here, and whether a recording is
        # loaded is one of the things the Export line reports.
        if hasattr(self, "export_hint"):
            self._update_export_hint()
            self._update_channel_check()
            self._update_y_controls()
            self._update_comments_action()
        if self.mode_banner.property("kind") != kind:
            self.mode_banner.setProperty("kind", kind)
            # A changed dynamic property only restyles after a repolish.
            self.mode_banner.style().unpolish(self.mode_banner)
            self.mode_banner.style().polish(self.mode_banner)

    def _build_sidebar(self) -> QWidget:
        """Numbered sections in the order the work is done, Export pinned below."""
        self.sidebar = QWidget()
        self.sidebar.setObjectName("sidebar")
        outer = QVBoxLayout(self.sidebar)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        sections = QWidget()
        sections_layout = QVBoxLayout(sections)
        sections_layout.setContentsMargins(8, 8, 8, 8)
        sections_layout.setSpacing(8)
        for section in (
            self._build_recording_section(),
            self._build_study_section(),
            self._build_window_section(),
            self._build_trigger_section(),
            self._build_ct_scan_section(),
            self._build_ct_method_section(),
        ):
            sections_layout.addWidget(section)
        sections_layout.addStretch(1)

        # Keeps itself wide enough for the widest section, so no control is
        # clipped behind the scroll bar.
        scroll = VerticalScrollArea()
        scroll.setWidget(sections)
        self.sidebar_scroll = scroll

        outer.addWidget(scroll, 1)
        # Outside the scroll area, so Export is always reachable.
        outer.addWidget(self._build_export_footer(), 0)
        # In this panel the wheel scrolls. It must never step a dropdown, a
        # number box or the slider that happens to be under the pointer.
        self._wheel_guard = guard_wheel(self.sidebar)
        return self.sidebar

    # ------------------------------------------------------------------
    # Sidebar sections
    # ------------------------------------------------------------------

    def _build_recording_section(self) -> QWidget:
        """Open a file and say which channel plays which role."""
        section = CollapsibleSection("1   Recording")
        self.recording_section = section

        open_btn = QPushButton("Open recording...")
        open_btn.setToolTip(
            "Open a LabChart recording and show all its blocks: a .mat "
            "export, or the native .adicht file directly."
        )
        open_btn.clicked.connect(self._browse_file)
        section.add_widget(open_btn)

        # Shortened in the middle when long, so a long file name cannot widen
        # the sidebar. The tooltip carries the full path.
        self.file_label = ElidedLabel("No recording open")
        self.file_label.setObjectName("hintLabel")
        section.add_widget(self.file_label)

        # Remarks from the reader of a native recording, such as a channel
        # that LabChart computes and that was recomputed or left out.
        self.load_notes = self._hint("")
        self.load_notes.hide()
        section.add_widget(self.load_notes)

        # Channels are named from the file once one is open. Rows stay
        # 1-indexed underneath, as everywhere else in this application.
        self.flow_row_spin = ChannelCombo("Channel")
        self.flow_row_spin.setValue(2)
        self.trigger_row_spin = ChannelCombo("Channel", allow_none=True)
        self.trigger_row_spin.setValue(1)
        self.pressure_row_spin = ChannelCombo("Channel", allow_none=True)
        self.pressure_row_spin.setValue(4)
        self.col_spin = ChannelCombo("Block")
        self.col_spin.setValue(3)

        form = self._form()
        form.addRow("Flow", self.flow_row_spin)
        form.addRow("Trigger", self.trigger_row_spin)
        form.addRow("Pressure", self.pressure_row_spin)
        form.addRow("Block", self.col_spin)
        section.add_layout(form)

        # Shown only when a chosen channel's title contradicts its role.
        self.channel_check = QLabel("")
        self.channel_check.setObjectName("warningBox")
        self.channel_check.setWordWrap(True)
        self.channel_check.hide()
        section.add_widget(self.channel_check)

        # `activated` fires for a choice the user makes, never for setValue,
        # so restoring a value in code does not redraw the plot.
        for picker in (self.flow_row_spin, self.trigger_row_spin,
                       self.pressure_row_spin, self.col_spin):
            picker.activated.connect(self._on_selection_changed)

        view_row = QHBoxLayout()
        self.overview_btn = QPushButton("All blocks")
        self.overview_btn.setToolTip("Show every channel and block of the recording.")
        self.overview_btn.clicked.connect(lambda: self._user_show_overview())
        view_row.addWidget(self.overview_btn)
        self.detail_btn = QPushButton("Selected block")
        self.detail_btn.setToolTip("Show the chosen block in detail.")
        self.detail_btn.clicked.connect(lambda: self._user_show_block())
        view_row.addWidget(self.detail_btn)
        section.add_layout(view_row)
        return section

    def _build_study_section(self) -> QWidget:
        """Choose the study type, which decides the processing and the sections."""
        section = CollapsibleSection("2   Study type")
        self.study_section = section

        self.workflow_combo = QComboBox()
        for index, (shown, engine, tip) in enumerate(_STUDY_TYPES):
            self.workflow_combo.addItem(shown, engine)
            self.workflow_combo.setItemData(index, tip, Qt.ToolTipRole)
        section.add_widget(self.workflow_combo)

        self.study_hint = self._hint(_STUDY_TYPES[0][2])
        section.add_widget(self.study_hint)

        self.workflow_combo.currentIndexChanged.connect(
            lambda _index: self._on_workflow_changed()
        )
        return section

    def _build_window_section(self) -> QWidget:
        """The Start and End times that bound the processed signal."""
        section = CollapsibleSection("3   Time window")
        self.window_section = section

        self.start_spin = QDoubleSpinBox()
        self.start_spin.setRange(0, 99999)
        self.start_spin.setDecimals(3)
        self.start_spin.setSingleStep(0.001)
        self.start_spin.setValue(0.0)

        self.end_spin = QDoubleSpinBox()
        self.end_spin.setRange(0, 99999)
        self.end_spin.setDecimals(3)
        self.end_spin.setSingleStep(0.001)
        self.end_spin.setValue(0.0)

        form = self._form()
        form.addRow("Start (s)", self.start_spin)
        form.addRow("End (s)", self.end_spin)
        section.add_layout(form)

        # The length of the window, or what is wrong with it.
        self.window_readout = self._hint("")
        section.add_widget(self.window_readout)

        row = QHBoxLayout()
        zoom_btn = QPushButton("Zoom to window")
        zoom_btn.setToolTip(
            "Fit the plot to Start/End with a small margin. Happens "
            "automatically on step select and when you commit Start/End."
        )
        zoom_btn.clicked.connect(lambda: self._zoom_to_window())
        row.addWidget(zoom_btn)
        clear_markers_btn = QPushButton("Clear markers")
        clear_markers_btn.setToolTip(
            "Remove every marker drawn on the plot: time markers, trigger "
            "ranges, CT landmarks and cycle labels."
        )
        clear_markers_btn.clicked.connect(self._clear_all_markers)
        row.addWidget(clear_markers_btn)
        section.add_layout(row)
        return section

    def _build_trigger_section(self) -> QWidget:
        """Pulse triggers for 4D image registration (every study type but CT)."""
        section = CollapsibleSection("4   Triggers")
        self.trigger_section = section

        self.total_triggers_label = self._readout("--")

        self.start_trigger_spin = QSpinBox()
        self.start_trigger_spin.setRange(1, 1)
        self.start_trigger_spin.setValue(1)

        self.num_triggers_spin = QSpinBox()
        self.num_triggers_spin.setRange(1, 1)
        self.num_triggers_spin.setValue(1)

        form = self._form()
        form.addRow("Triggers found", self.total_triggers_label)
        form.addRow("Start at trigger", self.start_trigger_spin)
        form.addRow("Number of triggers", self.num_triggers_spin)
        section.add_layout(form)

        mark_triggers_btn = QPushButton("Mark triggers")
        mark_triggers_btn.setToolTip(
            "Shade the chosen trigger range and set Start/End to it."
        )
        mark_triggers_btn.clicked.connect(self._mark_trigger_range)
        section.add_widget(mark_triggers_btn)
        return section

    def _build_ct_scan_section(self) -> QWidget:
        """Find the scan step in the trigger channel and lay the image grid on it."""
        section = CollapsibleSection("4   CT scan")
        self.ct_scan_section = section

        # What a tooltip says is not on screen. A new user does not hover.
        section.add_widget(self._hint(
            "The scan shows as a step on the trigger channel. Select it to "
            "set Start and End to the scan."
        ))

        detect_steps_btn = QPushButton("Detect Steps")
        detect_steps_btn.setToolTip(
            "Find the scan steps in the Trigger channel. This also runs when "
            "a block is opened."
        )
        detect_steps_btn.clicked.connect(self._detect_steps)
        section.add_widget(detect_steps_btn)

        self.ct_step_spin = QSpinBox()
        self.ct_step_spin.setRange(1, 1)
        self.ct_step_spin.setValue(1)
        select_step_btn = QPushButton("Select Step")
        select_step_btn.setToolTip(
            "Use this step: sets Start/End to it, draws the image grid and "
            "zooms the plot to it."
        )
        select_step_btn.clicked.connect(lambda: self._user_select_step())
        self.select_step_btn = select_step_btn
        step_row = QHBoxLayout()
        step_row.addWidget(self.ct_step_spin, 1)
        step_row.addWidget(select_step_btn)

        # Same type, range and precision as the grid's own dt box, which this
        # one mirrors. As an int spinbox it silently truncated any fractional
        # dt, leaving the CT export and the synthetic grid on different dt.
        self.ct_temporal_res_spin = QDoubleSpinBox()
        self.ct_temporal_res_spin.setRange(0.01, 10000.0)
        self.ct_temporal_res_spin.setDecimals(2)
        self.ct_temporal_res_spin.setSingleStep(10)
        self.ct_temporal_res_spin.setValue(200.0)
        self.ct_temporal_res_spin.setToolTip(
            "Time one image covers, taken from the reconstructed image series."
        )

        self.ct_total_images_spin = QSpinBox()
        self.ct_total_images_spin.setRange(1, 500)
        self.ct_total_images_spin.setValue(1)
        self.ct_total_images_spin.setToolTip(
            "Images in the selected step. Computed from the step when it is "
            "selected; edit it before Apply shift and image count to override."
        )

        # Bidirectional: the detected step edge can sit either side of the true
        # first image, so the correction has to move the window earlier as well
        # as later. Positive-only meant a late-detected edge was uncorrectable.
        self.ct_shift_spin = QSpinBox()
        self.ct_shift_spin.setRange(-2000, 2000)
        self.ct_shift_spin.setSingleStep(10)
        self.ct_shift_spin.setValue(0)
        self.ct_shift_spin.editingFinished.connect(self._on_shift_entry_changed)

        self.ct_shift_slider = QSlider(Qt.Horizontal)
        self.ct_shift_slider.setRange(-2000, 2000)
        self.ct_shift_slider.setSingleStep(10)
        self.ct_shift_slider.setPageStep(10)
        self.ct_shift_slider.setValue(0)
        self.ct_shift_slider.valueChanged.connect(self._on_shift_slider_changed)

        form = self._form()
        form.addRow("Step #", step_row)
        form.addRow("Temporal Res (ms)", self.ct_temporal_res_spin)
        form.addRow("Images in step", self.ct_total_images_spin)
        form.addRow("Start Shift (ms)", self.ct_shift_spin)
        # In the form, under its box, so it reads as the same setting.
        form.addRow("", self.ct_shift_slider)
        section.add_layout(form)
        section.add_widget(self._hint(
            "Temporal Res is the time one image covers. Images in step is "
            "counted from the step. If the step edge is not where the first "
            "image starts, Start Shift moves the scan window earlier "
            "(negative) or later; the slider does the same."
        ))

        apply_correction_btn = QPushButton("Apply shift and image count")
        apply_correction_btn.setToolTip(
            "Set Start to the detected step start plus the Start Shift, and "
            "End to Start plus Total Images times Temporal Res."
        )
        apply_correction_btn.clicked.connect(self._apply_ct_correction)
        section.add_widget(apply_correction_btn)
        section.add_widget(self._hint(
            "Sets Start to the step start plus the shift, and End to Start "
            "plus Images in step times Temporal Res."
        ))
        return section

    def _build_ct_method_section(self) -> QWidget:
        """Choose how the breath is built from the images; show that method only."""
        section = CollapsibleSection("5   Build the breath")
        self.ct_method_section = section

        self.ct_method_combo = QComboBox()
        for index, (shown, key, tip) in enumerate(_CT_METHODS):
            self.ct_method_combo.addItem(shown, key)
            self.ct_method_combo.setItemData(index, tip, Qt.ToolTipRole)
        section.add_widget(self.ct_method_combo)

        self.ct_method_hint = self._hint(_CT_METHODS[0][2])
        section.add_widget(self.ct_method_hint)

        self.ct_method_stack = QStackedWidget()
        self.parts_page = self._build_parts_page()
        self.landmarks_page = self._build_landmarks_page()
        self.cycle_page = self._build_cycle_page()
        for page in (self.parts_page, self.landmarks_page, self.cycle_page):
            self.ct_method_stack.addWidget(page)
        section.add_widget(self.ct_method_stack)

        self.ct_method_combo.currentIndexChanged.connect(
            lambda _index: self._on_ct_method_changed()
        )
        return section

    def _build_parts_page(self) -> QWidget:
        """Full manual method: ordered ranges of images assembled into a breath."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Manual mode is on exactly while this method is the chosen one, so the
        # switch itself is not shown. It stays a checkbox because its toggled
        # signal is what enters and leaves the mode.
        self.synth_enable_chk = QCheckBox("Enable full-manual mode", page)
        self.synth_enable_chk.toggled.connect(self._on_synth_mode_toggled)
        self.synth_enable_chk.hide()

        # Mirror of the Temporal Res box in the CT scan section. Both feed the
        # same grid, so only that one is shown. Double, because a dt derived
        # from an image count is rarely a whole millisecond.
        self.synth_dt_spin = QDoubleSpinBox(page)
        self.synth_dt_spin.setRange(0.01, 10000.0)
        self.synth_dt_spin.setDecimals(2)
        self.synth_dt_spin.setValue(float(self.ct_temporal_res_spin.value()))
        self.synth_dt_spin.hide()

        self.synth_images_spin = QSpinBox()
        self.synth_images_spin.setRange(1, 5000)
        self.synth_images_spin.setValue(1)
        self.synth_images_spin.setToolTip(
            "Number of images in the grid (ground truth from your "
            "reconstructed series). Temporal Res is left as typed."
        )

        self.synth_grid_offset_spin = QSpinBox()
        # Negative allowed: the detected step edge can sit after the true first
        # image, so the grid has to be shiftable in both directions.
        self.synth_grid_offset_spin.setRange(-600000, 600000)
        self.synth_grid_offset_spin.setValue(0)
        self.synth_grid_offset_spin.setToolTip(
            "Slides the whole grid earlier (negative) or later (positive) "
            "without changing Start/End. This moves the exported range."
        )

        self.synth_window_offset_spin = QSpinBox()
        self.synth_window_offset_spin.setRange(-600000, 600000)
        self.synth_window_offset_spin.setValue(0)
        self.synth_window_offset_spin.setToolTip(
            "Slides Start and End together, moving the whole processed range."
        )
        apply_win_offset_btn = QPushButton("Apply")
        apply_win_offset_btn.setToolTip("Shift Start/End by the window offset.")
        apply_win_offset_btn.clicked.connect(self._apply_window_offset)
        window_offset_row = QHBoxLayout()
        window_offset_row.addWidget(self.synth_window_offset_spin, 1)
        window_offset_row.addWidget(apply_win_offset_btn)

        form = self._form()
        form.addRow("Images in grid", self.synth_images_spin)
        form.addRow("Grid offset (ms)", self.synth_grid_offset_spin)
        form.addRow("Window offset (ms)", window_offset_row)
        layout.addLayout(form)

        grid_row = QHBoxLayout()
        self.synth_grid_label = QLabel("images: none")
        grid_row.addWidget(self.synth_grid_label, 1)
        show_grid_btn = QPushButton("Show Image Grid")
        # Explicit lambda: clicked() would otherwise pass its checked flag as
        # the `quiet` argument.
        show_grid_btn.clicked.connect(lambda: self._draw_synth_grid(quiet=False))
        grid_row.addWidget(show_grid_btn)
        layout.addLayout(grid_row)

        # States the span the grid covers, which is the range that is exported.
        self.synth_dt_warn_label = self._hint("")
        layout.addWidget(self.synth_dt_warn_label)

        layout.addWidget(self._hint(
            "On the plot, the grey numbers above the plot are this grid: the "
            "images that are cut into parts and exported. The blue numbers "
            "inside the plot are the images counted in the scan step "
            "(section 4). They match until you change Images in grid or "
            "Grid offset."
        ))
        layout.addWidget(self._hint(
            "An image is the slice between two grid lines. Set From and To "
            "(or use Pick on Plot), then Add Part. The order of the list is "
            "the order of the breath. From equal to To selects one image."
        ))

        # Range entry: explicit begin/end image numbers.
        self.part_from_spin = QSpinBox()
        self.part_from_spin.setRange(1, 1)
        self.part_to_spin = QSpinBox()
        self.part_to_spin.setRange(1, 1)
        range_row = QHBoxLayout()
        range_row.addWidget(QLabel("From"))
        range_row.addWidget(self.part_from_spin, 1)
        range_row.addWidget(QLabel("To"))
        range_row.addWidget(self.part_to_spin, 1)
        layout.addLayout(range_row)

        pick_row = QHBoxLayout()
        self.pick_part_btn = QPushButton("Pick on Plot")
        self.pick_part_btn.setToolTip("Click two images on the plot to fill From and To.")
        self.pick_part_btn.clicked.connect(self._arm_add_part)
        pick_row.addWidget(self.pick_part_btn)
        add_part_btn = QPushButton("Add Part")
        add_part_btn.setToolTip("Add a part covering From..To images, in assembly order.")
        add_part_btn.clicked.connect(self._add_part_from_spins)
        pick_row.addWidget(add_part_btn)
        layout.addLayout(pick_row)

        self.capture_hint = QLabel("")
        self.capture_hint.setObjectName("captureHint")
        self.capture_hint.setWordWrap(True)
        layout.addWidget(self.capture_hint)

        self.parts_list = QListWidget()
        self.parts_list.setMinimumHeight(88)
        self.parts_list.setMaximumHeight(140)
        layout.addWidget(self.parts_list)

        actions = QGridLayout()
        actions.setSpacing(4)
        for position, (_text, _slot) in enumerate([
            ("Move Up", self._move_part_up), ("Move Down", self._move_part_down),
            ("Edit Label", self._edit_part_label), ("Delete", self._delete_part),
            ("Clear Parts", self._clear_parts), ("Clear All", self._clear_synth_all),
        ]):
            _b = QPushButton(_text)
            _b.clicked.connect(_slot)
            actions.addWidget(_b, position // 3, position % 3)
        layout.addLayout(actions)

        # The Export button below the sections assembles and exports these.
        self.synth_status_label = self._hint("no parts")
        layout.addWidget(self.synth_status_label)
        return page

    def _build_landmarks_page(self) -> QWidget:
        """Landmark method: mark where inhale starts and where exhale ends."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        mark_inhale_btn = QPushButton("Mark Inhale Start")
        mark_inhale_btn.setToolTip("Then click the plot where inhale starts.")
        mark_inhale_btn.clicked.connect(self._mark_inhale_start)
        self.ct_inhale_label = self._readout("--")
        inhale_row = QHBoxLayout()
        inhale_row.addWidget(mark_inhale_btn, 1)
        inhale_row.addWidget(self.ct_inhale_label, 1)
        layout.addLayout(inhale_row)

        mark_exhale_btn = QPushButton("Mark Exhale End")
        mark_exhale_btn.setToolTip("Then click the plot where exhale ends.")
        mark_exhale_btn.clicked.connect(self._mark_exhale_end)
        self.ct_exhale_label = self._readout("--")
        exhale_row = QHBoxLayout()
        exhale_row.addWidget(mark_exhale_btn, 1)
        exhale_row.addWidget(self.ct_exhale_label, 1)
        layout.addLayout(exhale_row)

        clear_landmarks_btn = QPushButton("Clear Landmarks")
        clear_landmarks_btn.clicked.connect(self._clear_ct_landmarks)
        layout.addWidget(clear_landmarks_btn)
        layout.addStretch(1)
        return page

    def _build_cycle_page(self) -> QWidget:
        """Cycle method: click frames in breathing order to build one cycle."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.cycle_mode_btn = QPushButton("Label Cycle Frames")
        self.cycle_mode_btn.setCheckable(True)
        self.cycle_mode_btn.setToolTip(
            "Toggle: click the images in breathing order to build a cycle. "
            "A click takes the image that starts at the nearest grid line. "
            "Right click removes the most recent label."
        )
        self.cycle_mode_btn.toggled.connect(self._on_cycle_mode_toggled)
        self.cycle_count_label = self._readout("0 frames")
        label_row = QHBoxLayout()
        label_row.addWidget(self.cycle_mode_btn, 1)
        label_row.addWidget(self.cycle_count_label, 1)
        layout.addLayout(label_row)

        self.cycle_seq_label = self._hint("seq: none")
        self.cycle_seq_label.setToolTip("Original image indices in click order")
        layout.addWidget(self.cycle_seq_label)

        clear_cycle_btn = QPushButton("Clear Cycle Labels")
        clear_cycle_btn.clicked.connect(self._clear_cycle_labels)
        layout.addWidget(clear_cycle_btn)
        layout.addStretch(1)
        return page

    def _build_export_footer(self) -> QWidget:
        """Subject, output folder and the Export button, always on screen."""
        footer = QFrame()
        footer.setObjectName("exportFooter")
        layout = QVBoxLayout(footer)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        self.subject_edit = QLineEdit()
        self.subject_edit.setPlaceholderText("for example OSAMRI029")
        self.subject_edit.textChanged.connect(self._update_outdir)

        self.outdir_edit = QLineEdit(str(Path.cwd()))
        # textEdited fires for typing only, never for setText, so the folder
        # counts as hand picked exactly when the user typed it.
        self.outdir_edit.textEdited.connect(self._mark_outdir_user_set)
        outdir_browse_btn = QPushButton("...")
        outdir_browse_btn.setFixedWidth(32)
        outdir_browse_btn.setToolTip("Choose the output folder.")
        outdir_browse_btn.clicked.connect(self._browse_outdir)
        outdir_row = QHBoxLayout()
        outdir_row.addWidget(self.outdir_edit, 1)
        outdir_row.addWidget(outdir_browse_btn)

        form = self._form()
        form.addRow("Subject ID", self.subject_edit)
        form.addRow("Output folder", outdir_row)
        layout.addLayout(form)

        # Says what is still missing, or what exactly will be exported.
        self.export_hint = self._hint("")
        layout.addWidget(self.export_hint)

        self.execute_btn = QPushButton("Export")
        self.execute_btn.setObjectName("primaryButton")
        self.execute_btn.clicked.connect(lambda: self._export())
        layout.addWidget(self.execute_btn)
        return footer

    def _connect_live_updates(self) -> None:
        """Wire the inputs whose changes redraw the plot."""
        # Commit on Enter or focus-out rather than on every keystroke. With
        # Qt's default keyboard tracking, typing "15" emits valueChanged for
        # "1" first, so the grid would recompute for a one-image window and
        # flicker before settling. Stepping with the arrows still reacts at
        # once, because that path is not keyboard editing.
        for _spin in (
            self.start_spin, self.end_spin,
            self.ct_temporal_res_spin, self.ct_total_images_spin,
            self.synth_dt_spin, self.synth_images_spin,
            self.synth_window_offset_spin, self.synth_grid_offset_spin,
        ):
            _spin.setKeyboardTracking(False)

        # Grid inputs redraw the grid live, so the drawn numbers can never
        # disagree with where clicks actually land.
        self.synth_dt_spin.valueChanged.connect(self._on_synth_dt_changed)
        self.synth_images_spin.valueChanged.connect(self._on_synth_images_changed)
        self.synth_grid_offset_spin.valueChanged.connect(
            lambda _v: self._refresh_synth_grid_if_active()
        )
        self.ct_temporal_res_spin.valueChanged.connect(self._on_ct_dt_changed)
        self.start_spin.valueChanged.connect(lambda _v: self._refresh_synth_grid_if_active())
        self.end_spin.valueChanged.connect(lambda _v: self._refresh_synth_grid_if_active())
        # Re-zoom once the value is committed (Enter or focus out) rather than
        # on every keystroke, which would lurch the view while typing.
        self.start_spin.editingFinished.connect(self._zoom_to_window)
        self.end_spin.editingFinished.connect(self._zoom_to_window)

        # The line above Export tracks everything it depends on.
        self.subject_edit.textChanged.connect(lambda _t: self._update_export_hint())
        self.start_spin.valueChanged.connect(lambda _v: self._update_export_hint())
        self.end_spin.valueChanged.connect(lambda _v: self._update_export_hint())
        # Everything the list of blockers reads refreshes the line.
        for box in (self.ct_temporal_res_spin, self.synth_dt_spin, self.synth_images_spin):
            box.valueChanged.connect(lambda _v: self._update_export_hint())
        self.start_spin.valueChanged.connect(lambda _v: self._update_window_readout())
        self.end_spin.valueChanged.connect(lambda _v: self._update_window_readout())

    def _update_window_readout(self) -> None:
        """Show the window length, or say why the window is not usable yet."""
        start, end = self.start_spin.value(), self.end_spin.value()
        if start == 0 and end == 0:
            text = "Not set yet. Click the plot twice, or type the times."
        elif end <= start:
            text = "End must be after Start."
        else:
            text = f"Duration {end - start:.3f} s"
        self.window_readout.setText(text)

    def _on_processing_done(self, msg: str) -> None:
        """GUI-thread completion handler for the Execute worker."""
        self.status_bar.showMessage(msg)
        self.execute_btn.setEnabled(True)

    def _on_processing_failed(self, error_msg: str) -> None:
        """GUI-thread error handler for the Execute worker."""
        QMessageBox.critical(self, "Processing Error", error_msg)
        self.status_bar.showMessage(f"Error: {error_msg}")
        self.execute_btn.setEnabled(True)

    # ------------------------------------------------------------------
    # Helpers for Tk-StringVar-like access (thin wrappers)
    # ------------------------------------------------------------------

    def _get_start_var(self) -> str:
        return f"{self.start_spin.value():.3f}"

    def _get_end_var(self) -> str:
        return f"{self.end_spin.value():.3f}"

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _workflow_name(self) -> str:
        """The chosen study type as the handlers spell it: MRI, cpap,
        phase-contrast or CT."""
        return self.workflow_combo.currentData()

    def _ct_method(self) -> str:
        """The chosen CT method: parts, landmarks or cycle."""
        return self.ct_method_combo.currentData()

    def _on_workflow_changed(self) -> None:
        """Show only the sections that belong to the chosen study type."""
        index = self.workflow_combo.currentIndex()
        if (index != self._workflow_index_shown and self._plot_mode == "selected"
                and not self._ask_before_dropping(
                    "Changing the study type redraws the plot and")):
            self.workflow_combo.blockSignals(True)
            self.workflow_combo.setCurrentIndex(self._workflow_index_shown)
            self.workflow_combo.blockSignals(False)
            return
        self._workflow_index_shown = index
        is_ct = self._workflow_name() == "CT"
        self.study_hint.setText(
            self.workflow_combo.itemData(
                self.workflow_combo.currentIndex(), Qt.ToolTipRole
            )
        )
        entering_ct = is_ct and self.ct_scan_section.isHidden()
        self.ct_scan_section.setVisible(is_ct)
        self.ct_method_section.setVisible(is_ct)
        self.trigger_section.setVisible(not is_ct)
        self._sync_ct_method()
        # Trigger detection belongs to the study type: pulses for most, scan
        # steps for CT. Redo it for an open block, or Select Step finds none.
        if self.data is not None and self._plot_mode == "selected":
            self._plot_selected()
        if entering_ct:
            # The CT sections open below the fold. Bring them up, as the old
            # window did by jumping to its CT tab. Deferred until the sidebar
            # has laid the newly shown sections out.
            QTimer.singleShot(
                0, lambda: self._scroll_sidebar_to(self.ct_scan_section)
            )

    def _scroll_sidebar_to(self, section: QWidget) -> None:
        """Bring a sidebar section to the top of the visible area."""
        content = self.sidebar_scroll.widget()
        content.layout().activate()  # geometry of just-shown sections
        self.sidebar_scroll.verticalScrollBar().setValue(max(0, section.y() - 8))

    def _sync_ct_method(self) -> None:
        """Arm exactly the chosen CT method and disarm the other two.

        Manual mode used to have its own enable switch on another tab. It is
        now on while CT with the parts method is chosen, and off otherwise.
        Leaving a method also disarms its click mode, so a click can never
        land in a method that is no longer on screen.
        """
        is_ct = self._workflow_name() == "CT"
        method = self._ct_method() if is_ct else None
        if method != "cycle" and self.cycle_mode_btn.isChecked():
            self.cycle_mode_btn.setChecked(False)
        if method != "landmarks":
            self._ct_landmark_mode = None
        self.synth_enable_chk.setChecked(method == "parts")
        self._refresh_banner()
        self._update_export_hint()

    def _ct_export_inputs(self):
        """(cycle frames, inhale start, exhale end) of the chosen CT method.

        Only the method on screen feeds the export. Cycle labels used to
        override landmarks whenever any existed, even ones left over from
        before the user changed method.
        """
        method = self._ct_method()
        if method == "cycle":
            return list(self._ct_cycle_frames), None, None
        if method == "landmarks":
            return [], self._ct_inhale_start, self._ct_exhale_end
        return [], None, None

    def _seed_synth_image_count(self) -> None:
        """Start the parts image count from the selected step, once.

        Only while it still holds its initial 1. After that the count is the
        user's own and nothing rewrites it.
        """
        if (self.synth_images_spin.value() <= 1
                and self.ct_total_images_spin.value() > 1):
            self.synth_images_spin.setValue(self.ct_total_images_spin.value())

    def _marks_of_method(self, method: str) -> str:
        """What a CT method has on the plot, in words, or '' if nothing."""
        if method == "landmarks":
            marked = self._ct_inhale_start is not None or self._ct_exhale_end is not None
            return "the landmarks" if marked else ""
        n = len(self._ct_cycle_frames) if method == "cycle" else len(self._parts)
        noun = "cycle label" if method == "cycle" else "part"
        return f"{n} {noun}{'s' if n != 1 else ''}" if n else ""

    def _on_ct_method_changed(self) -> None:
        """Bring the chosen CT method's controls on screen, and only those.

        Leaving a method removes its marks, after asking. The marks of the
        method being left used to stay drawn although the export ignored
        them, so the plot showed one thing and the files held another.
        """
        new, old = self._ct_method(), self._active_ct_method
        if new != old:
            marks = self._marks_of_method(old)
            if marks and QMessageBox.question(
                self, "Change Method?",
                f"Only the method on screen is exported, so changing it "
                f"removes {marks}.\n\nContinue?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            ) != QMessageBox.Yes:
                self.ct_method_combo.blockSignals(True)
                self.ct_method_combo.setCurrentIndex(self.ct_method_combo.findData(old))
                self.ct_method_combo.blockSignals(False)
                return
            if marks:
                {"landmarks": self._clear_ct_landmarks,
                 "cycle": self._clear_cycle_labels,
                 "parts": self._clear_parts}[old]()
            self._active_ct_method = new
        index = self.ct_method_combo.currentIndex()
        self.ct_method_hint.setText(
            self.ct_method_combo.itemData(index, Qt.ToolTipRole)
        )
        self.ct_method_stack.setCurrentIndex(index)
        # A stacked widget is as tall as its tallest page. Letting the hidden
        # pages give up their height keeps a short method from leaving a gap.
        for i in range(self.ct_method_stack.count()):
            policy = QSizePolicy.Preferred if i == index else QSizePolicy.Ignored
            self.ct_method_stack.widget(i).setSizePolicy(QSizePolicy.Preferred, policy)
        self.ct_method_stack.adjustSize()
        self._sync_ct_method()
        if self._workflow_name() == "CT" and new != old:
            # Replaces a line about the method just left.
            self.status_bar.showMessage(_CT_NEXT_STEP[new] + ".")

    def _update_outdir(self, *_args) -> None:
        """Recompute output dir based on current mat file and subject.

        Leaves a folder the user picked by hand alone. Typing the Subject ID
        used to replace it with the default next to the recording.
        """
        if self.mat_filepath is None or self._outdir_user_set:
            return
        subject = self.subject_edit.text().strip()
        if subject:
            results_dir = self.mat_filepath.parent / f"FlowResults_{subject}"
        else:
            results_dir = self.mat_filepath.parent / "FlowResults"
        self.outdir_edit.setText(str(results_dir))

    # File types offered by the open dialog. Both kinds are listed together
    # first, so a native recording shows up without changing the filter.
    _OPEN_FILTER = (
        "LabChart recordings (*.mat *.adicht);;"
        "MATLAB exports (*.mat);;"
        "Native LabChart files (*.adicht);;"
        "All files (*.*)"
    )

    def _browse_file(self) -> None:
        """Open file dialog to select a recording (.mat export or .adicht)."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open LabChart recording", "", self._OPEN_FILTER,
        )
        if filepath:
            self.open_recording(filepath)

    def _set_busy(self, busy: bool) -> None:
        """Lock the window and show a wait cursor during a long load.

        A native recording takes many seconds to decode. Until it is in
        memory the previous recording still is, so nothing may act on it.
        """
        self.centralWidget().setEnabled(not busy)
        self.menuBar().setEnabled(not busy)
        if busy:
            QApplication.setOverrideCursor(Qt.WaitCursor)
        else:
            QApplication.restoreOverrideCursor()

    def _update_load_notes(self) -> None:
        """Show what the reader recomputed or left out, under the file name."""
        self.load_notes.setText("\n".join(self._load_notes))
        self.load_notes.setVisible(bool(self._load_notes))

    def open_recording(self, path) -> None:
        """Load a recording (.mat export or native .adicht) and show its blocks.

        One action from the user's side: choosing the file is enough to see
        the recording, with no separate load step.
        """
        path = Path(path)
        if self.data is not None:
            another = path != getattr(self, "_loaded_filepath", None)
            if not self._ask_before_dropping("Opening a recording", include_parts=another):
                return
        self.mat_filepath = path
        self.file_label.setText(self.mat_filepath.name)
        self.file_label.setToolTip(str(self.mat_filepath))
        # A new recording goes back to the default folder beside it.
        self._outdir_user_set = False
        self._update_outdir()
        self._load_and_overview()
        if self.data is None:  # the load failed and has already said why
            self.file_label.setText(f"Could not open {self.mat_filepath.name}")

    def _browse_outdir(self) -> None:
        """Open directory dialog to select output directory."""
        dirpath = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dirpath:
            self._set_output_dir(dirpath)

    def _set_output_dir(self, path) -> None:
        """Use a folder the user chose; it then stays put until a new recording."""
        self.outdir_edit.setText(str(path))
        self._outdir_user_set = True

    def _mark_outdir_user_set(self, *_args) -> None:
        self._outdir_user_set = True

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    _PRESSURE_MISSING = (
        "No pressure profile was produced: the pressure channel is empty in "
        "this block, or this study type has no pressure output."
    )

    def _pressure_warnings(self, requested: bool, note: Optional[str],
                           pressure_file: Path) -> List[str]:
        """Warnings for an export that wrote no pressure profile.

        Touches no widget, so the export worker thread may call it.
        """
        warnings = []
        if requested:
            warnings.append(
                f"No pressure profile: {note}." if note else self._PRESSURE_MISSING
            )
        # A pressure file from an earlier export would sit beside the new
        # flow profile and read as its partner.
        if pressure_file.exists():
            warnings.append(
                f"An older {pressure_file.name} is still in this folder. This "
                "export did not write it, so it does not belong to the new "
                "flow profile."
            )
        return warnings

    def _uses_assembly(self) -> bool:
        """True when Export assembles ordered parts (CT, parts method)."""
        return self._workflow_name() == "CT" and self._ct_method() == "parts"

    def _export_blockers(self) -> List[str]:
        """What is still missing before an export can run, in plain words."""
        blockers = []
        if self.data is None:
            blockers.append("open a recording")
        elif self.data.is_block_empty(self.flow_row_spin.value(), self.col_spin.value()):
            # Export on such a block used to end the program.
            blockers.append(
                "pick a block that has flow data (this one is empty on the "
                "flow channel)"
            )
        if not self.subject_edit.text().strip():
            blockers.append("enter a Subject ID")
        if self._uses_assembly():
            if not self._parts:
                blockers.append("add at least one part")
            if self._synth_grid() is None:
                # Shorter than one flow sample, or no image: no grid exists.
                blockers.append(
                    "set a Temporal Res of at least one flow sample "
                    f"({1000.0 / self._SYNTH_SAMPLE_RATE:g} ms) and at least one image"
                )
        elif self.end_spin.value() <= self.start_spin.value():
            blockers.append("set Start before End")
        elif self._workflow_name() == "CT":
            # The grid is not drawn for such a count, so the screen would
            # show one number of images and the files hold another.
            dt_ms = self.ct_temporal_res_spin.value()
            window = self.end_spin.value() - self.start_spin.value()
            images = int(round(window / (dt_ms / 1000.0))) if dt_ms > 0 else 0
            if images > self._MAX_GRID_IMAGES:
                blockers.append(
                    f"raise Temporal Res ({dt_ms:g} ms gives {images} images in "
                    f"this window, the limit is {self._MAX_GRID_IMAGES})"
                )
        return blockers

    def _ct_export_description(self, study: str, window: str) -> str:
        """What a CT export through Execute will really hold.

        With nothing marked, the landmark and cycle methods both export the
        window as recorded. Saying "landmarks method" then was untrue.
        """
        frames, inhale, _exhale = self._ct_export_inputs()
        if self._ct_method() == "cycle":
            if not frames:
                return (f"{study}, whole window as recorded "
                        f"(no cycle frame labelled), {window}")
            n = len(frames)
            return f"{study}, cycle frames method, {n} frame{'s' if n != 1 else ''}, {window}"
        if self._ct_inhale_start is None and self._ct_exhale_end is None:
            return f"{study}, whole window as recorded (no landmark marked), {window}"
        if inhale is None:
            return (f"{study}, whole window as recorded "
                    f"(inhale start not marked), {window}")
        if self._ct_exhale_end is None:
            return f"{study}, landmarks method (inhale start only), {window}"
        return f"{study}, landmarks method, {window}"

    def _update_export_hint(self) -> None:
        """State what is missing, or exactly what Export will write."""
        blockers = self._export_blockers()
        if blockers:
            self.export_hint.setText("Before export: " + ", ".join(blockers) + ".")
            return
        study = self.workflow_combo.currentText()
        if self._uses_assembly():
            n = len(self._parts)
            what = f"{study}, reorder parts method, {n} part{'s' if n != 1 else ''}"
        else:
            window = f"{self.start_spin.value():.3f} s to {self.end_spin.value():.3f} s"
            if self._workflow_name() == "CT":
                what = self._ct_export_description(study, window)
            else:
                what = f"{study}, {window}"
        self.export_hint.setText(f"Ready to export: {what}.")

    def _export_targets(self) -> List[Path]:
        """Every file the chosen export may write."""
        subject = self.subject_edit.text().strip()
        folder = Path(self.outdir_edit.text())
        names = [f"{subject}FlowProfile.csv"]
        # Pressure is written only when a pressure channel is chosen and has
        # data in this block, and never for phase contrast, which has no
        # pressure output. Naming a file that would not be written made the
        # question promise a replacement that did not happen.
        pressure_row = self.pressure_row_spin.value()
        if (pressure_row > 0
                and self._workflow_name() != "phase-contrast"
                and self.data is not None
                and not self.data.is_block_empty(pressure_row, self.col_spin.value())):
            names.append(f"{subject}PressureProfile.csv")
        if self._uses_assembly():
            names.append(f"{subject}ImageReorderMap.csv")
        else:
            names += [f"{subject}_trigger_info.txt", f"{subject}_sanity_check.png"]
        return [folder / name for name in names]

    def _confirm_overwrite(self, targets: List[Path]) -> bool:
        """Ask before replacing files of an earlier export. True means go on."""
        existing = [path for path in targets if path.exists()]
        if not existing:
            return True
        answer = QMessageBox.question(
            self, "Replace Existing Files?",
            f"{existing[0].parent} already holds:\n\n  "
            + "\n  ".join(path.name for path in existing)
            + "\n\nReplace them with this export?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        return answer == QMessageBox.Yes

    def _export(self) -> None:
        """Run the export that belongs to the chosen study type and CT method.

        This is the one exit. Execute on one tab and Assemble & Export on
        another used to be two, and which one applied was the user's problem.
        """
        # The button is disabled while an export runs, but the menu item and
        # its shortcut also land here.
        if not self.execute_btn.isEnabled():
            self.status_bar.showMessage("An export is already running.")
            return
        # The list behind the line above the button is the gate. The line
        # and the export used to keep separate lists, so the line could name
        # a problem while Export wrote the files, or say Ready while Export
        # refused. The message is the line itself, word for word.
        if self._export_blockers():
            self._update_export_hint()
            QMessageBox.warning(self, "Cannot Export Yet", self.export_hint.text())
            return
        # A channel named like another role would be exported as if it were
        # the right one, with nothing wrong in the output files.
        name_warnings = self._channel_name_warnings()
        if name_warnings and QMessageBox.question(
            self, "Check Channels",
            "\n".join(name_warnings)
            + "\n\nExport with these channels anyway?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        ) != QMessageBox.Yes:
            self.status_bar.showMessage("Export cancelled. No file was changed.")
            return
        if not self._confirm_overwrite(self._export_targets()):
            self.status_bar.showMessage("Export cancelled. No file was changed.")
            return
        if self._uses_assembly():
            self._assemble_and_export()
        else:
            self._execute_processing()

    def _show_export_result(self, report: dict) -> None:
        """Keep the report of a finished export and show it in a panel.

        Shown without blocking, so the main window stays usable while the
        panel is open.
        """
        self._last_export_report = report
        if self._result_dialog is not None:
            self._result_dialog.close()
            self._result_dialog.deleteLater()
        self._result_dialog = ExportResultDialog(report, parent=self)
        self._result_dialog.show()

    def _disconnect_click(self) -> None:
        """Disconnect canvas click callback if connected."""
        if self._click_cid is not None:
            self.canvas.mpl_disconnect(self._click_cid)
            self._click_cid = None

    def _reset_plot_state(self) -> None:
        """Drop references to old plot artists and reset trigger/CT state.

        Call this before ``fig.clear()`` or any replot. Lists are reassigned to
        ``[]`` rather than ``.clear()``-ed because ``fig.clear()`` invalidates
        the artists; later ``artist.remove()`` calls would raise and abort the
        replot (symptom: GUI stops drawing after switching blocks).
        """
        self._disconnect_click()
        self._overview_axes = {}
        self._envelopes = []
        self._time_markers = []
        self._trigger_markers = []
        self._ct_step_markers = []
        self._ct_step_labels = []
        self._ct_interval_markers = []
        self._ct_landmark_artists = []
        self._comment_artists = []
        self._comment_labels = {}
        self._comments_drawn = []
        self._comment_layout_key = None
        self._comment_labels_withheld = False
        self._y_reference = {}
        self._title_forms = []
        self._trigger_times = np.array([])
        self._trigger_count = 0
        self._ct_steps = []
        self._ct_selected_step = None
        self._ct_inhale_start = None
        self._ct_exhale_end = None
        self._ct_detected_start = None
        self._ct_detected_end = None
        self._ct_landmark_mode = None
        # Cycle labeling state (new mode, independent from landmarks)
        self._ct_cycle_frames = []
        self._ct_cycle_artists = []
        # Do not change self._ct_cycle_mode — it's a UI toggle, user controls it.
        # Full-manual synthetic-profile artists become invalid after fig.clear();
        # drop the references but keep self._parts (the user's labeling data) and
        # the UI toggle. Grid/overlay are redrawn on demand via "Show Image Grid".
        self._grid_artists = []
        self._part_artists = []
        self._capturing_part = False
        self._part_click_buffer = []
        self._capture_marker = None
        self.total_triggers_label.setText("--")
        self.ct_inhale_label.setText("--")
        self.ct_exhale_label.setText("--")
        # Count and sequence together. Resetting the count alone left the old
        # sequence on screen beside "0 frames".
        self._update_cycle_ui()

    def _load_and_overview(self) -> None:
        """Load the chosen recording and show the grid of all channels x blocks."""
        if self.mat_filepath is None:
            QMessageBox.warning(self, "No File", "Open a recording first.")
            return
        if getattr(self, "_loading", False):
            return  # processEvents below can re-dispatch a queued click here
        self._loading = True

        try:
            self._reset_plot_state()

            self.status_bar.showMessage(f"Loading {self.mat_filepath.name}...")
            QApplication.processEvents()

            # A native recording takes many seconds to decode, so say that the
            # application is busy. Whatever the reader recomputed or left out
            # comes back as notes and is shown under the file name.
            notes: List[str] = []
            self._load_notes = []
            name = self.mat_filepath.name

            def _show_progress(number: int, total: int, what: str) -> None:
                self.status_bar.showMessage(
                    f"Decoding {name}: {number} of {total} ({what})..."
                )
                # Keep the window painting, but take no clicks or keys while
                # the previous recording is still the one in memory.
                QApplication.processEvents(QEventLoop.ExcludeUserInputEvents)

            self._set_busy(True)
            try:
                self.data = load_recording(
                    self.mat_filepath, notes=notes, progress=_show_progress
                )
            except Exception as e:
                self._set_busy(False)
                self._update_load_notes()
                # A failed load must not leave the PREVIOUS file's data alive
                # under the new file's name: the plot would title old data with
                # the new name and Execute would export it beside the new file.
                self.data = None
                self._plot_mode = "empty"
                self.fig.clear()
                self.canvas.draw_idle()
                self._refresh_banner()
                QMessageBox.critical(self, "Load Error", str(e))
                self.status_bar.showMessage("Error loading file")
                return
            self._set_busy(False)
            self._load_notes = notes
            self._update_load_notes()

            # A different recording invalidates any parts labelled against the
            # previous file's grid, and with them the window, the image count
            # and the offsets, which all described a scan that is no longer
            # open. The image count goes back to 1 so the next selected step
            # seeds it again.
            if getattr(self, "_loaded_filepath", None) != self.mat_filepath:
                self._parts = []
                self._refresh_parts_list()
                for box, value in (
                    (self.start_spin, 0.0), (self.end_spin, 0.0),
                    (self.synth_images_spin, 1), (self.ct_total_images_spin, 1),
                    (self.synth_grid_offset_spin, 0), (self.synth_window_offset_spin, 0),
                    (self.ct_shift_spin, 0), (self.ct_shift_slider, 0),
                ):
                    # Silently: the handlers would redraw a grid on a plot
                    # that is about to be replaced.
                    box.blockSignals(True)
                    box.setValue(value)
                    box.blockSignals(False)
                self._update_window_readout()
            self._loaded_filepath = self.mat_filepath
        finally:
            self._loading = False
        self._show_overview()

    def _channel_titles(self) -> List[str]:
        """One title per channel of the loaded recording, blank where unnamed."""
        nch = self.data.num_channels
        titles = list(self.data.titles or [])
        return (titles + [""] * nch)[:nch]

    def _channel_text(self, row: int) -> str:
        """Short name for a channel row, such as "ch 2 Flow"."""
        if self.data is None or row < 1 or row > self.data.num_channels:
            return f"ch {row}"
        return f"ch {row} {self._channel_titles()[row - 1]}".strip()

    def _channel_name_warnings(self) -> List[str]:
        """One line per role whose chosen channel is named like something else.

        A role is flagged only when its channel's title lacks the role's word
        while another channel's title has it. Titles are a hint, so nothing
        is changed here: channel layouts differ between recordings and the
        choice stays the user's. On a real recording the default rows put
        Flow on an esophageal pressure channel, and nothing said so.
        """
        if self.data is None:
            return []
        titles = self._channel_titles()
        warnings = []
        for role, picker_name, words in _ROLE_TITLE_WORDS:
            row = getattr(self, picker_name).value()
            if row < 1 or row > len(titles):
                continue  # "None", or not a channel of this recording

            def fits(title: str) -> bool:
                return any(word in title.lower() for word in words)

            if fits(titles[row - 1]):
                continue
            named = [channel_label(i + 1, t) for i, t in enumerate(titles) if fits(t)]
            if named:
                warnings.append(
                    f"{role} is set to {channel_label(row, titles[row - 1])}, "
                    f"but {' and '.join(named)} "
                    f"{'is' if len(named) == 1 else 'are'} named like "
                    f"{role.lower()}."
                )
        return warnings

    def _update_channel_check(self) -> None:
        """Show or hide the channel name warning under the pickers."""
        warnings = self._channel_name_warnings()
        self.channel_check.setText("\n".join(warnings))
        self.channel_check.setVisible(bool(warnings))

    def _role_of(self, row: int) -> Optional[str]:
        """Role the pickers give a channel row: flow, trigger, pressure or None."""
        if row == self.flow_row_spin.value():
            return "flow"
        if row == self.trigger_row_spin.value():
            return "trigger"
        if row == self.pressure_row_spin.value():
            return "pressure"
        return None

    def _go_to_overview(self) -> None:
        """Return to the all-blocks overview of the loaded recording."""
        if self.data is None:
            self.status_bar.showMessage("Open a recording first.")
            return
        self._reset_plot_state()
        self._show_overview()

    def _plotted_selection(self) -> Tuple[int, int, int]:
        """(flow row, trigger row, block): everything the plot is drawn from.

        Pressure is not in it, because pressure is never plotted.
        """
        return (self.flow_row_spin.value(), self.trigger_row_spin.value(),
                self.col_spin.value())

    def _marks_at_risk(self, include_parts: bool) -> List[str]:
        """What a redraw would remove, in the words shown to the user."""
        marks = []
        if self._ct_inhale_start is not None or self._ct_exhale_end is not None:
            marks.append("the landmarks")
        n = len(self._ct_cycle_frames)
        if n:
            marks.append(f"{n} cycle label{'s' if n != 1 else ''}")
        n = len(self._parts)
        if include_parts and n:
            marks.append(f"{n} part{'s' if n != 1 else ''}")
        return marks

    def _ask_before_dropping(self, what: str, include_parts: bool = False) -> bool:
        """Ask before an action removes marks the user placed. True to go on.

        One rule for every control: landmarks, cycle labels and parts are
        never removed without a question. Only the Clear buttons, whose
        whole purpose is to remove them, do not ask. Before this rule the
        pickers asked while All blocks, Selected block, Select Step, the
        study type and opening a recording dropped the same marks silently.
        """
        at_risk = self._marks_at_risk(include_parts)
        if not at_risk:
            return True
        return QMessageBox.question(
            self, "Remove Markers?",
            f"{what} removes:\n\n  " + "\n  ".join(at_risk) + "\n\nContinue?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        ) == QMessageBox.Yes

    def _user_show_overview(self) -> None:
        """All blocks, from the button or the menu."""
        if self._ask_before_dropping("Showing all blocks"):
            self._go_to_overview()

    def _user_show_block(self) -> None:
        """Selected block, from the button or the menu."""
        if self._ask_before_dropping("Redrawing the block"):
            self._plot_selected()

    def _user_select_step(self) -> None:
        """Select Step, from the button."""
        if self._ask_before_dropping("Selecting a step redraws its image grid and"):
            self._select_ct_step()

    def _on_selection_changed(self, *_args) -> None:
        """Redraw the current view after the user changes a channel or block.

        The pickers used to need a separate Plot Selected click, so the plot
        could show one block while the pickers named another.

        Redrawing drops the markers placed on the old signal, so it happens
        only when something on the plot really changed, and it asks first when
        work would be lost. Landmarks used to vanish on a Pressure change
        while the line above Export still said "Ready".
        """
        if self.data is None:
            return
        self._update_channel_check()
        self._update_export_hint()
        new = self._plotted_selection()
        old = self._shown_selection
        if new == old:
            return  # the same entry again, or only Pressure changed
        was_detail = self._plot_mode == "selected"
        block_changed = old is not None and new[2] != old[2]
        if old is not None:
            # Parts name images of one scan, so they go with the block only.
            # They outlive the detail view, so this holds on the overview too.
            if not self._ask_before_dropping(
                    "Changing the channel or block redraws the plot and",
                    include_parts=block_changed):
                self.flow_row_spin.setValue(old[0])
                self.trigger_row_spin.setValue(old[1])
                self.col_spin.setValue(old[2])
                self._update_channel_check()
                self._update_export_hint()
                return
            if block_changed and self._parts:
                self._parts = []
                self._refresh_parts_list()
        col = new[2]
        if was_detail and not self.data.is_block_empty(self.flow_row_spin.value(), col):
            self._plot_selected()
            return
        self._go_to_overview()
        if was_detail:
            self.status_bar.showMessage(
                f"Block {col} has no data on the flow channel, so all blocks "
                "are shown. Pick another block or another Flow channel."
            )

    def _on_overview_click(self, event) -> None:
        """Open the block whose overview tile was clicked."""
        if self._plot_mode != "overview" or event.button != 1:
            return
        if self.toolbar.mode:  # zoom or pan owns the click
            return
        hit = next(
            (key for key, ax in self._overview_axes.items() if ax is event.inaxes),
            None,
        )
        if hit is None:
            return
        col = hit[1]
        flow_row = self.flow_row_spin.value()
        if self.data.is_block_empty(flow_row, col):
            self.status_bar.showMessage(
                f"Block {col} is empty on the flow channel "
                f"({self._channel_text(flow_row)}). Pick another block, or "
                "change the Flow channel on the left."
            )
            return
        if col != self.col_spin.value():
            # The parts list names images of the block it was built on.
            # Followed to another block, it cut that block by the wrong
            # image ranges on export.
            if not self._ask_before_dropping("Opening another block", include_parts=True):
                return
            if self._parts:
                self._parts = []
                self._refresh_parts_list()
        self.col_spin.setValue(col)
        self._plot_selected()

    # Groups of samples per overview tile. A tile is a few hundred pixels wide.
    _OVERVIEW_BINS = 250

    def _on_view_changed(self, ax) -> None:
        """Redraw the signal lines for the range now visible.

        Connected to every detail axes, because which of the linked axes
        reports a change depends on which one the user dragged.
        """
        x_lo, x_hi = ax.get_xlim()
        for envelope in self._envelopes:
            envelope.update(x_lo, x_hi)
        # Which labels crowd each other depends on the zoom.
        self._layout_comment_labels()
        self._layout_step_labels()
        self._fence_time_labels()

    def _fence_time_labels(self) -> None:
        """Keep every label placed at a time within the width of its plot.

        Step names, marker times, image numbers and cycle labels sit at a
        time on the x axis and at a height of the axes, often just above the
        top edge, so the plot does not clip them. Zoomed in elsewhere, such
        a label was still drawn, far outside the plot, and the layout
        engine, asked to make room for it, gave up with a warning on every
        redraw. Each is clipped to the width of its plot, not its height,
        and left out of the layout. Labels are added at many places and at
        any time, so this runs on every change of the view and takes each
        label once.
        """
        for ax in (self.ax_top, self.ax_bottom, self.ax_top_trig):
            if ax is None:
                continue
            fence = None
            for text in ax.texts:
                # A text has clipping switched on by default, with nothing to
                # clip against. Only a clip region means it is clipped.
                if (getattr(text, "_time_fenced", False)
                        or text.get_clip_path() is not None
                        or text.get_clip_box() is not None):
                    continue  # done before, or already clipped to the plot
                if fence is None:
                    fence = TransformedBbox(Bbox([[0.0, -1.0], [1.0, 2.0]]), ax.transAxes)
                # The backing shape of a label (the circle of a cycle label)
                # is drawn on its own, so it is fenced as well.
                for artist in (text, text.get_bbox_patch()):
                    if artist is not None:
                        artist.set_clip_path(None)
                        artist.set_clip_box(fence)
                        artist.set_clip_on(True)
                text.set_in_layout(False)
                text._time_fenced = True

    def _show_overview(self) -> None:
        """Draw the channels-x-blocks overview grid for the loaded file."""

        nch = self.data.num_channels
        nbl = self.data.num_blocks

        # Offer exactly the channels and blocks this recording has, by name.
        # A current choice past the last one is clamped to it by the picker.
        titles = self._channel_titles()
        for picker in (self.flow_row_spin, self.trigger_row_spin,
                       self.pressure_row_spin):
            picker.set_items(titles)
        self.col_spin.set_items([""] * nbl)

        # Build overview grid
        self.fig.clear()
        self._plot_mode = "overview"
        self.ax_top = None
        self.ax_bottom = None
        self.ax_top_trig = None
        self._overview_axes = {}

        for ch_idx in range(nch):
            row = ch_idx + 1  # 1-indexed
            role = self._role_of(row)
            color = _ROLE_COLORS.get(role, "0.45")
            # Each row is named, and says which role the pickers give it, so a
            # wrong channel assignment is visible before anything is exported.
            row_label = channel_label(row, titles[ch_idx])
            if role:
                row_label += f"\n({role})"
            for bl_idx in range(nbl):
                ax = self.fig.add_subplot(nch, nbl, ch_idx * nbl + bl_idx + 1)
                col = bl_idx + 1

                if self.data.is_block_empty(row, col):
                    ax.set_facecolor("#f0f0f0")
                    ax.text(
                        0.5, 0.5, "empty", ha="center", va="center",
                        fontsize=7, color="gray", transform=ax.transAxes,
                    )
                else:
                    time = self.data.get_time(row, col)
                    data = self.data.get_data(row, col)
                    # Reduced for speed, by the minimum and maximum of each
                    # group of samples. Keeping every Nth sample instead hid
                    # scan steps shorter than the stride.
                    ax.plot(
                        *minmax_envelope(time, data, time[0], time[-1],
                                         max_bins=self._OVERVIEW_BINS),
                        color=color, linewidth=0.5,
                    )

                if ch_idx == 0:
                    ax.set_title(f"Block {col}", fontsize=9, pad=3)
                if bl_idx == 0:
                    ax.set_ylabel(
                        row_label, rotation=0, ha="right", va="center",
                        fontsize=8, color=color if role else "0.25",
                        # A channel title is drawn as typed, never as a
                        # formula: a dollar sign pair raises while drawing.
                        parse_math=False,
                    )
                ax.tick_params(labelsize=6)
                # Only show axis labels on edges
                if ch_idx < nch - 1:
                    ax.set_xticklabels([])
                if bl_idx > 0:
                    ax.set_yticklabels([])

                self._overview_axes[(row, col)] = ax

        self.canvas.draw()
        self.toolbar.update()

        # A click on a tile opens that block. The slot is the one the detail
        # view uses for its own clicks, so only one handler is ever connected.
        self._disconnect_click()
        self._click_cid = self.canvas.mpl_connect(
            "button_press_event", self._on_overview_click
        )

        self.status_bar.showMessage(
            f"Overview: {nch} channels x {nbl} blocks | Click a block to open it"
        )
        self._shown_selection = self._plotted_selection()
        self._refresh_banner()

    def _plot_selected(self) -> None:
        """Plot a specific block in detail (flow + trigger top, flow-only bottom)."""
        if self.data is None:
            QMessageBox.warning(self, "No Data", "Open a recording first.")
            return

        flow_row = self.flow_row_spin.value()
        trigger_row = self.trigger_row_spin.value()  # 0 = none
        col = self.col_spin.value()
        nch = self.data.num_channels

        # Bounds check
        if flow_row < 1 or flow_row > nch:
            QMessageBox.warning(
                self, "Invalid Flow Row",
                f"Flow Row must be between 1 and {nch}.",
            )
            return
        if trigger_row > nch:
            QMessageBox.warning(
                self, "Invalid Trigger Row",
                f"Trigger Row must be 0 (none) or between 1 and {nch}.",
            )
            return
        if col < 1 or col > self.data.num_blocks:
            QMessageBox.warning(
                self, "Invalid Column",
                f"Column must be between 1 and {self.data.num_blocks}.",
            )
            return

        if self.data.is_block_empty(flow_row, col):
            QMessageBox.warning(self, "Empty Block", f"Flow block [{flow_row}, {col}] is empty.")
            self.status_bar.showMessage("Block is empty")
            return

        try:
            self._draw_detail(flow_row, trigger_row, col)
        except Exception as error:  # noqa: BLE001 (any failure ends the same way)
            # A view drawn half way takes no clicks and sits under a banner
            # that describes something else. Go back to a state that is whole.
            traceback.print_exc()
            self._reset_plot_state()
            self._show_overview()
            QMessageBox.critical(
                self, "Could Not Draw Block",
                f"Block {col} could not be drawn ({type(error).__name__}: "
                f"{error}).\n\nAll blocks are shown instead.",
            )

    def _draw_detail(self, flow_row: int, trigger_row: int, col: int) -> None:
        """Build the detail view of one block. ``_plot_selected`` checks the
        choice first and recovers if this fails."""
        # Clean up previous state (safe to call before fig.clear)
        self._reset_plot_state()

        time = self.data.get_time(flow_row, col)
        flow = self.data.get_data(flow_row, col)

        # Recreate detail axes with shared x-axis
        self.fig.clear()
        self._plot_mode = "selected"

        self.ax_top = self.fig.add_subplot(211)
        self.ax_bottom = self.fig.add_subplot(212, sharex=self.ax_top)
        self.ax_top_trig = None

        # The signal lines draw an envelope of the data at the current zoom
        # and are refreshed when the visible range changes. Drawing all of a
        # 92 million sample channel took 9.5 s per redraw on a real recording.
        for ax in (self.ax_top, self.ax_bottom):
            ax.callbacks.connect("xlim_changed", self._on_view_changed)

        # Top subplot: flow on primary y-axis
        self._envelopes.append(EnvelopeLine(
            self.ax_top, time, flow, "b-", linewidth=0.5, label="flow info"
        ))

        has_trigger = trigger_row >= 1 and not self.data.is_block_empty(trigger_row, col)
        is_ct = self._workflow_name() == "CT"
        if has_trigger:
            trigger = self.data.get_data(trigger_row, col)
            trigger_time = self.data.get_time(trigger_row, col)
            # Trigger on twin y-axis: different amplitude range won't squeeze flow.
            self.ax_top_trig = self.ax_top.twinx()
            self.ax_top_trig.callbacks.connect("xlim_changed", self._on_view_changed)
            self._envelopes.append(EnvelopeLine(
                self.ax_top_trig, trigger_time, trigger, "r-", linewidth=0.5,
                label="trigger info",
            ))
            self.ax_top_trig.set_ylabel("Trigger", color="red")
            self.ax_top_trig.tick_params(axis="y", labelcolor="red")
            # The whole range of the channel in this block, and it stays:
            # no zoom in time refits it (see _fit_y).
            trig_lim = self._channel_y_limits(trigger_row, col, -np.inf, np.inf)
            if trig_lim is not None:
                self.ax_top_trig.set_ylim(*trig_lim)

            if is_ct:
                # CT uses step (bridge) detection, not pulsatile peak detection
                self._detect_steps_from_data(trigger, trigger_time)
            else:
                # MRI/CPAP uses pulsatile peak detection
                self._detect_triggers(trigger, trigger_time)

        self.ax_top.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
        self.ax_top.set_xlabel("Time (s)")
        self.ax_top.set_ylabel("Amplitude")

        if is_ct and has_trigger:
            trigger_info = f" | {len(self._ct_steps)} step triggers"
        elif has_trigger:
            trigger_info = f" | {self._trigger_count} triggers"
        else:
            trigger_info = ""
        # Names, not bare row numbers, so the title says what is on screen.
        title_parts = [
            self.mat_filepath.name if self.mat_filepath else "recording",
            f"Block {col}",
            f"Flow: {self._channel_text(flow_row)}",
        ]
        if has_trigger:
            title_parts.append(f"Trigger: {self._channel_text(trigger_row)}")
        # Padded so the marker times and image numbers drawn along the top
        # edge of the axes sit under the title instead of across it.
        # File name and channel titles are drawn as typed, never as a formula.
        # Shorter forms follow for a window too narrow for the whole title:
        # the file name goes first, because the sidebar shows it too.
        self._title_forms = [
            " | ".join(title_parts) + trigger_info,
            " | ".join(title_parts[1:]) + trigger_info,
            " | ".join(title_parts[1:]),
            " | ".join(title_parts[1:3]),
            title_parts[1],
        ]
        self.ax_top.set_title(self._title_forms[0], pad=16, parse_math=False)
        self.ax_top.grid(True, alpha=0.3)
        self._refresh_top_legend()

        # Bottom subplot: flow only (for zoom/selection)
        self._envelopes.append(EnvelopeLine(
            self.ax_bottom, time, flow, "b-", linewidth=0.5
        ))
        self.ax_bottom.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
        self.ax_bottom.set_xlabel("Time (s)")
        self.ax_bottom.set_ylabel("Flow (L/s)")
        # Raised like the upper title, so the marker times along the top edge
        # sit under it. Without the room the End time printed into the title.
        self.ax_bottom.set_title("Flow only", pad=16)
        self.ax_bottom.grid(True, alpha=0.3)
        self.ax_bottom.format_coord = lambda x, y: f"time={x:.3f}s, flow={y:.4f}"

        n_comments = self._draw_comments()

        self.canvas.draw()
        self.toolbar.update()
        # After the draw, because the labels are placed in pixels and the
        # layout of the axes is only settled by a draw.
        if n_comments:
            self._layout_comment_labels()
        self._fit_title()
        self.canvas.draw_idle()

        # Connect click handler for data picker
        self._click_cid = self.canvas.mpl_connect("button_press_event", self._on_click)

        comment_info = (
            f" | {n_comments} comment{'s' if n_comments != 1 else ''}"
            if n_comments else ""
        )
        if self._comment_labels_withheld:
            comment_info += " (zoom in to read them)"
        self.status_bar.showMessage(
            f"Block {col}, {self._channel_text(flow_row)}: {len(time)} samples, "
            f"{time[-1]:.2f}s{trigger_info}{comment_info}"
        )
        self._shown_selection = self._plotted_selection()
        self._refresh_banner()

    # ------------------------------------------------------------------
    # LabChart comments
    # ------------------------------------------------------------------

    # Not used by any signal or marker on the plot.
    _COMMENT_COLOR = "#7c3aed"

    def _comment_label(self, comment) -> str:
        """The text drawn for a comment.

        A comment typed on a channel that is not plotted names that channel,
        so it is not read as a remark about the flow or the trigger.
        """
        shown = (None, self.flow_row_spin.value(), self.trigger_row_spin.value())
        if comment.channel in shown:
            return comment.text
        return f"{comment.text} ({self._channel_text(comment.channel)})"

    def _draw_comments(self) -> int:
        """Draw the open block's comments, as LabChart does: a dashed line at
        each time, with the text along it. Returns how many there are.

        The lines of one plot are a single collection in axes height
        coordinates, added without touching the data limits, so they never
        change a scale or the range of the time axis and stay cheap however
        many there are. The labels are made by ``_layout_comment_labels``.
        """
        self._comment_artists = []
        self._comment_labels = {}
        self._comment_layout_key = None
        self._comments_drawn = self.data.comments_in(self.col_spin.value())
        if not self._comments_drawn:
            return 0
        visible = self.comments_action.isChecked()
        segments = [[(c.time, 0.0), (c.time, 1.0)] for c in self._comments_drawn]
        for ax in (self.ax_top, self.ax_bottom):
            lines = LineCollection(
                segments, transform=ax.get_xaxis_transform(),
                colors=self._COMMENT_COLOR, linestyles=[(0, (4, 3))],
                linewidths=0.8, alpha=0.9, visible=visible,
            )
            ax.add_collection(lines, autolim=False)
            self._comment_artists.append(lines)
        return len(self._comments_drawn)

    # Where a comment label starts, as a fraction of the axes height, and the
    # room kept between two labels, in pixels.
    _COMMENT_LABEL_BASE = 0.04
    _COMMENT_LABEL_GAP_PX = 4.0
    # With more comments than this in view no label could be read. Only the
    # lines are drawn then, and the labels appear on zooming in.
    _COMMENT_LABELS_MAX = 60

    def _comment_text_artist(self, index: int):
        """The label of one comment, made the first time it comes into view."""
        label = self._comment_labels.get(index)
        if label is None:
            comment = self._comments_drawn[index]
            # On the topmost axes with a pale backing, so the signals do not
            # run through it. Along the line from the bottom edge: the top
            # edge carries the marker times and the image numbers.
            host = self.ax_top_trig if self.ax_top_trig is not None else self.ax_top
            label = host.text(
                comment.time, self._COMMENT_LABEL_BASE, self._comment_label(comment),
                transform=host.get_xaxis_transform(),
                rotation=90, ha="right", va="bottom", fontsize=8,
                color=self._COMMENT_COLOR, clip_on=True, zorder=5,
                # As typed. A dollar sign pair would be read as a formula,
                # and a broken one raises while drawing.
                parse_math=False,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5),
            )
            label.set_in_layout(False)
            self._comment_labels[index] = label
            self._comment_artists.append(label)
        return label

    def _layout_comment_labels(self) -> None:
        """Show the labels of the comments in view, and keep them apart.

        Every label starts at the bottom edge. One that would print over a
        neighbour at the current zoom is moved up to the first free height,
        and comes back down once zooming in separates them. On a real
        recording two comments 13 s apart in a 2300 s block were unreadable.
        """
        if not self._comments_drawn or self.ax_top is None:
            return
        box = self.ax_top.bbox
        if box.height <= 0:
            return
        shown = self.comments_action.isChecked()
        x_lo, x_hi = self.ax_top.get_xlim()
        # Three linked axes report every change of the view, one after the
        # other, and the labels sit on the last to be updated. Do the work
        # once per state: the key holds the view of both.
        host = self.ax_top_trig if self.ax_top_trig is not None else self.ax_top
        key = (x_lo, x_hi, host.get_xlim(), tuple(box.bounds), shown)
        if key == self._comment_layout_key:
            return
        self._comment_layout_key = key

        in_view = [i for i, c in enumerate(self._comments_drawn) if x_lo <= c.time <= x_hi]
        self._comment_labels_withheld = shown and len(in_view) > self._COMMENT_LABELS_MAX
        readable = shown and not self._comment_labels_withheld
        wanted = set(in_view) if readable else set()
        for index, label in self._comment_labels.items():
            if index not in wanted:
                label.set_visible(False)  # out of view, hidden, or too many
        if not wanted:
            return

        renderer = self.canvas.get_renderer()
        base_px = box.y0 + self._COMMENT_LABEL_BASE * box.height
        gap = self._COMMENT_LABEL_GAP_PX
        placed = []  # (right edge, bottom, top) in pixels, in time order
        for index in in_view:
            label = self._comment_text_artist(index)
            label.set_visible(True)
            label.set_y(self._COMMENT_LABEL_BASE)
            extent = label.get_window_extent(renderer)
            # The labels in reach sideways, nearest first. All labels are
            # one line of the same font, so all are equally wide.
            blocked = []
            for right, bottom, top in reversed(placed):
                if extent.x1 - right >= extent.width + gap:
                    break
                blocked.append((bottom, top))
            bottom = base_px
            for other_bottom, other_top in sorted(blocked):
                if bottom + extent.height + gap <= other_bottom:
                    break  # fits underneath this one
                bottom = max(bottom, other_top + gap)
            if bottom + extent.height > box.y1:
                bottom = base_px  # no room left above: accept the overlap
            label.set_y((bottom - box.y0) / box.height)
            placed.append((extent.x1, bottom, bottom + extent.height))

    def _set_comments_visible(self, visible: bool) -> None:
        """Show or hide the comments (View menu)."""
        for artist in self._comment_artists:
            artist.set_visible(visible)
        # Which labels are shown is the layout's decision (view, count).
        self._layout_comment_labels()
        self.canvas.draw_idle()

    def _update_comments_action(self) -> None:
        """The menu entry is live only while there are comments to show."""
        if not hasattr(self, "comments_action"):
            return  # the banner is first refreshed before the menus exist
        self.comments_action.setEnabled(
            self._plot_mode == "selected" and bool(self._comment_artists)
        )

    def _refresh_top_legend(self) -> None:
        """Combine handles from ax_top and ax_top_trig into one legend."""
        if self.ax_top is None:
            return
        handles, labels = self.ax_top.get_legend_handles_labels()
        if self.ax_top_trig is not None:
            h2, l2 = self.ax_top_trig.get_legend_handles_labels()
            handles.extend(h2)
            labels.extend(l2)
        if handles:
            self.ax_top.legend(handles, labels, loc="upper right", fontsize="small")

    def _detect_triggers(self, trigger_data: np.ndarray, trigger_time: np.ndarray) -> None:
        """Detect trigger pulses and update UI."""
        if len(trigger_data) == 0:
            self._trigger_times = np.array([])
            self._trigger_count = 0
            self.total_triggers_label.setText("0")
            return

        # Find peaks: height at half-max, minimum distance between peaks
        max_val = np.max(np.abs(trigger_data))
        if max_val == 0:
            self._trigger_times = np.array([])
            self._trigger_count = 0
            self.total_triggers_label.setText("0")
            return

        height_threshold = 0.5 * max_val
        min_distance = max(10, len(trigger_data) // 2000)

        peaks, _ = find_peaks(trigger_data, height=height_threshold, distance=min_distance)

        self._trigger_times = trigger_time[peaks]
        self._trigger_count = len(peaks)

        # Update trigger UI
        self.total_triggers_label.setText(str(self._trigger_count))
        if self._trigger_count > 0:
            self.start_trigger_spin.setMaximum(self._trigger_count)
            self.num_triggers_spin.setMaximum(self._trigger_count)
            # Clamp values
            if self.start_trigger_spin.value() > self._trigger_count:
                self.start_trigger_spin.setValue(1)
            if self.num_triggers_spin.value() > self._trigger_count:
                self.num_triggers_spin.setValue(1)

        # Plot small markers on detected peaks (on trigger axis so y-scale matches)
        if self._trigger_count > 0 and self.ax_top is not None:
            trig_ax = self.ax_top_trig if self.ax_top_trig is not None else self.ax_top
            trig_ax.plot(
                trigger_time[peaks], trigger_data[peaks],
                "rv", markersize=4, label=f"triggers ({self._trigger_count})",
            )
            self._refresh_top_legend()

    def _detect_steps_from_data(self, trigger_data: np.ndarray, trigger_time: np.ndarray) -> None:
        """Detect step (bridge) triggers from raw data and update CT UI."""
        from labchart2cfd.processing.step_detection import detect_steps

        # Rely on the shared default (DEFAULT_STEP_MIN_DURATION_S) so step
        # numbers stay consistent across the GUI and both CLI commands.
        self._ct_steps = detect_steps(trigger_data, trigger_time)

        if not self._ct_steps:
            return

        # Update step spinbox range
        self.ct_step_spin.setMaximum(len(self._ct_steps))
        self.ct_step_spin.setValue(1)

        # Draw step spans on the plot
        self._clear_ct_step_markers()
        for step in self._ct_steps:
            span = self.ax_top.axvspan(
                step["start_time"], step["end_time"],
                alpha=0.1, color="orange",
            )
            self._ct_step_markers.append(span)
            ann = self.ax_top.annotate(
                f"Step {step['index']}",
                xy=((step["start_time"] + step["end_time"]) / 2, 1.0),
                xycoords=("data", "axes fraction"),
                fontsize=7, color="orange", ha="center", va="bottom",
            )
            self._ct_step_markers.append(ann)
            self._ct_step_labels.append((step, ann))
        self._layout_step_labels()

    def _on_canvas_resized(self, _event) -> None:
        """Labels are kept apart in pixels, which a resize changes."""
        if self._plot_mode != "selected":
            return
        self._layout_comment_labels()
        self._layout_step_labels()
        self._fit_title()

    def _fit_title(self) -> None:
        """Show the longest form of the plot title that fits the plot area.

        A title is neither wrapped nor shortened by the plot library, so in
        a narrow window it ran off both edges and its ends could not be read.
        """
        if self.ax_top is None or not self._title_forms:
            return
        title = self.ax_top.title
        room = self.fig.bbox.width - 8
        renderer = self.canvas.get_renderer()
        for form in self._title_forms:
            title.set_text(form)
            if title.get_window_extent(renderer).width <= room:
                break

    # Steps whose middles are closer than this on screen share one label.
    _STEP_LABEL_MIN_PX = 50.0

    def _layout_step_labels(self) -> None:
        """Name steps together while they are too close to label apart.

        Three scan steps within 8 s of a 2300 s recording had "Step 5",
        "Step 6" and "Step 7" printed on one spot. Such a run is labelled
        once, "Steps 5 to 7", between its steps, and each step gets its own
        label back when zooming in separates them.
        """
        labels = self._ct_step_labels
        if not labels or self.ax_top is None:
            return
        middles = [(s["start_time"] + s["end_time"]) / 2 for s, _ann in labels]
        pixels = [self.ax_top.transData.transform((m, 0.0))[0] for m in middles]
        runs = [[0]]
        for i in range(1, len(labels)):
            if pixels[i] - pixels[i - 1] < self._STEP_LABEL_MIN_PX:
                runs[-1].append(i)
            else:
                runs.append([i])
        for run in runs:
            first, last = run[0], run[-1]
            for i in run[1:]:
                labels[i][1].set_visible(False)
            ann = labels[first][1]
            if first == last:
                text = f"Step {labels[first][0]['index']}"
            else:
                text = (f"Steps {labels[first][0]['index']} to "
                        f"{labels[last][0]['index']}")
            where = ((middles[first] + middles[last]) / 2, 1.0)
            ann.set_text(text)
            ann.xy = where
            ann.set_position(where)
            ann.set_visible(True)

    def _on_click(self, event) -> None:
        """Handle mouse click on the canvas for data picker."""
        if self._plot_mode != "selected":
            return
        # Include ax_top_trig (twin y-axis) so clicks over the trigger curve also register.
        valid_axes = (self.ax_top, self.ax_bottom, self.ax_top_trig)
        if event.inaxes not in valid_axes or event.inaxes is None:
            return
        # The same answer the banner shows, so the two cannot disagree.
        mode = self._click_mode()
        if mode == "toolbar":  # zoom or pan owns the click
            return
        picking_part = mode in ("part_first", "part_second")

        if event.button == 1:  # Left click
            if picking_part:
                self._capture_part_click(event.xdata)
            elif mode == "cycle":
                self._add_cycle_label(event.xdata)
            elif mode in ("inhale_start", "exhale_end"):
                self._add_ct_landmark(event.xdata)
            elif mode == "parts_idle":
                # Full-manual mode: ignore unarmed clicks. Time markers would
                # overwrite Start/End, and the image grid is computed live from
                # those, so a stray click would silently redefine the grid and
                # invalidate parts already added.
                self.status_bar.showMessage(
                    "Reorder parts: press 'Pick on Plot' first to select images."
                )
            elif mode in ("ct_landmarks_idle", "ct_cycle_idle"):
                self.status_bar.showMessage(_BANNERS[mode][1])
            else:
                self._add_time_marker(event.xdata)
        elif event.button == 3:  # Right click
            if picking_part:
                self._cancel_part_capture()
            elif mode == "cycle":
                self._remove_last_cycle_label()
            else:
                self._clear_time_markers()
        self._refresh_banner()

    def _align_time_marker_labels(self) -> None:
        """Put each marker's time on the outer side of its line.

        Centred on their lines, the two times of a short window printed into
        each other ("21.000s24.300s"). The earlier one now ends at its line
        and the later one starts at its line, so they cannot meet however
        close the markers are. A single marker keeps its time centred.
        """
        markers = sorted(self._time_markers, key=lambda m: m[0].get_xdata()[0])
        sides = ["center"] if len(markers) == 1 else ["right", "left"]
        for (_line_top, ann_top, _line_bot, ann_bot), side in zip(markers, sides):
            for ann in (ann_top, ann_bot):
                text = ann.get_text().strip()
                ann.set_ha(side)
                # One space keeps the time off its own line.
                ann.set_text({"right": text + " ", "left": " " + text}.get(side, text))

    def _add_time_marker(self, x_time: float) -> None:
        """Add a vertical time marker on both top and bottom plots."""
        if self.ax_bottom is None or self.ax_top is None:
            return

        # A third click restarts the pair. It used to draw a line that changed
        # nothing, so the visible markers stopped matching Start/End.
        if len(self._time_markers) >= 2:
            self._clear_time_markers()

        # Draw on top subplot
        line_top = self.ax_top.axvline(x=x_time, color="green", linestyle="--", linewidth=1, alpha=0.8)
        ann_top = self.ax_top.annotate(
            f"{x_time:.3f}s",
            xy=(x_time, 1.0), xycoords=("data", "axes fraction"),
            fontsize=8, color="green", fontweight="bold",
            ha="center", va="bottom",
        )
        # Draw on bottom subplot
        line_bot = self.ax_bottom.axvline(x=x_time, color="green", linestyle="--", linewidth=1, alpha=0.8)
        ann_bot = self.ax_bottom.annotate(
            f"{x_time:.3f}s",
            xy=(x_time, 1.0), xycoords=("data", "axes fraction"),
            fontsize=8, color="green", fontweight="bold",
            ha="center", va="bottom",
        )
        self._time_markers.append((line_top, ann_top, line_bot, ann_bot))
        self._align_time_marker_labels()

        # Auto-populate Start/End Time from first two markers
        marker_count = len(self._time_markers)
        if marker_count == 1:
            self.start_spin.setValue(x_time)
        elif marker_count == 2:
            self.end_spin.setValue(x_time)
            # Ensure start < end
            s = self.start_spin.value()
            e = self.end_spin.value()
            if e < s:
                self.start_spin.setValue(e)
                self.end_spin.setValue(s)

        self.canvas.draw_idle()
        self.status_bar.showMessage(
            f"Marker {marker_count} at {x_time:.3f}s | "
            f"Start={self._get_start_var()}s, End={self._get_end_var()}s"
        )

    # Fraction of the window added at each x end, and of the data range at
    # each y end, so the signal never touches the frame.
    _ZOOM_X_MARGIN = 0.05
    _ZOOM_Y_MARGIN = 0.10

    def _channel_y_limits(self, row: int, col: int, x_lo: float, x_hi: float):
        """(lo, hi) y limits for one channel's data inside [x_lo, x_hi].

        Read from the source arrays rather than the drawn artists, because the
        axes also carry marker lines whose y data is in axes coordinates and
        would otherwise dominate the range.
        """
        if self.data is None or row < 1 or row > self.data.num_channels:
            return None
        if self.data.is_block_empty(row, col):
            return None
        t = self.data.get_time(row, col)
        y = self.data.get_data(row, col)
        # Time rises with the sample index, so the samples in [x_lo, x_hi]
        # are one slice. A mask over a 92 million sample channel would copy
        # the whole of it to read its range.
        first = int(np.searchsorted(t, x_lo, side="left"))
        stop = int(np.searchsorted(t, x_hi, side="right"))
        if stop <= first:
            return None
        lo = float(np.min(y[first:stop]))
        hi = float(np.max(y[first:stop]))
        if not (np.isfinite(lo) and np.isfinite(hi)):
            # A NaN or an infinite sample. The range of the others is what
            # can be shown; an axis cannot take a limit that is not a number.
            finite = y[first:stop][np.isfinite(y[first:stop])]
            if finite.size == 0:
                return None
            lo, hi = float(finite.min()), float(finite.max())
        pad = (hi - lo) * self._ZOOM_Y_MARGIN
        if pad == 0:  # flat signal: keep the view from collapsing to a line
            pad = abs(hi) * self._ZOOM_Y_MARGIN or 1.0
        return lo - pad, hi + pad

    def _zoom_to_window(self, start_t: Optional[float] = None,
                        end_t: Optional[float] = None) -> None:
        """Zoom x and y to the processing window, with margin at every edge.

        The flow scale is fitted to the flow actually inside the window, so a
        short step buried in a long recording fills the view instead of being
        flattened by the full-recording range. The trigger scale is not
        touched.
        """
        if self.data is None or self.ax_top is None or self._plot_mode != "selected":
            return
        if start_t is None:
            start_t = self.start_spin.value()
        if end_t is None:
            end_t = self.end_spin.value()
        if end_t <= start_t:
            return

        pad = (end_t - start_t) * self._ZOOM_X_MARGIN
        self.ax_top.set_xlim(start_t - pad, end_t + pad)  # shared x: ax_bottom follows

        col = self.col_spin.value()

        # y comes from the window itself, not the padded span. Signal just
        # outside the window is often much larger, and letting it set the
        # range would flatten the very detail the zoom is meant to reveal.
        flow_lim = self._channel_y_limits(self.flow_row_spin.value(), col, start_t, end_t)
        if flow_lim is not None:
            self.ax_top.set_ylim(*flow_lim)
            if self.ax_bottom is not None:
                self.ax_bottom.set_ylim(*flow_lim)

        # The trigger scale is left as it is. Fitted to the window it showed
        # the noise on top of a scan step (2 mV on a real recording) and hid
        # the step. It holds the block's whole range or what the user set.

        self.canvas.draw_idle()
        self.status_bar.showMessage(
            f"Zoomed to {start_t:.3f}s - {end_t:.3f}s "
            f"(+/-{pad * 1000:.0f} ms margin)"
        )

    def _clear_time_markers(self) -> None:
        """Remove all time markers from both plots."""
        for artists in self._time_markers:
            _remove_all(artists)
        self._time_markers.clear()
        self.canvas.draw_idle()
        self.status_bar.showMessage("Time markers cleared")

    def _mark_trigger_range(self) -> None:
        """Highlight selected trigger range on the top plot."""
        if self._trigger_count == 0:
            QMessageBox.warning(self, "No Triggers", "No triggers detected. Load a block with trigger data first.")
            return
        if self.ax_top is None:
            QMessageBox.warning(
                self, "No Plot",
                "Open a block first: click it on the overview.",
            )
            return

        # Clear previous trigger range markers
        _remove_all(self._trigger_markers)
        self._trigger_markers.clear()

        start_idx = self.start_trigger_spin.value()  # 1-indexed
        num_triggers = self.num_triggers_spin.value()

        # Validate range
        if start_idx < 1 or start_idx > self._trigger_count:
            QMessageBox.warning(self, "Invalid Start", f"Start trigger must be 1-{self._trigger_count}.")
            return
        end_idx = start_idx + num_triggers - 1
        if end_idx > self._trigger_count:
            QMessageBox.warning(
                self, "Out of Range",
                f"Only {self._trigger_count - start_idx + 1} triggers available from #{start_idx}.",
            )
            return

        # Get time range (convert to 0-indexed for array access)
        t_start = self._trigger_times[start_idx - 1]
        t_end = self._trigger_times[end_idx - 1]

        # Draw shaded region
        span = self.ax_top.axvspan(t_start, t_end, alpha=0.15, color="green", label="selected range")
        self._trigger_markers.append(span)

        # Draw vertical lines at each selected trigger
        for i in range(start_idx - 1, end_idx):
            t = self._trigger_times[i]
            vl = self.ax_top.axvline(x=t, color="green", linestyle="-", linewidth=0.8, alpha=0.6)
            self._trigger_markers.append(vl)

        # Also mark on bottom plot
        span_b = self.ax_bottom.axvspan(t_start, t_end, alpha=0.1, color="green")
        self._trigger_markers.append(span_b)

        self._refresh_top_legend()
        self.canvas.draw_idle()

        # Auto-populate Start/End Time
        self.start_spin.setValue(t_start)
        self.end_spin.setValue(t_end)

        self.status_bar.showMessage(
            f"Triggers: starting at #{start_idx}, counting {num_triggers} "
            f"(frames {start_idx}-{end_idx}) | "
            f"Time: {t_start:.3f}s - {t_end:.3f}s"
        )

    def _clear_all_markers(self) -> None:
        """Clear both time markers and trigger range markers."""
        self._clear_time_markers()
        _remove_all(self._trigger_markers)
        self._trigger_markers.clear()
        self._clear_ct_step_markers()
        self._clear_ct_interval_markers()
        self._clear_ct_landmarks()
        self._clear_cycle_labels()  # drop stale cycle frames + artists
        if self.ax_top is not None:
            self._refresh_top_legend()
        self.canvas.draw_idle()
        self.status_bar.showMessage("All markers cleared")

    def _detect_steps(self) -> None:
        """Detect step triggers in the trigger channel for CT workflow (button handler)."""
        if self.data is None:
            QMessageBox.warning(self, "No Data", "Open a recording first.")
            return
        if self.ax_top is None:
            QMessageBox.warning(
                self, "No Plot",
                "Open a block first: click it on the overview.",
            )
            return

        trigger_row = self.trigger_row_spin.value()
        col = self.col_spin.value()

        if trigger_row < 1 or self.data.is_block_empty(trigger_row, col):
            QMessageBox.warning(
                self, "No Trigger",
                "No trigger data. Set Trigger (section 1, Recording) to a "
                "channel that has data in this block.",
            )
            return

        trigger_data = self.data.get_data(trigger_row, col)
        trigger_time = self.data.get_time(trigger_row, col)

        self._detect_steps_from_data(trigger_data, trigger_time)

        if not self._ct_steps:
            QMessageBox.information(self, "No Steps", "No step triggers detected.")
            return

        self.canvas.draw_idle()
        self.status_bar.showMessage(f"Detected {len(self._ct_steps)} step trigger(s)")

    def _clear_ct_step_markers(self) -> None:
        """Remove step trigger visualization markers."""
        _remove_all(self._ct_step_markers)
        self._ct_step_markers.clear()
        self._ct_step_labels = []

    def _select_ct_step(self) -> None:
        """Select a specific step trigger, draw interval lines, prepare for landmarks."""
        if not self._ct_steps:
            QMessageBox.warning(self, "No Steps", "Please detect steps first.")
            return

        step_num = self.ct_step_spin.value()
        if step_num < 1 or step_num > len(self._ct_steps):
            QMessageBox.warning(self, "Invalid Step", f"Step must be 1-{len(self._ct_steps)}.")
            return

        self._ct_selected_step = self._ct_steps[step_num - 1]
        step = self._ct_selected_step

        # Store original detected boundaries for manual correction reference
        self._ct_detected_start = step["start_time"]
        self._ct_detected_end = step["end_time"]

        # Reset shift to 0 when selecting a new step (suppress callback to avoid
        # premature redraw with stale frame count before _draw_ct_intervals is called)
        self._suppress_shift_slider = True
        self.ct_shift_spin.setValue(0)
        self.ct_shift_slider.setValue(0)
        step_dur_ms = (step["end_time"] - step["start_time"]) * 1000.0
        slider_max = max(int(step_dur_ms * 0.5), 500)
        # Symmetric: only raising the maximum left the minimum pinned at 0, so
        # the shift could never move the window earlier.
        self.ct_shift_slider.setRange(-slider_max, slider_max)
        self.ct_shift_spin.setRange(-slider_max, slider_max)
        self._suppress_shift_slider = False

        # Auto-fill start/end time from step boundaries
        self.start_spin.setValue(step["start_time"])
        self.end_spin.setValue(step["end_time"])

        # Highlight selected step more prominently
        self._clear_ct_step_markers()
        for s in self._ct_steps:
            color = "lime" if s["index"] == step_num else "orange"
            alpha = 0.25 if s["index"] == step_num else 0.08
            span = self.ax_top.axvspan(s["start_time"], s["end_time"], alpha=alpha, color=color)
            self._ct_step_markers.append(span)

        # Also mark on bottom plot
        span_b = self.ax_bottom.axvspan(step["start_time"], step["end_time"], alpha=0.15, color="lime")
        self._ct_step_markers.append(span_b)

        # Draw temporal resolution interval lines within the step
        self._draw_ct_intervals()
        # The step's image count is known now, so the parts grid can start
        # from it (only while its own count is still unset).
        self._seed_synth_image_count()

        # Bring the step into view so a short scan inside a long recording is
        # actually readable.
        self._zoom_to_window(step["start_time"], step["end_time"])

        # Reset landmarks AND cycle frames for the new step (stale cycle frames
        # from a prior step must not leak into this step's export).
        self._clear_ct_landmarks()
        self._clear_cycle_labels()

        # Start and End are the step now. Markers clicked for an earlier
        # window would show times that no longer apply.
        self._clear_time_markers()

        self.canvas.draw_idle()
        self.status_bar.showMessage(
            f"Step {step_num} selected ({step['duration']:.2f}s, "
            f"{self.ct_total_images_spin.value()} images) | "
            f"{_CT_NEXT_STEP[self._ct_method()]}"
        )
        self._refresh_banner()

    def _claim_click_mode(self) -> None:
        """Make the interaction modes exclusive.

        Cycle labelling and part capture sit above landmarks in the click
        dispatch, so arming a landmark while one of them was active silently
        stole the click and left the landmark armed indefinitely.
        """
        if self._ct_cycle_mode and hasattr(self, "cycle_mode_btn"):
            self.cycle_mode_btn.setChecked(False)
        self._cancel_part_capture()

    def _mark_inhale_start(self) -> None:
        """Activate click mode to place inhale start landmark."""
        self._claim_click_mode()
        self._ct_landmark_mode = "inhale_start"
        self.status_bar.showMessage("Click on plot to mark INHALE START")
        self._refresh_banner()

    def _mark_exhale_end(self) -> None:
        """Activate click mode to place exhale end landmark."""
        self._claim_click_mode()
        self._ct_landmark_mode = "exhale_end"
        self.status_bar.showMessage("Click on plot to mark EXHALE END")
        self._refresh_banner()

    # Most images the step grid is drawn for; the range of the Total Images box.
    _MAX_GRID_IMAGES = 500

    def _draw_ct_intervals(self, update_total_images_field: bool = True) -> None:
        """Draw temporal resolution interval lines within the step window.

        Args:
            update_total_images_field: If True, overwrite the Total Images entry
                with the auto-computed value. Set to False when the user has
                manually edited the field and we want to preserve their value.
        """
        if self.ax_top is None or self.ax_bottom is None:
            return
        if self._ct_selected_step is None:
            return

        # Clear previous interval artists
        self._clear_ct_interval_markers()

        temporal_res_ms = self.ct_temporal_res_spin.value()
        temporal_res_s = temporal_res_ms / 1000.0

        if temporal_res_s <= 0:
            return

        # Check for manual override: shift + manual frame count
        shift_ms = self.ct_shift_spin.value()

        try:
            manual_frames = self.ct_total_images_spin.value()
        except (ValueError, TypeError):
            manual_frames = None

        # The manual override applies only to a shift/correction preview
        # (update_total_images_field=False). Selecting a step must recompute
        # from the step itself, otherwise the spinbox's leftover value (1 by
        # default) is taken as the image count and the auto path never runs.
        manual_override = manual_frames is not None and manual_frames > 0 and not update_total_images_field

        if (manual_override and self._ct_detected_start is not None and shift_ms != 0):
            # Manual correction mode: use shifted start + user-specified frame count
            start_t = self._ct_detected_start + shift_ms / 1000.0
            n_images = manual_frames
        elif manual_override and self._ct_detected_start is not None:
            # Frame count override only (no shift)
            start_t = self._ct_detected_start
            n_images = manual_frames
        else:
            # Auto mode: compute from step boundaries
            step = self._ct_selected_step
            start_t = step["start_time"]
            end_t = step["end_time"]
            n_images = compute_image_count(end_t - start_t, temporal_res_s)

        # A Temporal Res far too small for the step asks for a line per image
        # on two axes: 0.01 ms in a 3 s step is 300,000 images, and drawing
        # them froze the window. The Total Images box stops at this many too.
        if n_images > self._MAX_GRID_IMAGES:
            self.canvas.draw_idle()
            self.status_bar.showMessage(
                f"Temporal Res {temporal_res_ms:g} ms gives {n_images} images in "
                f"this step, too many to draw (limit {self._MAX_GRID_IMAGES}). "
                "Check Temporal Res."
            )
            return

        # N images are bounded by N+1 lines.
        for i in range(n_images + 1):
            t = start_t + i * temporal_res_s
            for ax in (self.ax_top, self.ax_bottom):
                line = ax.axvline(x=t, color="blue", linestyle=":", linewidth=0.7, alpha=0.6)
                self._ct_interval_markers.append(line)

        # Label each image centred in its slice, so the number names the span
        # between two lines rather than a line itself.
        for i in range(1, n_images + 1):
            if i > 50:  # limit labels for readability
                break
            t_mid = start_t + (i - 0.5) * temporal_res_s
            ann = self.ax_top.annotate(
                str(i),
                xy=(t_mid, 0.02), xycoords=("data", "axes fraction"),
                fontsize=5, color="blue", ha="center", va="bottom",
            )
            self._ct_interval_markers.append(ann)

        total_images = n_images
        if update_total_images_field:
            self.ct_total_images_spin.setValue(total_images)

        # Populate Trigger Info box only on step selection, not during slider preview
        if update_total_images_field:
            self.total_triggers_label.setText(str(total_images))
            self.start_trigger_spin.setRange(1, total_images)
            self.num_triggers_spin.setRange(1, total_images)
            self.start_trigger_spin.setValue(1)
            self.num_triggers_spin.setValue(total_images)

        self.canvas.draw_idle()

    def _on_shift_slider_changed(self, value: int) -> None:
        """Sync slider value to entry and redraw interval lines as live preview."""
        if self._suppress_shift_slider:
            return
        self.ct_shift_spin.setValue(value)
        self._draw_ct_intervals(update_total_images_field=False)

    def _on_shift_entry_changed(self) -> None:
        """Sync entry value to slider and redraw interval lines."""
        val = self.ct_shift_spin.value()
        # Clamp to the slider's own range, both ends: clamping the low end to 0
        # silently discarded any negative (earlier) correction.
        val = max(self.ct_shift_slider.minimum(),
                  min(val, self.ct_shift_slider.maximum()))
        self.ct_shift_spin.setValue(val)
        # Suppress slider callback to avoid double-redraw
        self._suppress_shift_slider = True
        self.ct_shift_slider.setValue(val)
        self._suppress_shift_slider = False
        self._draw_ct_intervals(update_total_images_field=False)

    def _apply_ct_correction(self) -> None:
        """Apply manual correction: update start/end times from shift + frame count."""
        if self._ct_detected_start is None:
            QMessageBox.warning(self, "No Step", "Please select a step first.")
            return

        shift_ms = self.ct_shift_spin.value()

        manual_frames = self.ct_total_images_spin.value()
        if manual_frames < 1:
            QMessageBox.warning(self, "Invalid Frames", "Total images must be at least 1.")
            return

        temporal_res_ms = self.ct_temporal_res_spin.value()
        temporal_res_s = temporal_res_ms / 1000.0

        # Compute corrected boundaries. N images span N*dt (they are slices),
        # so the window ends at the closing line of the last image.
        corrected_start = self._ct_detected_start + shift_ms / 1000.0
        corrected_end = corrected_start + manual_frames * temporal_res_s

        # The corrected window must still have data behind it. A negative shift
        # can push it before the recording starts, and a large image count can
        # push it past the end; both used to be written to the spinboxes and
        # only failed later, deep in processing.
        if self.data is not None:
            try:
                t_axis = self.data.get_time(self.flow_row_spin.value(),
                                            self.col_spin.value())
            except ValueError:
                t_axis = None
            if t_axis is not None:
                rec_lo, rec_hi = float(t_axis[0]), float(t_axis[-1])
                if corrected_start < rec_lo or corrected_end > rec_hi:
                    QMessageBox.warning(
                        self, "Correction Outside Recording",
                        f"That correction gives {corrected_start:.3f}s to "
                        f"{corrected_end:.3f}s, outside the recording "
                        f"({rec_lo:.3f}s to {rec_hi:.3f}s).\n\n"
                        "Reduce the shift or the image count.",
                    )
                    return

        # Update the processing fields
        self.start_spin.setValue(corrected_start)
        self.end_spin.setValue(corrected_end)

        # Update trigger info box with corrected frame count
        self.total_triggers_label.setText(str(manual_frames))
        self.start_trigger_spin.setRange(1, manual_frames)
        self.num_triggers_spin.setRange(1, manual_frames)
        self.start_trigger_spin.setValue(1)
        self.num_triggers_spin.setValue(manual_frames)

        # Redraw fencepost lines with corrected values
        self._draw_ct_intervals(update_total_images_field=False)

        self.status_bar.showMessage(
            f"Correction applied: start shifted +{shift_ms:.0f}ms, "
            f"{manual_frames} images, "
            f"window {corrected_start:.3f}s - {corrected_end:.3f}s"
        )

    def _clear_ct_interval_markers(self) -> None:
        """Remove CT interval visualization markers."""
        _remove_all(self._ct_interval_markers)
        self._ct_interval_markers.clear()

    def _clear_ct_landmarks(self) -> None:
        """Clear CT landmark markers and reset landmark mode."""
        _remove_all(self._ct_landmark_artists)
        self._ct_landmark_artists.clear()
        self._ct_landmark_mode = None
        self._ct_inhale_start = None
        self._ct_exhale_end = None
        self.ct_inhale_label.setText("--")
        self.ct_exhale_label.setText("--")
        self.canvas.draw_idle()
        self._refresh_banner()

    def _landmark_snap_text(self, x_time: float) -> str:
        """Describe where the cut will actually land, or '' if not knowable.

        The splice snaps to the start of the nearest image so the exported
        waveform and image order describe the same instant; showing both times
        keeps that adjustment visible rather than silent.
        """
        step = self._ct_selected_step
        dt_s = self.ct_temporal_res_spin.value() / 1000.0
        if step is None or dt_s <= 0:
            return ""
        start_t = step["start_time"]
        n_images = compute_image_count(step["end_time"] - start_t, dt_s)
        if n_images < 1:
            return ""
        idx = time_to_image_index(x_time, start_t, dt_s, n_images)
        snapped = image_index_to_time(idx, start_t, dt_s)
        return (
            f" | cut snaps to image {idx + 1} at {snapped:.3f}s "
            f"({abs(snapped - x_time) * 1000:.0f} ms away)"
        )

    def _add_ct_landmark(self, x_time: float) -> None:
        """Place a CT landmark (inhale start or exhale end) at the clicked time."""
        if self.ax_bottom is None or self.ax_top is None:
            return

        if self._ct_landmark_mode == "inhale_start":
            color = "magenta"
            label = "Inhale Start"
            self._ct_inhale_start = x_time
            self.ct_inhale_label.setText(f"{x_time:.3f} s")

            # Remove previous inhale markers if re-placing
            self._remove_ct_landmark_by_label("Inhale Start")

            # Draw landmark on both plots
            for ax in (self.ax_top, self.ax_bottom):
                line = ax.axvline(x=x_time, color=color, linestyle="-", linewidth=1.5, alpha=0.9, label=label)
                ann = ax.annotate(
                    label, xy=(x_time, 0.95), xycoords=("data", "axes fraction"),
                    fontsize=7, color=color, fontweight="bold", ha="center", va="top",
                )
                self._ct_landmark_artists.extend([line, ann])

            self._ct_landmark_mode = None  # Single-click mode done
            self.canvas.draw_idle()
            self.status_bar.showMessage(
                f"Inhale start marked at {x_time:.3f}s" + self._landmark_snap_text(x_time)
            )

        elif self._ct_landmark_mode == "exhale_end":
            color = "darkviolet"
            label = "Exhale End"
            self._ct_exhale_end = x_time
            self.ct_exhale_label.setText(f"{x_time:.3f} s")

            # Remove previous exhale markers if re-placing
            self._remove_ct_landmark_by_label("Exhale End")

            for ax in (self.ax_top, self.ax_bottom):
                line = ax.axvline(x=x_time, color=color, linestyle="-", linewidth=1.5, alpha=0.9, label=label)
                ann = ax.annotate(
                    label, xy=(x_time, 0.95), xycoords=("data", "axes fraction"),
                    fontsize=7, color=color, fontweight="bold", ha="center", va="top",
                )
                self._ct_landmark_artists.extend([line, ann])

            self._ct_landmark_mode = None  # Single-click mode done
            self.canvas.draw_idle()
            self.status_bar.showMessage(f"Exhale end marked at {x_time:.3f}s")

    def _remove_ct_landmark_by_label(self, label: str) -> None:
        """Remove existing landmark artists (lines and annotations) matching label."""
        remaining = []
        for artist in self._ct_landmark_artists:
            if hasattr(artist, "get_text") and artist.get_text() == label:
                _safe_remove(artist)  # Annotation
            elif hasattr(artist, "get_label") and artist.get_label() == label:
                _safe_remove(artist)  # Line2D with label= kwarg
            else:
                remaining.append(artist)
        self._ct_landmark_artists = remaining

    # ------------------------------------------------------------------
    # Cycle frame labeling (new mode, independent of inhale/exhale)
    # ------------------------------------------------------------------

    def _on_cycle_mode_toggled(self, checked: bool) -> None:
        """Toggle cycle-labeling click mode."""
        self._ct_cycle_mode = checked
        if checked:
            # Stop landmark placement AND part capture so clicks don't race.
            # An armed 'Pick on Plot' outranks cycle mode in the click
            # dispatch, so leaving it armed meant the status bar announced
            # cycle mode while clicks silently rewrote the From/To boxes.
            self._ct_landmark_mode = None
            self._cancel_part_capture()
            self.status_bar.showMessage(
                "Cycle labelling on: click the images in breathing order "
                "(right click removes the last label)."
            )
        else:
            self.status_bar.showMessage("Cycle label mode off.")
        self._refresh_banner()

    def _cycle_start_time_dt(self) -> Optional[tuple]:
        """Return (start_t, dt, total_imgs) matching what _draw_ct_intervals uses.

        Returns None if a step hasn't been selected yet.
        """
        if self._ct_selected_step is None:
            return None
        dt = self.ct_temporal_res_spin.value() / 1000.0
        if dt <= 0:
            return None
        total_imgs = self.ct_total_images_spin.value()
        shift_ms = self.ct_shift_spin.value()
        if self._ct_detected_start is not None and (shift_ms != 0 or total_imgs > 0):
            start_t = self._ct_detected_start + shift_ms / 1000.0
        else:
            start_t = self._ct_selected_step["start_time"]
        return start_t, dt, total_imgs

    def _snap_to_fencepost(self, x_time: float) -> Optional[int]:
        """Snap a click time to the nearest fencepost 1-based image index."""
        meta = self._cycle_start_time_dt()
        if meta is None:
            return None
        start_t, dt, total_imgs = meta
        img = int(round((x_time - start_t) / dt)) + 1
        return max(1, min(img, total_imgs))

    def _add_cycle_label(self, x_time: float) -> None:
        """Add a cycle label at the fencepost nearest to the click."""
        if self._ct_selected_step is None:
            QMessageBox.warning(self, "No Step", "Select a CT step before labeling frames.")
            return

        img_idx = self._snap_to_fencepost(x_time)
        if img_idx is None:
            return

        # Stack repeated labels vertically so multiple visits to the same
        # fencepost stay visible. 0 prior visits → y=0.55, then +0.14 each.
        prior_visits = self._ct_cycle_frames.count(img_idx)
        self._ct_cycle_frames.append(img_idx)
        new_frame_num = len(self._ct_cycle_frames)

        start_t, dt, _ = self._cycle_start_time_dt()
        fencepost_time = start_t + (img_idx - 1) * dt
        y_pos = min(0.55 + prior_visits * 0.14, 0.95)

        ann = self.ax_top.annotate(
            str(new_frame_num),
            xy=(fencepost_time, y_pos),
            xycoords=("data", "axes fraction"),
            fontsize=9, color="white", fontweight="bold",
            ha="center", va="center",
            bbox=dict(boxstyle="circle,pad=0.25", fc="green", ec="darkgreen", lw=1),
        )
        self._ct_cycle_artists.append(ann)

        self._cycle_labels_changed()
        self.canvas.draw_idle()
        # The image taken is the one that starts at the nearest grid line,
        # which is not always the number under the pointer. Say which.
        self.status_bar.showMessage(
            f"Label {new_frame_num} is image {img_idx}, the image that starts "
            "at the grid line nearest your click."
        )

    def _remove_last_cycle_label(self) -> None:
        """Undo the most recently added cycle label."""
        if not self._ct_cycle_frames:
            return
        removed = self._ct_cycle_frames.pop()
        if self._ct_cycle_artists:
            last_artist = self._ct_cycle_artists.pop()
            try:
                last_artist.remove()
            except (ValueError, AttributeError):
                pass
        self._cycle_labels_changed()
        self.canvas.draw_idle()
        self.status_bar.showMessage(f"Removed the cycle label of image {removed}.")

    def _clear_cycle_labels(self) -> None:
        """Remove every cycle label and its artists."""
        for artist in self._ct_cycle_artists:
            try:
                artist.remove()
            except (ValueError, AttributeError):
                pass
        self._ct_cycle_artists = []
        self._ct_cycle_frames = []
        self._cycle_labels_changed()
        self.canvas.draw_idle()
        self.status_bar.showMessage("Cycle labels cleared.")

    def _update_cycle_ui(self) -> None:
        """Refresh the count + sequence labels in the CT controls."""
        n = len(self._ct_cycle_frames)
        self.cycle_count_label.setText(f"{n} frame{'s' if n != 1 else ''}")
        if n == 0:
            self.cycle_seq_label.setText("seq: none")
        else:
            seq = ",".join(str(i) for i in self._ct_cycle_frames)
            if len(seq) > 80:
                seq = seq[:77] + "..."
            self.cycle_seq_label.setText(f"seq: {seq}")

    def _cycle_labels_changed(self) -> None:
        """After the user adds or removes a cycle label: the readout, then
        the line above Export, which counts these frames."""
        self._update_cycle_ui()
        self._update_export_hint()

    # ------------------------------------------------------------------
    # Full-manual synthetic-profile mode
    # ------------------------------------------------------------------

    _PART_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                    "#8c564b", "#e377c2", "#17becf"]

    def _on_synth_mode_toggled(self, checked: bool) -> None:
        """Enter/leave full-manual synthetic-profile mode."""
        self._synth_mode = checked
        if checked:
            # Avoid click conflicts with the other CT interaction modes.
            if self._ct_cycle_mode and hasattr(self, "cycle_mode_btn"):
                self.cycle_mode_btn.setChecked(False)
            self._ct_landmark_mode = None
            self._seed_synth_image_count()
            # Quiet: the mode now comes on with the method, often before any
            # block is open, and that is not an error worth a dialog.
            self._draw_synth_grid(quiet=True)
            self.status_bar.showMessage(
                "Reorder parts: select a step, set the image count, then "
                "add parts in breathing order."
            )
        else:
            self._cancel_part_capture()
        self._refresh_banner()

    def _synth_grid(self):
        """Return (origin, dt, n_images) for the manual grid, or None.

        Temporal Res and the image count are **independent inputs**: the grid
        is exactly ``n_images`` slices of ``dt`` starting at the anchor, and
        neither box rewrites the other. Start/End and the step position the
        anchor and give visual reference only, so the grid is free to extend
        past the step end.

        ``dt`` is quantised to whole flow samples, because the export slices in
        samples. Drawing at a fractional dt while slicing at a rounded one is
        what produced "needs samples up to 300 but the window only has 297".
        """
        dt_typed = self.synth_dt_spin.value() / 1000.0
        n_images = self.synth_images_spin.value()
        if dt_typed <= 0 or n_images < 1:
            return None
        spi = int(round(dt_typed * self._SYNTH_SAMPLE_RATE))
        if spi < 1:
            return None
        dt = spi / self._SYNTH_SAMPLE_RATE
        origin = self.start_spin.value() + self.synth_grid_offset_spin.value() / 1000.0
        return origin, dt, n_images

    def _synth_export_window(self):
        """Return (origin, grid_end) the export must process, or None.

        The exported range is defined by the grid, not by Start/End, so the
        sample count always matches what the parts ask for.
        """
        meta = self._synth_grid()
        if meta is None:
            return None
        origin, dt, n_images = meta
        return origin, origin + n_images * dt

    # Flow sample rate the export resamples to; the grid quantises dt to it.
    _SYNTH_SAMPLE_RATE = 100.0

    def _on_synth_dt_changed(self, dt_ms: float) -> None:
        """Temporal Res edited: mirror to tab 3, redraw. Image count untouched."""
        if self._syncing_grid:
            return
        self._syncing_grid = True
        try:
            # Both boxes are QDoubleSpinBox with the same range and precision,
            # so the mirror is exact; rounding to int used to truncate any
            # fractional dt and split the two grids apart.
            self.ct_temporal_res_spin.setValue(float(dt_ms))
        finally:
            self._syncing_grid = False
        self._refresh_synth_grid_if_active()
        self._update_grid_readout()

    def _on_ct_dt_changed(self, dt_ms: float) -> None:
        """Tab 3's Temporal Res edited: mirror it and redraw BOTH grids."""
        if self._syncing_grid:
            return
        self._syncing_grid = True
        try:
            self.synth_dt_spin.setValue(float(dt_ms))
        finally:
            self._syncing_grid = False
        self._redraw_ct_intervals_if_active()
        self._refresh_synth_grid_if_active()
        self._update_grid_readout()

    def _redraw_ct_intervals_if_active(self) -> None:
        """Redraw tab 3's fencepost grid after dt changes.

        Without this the blue numbered grid kept the OLD dt while cycle
        snapping and the export used the NEW one, so a click was labelled and
        exported as a different image than the number drawn over it.
        """
        if (self._ct_selected_step is not None
                and self.ax_top is not None
                and self._plot_mode == "selected"):
            self._draw_ct_intervals(update_total_images_field=True)
            self.canvas.draw_idle()

    def _on_synth_images_changed(self, n_images: int) -> None:
        """Image count edited: redraw. Temporal Res is left exactly as typed."""
        if self._syncing_grid or n_images < 1:
            return
        self._refresh_synth_grid_if_active()
        self._update_grid_readout()

    def _update_grid_readout(self) -> None:
        """Show the span the grid covers, which is the range that gets exported."""
        meta = self._synth_grid()
        if meta is None:
            self.synth_dt_warn_label.setText("")
            return
        origin, dt, n_images = meta
        end = origin + n_images * dt
        note = (
            f"grid spans {n_images * dt:.3f} s  ({origin:.3f} s to {end:.3f} s) "
            f"= the exported range"
        )
        typed = self.synth_dt_spin.value() / 1000.0
        if abs(typed - dt) > 1e-9:
            note += (
                f"   |  dt snapped {typed * 1000:.2f} -> {dt * 1000:.2f} ms "
                f"({int(round(dt * self._SYNTH_SAMPLE_RATE))} samples "
                f"at {self._SYNTH_SAMPLE_RATE:.0f} Hz)"
            )
        self.synth_dt_warn_label.setText(note)

    def _refresh_synth_grid_if_active(self) -> None:
        """Redraw the grid if full-manual mode is on and a plot is showing."""
        if (getattr(self, "_synth_mode", False)
                and self.ax_top is not None
                and self._plot_mode == "selected"):
            self._draw_synth_grid(quiet=True)

    def _apply_window_offset(self) -> None:
        """Slide Start and End together by the window offset."""
        off = self.synth_window_offset_spin.value() / 1000.0
        if off == 0:
            return
        new_start = self.start_spin.value() + off
        if new_start < 0:
            QMessageBox.warning(self, "Bad Offset", "That offset puts Start before 0 s.")
            return
        self.start_spin.setValue(new_start)
        self.end_spin.setValue(self.end_spin.value() + off)
        self.synth_window_offset_spin.setValue(0)
        self._refresh_synth_grid_if_active()
        self.status_bar.showMessage(f"Window shifted by {off * 1000:.0f} ms.")

    def _clear_synth_grid(self) -> None:
        for artist in self._grid_artists:
            try:
                artist.remove()
            except (ValueError, AttributeError):
                pass
        self._grid_artists = []

    def _draw_synth_grid(self, quiet: bool = False) -> None:
        """Draw the image grid for the manual window.

        Args:
            quiet: When True, return silently instead of raising a dialog if
                the window is not usable yet. The live redraw uses this,
                because Start/End are transiently invalid while being typed
                and a modal dialog there would freeze the app.
        """
        if self.ax_top is None or self._plot_mode != "selected":
            if not quiet:
                QMessageBox.warning(self, "No Plot", "Select a block to view first.")
            return
        meta = self._synth_grid()
        if meta is None:
            if not quiet:
                QMessageBox.warning(
                    self, "Bad Window", "Set Start < End and a positive Temporal Res."
                )
            else:
                # Drop a stale grid rather than leaving it contradicting the boxes.
                self._clear_synth_grid()
                self.synth_grid_label.setText("images: none")
                self.canvas.draw_idle()
            return
        origin, dt, n_images = meta
        self.synth_grid_label.setText(
            f"images: 1..{n_images}  (dt={dt*1000:.0f} ms)"
        )
        self._update_grid_readout()
        # Bound the From/To image entry boxes to the grid.
        self.part_from_spin.setRange(1, n_images)
        self.part_to_spin.setRange(1, n_images)

        self._clear_synth_grid()
        label_every = max(1, n_images // 40)  # avoid clutter for long windows
        # N+1 boundary lines boundaries...
        for j in range(1, n_images + 2):
            t = origin + (j - 1) * dt
            line = self.ax_top.axvline(x=t, color="0.6", linestyle=":", linewidth=0.5, alpha=0.7)
            self._grid_artists.append(line)
        # ...and the image number centred inside the slice it names, so
        # "image i" reads as the span between two lines, not as a line.
        for i in range(1, n_images + 1):
            if (i - 1) % label_every and i != n_images:
                continue
            t_mid = origin + (i - 0.5) * dt
            ann = self.ax_top.annotate(
                str(i), xy=(t_mid, 1.0), xycoords=("data", "axes fraction"),
                fontsize=6, color="0.4", ha="center", va="bottom",
            )
            self._grid_artists.append(ann)
        self._draw_parts_overlay()
        self.canvas.draw_idle()

    def _snap_synth(self, x_time: float):
        """Return the image (slice) the click landed inside, 1..N, or None."""
        meta = self._synth_grid()
        if meta is None:
            return None
        origin, dt, n_images = meta
        # Epsilon: (0.6 - 0.0)/0.2 is 2.9999... in binary floating point, which
        # would put a click exactly on a boundary in the previous slice.
        img = int((x_time - origin) / dt + 1e-9) + 1
        return max(1, min(img, n_images))

    def _add_part_from_spins(self) -> None:
        """Add a part using the From/To image number boxes (the explicit path)."""
        if self._synth_grid() is None:
            QMessageBox.warning(
                self, "Bad Window",
                "Set Start < End and a positive Temporal Res, then 'Show Image Grid'.",
            )
            return
        a = self.part_from_spin.value()
        b = self.part_to_spin.value()
        if a > b:
            QMessageBox.warning(
                self, "Bad Range",
                "'From image' must not be greater than 'To image'. "
                "Set them equal to select a single image.",
            )
            return
        self._parts.append(ProfilePart(start_image=a, end_image=b, label=""))
        self._refresh_parts_list()
        self._draw_parts_overlay()
        self.canvas.draw_idle()
        self.capture_hint.setText("")
        self.status_bar.showMessage(f"Added part: images {a}-{b} (part {len(self._parts)}).")

    def _arm_add_part(self) -> None:
        """Arm two-click capture that fills the From/To boxes from plot clicks."""
        if not self.synth_enable_chk.isChecked():
            QMessageBox.information(
                self, "Choose Method",
                "Choose the CT study type and the 'Reorder image parts' "
                "method first.",
            )
            return
        # Exclusive click mode: a pending landmark must not survive underneath.
        self._ct_landmark_mode = None
        if self._synth_grid() is None:
            QMessageBox.warning(
                self, "Bad Window",
                "Set Start < End and a positive Temporal Res, then 'Show Image Grid'.",
            )
            return
        self._capturing_part = True
        self._part_click_buffer = []
        self.capture_hint.setText("Click the first image of the range on the plot.")
        self.status_bar.showMessage("Click the first image of the range (right click cancels).")
        self._refresh_banner()

    def _cancel_part_capture(self) -> None:
        self._capturing_part = False
        self._part_click_buffer = []
        self._clear_capture_marker()
        if hasattr(self, "capture_hint"):
            self.capture_hint.setText("")
        if hasattr(self, "mode_banner"):
            self._refresh_banner()

    def _draw_capture_marker(self, img: int) -> None:
        self._clear_capture_marker()
        meta = self._synth_grid()
        if meta is None or self.ax_top is None:
            return
        origin, dt, _ = meta
        # Shade the whole image slice, since an image is a span not a line.
        self._capture_marker = self.ax_top.axvspan(
            origin + (img - 1) * dt, origin + img * dt,
            color="red", alpha=0.25,
        )
        self.canvas.draw_idle()

    def _clear_capture_marker(self) -> None:
        if self._capture_marker is not None:
            try:
                self._capture_marker.remove()
            except (ValueError, AttributeError):
                pass
            self._capture_marker = None

    def _capture_part_click(self, x_time: float) -> None:
        """Fill From (1st click) / To (2nd click) from snapped plot clicks."""
        img = self._snap_synth(x_time)
        if img is None:
            self._cancel_part_capture()
            return
        self._part_click_buffer.append(img)
        if len(self._part_click_buffer) == 1:
            self.part_from_spin.setValue(img)
            self._draw_capture_marker(img)
            self.capture_hint.setText(f"First image: {img}. Now click the last image.")
            self.status_bar.showMessage(f"First image: {img}. Now click the last image of the range.")
            return
        a, b = sorted(self._part_click_buffer)
        self._capturing_part = False
        self._part_click_buffer = []
        self._clear_capture_marker()
        # a == b is a legal single-image part, same as typing From = To.
        self.part_from_spin.setValue(a)
        self.part_to_spin.setValue(b)
        self.capture_hint.setText(f"Images {a} to {b} chosen. Press Add Part.")
        self.status_bar.showMessage(f"Images {a} to {b} chosen. Press Add Part to add them.")

    def _draw_parts_overlay(self) -> None:
        """Shade each part's span on the top plot with its order number."""
        if self.ax_top is None:
            return
        for artist in self._part_artists:
            try:
                artist.remove()
            except (ValueError, AttributeError):
                pass
        self._part_artists = []
        meta = self._synth_grid()
        if meta is None:
            return
        origin, dt, _ = meta
        for idx, part in enumerate(self._parts):
            color = self._PART_COLORS[idx % len(self._PART_COLORS)]
            # Images a..b span from line a to line b+1.
            t_a = origin + (part.start_image - 1) * dt
            t_b = origin + part.end_image * dt
            span = self.ax_top.axvspan(t_a, t_b, color=color, alpha=0.18)
            self._part_artists.append(span)
            ann = self.ax_top.annotate(
                str(idx + 1), xy=((t_a + t_b) / 2, 0.92),
                xycoords=("data", "axes fraction"),
                fontsize=9, color="white", fontweight="bold", ha="center", va="center",
                bbox=dict(boxstyle="circle,pad=0.2", fc=color, ec="black", lw=0.5),
            )
            self._part_artists.append(ann)

    def _refresh_parts_list(self) -> None:
        self.parts_list.clear()
        meta = self._synth_grid()
        dt = meta[1] if meta else 0.2
        for idx, part in enumerate(self._parts):
            dur = (part.end_image - part.start_image + 1) * dt
            lbl = f" [{part.label}]" if part.label else ""
            QListWidgetItem(
                f"{idx + 1}.{lbl} images {part.start_image}-{part.end_image}  ({dur:.2f}s)",
                self.parts_list,
            )
        n = len(self._parts)
        self.synth_status_label.setText("no parts" if n == 0 else f"{n} part{'s' if n != 1 else ''}")
        self._update_export_hint()

    def _selected_part_row(self) -> int:
        return self.parts_list.currentRow()

    def _move_part_up(self) -> None:
        i = self._selected_part_row()
        if i > 0:
            self._parts[i - 1], self._parts[i] = self._parts[i], self._parts[i - 1]
            self._refresh_parts_list()
            self.parts_list.setCurrentRow(i - 1)
            self._draw_parts_overlay()
            self.canvas.draw_idle()

    def _move_part_down(self) -> None:
        i = self._selected_part_row()
        if 0 <= i < len(self._parts) - 1:
            self._parts[i + 1], self._parts[i] = self._parts[i], self._parts[i + 1]
            self._refresh_parts_list()
            self.parts_list.setCurrentRow(i + 1)
            self._draw_parts_overlay()
            self.canvas.draw_idle()

    def _edit_part_label(self) -> None:
        i = self._selected_part_row()
        if not (0 <= i < len(self._parts)):
            return
        text, ok = QInputDialog.getText(self, "Edit Label", "Part label:", text=self._parts[i].label)
        if ok:
            self._parts[i].label = text.strip()
            self._refresh_parts_list()

    def _delete_part(self) -> None:
        i = self._selected_part_row()
        if 0 <= i < len(self._parts):
            self._parts.pop(i)
            self._refresh_parts_list()
            self._draw_parts_overlay()
            self.canvas.draw_idle()

    def _clear_parts(self) -> None:
        """Clear only the parts list (the plot keeps its grid and markers)."""
        self._parts = []
        self._refresh_parts_list()
        self._draw_parts_overlay()
        self.canvas.draw_idle()

    def _clear_synth_all(self) -> None:
        """Clear everything this tab draws: parts, grid, markers, capture pin."""
        self._cancel_part_capture()  # also clears the red capture marker
        self._parts = []
        self._refresh_parts_list()
        _remove_all(self._part_artists)
        self._part_artists = []
        self._clear_synth_grid()
        self._clear_time_markers()
        self.synth_grid_label.setText("images: none")
        self.canvas.draw_idle()
        self.status_bar.showMessage("Cleared parts, image grid, and plot markers.")

    def _assemble_and_export(self) -> None:
        """Process the window, assemble parts, preview, and export 3 CSVs."""
        if self.data is None:
            QMessageBox.warning(self, "No Data", "Open a recording first.")
            return
        if not self._parts:
            QMessageBox.warning(self, "No Parts", "Add at least one part.")
            return
        subject = self.subject_edit.text().strip()
        if not subject:
            QMessageBox.warning(self, "Missing Subject", "Enter a Subject ID.")
            return
        meta = self._synth_grid()
        if meta is None:
            QMessageBox.warning(
                self, "Bad Grid", "Set a positive Temporal Res and at least one image."
            )
            return
        start_t, dt, n_images = meta
        grid_end = start_t + n_images * dt
        flow_row = self.flow_row_spin.value()
        col = self.col_spin.value()
        # Checked here, because reading an empty block raises, and an error
        # inside a button handler used to end the program.
        if self.data.is_block_empty(flow_row, col):
            QMessageBox.critical(
                self, "Empty Block",
                f"Block {col} is empty on the flow channel "
                f"({self._channel_text(flow_row)}). Pick a block that has "
                "flow data.",
            )
            return

        # The grid defines the exported range, so it must sit inside the
        # recording. Refuse rather than invent or silently truncate data.
        t_axis = self.data.get_time(flow_row, col)
        if start_t < float(t_axis[0]) or grid_end > float(t_axis[-1]):
            short_s = max(0.0, grid_end - float(t_axis[-1]))
            QMessageBox.critical(
                self, "Grid Outside Recording",
                f"The grid needs {n_images} x {dt * 1000:.0f} ms = "
                f"{n_images * dt:.3f} s starting at {start_t:.3f} s, "
                f"so it runs to {grid_end:.3f} s.\n\n"
                f"The recording covers {float(t_axis[0]):.3f} s to "
                f"{float(t_axis[-1]):.3f} s"
                + (f", short by {short_s:.3f} s "
                   f"({int(np.ceil(short_s / dt))} image(s))." if short_s > 0
                   else ".")
                + "\n\nReduce the image count, shorten Temporal Res, "
                  "or move the anchor with Start / Grid offset.",
            )
            return
        pressure_row = self.pressure_row_spin.value()
        include_pressure = pressure_row > 0
        output_dir = Path(self.outdir_edit.text())

        try:
            from labchart2cfd.io.csv_export import (
                export_flow_csv, export_pressure_csv, export_reorder_map_csv,
            )
            from labchart2cfd.workflows.factory import build_workflow

            wf = build_workflow(
                "ct", density=1.2, target_sample_rate=100.0,
                pressure_row=pressure_row if include_pressure else 1,
            )
            # Process exactly the grid's range, so the sample count always
            # matches what the parts need. Start/End are reference only.
            result = wf.process(
                self.data, flow_row, col, start_t, grid_end,
                inhale_start_time=None, exhale_end_time=None,
                temporal_resolution=dt, include_pressure=include_pressure,
            )
            # Validate against the grid's own slice count, not the workflow's
            # fencepost count: a part must name an image that has a slice.
            total_images = n_images
            asm = assemble_parts(
                result.mass_flow, self._parts, dt=dt,
                sample_rate=result.sample_rate, total_images=total_images,
                pressure=result.pressure,
                grid_offset_s=0.0,  # the processed array already starts at the anchor
                window_start_s=start_t,  # so the map reads in recording time
            )
        except Exception as exc:  # surface engine/workflow errors clearly
            QMessageBox.critical(self, "Assembly Failed", str(exc))
            return

        # Preview. Nothing is written unless Export is pressed there; the
        # preview used to have only Close and the export ran regardless.
        preview = AssemblyPreviewDialog(
            asm.time, asm.flow, len(asm.reorder_map), parent=self
        )
        if preview.exec_() != QDialog.Accepted:
            self.status_bar.showMessage("Export cancelled. No file was written.")
            return

        # Export
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            flow_file = output_dir / f"{subject}FlowProfile.csv"
            export_flow_csv(flow_file, asm.time, asm.flow)
            written = [flow_file]
            if asm.pressure is not None:
                pfile = output_dir / f"{subject}PressureProfile.csv"
                export_pressure_csv(pfile, asm.time, asm.pressure)
                written.append(pfile)
            map_file = output_dir / f"{subject}ImageReorderMap.csv"
            export_reorder_map_csv(map_file, asm.reorder_map)
            written.append(map_file)
        except Exception as exc:
            QMessageBox.critical(self, "Export Failed", str(exc))
            return

        order = ",".join(str(m[1]) for m in asm.reorder_map)
        if len(order) > 120:
            order = order[:117] + "..."
        warnings = []
        if asm.pressure is None:
            warnings = self._pressure_warnings(
                include_pressure, result.metadata.get("pressure_note"),
                output_dir / f"{subject}PressureProfile.csv",
            )
        self._show_export_result({
            "folder": output_dir,
            "files": written,
            "summary": [
                ("Samples", str(len(asm.flow))),
                ("Images", str(len(asm.reorder_map))),
                ("Image order (original numbers)", order),
                ("Time", "starts at 0 in the profile; the reorder map keeps "
                         "recording time"),
            ],
            "warnings": warnings,
            "plot": None,
        })
        self.status_bar.showMessage(f"Exported synthetic profile + reorder map for {subject}.")

    def _execute_processing(self) -> None:
        """Run the processing pipeline in a background thread."""
        # Validate inputs
        if self.data is None:
            QMessageBox.warning(self, "No Data", "Open a recording first.")
            return

        subject = self.subject_edit.text().strip()
        if not subject:
            QMessageBox.warning(self, "Missing Subject", "Please enter a Subject ID.")
            return

        start_time = self.start_spin.value()
        end_time = self.end_spin.value()

        if end_time <= start_time:
            QMessageBox.critical(self, "Invalid Times", "End time must be greater than start time.")
            return

        flow_row = self.flow_row_spin.value()
        pressure_row = self.pressure_row_spin.value()  # 0 = skip pressure
        col = self.col_spin.value()
        workflow_name = self._workflow_name()
        output_dir = Path(self.outdir_edit.text())
        include_pressure = pressure_row > 0

        # Validate against the data BEFORE spawning the worker, so a bad
        # window or block fails with a dialog instead of dying in the thread.
        if self.data.is_block_empty(flow_row, col):
            QMessageBox.critical(
                self, "Empty Block",
                f"Flow block [{flow_row}, {col}] is empty.",
            )
            return
        t_axis = self.data.get_time(flow_row, col)
        rec_lo, rec_hi = float(t_axis[0]), float(t_axis[-1])
        # The window must lie INSIDE the recording. Merely intersecting is not
        # enough: an overrun silently exported a truncated profile while the
        # trigger info still described the full requested window. The
        # synthetic-profile export already refused the identical overrun.
        if start_time < rec_lo or end_time > rec_hi:
            short = max(0.0, end_time - rec_hi) + max(0.0, rec_lo - start_time)
            QMessageBox.critical(
                self, "Window Outside Recording",
                f"The window {start_time:.3f}s to {end_time:.3f}s is not fully "
                f"inside the recording, which covers {rec_lo:.3f}s to "
                f"{rec_hi:.3f}s (short by {short:.3f}s).\n\n"
                "Adjust Start/End so the whole window has data behind it.",
            )
            return

        # Only the CT method on screen feeds the export. Cycle frames are
        # original image indices in click order (empty for the other methods).
        ct_cycle_frames, ct_inhale_start, ct_exhale_end = self._ct_export_inputs()

        # Cycle labels were snapped against the drawn fencepost grid, which is
        # anchored at detected_start + shift. The export windows at Start Time.
        # If those disagree (shift previewed but Apply Correction not clicked,
        # or Start typed by hand), every chunk would be taken from the wrong
        # place, so refuse rather than export silently shifted data.
        if ct_cycle_frames and self._ct_detected_start is not None:
            grid_anchor = (
                self._ct_detected_start + self.ct_shift_spin.value() / 1000.0
            )
            dt_s = self.ct_temporal_res_spin.value() / 1000.0
            if abs(start_time - grid_anchor) > dt_s / 2:
                QMessageBox.critical(
                    self, "Grid Out of Sync",
                    f"The cycle labels were placed on a grid anchored at "
                    f"{grid_anchor:.3f}s, but Start Time is {start_time:.3f}s. "
                    "Click 'Apply Correction' (or reset the Start Shift) so "
                    "the export uses the same grid you labelled.",
                )
                return
        # Standard/CPAP/CT read pressure from self.pressure_row on the workflow;
        # pick a safe default when user set "none" so the attribute exists.
        effective_pressure_row = pressure_row if include_pressure else 1

        # Capture trigger info from main thread for the background thread
        trigger_start = self.start_trigger_spin.value()
        trigger_num = self.num_triggers_spin.value()
        trigger_total = self._trigger_count

        # Capture CT-specific state
        ct_temporal_res_ms = float(self.ct_temporal_res_spin.value())
        # Manual "Total Images" the user set in the GUI (may be smaller than
        # the workflow's auto-computed fencepost count for the raw step window).
        ct_total_images_manual = int(self.ct_total_images_spin.value())

        # Snapshot the data reference: a Load Overview during processing must
        # not swap the recording out from under the worker mid-run.
        data = self.data

        # Disable button during processing
        self.execute_btn.setEnabled(False)
        self.status_bar.showMessage("Processing...")
        QApplication.processEvents()

        def _run() -> None:
            try:
                from labchart2cfd.io.csv_export import export_flow_csv, export_pressure_csv
                from labchart2cfd.workflows.factory import build_workflow

                # Select workflow
                if workflow_name == "CT":
                    wf = build_workflow("ct", density=1.2, pressure_row=effective_pressure_row)
                    # Cycle mode: skip rearrangement so result.time fencepost
                    # positions (i-1)*dt map to original image index i.
                    if ct_cycle_frames:
                        result = wf.process(
                            data, flow_row, col, start_time, end_time,
                            inhale_start_time=None,
                            exhale_end_time=None,
                            temporal_resolution=ct_temporal_res_ms / 1000.0,
                            include_pressure=include_pressure,
                        )
                    else:
                        result = wf.process(
                            data, flow_row, col, start_time, end_time,
                            inhale_start_time=ct_inhale_start,
                            exhale_end_time=ct_exhale_end,
                            temporal_resolution=ct_temporal_res_ms / 1000.0,
                            include_pressure=include_pressure,
                        )

                    total_imgs = result.metadata.get("total_images", 0)
                    dt_s = ct_temporal_res_ms / 1000.0

                    if ct_cycle_frames:
                        # Split into ascending sub-sequences. Images a..b are
                        # the slices from line a to line b+1, so every
                        # sub-sequence contributes m*dt and the total is N*dt.
                        # Slices are disjoint, so no seam is double-counted and
                        # the last sub-sequence needs no special case.
                        subseqs = []
                        current = [ct_cycle_frames[0]]
                        for i in range(1, len(ct_cycle_frames)):
                            if ct_cycle_frames[i] == ct_cycle_frames[i - 1] + 1:
                                current.append(ct_cycle_frames[i])
                            else:
                                subseqs.append(current)
                                current = [ct_cycle_frames[i]]
                        subseqs.append(current)

                        chunks_flow = []
                        chunks_pressure = []
                        for subseq in subseqs:
                            a, b = subseq[0], subseq[-1]
                            t_lo = (a - 1) * dt_s
                            t_hi = b * dt_s
                            mask = (result.time >= t_lo) & (result.time < t_hi)
                            chunks_flow.append(result.mass_flow[mask])
                            if result.pressure is not None:
                                chunks_pressure.append(result.pressure[mask])

                        if not any(len(c) for c in chunks_flow):
                            raise RuntimeError(
                                "Cycle labels produced an empty signal. Label at least "
                                "one pair of consecutive frames within each sub-sequence."
                            )
                        concat_flow = np.concatenate(chunks_flow)
                        sample_rate = result.sample_rate
                        result.time = np.arange(len(concat_flow)) / sample_rate
                        result.mass_flow = concat_flow
                        if result.pressure is not None:
                            result.pressure = np.concatenate(chunks_pressure)

                        inhale_img = ct_cycle_frames[0]
                        exhale_img = ct_cycle_frames[-1]
                        selected_indices = list(ct_cycle_frames)
                        result.metadata["cycle_mode"] = True
                        result.metadata["cycle_original_indices"] = list(ct_cycle_frames)
                        result.metadata["cycle_frame_count"] = len(ct_cycle_frames)
                        # The workflow's auto-computed total_images reflects the
                        # full step window, which may exceed what the user set
                        # manually (and labeled against). Override so the output
                        # matches the fencepost grid the user actually saw.
                        result.metadata["total_images"] = ct_total_images_manual
                        result.metadata["rearranged_image_indices"] = list(
                            range(ct_total_images_manual)
                        )
                        total_imgs = ct_total_images_manual
                    elif ct_inhale_start is not None and ct_exhale_end is not None:
                        # Clamp to the images that exist, so a landmark near
                        # the window edge cannot name a non-existent image.
                        inhale_img = time_to_image_index(
                            ct_inhale_start, start_time, dt_s, total_imgs
                        ) + 1  # 1-based
                        exhale_img = time_to_image_index(
                            ct_exhale_end, start_time, dt_s, total_imgs
                        ) + 1  # 1-based

                        if inhale_img <= exhale_img:
                            selected_indices = list(range(inhale_img, exhale_img + 1))
                        else:
                            selected_indices = (
                                list(range(inhale_img, total_imgs + 1))
                                + list(range(1, exhale_img + 1))
                            )
                    else:
                        inhale_img = 1
                        exhale_img = total_imgs
                        selected_indices = list(range(1, total_imgs + 1))

                    result.metadata["inhale_image"] = inhale_img
                    result.metadata["exhale_image"] = exhale_img
                    result.metadata["selected_num_images"] = len(selected_indices)
                    result.metadata["selected_image_indices"] = selected_indices
                elif workflow_name == "phase-contrast":
                    wf = build_workflow("phase-contrast", density=5.761)
                    result = wf.process(data, flow_row, col, start_time, end_time)
                elif workflow_name == "cpap":
                    wf = build_workflow("cpap", density=1.2, pressure_row=effective_pressure_row)
                    result = wf.process(
                        data, flow_row, col, start_time, end_time,
                        include_pressure=include_pressure,
                    )
                else:
                    wf = build_workflow("standard", density=1.2, pressure_row=effective_pressure_row)
                    result = wf.process(
                        data, flow_row, col, start_time, end_time,
                        include_pressure=include_pressure,
                    )

                output_dir.mkdir(parents=True, exist_ok=True)

                # Export flow CSV
                flow_file = output_dir / f"{subject}FlowProfile.csv"
                export_flow_csv(flow_file, result.time, result.mass_flow)

                # Export pressure CSV if available
                pressure_file = None
                if result.pressure is not None:
                    pressure_file = output_dir / f"{subject}PressureProfile.csv"
                    export_pressure_csv(pressure_file, result.time, result.pressure)

                # Save trigger info for 4D image registration
                trigger_file = output_dir / f"{subject}_trigger_info.txt"
                exported_duration = result.time[-1] if len(result.time) > 0 else 0.0

                with open(str(trigger_file), "w") as tf:
                    if workflow_name == "CT":
                        meta = result.metadata
                        tf.write(f"workflow: CT\n")
                        tf.write(f"step_start_time: {meta.get('step_start_time', 0):.3f}\n")
                        tf.write(f"step_end_time: {meta.get('step_end_time', 0):.3f}\n")
                        tf.write(f"step_duration: {meta.get('step_duration', 0):.3f}\n")
                        inhale_t = meta.get("inhale_start_time")
                        exhale_t = meta.get("exhale_end_time")
                        tf.write(f"inhale_start_time: {inhale_t:.3f}\n" if inhale_t is not None else "inhale_start_time: N/A\n")
                        tf.write(f"exhale_end_time: {exhale_t:.3f}\n" if exhale_t is not None else "exhale_end_time: N/A\n")
                        tf.write(f"temporal_resolution_s: {meta.get('temporal_resolution_s', 0.2):.4f}\n")
                        tf.write(f"total_images: {meta.get('total_images', 0)}\n")
                        tf.write(f"cut_image_index: {meta.get('cut_image_index', 0) + 1}\n")  # 1-based to match GUI labels
                        indices = meta.get("rearranged_image_indices", [])
                        tf.write(f"rearranged_image_indices: {','.join(str(i + 1) for i in indices)}\n")  # 1-based
                        tf.write(f"inhale_image: {meta.get('inhale_image', 1)}\n")
                        tf.write(f"exhale_image: {meta.get('exhale_image', 0)}\n")
                        tf.write(f"selected_num_images: {meta.get('selected_num_images', 0)}\n")
                        sel_indices = meta.get("selected_image_indices", [])
                        tf.write(f"selected_image_indices: {','.join(str(i) for i in sel_indices)}\n")
                        if meta.get("cycle_mode"):
                            cycle_orig = meta.get("cycle_original_indices", [])
                            tf.write(f"cycle_mode: true\n")
                            tf.write(f"cycle_frame_count: {meta.get('cycle_frame_count', 0)}\n")
                            tf.write(f"cycle_original_indices: {','.join(str(i) for i in cycle_orig)}\n")
                            mapping = ";".join(
                                f"{new_i+1}->{orig}" for new_i, orig in enumerate(cycle_orig)
                            )
                            tf.write(f"cycle_new_to_original: {mapping}\n")
                        else:
                            tf.write(f"cycle_mode: false\n")
                        tf.write(f"start_time_exported: 0.000\n")
                        tf.write(f"end_time_exported: {exported_duration:.3f}\n")
                    else:
                        end_trigger = trigger_start + trigger_num - 1
                        tf.write(f"start_trigger: {trigger_start}\n")
                        tf.write(f"num_triggers: {trigger_num}\n")
                        tf.write(f"end_trigger: {end_trigger}\n")
                        tf.write(f"total_triggers_in_block: {trigger_total}\n")
                        tf.write(f"start_time_original: {start_time:.3f}\n")
                        tf.write(f"end_time_original: {end_time:.3f}\n")
                        tf.write(f"start_time_exported: 0.000\n")
                        tf.write(f"end_time_exported: {exported_duration:.3f}\n")

                # Generate sanity-check PNG plot of exported data
                # Use Figure + FigureCanvasAgg directly to avoid backend switching
                from matplotlib.figure import Figure as MplFigure
                from matplotlib.backends.backend_agg import FigureCanvasAgg

                plot_file = output_dir / f"{subject}_sanity_check.png"
                n_plots = 2 if result.pressure is not None else 1
                fig_export = MplFigure(figsize=(10, 3 * n_plots), dpi=150)
                FigureCanvasAgg(fig_export)
                axes = fig_export.subplots(n_plots, 1, squeeze=False)[:, 0]

                axes[0].plot(result.time, result.mass_flow, "b-", linewidth=0.5)
                axes[0].axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
                axes[0].set_xlabel("Time (s)")
                axes[0].set_ylabel("Mass Flow Rate (kg/s)")
                axes[0].set_title(f"{subject} — Exported Flow Profile (zeroed time)")
                axes[0].grid(True, alpha=0.3)

                if result.pressure is not None:
                    axes[1].plot(result.time, result.pressure, "r-", linewidth=0.5)
                    axes[1].axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
                    axes[1].set_xlabel("Time (s)")
                    axes[1].set_ylabel("Pressure (Pa)")
                    axes[1].set_title(f"{subject} — Exported Pressure Profile (zeroed time)")
                    axes[1].grid(True, alpha=0.3)

                fig_export.tight_layout()
                fig_export.savefig(str(plot_file))

                # Update status on main thread
                msg = f"Done! Saved: {flow_file}"
                if pressure_file:
                    msg += f" and {pressure_file}"
                msg += f" and {trigger_file.name} and {plot_file.name}"
                msg += (
                    f" | Duration: {result.time[-1]:.2f}s"
                    f" | Rate: {result.sample_rate:.0f}Hz"
                    f" | Samples: {len(result.time)}"
                    f" | Drift: {result.drift_error:.6f}"
                    f" | Time zeroed for STAR-CCM+"
                )
                # A silently absent pressure profile looks the same as one that
                # was never requested; say why it is missing.
                pressure_note = result.metadata.get("pressure_note")
                if include_pressure and result.pressure is None and pressure_note:
                    msg += f" | WARNING: {pressure_note}"

                # The same outcome as a structured report, for the result
                # panel. Pressure asked for but absent is always a warning
                # there, with the specific reason when the workflow gave one.
                warnings = []
                if result.pressure is None:
                    warnings = self._pressure_warnings(
                        include_pressure, pressure_note,
                        output_dir / f"{subject}PressureProfile.csv",
                    )
                files = [flow_file]
                if pressure_file:
                    files.append(pressure_file)
                files += [trigger_file, plot_file]
                self._processing_report.emit({
                    "folder": output_dir,
                    "files": files,
                    "summary": [
                        ("Duration", f"{result.time[-1]:.2f} s"),
                        ("Sample rate", f"{result.sample_rate:.0f} Hz"),
                        ("Samples", str(len(result.time))),
                        ("Drift error", f"{result.drift_error:.6f}"),
                        ("Time", "starts at 0, as STAR-CCM+ expects"),
                    ],
                    "warnings": warnings,
                    "plot": plot_file,
                })

                self._processing_done.emit(msg)

            except Exception as e:
                import traceback
                traceback.print_exc()  # so the real error shows in the terminal
                self._processing_failed.emit(f"{type(e).__name__}: {e}")

        Thread(target=_run, daemon=True).start()


def report_unhandled_exception(exc_type, exc, tb) -> None:
    """Show an unexpected error instead of letting it end the program.

    PyQt aborts the whole process when an error escapes a button handler,
    which lost the session over something as small as an empty block. The
    traceback still goes to the terminal for whoever debugs it.
    """
    traceback.print_exception(exc_type, exc, tb)
    if QApplication.instance() is not None:
        QMessageBox.critical(
            None, "Unexpected Error",
            f"{exc_type.__name__}: {exc}\n\n"
            "The window stays open. If this happened during an export, check "
            "the output folder before using its files.",
        )


def install_exception_hook() -> None:
    """Route errors that nothing caught to ``report_unhandled_exception``."""
    sys.excepthook = report_unhandled_exception


def launch_gui() -> None:
    """Launch the Flow Profile GUI application."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    install_exception_hook()
    apply_theme(app)
    window = FlowProfileApp()
    window.show()
    app.exec()
