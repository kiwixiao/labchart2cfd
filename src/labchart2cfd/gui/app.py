"""PyQt5 GUI for interactive flow profile selection and processing.

Provides a desktop application that:
1. Loads LabChart .mat files via file browser
2. Displays an overview grid of all channels x blocks
3. Lets user select a specific block (row/col) for detailed view
4. Linked zoom between top and bottom subplots
5. Detects and counts trigger pulses with visual markers
6. Click-to-place time markers on flow curve for breath window selection
7. Trigger range selection for 4D image registration (start frame + count)
8. Runs the processing pipeline and exports CSV files
"""

import sys
from pathlib import Path
from threading import Thread
from typing import Optional, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from scipy.signal import find_peaks

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QPushButton, QLineEdit, QComboBox,
    QSpinBox, QDoubleSpinBox, QSlider, QFileDialog, QMessageBox,
    QStatusBar, QGridLayout, QSizePolicy, QFrame,
    QCheckBox, QListWidget, QListWidgetItem, QInputDialog, QDialog,
    QTabWidget, QScrollArea,
)

from labchart2cfd.io.labchart import LabChartData, load_labchart_mat
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


class AssemblyPreviewDialog(QDialog):
    """Popup showing the assembled synthetic flow profile."""

    def __init__(self, time, flow, n_images, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Synthetic Profile Preview")
        self.resize(640, 420)
        layout = QVBoxLayout(self)
        fig = Figure(figsize=(6, 3.5))
        canvas = FigureCanvasQTAgg(fig)
        ax = fig.add_subplot(111)
        ax.plot(time, flow, "b-", linewidth=0.8)
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
        ax.set_xlabel("Synthetic time (s)")
        ax.set_ylabel("Mass flow (kg/s)")
        ax.set_title(f"Assembled profile: {len(flow)} samples, {n_images} images")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        layout.addWidget(canvas)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)


class FlowProfileApp(QMainWindow):
    """Main GUI application for flow profile processing."""

    # Cross-thread completion signals for the Execute worker. QTimer.singleShot
    # from a plain threading.Thread never fires (no Qt event loop in that
    # thread), which left the button dead and error dialogs unshown; signal
    # emission from a foreign thread is queued to the GUI thread correctly.
    _processing_done = pyqtSignal(str)
    _processing_failed = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("CFD Flow Profile Processor")
        self.resize(1200, 800)
        self.setMinimumSize(900, 600)

        # State
        self.data: Optional[LabChartData] = None
        self.mat_filepath: Optional[Path] = None
        self._plot_mode = "empty"  # "empty", "overview", "selected"

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

    def _build_ui(self) -> None:
        """Build the complete UI layout."""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(5, 5, 5, 5)

        # --- Plot area (takes remaining space) ---
        plot_group = QGroupBox("Flow & Trigger Signals")
        plot_layout = QVBoxLayout(plot_group)

        self.fig = Figure(figsize=(12, 6), dpi=100, constrained_layout=True)
        # Show placeholder text instead of pre-created axes
        self.fig.text(
            0.5, 0.5, "Load a .mat file and click 'Load Overview' to begin",
            ha="center", va="center", fontsize=14, color="gray",
        )

        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Matplotlib navigation toolbar (zoom, pan, home, save)
        self.toolbar = NavigationToolbar2QT(self.canvas, plot_group)

        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)
        main_layout.addWidget(plot_group, stretch=1)

        # --- Controls area: tabbed + scrollable + height-capped so the plot
        # always dominates and panels never overlap/clip. ---
        self.tabs = QTabWidget()
        self.tabs.setMaximumHeight(370)
        self._tab_data_w = QWidget()
        self._tab_data_layout = QVBoxLayout(self._tab_data_w)
        self._tab_trig_w = QWidget()
        self._tab_trig_layout = QVBoxLayout(self._tab_trig_w)
        self._tab_ctstep_w = QWidget()
        self._tab_ctstep_layout = QVBoxLayout(self._tab_ctstep_w)
        self._tab_synth_w = QWidget()
        self._tab_synth_layout = QVBoxLayout(self._tab_synth_w)

        # -- Data Source group --
        file_group = QGroupBox("Data Source")
        file_grid = QGridLayout(file_group)

        # Row 0: file path + browse
        file_grid.addWidget(QLabel("File:"), 0, 0)
        self.file_label = QLabel("No file selected")
        self.file_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        file_grid.addWidget(self.file_label, 0, 1)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_file)
        file_grid.addWidget(browse_btn, 0, 2)
        file_grid.setColumnStretch(1, 1)

        # Row 1: channel assignment (flow/trigger/pressure rows) + column
        file_grid.addWidget(QLabel("Channels:"), 1, 0)
        spin_widget = QWidget()
        spin_hlayout = QHBoxLayout(spin_widget)
        spin_hlayout.setContentsMargins(0, 0, 0, 0)

        spin_hlayout.addWidget(QLabel("Flow Row:"))
        self.flow_row_spin = QSpinBox()
        self.flow_row_spin.setRange(1, 20)
        self.flow_row_spin.setValue(2)
        self.flow_row_spin.setFixedWidth(55)
        spin_hlayout.addWidget(self.flow_row_spin)

        spin_hlayout.addWidget(QLabel("  Trigger Row:"))
        self.trigger_row_spin = QSpinBox()
        self.trigger_row_spin.setRange(0, 20)  # 0 = no trigger
        self.trigger_row_spin.setValue(1)
        self.trigger_row_spin.setSpecialValueText("none")
        self.trigger_row_spin.setFixedWidth(65)
        spin_hlayout.addWidget(self.trigger_row_spin)

        spin_hlayout.addWidget(QLabel("  Pressure Row:"))
        self.pressure_row_spin = QSpinBox()
        self.pressure_row_spin.setRange(0, 20)  # 0 = no pressure export
        self.pressure_row_spin.setValue(4)
        self.pressure_row_spin.setSpecialValueText("none")
        self.pressure_row_spin.setFixedWidth(65)
        spin_hlayout.addWidget(self.pressure_row_spin)

        spin_hlayout.addWidget(QLabel("  Column:"))
        self.col_spin = QSpinBox()
        self.col_spin.setRange(1, 20)
        self.col_spin.setValue(3)
        self.col_spin.setFixedWidth(55)
        spin_hlayout.addWidget(self.col_spin)

        load_overview_btn = QPushButton("Load Overview")
        load_overview_btn.clicked.connect(self._load_and_overview)
        spin_hlayout.addSpacing(15)
        spin_hlayout.addWidget(load_overview_btn)

        plot_selected_btn = QPushButton("Plot Selected")
        plot_selected_btn.clicked.connect(self._plot_selected)
        spin_hlayout.addSpacing(5)
        spin_hlayout.addWidget(plot_selected_btn)

        spin_hlayout.addStretch()
        file_grid.addWidget(spin_widget, 1, 1, 1, 2)

        self._tab_data_layout.addWidget(file_group)

        # -- Trigger Info group --
        trigger_group = QGroupBox("Trigger Info")
        trigger_hlayout = QHBoxLayout(trigger_group)

        trigger_hlayout.addWidget(QLabel("Total Triggers:"))
        self.total_triggers_label = QLabel("--")
        self.total_triggers_label.setFixedWidth(50)
        self.total_triggers_label.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.total_triggers_label.setAlignment(Qt.AlignCenter)
        trigger_hlayout.addWidget(self.total_triggers_label)
        trigger_hlayout.addSpacing(15)

        trigger_hlayout.addWidget(QLabel("Start Trigger #:"))
        self.start_trigger_spin = QSpinBox()
        self.start_trigger_spin.setRange(1, 1)
        self.start_trigger_spin.setValue(1)
        self.start_trigger_spin.setFixedWidth(60)
        trigger_hlayout.addWidget(self.start_trigger_spin)
        trigger_hlayout.addSpacing(15)

        trigger_hlayout.addWidget(QLabel("# Triggers:"))
        self.num_triggers_spin = QSpinBox()
        self.num_triggers_spin.setRange(1, 1)
        self.num_triggers_spin.setValue(1)
        self.num_triggers_spin.setFixedWidth(60)
        trigger_hlayout.addWidget(self.num_triggers_spin)
        trigger_hlayout.addSpacing(15)

        mark_triggers_btn = QPushButton("Mark Triggers")
        mark_triggers_btn.clicked.connect(self._mark_trigger_range)
        trigger_hlayout.addWidget(mark_triggers_btn)

        clear_markers_btn = QPushButton("Clear Markers")
        clear_markers_btn.clicked.connect(self._clear_all_markers)
        trigger_hlayout.addWidget(clear_markers_btn)
        trigger_hlayout.addStretch()

        self._tab_trig_layout.addWidget(trigger_group)

        # -- Processing group --
        proc_group = QGroupBox("Processing")
        proc_grid = QGridLayout(proc_group)

        # Row 0: start/end time + subject ID
        proc_grid.addWidget(QLabel("Start Time (s):"), 0, 0)
        self.start_spin = QDoubleSpinBox()
        self.start_spin.setRange(0, 99999)
        self.start_spin.setDecimals(3)
        self.start_spin.setSingleStep(0.001)
        self.start_spin.setValue(0.0)
        self.start_spin.setFixedWidth(100)
        proc_grid.addWidget(self.start_spin, 0, 1)

        proc_grid.addWidget(QLabel("End Time (s):"), 0, 2)
        self.end_spin = QDoubleSpinBox()
        self.end_spin.setRange(0, 99999)
        self.end_spin.setDecimals(3)
        self.end_spin.setSingleStep(0.001)
        self.end_spin.setValue(0.0)
        self.end_spin.setFixedWidth(100)
        proc_grid.addWidget(self.end_spin, 0, 3)

        proc_grid.addWidget(QLabel("Subject ID:"), 0, 4)
        self.subject_edit = QLineEdit()
        self.subject_edit.setFixedWidth(120)
        self.subject_edit.textChanged.connect(self._update_outdir)
        proc_grid.addWidget(self.subject_edit, 0, 5)

        zoom_btn = QPushButton("Zoom to Window")
        zoom_btn.setToolTip(
            "Fit the plot to Start/End with a small margin. Happens "
            "automatically on step select and when you commit Start/End."
        )
        zoom_btn.clicked.connect(lambda: self._zoom_to_window())
        proc_grid.addWidget(zoom_btn, 0, 6)

        # Row 1: workflow + output dir + execute
        proc_grid.addWidget(QLabel("Workflow:"), 1, 0)
        self.workflow_combo = QComboBox()
        self.workflow_combo.addItems(["MRI", "cpap", "phase-contrast", "CT"])
        self.workflow_combo.setFixedWidth(120)
        self.workflow_combo.currentTextChanged.connect(self._on_workflow_changed)
        proc_grid.addWidget(self.workflow_combo, 1, 1)

        proc_grid.addWidget(QLabel("Output Dir:"), 1, 2)
        self.outdir_edit = QLineEdit(str(Path.cwd()))
        proc_grid.addWidget(self.outdir_edit, 1, 3, 1, 2)
        proc_grid.setColumnStretch(3, 1)

        outdir_browse_btn = QPushButton("...")
        outdir_browse_btn.setFixedWidth(30)
        outdir_browse_btn.clicked.connect(self._browse_outdir)
        proc_grid.addWidget(outdir_browse_btn, 1, 5)

        self.execute_btn = QPushButton("Execute")
        self.execute_btn.clicked.connect(self._execute_processing)
        proc_grid.addWidget(self.execute_btn, 1, 6)

        self._tab_data_layout.addWidget(proc_group)

        # -- CT-specific controls (hidden by default) --
        self.ct_group = QGroupBox("CT Step Trigger Controls")
        ct_grid = QGridLayout(self.ct_group)

        # Row 0: Detect Steps + Step selection + Temporal Res + Total Images
        detect_steps_btn = QPushButton("Detect Steps")
        detect_steps_btn.clicked.connect(self._detect_steps)
        ct_grid.addWidget(detect_steps_btn, 0, 0)

        ct_grid.addWidget(QLabel("Step #:"), 0, 1)
        self.ct_step_spin = QSpinBox()
        self.ct_step_spin.setRange(1, 1)
        self.ct_step_spin.setValue(1)
        self.ct_step_spin.setFixedWidth(60)
        ct_grid.addWidget(self.ct_step_spin, 0, 2)

        select_step_btn = QPushButton("Select Step")
        select_step_btn.clicked.connect(self._select_ct_step)
        ct_grid.addWidget(select_step_btn, 0, 3)

        ct_grid.addWidget(QLabel("Temporal Res (ms):"), 0, 4)
        # Same type, range and precision as the tab-4 dt box. As an int
        # spinbox this silently truncated any fractional dt mirrored from
        # tab 4, leaving the CT export and the synthetic grid on different dt.
        self.ct_temporal_res_spin = QDoubleSpinBox()
        self.ct_temporal_res_spin.setRange(0.01, 10000.0)
        self.ct_temporal_res_spin.setDecimals(2)
        self.ct_temporal_res_spin.setSingleStep(10)
        self.ct_temporal_res_spin.setValue(200.0)
        self.ct_temporal_res_spin.setFixedWidth(80)
        ct_grid.addWidget(self.ct_temporal_res_spin, 0, 5)

        ct_grid.addWidget(QLabel("Total Images:"), 0, 6)
        self.ct_total_images_spin = QSpinBox()
        self.ct_total_images_spin.setRange(1, 500)
        self.ct_total_images_spin.setValue(1)
        self.ct_total_images_spin.setFixedWidth(60)
        ct_grid.addWidget(self.ct_total_images_spin, 0, 7)

        # Row 1: Landmark buttons + time displays
        mark_inhale_btn = QPushButton("Mark Inhale Start")
        mark_inhale_btn.clicked.connect(self._mark_inhale_start)
        ct_grid.addWidget(mark_inhale_btn, 1, 0)

        self.ct_inhale_label = QLabel("--")
        self.ct_inhale_label.setFixedWidth(100)
        self.ct_inhale_label.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.ct_inhale_label.setAlignment(Qt.AlignCenter)
        ct_grid.addWidget(self.ct_inhale_label, 1, 1, 1, 2)

        mark_exhale_btn = QPushButton("Mark Exhale End")
        mark_exhale_btn.clicked.connect(self._mark_exhale_end)
        ct_grid.addWidget(mark_exhale_btn, 1, 3)

        self.ct_exhale_label = QLabel("--")
        self.ct_exhale_label.setFixedWidth(100)
        self.ct_exhale_label.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.ct_exhale_label.setAlignment(Qt.AlignCenter)
        ct_grid.addWidget(self.ct_exhale_label, 1, 4, 1, 2)

        clear_landmarks_btn = QPushButton("Clear Landmarks")
        clear_landmarks_btn.clicked.connect(self._clear_ct_landmarks)
        ct_grid.addWidget(clear_landmarks_btn, 1, 6, 1, 2)

        # Row 2: Start Shift (manual correction) + Apply button
        ct_grid.addWidget(QLabel("Start Shift (ms):"), 2, 0)
        self.ct_shift_spin = QSpinBox()
        self.ct_shift_spin.setRange(0, 2000)
        self.ct_shift_spin.setSingleStep(10)
        self.ct_shift_spin.setValue(0)
        self.ct_shift_spin.setFixedWidth(70)
        self.ct_shift_spin.editingFinished.connect(self._on_shift_entry_changed)
        ct_grid.addWidget(self.ct_shift_spin, 2, 1)

        self.ct_shift_slider = QSlider(Qt.Horizontal)
        self.ct_shift_slider.setRange(0, 2000)
        self.ct_shift_slider.setSingleStep(10)
        self.ct_shift_slider.setPageStep(10)
        self.ct_shift_slider.setValue(0)
        self.ct_shift_slider.setMinimumWidth(200)
        self.ct_shift_slider.valueChanged.connect(self._on_shift_slider_changed)
        ct_grid.addWidget(self.ct_shift_slider, 2, 2, 1, 4)

        apply_correction_btn = QPushButton("Apply Correction")
        apply_correction_btn.clicked.connect(self._apply_ct_correction)
        ct_grid.addWidget(apply_correction_btn, 2, 6, 1, 2)

        # Row 3: Cycle frame labeling (alternative to inhale/exhale trim)
        ct_grid.addWidget(QLabel("Cycle:"), 3, 0)
        self.cycle_mode_btn = QPushButton("Label Cycle Frames")
        self.cycle_mode_btn.setCheckable(True)
        self.cycle_mode_btn.setToolTip(
            "Toggle: click fenceposts in order to build a breathing cycle. "
            "Right-click removes the most recent label. Overrides inhale/exhale trim on export."
        )
        self.cycle_mode_btn.toggled.connect(self._on_cycle_mode_toggled)
        ct_grid.addWidget(self.cycle_mode_btn, 3, 1, 1, 2)

        self.cycle_count_label = QLabel("0 frames")
        self.cycle_count_label.setFixedWidth(100)
        self.cycle_count_label.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.cycle_count_label.setAlignment(Qt.AlignCenter)
        ct_grid.addWidget(self.cycle_count_label, 3, 3)

        clear_cycle_btn = QPushButton("Clear Cycle Labels")
        clear_cycle_btn.clicked.connect(self._clear_cycle_labels)
        ct_grid.addWidget(clear_cycle_btn, 3, 4, 1, 2)

        self.cycle_seq_label = QLabel("seq: —")
        self.cycle_seq_label.setToolTip("Original image indices in click order")
        ct_grid.addWidget(self.cycle_seq_label, 3, 6, 1, 2)

        self._tab_ctstep_layout.addWidget(self.ct_group)

        # ---- Full-manual synthetic-profile controls (shown with CT) ----
        self.synth_group = QGroupBox("Synthetic Profile (Full Manual)")
        synth_layout = QVBoxLayout(self.synth_group)

        srow1 = QHBoxLayout()
        self.synth_enable_chk = QCheckBox("Enable full-manual mode")
        self.synth_enable_chk.setToolTip(
            "Use Start/End (above) as the window and Temporal Res for the image "
            "grid; click image frames to build ordered parts; assemble a "
            "synthetic breathing profile + image reorder map."
        )
        self.synth_enable_chk.toggled.connect(self._on_synth_mode_toggled)
        srow1.addWidget(self.synth_enable_chk)
        show_grid_btn = QPushButton("Show Image Grid")
        # Explicit lambda: clicked() would otherwise pass its checked flag as
        # the `quiet` argument.
        show_grid_btn.clicked.connect(lambda: self._draw_synth_grid(quiet=False))
        srow1.addWidget(show_grid_btn)
        self.synth_grid_label = QLabel("images: —")
        srow1.addWidget(self.synth_grid_label)
        srow1.addStretch(1)
        synth_layout.addLayout(srow1)

        # Grid definition: dt and image count are two views of one grid, so
        # editing either recomputes the other. Offsets slide the window and
        # the grid independently.
        srow_grid = QHBoxLayout()
        srow_grid.addWidget(QLabel("Temporal Res (ms):"))
        # Double, because a dt derived from an image count is rarely a whole ms.
        self.synth_dt_spin = QDoubleSpinBox()
        self.synth_dt_spin.setRange(0.01, 10000.0)
        self.synth_dt_spin.setDecimals(2)
        self.synth_dt_spin.setValue(float(self.ct_temporal_res_spin.value()))
        self.synth_dt_spin.setFixedWidth(80)
        self.synth_dt_spin.setToolTip(
            "Seconds per image (one image owns exactly this much signal). "
            "Editing this recomputes the image count."
        )
        srow_grid.addWidget(self.synth_dt_spin)

        srow_grid.addWidget(QLabel("  Images:"))
        self.synth_images_spin = QSpinBox()
        self.synth_images_spin.setRange(1, 5000)
        self.synth_images_spin.setValue(1)
        self.synth_images_spin.setFixedWidth(70)
        self.synth_images_spin.setToolTip(
            "Number of images in the window (ground truth from your "
            "reconstructed series). Editing this recomputes Temporal Res."
        )
        srow_grid.addWidget(self.synth_images_spin)

        srow_grid.addWidget(QLabel("  Window offset (ms):"))
        self.synth_window_offset_spin = QSpinBox()
        self.synth_window_offset_spin.setRange(-600000, 600000)
        self.synth_window_offset_spin.setValue(0)
        self.synth_window_offset_spin.setFixedWidth(80)
        self.synth_window_offset_spin.setToolTip(
            "Slides Start and End together, moving the whole processed range."
        )
        srow_grid.addWidget(self.synth_window_offset_spin)
        apply_win_offset_btn = QPushButton("Apply")
        apply_win_offset_btn.setToolTip("Shift Start/End by the window offset.")
        apply_win_offset_btn.clicked.connect(self._apply_window_offset)
        srow_grid.addWidget(apply_win_offset_btn)

        srow_grid.addWidget(QLabel("  Grid offset (ms):"))
        self.synth_grid_offset_spin = QSpinBox()
        # Negative allowed: the detected step edge can sit after the true first
        # image, so the grid has to be shiftable in both directions.
        self.synth_grid_offset_spin.setRange(-600000, 600000)
        self.synth_grid_offset_spin.setValue(0)
        self.synth_grid_offset_spin.setFixedWidth(80)
        self.synth_grid_offset_spin.setToolTip(
            "Slides the whole grid earlier (negative) or later (positive) "
            "without changing Start/End. This moves the exported range."
        )
        srow_grid.addWidget(self.synth_grid_offset_spin)
        srow_grid.addStretch(1)
        synth_layout.addLayout(srow_grid)

        self.synth_dt_warn_label = QLabel("")
        self.synth_dt_warn_label.setStyleSheet("color: #b00;")
        synth_layout.addWidget(self.synth_dt_warn_label)

        help_lbl = QLabel(
            "① Enable full-manual mode   ② set From/To (or 'Pick on Plot')   "
            "③ Add Part   ④ repeat → Assemble && Export       "
            "An image is the slice between two grid lines; From=To selects one image."
        )
        help_lbl.setWordWrap(True)
        help_lbl.setStyleSheet("color: #555;")
        synth_layout.addWidget(help_lbl)

        # Range entry: explicit begin/end image numbers
        srow_range = QHBoxLayout()
        srow_range.addWidget(QLabel("From image:"))
        self.part_from_spin = QSpinBox()
        self.part_from_spin.setRange(1, 1)
        self.part_from_spin.setFixedWidth(70)
        srow_range.addWidget(self.part_from_spin)
        srow_range.addWidget(QLabel("To image:"))
        self.part_to_spin = QSpinBox()
        self.part_to_spin.setRange(1, 1)
        self.part_to_spin.setFixedWidth(70)
        srow_range.addWidget(self.part_to_spin)
        add_part_btn = QPushButton("Add Part")
        add_part_btn.setToolTip("Add a part covering From..To images, in assembly order.")
        add_part_btn.clicked.connect(self._add_part_from_spins)
        srow_range.addWidget(add_part_btn)
        self.pick_part_btn = QPushButton("Pick on Plot")
        self.pick_part_btn.setToolTip("Click two frames on the plot to fill From/To.")
        self.pick_part_btn.clicked.connect(self._arm_add_part)
        srow_range.addWidget(self.pick_part_btn)
        self.capture_hint = QLabel("")
        self.capture_hint.setStyleSheet("color: #c00; font-weight: bold;")
        srow_range.addWidget(self.capture_hint)
        srow_range.addStretch(1)
        synth_layout.addLayout(srow_range)

        self.parts_list = QListWidget()
        self.parts_list.setMaximumHeight(88)
        synth_layout.addWidget(self.parts_list)

        srow_actions = QHBoxLayout()
        for _text, _slot in [
            ("Move Up", self._move_part_up), ("Move Down", self._move_part_down),
            ("Edit Label", self._edit_part_label), ("Delete", self._delete_part),
            ("Clear Parts", self._clear_parts), ("Clear All", self._clear_synth_all),
        ]:
            _b = QPushButton(_text)
            _b.clicked.connect(_slot)
            srow_actions.addWidget(_b)
        srow_actions.addStretch(1)
        synth_layout.addLayout(srow_actions)

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

        srow3 = QHBoxLayout()
        assemble_btn = QPushButton("Assemble && Export")
        assemble_btn.clicked.connect(self._assemble_and_export)
        srow3.addWidget(assemble_btn)
        self.synth_status_label = QLabel("no parts")
        srow3.addWidget(self.synth_status_label)
        srow3.addStretch(1)
        synth_layout.addLayout(srow3)

        self._tab_synth_layout.addWidget(self.synth_group)

        # Finalize the tabbed control strip
        for _lay in (self._tab_data_layout, self._tab_trig_layout,
                     self._tab_ctstep_layout, self._tab_synth_layout):
            _lay.setContentsMargins(6, 6, 6, 6)
            _lay.addStretch(1)

        def _scroll(widget):
            sa = QScrollArea()
            sa.setWidgetResizable(True)
            sa.setFrameShape(QFrame.NoFrame)
            sa.setWidget(widget)
            return sa

        self.tabs.addTab(_scroll(self._tab_data_w), "1. Data && Window")
        self.tabs.addTab(_scroll(self._tab_trig_w), "2. Triggers (4D)")
        self._ct_step_tab = self.tabs.addTab(_scroll(self._tab_ctstep_w), "3. CT Step-Trigger")
        self._ct_synth_tab = self.tabs.addTab(_scroll(self._tab_synth_w), "4. CT Synthetic Profile")
        main_layout.addWidget(self.tabs, stretch=0)

        # Initialize CT-tab enablement from the current workflow.
        self._on_workflow_changed(self.workflow_combo.currentText())

        # --- Status bar ---
        self.status_bar = QStatusBar()
        self.status_bar.showMessage("Ready")
        self.setStatusBar(self.status_bar)

        # Worker-completion signals land on the GUI thread here.
        self._processing_done.connect(self._on_processing_done)
        self._processing_failed.connect(self._on_processing_failed)

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

    def _on_workflow_changed(self, text: str) -> None:
        """Enable the CT-only tabs and focus them when CT is selected."""
        if not hasattr(self, "_ct_synth_tab"):
            return  # tabs not built yet (called during construction)
        is_ct = text == "CT"
        self.tabs.setTabEnabled(self._ct_step_tab, is_ct)
        self.tabs.setTabEnabled(self._ct_synth_tab, is_ct)
        if not is_ct and self.synth_enable_chk.isChecked():
            self.synth_enable_chk.setChecked(False)
        if is_ct:
            self.tabs.setCurrentIndex(self._ct_synth_tab)

    def _update_outdir(self, *_args) -> None:
        """Recompute output dir based on current mat file and subject."""
        if self.mat_filepath is None:
            return
        subject = self.subject_edit.text().strip()
        if subject:
            results_dir = self.mat_filepath.parent / f"FlowResults_{subject}"
        else:
            results_dir = self.mat_filepath.parent / "FlowResults"
        self.outdir_edit.setText(str(results_dir))

    def _browse_file(self) -> None:
        """Open file dialog to select a .mat file."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Select LabChart .mat file", "",
            "MAT files (*.mat);;All files (*.*)",
        )
        if filepath:
            self.mat_filepath = Path(filepath)
            self.file_label.setText(str(self.mat_filepath))
            self._update_outdir()

    def _browse_outdir(self) -> None:
        """Open directory dialog to select output directory."""
        dirpath = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dirpath:
            self.outdir_edit.setText(dirpath)

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
        self._time_markers = []
        self._trigger_markers = []
        self._ct_step_markers = []
        self._ct_interval_markers = []
        self._ct_landmark_artists = []
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
        if hasattr(self, "cycle_count_label"):
            self.cycle_count_label.setText("0 frames")

    def _load_and_overview(self) -> None:
        """Load .mat file and show overview grid of all channels x blocks."""
        if self.mat_filepath is None:
            QMessageBox.warning(self, "No File", "Please select a .mat file first.")
            return
        if getattr(self, "_loading", False):
            return  # processEvents below can re-dispatch a queued click here
        self._loading = True

        try:
            self._reset_plot_state()

            self.status_bar.showMessage(f"Loading {self.mat_filepath.name}...")
            QApplication.processEvents()

            try:
                self.data = load_labchart_mat(self.mat_filepath)
            except Exception as e:
                # A failed load must not leave the PREVIOUS file's data alive
                # under the new file's name: the plot would title old data with
                # the new name and Execute would export it beside the new file.
                self.data = None
                self._plot_mode = "empty"
                self.fig.clear()
                self.canvas.draw_idle()
                QMessageBox.critical(self, "Load Error", str(e))
                self.status_bar.showMessage("Error loading file")
                return

            # A different recording invalidates any parts labelled against the
            # previous file's grid.
            if getattr(self, "_loaded_filepath", None) != self.mat_filepath:
                self._parts = []
                self._refresh_parts_list()
            self._loaded_filepath = self.mat_filepath
        finally:
            self._loading = False
        self._show_overview()

    def _show_overview(self) -> None:
        """Draw the channels-x-blocks overview grid for the loaded file."""

        nch = self.data.num_channels
        nbl = self.data.num_blocks

        # Update spinbox ranges to match actual data dimensions
        self.flow_row_spin.setMaximum(nch)
        self.trigger_row_spin.setMaximum(nch)  # 0 still allowed (= none)
        self.pressure_row_spin.setMaximum(nch)  # 0 still allowed (= none)
        self.col_spin.setMaximum(nbl)
        # Clamp current values into valid range
        if self.flow_row_spin.value() > nch:
            self.flow_row_spin.setValue(nch)
        if self.trigger_row_spin.value() > nch:
            self.trigger_row_spin.setValue(nch)
        if self.pressure_row_spin.value() > nch:
            self.pressure_row_spin.setValue(nch)
        if self.col_spin.value() > nbl:
            self.col_spin.setValue(nbl)

        # Build overview grid
        self.fig.clear()
        self._plot_mode = "overview"
        self.ax_top = None
        self.ax_bottom = None
        self.ax_top_trig = None

        axes = []
        for ch_idx in range(nch):
            row_axes = []
            for bl_idx in range(nbl):
                ax = self.fig.add_subplot(nch, nbl, ch_idx * nbl + bl_idx + 1)
                row = ch_idx + 1  # 1-indexed
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
                    # Downsample for speed
                    step = max(1, len(time) // 500)
                    ax.plot(time[::step], data[::step], "b-", linewidth=0.3)

                ax.set_title(f"[{row},{col}]", fontsize=7, pad=2)
                ax.tick_params(labelsize=5)
                # Only show axis labels on edges
                if ch_idx < nch - 1:
                    ax.set_xticklabels([])
                if bl_idx > 0:
                    ax.set_yticklabels([])

                row_axes.append(ax)
            axes.append(row_axes)

        self.canvas.draw()
        self.toolbar.update()

        self.status_bar.showMessage(
            f"Overview: {nch} channels x {nbl} blocks | "
            f"Select row/col and click 'Plot Selected' for detail"
        )

    def _plot_selected(self) -> None:
        """Plot a specific block in detail (flow + trigger top, flow-only bottom)."""
        if self.data is None:
            QMessageBox.warning(self, "No Data", "Please load a .mat file first (click 'Load Overview').")
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

        # Top subplot: flow on primary y-axis
        self.ax_top.plot(time, flow, "b-", linewidth=0.5, label="flow info")

        has_trigger = trigger_row >= 1 and not self.data.is_block_empty(trigger_row, col)
        is_ct = self.workflow_combo.currentText() == "CT"
        if has_trigger:
            trigger = self.data.get_data(trigger_row, col)
            trigger_time = self.data.get_time(trigger_row, col)
            # Trigger on twin y-axis: different amplitude range won't squeeze flow.
            self.ax_top_trig = self.ax_top.twinx()
            self.ax_top_trig.plot(trigger_time, trigger, "r-", linewidth=0.5, label="trigger info")
            self.ax_top_trig.set_ylabel("Trigger", color="red")
            self.ax_top_trig.tick_params(axis="y", labelcolor="red")

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
        self.ax_top.set_title(
            f"{self.mat_filepath.name} — Flow [{flow_row}, {col}]"
            f"{f' + Trigger [{trigger_row}, {col}]' if has_trigger else ''}"
            f"{trigger_info}"
        )
        self.ax_top.grid(True, alpha=0.3)
        self._refresh_top_legend()

        # Bottom subplot: flow only (for zoom/selection)
        self.ax_bottom.plot(time, flow, "b-", linewidth=0.5)
        self.ax_bottom.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
        self.ax_bottom.set_xlabel("Time (s)")
        self.ax_bottom.set_ylabel("Flow (L/s)")
        self.ax_bottom.set_title("Flow Signal — left-click to place time marker, right-click to clear")
        self.ax_bottom.grid(True, alpha=0.3)
        self.ax_bottom.format_coord = lambda x, y: f"time={x:.3f}s, flow={y:.4f}"

        self.canvas.draw()
        self.toolbar.update()

        # Connect click handler for data picker
        self._click_cid = self.canvas.mpl_connect("button_press_event", self._on_click)

        self.status_bar.showMessage(
            f"Flow [{flow_row}, {col}]: {len(time)} samples, {time[-1]:.2f}s{trigger_info} | "
            f"Left-click bottom plot to mark time, right-click to clear"
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

    def _on_click(self, event) -> None:
        """Handle mouse click on the canvas for data picker."""
        if self._plot_mode != "selected":
            return
        # Include ax_top_trig (twin y-axis) so clicks over the trigger curve also register.
        valid_axes = (self.ax_top, self.ax_bottom, self.ax_top_trig)
        if event.inaxes not in valid_axes or event.inaxes is None:
            return
        # Ignore clicks when toolbar is in zoom/pan mode
        if self.toolbar.mode:
            return

        if event.button == 1:  # Left click
            if self._capturing_part:
                self._capture_part_click(event.xdata)
            elif self._ct_cycle_mode:
                self._add_cycle_label(event.xdata)
            elif self._ct_landmark_mode is not None:
                self._add_ct_landmark(event.xdata)
            elif self._synth_mode:
                # Full-manual mode: ignore unarmed clicks. Time markers would
                # overwrite Start/End, and the image grid is computed live from
                # those, so a stray click would silently redefine the grid and
                # invalidate parts already added.
                self.status_bar.showMessage(
                    "Full-manual mode: click 'Pick on Plot' first to select frames."
                )
            else:
                self._add_time_marker(event.xdata)
        elif event.button == 3:  # Right click
            if self._capturing_part:
                self._cancel_part_capture()
            elif self._ct_cycle_mode:
                self._remove_last_cycle_label()
            else:
                self._clear_time_markers()

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
        mask = (t >= x_lo) & (t <= x_hi)
        if not np.any(mask):
            return None
        lo = float(np.min(y[mask]))
        hi = float(np.max(y[mask]))
        pad = (hi - lo) * self._ZOOM_Y_MARGIN
        if pad == 0:  # flat signal: keep the view from collapsing to a line
            pad = abs(hi) * self._ZOOM_Y_MARGIN or 1.0
        return lo - pad, hi + pad

    def _zoom_to_window(self, start_t: Optional[float] = None,
                        end_t: Optional[float] = None) -> None:
        """Zoom x and y to the processing window, with margin at every edge.

        Each y axis is scaled to the data actually inside the window, so a
        short step buried in a long recording fills the view instead of being
        flattened by the full-recording range.
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

        if self.ax_top_trig is not None:
            trig_lim = self._channel_y_limits(
                self.trigger_row_spin.value(), col, start_t, end_t
            )
            if trig_lim is not None:
                self.ax_top_trig.set_ylim(*trig_lim)

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
            QMessageBox.warning(self, "No Plot", "Please click 'Plot Selected' first.")
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
            QMessageBox.warning(self, "No Data", "Please load a .mat file first.")
            return
        if self.ax_top is None:
            QMessageBox.warning(self, "No Plot", "Please click 'Plot Selected' first.")
            return

        trigger_row = self.trigger_row_spin.value()
        col = self.col_spin.value()

        if trigger_row < 1 or self.data.is_block_empty(trigger_row, col):
            QMessageBox.warning(
                self, "No Trigger",
                "No trigger data. Set Trigger Row (in Data Source) to a non-empty channel.",
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
        self.ct_shift_slider.setMaximum(slider_max)
        self.ct_shift_spin.setMaximum(slider_max)
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

        # Bring the step into view so a short scan inside a long recording is
        # actually readable.
        self._zoom_to_window(step["start_time"], step["end_time"])

        # Reset landmarks AND cycle frames for the new step (stale cycle frames
        # from a prior step must not leak into this step's export).
        self._clear_ct_landmarks()
        self._clear_cycle_labels()

        self.canvas.draw_idle()
        self.status_bar.showMessage(
            f"Step {step_num} selected ({step['duration']:.2f}s, "
            f"{self.ct_total_images_spin.value()} images) | "
            f"Use buttons to mark Inhale Start and Exhale End"
        )

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

    def _mark_exhale_end(self) -> None:
        """Activate click mode to place exhale end landmark."""
        self._claim_click_mode()
        self._ct_landmark_mode = "exhale_end"
        self.status_bar.showMessage("Click on plot to mark EXHALE END")

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
        # Clamp to slider range to avoid desync
        slider_max = self.ct_shift_slider.maximum()
        val = max(0, min(val, slider_max))
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
            # Stop landmark placement so clicks don't race.
            self._ct_landmark_mode = None
            self.status_bar.showMessage(
                "CYCLE LABEL MODE: left-click fenceposts in order (right-click undoes last)."
            )
        else:
            self.status_bar.showMessage("Cycle label mode off.")

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

        self._update_cycle_ui()
        self.canvas.draw_idle()

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
        self._update_cycle_ui()
        self.canvas.draw_idle()
        self.status_bar.showMessage(f"Removed cycle label for original frame {removed}.")

    def _clear_cycle_labels(self) -> None:
        """Remove every cycle label and its artists."""
        for artist in self._ct_cycle_artists:
            try:
                artist.remove()
            except (ValueError, AttributeError):
                pass
        self._ct_cycle_artists = []
        self._ct_cycle_frames = []
        self._update_cycle_ui()
        self.canvas.draw_idle()
        self.status_bar.showMessage("Cycle labels cleared.")

    def _update_cycle_ui(self) -> None:
        """Refresh the count + sequence labels in the CT controls."""
        n = len(self._ct_cycle_frames)
        self.cycle_count_label.setText(f"{n} frame{'s' if n != 1 else ''}")
        if n == 0:
            self.cycle_seq_label.setText("seq: —")
        else:
            seq = ",".join(str(i) for i in self._ct_cycle_frames)
            if len(seq) > 80:
                seq = seq[:77] + "..."
            self.cycle_seq_label.setText(f"seq: {seq}")

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
            # Seed the image count from the step the first time only, as a
            # starting point. After that it is yours and nothing rewrites it.
            if self.synth_images_spin.value() <= 1 and self.ct_total_images_spin.value() > 1:
                self.synth_images_spin.setValue(self.ct_total_images_spin.value())
            self._draw_synth_grid()
            self.status_bar.showMessage(
                "FULL MANUAL: set Start/End + Temporal Res, then 'Add Part' and "
                "click two image frames per part."
            )
        else:
            self._cancel_part_capture()
            self.status_bar.showMessage("Full-manual mode off.")

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

    def _on_ct_dt_changed(self, dt_ms: int) -> None:
        """Tab 3's Temporal Res edited: mirror it into the synth grid."""
        if self._syncing_grid:
            return
        self._syncing_grid = True
        try:
            self.synth_dt_spin.setValue(float(dt_ms))
        finally:
            self._syncing_grid = False
        self._refresh_synth_grid_if_active()
        self._update_grid_readout()

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
                self.synth_grid_label.setText("images: —")
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
            QMessageBox.information(self, "Enable Mode", "Tick 'Enable full-manual mode' first.")
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
        self.capture_hint.setText("Click the START frame on the plot…")
        self.status_bar.showMessage("Click the START frame on the plot (right-click cancels).")

    def _cancel_part_capture(self) -> None:
        self._capturing_part = False
        self._part_click_buffer = []
        self._clear_capture_marker()
        if hasattr(self, "capture_hint"):
            self.capture_hint.setText("")

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
            self.capture_hint.setText(f"START = image {img} — click END frame")
            self.status_bar.showMessage(f"START = image {img}. Click the END frame.")
            return
        a, b = sorted(self._part_click_buffer)
        self._capturing_part = False
        self._part_click_buffer = []
        self._clear_capture_marker()
        # a == b is a legal single-image part, same as typing From = To.
        self.part_from_spin.setValue(a)
        self.part_to_spin.setValue(b)
        self.capture_hint.setText(f"range {a}..{b} ready — click 'Add Part'")
        self.status_bar.showMessage(f"Range {a}-{b} set. Click 'Add Part' to add it.")

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
        self.synth_grid_label.setText("images: —")
        self.canvas.draw_idle()
        self.status_bar.showMessage("Cleared parts, image grid, and plot markers.")

    def _assemble_and_export(self) -> None:
        """Process the window, assemble parts, preview, and export 3 CSVs."""
        if self.data is None:
            QMessageBox.warning(self, "No Data", "Load a .mat file first.")
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

        # Preview
        AssemblyPreviewDialog(asm.time, asm.flow, len(asm.reorder_map), parent=self).exec_()

        # Export
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            flow_file = output_dir / f"{subject}FlowProfile.csv"
            export_flow_csv(flow_file, asm.time, asm.flow)
            written = [flow_file.name]
            if asm.pressure is not None:
                pfile = output_dir / f"{subject}PressureProfile.csv"
                export_pressure_csv(pfile, asm.time, asm.pressure)
                written.append(pfile.name)
            map_file = output_dir / f"{subject}ImageReorderMap.csv"
            export_reorder_map_csv(map_file, asm.reorder_map)
            written.append(map_file.name)
        except Exception as exc:
            QMessageBox.critical(self, "Export Failed", str(exc))
            return

        order = ",".join(str(m[1]) for m in asm.reorder_map)
        if len(order) > 120:
            order = order[:117] + "..."
        QMessageBox.information(
            self, "Exported",
            f"Wrote to {output_dir}:\n  " + "\n  ".join(written)
            + f"\n\nAssembled {len(asm.flow)} samples, {len(asm.reorder_map)} images."
            + f"\nReorder (original image order):\n{order}",
        )
        self.status_bar.showMessage(f"Exported synthetic profile + reorder map for {subject}.")

    def _execute_processing(self) -> None:
        """Run the processing pipeline in a background thread."""
        # Validate inputs
        if self.data is None:
            QMessageBox.warning(self, "No Data", "Please load a .mat file first.")
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
        workflow_name = self.workflow_combo.currentText()
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
        if start_time > float(t_axis[-1]) or end_time < float(t_axis[0]):
            QMessageBox.critical(
                self, "Window Outside Recording",
                f"The window {start_time:.3f}s to {end_time:.3f}s does not "
                f"intersect the recording, which covers "
                f"{float(t_axis[0]):.3f}s to {float(t_axis[-1]):.3f}s.",
            )
            return

        # Cycle labels were snapped against the drawn fencepost grid, which is
        # anchored at detected_start + shift. The export windows at Start Time.
        # If those disagree (shift previewed but Apply Correction not clicked,
        # or Start typed by hand), every chunk would be taken from the wrong
        # place, so refuse rather than export silently shifted data.
        if self._ct_cycle_frames and self._ct_detected_start is not None:
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
        ct_inhale_start = self._ct_inhale_start
        ct_exhale_end = self._ct_exhale_end
        ct_selected_step = self._ct_selected_step
        ct_temporal_res_ms = float(self.ct_temporal_res_spin.value())
        # Cycle-labeling: ordered original image indices (empty → landmark path)
        ct_cycle_frames = list(self._ct_cycle_frames)
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

                self._processing_done.emit(msg)

            except Exception as e:
                import traceback
                traceback.print_exc()  # so the real error shows in the terminal
                self._processing_failed.emit(f"{type(e).__name__}: {e}")

        Thread(target=_run, daemon=True).start()


def launch_gui() -> None:
    """Launch the Flow Profile GUI application."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    window = FlowProfileApp()
    window.show()
    app.exec()
