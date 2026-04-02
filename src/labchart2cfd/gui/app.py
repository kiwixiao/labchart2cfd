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

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QPushButton, QLineEdit, QComboBox,
    QSpinBox, QDoubleSpinBox, QSlider, QFileDialog, QMessageBox,
    QStatusBar, QGridLayout, QSizePolicy, QFrame,
)

from labchart2cfd.io.labchart import LabChartData, load_labchart_mat


class FlowProfileApp(QMainWindow):
    """Main GUI application for flow profile processing."""

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

        self.fig = Figure(figsize=(12, 6), dpi=100)
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

        # --- Controls area (bottom, fixed height) ---
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setContentsMargins(0, 0, 0, 0)

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

        # Row 1: row/col spinboxes + Load Overview + Plot Selected
        file_grid.addWidget(QLabel("Row:"), 1, 0)
        spin_widget = QWidget()
        spin_hlayout = QHBoxLayout(spin_widget)
        spin_hlayout.setContentsMargins(0, 0, 0, 0)

        self.row_spin = QSpinBox()
        self.row_spin.setRange(1, 20)
        self.row_spin.setValue(2)
        self.row_spin.setFixedWidth(60)
        spin_hlayout.addWidget(self.row_spin)

        spin_hlayout.addWidget(QLabel("  Column:"))
        self.col_spin = QSpinBox()
        self.col_spin.setRange(1, 20)
        self.col_spin.setValue(3)
        self.col_spin.setFixedWidth(60)
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

        controls_layout.addWidget(file_group)

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

        controls_layout.addWidget(trigger_group)

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

        controls_layout.addWidget(proc_group)

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
        self.ct_temporal_res_spin = QSpinBox()
        self.ct_temporal_res_spin.setRange(10, 2000)
        self.ct_temporal_res_spin.setSingleStep(10)
        self.ct_temporal_res_spin.setValue(200)
        self.ct_temporal_res_spin.setFixedWidth(70)
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

        self.ct_group.setVisible(False)
        controls_layout.addWidget(self.ct_group)

        main_layout.addWidget(controls_widget, stretch=0)

        # --- Status bar ---
        self.status_bar = QStatusBar()
        self.status_bar.showMessage("Ready")
        self.setStatusBar(self.status_bar)

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
        """Toggle CT controls visibility based on workflow selection."""
        self.ct_group.setVisible(text == "CT")

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

    def _load_and_overview(self) -> None:
        """Load .mat file and show overview grid of all channels x blocks."""
        if self.mat_filepath is None:
            QMessageBox.warning(self, "No File", "Please select a .mat file first.")
            return

        self._disconnect_click()
        self._time_markers.clear()
        self._trigger_times = np.array([])
        self._trigger_count = 0
        self.total_triggers_label.setText("--")

        self.status_bar.showMessage(f"Loading {self.mat_filepath.name}...")
        QApplication.processEvents()

        try:
            self.data = load_labchart_mat(self.mat_filepath)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))
            self.status_bar.showMessage("Error loading file")
            return

        nch = self.data.num_channels
        nbl = self.data.num_blocks

        # Update spinbox ranges to match actual data dimensions
        self.row_spin.setMaximum(nch)
        self.col_spin.setMaximum(nbl)
        # Clamp current values into valid range
        if self.row_spin.value() > nch:
            self.row_spin.setValue(nch)
        if self.col_spin.value() > nbl:
            self.col_spin.setValue(nbl)

        # Build overview grid
        self.fig.clear()
        self._plot_mode = "overview"
        self.ax_top = None
        self.ax_bottom = None

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

        try:
            self.fig.subplots_adjust(hspace=0.4, wspace=0.3)
            self.fig.tight_layout()
        except Exception:
            self.fig.subplots_adjust(hspace=0.4, wspace=0.3)

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

        row = self.row_spin.value()
        col = self.col_spin.value()

        # Bounds check
        if row < 1 or row > self.data.num_channels:
            QMessageBox.warning(
                self, "Invalid Row",
                f"Row must be between 1 and {self.data.num_channels}.",
            )
            return
        if col < 1 or col > self.data.num_blocks:
            QMessageBox.warning(
                self, "Invalid Column",
                f"Column must be between 1 and {self.data.num_blocks}.",
            )
            return

        if self.data.is_block_empty(row, col):
            QMessageBox.warning(self, "Empty Block", f"Block [{row}, {col}] is empty.")
            self.status_bar.showMessage("Block is empty")
            return

        # Clean up previous state
        self._disconnect_click()
        self._time_markers.clear()
        self._trigger_markers.clear()

        time = self.data.get_time(row, col)
        flow = self.data.get_data(row, col)

        # Recreate detail axes with shared x-axis
        self.fig.clear()
        self._plot_mode = "selected"

        self.ax_top = self.fig.add_subplot(211)
        self.ax_bottom = self.fig.add_subplot(212, sharex=self.ax_top)

        # Top subplot: flow + trigger overlay
        self.ax_top.plot(time, flow, "b-", linewidth=0.5, label="flow info")

        trigger_row = row - 1
        has_trigger = trigger_row >= 1 and not self.data.is_block_empty(trigger_row, col)
        is_ct = self.workflow_combo.currentText() == "CT"
        if has_trigger:
            trigger = self.data.get_data(trigger_row, col)
            trigger_time = self.data.get_time(trigger_row, col)
            self.ax_top.plot(trigger_time, trigger, "r-", linewidth=0.5, label="trigger info")

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
            f"{self.mat_filepath.name} — Block [{row}, {col}]: Flow + Trigger{trigger_info}"
        )
        self.ax_top.grid(True, alpha=0.3)
        self.ax_top.legend(loc="upper right", fontsize="small")

        # Bottom subplot: flow only (for zoom/selection)
        self.ax_bottom.plot(time, flow, "b-", linewidth=0.5)
        self.ax_bottom.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
        self.ax_bottom.set_xlabel("Time (s)")
        self.ax_bottom.set_ylabel("Flow (L/s)")
        self.ax_bottom.set_title("Flow Signal — left-click to place time marker, right-click to clear")
        self.ax_bottom.grid(True, alpha=0.3)
        self.ax_bottom.format_coord = lambda x, y: f"time={x:.3f}s, flow={y:.4f}"

        self.fig.tight_layout()
        self.canvas.draw()
        self.toolbar.update()

        # Connect click handler for data picker
        self._click_cid = self.canvas.mpl_connect("button_press_event", self._on_click)

        self.status_bar.showMessage(
            f"Block [{row}, {col}]: {len(time)} samples, {time[-1]:.2f}s{trigger_info} | "
            f"Left-click bottom plot to mark time, right-click to clear"
        )

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

        # Plot small markers on detected peaks
        if self._trigger_count > 0 and self.ax_top is not None:
            self.ax_top.plot(
                trigger_time[peaks], trigger_data[peaks],
                "rv", markersize=4, label=f"triggers ({self._trigger_count})",
            )
            self.ax_top.legend(loc="upper right", fontsize="small")

    def _detect_steps_from_data(self, trigger_data: np.ndarray, trigger_time: np.ndarray) -> None:
        """Detect step (bridge) triggers from raw data and update CT UI."""
        from labchart2cfd.processing.step_detection import detect_steps

        self._ct_steps = detect_steps(trigger_data, trigger_time, min_duration_s=1.0)

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
        if event.inaxes not in (self.ax_top, self.ax_bottom):
            return
        # Ignore clicks when toolbar is in zoom/pan mode
        if self.toolbar.mode:
            return

        if event.button == 1:  # Left click — add marker
            if self._ct_landmark_mode is not None:
                self._add_ct_landmark(event.xdata)
            else:
                self._add_time_marker(event.xdata)
        elif event.button == 3:  # Right click — clear all markers
            self._clear_time_markers()

    def _add_time_marker(self, x_time: float) -> None:
        """Add a vertical time marker on both top and bottom plots."""
        if self.ax_bottom is None or self.ax_top is None:
            return

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

    def _clear_time_markers(self) -> None:
        """Remove all time markers from both plots."""
        for artists in self._time_markers:
            for artist in artists:
                artist.remove()
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
        for artist in self._trigger_markers:
            artist.remove()
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

        self.ax_top.legend(loc="upper right", fontsize="small")
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
        for artist in self._trigger_markers:
            artist.remove()
        self._trigger_markers.clear()
        self._clear_ct_step_markers()
        self._clear_ct_interval_markers()
        self._clear_ct_landmarks()
        if self.ax_top is not None:
            self.ax_top.legend(loc="upper right", fontsize="small")
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

        row = self.row_spin.value()
        col = self.col_spin.value()
        trigger_row = row - 1

        if trigger_row < 1 or self.data.is_block_empty(trigger_row, col):
            QMessageBox.warning(self, "No Trigger", "No trigger data found in the row above the flow channel.")
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
        for artist in self._ct_step_markers:
            artist.remove()
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

        # Reset landmarks for new step
        self._clear_ct_landmarks()

        self.canvas.draw_idle()
        self.status_bar.showMessage(
            f"Step {step_num} selected ({step['duration']:.2f}s, "
            f"{self.ct_total_images_spin.value()} images) | "
            f"Use buttons to mark Inhale Start and Exhale End"
        )

    def _mark_inhale_start(self) -> None:
        """Activate click mode to place inhale start landmark."""
        self._ct_landmark_mode = "inhale_start"
        self.status_bar.showMessage("Click on plot to mark INHALE START")

    def _mark_exhale_end(self) -> None:
        """Activate click mode to place exhale end landmark."""
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

        if (manual_frames is not None and manual_frames > 0
                and self._ct_detected_start is not None and shift_ms != 0):
            # Manual correction mode: use shifted start + user-specified frame count
            start_t = self._ct_detected_start + shift_ms / 1000.0
            num_lines = manual_frames
        elif manual_frames is not None and manual_frames > 0 and self._ct_detected_start is not None:
            # Frame count override only (no shift)
            start_t = self._ct_detected_start
            num_lines = manual_frames
        else:
            # Auto mode: compute from step boundaries
            step = self._ct_selected_step
            start_t = step["start_time"]
            end_t = step["end_time"]
            num_lines = int(np.round((end_t - start_t) / temporal_res_s)) + 1

        # Draw fencepost lines: each line = 1 image
        for i in range(num_lines):
            t = start_t + i * temporal_res_s
            for ax in (self.ax_top, self.ax_bottom):
                line = ax.axvline(x=t, color="blue", linestyle=":", linewidth=0.7, alpha=0.6)
                self._ct_interval_markers.append(line)

            # Label AT each fencepost line (not between)
            img_idx = i + 1
            if img_idx <= 50:  # limit labels for readability
                ann = self.ax_top.annotate(
                    str(img_idx),
                    xy=(t, 0.02), xycoords=("data", "axes fraction"),
                    fontsize=5, color="blue", ha="center", va="bottom",
                )
                self._ct_interval_markers.append(ann)

        total_images = num_lines
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

        # Compute corrected boundaries
        corrected_start = self._ct_detected_start + shift_ms / 1000.0
        corrected_end = corrected_start + (manual_frames - 1) * temporal_res_s

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
        for artist in self._ct_interval_markers:
            artist.remove()
        self._ct_interval_markers.clear()

    def _clear_ct_landmarks(self) -> None:
        """Clear CT landmark markers and reset landmark mode."""
        for artist in self._ct_landmark_artists:
            artist.remove()
        self._ct_landmark_artists.clear()
        self._ct_landmark_mode = None
        self._ct_inhale_start = None
        self._ct_exhale_end = None
        self.ct_inhale_label.setText("--")
        self.ct_exhale_label.setText("--")
        self.canvas.draw_idle()

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
            self.status_bar.showMessage(f"Inhale start marked at {x_time:.3f}s")

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
                artist.remove()  # Annotation
            elif hasattr(artist, "get_label") and artist.get_label() == label:
                artist.remove()  # Line2D with label= kwarg
            else:
                remaining.append(artist)
        self._ct_landmark_artists = remaining

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

        row = self.row_spin.value()
        col = self.col_spin.value()
        workflow_name = self.workflow_combo.currentText()
        output_dir = Path(self.outdir_edit.text())

        # Capture trigger info from main thread for the background thread
        trigger_start = self.start_trigger_spin.value()
        trigger_num = self.num_triggers_spin.value()
        trigger_total = self._trigger_count

        # Capture CT-specific state
        ct_inhale_start = self._ct_inhale_start
        ct_exhale_end = self._ct_exhale_end
        ct_selected_step = self._ct_selected_step
        ct_temporal_res_ms = float(self.ct_temporal_res_spin.value())

        # Disable button during processing
        self.execute_btn.setEnabled(False)
        self.status_bar.showMessage("Processing...")
        QApplication.processEvents()

        def _run() -> None:
            try:
                from labchart2cfd.io.csv_export import export_flow_csv, export_pressure_csv
                from labchart2cfd.processing.rearrangement import time_to_image_index
                from labchart2cfd.workflows import (
                    StandardOSAMRIWorkflow,
                    CPAPWorkflow,
                    PhaseContrastWorkflow,
                    CTWorkflow,
                )

                # Select workflow
                if workflow_name == "CT":
                    wf = CTWorkflow(density=1.2)
                    result = wf.process(
                        self.data, row, col, start_time, end_time,
                        inhale_start_time=ct_inhale_start,
                        exhale_end_time=ct_exhale_end,
                        temporal_resolution=ct_temporal_res_ms / 1000.0,
                    )

                    # Determine selected images from inhale/exhale landmarks
                    # All indices are 1-based to match blue fencepost line labels
                    total_imgs = result.metadata.get("total_images", 0)

                    if ct_inhale_start is not None and ct_exhale_end is not None:
                        inhale_img = time_to_image_index(
                            ct_inhale_start, start_time, ct_temporal_res_ms / 1000.0
                        ) + 1  # 1-based
                        exhale_img = time_to_image_index(
                            ct_exhale_end, start_time, ct_temporal_res_ms / 1000.0
                        ) + 1  # 1-based

                        if inhale_img <= exhale_img:
                            # Normal: inhale LEFT of exhale -> straight sequence
                            selected_indices = list(range(inhale_img, exhale_img + 1))
                        else:
                            # Wrapping: exhale LEFT of inhale -> wrap around
                            selected_indices = (
                                list(range(inhale_img, total_imgs + 1))
                                + list(range(1, exhale_img + 1))
                            )
                    else:
                        # No landmarks -> use full range
                        inhale_img = 1
                        exhale_img = total_imgs
                        selected_indices = list(range(1, total_imgs + 1))

                    result.metadata["inhale_image"] = inhale_img
                    result.metadata["exhale_image"] = exhale_img
                    result.metadata["selected_num_images"] = len(selected_indices)
                    result.metadata["selected_image_indices"] = selected_indices
                elif workflow_name == "phase-contrast":
                    wf = PhaseContrastWorkflow(density=5.761)
                    result = wf.process(self.data, row, col, start_time, end_time)
                elif workflow_name == "cpap":
                    wf = CPAPWorkflow(density=1.2)
                    result = wf.process(self.data, row, col, start_time, end_time)
                else:
                    wf = StandardOSAMRIWorkflow(density=1.2)
                    result = wf.process(self.data, row, col, start_time, end_time)

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

                def _done():
                    self.status_bar.showMessage(msg)
                    self.execute_btn.setEnabled(True)

                QTimer.singleShot(0, _done)

            except Exception as e:
                error_msg = str(e)

                def _error():
                    QMessageBox.critical(self, "Processing Error", error_msg)
                    self.status_bar.showMessage(f"Error: {error_msg}")
                    self.execute_btn.setEnabled(True)

                QTimer.singleShot(0, _error)

        Thread(target=_run, daemon=True).start()


def launch_gui() -> None:
    """Launch the Flow Profile GUI application."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    window = FlowProfileApp()
    window.show()
    app.exec()
