"""Tkinter GUI for interactive flow profile selection and processing.

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

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from threading import Thread
from typing import Optional, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy.signal import find_peaks

from labchart2cfd.io.labchart import LabChartData, load_labchart_mat


class FlowProfileApp:
    """Main GUI application for flow profile processing."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("CFD Flow Profile Processor")
        self.root.geometry("1200x800")
        self.root.minsize(900, 600)

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

        self._build_ui()

    def _build_ui(self) -> None:
        """Build the complete UI layout."""
        # Main container
        main_frame = ttk.Frame(self.root, padding=5)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Controls area (bottom, packed FIRST so it always gets space) ---
        controls_frame = ttk.Frame(main_frame)
        controls_frame.pack(side=tk.BOTTOM, fill=tk.X)

        # --- Status bar (above controls packing order, below controls visually) ---
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=(5, 0))

        # --- Plot area (takes remaining space) ---
        plot_frame = ttk.LabelFrame(main_frame, text="Flow & Trigger Signals", padding=5)
        plot_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        self.fig = Figure(figsize=(12, 6), dpi=100)
        # Show placeholder text instead of pre-created axes
        self.fig.text(
            0.5, 0.5, "Load a .mat file and click 'Load Overview' to begin",
            ha="center", va="center", fontsize=14, color="gray",
        )

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()

        # Matplotlib navigation toolbar (zoom, pan, home, save)
        self.toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        self.toolbar.update()

        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Row 1: File loading — multi-row layout for responsiveness
        file_frame = ttk.LabelFrame(controls_frame, text="Data Source", padding=5)
        file_frame.pack(fill=tk.X, pady=(0, 5))

        # File frame Row 0: file path + browse
        ttk.Label(file_frame, text="File:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.file_var = tk.StringVar(value="No file selected")
        ttk.Label(file_frame, textvariable=self.file_var).grid(
            row=0, column=1, sticky=tk.EW, padx=(0, 5),
        )
        ttk.Button(file_frame, text="Browse...", command=self._browse_file).grid(
            row=0, column=2, sticky=tk.E,
        )
        file_frame.columnconfigure(1, weight=1)  # path label stretches

        # File frame Row 1: row/col spinboxes + Load Overview + Plot Selected
        ttk.Label(file_frame, text="Row:").grid(row=1, column=0, sticky=tk.W, padx=(0, 5), pady=(5, 0))

        spin_frame = ttk.Frame(file_frame)
        spin_frame.grid(row=1, column=1, sticky=tk.W, pady=(5, 0))

        self.row_var = tk.IntVar(value=2)
        self.row_spin = ttk.Spinbox(spin_frame, from_=1, to=20, textvariable=self.row_var, width=5)
        self.row_spin.pack(side=tk.LEFT)

        ttk.Label(spin_frame, text="  Column:").pack(side=tk.LEFT)
        self.col_var = tk.IntVar(value=3)
        self.col_spin = ttk.Spinbox(spin_frame, from_=1, to=20, textvariable=self.col_var, width=5)
        self.col_spin.pack(side=tk.LEFT)

        ttk.Button(spin_frame, text="Load Overview", command=self._load_and_overview).pack(
            side=tk.LEFT, padx=(15, 5),
        )
        ttk.Button(spin_frame, text="Plot Selected", command=self._plot_selected).pack(
            side=tk.LEFT, padx=(5, 0),
        )

        # --- Trigger range frame ---
        trigger_frame = ttk.LabelFrame(controls_frame, text="Trigger Info", padding=5)
        trigger_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(trigger_frame, text="Total Triggers:").pack(side=tk.LEFT, padx=(0, 5))
        self.total_triggers_var = tk.StringVar(value="--")
        ttk.Label(trigger_frame, textvariable=self.total_triggers_var, width=6,
                  relief=tk.SUNKEN, anchor=tk.CENTER).pack(side=tk.LEFT, padx=(0, 15))

        ttk.Label(trigger_frame, text="Start Trigger #:").pack(side=tk.LEFT, padx=(0, 5))
        self.start_trigger_var = tk.IntVar(value=1)
        self.start_trigger_spin = ttk.Spinbox(
            trigger_frame, from_=1, to=1, textvariable=self.start_trigger_var, width=5,
        )
        self.start_trigger_spin.pack(side=tk.LEFT, padx=(0, 15))

        ttk.Label(trigger_frame, text="# Triggers:").pack(side=tk.LEFT, padx=(0, 5))
        self.num_triggers_var = tk.IntVar(value=1)
        self.num_triggers_spin = ttk.Spinbox(
            trigger_frame, from_=1, to=1, textvariable=self.num_triggers_var, width=5,
        )
        self.num_triggers_spin.pack(side=tk.LEFT, padx=(0, 15))

        ttk.Button(trigger_frame, text="Mark Triggers", command=self._mark_trigger_range).pack(
            side=tk.LEFT, padx=(5, 5),
        )
        ttk.Button(trigger_frame, text="Clear Markers", command=self._clear_all_markers).pack(
            side=tk.LEFT, padx=(5, 0),
        )

        # Row 2: Time window and processing — multi-row layout
        proc_frame = ttk.LabelFrame(controls_frame, text="Processing", padding=5)
        proc_frame.pack(fill=tk.X, pady=(0, 5))

        # Proc Row 0: start/end time + subject ID
        ttk.Label(proc_frame, text="Start Time (s):").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.start_var = tk.StringVar(value="0.0")
        ttk.Entry(proc_frame, textvariable=self.start_var, width=12).grid(row=0, column=1)

        ttk.Label(proc_frame, text="End Time (s):").grid(row=0, column=2, padx=(10, 5))
        self.end_var = tk.StringVar(value="0.0")
        ttk.Entry(proc_frame, textvariable=self.end_var, width=12).grid(row=0, column=3)

        ttk.Label(proc_frame, text="Subject ID:").grid(row=0, column=4, padx=(10, 5))
        self.subject_var = tk.StringVar(value="")
        self.subject_var.trace_add("write", self._update_outdir)
        ttk.Entry(proc_frame, textvariable=self.subject_var, width=15).grid(row=0, column=5)

        # Proc Row 1: workflow + output dir + execute
        ttk.Label(proc_frame, text="Workflow:").grid(row=1, column=0, sticky=tk.W, padx=(0, 5), pady=(5, 0))
        self.workflow_var = tk.StringVar(value="MRI")
        workflow_combo = ttk.Combobox(
            proc_frame, textvariable=self.workflow_var,
            values=["MRI", "cpap", "phase-contrast"],
            state="readonly", width=14,
        )
        workflow_combo.grid(row=1, column=1, pady=(5, 0))

        ttk.Label(proc_frame, text="Output Dir:").grid(row=1, column=2, padx=(10, 5), pady=(5, 0))
        self.outdir_var = tk.StringVar(value=str(Path.cwd()))
        ttk.Entry(proc_frame, textvariable=self.outdir_var).grid(
            row=1, column=3, columnspan=2, sticky=tk.EW, pady=(5, 0),
        )
        proc_frame.columnconfigure(3, weight=1)  # output dir stretches

        ttk.Button(proc_frame, text="...", command=self._browse_outdir, width=3).grid(
            row=1, column=5, padx=(5, 0), pady=(5, 0),
        )

        self.execute_btn = ttk.Button(proc_frame, text="Execute", command=self._execute_processing)
        self.execute_btn.grid(row=1, column=6, padx=(15, 0), pady=(5, 0))

    def _update_outdir(self, *_args) -> None:
        """Recompute output dir based on current mat file and subject."""
        if not hasattr(self, "mat_filepath") or self.mat_filepath is None:
            return
        subject = self.subject_var.get().strip()
        if subject:
            results_dir = self.mat_filepath.parent / f"FlowResults_{subject}"
        else:
            results_dir = self.mat_filepath.parent / "FlowResults"
        self.outdir_var.set(str(results_dir))

    def _browse_file(self) -> None:
        """Open file dialog to select a .mat file."""
        filepath = filedialog.askopenfilename(
            title="Select LabChart .mat file",
            filetypes=[("MAT files", "*.mat"), ("All files", "*.*")],
        )
        if filepath:
            self.mat_filepath = Path(filepath)
            self.file_var.set(str(self.mat_filepath))
            self._update_outdir()

    def _browse_outdir(self) -> None:
        """Open directory dialog to select output directory."""
        dirpath = filedialog.askdirectory(title="Select Output Directory")
        if dirpath:
            self.outdir_var.set(dirpath)

    def _disconnect_click(self) -> None:
        """Disconnect canvas click callback if connected."""
        if self._click_cid is not None:
            self.canvas.mpl_disconnect(self._click_cid)
            self._click_cid = None

    def _load_and_overview(self) -> None:
        """Load .mat file and show overview grid of all channels x blocks."""
        if self.mat_filepath is None:
            messagebox.showwarning("No File", "Please select a .mat file first.")
            return

        self._disconnect_click()
        self._time_markers.clear()
        self._trigger_times = np.array([])
        self._trigger_count = 0
        self.total_triggers_var.set("--")

        self.status_var.set(f"Loading {self.mat_filepath.name}...")
        self.root.update_idletasks()

        try:
            self.data = load_labchart_mat(self.mat_filepath)
        except Exception as e:
            messagebox.showerror("Load Error", str(e))
            self.status_var.set("Error loading file")
            return

        nch = self.data.num_channels
        nbl = self.data.num_blocks

        # Update spinbox ranges to match actual data dimensions
        self.row_spin.configure(to=nch)
        self.col_spin.configure(to=nbl)
        # Clamp current values into valid range
        if self.row_var.get() > nch:
            self.row_var.set(nch)
        if self.col_var.get() > nbl:
            self.col_var.set(nbl)

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

        self.status_var.set(
            f"Overview: {nch} channels x {nbl} blocks | "
            f"Select row/col and click 'Plot Selected' for detail"
        )

    def _plot_selected(self) -> None:
        """Plot a specific block in detail (flow + trigger top, flow-only bottom)."""
        if self.data is None:
            messagebox.showwarning("No Data", "Please load a .mat file first (click 'Load Overview').")
            return

        row = self.row_var.get()
        col = self.col_var.get()

        # Bounds check
        if row < 1 or row > self.data.num_channels:
            messagebox.showwarning(
                "Invalid Row",
                f"Row must be between 1 and {self.data.num_channels}.",
            )
            return
        if col < 1 or col > self.data.num_blocks:
            messagebox.showwarning(
                "Invalid Column",
                f"Column must be between 1 and {self.data.num_blocks}.",
            )
            return

        if self.data.is_block_empty(row, col):
            messagebox.showwarning("Empty Block", f"Block [{row}, {col}] is empty.")
            self.status_var.set("Block is empty")
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
        if has_trigger:
            trigger = self.data.get_data(trigger_row, col)
            trigger_time = self.data.get_time(trigger_row, col)
            self.ax_top.plot(trigger_time, trigger, "r-", linewidth=0.5, label="trigger info")

            # Detect trigger pulses
            self._detect_triggers(trigger, trigger_time)

        self.ax_top.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
        self.ax_top.set_xlabel("Time (s)")
        self.ax_top.set_ylabel("Amplitude")

        trigger_info = f" | {self._trigger_count} triggers" if has_trigger else ""
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

        self.status_var.set(
            f"Block [{row}, {col}]: {len(time)} samples, {time[-1]:.2f}s{trigger_info} | "
            f"Left-click bottom plot to mark time, right-click to clear"
        )

    def _detect_triggers(self, trigger_data: np.ndarray, trigger_time: np.ndarray) -> None:
        """Detect trigger pulses and update UI."""
        if len(trigger_data) == 0:
            self._trigger_times = np.array([])
            self._trigger_count = 0
            self.total_triggers_var.set("0")
            return

        # Find peaks: height at half-max, minimum distance between peaks
        max_val = np.max(np.abs(trigger_data))
        if max_val == 0:
            self._trigger_times = np.array([])
            self._trigger_count = 0
            self.total_triggers_var.set("0")
            return

        height_threshold = 0.5 * max_val
        min_distance = max(10, len(trigger_data) // 2000)

        peaks, _ = find_peaks(trigger_data, height=height_threshold, distance=min_distance)

        self._trigger_times = trigger_time[peaks]
        self._trigger_count = len(peaks)

        # Update trigger UI
        self.total_triggers_var.set(str(self._trigger_count))
        if self._trigger_count > 0:
            self.start_trigger_spin.configure(to=self._trigger_count)
            self.num_triggers_spin.configure(to=self._trigger_count)
            # Clamp values
            if self.start_trigger_var.get() > self._trigger_count:
                self.start_trigger_var.set(1)
            if self.num_triggers_var.get() > self._trigger_count:
                self.num_triggers_var.set(1)

        # Plot small markers on detected peaks
        if self._trigger_count > 0 and self.ax_top is not None:
            self.ax_top.plot(
                trigger_time[peaks], trigger_data[peaks],
                "rv", markersize=4, label=f"triggers ({self._trigger_count})",
            )
            self.ax_top.legend(loc="upper right", fontsize="small")

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
            self.start_var.set(f"{x_time:.3f}")
        elif marker_count == 2:
            self.end_var.set(f"{x_time:.3f}")
            # Ensure start < end
            try:
                s = float(self.start_var.get())
                e = float(self.end_var.get())
                if e < s:
                    self.start_var.set(f"{e:.3f}")
                    self.end_var.set(f"{s:.3f}")
            except ValueError:
                pass

        self.canvas.draw_idle()
        self.status_var.set(
            f"Marker {marker_count} at {x_time:.3f}s | "
            f"Start={self.start_var.get()}s, End={self.end_var.get()}s"
        )

    def _clear_time_markers(self) -> None:
        """Remove all time markers from both plots."""
        for artists in self._time_markers:
            for artist in artists:
                artist.remove()
        self._time_markers.clear()
        self.canvas.draw_idle()
        self.status_var.set("Time markers cleared")

    def _mark_trigger_range(self) -> None:
        """Highlight selected trigger range on the top plot."""
        if self._trigger_count == 0:
            messagebox.showwarning("No Triggers", "No triggers detected. Load a block with trigger data first.")
            return
        if self.ax_top is None:
            messagebox.showwarning("No Plot", "Please click 'Plot Selected' first.")
            return

        # Clear previous trigger range markers
        for artist in self._trigger_markers:
            artist.remove()
        self._trigger_markers.clear()

        start_idx = self.start_trigger_var.get()  # 1-indexed
        num_triggers = self.num_triggers_var.get()

        # Validate range
        if start_idx < 1 or start_idx > self._trigger_count:
            messagebox.showwarning("Invalid Start", f"Start trigger must be 1-{self._trigger_count}.")
            return
        end_idx = start_idx + num_triggers - 1
        if end_idx > self._trigger_count:
            messagebox.showwarning(
                "Out of Range",
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
        self.start_var.set(f"{t_start:.3f}")
        self.end_var.set(f"{t_end:.3f}")

        self.status_var.set(
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
        if self.ax_top is not None:
            self.ax_top.legend(loc="upper right", fontsize="small")
        self.canvas.draw_idle()
        self.status_var.set("All markers cleared")

    def _execute_processing(self) -> None:
        """Run the processing pipeline in a background thread."""
        # Validate inputs
        if self.data is None:
            messagebox.showwarning("No Data", "Please load a .mat file first.")
            return

        subject = self.subject_var.get().strip()
        if not subject:
            messagebox.showwarning("Missing Subject", "Please enter a Subject ID.")
            return

        try:
            start_time = float(self.start_var.get())
            end_time = float(self.end_var.get())
        except ValueError:
            messagebox.showerror("Invalid Times", "Start and End times must be numbers.")
            return

        if end_time <= start_time:
            messagebox.showerror("Invalid Times", "End time must be greater than start time.")
            return

        row = self.row_var.get()
        col = self.col_var.get()
        workflow_name = self.workflow_var.get()
        output_dir = Path(self.outdir_var.get())

        # Capture trigger info from main thread for the background thread
        trigger_start = self.start_trigger_var.get()
        trigger_num = self.num_triggers_var.get()
        trigger_total = self._trigger_count

        # Disable button during processing
        self.execute_btn.config(state=tk.DISABLED)
        self.status_var.set("Processing...")
        self.root.update_idletasks()

        def _run() -> None:
            try:
                from labchart2cfd.io.csv_export import export_flow_csv, export_pressure_csv
                from labchart2cfd.workflows import (
                    StandardOSAMRIWorkflow,
                    CPAPWorkflow,
                    PhaseContrastWorkflow,
                )

                # Select workflow
                if workflow_name == "phase-contrast":
                    wf = PhaseContrastWorkflow(density=5.761)
                elif workflow_name == "cpap":
                    wf = CPAPWorkflow(density=1.2)
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
                end_trigger = trigger_start + trigger_num - 1
                exported_duration = result.time[-1] if len(result.time) > 0 else 0.0
                with open(str(trigger_file), "w") as tf:
                    tf.write(f"start_trigger: {trigger_start}\n")
                    tf.write(f"num_triggers: {trigger_num}\n")
                    tf.write(f"end_trigger: {end_trigger}\n")
                    tf.write(f"total_triggers_in_block: {trigger_total}\n")
                    tf.write(f"start_time_original: {start_time:.3f}\n")
                    tf.write(f"end_time_original: {end_time:.3f}\n")
                    tf.write(f"start_time_exported: 0.000\n")
                    tf.write(f"end_time_exported: {exported_duration:.3f}\n")

                # Generate sanity-check PNG plot of exported data
                import matplotlib
                matplotlib.use("Agg")  # non-interactive backend for file saving
                import matplotlib.pyplot as plt

                plot_file = output_dir / f"{subject}_sanity_check.png"
                n_plots = 2 if result.pressure is not None else 1
                fig_export, axes = plt.subplots(n_plots, 1, figsize=(10, 3 * n_plots), dpi=150)
                if n_plots == 1:
                    axes = [axes]

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
                plt.close(fig_export)

                # Restore TkAgg backend for the GUI
                matplotlib.use("TkAgg")

                # Update status on main thread
                def _done():
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
                    self.status_var.set(msg)
                    self.execute_btn.config(state=tk.NORMAL)

                self.root.after(0, _done)

            except Exception as e:
                def _error():
                    messagebox.showerror("Processing Error", str(e))
                    self.status_var.set(f"Error: {e}")
                    self.execute_btn.config(state=tk.NORMAL)

                self.root.after(0, _error)

        Thread(target=_run, daemon=True).start()


def launch_gui() -> None:
    """Launch the Flow Profile GUI application."""
    root = tk.Tk()
    FlowProfileApp(root)
    root.mainloop()
