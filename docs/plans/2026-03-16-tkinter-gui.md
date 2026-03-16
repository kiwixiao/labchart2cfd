# Flow Profile GUI Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Tkinter desktop GUI that loads LabChart .mat files, displays interactive flow signals with zoom/pan, accepts user-specified time windows, and executes the full processing pipeline to export Star-CCM+ compatible CSV files.

**Architecture:** Single new module `src/labchart2cfd/gui/app.py` containing a `FlowProfileApp` class that embeds a matplotlib canvas in a Tkinter window. The GUI reuses all existing modules (`io.labchart`, `workflows`, `io.csv_export`) without modification. A new CLI command `labchart2cfd gui` launches the app.

**Tech Stack:** Tkinter (stdlib), matplotlib (existing dependency — `FigureCanvasTkAgg`, `NavigationToolbar2Tk`), threading (stdlib for background processing)

---

## Chunk 1: GUI Module and CLI Entry Point

### Task 1: Create GUI module skeleton with file loading

**Files:**
- Create: `src/labchart2cfd/gui/__init__.py`
- Create: `src/labchart2cfd/gui/app.py`

- [ ] **Step 1: Create the gui package init**

```python
"""GUI module for interactive flow profile selection and processing."""
```

Write this to `src/labchart2cfd/gui/__init__.py`.

- [ ] **Step 2: Create the main GUI app file with window layout**

Create `src/labchart2cfd/gui/app.py` with:

```python
"""Tkinter GUI for interactive flow profile selection and processing.

Provides a desktop application that:
1. Loads LabChart .mat files via file browser
2. Displays flow signal with matplotlib zoom/pan toolbar
3. Accepts start/end time input for breath window selection
4. Runs the processing pipeline and exports CSV files
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from threading import Thread
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

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

        self._build_ui()

    def _build_ui(self) -> None:
        """Build the complete UI layout."""
        # Main container
        main_frame = ttk.Frame(self.root, padding=5)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Plot area (top, takes most space) ---
        plot_frame = ttk.LabelFrame(main_frame, text="Flow Signal", padding=5)
        plot_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        self.fig = Figure(figsize=(12, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Flow (L/s)")
        self.ax.set_title("Load a .mat file to begin")
        self.ax.grid(True, alpha=0.3)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()

        # Matplotlib navigation toolbar (zoom, pan, home, save)
        self.toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        self.toolbar.update()

        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # --- Controls area (bottom) ---
        controls_frame = ttk.Frame(main_frame)
        controls_frame.pack(fill=tk.X)

        # Row 1: File loading
        file_frame = ttk.LabelFrame(controls_frame, text="Data Source", padding=5)
        file_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(file_frame, text="File:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.file_var = tk.StringVar(value="No file selected")
        ttk.Label(file_frame, textvariable=self.file_var, width=60).grid(row=0, column=1, sticky=tk.W)
        ttk.Button(file_frame, text="Browse...", command=self._browse_file).grid(row=0, column=2, padx=5)

        ttk.Label(file_frame, text="Row:").grid(row=0, column=3, padx=(10, 5))
        self.row_var = tk.IntVar(value=2)
        ttk.Spinbox(file_frame, from_=1, to=20, textvariable=self.row_var, width=5).grid(row=0, column=4)

        ttk.Label(file_frame, text="Column:").grid(row=0, column=5, padx=(10, 5))
        self.col_var = tk.IntVar(value=3)
        ttk.Spinbox(file_frame, from_=1, to=20, textvariable=self.col_var, width=5).grid(row=0, column=6)

        ttk.Button(file_frame, text="Load & Plot", command=self._load_and_plot).grid(row=0, column=7, padx=(15, 0))

        # Row 2: Time window and processing
        proc_frame = ttk.LabelFrame(controls_frame, text="Processing", padding=5)
        proc_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(proc_frame, text="Start Time (s):").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.start_var = tk.StringVar(value="0.0")
        ttk.Entry(proc_frame, textvariable=self.start_var, width=12).grid(row=0, column=1)

        ttk.Label(proc_frame, text="End Time (s):").grid(row=0, column=2, padx=(10, 5))
        self.end_var = tk.StringVar(value="0.0")
        ttk.Entry(proc_frame, textvariable=self.end_var, width=12).grid(row=0, column=3)

        ttk.Label(proc_frame, text="Subject ID:").grid(row=0, column=4, padx=(10, 5))
        self.subject_var = tk.StringVar(value="")
        ttk.Entry(proc_frame, textvariable=self.subject_var, width=15).grid(row=0, column=5)

        ttk.Label(proc_frame, text="Workflow:").grid(row=0, column=6, padx=(10, 5))
        self.workflow_var = tk.StringVar(value="standard")
        workflow_combo = ttk.Combobox(
            proc_frame, textvariable=self.workflow_var,
            values=["standard", "cpap", "phase-contrast"],
            state="readonly", width=14,
        )
        workflow_combo.grid(row=0, column=7)

        ttk.Label(proc_frame, text="Output Dir:").grid(row=0, column=8, padx=(10, 5))
        self.outdir_var = tk.StringVar(value=str(Path.cwd()))
        ttk.Entry(proc_frame, textvariable=self.outdir_var, width=25).grid(row=0, column=9)
        ttk.Button(proc_frame, text="...", command=self._browse_outdir, width=3).grid(row=0, column=10)

        self.execute_btn = ttk.Button(proc_frame, text="Execute", command=self._execute_processing)
        self.execute_btn.grid(row=0, column=11, padx=(15, 0))

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, pady=(5, 0))

    def _browse_file(self) -> None:
        """Open file dialog to select a .mat file."""
        filepath = filedialog.askopenfilename(
            title="Select LabChart .mat file",
            filetypes=[("MAT files", "*.mat"), ("All files", "*.*")],
        )
        if filepath:
            self.mat_filepath = Path(filepath)
            self.file_var.set(str(self.mat_filepath))

    def _browse_outdir(self) -> None:
        """Open directory dialog to select output directory."""
        dirpath = filedialog.askdirectory(title="Select Output Directory")
        if dirpath:
            self.outdir_var.set(dirpath)

    def _load_and_plot(self) -> None:
        """Load the selected .mat file and plot the flow signal."""
        if self.mat_filepath is None:
            messagebox.showwarning("No File", "Please select a .mat file first.")
            return

        self.status_var.set(f"Loading {self.mat_filepath.name}...")
        self.root.update_idletasks()

        try:
            self.data = load_labchart_mat(self.mat_filepath)
        except Exception as e:
            messagebox.showerror("Load Error", str(e))
            self.status_var.set("Error loading file")
            return

        row = self.row_var.get()
        col = self.col_var.get()

        if self.data.is_block_empty(row, col):
            messagebox.showwarning("Empty Block", f"Block [{row}, {col}] is empty.")
            self.status_var.set("Block is empty")
            return

        time = self.data.get_time(row, col)
        flow = self.data.get_data(row, col)

        # Plot
        self.ax.clear()
        self.ax.plot(time, flow, "b-", linewidth=0.5)
        self.ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Flow (L/s)")
        self.ax.set_title(f"{self.mat_filepath.name} — Block [{row}, {col}]")
        self.ax.grid(True, alpha=0.3)
        self.ax.format_coord = lambda x, y: f"time={x:.3f}s, flow={y:.4f}"
        self.fig.tight_layout()
        self.canvas.draw()

        self.status_var.set(
            f"Loaded: {self.data.num_channels} channels, {self.data.num_blocks} blocks | "
            f"Block [{row}, {col}]: {len(time)} samples, {time[-1]:.2f}s"
        )

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

                # Update status on main thread
                def _done():
                    msg = f"Done! Saved: {flow_file}"
                    if pressure_file:
                        msg += f" and {pressure_file}"
                    msg += (
                        f" | Duration: {result.time[-1]:.2f}s"
                        f" | Rate: {result.sample_rate:.0f}Hz"
                        f" | Samples: {len(result.time)}"
                        f" | Drift: {result.drift_error:.6f}"
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
```

- [ ] **Step 3: Verify the file was created correctly**

Run: `python -c "from labchart2cfd.gui.app import FlowProfileApp; print('Import OK')"`

Expected: `Import OK` (on a system with display; may fail on headless)

- [ ] **Step 4: Commit**

```bash
git add src/labchart2cfd/gui/__init__.py src/labchart2cfd/gui/app.py
git commit -m "feat: add Tkinter GUI for interactive flow profile selection"
```

---

### Task 2: Add CLI entry point for the GUI

**Files:**
- Modify: `src/labchart2cfd/cli/main.py` (add `gui` command at end of file, before `if __name__`)

- [ ] **Step 1: Add the gui command to the CLI**

Add this command to `src/labchart2cfd/cli/main.py`, after the `version` command and before the `if __name__ == "__main__"` block:

```python
@app.command()
def gui() -> None:
    """Launch the interactive GUI for flow profile selection and processing.

    Opens a desktop window with:
    - File browser to load .mat files
    - Interactive plot with zoom/pan for identifying breath windows
    - Time window input and processing controls
    """
    from labchart2cfd.gui.app import launch_gui
    launch_gui()
```

- [ ] **Step 2: Verify the CLI command is registered**

Run: `python -m labchart2cfd.cli.main --help`

Expected: Output should list `gui` alongside `process`, `validate`, `visualize`, `version`

- [ ] **Step 3: Commit**

```bash
git add src/labchart2cfd/cli/main.py
git commit -m "feat: add 'gui' CLI command to launch desktop app"
```

---

### Task 3: Manual smoke test

- [ ] **Step 1: Launch the GUI**

Run: `cd /Users/xiaz9n/Dropbox/CCHMCProjects/MatlabFiles/convertFlowprofile/flowprofile_python && python -m labchart2cfd.cli.main gui`

Expected: A Tkinter window opens with:
- A blank matplotlib plot area at top with navigation toolbar
- Controls at bottom: Browse button, Row/Column spinboxes, Load & Plot button
- Processing row: Start/End time fields, Subject ID, Workflow dropdown, Output Dir, Execute button
- Status bar at bottom showing "Ready"

- [ ] **Step 2: Test loading a .mat file**

In the GUI:
1. Click "Browse..." → navigate to `/Users/xiaz9n/Dropbox/CCHMCProjects/MatlabFiles/convertFlowprofile/` → select a small `.mat` file (e.g., `OSAMRI_0029_07012021.mat`)
2. Set Row=2, Column=3
3. Click "Load & Plot"

Expected: Flow signal appears in the plot. Zoom/pan toolbar works. Status bar shows channel/block info.

- [ ] **Step 3: Test full processing**

In the GUI:
1. Use zoom to identify a breath window, note start/end times
2. Type start/end times into the text boxes
3. Enter Subject ID (e.g., "OSAMRI029_test")
4. Select workflow "standard"
5. Click "Execute"

Expected: Status bar shows "Processing..." then "Done! Saved: ..." with file path and summary stats.

- [ ] **Step 4: Verify output files**

Check that `OSAMRI029_testFlowProfile.csv` was created in the output directory with the correct Star-CCM+ format (quoted headers, numeric data).

- [ ] **Step 5: Commit (if any fixes were needed)**

```bash
git commit -am "fix: adjustments from GUI smoke test"
```
