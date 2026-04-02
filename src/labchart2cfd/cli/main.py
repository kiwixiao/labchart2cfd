"""CLI interface for labchart2cfd.

Provides commands for processing, validating, and visualizing flow profile data.
"""

from enum import Enum
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="labchart2cfd",
    help="Convert LabChart .mat flow profile data to CSV for CFD simulation boundary conditions",
    no_args_is_help=True,
)
console = Console()


class WorkflowType(str, Enum):
    """Available workflow types."""
    STANDARD = "standard"
    CPAP = "cpap"
    PHASE_CONTRAST = "phase-contrast"
    CT = "ct"


@app.command()
def process(
    input_file: Annotated[
        Path,
        typer.Argument(help="Path to LabChart .mat file"),
    ],
    subject: Annotated[
        str,
        typer.Argument(help="Subject identifier (e.g., OSAMRI029)"),
    ],
    row: Annotated[
        int,
        typer.Option("--row", "-r", help="Flow data row (1-indexed)"),
    ] = 2,
    column: Annotated[
        int,
        typer.Option("--column", "-c", help="Block column (1-indexed)"),
    ] = 3,
    start: Annotated[
        float,
        typer.Option("--start", "-s", help="Start time in seconds"),
    ] = 0.0,
    end: Annotated[
        float,
        typer.Option("--end", "-e", help="End time in seconds"),
    ] = 0.0,
    workflow: Annotated[
        WorkflowType,
        typer.Option("--workflow", "-w", help="Processing workflow type"),
    ] = WorkflowType.STANDARD,
    output_dir: Annotated[
        Optional[Path],
        typer.Option("--output-dir", "-o", help="Output directory"),
    ] = None,
    no_pressure: Annotated[
        bool,
        typer.Option("--no-pressure", help="Skip pressure processing"),
    ] = False,
    bag_id: Annotated[
        Optional[str],
        typer.Option("--bag", help="Xenon bag ID (Bag1/Bag2/Bag4) for phase-contrast"),
    ] = None,
    density: Annotated[
        Optional[float],
        typer.Option("--density", help="Override gas density (kg/m³)"),
    ] = None,
    sample_rate: Annotated[
        float,
        typer.Option("--sample-rate", help="Target sample rate (Hz)"),
    ] = 100.0,
    inhale_start: Annotated[
        Optional[float],
        typer.Option("--inhale-start", help="CT: inhale start time in seconds"),
    ] = None,
    exhale_end: Annotated[
        Optional[float],
        typer.Option("--exhale-end", help="CT: exhale end time in seconds"),
    ] = None,
    temporal_resolution: Annotated[
        float,
        typer.Option("--temporal-resolution", help="CT: temporal resolution in ms (default 200)"),
    ] = 200.0,
    step_number: Annotated[
        Optional[int],
        typer.Option("--step-number", help="CT: step trigger number to use (1-indexed)"),
    ] = None,
) -> None:
    """Process a LabChart .mat file and export Star-CCM+ compatible CSV.

    Example:
        labchart2cfd process input.mat OSAMRI029 -s 12.97 -e 16.16
        labchart2cfd process input.mat CT001 -w ct --step-number 1 --inhale-start 5.2 --temporal-resolution 200
    """
    from labchart2cfd.io.labchart import load_labchart_mat
    from labchart2cfd.io.csv_export import export_flow_csv, export_pressure_csv
    from labchart2cfd.workflows import (
        StandardOSAMRIWorkflow,
        CPAPWorkflow,
        PhaseContrastWorkflow,
        CTWorkflow,
    )

    # Validate input file
    if not input_file.exists():
        console.print(f"[red]Error:[/red] File not found: {input_file}")
        raise typer.Exit(1)

    # Validate times
    if end <= start and end != 0.0:
        console.print("[red]Error:[/red] End time must be greater than start time")
        raise typer.Exit(1)

    # Set output directory
    if output_dir is None:
        output_dir = Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    console.print(f"Loading [cyan]{input_file}[/cyan]...")
    try:
        data = load_labchart_mat(input_file)
    except Exception as e:
        console.print(f"[red]Error loading file:[/red] {e}")
        raise typer.Exit(1)

    console.print(f"  Loaded: {data.num_channels} channels, {data.num_blocks} blocks")

    # Validate block exists
    if data.is_block_empty(row, column):
        console.print(f"[red]Error:[/red] Block [{row}, {column}] is empty")
        raise typer.Exit(1)

    # Auto-detect end time if not specified
    if end == 0.0:
        time_array = data.get_time(row, column)
        end = float(time_array[-1])
        console.print(f"  Using full time range: {start:.2f} - {end:.2f}s")

    # Select and configure workflow
    if workflow == WorkflowType.CT:
        if density is None:
            density = 1.2  # Air
        wf = CTWorkflow(
            target_sample_rate=sample_rate,
            density=density,
        )
        # If step_number is provided, auto-detect step boundaries
        if step_number is not None:
            from labchart2cfd.processing.step_detection import detect_steps
            trigger_row = row - 1
            if trigger_row < 1 or data.is_block_empty(trigger_row, column):
                console.print("[red]Error:[/red] No trigger channel found (expected row above flow channel)")
                raise typer.Exit(1)
            trigger_data = data.get_data(trigger_row, column)
            trigger_time = data.get_time(trigger_row, column)
            steps = detect_steps(trigger_data, trigger_time)
            if step_number < 1 or step_number > len(steps):
                console.print(f"[red]Error:[/red] Step {step_number} not found. Detected {len(steps)} step(s).")
                raise typer.Exit(1)
            selected_step = steps[step_number - 1]
            start = selected_step["start_time"]
            end = selected_step["end_time"]
            console.print(f"  Using step {step_number}: {start:.3f}s - {end:.3f}s ({selected_step['duration']:.2f}s)")

    elif workflow == WorkflowType.PHASE_CONTRAST:
        if density is None:
            density = 5.761  # Xenon
        target_rate = 1000.0 if sample_rate == 100.0 else sample_rate
        wf = PhaseContrastWorkflow(
            target_sample_rate=target_rate,
            density=density,
        )
    elif workflow == WorkflowType.CPAP:
        if density is None:
            density = 1.2  # Air
        wf = CPAPWorkflow(
            target_sample_rate=sample_rate,
            density=density,
        )
    else:  # STANDARD
        if density is None:
            density = 1.2  # Air
        wf = StandardOSAMRIWorkflow(
            target_sample_rate=sample_rate,
            density=density,
        )

    console.print(f"Using [green]{wf.name}[/green] workflow")

    # Process
    try:
        if workflow == WorkflowType.CT:
            result = wf.process(
                data, row, column, start, end,
                inhale_start_time=inhale_start,
                exhale_end_time=exhale_end,
                temporal_resolution=temporal_resolution / 1000.0,
                include_pressure=not no_pressure,
            )
        elif workflow == WorkflowType.PHASE_CONTRAST and bag_id:
            result = wf.process_with_bag_config(data, row, column, bag_id)
        else:
            result = wf.process(
                data, row, column, start, end,
                include_pressure=not no_pressure,
            )
    except Exception as e:
        console.print(f"[red]Error during processing:[/red] {e}")
        raise typer.Exit(1)

    # Export flow CSV
    flow_file = output_dir / f"{subject}FlowProfile.csv"
    export_flow_csv(flow_file, result.time, result.mass_flow)
    console.print(f"[green]Created:[/green] {flow_file}")

    # Export pressure CSV if available
    if result.pressure is not None:
        pressure_file = output_dir / f"{subject}PressureProfile.csv"
        export_pressure_csv(pressure_file, result.time, result.pressure)
        console.print(f"[green]Created:[/green] {pressure_file}")

    # Summary
    console.print("\n[bold]Processing Summary:[/bold]")
    table = Table(show_header=False)
    table.add_column("Property", style="cyan")
    table.add_column("Value")
    table.add_row("Time range", f"{result.original_start_time:.2f} - {result.original_end_time:.2f} s")
    table.add_row("Duration", f"{result.time[-1]:.2f} s")
    table.add_row("Sample rate", f"{result.sample_rate:.0f} Hz")
    table.add_row("Samples", str(len(result.time)))
    table.add_row("Drift error", f"{result.drift_error:.6f}")
    console.print(table)


@app.command()
def validate(
    input_file: Annotated[
        Path,
        typer.Argument(help="Path to LabChart .mat file"),
    ],
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show detailed block information"),
    ] = False,
) -> None:
    """Validate and describe a LabChart .mat file structure.

    Shows channels, blocks, sample rates, and data availability.
    """
    from labchart2cfd.io.labchart import describe_mat_structure

    if not input_file.exists():
        console.print(f"[red]Error:[/red] File not found: {input_file}")
        raise typer.Exit(1)

    console.print(f"Analyzing [cyan]{input_file}[/cyan]...")

    try:
        info = describe_mat_structure(input_file)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    if "error" in info:
        console.print(f"[red]Error:[/red] {info['error']}")
        raise typer.Exit(1)

    # Summary table
    console.print("\n[bold]File Structure:[/bold]")
    table = Table(show_header=False)
    table.add_column("Property", style="cyan")
    table.add_column("Value")
    table.add_row("Channels", str(info["num_channels"]))
    table.add_row("Blocks", str(info["num_blocks"]))
    table.add_row("Total samples", f"{info['total_samples']:,}")
    console.print(table)

    # Channel titles
    if info["titles"]:
        console.print("\n[bold]Channel Titles:[/bold]")
        for i, title in enumerate(info["titles"], 1):
            console.print(f"  {i}. {title}")

    # Block occupancy grid
    console.print("\n[bold]Block Occupancy:[/bold] (✓ = data, · = empty)")
    for ch in range(info["num_channels"]):
        row_str = f"  Ch{ch+1}: "
        for bl in range(info["num_blocks"]):
            block = next(
                (b for b in info["blocks"]
                 if b["channel"] == ch + 1 and b["block"] == bl + 1),
                None
            )
            if block and not block["empty"]:
                row_str += "✓ "
            else:
                row_str += "· "
        console.print(row_str)

    # Verbose: show block details
    if verbose:
        console.print("\n[bold]Block Details:[/bold]")
        table = Table()
        table.add_column("Ch", justify="right")
        table.add_column("Block", justify="right")
        table.add_column("Samples", justify="right")
        table.add_column("Rate (Hz)", justify="right")
        table.add_column("Duration (s)", justify="right")

        for block in info["blocks"]:
            if not block["empty"]:
                table.add_row(
                    str(block["channel"]),
                    str(block["block"]),
                    f"{block['samples']:,}",
                    f"{block['sample_rate']:.1f}",
                    f"{block['duration_s']:.2f}",
                )
        console.print(table)


@app.command()
def visualize(
    input_file: Annotated[
        Path,
        typer.Argument(help="Path to LabChart .mat file"),
    ],
    row: Annotated[
        int,
        typer.Option("--row", "-r", help="Flow data row (1-indexed)"),
    ] = 2,
    column: Annotated[
        int,
        typer.Option("--column", "-c", help="Block column (1-indexed)"),
    ] = 3,
    save: Annotated[
        Optional[Path],
        typer.Option("--save", "-s", help="Save plot to file instead of displaying"),
    ] = None,
) -> None:
    """Visualize flow data to help identify start/end times.

    Displays an interactive plot of the flow signal.
    """
    from labchart2cfd.io.labchart import load_labchart_mat
    from labchart2cfd.visualization.plots import plot_flow_signal

    if not input_file.exists():
        console.print(f"[red]Error:[/red] File not found: {input_file}")
        raise typer.Exit(1)

    console.print(f"Loading [cyan]{input_file}[/cyan]...")
    data = load_labchart_mat(input_file)

    if data.is_block_empty(row, column):
        console.print(f"[red]Error:[/red] Block [{row}, {column}] is empty")
        raise typer.Exit(1)

    time = data.get_time(row, column)
    flow = data.get_data(row, column)

    console.print(f"Plotting block [{row}, {column}]: {len(time)} samples")

    plot_flow_signal(time, flow, title=f"Block [{row}, {column}]", save_path=save)

    if save:
        console.print(f"[green]Saved:[/green] {save}")


@app.command(name="detect-steps")
def detect_steps_cmd(
    input_file: Annotated[
        Path,
        typer.Argument(help="Path to LabChart .mat file"),
    ],
    row: Annotated[
        int,
        typer.Option("--row", "-r", help="Flow data row (1-indexed, trigger is row-1)"),
    ] = 2,
    column: Annotated[
        int,
        typer.Option("--column", "-c", help="Block column (1-indexed)"),
    ] = 3,
    threshold: Annotated[
        float,
        typer.Option("--threshold", help="Threshold fraction of max amplitude (0-1)"),
    ] = 0.5,
    min_duration: Annotated[
        float,
        typer.Option("--min-duration", help="Minimum step duration in seconds"),
    ] = 0.5,
) -> None:
    """Detect step triggers in a LabChart .mat file (for CT workflow).

    Reports each detected step with start/end times and duration.
    """
    from labchart2cfd.io.labchart import load_labchart_mat
    from labchart2cfd.processing.step_detection import detect_steps

    if not input_file.exists():
        console.print(f"[red]Error:[/red] File not found: {input_file}")
        raise typer.Exit(1)

    console.print(f"Loading [cyan]{input_file}[/cyan]...")
    data = load_labchart_mat(input_file)

    trigger_row = row - 1
    if trigger_row < 1 or data.is_block_empty(trigger_row, column):
        console.print(f"[red]Error:[/red] No trigger data at row {trigger_row}, column {column}")
        raise typer.Exit(1)

    trigger_data = data.get_data(trigger_row, column)
    trigger_time = data.get_time(trigger_row, column)

    steps = detect_steps(trigger_data, trigger_time, threshold, min_duration)

    if not steps:
        console.print("[yellow]No step triggers detected.[/yellow]")
        return

    console.print(f"\n[bold]Detected {len(steps)} step trigger(s):[/bold]")
    table = Table()
    table.add_column("Step #", justify="right")
    table.add_column("Start (s)", justify="right")
    table.add_column("End (s)", justify="right")
    table.add_column("Duration (s)", justify="right")

    for step in steps:
        table.add_row(
            str(step["index"]),
            f"{step['start_time']:.3f}",
            f"{step['end_time']:.3f}",
            f"{step['duration']:.2f}",
        )
    console.print(table)


@app.command()
def version() -> None:
    """Show version information."""
    from labchart2cfd import __version__
    console.print(f"labchart2cfd version {__version__}")


@app.command()
def gui() -> None:
    """Launch the interactive GUI for flow profile selection and processing.

    Opens a desktop window with:
    - File browser to load .mat files
    - Interactive plot with zoom/pan for identifying breath windows
    - Time window input and processing controls
    """
    import sys
    import os

    # macOS: must launch via 'open python.app' so macOS treats it as a real
    # GUI app with keyboard focus. Use env var to prevent infinite relaunch.
    if (sys.platform == "darwin"
            and os.environ.get("LABCHART2CFD_GUI") != "1"):
        env_dir = os.path.dirname(os.path.dirname(sys.executable))
        python_app = os.path.join(env_dir, "python.app")
        if os.path.isdir(python_app):
            os.execlp(
                "open", "open",
                "--env", "LABCHART2CFD_GUI=1",
                python_app, "--args", "-m", "labchart2cfd", "gui",
            )

    from labchart2cfd.gui.app import launch_gui
    launch_gui()


if __name__ == "__main__":
    app()
