"""Plotting functions for flow profile visualization.

Provides matplotlib-based plotting for analyzing and validating flow data.
"""

from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

from labchart2cfd.workflows.base import WorkflowResult


def plot_flow_signal(
    time: NDArray[np.float64],
    flow: NDArray[np.float64],
    title: str = "Flow Signal",
    xlabel: str = "Time (s)",
    ylabel: str = "Flow (L/s)",
    save_path: Optional[Path] = None,
    figsize: tuple[float, float] = (12, 6),
) -> None:
    """Plot a flow signal with time axis.

    Useful for identifying start/end times for extraction.

    Args:
        time: Time array in seconds
        flow: Flow data array
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: Path to save figure (if None, displays interactively)
        figsize: Figure size (width, height) in inches
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(time, flow, "b-", linewidth=0.5)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Add interactive cursor info
    ax.format_coord = lambda x, y: f"time={x:.3f}s, flow={y:.4f}"

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_processing_summary(
    result: WorkflowResult,
    title_prefix: str = "",
    save_path: Optional[Path] = None,
    figsize: tuple[float, float] = (12, 8),
) -> None:
    """Plot a summary of processing results.

    Shows mass flow rate and pressure (if available) after processing.

    Args:
        result: WorkflowResult from a workflow
        title_prefix: Prefix for plot titles
        save_path: Path to save figure
        figsize: Figure size (width, height) in inches
    """
    has_pressure = result.pressure is not None
    nrows = 2 if has_pressure else 1

    fig, axes = plt.subplots(nrows, 1, figsize=figsize, sharex=True)
    if nrows == 1:
        axes = [axes]

    # Mass flow rate plot
    ax_flow = axes[0]
    ax_flow.plot(result.time, result.mass_flow, "b-", linewidth=0.8)
    ax_flow.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    ax_flow.set_ylabel("Mass Flow Rate (kg/s)")
    ax_flow.set_title(f"{title_prefix}Mass Flow Rate")
    ax_flow.grid(True, alpha=0.3)

    # Pressure plot
    if has_pressure:
        ax_pressure = axes[1]
        ax_pressure.plot(result.time, result.pressure, "r-", linewidth=0.8)
        ax_pressure.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
        ax_pressure.set_ylabel("Pressure (Pa)")
        ax_pressure.set_title(f"{title_prefix}Pressure Profile")
        ax_pressure.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")

    # Add metadata annotation
    metadata_text = (
        f"Sample rate: {result.sample_rate:.0f} Hz\n"
        f"Drift error: {result.drift_error:.6f}\n"
        f"Original window: {result.original_start_time:.2f} - {result.original_end_time:.2f}s"
    )
    fig.text(0.02, 0.02, metadata_text, fontsize=8, family="monospace",
             verticalalignment="bottom", alpha=0.7)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_comparison(
    time1: NDArray[np.float64],
    data1: NDArray[np.float64],
    time2: NDArray[np.float64],
    data2: NDArray[np.float64],
    label1: str = "Python",
    label2: str = "MATLAB",
    title: str = "Comparison",
    ylabel: str = "Value",
    save_path: Optional[Path] = None,
    figsize: tuple[float, float] = (12, 8),
) -> None:
    """Plot comparison between two datasets.

    Useful for validating Python output against MATLAB reference.

    Args:
        time1: First time array
        data1: First data array
        time2: Second time array
        data2: Second data array
        label1: Label for first dataset
        label2: Label for second dataset
        title: Plot title
        ylabel: Y-axis label
        save_path: Path to save figure
        figsize: Figure size (width, height) in inches
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])

    # Main comparison plot
    ax_main = axes[0]
    ax_main.plot(time1, data1, "b-", linewidth=0.8, label=label1, alpha=0.8)
    ax_main.plot(time2, data2, "r--", linewidth=0.8, label=label2, alpha=0.8)
    ax_main.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    ax_main.set_ylabel(ylabel)
    ax_main.set_title(title)
    ax_main.legend()
    ax_main.grid(True, alpha=0.3)

    # Difference plot (if same length, interpolate otherwise)
    ax_diff = axes[1]
    if len(time1) == len(time2):
        diff = data1 - data2
        ax_diff.plot(time1, diff, "g-", linewidth=0.8)
        max_diff = np.max(np.abs(diff))
        rmse = np.sqrt(np.mean(diff**2))
        ax_diff.set_title(f"Difference (Max: {max_diff:.2e}, RMSE: {rmse:.2e})")
    else:
        # Interpolate to common time base
        from scipy.interpolate import CubicSpline
        common_time = np.linspace(
            max(time1[0], time2[0]),
            min(time1[-1], time2[-1]),
            min(len(time1), len(time2))
        )
        interp1 = CubicSpline(time1, data1)(common_time)
        interp2 = CubicSpline(time2, data2)(common_time)
        diff = interp1 - interp2
        ax_diff.plot(common_time, diff, "g-", linewidth=0.8)
        max_diff = np.max(np.abs(diff))
        rmse = np.sqrt(np.mean(diff**2))
        ax_diff.set_title(f"Difference (interpolated, Max: {max_diff:.2e}, RMSE: {rmse:.2e})")

    ax_diff.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    ax_diff.set_xlabel("Time (s)")
    ax_diff.set_ylabel("Difference")
    ax_diff.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_drift_correction(
    time: NDArray[np.float64],
    original_flow: NDArray[np.float64],
    corrected_flow: NDArray[np.float64],
    cumulative_volume: NDArray[np.float64],
    corrected_volume: NDArray[np.float64],
    drift_error: float,
    title_prefix: str = "",
    save_path: Optional[Path] = None,
    figsize: tuple[float, float] = (12, 10),
) -> None:
    """Plot drift correction details.

    Shows flow before/after correction and cumulative volumes.
    Matches MATLAB visualization style.

    Args:
        time: Time array
        original_flow: Flow before drift correction
        corrected_flow: Flow after drift correction
        cumulative_volume: Cumulative volume before correction
        corrected_volume: Cumulative volume after correction
        drift_error: Calculated drift error rate
        title_prefix: Prefix for titles
        save_path: Path to save figure
        figsize: Figure size
    """
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    # Cumulative volume before correction
    ax1 = axes[0]
    ax1.plot(time, cumulative_volume, "b-", linewidth=0.8)
    ax1.set_ylabel("Volume (L)")
    ax1.set_title(f"{title_prefix}Cumulative Flow Volume (before correction)")
    ax1.grid(True, alpha=0.3)

    # Flow rate
    ax2 = axes[1]
    ax2.plot(time, original_flow, "b-", linewidth=0.5, alpha=0.5, label="Original")
    ax2.plot(time, corrected_flow, "r-", linewidth=0.8, label="Corrected")
    ax2.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    ax2.set_ylabel("Flow Rate (L/s)")
    ax2.set_title(f"Flow Rate (drift error: {drift_error:.6f})")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Cumulative volume after correction
    ax3 = axes[2]
    ax3.plot(time, corrected_volume, "r-", linewidth=0.8)
    ax3.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Volume (L)")
    ax3.set_title(f"{title_prefix}Cumulative Flow Volume (corrected)")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
