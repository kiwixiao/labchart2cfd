"""Base workflow class for flow profile processing."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from labchart2cfd.io.labchart import LabChartData


@dataclass
class WorkflowResult:
    """Result container for workflow processing.

    Attributes:
        time: Processed time array (starts at 0)
        mass_flow: Processed mass flow rate in kg/s
        pressure: Processed pressure in Pa (optional, not all workflows have pressure)
        drift_error: Drift error rate that was corrected
        sample_rate: Final sample rate in Hz
        original_start_time: Original start time of the extracted window
        original_end_time: Original end time of the extracted window
    """
    time: NDArray[np.float64]
    mass_flow: NDArray[np.float64]
    pressure: Optional[NDArray[np.float64]] = None
    drift_error: float = 0.0
    sample_rate: float = 100.0
    original_start_time: float = 0.0
    original_end_time: float = 0.0
    metadata: dict = field(default_factory=dict)


class BaseWorkflow(ABC):
    """Abstract base class for flow profile processing workflows.

    Each workflow implementation handles a specific type of measurement:
    - StandardOSAMRIWorkflow: Standard pneumotach measurements
    - CPAPWorkflow: CPAP-assisted breathing measurements
    - PhaseContrastWorkflow: Xenon phase contrast measurements
    """

    # Default processing parameters
    default_flow_row: int = 2  # Channel for flow data
    default_pressure_row: int = 4  # Channel for pressure data
    default_target_sample_rate: float = 100.0  # Hz
    default_smoothing_window: int = 5  # Points

    def __init__(
        self,
        target_sample_rate: float = 100.0,
        smoothing_window: int = 5,
        density: float = 1.2,
    ):
        """Initialize workflow with processing parameters.

        Args:
            target_sample_rate: Target sample rate after resampling (Hz)
            smoothing_window: Window size for moving average smoothing
            density: Gas density in kg/m³ (1.2 for air, 5.761 for xenon)
        """
        self.target_sample_rate = target_sample_rate
        self.smoothing_window = smoothing_window
        self.density = density

    @abstractmethod
    def process(
        self,
        data: LabChartData,
        row: int,
        column: int,
        start_time: float,
        end_time: float,
        **kwargs,
    ) -> WorkflowResult:
        """Process flow profile data.

        Args:
            data: Loaded LabChart data
            row: Channel row (1-indexed)
            column: Block column (1-indexed)
            start_time: Start time of the window to extract (seconds)
            end_time: End time of the window to extract (seconds)
            **kwargs: Additional workflow-specific parameters

        Returns:
            WorkflowResult with processed data
        """
        pass

    def _extract_time_window(
        self,
        time: NDArray[np.float64],
        data: NDArray[np.float64],
        start_time: float,
        end_time: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Extract data within a time window.

        Args:
            time: Full time array
            data: Full data array
            start_time: Start time (seconds)
            end_time: End time (seconds)

        Returns:
            Tuple of (windowed_time, windowed_data)
        """
        mask = (time >= start_time) & (time <= end_time)
        return time[mask], data[mask]

    def _normalize_time(
        self,
        time: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Normalize time array to start at zero.

        Args:
            time: Time array

        Returns:
            Time array starting at 0
        """
        return time - time[0]

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the workflow."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Brief description of the workflow."""
        pass
