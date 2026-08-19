"""Base workflow class for flow profile processing."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline

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

        Raises:
            ValueError: If the window does not intersect the data. An empty
                window used to propagate and die deep in resampling with a
                bare IndexError.
        """
        mask = (time >= start_time) & (time <= end_time)
        if not np.any(mask):
            raise ValueError(
                f"Requested window [{start_time:.3f}, {end_time:.3f}]s does "
                f"not intersect the data, which covers "
                f"[{time[0]:.3f}, {time[-1]:.3f}]s."
            )
        return time[mask], data[mask]

    def _pressure_on_time_base(
        self,
        data: LabChartData,
        pressure_row: int,
        column: int,
        target_time: NDArray[np.float64],
    ) -> Optional[NDArray[np.float64]]:
        """Interpolate the pressure channel onto the flow's resampled time base.

        Pressure is read on its OWN time axis, because LabChart routinely
        records channels at different sample rates and masking pressure with
        the flow channel's time array crashed on any rate mismatch.

        Pressure is optional and must never change the flow: rather than
        trimming both to a common length (which silently shortened the
        exported flow when the pressure channel did not span the window), this
        returns None when pressure cannot cover ``target_time``, leaving the
        flow export intact and the omission visible in metadata.

        Args:
            data: Loaded LabChart data
            pressure_row: Pressure channel row (1-indexed)
            column: Block column (1-indexed)
            target_time: The flow's resampled time axis, in original units.

        Returns:
            Pressure in cmH2O sampled at ``target_time``, or None if the
            channel does not cover that span.
        """
        p_time = data.get_time(pressure_row, column)
        p_data = data.get_data(pressure_row, column)
        if len(p_time) < 2:
            return None
        # Require full coverage so the spline never extrapolates.
        if p_time[0] > target_time[0] or p_time[-1] < target_time[-1]:
            return None
        return CubicSpline(p_time, p_data)(target_time)

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
