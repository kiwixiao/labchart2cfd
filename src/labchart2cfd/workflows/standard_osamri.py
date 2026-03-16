"""Standard OSAMRI workflow for flow profile processing.

This workflow handles the most common case (49 of 62 scripts):
1. Load .mat → extract block[row, column]
2. DC drift correction on FULL signal: error = abs(cumtrapz[-1]) / time[-1]
3. Correct: flow + error (positive sign)
4. Extract time window [start, end]
5. Convert: mass_flow = flow * 1e-3 * density * -1
6. Resample to 100 Hz with spline
7. Smooth with 5-point moving average
8. Export CSV
"""

from typing import Optional

import numpy as np

from labchart2cfd.io.labchart import LabChartData
from labchart2cfd.processing.drift_correction import correct_flow_drift
from labchart2cfd.processing.resampling import resample_multiple
from labchart2cfd.processing.smoothing import smooth_moving_average
from labchart2cfd.processing.unit_conversion import (
    liters_per_second_to_kg_per_second,
    cmh2o_to_pascal,
    AIR_DENSITY_KG_M3,
)
from labchart2cfd.workflows.base import BaseWorkflow, WorkflowResult


class StandardOSAMRIWorkflow(BaseWorkflow):
    """Standard OSAMRI workflow for pneumotach measurements.

    Key characteristics:
    - Drift correction on FULL signal before windowing
    - Uses flow + error correction (positive sign)
    - Processes both flow (row 2) and pressure (row 4)
    - Resamples to 100 Hz
    - 5-point moving average smoothing
    """

    def __init__(
        self,
        target_sample_rate: float = 100.0,
        smoothing_window: int = 5,
        density: float = AIR_DENSITY_KG_M3,
        pressure_row: int = 4,
    ):
        """Initialize standard OSAMRI workflow.

        Args:
            target_sample_rate: Target sample rate after resampling (Hz)
            smoothing_window: Window size for moving average smoothing
            density: Air density in kg/m³ (default: 1.2)
            pressure_row: Row containing pressure data (default: 4)
        """
        super().__init__(target_sample_rate, smoothing_window, density)
        self.pressure_row = pressure_row

    @property
    def name(self) -> str:
        return "Standard OSAMRI"

    @property
    def description(self) -> str:
        return "Standard pneumotach workflow with drift correction on full signal"

    def process(
        self,
        data: LabChartData,
        row: int,
        column: int,
        start_time: float,
        end_time: float,
        include_pressure: bool = True,
        **kwargs,
    ) -> WorkflowResult:
        """Process flow profile using standard OSAMRI workflow.

        Args:
            data: Loaded LabChart data
            row: Flow channel row (1-indexed, typically 2)
            column: Block column (1-indexed)
            start_time: Start time of the window to extract (seconds)
            end_time: End time of the window to extract (seconds)
            include_pressure: Whether to process pressure data (default: True)

        Returns:
            WorkflowResult with processed flow and pressure data
        """
        # Extract full flow data for drift correction
        full_flow = data.get_data(row, column)
        full_time = data.get_time(row, column)

        # Step 1: DC drift correction on FULL signal
        # MATLAB: flow_vol = cumtrapz(time, flow);
        #         error = abs(flow_vol(end)) / time(end);
        #         correctedFlow = flow + error;
        corrected_flow, drift_error = correct_flow_drift(
            full_time, full_flow, sign=1  # Standard uses + error
        )

        # Step 2: Extract time window
        time_windowed, flow_windowed = self._extract_time_window(
            full_time, corrected_flow, start_time, end_time
        )

        # Step 3: Convert to mass flow
        # MATLAB: massFlow = FlowU .* 1e-3 .* rho * -1;
        mass_flow = liters_per_second_to_kg_per_second(
            flow_windowed, self.density, invert_sign=True
        )

        # Step 4: Handle pressure if requested
        pressure_resampled: Optional[np.ndarray] = None
        if include_pressure and not data.is_block_empty(self.pressure_row, column):
            full_pressure = data.get_data(self.pressure_row, column)
            _, pressure_windowed = self._extract_time_window(
                full_time, full_pressure, start_time, end_time
            )
            # Convert cmH2O to Pa
            pressure_pa = cmh2o_to_pascal(pressure_windowed)

            # Step 5: Resample both flow and pressure
            time_resampled, mass_flow_resampled, pressure_resampled = resample_multiple(
                time_windowed,
                mass_flow,
                pressure_pa,
                target_rate_hz=self.target_sample_rate,
            )

            # Step 6: Smooth both
            mass_flow_smooth = smooth_moving_average(
                mass_flow_resampled, self.smoothing_window
            )
            pressure_smooth = smooth_moving_average(
                pressure_resampled, self.smoothing_window
            )
            pressure_resampled = pressure_smooth
        else:
            # Step 5: Resample flow only
            time_resampled, mass_flow_resampled = resample_multiple(
                time_windowed,
                mass_flow,
                target_rate_hz=self.target_sample_rate,
            )

            # Step 6: Smooth
            mass_flow_smooth = smooth_moving_average(
                mass_flow_resampled, self.smoothing_window
            )

        # Step 7: Normalize time to start at 0
        time_normalized = self._normalize_time(time_resampled)

        return WorkflowResult(
            time=time_normalized,
            mass_flow=mass_flow_smooth,
            pressure=pressure_resampled,
            drift_error=drift_error,
            sample_rate=self.target_sample_rate,
            original_start_time=start_time,
            original_end_time=end_time,
            metadata={
                "workflow": self.name,
                "density_kg_m3": self.density,
                "smoothing_window": self.smoothing_window,
            },
        )
