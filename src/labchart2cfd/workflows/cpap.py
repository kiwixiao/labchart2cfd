"""CPAP workflow for flow profile processing.

This workflow handles CPAP-assisted breathing measurements (4 of 62 scripts).
Key differences from Standard OSAMRI:
1. DC drift correction on WINDOWED data (not full signal)
2. Uses flow - error (negative sign, opposite of standard)

Processing steps:
1. Load .mat → extract block[row, column]
2. Extract time window [start, end] FIRST
3. DC drift on windowed data: error = abs(cumtrapz[-1]) / Dt
4. Correct: flow - error (negative sign)
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


class CPAPWorkflow(BaseWorkflow):
    """CPAP workflow for CPAP-assisted breathing measurements.

    Key characteristics:
    - Window extraction BEFORE drift correction
    - Drift correction on WINDOWED data only
    - Uses flow - error correction (negative sign)
    - Otherwise same processing as Standard OSAMRI

    This accounts for the fact that CPAP measurements may have different
    baseline characteristics within the breathing cycle window compared
    to the full recording.
    """

    def __init__(
        self,
        target_sample_rate: float = 100.0,
        smoothing_window: int = 5,
        density: float = AIR_DENSITY_KG_M3,
        pressure_row: int = 4,
    ):
        """Initialize CPAP workflow.

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
        return "CPAP"

    @property
    def description(self) -> str:
        return "CPAP workflow with drift correction on windowed data"

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
        """Process flow profile using CPAP workflow.

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
        # Extract full data
        full_flow = data.get_data(row, column)
        full_time = data.get_time(row, column)

        # Step 1: Extract time window FIRST (different from Standard)
        # MATLAB: flow_picked = flow(timeIndicesToUse);
        #         time_picked = time(timeIndicesToUse);
        time_windowed, flow_windowed = self._extract_time_window(
            full_time, full_flow, start_time, end_time
        )

        # Step 2: DC drift correction on WINDOWED data
        # MATLAB: Dt = max(time_picked) - min(time_picked);
        #         flow_vol_picked = cumtrapz(time_picked, flow_picked);
        #         error_picked = abs(flow_vol_picked(end)) / Dt;
        #         correctedFlow = flow_picked - error_picked;  # NOTE: minus sign!
        corrected_flow, drift_error = correct_flow_drift(
            time_windowed, flow_windowed, sign=-1  # CPAP uses - error
        )

        # Step 3: Convert to mass flow
        mass_flow = liters_per_second_to_kg_per_second(
            corrected_flow, self.density, invert_sign=True
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

            # Step 5: Resample both
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
                "drift_on_windowed": True,  # Key difference from Standard
            },
        )
