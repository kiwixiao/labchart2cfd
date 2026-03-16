"""Phase Contrast (Xenon) workflow for flow profile processing.

This workflow handles xenon phase contrast measurements (3 of 62 scripts).
Key differences from Standard OSAMRI:
1. Uses xenon density (5.761 kg/m³) instead of air
2. Voltage-based calibration with xenon bag reference
3. Zero-shift baseline correction (shift curve to y=0)
4. No cumtrapz drift correction - uses voltage integral calibration
5. Higher sample rate (1000 Hz instead of 100 Hz)
6. No pressure output

Processing steps:
1. Load .mat → extract block[row, column]
2. Zero-shift: subtract mean of baseline region
3. Extract time window based on bag selection
4. Convert voltage to mass flow using calibration
5. Resample to 1000 Hz with spline
6. Smooth with 5-point moving average
7. Export CSV
"""

from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import trapezoid

from labchart2cfd.io.labchart import LabChartData
from labchart2cfd.processing.resampling import resample_to_rate
from labchart2cfd.processing.smoothing import smooth_moving_average
from labchart2cfd.processing.unit_conversion import XENON_DENSITY_KG_M3
from labchart2cfd.workflows.base import BaseWorkflow, WorkflowResult


# Bag-specific trigger times and calibration windows from PC_QX_FlowProfileMasterCode.m
BAG_CONFIG = {
    "Bag1": {
        "shift_time": 20.0,  # Time point to average baseline up to
        "dynamic_start": 57.11,  # Dynamic region for calibration
        "dynamic_end": 60.23,
        "window_start": 53.99,  # Full extraction window
        "window_end": 85.19,
        "trigger_times": [53.99, 57.11, 60.23, 63.35, 66.47, 69.59, 72.71, 75.83, 78.95, 82.07, 85.19],
    },
    "Bag2": {
        "shift_time": 8.0,
        "dynamic_start": 62.44,
        "dynamic_end": 65.56,
        "window_start": 59.32,
        "window_end": 90.52,
        "trigger_times": [59.32, 62.44, 65.56, 68.68, 71.80, 74.92, 78.04, 81.16, 84.28, 87.40, 90.52],
    },
    "Bag4": {
        "shift_time": 20.0,
        "dynamic_start": 94.12,
        "dynamic_end": 97.02,
        "window_start": 91.21,
        "window_end": 120.2,
        "trigger_times": [91.21, 94.12, 97.02, 99.92, 102.8, 105.7, 108.6, 111.5, 114.4, 117.3, 120.2],
    },
}


class PhaseContrastWorkflow(BaseWorkflow):
    """Phase contrast workflow for xenon measurements.

    Key characteristics:
    - Uses xenon density (5.761 kg/m³)
    - Voltage signal requiring calibration
    - Zero-shift baseline correction
    - Volume integral calibration (cumtrapz gives V_per_liter)
    - 1000 Hz output sample rate
    - No pressure output
    """

    def __init__(
        self,
        target_sample_rate: float = 1000.0,  # Higher rate for xenon
        smoothing_window: int = 5,
        density: float = XENON_DENSITY_KG_M3,
        flow_row: int = 2,
    ):
        """Initialize phase contrast workflow.

        Args:
            target_sample_rate: Target sample rate after resampling (Hz)
            smoothing_window: Window size for moving average smoothing
            density: Xenon density in kg/m³ (default: 5.761)
            flow_row: Row containing flow voltage data (default: 2)
        """
        super().__init__(target_sample_rate, smoothing_window, density)
        self.flow_row = flow_row

    @property
    def name(self) -> str:
        return "Phase Contrast (Xenon)"

    @property
    def description(self) -> str:
        return "Xenon phase contrast workflow with voltage calibration"

    def _zero_shift(
        self,
        time: NDArray[np.float64],
        data: NDArray[np.float64],
        shift_time: float,
    ) -> NDArray[np.float64]:
        """Zero-shift data by subtracting baseline mean.

        The MATLAB code finds the mean of data up to a certain time point
        (where the signal is flat/baseline) and subtracts it.

        Args:
            time: Time array
            data: Voltage data
            shift_time: Time point up to which to calculate baseline mean

        Returns:
            Zero-shifted data
        """
        # Find index where time equals or exceeds shift_time
        # MATLAB: tindex_shift = find(timeCell{2,1} == shift_time);
        #         y_shift = mean(dataCell{2,1}(1:tindex_shift(end)));
        shift_idx = np.searchsorted(time, shift_time)
        if shift_idx > 0:
            baseline_mean = np.mean(data[:shift_idx])
            return data - baseline_mean
        return data

    def _get_voltage_calibration(
        self,
        time: NDArray[np.float64],
        data: NDArray[np.float64],
    ) -> float:
        """Calculate voltage-to-volume calibration factor.

        Uses cumulative trapezoid integration to find total voltage-seconds,
        which corresponds to 1 liter of gas.

        MATLAB: vol = cumtrapz(timeU, flowV);
                v_v = max(vol);  # voltage-seconds per liter

        Args:
            time: Time array for the extraction window
            data: Zero-shifted voltage data for the extraction window

        Returns:
            Calibration factor (voltage-seconds per liter)
        """
        volume_integral = trapezoid(data, time)
        return abs(volume_integral) if volume_integral != 0 else 1.0

    def process(
        self,
        data: LabChartData,
        row: int,
        column: int,
        start_time: float,
        end_time: float,
        bag_id: Optional[str] = None,
        calibration_factor: Optional[float] = None,
        shift_time: Optional[float] = None,
        **kwargs,
    ) -> WorkflowResult:
        """Process flow profile using phase contrast workflow.

        Args:
            data: Loaded LabChart data
            row: Flow channel row (1-indexed, typically 2)
            column: Block column (1-indexed)
            start_time: Start time of the window to extract (seconds)
            end_time: End time of the window to extract (seconds)
            bag_id: Bag identifier ("Bag1", "Bag2", or "Bag4") for preset config
            calibration_factor: Manual calibration factor (voltage per liter)
            shift_time: Time point for baseline calculation (overrides bag config)

        Returns:
            WorkflowResult with processed flow data (no pressure)
        """
        # Get full data
        full_flow = data.get_data(row, column)
        full_time = data.get_time(row, column)

        # Determine shift_time from bag config or parameter
        effective_shift_time = shift_time
        bag_config = None
        if bag_id and bag_id in BAG_CONFIG:
            bag_config = BAG_CONFIG[bag_id]
            if effective_shift_time is None:
                effective_shift_time = bag_config["shift_time"]
            # Use bag config times if start/end not specified
            if start_time == 0 and end_time == 0:
                start_time = bag_config["window_start"]
                end_time = bag_config["window_end"]
        elif effective_shift_time is None:
            # Default to 10 seconds if no config
            effective_shift_time = 10.0

        # Step 1: Zero-shift the data
        # MATLAB: y_shift = mean(dataCell{2,1}(1:tindex_shift(end)));
        #         dataCell_update = dataCell{2,1} - y_shift;
        flow_shifted = self._zero_shift(full_time, full_flow, effective_shift_time)

        # Step 2: Extract time window
        time_windowed, flow_windowed = self._extract_time_window(
            full_time, flow_shifted, start_time, end_time
        )

        # Step 3: Determine calibration factor
        if calibration_factor is None:
            # Calculate from voltage integral
            # MATLAB: vol = cumtrapz(timeU, flowV);
            #         v_v = max(vol);
            calibration_factor = self._get_voltage_calibration(time_windowed, flow_windowed)

        # Step 4: Convert voltage to mass flow
        # MATLAB: correctedFlow = flowV / v_v * rho / 1000;
        # flowV / v_v gives L/s, then * rho / 1000 gives kg/s
        flow_lps = flow_windowed / calibration_factor  # L/s
        mass_flow = flow_lps * self.density / 1000  # kg/s

        # Step 5: Resample to target rate (1000 Hz for xenon)
        time_resampled, mass_flow_resampled = resample_to_rate(
            time_windowed, mass_flow, self.target_sample_rate
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
            pressure=None,  # Phase contrast has no pressure
            drift_error=0.0,  # No drift correction in phase contrast
            sample_rate=self.target_sample_rate,
            original_start_time=start_time,
            original_end_time=end_time,
            metadata={
                "workflow": self.name,
                "density_kg_m3": self.density,
                "smoothing_window": self.smoothing_window,
                "bag_id": bag_id,
                "calibration_factor": calibration_factor,
                "shift_time": effective_shift_time,
            },
        )

    def process_with_bag_config(
        self,
        data: LabChartData,
        row: int,
        column: int,
        bag_id: str,
        **kwargs,
    ) -> WorkflowResult:
        """Process using preset bag configuration.

        Convenience method that uses BAG_CONFIG for timing parameters.

        Args:
            data: Loaded LabChart data
            row: Flow channel row (1-indexed)
            column: Block column (1-indexed)
            bag_id: Bag identifier ("Bag1", "Bag2", or "Bag4")

        Returns:
            WorkflowResult with processed flow data

        Raises:
            ValueError: If bag_id is not recognized
        """
        if bag_id not in BAG_CONFIG:
            raise ValueError(f"Unknown bag_id: {bag_id}. Use one of {list(BAG_CONFIG.keys())}")

        config = BAG_CONFIG[bag_id]
        return self.process(
            data=data,
            row=row,
            column=column,
            start_time=config["window_start"],
            end_time=config["window_end"],
            bag_id=bag_id,
            shift_time=config["shift_time"],
            **kwargs,
        )
