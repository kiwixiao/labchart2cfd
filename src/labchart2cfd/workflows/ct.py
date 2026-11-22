"""CT workflow for flow profile processing.

CT scans use step triggers. Within a step window, the breathing cycle
starts at an arbitrary phase. The user marks the inhale start time,
and the flow profile is rearranged so it begins at inhale onset.

Pipeline:
1. Extract step window from full signal
2. Rearrange at inhale_start_time
3. Drift correction on rearranged signal (sign=+1)
4. Convert L/s → kg/s
5. Resample to 100 Hz
6. Smooth with moving average
7. Same for pressure if available
8. Compute image mapping and store in metadata
"""

from typing import Optional, Tuple

import numpy as np

from labchart2cfd.io.labchart import LabChartData
from labchart2cfd.processing.drift_correction import (
    apply_drift_correction,
    calculate_drift_error,
    correct_flow_drift,
)
from labchart2cfd.processing.rearrangement import (
    rearrange_at_landmark,
    compute_image_count,
    time_to_image_index,
    image_index_to_time,
    rearrange_image_indices,
)
from labchart2cfd.processing.resampling import resample_to_rate
from labchart2cfd.processing.smoothing import smooth_moving_average
from labchart2cfd.processing.unit_conversion import (
    liters_per_second_to_kg_per_second,
    cmh2o_to_pascal,
    AIR_DENSITY_KG_M3,
)
from labchart2cfd.workflows.base import BaseWorkflow, WorkflowResult


class CTWorkflow(BaseWorkflow):
    """CT workflow for step-triggered flow profile processing.

    Key characteristics:
    - Uses step triggers (sustained high level) instead of pulsatile triggers
    - Rearranges signal at inhale onset so flow profile starts at inhale
    - Drift correction on rearranged signal (positive sign)
    - Computes image index mapping based on temporal resolution
    """

    def __init__(
        self,
        target_sample_rate: float = 100.0,
        smoothing_window: int = 5,
        density: float = AIR_DENSITY_KG_M3,
        pressure_row: int = 4,
    ):
        super().__init__(target_sample_rate, smoothing_window, density)
        self.pressure_row = pressure_row

    @property
    def name(self) -> str:
        return "CT"

    @property
    def description(self) -> str:
        return "CT step-trigger workflow with signal rearrangement at inhale onset"

    def process(
        self,
        data: LabChartData,
        row: int,
        column: int,
        start_time: float,
        end_time: float,
        inhale_start_time: Optional[float] = None,
        exhale_end_time: Optional[float] = None,
        temporal_resolution: float = 0.2,
        include_pressure: bool = True,
        drift_range: Optional[Tuple[float, float]] = None,
        **kwargs,
    ) -> WorkflowResult:
        """Process flow profile using CT workflow.

        Args:
            data: Loaded LabChart data
            row: Flow channel row (1-indexed)
            column: Block column (1-indexed)
            start_time: Step window start time (seconds)
            end_time: Step window end time (seconds)
            inhale_start_time: Time of inhale onset within the step (seconds).
                If None, no rearrangement is performed.
            exhale_end_time: Time of exhale end within the step (seconds).
                Stored in metadata for reference.
            temporal_resolution: CT temporal resolution in seconds (default 0.2s = 200ms)
            include_pressure: Whether to process pressure data
            drift_range: (t0, t1) in recording time. When given, the drift is
                the mean of the raw flow over this range (one whole breath the
                user picked), subtracted from the window. When None, the mean
                over the window itself, as before.

        Returns:
            WorkflowResult with processed and rearranged data

        Raises:
            ValueError: If drift_range is empty or reaches outside the
                recording. Part of a breath would be measured and reported
                as the drift without a word.
        """
        full_flow = data.get_data(row, column)
        full_time = data.get_time(row, column)
        if drift_range is not None:
            d0, d1 = float(drift_range[0]), float(drift_range[1])
            if not d1 > d0:
                raise ValueError(
                    f"The drift breath must end after it starts ({d0:.3f} s to {d1:.3f} s)."
                )
            if d0 < full_time[0] - 1e-9 or d1 > full_time[-1] + 1e-9:
                raise ValueError(
                    f"The drift breath {d0:.3f} s to {d1:.3f} s reaches outside "
                    f"the recording ({full_time[0]:.3f} s to {full_time[-1]:.3f} s)."
                )
            drift_range = (d0, d1)

        # Step 1: Extract step window
        time_windowed, flow_windowed = self._extract_time_window(
            full_time, full_flow, start_time, end_time
        )

        # Step 2: Rearrange so the profile starts at inhale onset.
        # With both landmarks in normal order (inhale left of exhale) the
        # window already runs inhale->exhale and no splice is needed; the
        # wrapping case (exhale left of inhale) splices. With ONLY the inhale
        # given, rearrange at it: the old condition also required exhale to be
        # supplied, so the CLI's documented `--inhale-start` invocation
        # silently did nothing while metadata reported the time as honoured.
        needs_rearrange = inhale_start_time is not None and (
            exhale_end_time is None or exhale_end_time < inhale_start_time
        )

        # Snap the cut to the start of the nearest image BEFORE splicing, so
        # the exported waveform and the exported image order describe the same
        # instant. Cutting at the raw click while reporting a rounded image
        # index would leave them misaligned by up to half an image.
        step_duration = end_time - start_time
        total_images = compute_image_count(step_duration, temporal_resolution)
        cut_image_idx = 0
        cut_time = inhale_start_time
        if needs_rearrange:
            cut_image_idx = time_to_image_index(
                inhale_start_time, start_time, temporal_resolution, total_images
            )
            cut_time = image_index_to_time(
                cut_image_idx, start_time, temporal_resolution
            )

        # Keep the window's original time base: the splice below rewrites
        # time_windowed to start at 0, but pressure must be sampled against
        # the recording's own clock before being spliced the same way.
        time_windowed_raw = time_windowed.copy()

        if needs_rearrange:
            time_windowed, flow_windowed = rearrange_at_landmark(
                time_windowed, flow_windowed, cut_time
            )

        # Step 3: Drift correction on rearranged signal, or with the error
        # measured on the breath the user picked.
        if drift_range is None:
            corrected_flow, drift_error = correct_flow_drift(
                time_windowed, flow_windowed
            )
        else:
            t_breath, f_breath = self._extract_time_window(
                full_time, full_flow, drift_range[0], drift_range[1]
            )
            if len(t_breath) < 2:
                raise ValueError(
                    f"The drift breath {drift_range[0]:.3f} s to "
                    f"{drift_range[1]:.3f} s holds fewer than 2 samples."
                )
            drift_error = calculate_drift_error(t_breath, f_breath)
            corrected_flow = apply_drift_correction(flow_windowed, drift_error)

        # Step 4: Convert to mass flow (L/s → kg/s)
        mass_flow = liters_per_second_to_kg_per_second(
            corrected_flow, self.density, invert_sign=True
        )

        # Step 5: Resample the flow. The flow alone sets the output length;
        # pressure is optional and must never reshape it.
        time_resampled, mass_flow_resampled = resample_to_rate(
            time_windowed, mass_flow, self.target_sample_rate
        )
        mass_flow_smooth = smooth_moving_average(mass_flow_resampled, self.smoothing_window)

        # Step 6: Pressure. Sampled on its own time base over the SAME window,
        # then spliced at the same instant as the flow so the two stay aligned.
        pressure_resampled: Optional[np.ndarray] = None
        pressure_note: Optional[str] = None
        if include_pressure and not data.is_block_empty(self.pressure_row, column):
            # Sample pressure on the RAW window grid, then splice it with the
            # same function and cut instant as the flow, then resample. Doing
            # the splice after resampling instead rotated a 10 ms output sample
            # where the flow rotated a 1 ms raw one, leaving the exported
            # pressure with a sub-sample phase offset against the flow.
            p_raw = self._pressure_on_time_base(
                data, self.pressure_row, column, time_windowed_raw
            )
            if p_raw is None:
                pressure_note = (
                    f"pressure row {self.pressure_row} does not cover the "
                    "requested window; flow exported without pressure"
                )
            else:
                pressure_pa = cmh2o_to_pascal(p_raw)
                if needs_rearrange:
                    _, pressure_pa = rearrange_at_landmark(
                        time_windowed_raw, pressure_pa, cut_time
                    )
                # time_windowed is the flow's post-splice base, same length and
                # span, so both resample to identical counts.
                _, pressure_resampled = resample_to_rate(
                    time_windowed, pressure_pa, self.target_sample_rate
                )
                pressure_resampled = smooth_moving_average(
                    pressure_resampled, self.smoothing_window
                )

        # Normalize time to start at 0
        time_normalized = self._normalize_time(time_resampled)

        # Step 8: Image mapping (count and cut index computed above, since the
        # splice needs the snapped cut time).
        rearranged_indices = rearrange_image_indices(total_images, cut_image_idx)

        metadata = {
            "workflow": self.name,
            "density_kg_m3": self.density,
            "smoothing_window": self.smoothing_window,
            "step_start_time": start_time,
            "step_end_time": end_time,
            "step_duration": step_duration,
            "inhale_start_time": inhale_start_time,
            # The instant actually cut at, after snapping to an image start.
            "inhale_snapped_time": cut_time,
            "exhale_end_time": exhale_end_time,
            "temporal_resolution_s": temporal_resolution,
            "total_images": total_images,
            "cut_image_index": cut_image_idx,
            "pressure_note": pressure_note,
            "drift_range": drift_range,
            "rearranged_image_indices": rearranged_indices.tolist(),
        }

        return WorkflowResult(
            time=time_normalized,
            mass_flow=mass_flow_smooth,
            pressure=pressure_resampled,
            drift_error=drift_error,
            sample_rate=self.target_sample_rate,
            original_start_time=start_time,
            original_end_time=end_time,
            metadata=metadata,
        )
