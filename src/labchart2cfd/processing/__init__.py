"""Signal processing functions for flow profile data."""

from labchart2cfd.processing.drift_correction import (
    calculate_drift_error,
    apply_drift_correction,
)
from labchart2cfd.processing.resampling import resample_to_rate
from labchart2cfd.processing.smoothing import smooth_moving_average
from labchart2cfd.processing.unit_conversion import (
    liters_per_second_to_kg_per_second,
    cmh2o_to_pascal,
)

__all__ = [
    "calculate_drift_error",
    "apply_drift_correction",
    "resample_to_rate",
    "smooth_moving_average",
    "liters_per_second_to_kg_per_second",
    "cmh2o_to_pascal",
]
