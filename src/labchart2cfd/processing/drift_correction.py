"""DC drift correction for flow measurements.

Flow sensors can exhibit baseline drift over time, causing the integrated
volume to not return to zero after a breathing cycle. This module provides
functions to calculate and correct for this drift.

The correction subtracts the signed mean drift rate:

    error     = cumtrapz(flow, time)[-1] / (time[-1] - time[0])
    corrected = flow - error

so the integrated volume returns to zero for either drift polarity.

History: the MATLAB original (and the first port of it) computed
``abs(flow_vol(end)) / time(end)`` and applied it with a per-workflow sign
(+1 standard, -1 CPAP). Taking abs() discards the drift's direction, so
whenever the actual polarity did not match the assumed one the "correction"
doubled the drift instead of removing it. The signed form needs no per
workflow sign and is correct for both directions.
"""

from typing import Tuple

import numpy as np
from scipy.integrate import cumulative_trapezoid


def calculate_drift_error(time, flow):
    # type: (np.ndarray, np.ndarray) -> float
    """Calculate the signed drift error rate from flow data.

    Integrates the flow to volume, then divides the net (signed) final volume
    by the elapsed time to get the mean drift rate.

    Args:
        time: Time array in seconds (at least 2 samples)
        flow: Flow rate array (L/s or any consistent units)

    Returns:
        Signed drift error rate in flow units per second. Positive means the
        signal drifts positive.

    Raises:
        ValueError: If the window has fewer than 2 samples or zero duration.
    """
    if len(time) < 2:
        raise ValueError(
            f"Drift correction needs at least 2 samples, got {len(time)}."
        )
    total_time = time[-1] - time[0]
    if total_time <= 0:
        raise ValueError(
            f"Drift correction needs a positive time span, got {total_time}."
        )

    # cumulative_trapezoid returns array of length n-1, use initial=0 to match MATLAB
    flow_volume = cumulative_trapezoid(flow, time, initial=0)

    return float(flow_volume[-1] / total_time)


def apply_drift_correction(flow, error):
    # type: (np.ndarray, float) -> np.ndarray
    """Subtract the signed drift error rate from the flow.

    Args:
        flow: Flow rate array
        error: Signed drift error rate (from calculate_drift_error)

    Returns:
        Drift-corrected flow array whose integral returns to ~zero.
    """
    return flow - error


def correct_flow_drift(time, flow):
    # type: (np.ndarray, np.ndarray) -> Tuple[np.ndarray, float]
    """Calculate and apply drift correction in one step.

    Args:
        time: Time array in seconds
        flow: Flow rate array

    Returns:
        Tuple of (corrected_flow, signed_error_rate)
    """
    error = calculate_drift_error(time, flow)
    corrected = apply_drift_correction(flow, error)
    return corrected, error
