"""DC drift correction for flow measurements.

Flow sensors can exhibit baseline drift over time, causing the integrated
volume to not return to zero after a breathing cycle. This module provides
functions to calculate and correct for this drift.

The correction is based on cumulative integration:
    error = abs(cumtrapz(time, flow)[-1]) / time[-1]

This represents the average offset needed to make the integrated volume
return to zero.
"""

from typing import Tuple

import numpy as np
from scipy.integrate import cumulative_trapezoid


def calculate_drift_error(time, flow):
    # type: (np.ndarray, np.ndarray) -> float
    """Calculate the drift error rate from flow data.

    Uses cumulative trapezoidal integration to find the total volume
    drift, then divides by the total time to get the drift rate.

    Args:
        time: Time array in seconds
        flow: Flow rate array (L/s or any consistent units)

    Returns:
        Drift error rate in flow units per second
    """
    # cumulative_trapezoid returns array of length n-1, use initial=0 to match MATLAB
    flow_volume = cumulative_trapezoid(flow, time, initial=0)

    # Get total time span
    total_time = time[-1] - time[0]

    # Calculate error rate: absolute final volume divided by time
    # This matches MATLAB: error = abs(flow_vol(end)) / time(end)
    error = abs(flow_volume[-1]) / total_time

    return float(error)


def apply_drift_correction(flow, error, sign=1):
    # type: (np.ndarray, float, int) -> np.ndarray
    """Apply drift correction to flow data.

    Args:
        flow: Flow rate array
        error: Drift error rate (from calculate_drift_error)
        sign: Correction sign (+1 for standard OSAMRI, -1 for CPAP)
              - Standard: correctedFlow = flow + error
              - CPAP: correctedFlow = flow - error

    Returns:
        Drift-corrected flow array
    """
    return flow + (sign * error)


def correct_flow_drift(time, flow, sign=1):
    # type: (np.ndarray, np.ndarray, int) -> Tuple[np.ndarray, float]
    """Calculate and apply drift correction in one step.

    Convenience function that combines calculate_drift_error and
    apply_drift_correction.

    Args:
        time: Time array in seconds
        flow: Flow rate array
        sign: Correction sign (+1 for standard, -1 for CPAP)

    Returns:
        Tuple of (corrected_flow, error_rate)
    """
    error = calculate_drift_error(time, flow)
    corrected = apply_drift_correction(flow, error, sign)
    return corrected, error
