"""Resampling functions for flow profile data.

Provides interpolation to a desired sample rate using cubic spline
interpolation (MATLAB interp1(..., 'spline') equivalent).

Sample-count convention: a window of duration ``period`` at ``rate`` holds
``round(period * rate)`` intervals and therefore ``round(...) + 1`` samples,
endpoint inclusive. Rounding (not truncation) makes the count immune to float
representation of the window span: ``int(409.99999999999994)`` used to drop a
sample and fail the CT synthetic-profile assembly for ~16% of grid anchors.
The endpoint-inclusive fencepost also makes the achieved spacing equal
``period / round(period*rate)``, i.e. the nominal rate, where the old
``int(period*rate)`` count silently delivered 99.75 Hz for a "100 Hz" request.
"""

import numpy as np
from scipy.interpolate import CubicSpline
from typing import Tuple


def _target_indices(time, target_rate_hz):
    # type: (np.ndarray, float) -> np.ndarray
    """Index positions to interpolate at, for the shared fencepost count."""
    if len(time) < 2:
        raise ValueError(
            f"Resampling needs at least 2 samples, got {len(time)}."
        )
    if target_rate_hz <= 0:
        raise ValueError(f"target_rate_hz must be positive, got {target_rate_hz}.")
    period = time[-1] - time[0]
    intervals = int(round(period * target_rate_hz))
    if intervals < 1:
        raise ValueError(
            f"Window of {period:.6f}s holds no complete interval at "
            f"{target_rate_hz:g} Hz; widen the window or raise the rate."
        )
    return np.linspace(0, len(time) - 1, intervals + 1)


def resample_to_rate(time, data, target_rate_hz=100.0):
    # type: (np.ndarray, np.ndarray, float) -> Tuple[np.ndarray, np.ndarray]
    """Resample data to a target sample rate using cubic spline interpolation.

    Args:
        time: Original time array
        data: Original data array
        target_rate_hz: Target sample rate in Hz (default: 100)

    Returns:
        Tuple of (resampled_time, resampled_data), endpoint inclusive, with
        ``round(period * rate) + 1`` samples.

    Raises:
        ValueError: For a window shorter than one output interval, fewer than
            2 input samples, or a non-positive rate.
    """
    target_indices = _target_indices(time, target_rate_hz)

    original_indices = np.arange(len(time))
    resampled_time = CubicSpline(original_indices, time)(target_indices)
    resampled_data = CubicSpline(original_indices, data)(target_indices)

    return resampled_time, resampled_data


def resample_multiple(time, *data_arrays, target_rate_hz=100.0):
    # type: (np.ndarray, *np.ndarray, float) -> Tuple[np.ndarray, ...]
    """Resample multiple data arrays sharing one time base.

    Useful when processing both flow and pressure simultaneously. The rate is
    an explicit keyword (a misspelled keyword used to fall back silently to
    100 Hz via kwargs.get).

    Args:
        time: Original time array
        *data_arrays: One or more data arrays to resample
        target_rate_hz: Target sample rate in Hz (default: 100)

    Returns:
        Tuple of (resampled_time, resampled_data1, resampled_data2, ...)
    """
    target_indices = _target_indices(time, target_rate_hz)

    original_indices = np.arange(len(time))
    resampled_time = CubicSpline(original_indices, time)(target_indices)

    resampled_arrays = [resampled_time]
    for data in data_arrays:
        resampled_arrays.append(
            CubicSpline(original_indices, data)(target_indices)
        )

    return tuple(resampled_arrays)
