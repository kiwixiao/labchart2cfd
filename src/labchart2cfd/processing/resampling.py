"""Resampling functions for flow profile data.

Provides interpolation to a desired sample rate using cubic spline
interpolation, matching MATLAB's interp1(..., 'spline') behavior.
"""

from typing import Tuple

import numpy as np
from scipy.interpolate import CubicSpline


def resample_to_rate(time, data, target_rate_hz=100.0):
    # type: (np.ndarray, np.ndarray, float) -> Tuple[np.ndarray, np.ndarray]
    """Resample data to a target sample rate using cubic spline interpolation.

    This matches the MATLAB workflow:
        period = endTime - startTime
        desiredSamples = period * desiredSampleRate
        DS = linspace(1, size(TimeU, 2), desiredSamples)
        dataDS = interp1(1:size(TimeU, 2), data, DS, 'spline')

    Args:
        time: Original time array
        data: Original data array
        target_rate_hz: Target sample rate in Hz (default: 100)

    Returns:
        Tuple of (resampled_time, resampled_data)
    """
    # Calculate period and desired number of samples
    period = time[-1] - time[0]
    num_samples = int(period * target_rate_hz)

    # MATLAB uses linspace on indices, then interpolates
    # We'll do the same for exact matching
    original_indices = np.arange(len(time))
    target_indices = np.linspace(0, len(time) - 1, num_samples)

    # Create spline interpolators
    time_spline = CubicSpline(original_indices, time)
    data_spline = CubicSpline(original_indices, data)

    # Interpolate
    resampled_time = time_spline(target_indices)
    resampled_data = data_spline(target_indices)

    return resampled_time, resampled_data


def resample_multiple(time, *data_arrays, **kwargs):
    # type: (np.ndarray, *np.ndarray, **float) -> Tuple[np.ndarray, ...]
    """Resample multiple data arrays to the same time base.

    Useful when processing both flow and pressure simultaneously.

    Args:
        time: Original time array
        *data_arrays: One or more data arrays to resample
        target_rate_hz: Target sample rate in Hz (default: 100)

    Returns:
        Tuple of (resampled_time, resampled_data1, resampled_data2, ...)
    """
    target_rate_hz = kwargs.get('target_rate_hz', 100.0)

    period = time[-1] - time[0]
    num_samples = int(period * target_rate_hz)

    original_indices = np.arange(len(time))
    target_indices = np.linspace(0, len(time) - 1, num_samples)

    # Interpolate time
    time_spline = CubicSpline(original_indices, time)
    resampled_time = time_spline(target_indices)

    # Interpolate each data array
    resampled_arrays = [resampled_time]
    for data in data_arrays:
        data_spline = CubicSpline(original_indices, data)
        resampled_arrays.append(data_spline(target_indices))

    return tuple(resampled_arrays)
