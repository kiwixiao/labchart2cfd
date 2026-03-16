"""Smoothing functions for flow profile data.

MATLAB's smooth() function uses a 5-point moving average by default.
This module provides equivalent functionality.
"""

import numpy as np


def smooth_moving_average(data, window_size=5):
    # type: (np.ndarray, int) -> np.ndarray
    """Apply moving average smoothing to data.

    Matches MATLAB's smooth() function behavior with default 5-point window.
    At the edges, uses smaller windows (asymmetric smoothing).

    MATLAB's smooth() for a 5-point window:
    - y(1) = y(1)  (no change)
    - y(2) = (y(1) + y(2) + y(3)) / 3
    - y(3) = (y(1) + y(2) + y(3) + y(4) + y(5)) / 5
    - y(n-1) = (y(n-2) + y(n-1) + y(n)) / 3
    - y(n) = y(n)  (no change)

    Args:
        data: Input data array
        window_size: Size of the moving average window (default: 5, must be odd)

    Returns:
        Smoothed data array of the same length
    """
    if window_size % 2 == 0:
        window_size += 1  # Ensure odd window size

    n = len(data)
    half_window = window_size // 2
    result = np.zeros_like(data)

    for i in range(n):
        # Determine window bounds
        start = max(0, i - half_window)
        end = min(n, i + half_window + 1)

        # For MATLAB compatibility at edges:
        # First and last points use progressively smaller windows
        if i < half_window:
            # Left edge: use points from 0 to 2*i+1 (centered on i)
            start = 0
            end = min(n, 2 * i + 1)
            if end <= start:
                end = start + 1
        elif i >= n - half_window:
            # Right edge: use points from 2*i-n+1 to n
            start = max(0, 2 * i - n + 1)
            end = n

        result[i] = np.mean(data[start:end])

    return result


def smooth_savgol(data, window_size=5, polyorder=2):
    # type: (np.ndarray, int, int) -> np.ndarray
    """Apply Savitzky-Golay filter for smoothing.

    Alternative to moving average that preserves peaks better.
    Not used in standard MATLAB workflow but available for comparison.

    Args:
        data: Input data array
        window_size: Size of the filter window (must be odd)
        polyorder: Order of the polynomial used for fitting

    Returns:
        Smoothed data array
    """
    from scipy.signal import savgol_filter

    if window_size % 2 == 0:
        window_size += 1

    if window_size <= polyorder:
        window_size = polyorder + 1
        if window_size % 2 == 0:
            window_size += 1

    return savgol_filter(data, window_size, polyorder)
