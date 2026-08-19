"""Rearrangement logic for CT workflow.

CT scans capture breathing at an arbitrary phase within a step trigger window.
The user marks where inhale starts, and the signal is rearranged so that
it begins at inhale onset: [inhale_start → step_end] + [step_start → inhale_start].

Image indices are similarly rearranged to match the reordered flow profile.
"""

from typing import Optional

import numpy as np
from numpy.typing import NDArray


def rearrange_at_landmark(
    time: NDArray[np.float64],
    data: NDArray[np.float64],
    cut_time: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Splice signal at a landmark time so it starts at that point.

    Produces: [cut_time → end] + [start → cut_time], with time normalized to 0.

    Args:
        time: Time array for the step window
        data: Data array (same length as time)
        cut_time: Time at which to cut and rearrange

    Returns:
        Tuple of (new_time, new_data) with time starting at 0
    """
    # Find the index closest to cut_time
    cut_idx = int(np.argmin(np.abs(time - cut_time)))

    # Rearrange: [cut_idx:] + [:cut_idx]
    new_data = np.concatenate([data[cut_idx:], data[:cut_idx]])

    # Build new time: continuous from 0, preserving original sample spacing
    total_duration = time[-1] - time[0]
    new_time = np.linspace(0.0, total_duration, len(new_data))

    return new_time, new_data


def compute_image_count(
    step_duration_s: float,
    temporal_resolution_s: float,
) -> int:
    """Compute the number of CT images acquired in a step window.

    An image is the slice between two consecutive grid lines, so a window of
    duration D holds D/dt images bounded by D/dt + 1 lines. A scanner running
    4 s at 200 ms produces 20 images, not 21.

    Args:
        step_duration_s: Duration of the step trigger window in seconds
        temporal_resolution_s: Time per image in seconds

    Returns:
        Number of images (slices) in the window
    """
    if temporal_resolution_s <= 0:
        raise ValueError("temporal_resolution_s must be positive")
    # floor(x+0.5): half-up like MATLAB's round. np.round is banker's
    # rounding (round(20.5) == 20), which disagrees at exact half boundaries.
    return int(np.floor(step_duration_s / temporal_resolution_s + 0.5))


def time_to_image_index(
    landmark_time: float,
    step_start_time: float,
    temporal_resolution_s: float,
    total_images: Optional[int] = None,
) -> int:
    """Index of the image whose start is nearest ``landmark_time``.

    A landmark is a point in time, so it snaps to the nearest image
    *boundary*: this answers "which image starts here", not "which image
    contains this". Rounding keeps a click that sits a hair before a boundary
    from falling back into the previous image.

    Args:
        landmark_time: Absolute time of the landmark
        step_start_time: Absolute time of step start
        temporal_resolution_s: Time per image in seconds
        total_images: When given, clamp the result to ``0..total_images-1``
            so the index always names an image that exists.

    Returns:
        0-based index of the image starting nearest the landmark
    """
    if temporal_resolution_s <= 0:
        raise ValueError("temporal_resolution_s must be positive")
    offset = landmark_time - step_start_time
    # Half-up rounding, consistent with compute_image_count.
    idx = int(np.floor(offset / temporal_resolution_s + 0.5))
    if total_images is not None:
        idx = max(0, min(idx, total_images - 1))
    return idx


def image_index_to_time(
    image_index: int,
    step_start_time: float,
    temporal_resolution_s: float,
) -> float:
    """Absolute start time of a 0-based image index.

    Inverse of :func:`time_to_image_index`. Used to snap the flow splice to
    the same instant the image index names, so the exported waveform and the
    exported image order describe the same moment.

    Args:
        image_index: 0-based image index
        step_start_time: Absolute time of step start
        temporal_resolution_s: Time per image in seconds

    Returns:
        Absolute time at which that image begins
    """
    if temporal_resolution_s <= 0:
        raise ValueError("temporal_resolution_s must be positive")
    return step_start_time + image_index * temporal_resolution_s


def rearrange_image_indices(
    total_images: int,
    cut_image_index: int,
) -> NDArray[np.int64]:
    """Rearrange image indices so they start at the cut point.

    Example: total=50, cut=20 → [20, 21, ..., 49, 0, 1, ..., 19]

    Args:
        total_images: Total number of images in the step
        cut_image_index: 0-based index at which to cut

    Returns:
        Array of rearranged 0-based image indices
    """
    if total_images <= 0:
        return np.array([], dtype=np.int64)
    cut_image_index = max(0, min(cut_image_index, total_images - 1))
    indices = np.arange(total_images, dtype=np.int64)
    return np.roll(indices, -cut_image_index)
