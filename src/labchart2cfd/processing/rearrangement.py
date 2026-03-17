"""Rearrangement logic for CT workflow.

CT scans capture breathing at an arbitrary phase within a step trigger window.
The user marks where inhale starts, and the signal is rearranged so that
it begins at inhale onset: [inhale_start → step_end] + [step_start → inhale_start].

Image indices are similarly rearranged to match the reordered flow profile.
"""

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
    """Compute number of CT images in a step window (fencepost count).

    Uses fencepost semantics: N intervals produce N+1 boundary lines,
    and each boundary is one image. Both the start and end of the step
    count as images.

    Args:
        step_duration_s: Duration of the step trigger window in seconds
        temporal_resolution_s: Time per image frame in seconds

    Returns:
        Number of images (N+1 fenceposts for N intervals)
    """
    if temporal_resolution_s <= 0:
        raise ValueError("temporal_resolution_s must be positive")
    return int(np.round(step_duration_s / temporal_resolution_s)) + 1


def time_to_image_index(
    landmark_time: float,
    step_start_time: float,
    temporal_resolution_s: float,
) -> int:
    """Convert a landmark time to the nearest fencepost image index.

    Uses round() to snap to the nearest fencepost boundary, avoiding
    off-by-one errors when the click is slightly before a boundary.

    Args:
        landmark_time: Absolute time of the landmark
        step_start_time: Absolute time of step start
        temporal_resolution_s: Time per image frame in seconds

    Returns:
        0-based image index corresponding to the nearest fencepost
    """
    if temporal_resolution_s <= 0:
        raise ValueError("temporal_resolution_s must be positive")
    offset = landmark_time - step_start_time
    return int(np.round(offset / temporal_resolution_s))


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
