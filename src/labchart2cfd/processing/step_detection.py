"""Step trigger detection for CT workflow.

CT scans use step triggers (sustained high level) rather than pulsatile triggers.
A rising edge marks scan start and a falling edge marks scan end.
"""

from typing import List

import numpy as np
from numpy.typing import NDArray


# Shared defaults so the CLI (`process` and `detect-steps`) and the GUI all
# detect the same steps. Keeping these in one place prevents the call sites from
# drifting apart (a past divergence bug came from mismatched min-duration values).
DEFAULT_STEP_THRESHOLD_FRACTION = 0.5
DEFAULT_STEP_MIN_DURATION_S = 0.5


def detect_steps(
    trigger_data: NDArray[np.float64],
    trigger_time: NDArray[np.float64],
    threshold_fraction: float = DEFAULT_STEP_THRESHOLD_FRACTION,
    min_duration_s: float = DEFAULT_STEP_MIN_DURATION_S,
) -> List[dict]:
    """Detect step triggers (sustained high-level regions) in trigger data.

    Uses threshold-based edge detection: rising edge = step start,
    falling edge = step end.

    Args:
        trigger_data: Trigger signal amplitude array
        trigger_time: Corresponding time array
        threshold_fraction: Fraction of max amplitude to use as threshold (0-1)
        min_duration_s: Minimum step duration in seconds to keep

    Returns:
        List of dicts with keys:
            index (int): 1-indexed step number
            start_time (float): Time of rising edge
            end_time (float): Time of falling edge
            start_idx (int): Array index of rising edge
            end_idx (int): Array index of falling edge
            duration (float): Step duration in seconds
    """
    if len(trigger_data) == 0 or len(trigger_time) == 0:
        return []

    lo = float(np.min(trigger_data))
    hi = float(np.max(trigger_data))
    if hi == lo:
        return []  # flat signal: nothing steps

    # Threshold sits inside the signal's RANGE, not at a fraction of max(|x|).
    # A trigger idling at 4 V stepping to 5 V is entirely above 0.5*max(|x|),
    # which used to make the "step" span the whole recording.
    span = hi - lo

    # Polarity is anchored to the RESTING level, taken as whichever extreme the
    # recording starts at (acquisition begins before the scanner fires). A step
    # is then any excursion away from that level, which works for a positive or
    # a negative going trigger and, unlike a median/duty-cycle heuristic, does
    # not invert when a step happens to occupy more than half the recording.
    starts_low = abs(float(trigger_data[0]) - lo) <= abs(float(trigger_data[0]) - hi)
    if starts_low:
        high = trigger_data >= lo + threshold_fraction * span
    else:
        high = trigger_data <= hi - threshold_fraction * span

    # Detect edges by diff: +1 = rising, -1 = falling
    edges = np.diff(high.astype(np.int8))
    rising_indices = np.where(edges == 1)[0] + 1  # index of first high sample
    # First sample BELOW threshold: the step's end boundary. Using the last
    # high sample understated every duration by one sample period (a 4.000 s
    # step reported as 3.999 s).
    falling_indices = np.where(edges == -1)[0] + 1

    # Handle edge cases where signal starts or ends high
    if len(rising_indices) == 0 and len(falling_indices) == 0:
        if high[0]:
            # Entire signal is high — one step spanning the whole duration
            rising_indices = np.array([0])
            falling_indices = np.array([len(trigger_data) - 1])
        else:
            return []

    if len(rising_indices) > 0 and len(falling_indices) > 0:
        # If first falling edge comes before first rising edge,
        # signal starts high
        if falling_indices[0] < rising_indices[0]:
            rising_indices = np.concatenate([[0], rising_indices])
        # If last rising edge comes after last falling edge,
        # signal ends high
        if rising_indices[-1] > falling_indices[-1]:
            falling_indices = np.concatenate([falling_indices, [len(trigger_data) - 1]])
    elif len(rising_indices) > 0:
        # Only rising edges, signal ends high
        falling_indices = np.array([len(trigger_data) - 1] * len(rising_indices))
    elif len(falling_indices) > 0:
        # Only falling edges, signal starts high
        rising_indices = np.array([0] * len(falling_indices))

    # Pair rising and falling edges
    n_steps = min(len(rising_indices), len(falling_indices))
    steps = []
    step_num = 0

    for i in range(n_steps):
        start_idx = int(rising_indices[i])
        end_idx = int(falling_indices[i])
        start_t = float(trigger_time[start_idx])
        end_t = float(trigger_time[end_idx])
        duration = end_t - start_t

        if duration >= min_duration_s:
            step_num += 1
            steps.append({
                "index": step_num,
                "start_time": start_t,
                "end_time": end_t,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "duration": duration,
            })

    return steps
