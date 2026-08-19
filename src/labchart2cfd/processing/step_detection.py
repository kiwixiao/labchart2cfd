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
DEFAULT_STEP_POLARITY = "active-high"


def detect_steps(
    trigger_data: NDArray[np.float64],
    trigger_time: NDArray[np.float64],
    threshold_fraction: float = DEFAULT_STEP_THRESHOLD_FRACTION,
    min_duration_s: float = DEFAULT_STEP_MIN_DURATION_S,
    polarity: str = DEFAULT_STEP_POLARITY,
) -> List[dict]:
    """Detect step triggers (sustained level regions) in trigger data.

    Threshold-based edge detection: the threshold sits inside the signal's
    RANGE (min + fraction*(max-min)), so a trigger idling at a non-zero
    baseline is handled; using a fraction of max(abs(x)) made such a signal
    sit entirely above threshold and reported the whole recording as one step.

    Args:
        trigger_data: Trigger signal amplitude array
        trigger_time: Corresponding time array
        threshold_fraction: Where the threshold sits inside the signal range (0-1)
        min_duration_s: Minimum step duration in seconds to keep
        polarity: "active-high" (default, the CT convention: the scan is the
            raised level) or "active-low" for an inverted trigger. Declared
            rather than inferred; see the note in the body.

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

    # Polarity is DECLARED, not inferred. Three successive auto-detection rules
    # (median occupancy, first-sample anchor, edge agreement) each passed the
    # cases they were written for and inverted a different real one: a step
    # occupying most of the recording, a block beginning mid-scan, a recording
    # truncated at both ends, a single noisy edge sample. A two-level signal
    # simply does not say which level is the scan, so guessing trades one
    # silent failure for another. CT step triggers are active high, which is
    # the default and matches the pre-existing behaviour; an inverted rig can
    # say so explicitly.
    if polarity == "active-high":
        high = trigger_data >= lo + threshold_fraction * span
    elif polarity == "active-low":
        high = trigger_data <= hi - threshold_fraction * span
    else:
        raise ValueError(
            f"polarity must be 'active-high' or 'active-low', got {polarity!r}."
        )

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
