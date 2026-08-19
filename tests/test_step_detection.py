"""Tests for CT step-trigger detection.

Covers the shared default constants (so the CLI, detect-steps command, and GUI
cannot silently disagree on which steps exist) plus basic detection behavior.
"""
import numpy as np
import pytest

from labchart2cfd.processing.step_detection import (
    detect_steps,
    DEFAULT_STEP_MIN_DURATION_S,
    DEFAULT_STEP_THRESHOLD_FRACTION,
)


def _two_step_signal():
    fs = 1000.0
    t = np.arange(0, 10, 1 / fs)
    trig = np.zeros_like(t)
    trig[(t >= 1.0) & (t < 1.7)] = 5.0   # 0.7 s plateau
    trig[(t >= 4.0) & (t < 6.5)] = 5.0   # 2.5 s plateau
    return trig, t


def test_shared_defaults_are_half_second():
    assert DEFAULT_STEP_MIN_DURATION_S == 0.5
    assert DEFAULT_STEP_THRESHOLD_FRACTION == 0.5


def test_default_min_duration_keeps_short_step():
    # With the shared default (no min_duration passed), a 0.7 s step is kept.
    trig, t = _two_step_signal()
    steps = detect_steps(trig, t)
    assert len(steps) == 2
    durations = sorted(round(s["duration"], 1) for s in steps)
    assert durations == [0.7, 2.5]


def test_larger_min_duration_drops_short_step():
    trig, t = _two_step_signal()
    steps = detect_steps(trig, t, min_duration_s=1.0)
    assert len(steps) == 1
    assert round(steps[0]["duration"], 1) == 2.5


def test_empty_input_returns_no_steps():
    assert detect_steps(np.array([]), np.array([])) == []


def test_dc_offset_baseline_does_not_swallow_the_recording():
    """Threshold must come from the signal's range, not max(|x|).

    A trigger idling at 4 V and stepping to 5 V sat entirely above the old
    0.5*max(|x|) = 2.5 V threshold, so the 'step' was the whole recording and
    the CT window was silently wrong.
    """
    fs = 1000.0
    t = np.arange(0, 20, 1 / fs)
    trig = np.full_like(t, 4.0)
    trig[(t >= 3.0) & (t < 7.0)] = 5.0
    trig[(t >= 12.0) & (t < 16.0)] = 5.0

    steps = detect_steps(trig, t)

    assert len(steps) == 2
    assert steps[0]["start_time"] == pytest.approx(3.0, abs=2 / fs)
    assert steps[0]["duration"] == pytest.approx(4.0, abs=2 / fs)
    assert steps[1]["start_time"] == pytest.approx(12.0, abs=2 / fs)


def test_negative_going_trigger_is_detected():
    """A trigger stepping DOWN from baseline is still a step."""
    fs = 1000.0
    t = np.arange(0, 20, 1 / fs)
    trig = np.zeros_like(t)
    trig[(t >= 5.0) & (t < 9.0)] = -5.0

    steps = detect_steps(trig, t)

    assert len(steps) == 1
    assert steps[0]["start_time"] == pytest.approx(5.0, abs=2 / fs)
    assert steps[0]["duration"] == pytest.approx(4.0, abs=2 / fs)


def test_flat_signal_returns_no_steps():
    t = np.arange(0, 5, 0.001)
    assert detect_steps(np.full_like(t, 3.3), t) == []


def test_duration_covers_the_full_step():
    """The step ends at the first sample BELOW threshold, so a genuine 4.000 s
    step is not reported as 3.999 s (one sample short)."""
    fs = 1000.0
    t = np.arange(0, 10, 1 / fs)
    trig = np.zeros_like(t)
    trig[(t >= 2.0) & (t < 6.0)] = 5.0

    steps = detect_steps(trig, t)

    assert len(steps) == 1
    assert steps[0]["duration"] == pytest.approx(4.0, abs=1e-9)
