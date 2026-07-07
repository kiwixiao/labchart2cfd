"""Tests for CT step-trigger detection.

Covers the shared default constants (so the CLI, detect-steps command, and GUI
cannot silently disagree on which steps exist) plus basic detection behavior.
"""
import numpy as np

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
