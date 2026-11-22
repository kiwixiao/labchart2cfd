"""Tests for in equals out on a composed breath (design 2026-11-22, 4.7).

The CFD repeats the exported breath, so any net volume accumulates every
cycle. The complete half is kept as recorded and the composed half is scaled
by one factor so the net volume is zero. Timing is never changed.
"""
import numpy as np
import pytest
from scipy.integrate import trapezoid

from labchart2cfd.processing.volume_balance import balance_halves, mass_volume_to_ml

RATE = 100.0


def _breath():
    """One 2 s breath with 0.1 more above zero than below: net is not zero."""
    t = np.arange(0, 2.0, 1 / RATE)
    return np.sin(2 * np.pi * 0.5 * t) + 0.1


def test_net_volume_is_zero_after_balancing():
    flow = _breath()
    assert abs(trapezoid(flow, dx=1 / RATE)) > 0.1        # the case to fix

    result = balance_halves(flow, RATE, keep="positive")

    assert abs(trapezoid(result.flow, dx=1 / RATE)) < 1e-12


def test_kept_half_is_unchanged_sample_by_sample():
    flow = _breath()

    result = balance_halves(flow, RATE, keep="positive")

    kept = flow > 0
    assert np.array_equal(result.flow[kept], flow[kept])
    assert not np.array_equal(result.flow[~kept], flow[~kept])


def test_no_sample_changes_sign():
    """Each zero crossing stays between the same two samples."""
    flow = _breath()

    result = balance_halves(flow, RATE, keep="negative")

    assert np.array_equal(np.sign(result.flow), np.sign(flow))


def test_factor_is_the_volume_ratio():
    flow = _breath()
    above = trapezoid(np.where(flow > 0, flow, 0.0), dx=1 / RATE)
    below = trapezoid(np.where(flow < 0, flow, 0.0), dx=1 / RATE)

    result = balance_halves(flow, RATE, keep="positive")

    assert result.factor == pytest.approx(-above / below)
    assert result.kept_volume == pytest.approx(above)
    assert result.scaled_volume_before == pytest.approx(below)


def test_a_half_with_no_volume_raises():
    """All samples above zero: there is no other half to scale."""
    flow = np.abs(_breath()) + 0.01

    with pytest.raises(ValueError, match="no volume"):
        balance_halves(flow, RATE, keep="positive")


def test_an_unknown_half_raises():
    with pytest.raises(ValueError, match="keep"):
        balance_halves(_breath(), RATE, keep="inhale")


def test_mass_volume_to_ml():
    """1.2 g of air at 1.2 kg/m3 is 1 L."""
    assert mass_volume_to_ml(0.0012, density=1.2) == pytest.approx(1000.0)
