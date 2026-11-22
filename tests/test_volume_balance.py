"""Tests for in equals out on a composed breath (design 2026-11-22, 4.7).

The CFD repeats the exported breath, so any net volume accumulates every
cycle. The complete half is kept as recorded and the composed half is scaled
by one factor so the net volume is zero. Timing is never changed.
"""
import numpy as np
import pytest
from scipy.integrate import trapezoid

from labchart2cfd.processing.volume_balance import (
    balance_choices, balance_halves, default_keep, half_volumes, mass_volume_to_ml,
)

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


# ----------------------------------------------- what the preview offers

def test_balance_choices_offers_both_halves():
    flow = _breath()

    choices = balance_choices(flow, RATE)

    assert set(choices) == {"positive", "negative"}
    for keep, result in choices.items():
        assert np.array_equal(result.flow, balance_halves(flow, RATE, keep).flow)


def test_balance_choices_is_empty_when_a_half_holds_no_volume():
    """Keeping the empty half would scale the other one to zero."""
    assert balance_choices(np.abs(_breath()) + 0.01, RATE) == {}


def test_default_keep_is_the_half_the_extra_flow_is_not_in():
    flow = _breath()                      # above zero for t < 1 s, below after
    assert default_keep(flow, RATE, []) == "positive"
    assert default_keep(flow, RATE, [(0.2, 0.5)]) == "negative"
    assert default_keep(flow, RATE, [(1.3, 1.6)]) == "positive"


def test_half_volumes_integrate_over_the_time_column():
    time = np.arange(0, 2.0, 1 / RATE)
    flow = _breath()

    above, below = half_volumes(time, flow)

    assert above == pytest.approx(trapezoid(np.where(flow > 0, flow, 0.0), time))
    assert below == pytest.approx(trapezoid(np.where(flow < 0, flow, 0.0), time))
    assert below < 0 < above


def _blocks(ratio):
    """50 samples at -1 and 50 at +ratio, each block starting and ending at
    zero, so the half below zero holds 50 dx and the half above 50 dx x ratio."""
    return np.r_[0.0, -np.ones(50), 0.0, ratio * np.ones(50), 0.0]


@pytest.mark.parametrize("keep", ["positive", "negative"])
def test_a_half_with_almost_no_volume_is_refused(keep):
    """Resampling leaves residue samples across zero. One sample at +2.3e-19
    in a breath below zero gave a factor of 0 (a file of zeros) keeping the
    half above, or about 1e17 (a spike of 81 L/s) keeping the half below,
    and both passed the read back of the written file."""
    flow = -4e-4 * np.sin(np.linspace(0.0, np.pi, 61))[1:-1]
    flow[30] = 2.3e-19

    with pytest.raises(ValueError, match="almost no volume"):
        balance_halves(flow, RATE, keep=keep)
    assert balance_choices(flow, RATE) == {}


def test_a_half_of_two_percent_is_still_balanced():
    """The refusal starts below 1 percent of the other half."""
    result = balance_halves(_blocks(0.02), RATE, keep="negative")
    assert result.factor == pytest.approx(50.0)
    with pytest.raises(ValueError, match="almost no volume"):
        balance_halves(_blocks(0.005), RATE, keep="negative")
