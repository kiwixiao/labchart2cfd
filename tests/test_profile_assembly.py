"""Tests for the CT synthetic-profile assembly engine.

The engine works on the already-processed window (process-then-assemble).
Image i sits at processed time t_i = (i-1)*dt; with dt=0.2 and 100 Hz the
samples-per-image (spi) is 20.
"""
import numpy as np
import pytest

from labchart2cfd.processing.profile_assembly import ProfilePart, assemble_parts


def _flow(n):
    """Ramp 0..n-1 so a sample's value equals its index (easy to assert)."""
    return np.arange(n, dtype=float)


def test_single_part_reorders_and_lengths():
    # N=11 images, dt=0.2, 100 Hz -> spi=20, window len = 201 samples
    flow = _flow(201)
    r = assemble_parts(flow, [ProfilePart(4, 9)], dt=0.2, sample_rate=100.0, total_images=11)
    # last (only) part is closed: samples [60 .. 160] inclusive -> 101 samples
    assert len(r.flow) == 101
    assert r.flow[0] == 60.0 and r.flow[-1] == 160.0
    # reorder map lists images 4..9 inclusive
    assert [m[1] for m in r.reorder_map] == [4, 5, 6, 7, 8, 9]
    assert r.reorder_map[0] == (1, 4, pytest.approx(0.6))
    # synthetic time starts at 0 at the sample rate
    assert r.time[0] == 0.0
    assert r.time[1] == pytest.approx(0.01)


def test_two_parts_join_no_double_count():
    flow = _flow(201)
    r = assemble_parts(
        flow, [ProfilePart(4, 9), ProfilePart(1, 3)],
        dt=0.2, sample_rate=100.0, total_images=11,
    )
    # part1 non-last half-open [60,160) = 100 samples; part2 last closed [0,40] = 41
    assert len(r.flow) == 141
    assert r.flow[0] == 60.0
    assert r.flow[99] == 159.0
    assert r.flow[100] == 0.0
    assert r.flow[-1] == 40.0
    assert [m[1] for m in r.reorder_map] == [4, 5, 6, 7, 8, 9, 1, 2, 3]


def test_pressure_assembled_parallel():
    flow = _flow(201)
    pres = _flow(201) * 10
    r = assemble_parts(
        flow, [ProfilePart(2, 4)],
        dt=0.2, sample_rate=100.0, total_images=11, pressure=pres,
    )
    assert r.pressure is not None
    assert len(r.pressure) == len(r.flow)
    assert r.pressure[0] == 200.0  # image 2 -> sample 20 -> 20*10


def test_empty_parts_raises():
    with pytest.raises(ValueError):
        assemble_parts(_flow(201), [], dt=0.2, sample_rate=100.0, total_images=11)


@pytest.mark.parametrize("a,b", [(0, 3), (3, 3), (5, 2), (9, 12)])
def test_invalid_parts_raise(a, b):
    with pytest.raises(ValueError):
        assemble_parts(_flow(201), [ProfilePart(a, b)], dt=0.2, sample_rate=100.0, total_images=11)
