"""Tests for the CT synthetic-profile assembly engine.

The engine works on the already-processed window (process-then-assemble).

Image model: grid lines bound the images, and image i is the slice between
line i and line i+1. So N images are bounded by N+1 lines, and one image owns
exactly dt of signal. With dt=0.2 at 100 Hz the samples-per-image (spi) is 20.
"""
import numpy as np
import pytest

from labchart2cfd.processing.profile_assembly import ProfilePart, assemble_parts

SPI = 20  # dt=0.2 * 100 Hz


def _flow(n):
    """Ramp 0..n-1 so a sample's value equals its index (easy to assert)."""
    return np.arange(n, dtype=float)


def _assemble(parts, flow=None, **kw):
    kw.setdefault("dt", 0.2)
    kw.setdefault("sample_rate", 100.0)
    kw.setdefault("total_images", 10)
    return assemble_parts(_flow(201) if flow is None else flow, parts, **kw)


def test_single_part_covers_one_slice_per_image():
    # Images 4..9 = 6 slices = 6*20 = 120 samples, starting at line 4 (sample 60).
    r = _assemble([ProfilePart(4, 9)])

    assert len(r.flow) == 6 * SPI
    assert r.flow[0] == 60.0
    assert r.flow[-1] == 179.0
    assert [m[1] for m in r.reorder_map] == [4, 5, 6, 7, 8, 9]
    assert r.reorder_map[0] == (1, 4, pytest.approx(0.6))
    assert r.time[0] == 0.0
    assert r.time[1] == pytest.approx(0.01)


def test_two_parts_concatenate_in_order():
    r = _assemble([ProfilePart(4, 9), ProfilePart(1, 3)])

    assert len(r.flow) == (6 + 3) * SPI
    assert r.flow[0] == 60.0            # part 1 starts at image 4
    assert r.flow[119] == 179.0         # part 1 ends at image 9
    assert r.flow[120] == 0.0           # part 2 starts at image 1
    assert [m[1] for m in r.reorder_map] == [4, 5, 6, 7, 8, 9, 1, 2, 3]


def test_single_image_part_is_allowed():
    """a == b is legal now: one image contributes exactly dt of signal."""
    r = _assemble([ProfilePart(5, 5)])

    assert len(r.flow) == SPI
    assert r.flow[0] == 80.0            # line 5 -> sample (5-1)*20
    assert [m[1] for m in r.reorder_map] == [5]


@pytest.mark.parametrize(
    "parts",
    [
        [ProfilePart(1, 1)],
        [ProfilePart(4, 9)],
        [ProfilePart(4, 9), ProfilePart(1, 3)],
        [ProfilePart(10, 10), ProfilePart(1, 2), ProfilePart(5, 7)],
    ],
)
def test_flow_and_map_never_disagree(parts):
    """The invariant that was broken before: one map row per image, spi samples each."""
    r = _assemble(parts)

    assert len(r.flow) == len(r.reorder_map) * SPI


def test_grid_offset_shifts_every_slice():
    # 0.1 s offset at 100 Hz shifts the grid origin by 10 samples.
    r = _assemble([ProfilePart(1, 1)], grid_offset_s=0.1)

    assert len(r.flow) == SPI
    assert r.flow[0] == 10.0


def test_pressure_assembled_parallel():
    r = _assemble([ProfilePart(2, 4)], pressure=_flow(201) * 10)

    assert r.pressure is not None
    assert len(r.pressure) == len(r.flow)
    assert r.pressure[0] == 200.0       # line 2 -> sample 20 -> 20*10


def test_empty_parts_raises():
    with pytest.raises(ValueError):
        _assemble([])


@pytest.mark.parametrize("a,b", [(0, 3), (5, 2), (9, 12)])
def test_invalid_parts_raise(a, b):
    with pytest.raises(ValueError):
        _assemble([ProfilePart(a, b)])


def test_part_past_end_of_flow_raises():
    """A slice must not read past the processed window."""
    with pytest.raises(ValueError):
        _assemble([ProfilePart(10, 10)], flow=_flow(150))
