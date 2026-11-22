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


def test_map_time_is_absolute_while_flow_time_starts_at_zero():
    """The two files speak different clocks, on purpose.

    The flow CSV is the synthetic breath, so its axis starts at 0. The reorder
    map points back at the scan, so its time must be findable on the original
    recording's axis.
    """
    anchor = 1548.930
    r = _assemble([ProfilePart(9, 15)], flow=_flow(301),
                  total_images=15, window_start_s=anchor)

    # Output clock: the assembled profile always begins at 0.
    assert r.time[0] == 0.0

    # Input clock: image 9 begins 8 slices after the anchor.
    assert r.reorder_map[0] == (1, 9, pytest.approx(anchor + 8 * 0.2))
    assert r.reorder_map[1][2] == pytest.approx(anchor + 9 * 0.2)


def test_map_time_defaults_to_window_relative():
    """Without an anchor the map stays relative, so existing callers are safe."""
    r = _assemble([ProfilePart(4, 9)])

    assert r.reorder_map[0] == (1, 4, pytest.approx(0.6))


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


# ------------- design 2026-11-22: any Temporal Res, extra flow, frame times

def test_fractional_dt_rounds_each_boundary_on_its_own():
    """275 ms is 27.5 samples at 100 Hz. Each line goes to the sample nearest
    k x dt, so the slices are 28, 27, 28, 27 samples and never drift."""
    r = _assemble([ProfilePart(1, 4)], flow=_flow(200), dt=0.275, total_images=4)

    starts = [round(f.profile_start_s * 100) for f in r.frames]
    ends = [round(f.profile_end_s * 100) for f in r.frames]
    assert starts == [0, 28, 55, 83]
    assert ends == [28, 55, 83, 110]
    assert len(r.flow) == 110
    assert r.flow[28] == 28.0 and r.flow[-1] == 109.0


def test_frame_times_name_the_samples_used():
    r = _assemble([ProfilePart(3, 4), ProfilePart(1, 2)], flow=_flow(200),
                  dt=0.275, total_images=4, window_start_s=10.0)

    boundaries = [0, 28, 55, 83, 110]
    for f in r.frames:
        assert f.original_time_s == pytest.approx(10.0 + boundaries[f.original_image - 1] / 100)
    assert [m[2] for m in r.reorder_map] == [f.original_time_s for f in r.frames]


def test_extra_flow_follows_its_part():
    """Images 4..6, then recorded flow until 1.5 s, then images 1..3."""
    r = _assemble([ProfilePart(4, 6, extra_until_s=1.5), ProfilePart(1, 3)],
                  flow=_flow(300), total_images=6)

    expected = np.concatenate([np.arange(60, 150), np.arange(0, 60)]).astype(float)
    assert np.array_equal(r.flow, expected)
    third, fourth = r.frames[2], r.frames[3]
    assert third.original_image == 6
    assert third.profile_end_s == pytest.approx(0.6)
    assert third.extra_flow_s == pytest.approx(0.3)
    assert third.extra_flow_until_s == pytest.approx(1.5)
    assert fourth.profile_start_s == pytest.approx(0.9)
    assert r.frames[0].extra_flow_s == 0.0 and r.frames[0].extra_flow_until_s is None
    assert r.extra_spans == [(pytest.approx(0.6), pytest.approx(0.9))]


def test_extra_flow_after_the_last_part_is_allowed():
    r = _assemble([ProfilePart(1, 3, extra_until_s=1.0)], flow=_flow(300), total_images=6)

    assert len(r.flow) == 100
    assert r.frames[-1].extra_flow_s == pytest.approx(0.4)


def test_extra_flow_before_the_next_image_is_refused():
    """Image 4 follows image 3 in the recording: extra flow after 3 would
    play the start of image 4 twice."""
    with pytest.raises(ValueError, match="next image"):
        _assemble([ProfilePart(1, 3, extra_until_s=0.8), ProfilePart(4, 6)],
                  flow=_flow(300), total_images=6)


@pytest.mark.parametrize("until, message", [(0.5, "after"), (3.5, "recording")])
def test_extra_flow_must_end_after_its_image_and_inside_the_flow(until, message):
    with pytest.raises(ValueError, match=message):
        _assemble([ProfilePart(1, 3, extra_until_s=until)], flow=_flow(300), total_images=6)
