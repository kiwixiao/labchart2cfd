"""Tests for CT rearrangement and image-index math.

Image model (shared with the synthetic-profile engine): grid lines bound the
images and an image is the slice between two lines, so a window of duration D
at resolution dt holds D/dt images bounded by D/dt + 1 lines.

A landmark is a point in time, so it snaps to the nearest image *boundary*:
the answer to "which image starts here", not "which image contains this".
"""
import numpy as np
import pytest

from labchart2cfd.processing.rearrangement import (
    compute_image_count,
    image_index_to_time,
    rearrange_at_landmark,
    rearrange_image_indices,
    time_to_image_index,
)


def test_image_count_is_slices_not_lines():
    # 2.0 s at 200 ms is 10 acquired images (11 boundary lines).
    assert compute_image_count(2.0, 0.2) == 10
    assert compute_image_count(4.0, 0.2) == 20
    assert compute_image_count(1.0, 0.2) == 5


def test_image_count_rounds_to_nearest_slice():
    assert compute_image_count(2.05, 0.2) == 10
    assert compute_image_count(2.15, 0.2) == 11


def test_image_count_rejects_bad_resolution():
    with pytest.raises(ValueError):
        compute_image_count(2.0, 0.0)


def test_landmark_snaps_to_nearest_image_boundary():
    # 0.75 s is nearest the line at 0.8 s, which starts image index 4.
    assert time_to_image_index(0.75, 0.0, 0.2) == 4
    assert time_to_image_index(0.69, 0.0, 0.2) == 3
    assert time_to_image_index(1.27, 0.0, 0.2) == 6


def test_landmark_index_is_clamped_to_available_images():
    # Without a bound the raw round would give 10, which has no slice.
    assert time_to_image_index(2.0, 0.0, 0.2, total_images=10) == 9
    assert time_to_image_index(-5.0, 0.0, 0.2, total_images=10) == 0


def test_image_index_to_time_is_the_inverse():
    assert image_index_to_time(4, 0.0, 0.2) == pytest.approx(0.8)
    assert image_index_to_time(6, 1.0, 0.2) == pytest.approx(2.2)


@pytest.mark.parametrize("t", [0.05, 0.31, 0.75, 1.27, 1.9])
def test_snapping_never_moves_a_landmark_more_than_half_an_image(t):
    idx = time_to_image_index(t, 0.0, 0.2)
    snapped = image_index_to_time(idx, 0.0, 0.2)

    assert abs(snapped - t) <= 0.1 + 1e-9  # dt/2


def test_rearrange_image_indices_rolls_to_cut():
    out = rearrange_image_indices(10, 6)

    assert out.tolist() == [6, 7, 8, 9, 0, 1, 2, 3, 4, 5]


def test_rearrange_at_landmark_splices_and_normalizes_time():
    time = np.linspace(0.0, 1.0, 11)
    data = np.arange(11, dtype=float)

    new_time, new_data = rearrange_at_landmark(time, data, 0.5)

    assert new_data[0] == 5.0          # starts at the landmark
    assert new_time[0] == 0.0          # time normalized
    assert len(new_data) == len(data)
