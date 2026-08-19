"""Tests for the LabChart .mat reader.

The reader previously had zero coverage and trusted the file's own index
tables. These tests pin the happy path against synthetic files and, more
importantly, pin that corrupted or truncated files are REJECTED rather than
silently loading clamped or misattributed data.
"""
import numpy as np
import pytest
from scipy.io import savemat

from labchart2cfd.io.labchart import describe_mat_structure, load_labchart_mat


def _write_mat(path, n=100, fs=100.0, tweak=None):
    """Two channels, one block, ch1 = ramp, ch2 = ramp*10."""
    payload = {
        "data": np.concatenate([np.arange(n, dtype=float),
                                np.arange(n, dtype=float) * 10]),
        "datastart": np.array([[1.0], [n + 1.0]]),
        "dataend": np.array([[float(n)], [2.0 * n]]),
        "samplerate": np.array([[fs], [fs]]),
        "titles": np.array(["flow ", "press"]),
    }
    if tweak:
        payload.update(tweak)
    savemat(str(path), payload)
    return path


def test_valid_file_loads_with_correct_indexing(tmp_path):
    mat = _write_mat(tmp_path / "ok.mat")

    d = load_labchart_mat(mat)

    assert d.num_channels == 2 and d.num_blocks == 1
    assert d.get_data(1, 1)[0] == 0.0 and d.get_data(1, 1)[-1] == 99.0
    assert d.get_data(2, 1)[0] == 0.0 and d.get_data(2, 1)[-1] == 990.0
    assert d.get_time(1, 1)[-1] == pytest.approx(0.99)
    assert d.titles == ["flow", "press"]


def test_truncated_file_is_rejected_not_clamped(tmp_path):
    """dataend beyond the data must raise. Python slicing used to clamp, so a
    partially copied file loaded 100 samples while claiming 1000, and validate
    reported the declared length while process exported the short one."""
    mat = _write_mat(tmp_path / "trunc.mat",
                     tweak={"dataend": np.array([[1000.0], [2000.0]])})

    with pytest.raises(ValueError, match="beyond"):
        load_labchart_mat(mat)


def test_nonpositive_datastart_other_than_sentinel_is_rejected(tmp_path):
    """Only exactly -1 means 'empty block'. A 0 or -5 used to become a negative
    Python index and returned ANOTHER channel's tail samples as this block."""
    mat = _write_mat(tmp_path / "bad.mat",
                     tweak={"datastart": np.array([[0.0], [101.0]])})

    with pytest.raises(ValueError, match="datastart"):
        load_labchart_mat(mat)


def test_sentinel_minus_one_still_means_empty(tmp_path):
    mat = _write_mat(tmp_path / "empty.mat",
                     tweak={"datastart": np.array([[-1.0], [101.0]]),
                            "dataend": np.array([[-1.0], [200.0]])})

    d = load_labchart_mat(mat)

    assert d.is_block_empty(1, 1)
    assert not d.is_block_empty(2, 1)


def test_missing_samplerate_gives_a_named_error(tmp_path):
    path = tmp_path / "nosr.mat"
    savemat(str(path), {
        "data": np.arange(10.0),
        "datastart": np.array([[1.0]]),
        "dataend": np.array([[10.0]]),
    })

    with pytest.raises(ValueError, match="samplerate"):
        load_labchart_mat(path)


def test_scaleunits_without_scaleoffset_gives_a_named_error(tmp_path):
    mat = _write_mat(tmp_path / "noso.mat",
                     tweak={"scaleunits": np.array([[1.0], [1.0]])})

    with pytest.raises(ValueError, match="scaleoffset"):
        load_labchart_mat(mat)


def test_cell_array_titles_do_not_crash(tmp_path):
    """A file re-saved from MATLAB stores titles as a cell array (object
    dtype), which used to crash the chr() fallback."""
    mat = _write_mat(tmp_path / "cell.mat",
                     tweak={"titles": np.array([["flow", "press"]], dtype=object)})

    d = load_labchart_mat(mat)

    assert d.titles == ["flow", "press"]


def test_blank_title_keeps_every_channel_in_its_own_slot(tmp_path):
    """A blank cell used to be dropped, shifting later channels up a slot so
    validate named the wrong channel and the user picked the wrong --row."""
    mat = _write_mat(
        tmp_path / "blank.mat",
        tweak={"titles": np.array([["", "press"]], dtype=object)},
    )

    d = load_labchart_mat(mat)

    assert d.titles == ["", "press"]     # not ["press", ...]


def test_nested_cell_titles_are_unwrapped(tmp_path):
    mat = _write_mat(
        tmp_path / "nested.mat",
        tweak={"titles": np.array(
            [[np.array(["flow"], dtype=object), np.array(["press"], dtype=object)]],
            dtype=object)},
    )

    d = load_labchart_mat(mat)

    assert d.titles == ["flow", "press"]


def test_scaleoffset_without_scaleunits_is_rejected(tmp_path):
    """The pair is all-or-nothing; accepting one silently applied no scaling."""
    mat = _write_mat(tmp_path / "half.mat",
                     tweak={"scaleoffset": np.array([[1.0], [1.0]])})

    with pytest.raises(ValueError, match="scaleunits"):
        load_labchart_mat(mat)


def test_describe_and_load_agree_on_scaling_fields(tmp_path):
    """validate must not bless a file process refuses."""
    mat = _write_mat(tmp_path / "half2.mat",
                     tweak={"scaleunits": np.array([[1.0], [1.0]])})

    info = describe_mat_structure(mat)

    assert "error" in info
    with pytest.raises(ValueError):
        load_labchart_mat(mat)


def test_nan_sample_rate_is_rejected(tmp_path):
    mat = _write_mat(tmp_path / "nan.mat",
                     tweak={"samplerate": np.array([[np.nan], [100.0]])})

    with pytest.raises(ValueError, match="sample rate"):
        load_labchart_mat(mat)


def test_describe_returns_error_dict_on_mismatched_shapes(tmp_path):
    """Contract: describe returns {'error': ...}, never raises."""
    mat = _write_mat(tmp_path / "shape.mat",
                     tweak={"samplerate": np.array([[100.0]])})   # 1x1 vs 2x1

    info = describe_mat_structure(mat)

    assert "error" in info


def test_describe_requires_the_data_variable(tmp_path):
    """describe used to report a full healthy structure for a file with no
    data at all, so validate passed and process failed."""
    path = tmp_path / "nodata.mat"
    savemat(str(path), {
        "datastart": np.array([[1.0]]),
        "dataend": np.array([[10.0]]),
        "samplerate": np.array([[100.0]]),
    })

    info = describe_mat_structure(path)

    assert "error" in info


def test_describe_flags_inverted_block_indices(tmp_path):
    mat = _write_mat(tmp_path / "inv.mat",
                     tweak={"dataend": np.array([[40.0], [60.0]]),
                            "datastart": np.array([[80.0], [101.0]])})

    info = describe_mat_structure(mat)

    ch1 = next(b for b in info["blocks"] if b["channel"] == 1)
    assert ch1.get("corrupt") is True
