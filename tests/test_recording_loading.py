"""Tests for opening a recording by file type, and for native .adicht files.

A LabChart recording reaches the application either as LabChart's own MATLAB
export (.mat) or as the native file (.adicht). Both must end up as the same
``LabChartData``.

Real .adicht files are patient recordings and cannot live in the repository,
so the test that compares a native file with LabChart's export of it runs only
when a local pair is named:

    LABCHART2CFD_ADICHT_FILE=/path/rec.adicht LABCHART2CFD_ADICHT_MAT=/path/rec.mat pytest tests/test_recording_loading.py
"""
import os
from pathlib import Path

import numpy as np
import pytest

from labchart2cfd.io.adicht import AdichtFormatError
from labchart2cfd.io.labchart import LabChartData
from labchart2cfd.io.recording import RECORDING_SUFFIXES, load_recording


# ------------------------------------------------------------ by file type

def test_a_mat_export_is_read_by_the_mat_loader(mat_file):
    data = load_recording(mat_file)

    assert isinstance(data, LabChartData)
    assert (data.num_channels, data.num_blocks) == (4, 3)


def test_an_adicht_file_is_sent_to_the_adicht_loader(monkeypatch, tmp_path):
    seen = []

    def _fake(path, notes=None, progress=None):
        seen.append(Path(path))
        return "decoded"

    monkeypatch.setattr("labchart2cfd.io.recording.load_adicht", _fake)

    assert load_recording(tmp_path / "rec.adicht") == "decoded"
    assert load_recording(tmp_path / "REC.ADICHT") == "decoded"    # any case
    assert [p.name for p in seen] == ["rec.adicht", "REC.ADICHT"]


def test_an_unknown_file_type_is_refused_by_name(tmp_path):
    with pytest.raises(ValueError) as error:
        load_recording(tmp_path / "rec.csv")

    assert ".mat" in str(error.value) and ".adicht" in str(error.value)


def test_both_recording_types_are_listed():
    assert set(RECORDING_SUFFIXES) == {".mat", ".adicht"}


def test_a_file_that_is_not_adicht_is_refused_with_the_reason(tmp_path):
    """The reader must refuse, never return numbers from a file it cannot read."""
    fake = tmp_path / "rec.adicht"
    fake.write_bytes(b"this is not a LabChart recording " * 100)

    with pytest.raises(AdichtFormatError) as error:
        load_recording(fake)

    assert "ADInstruments" in str(error.value)


def test_an_empty_adicht_file_is_refused(tmp_path):
    empty = tmp_path / "rec.adicht"
    empty.write_bytes(b"")

    with pytest.raises(AdichtFormatError):
        load_recording(empty)


# ------------------------------------- decoding, with a stand-in for the file

class _FakeAdicht:
    """Stands in for AdichtFile: 3 channels, 2 records, channel 3 not reproducible."""

    n_channels, n_records = 3, 2
    titles = ["TRIGGER", "FLOW", "CALC"]
    n_samples = [[4, 0], [4, 3], [4, 3]]
    sample_rate = [[10.0, 10.0], [10.0, 10.0], [10.0, 10.0]]

    def __init__(self, path):
        self.path = Path(path)
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.closed = True

    def is_exportable(self, channel):
        return channel != 2

    def data(self, channel, record):
        # A value that single precision cannot hold exactly.
        return np.full(self.n_samples[channel][record], 0.1 + channel, dtype=np.float64)


@pytest.fixture
def fake_adicht(monkeypatch):
    monkeypatch.setattr("labchart2cfd.io.recording.AdichtFile", _FakeAdicht)
    monkeypatch.setattr(
        "labchart2cfd.io.recording.plan_export",
        lambda adi: ([], ["LEFT OUT: channel 3 (CALC) carries a calculation."]),
    )


def test_native_values_are_rounded_as_labcharts_export_stores_them(fake_adicht, tmp_path):
    """Opening the native file must give the numbers the export gives."""
    data = load_recording(tmp_path / "rec.adicht")

    assert data.get_data(2, 1)[0] == np.float64(np.float32(1.1))
    assert data.get_data(2, 1)[0] != 1.1
    assert data.get_data(2, 1).dtype == np.float64


def test_native_time_starts_at_zero_in_every_record(fake_adicht, tmp_path):
    data = load_recording(tmp_path / "rec.adicht")

    assert np.array_equal(data.get_time(2, 1), [0.0, 0.1, 0.2, 0.3])
    assert np.array_equal(data.get_time(2, 2), [0.0, 0.1, 0.2])


def test_a_channel_that_cannot_be_reproduced_is_left_empty_and_named(fake_adicht, tmp_path):
    notes = []

    data = load_recording(tmp_path / "rec.adicht", notes=notes)

    assert data.is_block_empty(3, 1) and data.is_block_empty(3, 2)
    assert notes == ["LEFT OUT: channel 3 (CALC) carries a calculation."]


def test_a_record_without_samples_is_an_empty_block(fake_adicht, tmp_path):
    data = load_recording(tmp_path / "rec.adicht")

    assert data.is_block_empty(1, 2)
    assert not data.is_block_empty(1, 1)


def test_native_recording_keeps_its_titles_and_shape(fake_adicht, tmp_path):
    data = load_recording(tmp_path / "rec.adicht")

    assert (data.num_channels, data.num_blocks) == (3, 2)
    assert data.titles == ["TRIGGER", "FLOW", "CALC"]


def test_decoding_reports_progress_for_each_channel_record(fake_adicht, tmp_path):
    """The decode takes many seconds on a real file; the caller shows progress."""
    calls = []

    load_recording(tmp_path / "rec.adicht",
                   progress=lambda done, total, what: calls.append((done, total, what)))

    # Channel 3 is left out and one record of channel 1 is empty: 3 to decode.
    assert [c[:2] for c in calls] == [(1, 3), (2, 3), (3, 3)]
    assert "FLOW" in calls[1][2] and "record 1" in calls[1][2]


# ----------------------------------------------- a real pair, when available

_ADICHT = os.environ.get("LABCHART2CFD_ADICHT_FILE", "")
_EXPORT = os.environ.get("LABCHART2CFD_ADICHT_MAT", "")
_have_pair = bool(_ADICHT and _EXPORT) and Path(_ADICHT).exists() and Path(_EXPORT).exists()


@pytest.mark.skipif(
    not _have_pair,
    reason="needs a real recording: set LABCHART2CFD_ADICHT_FILE and LABCHART2CFD_ADICHT_MAT",
)
def test_opening_the_adicht_gives_the_same_numbers_as_labcharts_export():
    """The acceptance test: native file and LabChart's export are one recording."""
    notes = []
    native = load_recording(_ADICHT, notes=notes)
    export = load_recording(_EXPORT)

    assert (native.num_channels, native.num_blocks) == (export.num_channels, export.num_blocks)
    assert [t.strip() for t in native.titles] == [t.strip() for t in export.titles]
    for ch in range(1, export.num_channels + 1):
        for bl in range(1, export.num_blocks + 1):
            where = f"channel {ch}, block {bl}"
            assert native.is_block_empty(ch, bl) == export.is_block_empty(ch, bl), where
            if export.is_block_empty(ch, bl):
                continue
            assert np.array_equal(native.get_data(ch, bl), export.get_data(ch, bl)), where
            assert np.array_equal(native.get_time(ch, bl), export.get_time(ch, bl)), where
