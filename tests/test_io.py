"""Tests for I/O functions.

Tests for loading LabChart .mat files and exporting CSV.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from labchart2cfd.io.csv_export import (
    export_flow_csv,
    export_pressure_csv,
    export_generic_csv,
    export_reorder_map_csv,
)


class TestCSVExport:
    """Tests for CSV export functions."""

    def test_export_flow_csv_creates_file(self):
        """Flow CSV export should create a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_flow.csv"
            time = np.linspace(0, 1, 100)
            flow = np.sin(2 * np.pi * time)

            result = export_flow_csv(filepath, time, flow)

            assert result.exists()
            assert result == filepath

    def test_export_flow_csv_header(self):
        """Flow CSV should have correct header."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_flow.csv"
            time = np.array([0.0, 0.1, 0.2])
            flow = np.array([0.001, 0.002, 0.003])

            export_flow_csv(filepath, time, flow)

            with open(filepath) as f:
                header = f.readline().strip()

            assert header == '"time (s)","Massflowrate (kg/s)"'

    def test_export_flow_csv_keeps_small_magnitudes(self):
        """Mass flow is ~1e-4 kg/s; the format must not quantize it away.

        The old fixed %.5f collapsed a pediatric waveform to ~61 distinct
        values (1.6% error at peak). Scientific notation keeps the shape at
        any magnitude.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_flow.csv"
            time = np.array([0.0, 0.01])
            flow = np.array([3.00123456e-4, -2.98765432e-4])

            export_flow_csv(filepath, time, flow)

            with open(filepath) as f:
                lines = f.read().splitlines()

            assert lines[1] == "0.000000,3.001234560e-04"
            assert lines[2] == "0.010000,-2.987654320e-04"

    def test_export_rejects_length_mismatch(self):
        """zip() used to truncate silently: 10 times + 7 values wrote 7 rows."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="length"):
                export_flow_csv(
                    Path(tmpdir) / "x.csv", np.arange(10.0), np.arange(7.0)
                )

    def test_export_rejects_non_finite_values(self):
        """NaN/inf must not reach a CFD boundary-condition file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="finite"):
                export_flow_csv(
                    Path(tmpdir) / "x.csv",
                    np.array([0.0, 0.01]),
                    np.array([1.0, np.nan]),
                )

    def test_export_pressure_csv_header(self):
        """Pressure CSV should have correct header."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_pressure.csv"
            time = np.array([0.0, 0.1])
            pressure = np.array([100.0, 200.0])

            export_pressure_csv(filepath, time, pressure)

            with open(filepath) as f:
                header = f.readline().strip()

            assert header == '"time (s)","Pressure (Pa)"'

    def test_export_csv_line_endings(self):
        """CSV uses CRLF exactly — never the doubled \\r\\r\\n Windows produced."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.csv"
            time = np.array([0.0, 0.1, 0.2])
            flow = np.array([0.001, 0.002, 0.003])

            export_flow_csv(filepath, time, flow)

            with open(filepath, "rb") as f:
                content = f.read()

            # 1 header + 3 data rows, each terminated by exactly one CRLF
            assert content.count(b"\r\n") == 4
            assert b"\r\r\n" not in content
            # no bare CR left once CRLFs are removed
            assert b"\r" not in content.replace(b"\r\n", b"")

    def test_export_reorder_map_csv(self):
        """Reorder map writes header + (new_frame, original_image, time) rows."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "map.csv"
            reorder_map = [(1, 4, 0.6), (2, 5, 0.8), (3, 1, 0.0)]

            export_reorder_map_csv(filepath, reorder_map)

            with open(filepath, "rb") as f:
                content = f.read()
            lines = content.decode().split("\r\n")
            assert lines[0] == "new_frame,original_image,original_time_s"
            assert lines[1] == "1,4,0.600000"
            assert lines[3] == "3,1,0.000000"
            assert b"\r\r\n" not in content

    def test_export_frame_table_csv_writes_seven_columns(self):
        """The frame table keeps the map's first three columns and adds where
        each frame sits in the profile and any extra flow after it."""
        from labchart2cfd.io.csv_export import export_frame_table_csv
        from labchart2cfd.processing.profile_assembly import FrameRow
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "map.csv"
            frames = [
                FrameRow(1, 4, 0.6, 0.0, 0.2, 0.0, None),
                FrameRow(2, 5, 0.8, 0.2, 0.4, 0.3, 1.3),
            ]

            export_frame_table_csv(filepath, frames)

            with open(filepath, "rb") as f:
                content = f.read()
            lines = content.decode().split("\r\n")
            assert lines[0] == (
                "new_frame,original_image,original_time_s,profile_start_s,"
                "profile_end_s,extra_flow_s,extra_flow_until_s"
            )
            assert lines[1] == "1,4,0.600000,0.000000,0.200000,0.000000,"
            assert lines[2] == "2,5,0.800000,0.200000,0.400000,0.300000,1.300000"
            assert b"\r\r\n" not in content

    def test_export_generic_csv_custom_header(self):
        """Generic CSV export should use custom header."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_generic.csv"
            time = np.array([0.0, 0.1])
            data = np.array([1.0, 2.0])

            export_generic_csv(filepath, time, data, "Custom Value (units)")

            with open(filepath) as f:
                header = f.readline().strip()

            assert header == '"time (s)","Custom Value (units)"'

    def test_flow_and_pressure_byte_exact(self):
        """Pin the exact bytes of the flow and pressure writers so the
        de-duplication refactor stays provably output-preserving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            time = np.array([0.0, 0.1])
            flow = np.array([1.5, -2.25])
            pressure = np.array([100.0, 200.0])

            fp = Path(tmpdir) / "f.csv"
            pp = Path(tmpdir) / "p.csv"
            export_flow_csv(fp, time, flow)
            export_pressure_csv(pp, time, pressure)

            assert fp.read_bytes() == (
                b'"time (s)","Massflowrate (kg/s)"\r\n'
                b"0.000000,1.500000000e+00\r\n"
                b"0.100000,-2.250000000e+00\r\n"
            )
            assert pp.read_bytes() == (
                b'"time (s)","Pressure (Pa)"\r\n'
                b"0.000000,1.000000000e+02\r\n"
                b"0.100000,2.000000000e+02\r\n"
            )


class TestLabChartData:
    """Tests for LabChartData class."""

    def test_data_indexing_1_based(self):
        """LabChartData should use 1-based indexing like MATLAB."""
        from labchart2cfd.io.labchart import LabChartData

        # Create mock data
        data_cell = [[np.array([1, 2, 3]), np.array([4, 5, 6])]]
        time_cell = [[np.array([0, 1, 2]), np.array([0, 1, 2])]]

        lcd = LabChartData(
            data_cell=data_cell,
            time_cell=time_cell,
            num_channels=1,
            num_blocks=2,
        )

        # Row 1, Column 1 should return first array
        assert_allclose(lcd.get_data(1, 1), np.array([1, 2, 3]))
        # Row 1, Column 2 should return second array
        assert_allclose(lcd.get_data(1, 2), np.array([4, 5, 6]))

    def test_is_block_empty(self):
        """is_block_empty should detect None entries."""
        from labchart2cfd.io.labchart import LabChartData

        data_cell = [[np.array([1, 2, 3]), None]]
        time_cell = [[np.array([0, 1, 2]), None]]

        lcd = LabChartData(
            data_cell=data_cell,
            time_cell=time_cell,
            num_channels=1,
            num_blocks=2,
        )

        assert not lcd.is_block_empty(1, 1)
        assert lcd.is_block_empty(1, 2)

    def test_get_data_empty_block_raises(self):
        """Accessing empty block should raise ValueError."""
        from labchart2cfd.io.labchart import LabChartData

        data_cell = [[None]]
        time_cell = [[None]]

        lcd = LabChartData(
            data_cell=data_cell,
            time_cell=time_cell,
            num_channels=1,
            num_blocks=1,
        )

        with pytest.raises(ValueError, match="empty"):
            lcd.get_data(1, 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
