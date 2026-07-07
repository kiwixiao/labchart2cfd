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

    def test_export_flow_csv_precision(self):
        """Flow CSV should use specified precision."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_flow.csv"
            time = np.array([0.0, 0.1])
            flow = np.array([0.123456789, -0.987654321])

            export_flow_csv(filepath, time, flow, precision=5)

            with open(filepath) as f:
                lines = f.readlines()

            # Check data lines have 5 decimal places
            data_line = lines[1].strip()
            parts = data_line.split(",")
            assert parts[1] == "0.12346"  # rounded to 5 decimals

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
            assert lines[1] == "1,4,0.60000"
            assert lines[3] == "3,1,0.00000"
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
                b"0.00000,1.50000\r\n"
                b"0.10000,-2.25000\r\n"
            )
            assert pp.read_bytes() == (
                b'"time (s)","Pressure (Pa)"\r\n'
                b"0.00000,100.00000\r\n"
                b"0.10000,200.00000\r\n"
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
