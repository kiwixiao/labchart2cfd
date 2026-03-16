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
        """CSV should use Windows line endings (CRLF)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.csv"
            time = np.array([0.0])
            flow = np.array([0.001])

            export_flow_csv(filepath, time, flow)

            with open(filepath, "rb") as f:
                content = f.read()

            assert b"\r\n" in content

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
