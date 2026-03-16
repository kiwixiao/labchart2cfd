"""Export data to Star-CCM+ compatible CSV format.

Star-CCM+ expects CSV files with:
- Header row with quoted column names
- Data rows with 5 decimal precision
- Windows-style line endings (CRLF)
"""

from pathlib import Path
from typing import Union

import numpy as np


def export_flow_csv(filepath, time, mass_flow, precision=5):
    # type: (Union[str, Path], np.ndarray, np.ndarray, int) -> Path
    """Export mass flow rate data to Star-CCM+ compatible CSV.

    Args:
        filepath: Output file path
        time: Time array in seconds
        mass_flow: Mass flow rate array in kg/s
        precision: Decimal precision for output values

    Returns:
        Path to the created file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Star-CCM+ expects quoted headers
    header = '"time (s)","Massflowrate (kg/s)"'

    # Build data lines with specified precision
    fmt = "%.{}f".format(precision)
    lines = [header]
    for t, m in zip(time, mass_flow):
        lines.append("{},{}".format(fmt % t, fmt % m))

    # Write with Windows line endings (as MATLAB's fprintf does)
    with open(str(filepath), "w") as f:
        f.write("\r\n".join(lines) + "\r\n")

    return filepath


def export_pressure_csv(filepath, time, pressure, precision=5):
    # type: (Union[str, Path], np.ndarray, np.ndarray, int) -> Path
    """Export pressure data to Star-CCM+ compatible CSV.

    Args:
        filepath: Output file path
        time: Time array in seconds
        pressure: Pressure array in Pa

    Returns:
        Path to the created file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Star-CCM+ expects quoted headers
    header = '"time (s)","Pressure (Pa)"'

    # Build data lines with specified precision
    fmt = "%.{}f".format(precision)
    lines = [header]
    for t, p in zip(time, pressure):
        lines.append("{},{}".format(fmt % t, fmt % p))

    # Write with Windows line endings
    with open(str(filepath), "w") as f:
        f.write("\r\n".join(lines) + "\r\n")

    return filepath


def export_generic_csv(filepath, time, data, data_header, precision=5):
    # type: (Union[str, Path], np.ndarray, np.ndarray, str, int) -> Path
    """Export generic time-series data to Star-CCM+ compatible CSV.

    Args:
        filepath: Output file path
        time: Time array in seconds
        data: Data array
        data_header: Header name for the data column (will be quoted)
        precision: Decimal precision for output values

    Returns:
        Path to the created file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Ensure header is quoted
    if not data_header.startswith('"'):
        data_header = '"{}"'.format(data_header)

    header = '"time (s)",{}'.format(data_header)

    # Build data lines
    fmt = "%.{}f".format(precision)
    lines = [header]
    for t, d in zip(time, data):
        lines.append("{},{}".format(fmt % t, fmt % d))

    # Write with Windows line endings
    with open(str(filepath), "w") as f:
        f.write("\r\n".join(lines) + "\r\n")

    return filepath
