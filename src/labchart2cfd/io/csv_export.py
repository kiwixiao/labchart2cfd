"""Export data to CFD-compatible CSV format (Star-CCM+ conventions).

Star-CCM+ expects CSV files with:
- Header row with quoted column names
- Windows-style line endings (CRLF)

Number formats: the time column is fixed-point microseconds (%.6f); the data
column is scientific notation (%.<precision>e). Mass flow is of order 1e-4
kg/s, and the old fixed %.5f quantized a pediatric waveform to ~61 distinct
values (1.6% error at peak). Scientific notation keeps the waveform shape at
any magnitude.

All writers validate that arrays have matching lengths (zip used to truncate
silently) and contain only finite values (NaN/inf must never reach a CFD
boundary-condition file).
"""

from pathlib import Path
from typing import Union

import numpy as np


def _validate_series(time, data, data_name):
    # type: (np.ndarray, np.ndarray, str) -> None
    """Reject silently-truncating or non-finite input before writing."""
    if len(time) != len(data):
        raise ValueError(
            f"time and {data_name} have different lengths "
            f"({len(time)} vs {len(data)}); refusing to write a truncated CSV."
        )
    if len(time) == 0:
        raise ValueError(f"time and {data_name} are empty; nothing to write.")
    if not np.all(np.isfinite(time)):
        raise ValueError("time contains non-finite values (NaN or inf).")
    if not np.all(np.isfinite(data)):
        raise ValueError(f"{data_name} contains non-finite values (NaN or inf).")


def export_generic_csv(filepath, time, data, data_header, precision=9):
    # type: (Union[str, Path], np.ndarray, np.ndarray, str, int) -> Path
    """Export generic time-series data to Star-CCM+ compatible CSV.

    Args:
        filepath: Output file path
        time: Time array in seconds
        data: Data array
        data_header: Header name for the data column (will be quoted)
        precision: Significant decimal digits of the data mantissa

    Returns:
        Path to the created file

    Raises:
        ValueError: On length mismatch, empty input, or non-finite values.
    """
    _validate_series(time, data, "data")

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Ensure header is quoted
    if not data_header.startswith('"'):
        data_header = '"{}"'.format(data_header)

    header = '"time (s)",{}'.format(data_header)

    data_fmt = "%.{}e".format(precision)
    lines = [header]
    for t, d in zip(time, data):
        lines.append("{},{}".format("%.6f" % t, data_fmt % d))

    # CRLF explicitly; newline="" stops Windows doubling it to \r\r\n.
    # ascii so the bytes are identical on every platform locale.
    with open(str(filepath), "w", newline="", encoding="ascii") as f:
        f.write("\r\n".join(lines) + "\r\n")

    return filepath


def export_flow_csv(filepath, time, mass_flow, precision=9):
    # type: (Union[str, Path], np.ndarray, np.ndarray, int) -> Path
    """Export mass flow rate data to Star-CCM+ compatible CSV.

    Args:
        filepath: Output file path
        time: Time array in seconds
        mass_flow: Mass flow rate array in kg/s
        precision: Significant decimal digits of the mantissa

    Returns:
        Path to the created file
    """
    return export_generic_csv(
        filepath, time, mass_flow, '"Massflowrate (kg/s)"', precision
    )


def export_pressure_csv(filepath, time, pressure, precision=9):
    # type: (Union[str, Path], np.ndarray, np.ndarray, int) -> Path
    """Export pressure data to Star-CCM+ compatible CSV.

    Args:
        filepath: Output file path
        time: Time array in seconds
        pressure: Pressure array in Pa
        precision: Significant decimal digits of the mantissa

    Returns:
        Path to the created file
    """
    return export_generic_csv(
        filepath, time, pressure, '"Pressure (Pa)"', precision
    )


def export_reorder_map_csv(filepath, reorder_map):
    # type: (Union[str, Path], list) -> Path
    """Export the CT image reorder map to CSV.

    The map tells you, for each position in the assembled synthetic profile,
    which original image it came from and where that image sits in the
    original recording (absolute time), so the CT image stack can be
    re-aligned to the synthetic breathing cycle.

    Args:
        filepath: Output file path.
        reorder_map: List of (new_frame, original_image, original_time_s) rows.

    Returns:
        Path to the created file.

    Raises:
        ValueError: If the map is empty or contains non-finite times.
    """
    if not reorder_map:
        raise ValueError("reorder_map is empty; nothing to write.")

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    header = "new_frame,original_image,original_time_s"
    lines = [header]
    for new_frame, original_image, original_time_s in reorder_map:
        if not np.isfinite(original_time_s):
            raise ValueError(
                f"reorder_map contains a non-finite time for frame {new_frame}."
            )
        lines.append(
            "{},{},{}".format(
                int(new_frame), int(original_image), "%.6f" % original_time_s
            )
        )

    with open(str(filepath), "w", newline="", encoding="ascii") as f:
        f.write("\r\n".join(lines) + "\r\n")

    return filepath


FRAME_TABLE_HEADER = (
    "new_frame,original_image,original_time_s,profile_start_s,"
    "profile_end_s,extra_flow_s,extra_flow_until_s"
)


def export_frame_table_csv(filepath, frames):
    # type: (Union[str, Path], list) -> Path
    """Export the CT frame table: the reorder map plus where each frame sits.

    The first three columns are the reorder map's, unchanged. The others say
    where each frame starts and ends in the composed profile, and how long
    the recorded flow kept after it is (0 when none) and where in the
    recording that flow ends (empty when none). Registration can then align
    each image to its exact stretch of the breathing profile.

    Args:
        filepath: Output file path.
        frames: List of FrameRow (labchart2cfd.processing.profile_assembly).

    Returns:
        Path to the created file.

    Raises:
        ValueError: If the table is empty or contains a non-finite time.
    """
    if not frames:
        raise ValueError("The frame table is empty; nothing to write.")

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    lines = [FRAME_TABLE_HEADER]
    for f in frames:
        times = [f.original_time_s, f.profile_start_s, f.profile_end_s, f.extra_flow_s]
        if f.extra_flow_until_s is not None:
            times.append(f.extra_flow_until_s)
        if not all(np.isfinite(t) for t in times):
            raise ValueError(f"The frame table has a non-finite time for frame {f.new_frame}.")
        until = "" if f.extra_flow_until_s is None else "%.6f" % f.extra_flow_until_s
        lines.append("{},{},{},{},{},{},{}".format(
            int(f.new_frame), int(f.original_image),
            "%.6f" % f.original_time_s, "%.6f" % f.profile_start_s,
            "%.6f" % f.profile_end_s, "%.6f" % f.extra_flow_s, until,
        ))

    with open(str(filepath), "w", newline="", encoding="ascii") as f:
        f.write("\r\n".join(lines) + "\r\n")

    return filepath
