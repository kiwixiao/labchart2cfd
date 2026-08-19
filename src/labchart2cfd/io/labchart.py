"""Load LabChart-exported .mat files.

This module ports the MATLAB LabChart.m function to Python, handling the
cell array structure of LabChart exports.

LabChart exports contain:
- data: 1D array of all sample values
- datastart/dataend: [numchannels, numblocks] indices into data array
- samplerate: [numchannels, numblocks] sample rates per channel/block
- scaleunits/scaleoffset: scaling factors for 16-bit data
- titles: channel names
- unittext/unittextmap: unit strings per channel
"""

from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import scipy.io


class LabChartData:
    """Container for LabChart .mat file data.

    Attributes:
        data_cell: 2D list [channels][blocks] of data arrays
        time_cell: 2D list [channels][blocks] of time arrays
        num_channels: Number of channels in the recording
        num_blocks: Number of blocks (segments) in the recording
        titles: Channel titles/names
        filepath: Source file path
    """

    def __init__(
        self,
        data_cell,  # type: List[List[Optional[np.ndarray]]]
        time_cell,  # type: List[List[Optional[np.ndarray]]]
        num_channels,  # type: int
        num_blocks,  # type: int
        titles=None,  # type: Optional[List[str]]
        filepath=None,  # type: Optional[Path]
    ):
        self.data_cell = data_cell
        self.time_cell = time_cell
        self.num_channels = num_channels
        self.num_blocks = num_blocks
        self.titles = titles if titles is not None else []
        self.filepath = filepath

    def get_data(self, row, column):
        # type: (int, int) -> np.ndarray
        """Get data for a specific channel/block (1-indexed like MATLAB).

        Args:
            row: Channel index (1-indexed)
            column: Block index (1-indexed)

        Returns:
            Data array for the specified channel/block

        Raises:
            ValueError: If the specified block is empty or indices are invalid
        """
        # Convert from 1-indexed (MATLAB convention) to 0-indexed
        ch = row - 1
        bl = column - 1

        if ch < 0 or ch >= self.num_channels:
            raise ValueError("Row {} out of range [1, {}]".format(row, self.num_channels))
        if bl < 0 or bl >= self.num_blocks:
            raise ValueError("Column {} out of range [1, {}]".format(column, self.num_blocks))

        data = self.data_cell[ch][bl]
        if data is None:
            raise ValueError("Block [{}, {}] is empty".format(row, column))
        return data

    def get_time(self, row, column):
        # type: (int, int) -> np.ndarray
        """Get time array for a specific channel/block (1-indexed like MATLAB).

        Args:
            row: Channel index (1-indexed)
            column: Block index (1-indexed)

        Returns:
            Time array for the specified channel/block

        Raises:
            ValueError: If the specified block is empty or indices are invalid
        """
        ch = row - 1
        bl = column - 1

        if ch < 0 or ch >= self.num_channels:
            raise ValueError("Row {} out of range [1, {}]".format(row, self.num_channels))
        if bl < 0 or bl >= self.num_blocks:
            raise ValueError("Column {} out of range [1, {}]".format(column, self.num_blocks))

        time = self.time_cell[ch][bl]
        if time is None:
            raise ValueError("Block [{}, {}] is empty".format(row, column))
        return time

    def is_block_empty(self, row, column):
        # type: (int, int) -> bool
        """Check if a specific block is empty (1-indexed).

        Args:
            row: Channel index (1-indexed)
            column: Block index (1-indexed)

        Returns:
            True if block is empty, False otherwise
        """
        ch = row - 1
        bl = column - 1

        if ch < 0 or ch >= self.num_channels:
            return True
        if bl < 0 or bl >= self.num_blocks:
            return True

        return self.data_cell[ch][bl] is None


def _extract_titles(mat, num_channels):
    # type: (dict, int) -> List[str]
    """Parse channel titles from any of the layouts LabChart/MATLAB produce.

    LabChart's exporter writes a char matrix (rows of fixed-width chars).
    Re-saving from MATLAB turns it into a cell array, which loadmat delivers
    as an object array; the old chr() fallback crashed on that.
    """
    if "titles" not in mat:
        return []

    def _one_title(item) -> str:
        """Exactly one title per cell, however deeply loadmat nested it.

        Returning a variable number of strings per cell shifted every later
        channel up a slot whenever a cell was blank, so `validate` printed the
        wrong channel numbers and the user selected the wrong --row.
        """
        arr = np.asarray(item).ravel()
        if arr.size == 0:
            return ""                       # blank cell keeps its position
        if arr.size == 1:
            value = arr[0]
            if isinstance(value, np.ndarray):
                return _one_title(value)    # cell inside a cell
            return str(value).strip()
        return "".join(str(x) for x in arr).strip()

    titles_raw = mat["titles"]
    # Cell array: object dtype, possibly nested by loadmat.
    if getattr(titles_raw, "dtype", None) is not None and titles_raw.dtype == object:
        items = [_one_title(item) for item in titles_raw.ravel()]
    else:
        items = []
        for title_item in titles_raw:
            if isinstance(title_item, str):
                items.append(title_item.strip())
            else:
                # Char matrix row: join character codes.
                items.append("".join(chr(c) for c in title_item if c != 0).strip())

    # Pad so index i is always channel i+1, never a shifted neighbour.
    items = items[:num_channels]
    items.extend([""] * (num_channels - len(items)))
    return items


def load_labchart_mat(filepath):
    # type: (Union[str, Path]) -> LabChartData
    """Load a LabChart-exported .mat file.

    This function mirrors the behavior of LabChart.m, parsing the cell array
    structure and applying scaling factors to convert raw data to physical units.

    Args:
        filepath: Path to the .mat file

    Returns:
        LabChartData object containing parsed data and time arrays

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file doesn't contain expected LabChart data structure
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError("File not found: {}".format(filepath))

    # Load the .mat file
    mat = scipy.io.loadmat(str(filepath), squeeze_me=False)

    # Validate required variables exist
    if "data" not in mat:
        raise ValueError(
            "No 'data' variable found. Select a .mat file created with "
            "Export Matlab 3.0 or later (LabChart for Windows 7.2 or later)"
        )

    # Required companion fields, with errors that name what is missing
    # (a bare KeyError told the user nothing).
    for field in ("datastart", "dataend", "samplerate"):
        if field not in mat:
            raise ValueError(
                f"No '{field}' variable found in {filepath.name}; "
                "not a complete LabChart export."
            )

    # Extract core arrays (ravel avoids flatten's unconditional copy)
    data = mat["data"].ravel()
    datastart = mat["datastart"]
    dataend = mat["dataend"]
    samplerate = mat["samplerate"]

    # Optional scaling (16-bit data). The pair is all-or-nothing: either both
    # fields are present or neither is. Accepting one silently applied no
    # scaling at all while the mirror case was a hard error.
    has_units = "scaleunits" in mat
    has_offset = "scaleoffset" in mat
    if has_units != has_offset:
        missing = "scaleoffset" if has_units else "scaleunits"
        raise ValueError(
            f"'{filepath.name}' has "
            f"{'scaleunits' if has_units else 'scaleoffset'} but no {missing}; "
            "not a complete LabChart export."
        )
    has_scaling = has_units
    if has_scaling:
        scaleunits = mat["scaleunits"]
        scaleoffset = mat["scaleoffset"]

    # Get dimensions
    num_channels, num_blocks = datastart.shape

    titles = _extract_titles(mat, num_channels)

    # Initialize cell arrays
    data_cell = [[None for _ in range(num_blocks)] for _ in range(num_channels)]
    time_cell = [[None for _ in range(num_blocks)] for _ in range(num_channels)]

    # Extract data for each channel and block. The index tables come from the
    # file itself, so they are validated: Python slicing would silently clamp
    # a truncated file, and a non-positive start other than the -1 sentinel
    # would wrap into a negative index and return ANOTHER channel's samples.
    for ch in range(num_channels):
        for bl in range(num_blocks):
            # datastart == -1 indicates empty block
            start_idx = int(datastart[ch, bl])
            if start_idx == -1:
                continue
            if start_idx < 1:
                raise ValueError(
                    f"Corrupt datastart {start_idx} for channel {ch + 1}, "
                    f"block {bl + 1} in {filepath.name}: only -1 (empty) or "
                    "a 1-based index is valid."
                )

            end_idx = int(dataend[ch, bl])
            if end_idx < start_idx:
                raise ValueError(
                    f"Corrupt indices for channel {ch + 1}, block {bl + 1} in "
                    f"{filepath.name}: dataend {end_idx} is before "
                    f"datastart {start_idx}."
                )
            if end_idx > data.size:
                raise ValueError(
                    f"Channel {ch + 1}, block {bl + 1} in {filepath.name} "
                    f"claims samples up to {end_idx} but the file holds only "
                    f"{data.size}; the file is truncated (beyond data). "
                    "Re-export or re-copy it."
                )

            # Extract raw data (MATLAB 1-indexed inclusive -> Python slice)
            pdata = data[start_idx - 1 : end_idx].astype(np.float64)

            # Apply scaling if available (16-bit data)
            if has_scaling:
                pdata = (pdata + scaleoffset[ch, bl]) * scaleunits[ch, bl]

            # Generate time array
            sr = float(samplerate[ch, bl])
            if not np.isfinite(sr) or sr <= 0:
                raise ValueError(
                    f"Invalid sample rate {sr} for channel {ch + 1}, "
                    f"block {bl + 1} in {filepath.name}."
                )
            ptime = np.arange(len(pdata)) / sr

            data_cell[ch][bl] = pdata
            time_cell[ch][bl] = ptime

    return LabChartData(
        data_cell=data_cell,
        time_cell=time_cell,
        num_channels=num_channels,
        num_blocks=num_blocks,
        titles=titles,
        filepath=filepath,
    )


def describe_mat_structure(filepath):
    # type: (Union[str, Path]) -> dict
    """Describe the structure of a LabChart .mat file.

    Useful for debugging and understanding the data layout.

    Args:
        filepath: Path to the .mat file

    Returns:
        Dictionary with structure information including dimensions,
        sample rates, and block occupancy
    """
    filepath = Path(filepath)
    mat = scipy.io.loadmat(str(filepath), squeeze_me=False)

    data = mat.get("data")
    datastart = mat.get("datastart")
    dataend = mat.get("dataend")
    samplerate = mat.get("samplerate")

    if datastart is None:
        return {"error": "Not a valid LabChart .mat file (no datastart)"}
    if data is None:
        # Without the data variable there is nothing to process; reporting a
        # healthy structure here made `validate` pass on files `process`
        # would then reject.
        return {"error": "Not a valid LabChart .mat file (no data variable)"}
    if dataend is None or samplerate is None:
        return {"error": "Not a valid LabChart .mat file (missing dataend/samplerate)"}
    # Same all-or-nothing scaling rule the loader enforces, so validate cannot
    # bless a file process will refuse.
    if ("scaleunits" in mat) != ("scaleoffset" in mat):
        return {
            "error": "Not a valid LabChart .mat file "
                     "(scaleunits and scaleoffset must both be present)"
        }

    num_channels, num_blocks = datastart.shape
    # Index tables may disagree in shape on a malformed file; report that
    # rather than raising IndexError and breaking the documented contract.
    if (getattr(dataend, "shape", None) != datastart.shape
            or getattr(samplerate, "shape", None) != datastart.shape):
        return {
            "error": "Not a valid LabChart .mat file "
                     "(datastart, dataend and samplerate have different shapes)"
        }
    total_available = data.size

    # Build block info; declared indices are checked against the actual data
    # so a truncated or corrupt file is visibly flagged rather than described
    # as healthy.
    blocks = []
    for ch in range(num_channels):
        for bl in range(num_blocks):
            start = int(datastart[ch, bl])
            if start == -1:
                blocks.append({
                    "channel": ch + 1,
                    "block": bl + 1,
                    "empty": True,
                })
                continue

            end = int(dataend[ch, bl])
            sr = float(samplerate[ch, bl])
            if (start < 1 or end < start or end > total_available
                    or not np.isfinite(sr) or sr <= 0):
                blocks.append({
                    "channel": ch + 1,
                    "block": bl + 1,
                    "empty": False,
                    "corrupt": True,
                    "declared_start": start,
                    "declared_end": end,
                })
                continue

            n_samples = end - start + 1
            blocks.append({
                "channel": ch + 1,
                "block": bl + 1,
                "empty": False,
                "samples": n_samples,
                "sample_rate": sr,
                "duration_s": n_samples / sr,
            })

    titles = _extract_titles(mat, num_channels)

    return {
        "filepath": str(filepath),
        "num_channels": num_channels,
        "num_blocks": num_blocks,
        "titles": titles,
        "total_samples": int(total_available),
        "blocks": blocks,
    }
