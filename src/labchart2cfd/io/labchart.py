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

    # Extract core arrays
    data = mat["data"].flatten()
    datastart = mat["datastart"]
    dataend = mat["dataend"]
    samplerate = mat["samplerate"]

    # Optional scaling (16-bit data)
    has_scaling = "scaleunits" in mat
    if has_scaling:
        scaleunits = mat["scaleunits"]
        scaleoffset = mat["scaleoffset"]

    # Get dimensions
    num_channels, num_blocks = datastart.shape

    # Parse titles if available
    titles = []
    if "titles" in mat:
        titles_raw = mat["titles"]
        for i in range(num_channels):
            if i < len(titles_raw):
                title_item = titles_raw[i]
                # Handle both string and character array formats
                if isinstance(title_item, str):
                    title = title_item.strip()
                else:
                    # Character array: join characters
                    title = "".join(chr(c) for c in title_item if c != 0).strip()
                titles.append(title)

    # Initialize cell arrays
    data_cell = [[None for _ in range(num_blocks)] for _ in range(num_channels)]
    time_cell = [[None for _ in range(num_blocks)] for _ in range(num_channels)]

    # Extract data for each channel and block
    for ch in range(num_channels):
        for bl in range(num_blocks):
            # datastart == -1 indicates empty block
            start_idx = int(datastart[ch, bl])
            if start_idx == -1:
                continue

            end_idx = int(dataend[ch, bl])

            # Extract raw data (MATLAB uses 1-indexed, Python uses 0-indexed)
            # datastart/dataend are 1-indexed in MATLAB, but loadmat converts
            # them as-is, so we need to subtract 1 for start and keep end as-is
            # for Python's exclusive end slicing
            pdata = data[start_idx - 1 : end_idx].astype(np.float64)

            # Apply scaling if available (16-bit data)
            if has_scaling:
                pdata = (pdata + scaleoffset[ch, bl]) * scaleunits[ch, bl]

            # Generate time array
            sr = float(samplerate[ch, bl])
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
        return {"error": "Not a valid LabChart .mat file"}

    num_channels, num_blocks = datastart.shape

    # Build block info
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
            else:
                end = int(dataend[ch, bl])
                sr = float(samplerate[ch, bl])
                n_samples = end - start + 1
                duration = n_samples / sr
                blocks.append({
                    "channel": ch + 1,
                    "block": bl + 1,
                    "empty": False,
                    "samples": n_samples,
                    "sample_rate": sr,
                    "duration_s": duration,
                })

    # Parse titles
    titles = []
    if "titles" in mat:
        titles_raw = mat["titles"]
        for i in range(num_channels):
            if i < len(titles_raw):
                title_item = titles_raw[i]
                if isinstance(title_item, str):
                    title = title_item.strip()
                else:
                    title = "".join(chr(c) for c in title_item if c != 0).strip()
                titles.append(title)

    return {
        "filepath": str(filepath),
        "num_channels": num_channels,
        "num_blocks": num_blocks,
        "titles": titles,
        "total_samples": len(data.flatten()) if data is not None else 0,
        "blocks": blocks,
    }
