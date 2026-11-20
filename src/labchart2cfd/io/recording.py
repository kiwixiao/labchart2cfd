"""Open a LabChart recording of either kind as ``LabChartData``.

A recording reaches the application as LabChart's own MATLAB export (.mat) or
as the native file (.adicht). Both come out of ``load_recording`` in the same
form, so nothing downstream needs to know which one was opened.
"""

from pathlib import Path
from typing import Callable, List, Optional, Union

import numpy as np

from labchart2cfd.io.adicht import AdichtFile, plan_export
from labchart2cfd.io.labchart import Comment, LabChartData, load_labchart_mat

# File types that hold a recording, lower case, in the order they are offered.
RECORDING_SUFFIXES = (".mat", ".adicht")


ProgressCallback = Callable[[int, int, str], None]


def load_adicht(filepath: Union[str, Path],
                notes: Optional[List[str]] = None,
                progress: Optional[ProgressCallback] = None) -> LabChartData:
    """Decode a native LabChart recording.

    The values are rounded to single precision before use, because that is
    how LabChart's MATLAB export stores them. Opening the native file
    therefore gives exactly the numbers that opening the export gives, which
    is what the comparison against real exports checks.

    A channel whose LabChart calculation cannot be reproduced is left empty,
    like an empty block. A file with anything the reader has not been
    verified on raises ``AdichtFormatError``.

    Args:
        filepath: Path to the .adicht file. It is opened read only.
        notes: If given, lines about channels that were recomputed or left
            out are appended to it, for showing to the user.
        progress: If given, called before each channel of each record is
            decoded as ``progress(number, total, description)``. A long
            recording takes many seconds, so a caller can show where it is.

    Returns:
        The recording, with 1-indexed channels (rows) and records (columns).
    """
    filepath = Path(filepath)
    with AdichtFile(filepath) as adi:
        _cells, reader_notes = plan_export(adi)
        n_channels, n_records = adi.n_channels, adi.n_records
        titles = list(adi.titles)
        data_cell = [[None for _ in range(n_records)] for _ in range(n_channels)]
        time_cell = [[None for _ in range(n_records)] for _ in range(n_channels)]
        # A channel that cannot be reproduced stays empty (reader_notes says
        # which and why), and so does a record that holds no samples.
        to_decode = [
            (ch, rec)
            for ch in range(n_channels) if adi.is_exportable(ch)
            for rec in range(n_records) if adi.n_samples[ch][rec] > 0
        ]
        for number, (ch, rec) in enumerate(to_decode, start=1):
            if progress is not None:
                progress(number, len(to_decode),
                         f"channel {ch + 1} {titles[ch]}".rstrip() + f", record {rec + 1}")
            values = adi.data(ch, rec).astype(np.float32).astype(np.float64)
            data_cell[ch][rec] = values
            # The same time base the .mat loader builds: sample index over
            # the sample rate, starting at 0 in every record.
            time_cell[ch][rec] = (
                np.arange(adi.n_samples[ch][rec]) / float(adi.sample_rate[ch][rec])
            )

        # The native reader counts records and channels from 0 and marks a
        # comment on all channels with -1. Its time is the tick over the
        # record's tick rate, the same clock as the samples.
        comments = [
            Comment(
                block=com["record"] + 1,
                channel=None if com["channel"] < 0 else com["channel"] + 1,
                time=float(com["time_s"]),
                text=com["text"],
            )
            for com in adi.comments
        ]

    if notes is not None:
        notes.extend(reader_notes)
    return LabChartData(
        data_cell=data_cell,
        time_cell=time_cell,
        num_channels=n_channels,
        num_blocks=n_records,
        titles=titles,
        filepath=filepath,
        comments=comments,
    )


def load_recording(filepath: Union[str, Path],
                   notes: Optional[List[str]] = None,
                   progress: Optional[ProgressCallback] = None) -> LabChartData:
    """Open a recording, choosing the reader by the file's extension.

    Args:
        filepath: A LabChart .mat export or a native .adicht file.
        notes: If given, remarks from the reader are appended to it.
        progress: If given, called as a native recording is decoded (see
            ``load_adicht``). A .mat export loads in one step and reports none.

    Raises:
        ValueError: If the extension is neither .mat nor .adicht, or the file
            is not what its extension says.
    """
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()
    if suffix == ".mat":
        return load_labchart_mat(filepath, notes=notes)
    if suffix == ".adicht":
        return load_adicht(filepath, notes=notes, progress=progress)
    raise ValueError(
        f"Cannot open '{filepath.name}': expected a LabChart .mat export or "
        "a native .adicht recording."
    )
