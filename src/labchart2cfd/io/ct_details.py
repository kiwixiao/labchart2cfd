"""The CT details file, ``{subject}_trigger_info.txt``.

Plain ``key: value`` lines, one per fact, so a person can read it and a
script can parse it. Every CT export writes it (design 2026-11-22, 4.3): what
was cut, where the drift was measured, the order of the frames, and the name
of the frame table it goes with.
"""

from pathlib import Path
from typing import List, Tuple, Union


def write_ct_details(filepath: Union[str, Path], entries: List[Tuple[str, str]]) -> Path:
    """Write the details file.

    Args:
        filepath: Output file path.
        entries: (key, value) pairs in the order they are written. A key may
            repeat (one ``extra_flow`` line per extra flow).

    Returns:
        Path to the created file.

    Raises:
        ValueError: If a key is empty or holds ": " or a line break, or a value
            holds a line break; either would make the file unreadable.
    """
    lines = []
    for key, value in entries:
        value = str(value)
        if not key or ": " in key or "\n" in key or "\n" in value:
            raise ValueError(f"Cannot write the details line {key!r}: {value!r}.")
        lines.append(f"{key}: {value}")
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text("\n".join(lines) + "\n")
    return filepath
