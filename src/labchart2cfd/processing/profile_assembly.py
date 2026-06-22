"""Synthetic breathing-profile assembly for the CT workflow.

In 4D CT the scanner captures images at uncontrolled breathing phases, so the
images are not a clean inhale->exhale cycle in acquisition order. The clinician
defines a window, then labels an ordered list of *parts* (contiguous ranges of
numbered image markers). This engine stitches the parts of the already-processed
flow into one synthetic profile and emits the matching image reorder map.

Process-then-assemble: the caller processes the whole window first (drift,
unit conversion, resample, smooth) and passes the processed flow here; this
engine only reorders/concatenates. It is intentionally free of any GUI or
plotting imports so it can be unit-tested headless.

Fencepost convention (image i sits at processed time t_i = (i-1)*dt):
- Flow: a part (a, b) contributes samples in the half-open span
  [(a-1)*spi, (b-1)*spi); the LAST part uses the closed span so the assembled
  profile spans end-to-end while internal joins never double-count the seam.
- Reorder map: a part (a, b) lists images a..b inclusive (b-a+1 entries).
Flow and map intentionally live at different granularities.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass
class ProfilePart:
    """One contiguous range of image markers, in assembly order.

    Args:
        start_image: 1-based image number, inclusive.
        end_image: 1-based image number, inclusive (must be > start_image).
        label: Optional descriptive tag ("inhale", "exhale tail"); not used
            by the math.
    """

    start_image: int
    end_image: int
    label: str = ""


@dataclass
class AssemblyResult:
    """Result of assembling parts into a synthetic profile.

    Attributes:
        time: Synthetic time array, starting at 0 at the sample rate.
        flow: Assembled mass-flow (deliverable A).
        pressure: Assembled pressure, or None if no pressure was supplied.
        reorder_map: One row per image, (new_frame, original_image,
            original_time_s) — deliverable B.
    """

    time: NDArray[np.float64]
    flow: NDArray[np.float64]
    pressure: Optional[NDArray[np.float64]]
    reorder_map: List[Tuple[int, int, float]] = field(default_factory=list)


def assemble_parts(
    flow: NDArray[np.float64],
    parts: List[ProfilePart],
    dt: float,
    sample_rate: float,
    total_images: int,
    pressure: Optional[NDArray[np.float64]] = None,
) -> AssemblyResult:
    """Assemble an ordered list of parts into a synthetic profile + reorder map.

    Args:
        flow: Processed mass-flow over the full window (uniform sample rate).
        parts: Ordered parts; list order is the assembly order.
        dt: Temporal resolution (seconds per image marker, e.g. 0.2).
        sample_rate: Sample rate of ``flow`` in Hz (e.g. 100.0).
        total_images: Number of image markers in the window (1..N).
        pressure: Optional processed pressure aligned with ``flow``.

    Returns:
        AssemblyResult with assembled flow/pressure, synthetic time, and the
        per-image reorder map.

    Raises:
        ValueError: if ``parts`` is empty, ``dt``/``sample_rate`` are
            non-positive, a part violates ``1 <= start < end <= total_images``,
            or the assembled flow is empty.
    """
    if not parts:
        raise ValueError("No parts to assemble; label at least one part.")
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")

    spi = int(round(dt * sample_rate))
    if spi < 1:
        raise ValueError(
            f"samples-per-image is {spi}; dt*sample_rate must be >= 1."
        )

    for p in parts:
        if not (1 <= p.start_image < p.end_image <= total_images):
            raise ValueError(
                "Each part must satisfy 1 <= start_image < end_image <= "
                f"{total_images}; got {p.start_image}..{p.end_image}."
            )

    flow_chunks: List[NDArray[np.float64]] = []
    pressure_chunks: List[NDArray[np.float64]] = []
    reorder_map: List[Tuple[int, int, float]] = []
    new_frame = 0
    last_idx = len(parts) - 1

    for idx, p in enumerate(parts):
        lo = (p.start_image - 1) * spi
        hi = (p.end_image - 1) * spi
        stop = hi + 1 if idx == last_idx else hi  # last part is closed
        flow_chunks.append(flow[lo:stop])
        if pressure is not None:
            pressure_chunks.append(pressure[lo:stop])
        for img in range(p.start_image, p.end_image + 1):
            new_frame += 1
            reorder_map.append((new_frame, img, (img - 1) * dt))

    assembled_flow = np.concatenate(flow_chunks)
    if assembled_flow.size == 0:
        raise ValueError("Assembled flow is empty; check the parts and window.")
    assembled_pressure = (
        np.concatenate(pressure_chunks) if pressure is not None else None
    )
    time = np.arange(assembled_flow.size) / sample_rate

    return AssemblyResult(
        time=time,
        flow=assembled_flow,
        pressure=assembled_pressure,
        reorder_map=reorder_map,
    )
