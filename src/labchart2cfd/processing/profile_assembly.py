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

Image model. The CT trigger channel marks only the start and end of the whole
dynamic sequence; there are no per-image triggers, so the image grid is a
uniform ruler laid across the window. Grid *lines* bound the images, and an
*image* is the slice between two consecutive lines::

    line j at   grid_offset + (j-1)*dt        j = 1 .. N+1
    image i  =  [line_i, line_i+1)            i = 1 .. N

So N images are bounded by N+1 lines, and one image owns exactly dt of signal.
A part (a, b) therefore means images a..b:

- Flow: samples [off + (a-1)*spi, off + b*spi), which is (b-a+1)*spi samples.
- Reorder map: images a..b inclusive, which is (b-a+1) rows.

Flow and map agree by construction (one map row per image, spi samples each),
and because the slices are disjoint there is no seam to double-count.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass
class ProfilePart:
    """One contiguous run of images, in assembly order.

    Args:
        start_image: 1-based image number, inclusive.
        end_image: 1-based image number, inclusive. May equal start_image,
            which selects that single image (one dt of signal).
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
            original_time_s) — deliverable B. ``original_time_s`` is where the
            image sits in the ORIGINAL recording, while ``time`` above is the
            synthetic profile's own axis starting at 0; the two clocks are
            deliberately different.
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
    grid_offset_s: float = 0.0,
    window_start_s: float = 0.0,
) -> AssemblyResult:
    """Assemble an ordered list of parts into a synthetic profile + reorder map.

    Args:
        flow: Processed mass-flow over the full window (uniform sample rate).
        parts: Ordered parts; list order is the assembly order.
        dt: Seconds per image (e.g. 0.2). One image owns exactly this much
            signal.
        sample_rate: Sample rate of ``flow`` in Hz (e.g. 100.0).
        total_images: Number of images in the window (1..N).
        pressure: Optional processed pressure aligned with ``flow``.
        grid_offset_s: Seconds from the start of ``flow`` to grid line 1.
        window_start_s: Absolute time of the first sample of ``flow`` in the
            original recording. Added to the reorder map's time column so it
            can be found on the recording's own axis; leave at 0 to keep that
            column relative to the window.

    Returns:
        AssemblyResult with assembled flow/pressure, synthetic time, and the
        per-image reorder map.

    Raises:
        ValueError: if ``parts`` is empty, ``dt``/``sample_rate`` are
            non-positive, ``grid_offset_s`` is negative, a part violates
            ``1 <= start <= end <= total_images``, a part would read past the
            end of ``flow``, or the assembled flow is empty.
    """
    if not parts:
        raise ValueError("No parts to assemble; label at least one part.")
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")
    if grid_offset_s < 0:
        raise ValueError("grid_offset_s must not be negative.")

    spi = int(round(dt * sample_rate))
    if spi < 1:
        raise ValueError(
            f"samples-per-image is {spi}; dt*sample_rate must be >= 1."
        )

    off = int(round(grid_offset_s * sample_rate))

    for p in parts:
        if not (1 <= p.start_image <= p.end_image <= total_images):
            raise ValueError(
                "Each part must satisfy 1 <= start_image <= end_image <= "
                f"{total_images}; got {p.start_image}..{p.end_image}."
            )
        if off + p.end_image * spi > len(flow):
            raise ValueError(
                f"Part {p.start_image}..{p.end_image} needs samples up to "
                f"{off + p.end_image * spi} but the processed window only has "
                f"{len(flow)}. Shorten the part, or widen the window."
            )

    flow_chunks: List[NDArray[np.float64]] = []
    pressure_chunks: List[NDArray[np.float64]] = []
    reorder_map: List[Tuple[int, int, float]] = []
    new_frame = 0

    for p in parts:
        # Image i is the slice [line_i, line_i+1), so images a..b span from
        # line a up to line b+1. Slices are disjoint: no seam to double-count.
        lo = off + (p.start_image - 1) * spi
        stop = off + p.end_image * spi
        flow_chunks.append(flow[lo:stop])
        if pressure is not None:
            pressure_chunks.append(pressure[lo:stop])
        for img in range(p.start_image, p.end_image + 1):
            new_frame += 1
            reorder_map.append(
                (new_frame, img, window_start_s + grid_offset_s + (img - 1) * dt)
            )

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
