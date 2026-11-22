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

Any Temporal Res (design 2026-11-22, 4.1a). dt is used as typed. Line k is
the sample nearest k x dt, rounded half up on its own, so a dt that is not a
whole number of samples (275 ms at 100 Hz is 27.5) gives slices of 28 and 27
samples that never drift from the grid. The frame table names the samples
actually used. At 100 and 200 ms every line falls on a sample and nothing
changes.

Extra flow (4.2). A part may keep the recorded flow after its last image up
to ``extra_until_s`` before the next part starts. No image belongs to that
flow; the frame table says how long it is and where it ends.
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
        extra_until_s: Optional time in the recording. The recorded flow
            after the last image of the part is kept up to this time, before
            the next part starts. None keeps no extra flow.
    """

    start_image: int
    end_image: int
    label: str = ""
    extra_until_s: Optional[float] = None


@dataclass
class FrameRow:
    """One frame of the composed profile, for the frame table.

    Attributes:
        new_frame: 1-based position in the composed profile.
        original_image: 1-based image number in the scan.
        original_time_s: Where the image starts in the recording (the sample
            actually used).
        profile_start_s: Where the frame starts in the composed profile.
        profile_end_s: Where the frame ends in the composed profile.
        extra_flow_s: Length of the recorded flow kept after this frame
            before the next one, 0 when there is none.
        extra_flow_until_s: Time in the recording where that extra flow
            ends (the sample actually used), None when there is none.
    """

    new_frame: int
    original_image: int
    original_time_s: float
    profile_start_s: float
    profile_end_s: float
    extra_flow_s: float
    extra_flow_until_s: Optional[float]


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
        frames: One FrameRow per frame, the full frame table.
        extra_spans: (start, end) in profile time of each stretch of extra
            flow, for drawing.
    """

    time: NDArray[np.float64]
    flow: NDArray[np.float64]
    pressure: Optional[NDArray[np.float64]]
    reorder_map: List[Tuple[int, int, float]] = field(default_factory=list)
    frames: List[FrameRow] = field(default_factory=list)
    extra_spans: List[Tuple[float, float]] = field(default_factory=list)


def _grid_line(k: int, off: int, exact_spi: float) -> int:
    """Sample index of grid line k (line 0 starts image 1), rounded half up."""
    return off + int(np.floor(k * exact_spi + 0.5 + 1e-9))


def _sample_at(t: float, window_start_s: float, sample_rate: float) -> int:
    """Sample index of a recording time, rounded half up."""
    return int(np.floor((t - window_start_s) * sample_rate + 0.5 + 1e-9))


def extra_flow_problems(
    parts: List[ProfilePart],
    dt: float,
    sample_rate: float,
    grid_offset_s: float = 0.0,
    window_start_s: float = 0.0,
) -> List[Tuple[int, str]]:
    """The extra flow rules, checked on whole samples as the cut is made.

    One function for the engine and the screen: ``assemble_parts`` raises the
    first problem, and the line above Export lists them all, so the two
    cannot disagree by a sample (design 2026-11-22, 4.2). An extra flow must
    end after its image ends, and is refused when the next part starts at
    the next image, which already follows it in the recording.

    Args:
        parts: Ordered parts, as for ``assemble_parts``.
        dt: Seconds per image as typed.
        sample_rate: Sample rate of the processed flow in Hz.
        grid_offset_s: Seconds from the start of the processed flow to grid
            line 1.
        window_start_s: Recording time of the first processed sample.

    Returns:
        (index of the part, rule broken) for each problem, in part order.
        The rule reads after "Extra flow after image N".
    """
    exact_spi = dt * sample_rate
    off = int(round(grid_offset_s * sample_rate))
    problems: List[Tuple[int, str]] = []
    for i, p in enumerate(parts):
        if p.extra_until_s is None:
            continue
        end_line = _grid_line(p.end_image, off, exact_spi)
        if _sample_at(p.extra_until_s, window_start_s, sample_rate) <= end_line:
            image_end_s = window_start_s + end_line / sample_rate
            problems.append((i, (
                f"must end after that image ends at {image_end_s:.3f} s; it "
                f"ends at {p.extra_until_s:.3f} s"
            )))
        elif i + 1 < len(parts) and parts[i + 1].start_image == p.end_image + 1:
            problems.append((i, (
                f"is not possible: the next part starts at the next image, "
                f"{p.end_image + 1}, which already follows it in the recording"
            )))
    return problems


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
        dt: Seconds per image as typed (e.g. 0.2 or 0.275). Each grid line
            is rounded to the nearest sample on its own.
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
            non-positive, an image is shorter than one sample,
            ``grid_offset_s`` is negative, a part violates
            ``1 <= start <= end <= total_images``, a part would read past the
            end of ``flow``, extra flow does not end after its image, reaches
            past ``flow``, or is followed by the next image, or the assembled
            flow is empty.
    """
    if not parts:
        raise ValueError("No parts to assemble; label at least one part.")
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")
    if grid_offset_s < 0:
        raise ValueError("grid_offset_s must not be negative.")

    exact_spi = dt * sample_rate
    if exact_spi < 1:
        raise ValueError(
            f"samples-per-image is {exact_spi:.3f}; dt*sample_rate must be >= 1."
        )

    exact_off = grid_offset_s * sample_rate
    off = int(round(exact_off))
    if abs(exact_off - off) > 1e-9:
        raise ValueError(
            f"grid_offset_s*sample_rate is {exact_off:.9f}, not a whole "
            "number of samples; quantise the offset to a multiple of "
            "1/sample_rate."
        )

    def line(k: int) -> int:
        return _grid_line(k, off, exact_spi)

    def sample_at(t: float) -> int:
        return _sample_at(t, window_start_s, sample_rate)

    for i, p in enumerate(parts):
        if not (1 <= p.start_image <= p.end_image <= total_images):
            raise ValueError(
                "Each part must satisfy 1 <= start_image <= end_image <= "
                f"{total_images}; got {p.start_image}..{p.end_image}."
            )
        if line(p.end_image) > len(flow):
            raise ValueError(
                f"Part {p.start_image}..{p.end_image} needs samples up to "
                f"{line(p.end_image)} but the processed window only has "
                f"{len(flow)}. Shorten the part, or widen the window."
            )
        if p.extra_until_s is None:
            continue
        if sample_at(p.extra_until_s) > len(flow):
            flow_end_s = window_start_s + len(flow) / sample_rate
            raise ValueError(
                f"Extra flow after image {p.end_image} runs to "
                f"{p.extra_until_s:.3f} s, past the end of the processed "
                f"recording at {flow_end_s:.3f} s."
            )
    problems = extra_flow_problems(
        parts, dt, sample_rate, grid_offset_s=grid_offset_s, window_start_s=window_start_s
    )
    if problems:
        i, rule = problems[0]
        raise ValueError(f"Extra flow after image {parts[i].end_image} {rule}.")

    flow_chunks: List[NDArray[np.float64]] = []
    pressure_chunks: List[NDArray[np.float64]] = []
    frames: List[FrameRow] = []
    extra_spans: List[Tuple[float, float]] = []
    cursor = 0  # samples written to the composed profile so far

    for p in parts:
        # Image i is the slice [line i-1, line i), so images a..b span from
        # line a-1 up to line b. Slices are disjoint: no seam to double-count.
        lo = line(p.start_image - 1)
        stop = line(p.end_image)
        if p.extra_until_s is not None:
            stop_with_extra = sample_at(p.extra_until_s)
        else:
            stop_with_extra = stop
        flow_chunks.append(flow[lo:stop_with_extra])
        if pressure is not None:
            pressure_chunks.append(pressure[lo:stop_with_extra])
        for img in range(p.start_image, p.end_image + 1):
            first, last = line(img - 1), line(img)
            frames.append(FrameRow(
                new_frame=len(frames) + 1,
                original_image=img,
                original_time_s=window_start_s + first / sample_rate,
                profile_start_s=cursor / sample_rate,
                profile_end_s=(cursor + last - first) / sample_rate,
                extra_flow_s=0.0,
                extra_flow_until_s=None,
            ))
            cursor += last - first
        if stop_with_extra > stop:
            extra = stop_with_extra - stop
            frames[-1].extra_flow_s = extra / sample_rate
            frames[-1].extra_flow_until_s = window_start_s + stop_with_extra / sample_rate
            extra_spans.append((cursor / sample_rate, (cursor + extra) / sample_rate))
            cursor += extra

    assembled_flow = np.concatenate(flow_chunks)
    if assembled_flow.size == 0:
        raise ValueError("Assembled flow is empty; check the parts and window.")
    assembled_pressure = (
        np.concatenate(pressure_chunks) if pressure is not None else None
    )
    time = np.arange(assembled_flow.size) / sample_rate
    reorder_map = [(f.new_frame, f.original_image, f.original_time_s) for f in frames]

    return AssemblyResult(
        time=time,
        flow=assembled_flow,
        pressure=assembled_pressure,
        reorder_map=reorder_map,
        frames=frames,
        extra_spans=extra_spans,
    )
