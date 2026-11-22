"""In equals out for a composed CT breath (design 2026-11-22, 4.7).

The CFD repeats the exported breath cycle after cycle, so any net volume left
in it accumulates. A composed breath has one complete half, recorded as it
was, and one composed half, put together from images and extra flow. The
complete half is kept sample by sample, and every sample of the composed half
is multiplied by one factor k so that the net volume is exactly zero.

Write the profile as q = p + n, with p the samples above zero and n the
samples below zero (each with the other set to 0). The volume is the
trapezoid integral, the volume a solver sees when it interpolates the table
linearly between samples. It is linear, so with the samples below zero as the
composed half net(k) = trapz(p) + k trapz(n), and k = -trapz(p) / trapz(n)
gives zero. Scaling keeps the sign of every sample, so each zero crossing
stays between the same two samples, and no time is changed.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import trapezoid

from labchart2cfd.processing.unit_conversion import AIR_DENSITY_KG_M3

_HALVES = ("positive", "negative")


@dataclass
class BalanceResult:
    """A breath whose composed half was scaled so in equals out.

    Attributes:
        flow: The balanced flow, same length and time base as the input.
        factor: The factor the composed half was multiplied by.
        kept_volume: Signed trapezoid volume of the complete half (flow
            units x s), unchanged by the balance.
        scaled_volume_before: Signed trapezoid volume of the composed half
            before scaling.
    """

    flow: NDArray[np.float64]
    factor: float
    kept_volume: float
    scaled_volume_before: float


def balance_halves(flow: NDArray[np.float64], sample_rate: float, keep: str) -> BalanceResult:
    """Scale the composed half of a breath so its net volume is zero.

    Args:
        flow: Uniformly sampled flow of one breath.
        sample_rate: Samples per second of ``flow``.
        keep: The complete half, kept as it is: "positive" (the samples
            above zero) or "negative" (the samples below zero).

    Returns:
        BalanceResult with the balanced flow and the factor used.

    Raises:
        ValueError: If ``keep`` is not "positive" or "negative", the sample
            rate is not positive, or the half to scale holds no volume.
    """
    if keep not in _HALVES:
        raise ValueError(f"keep must be 'positive' or 'negative', got {keep!r}.")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")
    flow = np.asarray(flow, dtype=float)
    above = np.where(flow > 0, flow, 0.0)
    below = np.where(flow < 0, flow, 0.0)
    kept, scaled = (above, below) if keep == "positive" else (below, above)
    dx = 1.0 / sample_rate
    kept_volume = float(trapezoid(kept, dx=dx))
    scaled_volume = float(trapezoid(scaled, dx=dx))
    if scaled_volume == 0.0:
        raise ValueError(
            "the half to scale holds no volume: every sample lies on one side "
            "of zero, so in and out cannot be matched."
        )
    factor = -kept_volume / scaled_volume
    return BalanceResult(
        flow=kept + factor * scaled,
        factor=factor,
        kept_volume=kept_volume,
        scaled_volume_before=scaled_volume,
    )


def balance_choices(flow: NDArray[np.float64], sample_rate: float) -> Dict[str, BalanceResult]:
    """Both balances the preview offers, keyed by the half kept, or none.

    When one half holds no volume, keeping the other would scale nothing and
    keeping the empty one would scale the other to zero. Neither is a breath
    with in equal to out, so no choice is offered.

    Args:
        flow: Uniformly sampled flow of one breath.
        sample_rate: Samples per second of ``flow``.

    Returns:
        {"positive": ..., "negative": ...}, or {} when a half holds no volume.
    """
    try:
        return {keep: balance_halves(flow, sample_rate, keep) for keep in _HALVES}
    except ValueError:
        return {}


def default_keep(
    flow: NDArray[np.float64], sample_rate: float, extra_spans: List[Tuple[float, float]]
) -> str:
    """The complete half to offer first: the one the extra flow is not in.

    The extra flow belongs to the composed half. Its side of zero is the sign
    of the mean flow over the extra flow samples. With no extra flow, the
    half above zero.

    Args:
        flow: The composed breath.
        sample_rate: Samples per second of ``flow``.
        extra_spans: (start, end) in profile time of each extra flow.

    Returns:
        "positive" or "negative".
    """
    flow = np.asarray(flow, dtype=float)
    samples = [flow[int(round(a * sample_rate)):int(round(b * sample_rate))]
               for a, b in extra_spans]
    samples = [chunk for chunk in samples if chunk.size]
    if not samples:
        return "positive"
    return "negative" if float(np.mean(np.concatenate(samples))) > 0 else "positive"


def half_volumes(time: NDArray[np.float64], flow: NDArray[np.float64]) -> Tuple[float, float]:
    """Signed trapezoid volumes of the samples above and below zero.

    Integrated over the given time column, so a file read back is measured
    on its own time axis.

    Returns:
        (above, below); below is zero or negative.
    """
    flow = np.asarray(flow, dtype=float)
    above = float(trapezoid(np.where(flow > 0, flow, 0.0), time))
    below = float(trapezoid(np.where(flow < 0, flow, 0.0), time))
    return above, below


def mass_volume_to_ml(volume_kg: float, density: float = AIR_DENSITY_KG_M3) -> float:
    """A volume given as mass (the integral of a mass flow, kg) in mL.

    Args:
        volume_kg: Mass of gas in kg.
        density: Gas density in kg/m^3 (1.2 for air, as the export uses).

    Returns:
        The volume in mL.
    """
    return volume_kg / density * 1e6
