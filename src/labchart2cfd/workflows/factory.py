"""Workflow construction factory.

Single source of truth for building a workflow from a type name, shared by the
CLI (`cli/main.py`) and GUI (`gui/app.py`). Centralizes the per-type gas-density
defaults and class selection that both surfaces previously duplicated inline.

Omitted keyword arguments fall through to each workflow class's own defaults, so
callers that only pass some parameters get identical objects to a direct
constructor call.
"""
from typing import Optional

from labchart2cfd.processing.unit_conversion import AIR_DENSITY_KG_M3, XENON_DENSITY_KG_M3
from labchart2cfd.workflows.base import BaseWorkflow
from labchart2cfd.workflows.cpap import CPAPWorkflow
from labchart2cfd.workflows.ct import CTWorkflow
from labchart2cfd.workflows.phase_contrast import PhaseContrastWorkflow
from labchart2cfd.workflows.standard_osamri import StandardOSAMRIWorkflow


# type name (normalized) -> (class, default gas density, accepts pressure_row)
_WORKFLOWS = {
    "standard": (StandardOSAMRIWorkflow, AIR_DENSITY_KG_M3, True),
    "cpap": (CPAPWorkflow, AIR_DENSITY_KG_M3, True),
    "ct": (CTWorkflow, AIR_DENSITY_KG_M3, True),
    "phase-contrast": (PhaseContrastWorkflow, XENON_DENSITY_KG_M3, False),
}


def _normalize(workflow_type) -> str:
    """Normalize a workflow type (enum or str, any case/underscore) to a key."""
    name = getattr(workflow_type, "value", workflow_type)
    return str(name).lower().replace("_", "-")


def build_workflow(
    workflow_type,
    *,
    density: Optional[float] = None,
    target_sample_rate: Optional[float] = None,
    pressure_row: Optional[int] = None,
) -> BaseWorkflow:
    """Construct a workflow instance for the given type.

    Args:
        workflow_type: Workflow type as a string ("standard", "cpap",
            "phase-contrast", "ct"; case/underscore-insensitive) or a
            WorkflowType enum.
        density: Gas density in kg/m³. When None, the per-type default is used
            (air for standard/cpap/ct, xenon for phase-contrast).
        target_sample_rate: Target output sample rate in Hz. When None, the
            workflow class default is used (100 Hz, or 1000 Hz for phase
            contrast).
        pressure_row: Pressure channel row (1-indexed). Ignored for phase
            contrast, which has no pressure output. When None, the class default
            is used.

    Returns:
        A configured BaseWorkflow subclass instance.

    Raises:
        ValueError: If workflow_type is not recognized.
    """
    key = _normalize(workflow_type)
    if key not in _WORKFLOWS:
        raise ValueError(
            f"Unknown workflow type: {workflow_type!r}. "
            f"Use one of {sorted(_WORKFLOWS)}."
        )

    workflow_cls, default_density, accepts_pressure_row = _WORKFLOWS[key]

    kwargs = {"density": density if density is not None else default_density}
    if target_sample_rate is not None:
        kwargs["target_sample_rate"] = target_sample_rate
    if accepts_pressure_row and pressure_row is not None:
        kwargs["pressure_row"] = pressure_row

    return workflow_cls(**kwargs)
