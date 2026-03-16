"""Workflow implementations for different flow profile processing types."""

from labchart2cfd.workflows.base import BaseWorkflow, WorkflowResult
from labchart2cfd.workflows.standard_osamri import StandardOSAMRIWorkflow
from labchart2cfd.workflows.cpap import CPAPWorkflow
from labchart2cfd.workflows.phase_contrast import PhaseContrastWorkflow

__all__ = [
    "BaseWorkflow",
    "WorkflowResult",
    "StandardOSAMRIWorkflow",
    "CPAPWorkflow",
    "PhaseContrastWorkflow",
]
