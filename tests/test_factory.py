"""Tests for the workflow factory (build_workflow).

The factory is the single source of truth for workflow construction shared by
the CLI and GUI. These tests pin that it reproduces, attribute-for-attribute,
exactly what each call site used to build inline — so wiring both surfaces to
it is behavior-preserving.
"""
import pytest

from labchart2cfd.workflows import (
    StandardOSAMRIWorkflow,
    CPAPWorkflow,
    PhaseContrastWorkflow,
    CTWorkflow,
)
from labchart2cfd.workflows.factory import build_workflow


def test_standard_cli_pattern():
    # CLI: StandardOSAMRIWorkflow(target_sample_rate=sample_rate, density=density)
    wf = build_workflow("standard", density=1.2, target_sample_rate=100.0)
    assert isinstance(wf, StandardOSAMRIWorkflow)
    assert wf.density == 1.2
    assert wf.target_sample_rate == 100.0
    assert wf.pressure_row == 4  # class default preserved when not passed


def test_standard_gui_pattern():
    # GUI: StandardOSAMRIWorkflow(density=1.2, pressure_row=effective_pressure_row)
    wf = build_workflow("standard", density=1.2, pressure_row=5)
    assert isinstance(wf, StandardOSAMRIWorkflow)
    assert wf.density == 1.2
    assert wf.pressure_row == 5
    assert wf.target_sample_rate == 100.0  # class default preserved


def test_cpap_gui_pattern():
    wf = build_workflow("cpap", density=1.2, pressure_row=4)
    assert isinstance(wf, CPAPWorkflow)
    assert wf.density == 1.2
    assert wf.pressure_row == 4
    assert wf.target_sample_rate == 100.0


def test_ct_gui_pattern():
    # GUI: CTWorkflow(density=1.2, pressure_row=effective_pressure_row)
    wf = build_workflow("ct", density=1.2, pressure_row=4)
    assert isinstance(wf, CTWorkflow)
    assert wf.density == 1.2
    assert wf.pressure_row == 4
    assert wf.target_sample_rate == 100.0


def test_ct_cli_pattern():
    # CLI: CTWorkflow(target_sample_rate=sample_rate, density=density)
    wf = build_workflow("ct", density=1.2, target_sample_rate=250.0)
    assert isinstance(wf, CTWorkflow)
    assert wf.target_sample_rate == 250.0


def test_phase_contrast_defaults_rate_1000():
    # GUI: PhaseContrastWorkflow(density=5.761) -> class default rate 1000
    wf = build_workflow("phase-contrast", density=5.761)
    assert isinstance(wf, PhaseContrastWorkflow)
    assert wf.density == 5.761
    assert wf.target_sample_rate == 1000.0


def test_phase_contrast_explicit_rate():
    # CLI: PhaseContrastWorkflow(target_sample_rate=target_rate, density=density)
    wf = build_workflow("phase-contrast", density=5.761, target_sample_rate=500.0)
    assert wf.target_sample_rate == 500.0


def test_density_defaults_by_type_when_omitted():
    # CLI resolves density to 1.2 (air) or 5.761 (xenon) when --density not given
    assert build_workflow("standard").density == 1.2
    assert build_workflow("cpap").density == 1.2
    assert build_workflow("ct").density == 1.2
    assert build_workflow("phase-contrast").density == 5.761


def test_accepts_gui_and_cli_type_spellings():
    # GUI passes "CT"/"phase-contrast"/"cpap"; CLI passes WorkflowType enum values
    assert isinstance(build_workflow("CT"), CTWorkflow)
    assert isinstance(build_workflow("phase-contrast"), PhaseContrastWorkflow)
    assert isinstance(build_workflow("standard"), StandardOSAMRIWorkflow)


def test_rejects_unknown_type():
    with pytest.raises(ValueError):
        build_workflow("nonsense")
