"""Pydantic configuration models for flow profile processing.

These models provide validation and documentation for processing parameters.
Compatible with pydantic v1.x.
"""

from enum import Enum
from pathlib import Path
from typing import Optional, Union

from pydantic import BaseModel, Field, validator


class WorkflowType(str, Enum):
    """Available workflow types."""
    STANDARD = "standard"
    CPAP = "cpap"
    PHASE_CONTRAST = "phase_contrast"


class ProcessingConfig(BaseModel):
    """Base processing configuration.

    Attributes:
        input_file: Path to LabChart .mat file
        subject: Subject identifier for output filenames
        row: Flow data channel row (1-indexed)
        column: Block column (1-indexed)
        start_time: Start time of extraction window (seconds)
        end_time: End time of extraction window (seconds)
        output_dir: Output directory for CSV files
    """
    input_file: Path
    subject: str
    row: int = Field(default=2, ge=1, description="Flow channel row (1-indexed)")
    column: int = Field(default=3, ge=1, description="Block column (1-indexed)")
    start_time: float = Field(default=0.0, ge=0.0, description="Start time (seconds)")
    end_time: float = Field(default=0.0, ge=0.0, description="End time (seconds)")
    output_dir: Path = Field(default=Path("."), description="Output directory")

    @validator("input_file")
    def validate_input_exists(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"Input file does not exist: {v}")
        return v

    @validator("end_time")
    def validate_end_after_start(cls, v: float, values) -> float:
        start = values.get("start_time", 0.0)
        if v != 0.0 and v <= start:
            raise ValueError("End time must be greater than start time")
        return v


class WorkflowConfig(BaseModel):
    """Common workflow configuration.

    Attributes:
        target_sample_rate: Output sample rate in Hz
        smoothing_window: Moving average window size
        include_pressure: Whether to process pressure data
    """
    target_sample_rate: float = Field(default=100.0, gt=0, description="Target Hz")
    smoothing_window: int = Field(default=5, ge=3, le=21, description="Window size")
    include_pressure: bool = Field(default=True, description="Process pressure")


class StandardConfig(WorkflowConfig):
    """Standard OSAMRI workflow configuration.

    Drift correction on full signal, flow + error.

    Attributes:
        density: Air density in kg/m³
        pressure_row: Channel row for pressure data
    """
    workflow_type: str = "standard"
    density: float = Field(default=1.2, gt=0, description="Air density kg/m³")
    pressure_row: int = Field(default=4, ge=1, description="Pressure channel row")


class CPAPConfig(WorkflowConfig):
    """CPAP workflow configuration.

    Drift correction on windowed data, flow - error.

    Attributes:
        density: Air density in kg/m³
        pressure_row: Channel row for pressure data
    """
    workflow_type: str = "cpap"
    density: float = Field(default=1.2, gt=0, description="Air density kg/m³")
    pressure_row: int = Field(default=4, ge=1, description="Pressure channel row")


class PhaseContrastConfig(WorkflowConfig):
    """Phase contrast (xenon) workflow configuration.

    Voltage calibration, no drift correction.

    Attributes:
        density: Xenon density in kg/m³
        bag_id: Xenon bag identifier (Bag1, Bag2, Bag4)
        shift_time: Time point for baseline calculation
        calibration_factor: Manual voltage-per-liter factor
    """
    workflow_type: str = "phase_contrast"
    density: float = Field(default=5.761, gt=0, description="Xenon density kg/m³")
    target_sample_rate: float = Field(default=1000.0, gt=0, description="Target Hz")
    include_pressure: bool = Field(default=False, description="No pressure for xenon")
    bag_id: Optional[str] = Field(default=None, regex=r"^Bag[124]$")
    shift_time: Optional[float] = Field(default=None, ge=0, description="Baseline time")
    calibration_factor: Optional[float] = Field(default=None, gt=0)


class SubjectConfig(BaseModel):
    """Complete configuration for a subject processing run.

    Combines processing parameters with workflow-specific settings.
    """
    processing: ProcessingConfig
    workflow: Union[StandardConfig, CPAPConfig, PhaseContrastConfig]

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "SubjectConfig":
        """Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            SubjectConfig instance
        """
        import yaml
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        return cls.parse_obj(data)

    def to_yaml(self, yaml_path: Path) -> None:
        """Save configuration to YAML file.

        Args:
            yaml_path: Path to save YAML file
        """
        import yaml
        with open(yaml_path, "w") as f:
            yaml.dump(self.dict(), f, default_flow_style=False)
