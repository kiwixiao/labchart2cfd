"""Tests for the workflow layer.

Currently pins the CT workflow's no-rearrange ("full manual window") path,
which the synthetic-profile GUI mode relies on: process the whole window
unrearranged so the assembly engine can splice the processed signal.
"""
import numpy as np

from labchart2cfd.io.labchart import LabChartData
from labchart2cfd.workflows import CTWorkflow


def _single_channel_data(duration_s=2.0, raw_rate_hz=1000.0):
    """One flow channel, one block: a clean sinusoid over [0, duration]."""
    n = int(duration_s * raw_rate_hz) + 1
    t = np.linspace(0.0, duration_s, n)
    flow = np.sin(2 * np.pi * 0.5 * t)  # 0.5 Hz breathing-like signal
    return LabChartData(
        data_cell=[[flow]],
        time_cell=[[t]],
        num_channels=1,
        num_blocks=1,
    )


def test_ct_window_no_rearrange_image_map():
    data = _single_channel_data(duration_s=2.0)
    wf = CTWorkflow(target_sample_rate=100.0, density=1.2)

    result = wf.process(
        data, row=1, column=1, start_time=0.0, end_time=2.0,
        inhale_start_time=None, exhale_end_time=None,
        temporal_resolution=0.2, include_pressure=False,
    )

    # Resampled to 100 Hz, time normalized to start at 0
    assert result.sample_rate == 100.0
    assert result.time[0] == 0.0
    assert len(result.time) == len(result.mass_flow)
    assert len(result.mass_flow) > 0

    # No rearrangement requested -> cut index 0; fencepost image count = 2.0/0.2 + 1
    assert result.metadata["total_images"] == 11
    assert result.metadata["cut_image_index"] == 0
    assert result.pressure is None
