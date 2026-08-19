"""Tests for the workflow layer.

Currently pins the CT workflow's no-rearrange ("full manual window") path,
which the synthetic-profile GUI mode relies on: process the whole window
unrearranged so the assembly engine can splice the processed signal.
"""
import numpy as np
import pytest
from scipy.integrate import cumulative_trapezoid, trapezoid

from labchart2cfd.io.labchart import LabChartData
from labchart2cfd.workflows import CTWorkflow, PhaseContrastWorkflow


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

    # No rearrangement requested -> cut index 0. Image count is the number of
    # slices acquired, 2.0/0.2 = 10 (bounded by 11 grid lines).
    assert result.metadata["total_images"] == 10
    assert result.metadata["cut_image_index"] == 0
    assert result.pressure is None


def test_ct_cut_snaps_to_image_start_so_flow_and_images_align():
    """The splice must happen at the instant the reported image index names.

    Cutting at the raw click while reporting a rounded image index would leave
    the exported waveform and the exported image order describing moments up
    to half an image apart.
    """
    data = _single_channel_data(duration_s=2.0)
    wf = CTWorkflow(target_sample_rate=100.0, density=1.2)

    # exhale left of inhale -> the wrapping case, which triggers rearrangement.
    result = wf.process(
        data, row=1, column=1, start_time=0.0, end_time=2.0,
        inhale_start_time=1.27, exhale_end_time=0.5,
        temporal_resolution=0.2, include_pressure=False,
    )

    # 1.27 s is nearest the line at 1.2 s, which starts image index 6.
    assert result.metadata["cut_image_index"] == 6
    assert result.metadata["inhale_snapped_time"] == pytest.approx(1.2)
    assert result.metadata["inhale_start_time"] == 1.27  # request preserved
    # The rearranged order starts at the cut image and wraps.
    assert result.metadata["rearranged_image_indices"][0] == 6
    assert len(result.metadata["rearranged_image_indices"]) == 10


def test_ct_rearranges_with_only_inhale_given():
    """The CLI's documented invocation passes --inhale-start without
    --exhale-end. The old condition additionally required exhale < inhale, so
    that call silently skipped rearrangement and the profile started
    mid-cycle while metadata reported the inhale time as honoured."""
    data = _single_channel_data(duration_s=2.0)
    wf = CTWorkflow(target_sample_rate=100.0, density=1.2)

    result = wf.process(
        data, row=1, column=1, start_time=0.0, end_time=2.0,
        inhale_start_time=1.27, exhale_end_time=None,
        temporal_resolution=0.2, include_pressure=False,
    )

    assert result.metadata["cut_image_index"] == 6
    assert result.metadata["inhale_snapped_time"] == pytest.approx(1.2)
    assert result.metadata["rearranged_image_indices"][0] == 6


def _two_rate_data(flow_hz=1000.0, pressure_hz=200.0, duration_s=10.0):
    """Four channels so the default pressure_row=4 resolves: flow on ch2 at
    flow_hz, pressure on ch4 at a DIFFERENT rate."""
    tf = np.arange(0.0, duration_s, 1.0 / flow_hz)
    tp = np.arange(0.0, duration_s, 1.0 / pressure_hz)
    flow = np.sin(2 * np.pi * 0.5 * tf)
    pressure = 5.0 + np.cos(2 * np.pi * 0.5 * tp)  # cmH2O
    empty = None
    return LabChartData(
        data_cell=[[flow * 0.0], [flow], [empty], [pressure]],
        time_cell=[[tf], [tf], [empty], [tp]],
        num_channels=4,
        num_blocks=1,
    )


def test_pressure_at_a_different_sample_rate_is_windowed_on_its_own_time():
    """All workflows masked pressure with the FLOW channel's time array, which
    crashed with a boolean-index length mismatch whenever the two channels
    were recorded at different rates (routine in LabChart)."""
    from labchart2cfd.workflows import StandardOSAMRIWorkflow

    data = _two_rate_data()
    wf = StandardOSAMRIWorkflow(target_sample_rate=100.0, density=1.2)

    result = wf.process(
        data, row=2, column=1, start_time=2.0, end_time=6.0,
        include_pressure=True,
    )

    assert result.pressure is not None
    assert len(result.pressure) == len(result.mass_flow)
    # cmH2O converted to Pa: 5 cmH2O ~ 490 Pa, so values sit near there.
    assert 300.0 < float(np.mean(result.pressure)) < 700.0


def test_window_outside_data_raises_a_named_error():
    """An out-of-range window used to die deep in resampling with a bare
    IndexError; now the window extraction itself explains the problem."""
    data = _single_channel_data(duration_s=2.0)
    wf = CTWorkflow(target_sample_rate=100.0, density=1.2)

    with pytest.raises(ValueError, match="window"):
        wf.process(
            data, row=1, column=1, start_time=500.0, end_time=520.0,
            inhale_start_time=None, exhale_end_time=None,
            temporal_resolution=0.2, include_pressure=False,
        )


def test_phase_contrast_calibration_uses_peak_cumulative():
    """Voltage calibration must be the PEAK of the cumulative integral.

    The MATLAB reference (documented in the workflow) is::

        vol = cumtrapz(timeU, flowV)
        v_v = max(vol)

    For a balanced breath the cumulative curve rises to a peak during inhale
    then falls back toward zero during exhale, so the *total* integral
    (scipy.integrate.trapezoid) collapses to ~0 while the peak stays large.
    Using the total integral therefore corrupts the calibration factor.
    """
    t = np.linspace(0.0, 4.0, 4001)
    data = np.sin(2 * np.pi * 0.25 * t)  # one full breath, net area ~= 0

    expected_peak = float(np.max(cumulative_trapezoid(data, t, initial=0)))
    total_integral = float(trapezoid(data, t))

    # Sanity: peak and total are sharply different for this signal.
    assert expected_peak > 1.0
    assert abs(total_integral) < 0.1

    wf = PhaseContrastWorkflow()
    cal = wf._get_voltage_calibration(t, data)

    assert cal == pytest.approx(expected_peak, rel=1e-6)
