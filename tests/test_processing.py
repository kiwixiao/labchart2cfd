"""Tests for processing functions.

These tests verify Python processing functions match MATLAB behavior.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from labchart2cfd.processing.drift_correction import (
    calculate_drift_error,
    apply_drift_correction,
    correct_flow_drift,
)
from labchart2cfd.processing.resampling import resample_to_rate
from labchart2cfd.processing.smoothing import smooth_moving_average
from labchart2cfd.processing.unit_conversion import (
    liters_per_second_to_kg_per_second,
    cmh2o_to_pascal,
    AIR_DENSITY_KG_M3,
    CMH2O_TO_PA,
)


class TestDriftCorrection:
    """Tests for drift correction functions.

    The error rate is SIGNED (net volume / elapsed time) and the correction
    subtracts it, so the integrated volume returns to zero for either drift
    polarity. The old form used abs() plus a per-workflow sign, which doubled
    the drift whenever the polarity did not match the assumed one.
    """

    def test_calculate_drift_error_zero_drift(self):
        """Perfect sinusoidal flow should have near-zero drift."""
        time = np.linspace(0, 2 * np.pi, 1000)
        flow = np.sin(time)
        error = calculate_drift_error(time, flow)
        assert_allclose(error, 0.0, atol=1e-3)

    def test_drift_error_is_signed(self):
        """The error rate carries the drift's direction."""
        # Whole periods, so the sine itself integrates to ~0 and only the
        # offset determines the sign.
        time = np.linspace(0, 4 * np.pi, 4001)
        assert calculate_drift_error(time, np.sin(time) + 0.1) > 0.05
        assert calculate_drift_error(time, np.sin(time) - 0.1) < -0.05

    def test_apply_drift_correction_subtracts_signed_error(self):
        flow = np.array([1.0, 2.0, 3.0])
        assert_allclose(apply_drift_correction(flow, 0.5), [0.5, 1.5, 2.5])
        assert_allclose(apply_drift_correction(flow, -0.5), [1.5, 2.5, 3.5])

    @pytest.mark.parametrize("offset", [+0.005, -0.005])
    def test_correction_zeroes_net_volume_for_both_polarities(self, offset):
        """The invariant that was broken: after correction, the integrated
        volume must return to ~0 regardless of drift direction. The old code
        turned a +0.5 L spurious volume into +1.0 L."""
        from scipy.integrate import cumulative_trapezoid

        time = np.arange(0, 100, 0.001)
        flow = 0.5 * np.sin(2 * np.pi * 0.25 * time) + offset

        corrected, error = correct_flow_drift(time, flow)

        v_after = cumulative_trapezoid(corrected, time, initial=0)[-1]
        assert_allclose(v_after, 0.0, atol=1e-6)
        assert_allclose(error, offset, atol=1e-6)

    def test_single_sample_window_raises_not_nan(self):
        with pytest.raises(ValueError):
            calculate_drift_error(np.array([1.0]), np.array([0.5]))


class TestResampling:
    """Tests for resampling functions."""

    def test_resample_fencepost_count_and_true_rate(self):
        """N intervals need N+1 samples, and the achieved rate is the nominal one.

        The old count int(period*rate) with endpoint-inclusive linspace gave
        spacing period/(N-1): a "100 Hz" request actually produced 99.75 Hz,
        which drifted the CT image map ~9.5 ms across a 4 s window.
        """
        time = np.linspace(0, 3.0, 3001)  # 1000 Hz, exactly 3 s
        data = np.sin(2 * np.pi * time)

        time_out, data_out = resample_to_rate(time, data, 100)

        assert len(time_out) == 301          # 300 intervals + 1
        assert len(data_out) == 301
        spacing = np.diff(time_out)
        assert_allclose(spacing, 0.01, rtol=1e-9)  # truly 100 Hz

    def test_resample_count_is_immune_to_float_fuzz(self):
        """Regression: a window whose float span lands a hair under the exact
        value must not lose a sample. int(409.99999999999994) == 409 used to
        make ~16% of grid anchors fail the synthetic-profile assembly with
        'needs samples up to N but the window only has N-1'."""
        t = np.arange(0, 10, 0.001)          # 1000 Hz
        mask = (t >= 0.137) & (t <= 4.137)   # 4.000 s window off the exact grid
        data = np.sin(t)

        time_out, _ = resample_to_rate(t[mask], data[mask], 100)

        # 400 intervals + 1, never 399/400 depending on float luck.
        assert len(time_out) == 401

    def test_resample_preserves_time_range(self):
        """Resampled time should cover same range."""
        time = np.linspace(5.0, 8.0, 3000)
        data = np.sin(time)

        time_out, data_out = resample_to_rate(time, data, 100)

        assert_allclose(time_out[0], time[0], rtol=0.01)
        assert_allclose(time_out[-1], time[-1], rtol=0.01)

    def test_resample_rejects_degenerate_window(self):
        """A window too short to hold one interval raises instead of returning
        an empty array that looks like success."""
        time = np.linspace(0, 0.004, 5)      # 4 ms at 1000 Hz
        with pytest.raises(ValueError):
            resample_to_rate(time, np.ones(5), 100)


class TestSmoothing:
    """Tests for smoothing functions."""

    def test_smooth_constant(self):
        """Smoothing constant data should return same data."""
        data = np.ones(100) * 5.0
        smoothed = smooth_moving_average(data, window_size=5)
        assert_allclose(smoothed, data)

    def test_smooth_output_length(self):
        """Smoothed data should have same length as input."""
        data = np.random.randn(100)
        smoothed = smooth_moving_average(data, window_size=5)
        assert len(smoothed) == len(data)

    def test_smooth_reduces_noise(self):
        """Smoothing should reduce standard deviation of noisy signal."""
        np.random.seed(42)
        signal = np.ones(1000)
        noisy = signal + np.random.randn(1000) * 0.1
        smoothed = smooth_moving_average(noisy, window_size=5)

        # Smoothed should have lower std dev
        assert np.std(smoothed) < np.std(noisy)


class TestUnitConversion:
    """Tests for unit conversion functions."""

    def test_liters_to_kg_per_second(self):
        """Test L/s to kg/s conversion."""
        flow_lps = np.array([1.0])  # 1 L/s
        density = 1.2  # kg/m³

        # 1 L/s = 0.001 m³/s
        # 0.001 m³/s * 1.2 kg/m³ = 0.0012 kg/s
        # With invert_sign=True: -0.0012 kg/s
        mass_flow = liters_per_second_to_kg_per_second(flow_lps, density, invert_sign=True)
        expected = np.array([-0.0012])
        assert_allclose(mass_flow, expected)

    def test_liters_to_kg_per_second_no_invert(self):
        """Test L/s to kg/s without sign inversion."""
        flow_lps = np.array([1.0])
        mass_flow = liters_per_second_to_kg_per_second(
            flow_lps, AIR_DENSITY_KG_M3, invert_sign=False
        )
        expected = np.array([0.0012])
        assert_allclose(mass_flow, expected)

    def test_cmh2o_to_pascal(self):
        """Test cmH2O to Pascal conversion."""
        pressure_cmh2o = np.array([1.0])
        pressure_pa = cmh2o_to_pascal(pressure_cmh2o)
        expected = np.array([CMH2O_TO_PA])  # 98.0665 Pa
        assert_allclose(pressure_pa, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
