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
    """Tests for drift correction functions."""

    def test_calculate_drift_error_zero_drift(self):
        """Perfect sinusoidal flow should have near-zero drift."""
        time = np.linspace(0, 2 * np.pi, 1000)
        # Perfect sine wave integrates to zero over full period
        flow = np.sin(time)
        error = calculate_drift_error(time, flow)
        assert_allclose(error, 0.0, atol=1e-3)

    def test_calculate_drift_error_with_offset(self):
        """Constant offset should be detected as drift."""
        time = np.linspace(0, 10, 1000)
        offset = 0.1
        flow = np.sin(time) + offset
        error = calculate_drift_error(time, flow)
        # Error should be approximately the offset
        assert error > 0.05

    def test_apply_drift_correction_positive(self):
        """Standard OSAMRI uses positive correction (flow + error)."""
        flow = np.array([1.0, 2.0, 3.0])
        error = 0.5
        corrected = apply_drift_correction(flow, error, sign=1)
        expected = np.array([1.5, 2.5, 3.5])
        assert_allclose(corrected, expected)

    def test_apply_drift_correction_negative(self):
        """CPAP uses negative correction (flow - error)."""
        flow = np.array([1.0, 2.0, 3.0])
        error = 0.5
        corrected = apply_drift_correction(flow, error, sign=-1)
        expected = np.array([0.5, 1.5, 2.5])
        assert_allclose(corrected, expected)


class TestResampling:
    """Tests for resampling functions."""

    def test_resample_to_rate_length(self):
        """Output length should match target sample rate."""
        time = np.linspace(0, 3.0, 3000)  # 1000 Hz, 3 seconds
        data = np.sin(2 * np.pi * time)
        target_rate = 100  # Hz

        time_out, data_out = resample_to_rate(time, data, target_rate)

        expected_samples = int(3.0 * target_rate)
        assert len(time_out) == expected_samples
        assert len(data_out) == expected_samples

    def test_resample_preserves_time_range(self):
        """Resampled time should cover same range."""
        time = np.linspace(5.0, 8.0, 3000)
        data = np.sin(time)

        time_out, data_out = resample_to_rate(time, data, 100)

        assert_allclose(time_out[0], time[0], rtol=0.01)
        assert_allclose(time_out[-1], time[-1], rtol=0.01)


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
