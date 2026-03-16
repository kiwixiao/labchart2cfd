"""Unit conversion functions for flow profile data.

Physical constants and conversion factors used in CFD flow profile processing.
"""

import numpy as np


# Physical constants
AIR_DENSITY_KG_M3 = 1.2  # Standard air density at ~20°C
XENON_DENSITY_KG_M3 = 5.761  # Xenon gas density

# Conversion factors
CMH2O_TO_PA = 98.0665  # 1 cmH2O = 98.0665 Pa


def liters_per_second_to_kg_per_second(flow_lps, density_kg_m3=AIR_DENSITY_KG_M3, invert_sign=True):
    # type: (np.ndarray, float, bool) -> np.ndarray
    """Convert volumetric flow rate to mass flow rate.

    Args:
        flow_lps: Flow rate in L/s
        density_kg_m3: Gas density in kg/m³ (default: air at 1.2 kg/m³)
        invert_sign: If True, multiply by -1 (inhale is negative in measurements,
                     but positive outflow in CFD convention)

    Returns:
        Mass flow rate in kg/s

    Note:
        The MATLAB code applies: massFlow = flow * 1e-3 * rho * -1
        - 1e-3 converts L/s to m³/s (1 L = 0.001 m³)
        - rho (density) converts m³/s to kg/s
        - -1 inverts sign for CFD convention (bronchi outlet = positive)
    """
    # L/s -> m³/s -> kg/s
    mass_flow = flow_lps * 1e-3 * density_kg_m3

    if invert_sign:
        mass_flow = -mass_flow

    return mass_flow


def cmh2o_to_pascal(pressure_cmh2o):
    # type: (np.ndarray) -> np.ndarray
    """Convert pressure from cmH2O to Pascal.

    Args:
        pressure_cmh2o: Pressure in cmH2O

    Returns:
        Pressure in Pascal
    """
    return pressure_cmh2o * CMH2O_TO_PA


def voltage_to_flow_xenon(voltage, calibration_voltage_per_liter=1.0, density_kg_m3=XENON_DENSITY_KG_M3):
    # type: (np.ndarray, float, float) -> np.ndarray
    """Convert voltage signal to mass flow for xenon phase contrast.

    Used in phase contrast workflow where the flow sensor outputs voltage
    proportional to flow rate. A calibration factor is used to convert
    voltage to liters.

    Args:
        voltage: Raw voltage signal
        calibration_voltage_per_liter: Voltage output for 1 L of gas flow
                                       (determined from calibration bag)
        density_kg_m3: Gas density in kg/m³ (default: xenon at 5.761 kg/m³)

    Returns:
        Mass flow rate in kg/s
    """
    # voltage -> L/s -> m³/s -> kg/s
    flow_lps = voltage / calibration_voltage_per_liter
    mass_flow = flow_lps * 1e-3 * density_kg_m3
    return mass_flow
