"""
Physical and astronomical constants for satellite atmospheric modeling.

This module contains all the physical constants and tabulated data
needed for the Harris-Priester atmospheric density model.

Author: PhD Design Research
Date: June 2025
"""

import numpy as np

# ============================================================================
# EARTH CONSTANTS
# ============================================================================

EARTH_CONSTANTS = {
    'radius': 6.378137e6,           # Earth equatorial radius [m] (WGS84)
    'mu': 3.986004418e14,           # Earth gravitational parameter [m³/s²]
    'rotation_rate': 7.2921159e-5,  # Earth rotation rate [rad/s]
    'mass': 5.972168e24,            # Earth mass [kg]
    'j2': 1.08262668e-3,            # Second zonal harmonic coefficient
}

# ============================================================================
# PHYSICAL CONSTANTS
# ============================================================================

PHYSICAL_CONSTANTS = {
    'k_boltzmann': 1.380649e-23,    # Boltzmann constant [J/K]
    'gas_constant': 8.314462618,    # Universal gas constant [J/(mol⋅K)]
    'avogadro': 6.02214076e23,      # Avogadro constant [1/mol]
    'g0': 9.80665,                  # Standard gravity [m/s²]
}

# ============================================================================
# CONVERSION CONSTANTS
# ============================================================================

CONVERSIONS = {
    'deg_to_rad': np.pi / 180.0,
    'rad_to_deg': 180.0 / np.pi,
    'days_to_sec': 86400.0,
    'hours_to_sec': 3600.0,
    'km_to_m': 1000.0,
}

# ============================================================================
# HARRIS-PRIESTER ATMOSPHERIC MODEL DATA
# ============================================================================

HARRIS_PRIESTER = {
    # Altitude levels [km] - converted to meters
    'altitudes': np.array([
        100, 120, 130, 140, 150, 160, 170, 180, 190, 200,
        210, 220, 230, 240, 250, 260, 270, 280, 290, 300,
        320, 340, 360, 380, 400, 420, 440, 460, 480, 500,
        520, 540, 560, 580, 600, 620, 640, 660, 680, 700,
        720, 740, 760, 780, 800, 840, 880, 920, 960, 1000
    ]) * 1000.0,  # Convert km to meters
    
    # Minimum density (night/winter conditions) [kg/m³]
    'rho_min': np.array([
        5.297e-07, 2.396e-08, 8.770e-09, 3.899e-09, 2.122e-09, 1.263e-09,
        8.008e-10, 5.283e-10, 3.617e-10, 2.557e-10, 1.839e-10, 1.341e-10,
        9.949e-11, 7.488e-11, 5.709e-11, 4.403e-11, 3.430e-11, 2.697e-11,
        2.139e-11, 1.708e-11, 1.099e-11, 7.214e-12, 4.824e-12, 3.274e-12,
        2.249e-12, 1.558e-12, 1.091e-12, 7.701e-13, 5.474e-13, 3.916e-13,
        2.819e-13, 2.042e-13, 1.488e-13, 1.092e-13, 8.070e-14, 5.937e-14,
        4.398e-14, 3.265e-14, 2.438e-14, 1.828e-14, 1.370e-14, 1.028e-14,
        7.751e-15, 5.854e-15, 4.429e-15, 2.541e-15, 1.457e-15, 8.373e-16,
        4.809e-16, 2.768e-16
    ]),
    
    # Maximum density (day/summer conditions) [kg/m³]
    'rho_max': np.array([
        5.297e-07, 2.396e-08, 8.770e-09, 3.899e-09, 2.122e-09, 1.263e-09,
        8.008e-10, 5.283e-10, 3.617e-10, 2.557e-10, 1.839e-10, 1.341e-10,
        9.949e-11, 7.488e-11, 5.709e-11, 4.403e-11, 3.430e-11, 2.697e-11,
        2.139e-11, 1.708e-11, 1.099e-11, 7.214e-12, 4.824e-12, 3.274e-12,
        2.249e-12, 1.558e-12, 1.091e-12, 7.701e-13, 5.474e-13, 3.916e-13,
        2.819e-13, 2.042e-13, 1.488e-13, 1.092e-13, 8.070e-14, 5.937e-14,
        4.398e-14, 3.265e-14, 2.438e-14, 1.828e-14, 1.370e-14, 1.028e-14,
        7.751e-15, 5.854e-15, 4.429e-15, 2.541e-15, 1.457e-15, 8.373e-16,
        4.809e-16, 2.768e-16
    ]) * 4.0,  # Day densities are typically 2-4 times higher than night
    
    # Model parameters
    'n_diurnal': 2,        # Cosine exponent for diurnal variation
    'm_annual': 6,         # Cosine exponent for annual variation  
    'lag_angle': 30.0,     # Phase lag angle [degrees]
    'f107_ref': 150.0,     # Reference F10.7 solar flux
    'temp_exo_ref': 900.0, # Reference exospheric temperature [K]
}

# Export all constants
__all__ = [
    'EARTH_CONSTANTS', 'PHYSICAL_CONSTANTS', 'CONVERSIONS', 'HARRIS_PRIESTER'
]
