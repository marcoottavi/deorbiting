"""Test the simplified main.py functionality"""

from main import run_simulation

# Test parameters for ISS-like orbit
test_params = {
    'semi_major_axis': 6.78e6,  # ~400 km altitude
    'eccentricity': 0.001,
    'inclination': 51.6,
    'raan': 0,
    'arg_perigee': 0,
    'true_anomaly': 0,
    'time_span': 86400,  # 1 day
    'time_step': 300,    # 5 minutes
    'include_j2': True,
    'include_drag': False,
    'include_thrust': False,
    'spacecraft_mass': 500.0,
    'ballistic_coefficient': 100.0,
    'solar_flux': 150.0,
    'thrust_magnitude': 1.0,
    'specific_impulse': 2000.0,
    'thrust_direction': "prograde",
    'enable_altitude_event': False,
    'altitude_threshold': 100.0
}

print("Testing simplified main.py functionality...")
success = run_simulation(test_params)
print(f"Test {'PASSED' if success else 'FAILED'}")
