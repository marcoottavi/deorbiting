"""
Test and demonstration script for the orbital propagator.

This script demonstrates the capabilities of the orbital propagator including:
- Basic orbital propagation
- J2 perturbations
- Atmospheric drag using Harris-Priester model
- Constant thrust with mass variation
- Comparison of different perturbation effects

Author: PhD Design Research
Date: June 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from orbital_elements import OrbitalElements
from orbital_propagator import OrbitalPropagator
from harris_priester import HarrisPriesterAtmosphere
from constants import EARTH_CONSTANTS, CONVERSIONS


def test_basic_propagation():
    """Test basic orbital propagation without perturbations."""
    
    print("="*60)
    print("TEST 1: BASIC ORBITAL PROPAGATION (KEPLERIAN)")
    print("="*60)
    
    # Create initial orbit (ISS-like)
    initial_orbit = OrbitalElements(
        semi_major_axis=6.78e6,  # ~400 km altitude
        eccentricity=0.0001,
        inclination=51.6,        # ISS inclination
        raan=0.0,
        arg_perigee=0.0,
        true_anomaly=0.0
    )
    
    print(f"Initial orbit:\n{initial_orbit}")
    
    # Create propagator
    propagator = OrbitalPropagator()
    
    # Propagate for 1 day with no perturbations
    results = propagator.propagate(
        initial_elements=initial_orbit,
        time_span=86400,  # 1 day
        include_j2=False,
        include_drag=False,
        include_thrust=False,
        time_step=300  # 5 minutes
    )
    
    if results['success']:
        print(f"\nPropagation successful!")
        print(f"Final altitude: {results['altitudes'][-1]/1000:.1f} km")
        print(f"Altitude change: {(results['altitudes'][-1] - results['altitudes'][0])/1000:.6f} km")
        print("(Should be ~0 for Keplerian motion)")
    
    return results


def test_j2_perturbation():
    """Test J2 perturbation effects."""
    
    print("\n" + "="*60)
    print("TEST 2: J2 PERTURBATION EFFECTS")
    print("="*60)
    
    # Create initial orbit
    initial_orbit = OrbitalElements(
        semi_major_axis=7.0e6,  # ~600 km altitude
        eccentricity=0.01,
        inclination=98.0,       # Sun-synchronous-like
        raan=0.0,
        arg_perigee=90.0,
        true_anomaly=0.0
    )
    
    print(f"Initial orbit:\n{initial_orbit}")
    
    propagator = OrbitalPropagator()
    
    # Propagate with and without J2
    print("\nPropagating with J2 perturbations...")
    results_j2 = propagator.propagate(
        initial_elements=initial_orbit,
        time_span=86400 * 7,  # 1 week
        include_j2=True,
        include_drag=False,
        include_thrust=False,
        time_step=3600  # 1 hour
    )
    
    print("\nPropagating without J2 perturbations...")
    results_no_j2 = propagator.propagate(
        initial_elements=initial_orbit,
        time_span=86400 * 7,  # 1 week
        include_j2=False,
        include_drag=False,
        include_thrust=False,
        time_step=3600  # 1 hour
    )
    
    if results_j2['success'] and results_no_j2['success']:
        # Calculate RAAN and argument of perigee drift
        raan_initial = initial_orbit.raan * CONVERSIONS['rad_to_deg']
        argp_initial = initial_orbit.w * CONVERSIONS['rad_to_deg']
        
        raan_final_j2 = results_j2['orbital_elements'][-1].raan * CONVERSIONS['rad_to_deg']
        argp_final_j2 = results_j2['orbital_elements'][-1].w * CONVERSIONS['rad_to_deg']
        
        raan_drift = raan_final_j2 - raan_initial
        argp_drift = argp_final_j2 - argp_initial
        
        print(f"\nJ2 Effects over 1 week:")
        print(f"RAAN drift: {raan_drift:.3f}°")
        print(f"Argument of perigee drift: {argp_drift:.3f}°")
    
    return results_j2, results_no_j2


def test_atmospheric_drag():
    """Test atmospheric drag effects."""
    
    print("\n" + "="*60)
    print("TEST 3: ATMOSPHERIC DRAG EFFECTS")
    print("="*60)
    
    # Create low Earth orbit
    initial_orbit = OrbitalElements(
        semi_major_axis=6.65e6,  # ~270 km altitude
        eccentricity=0.001,
        inclination=51.6,
        raan=0.0,
        arg_perigee=0.0,
        true_anomaly=0.0
    )
    
    print(f"Initial orbit:\n{initial_orbit}")
    
    # Spacecraft parameters
    ballistic_coefficient = 50.0  # kg/m² (typical for small satellite)
    
    print(f"Ballistic coefficient: {ballistic_coefficient} kg/m²")
    
    propagator = OrbitalPropagator()
    
    # Set moderate solar activity
    propagator.atmosphere.set_solar_conditions(f107_flux=150.0)
    
    print("\nPropagating with atmospheric drag...")
    results_drag = propagator.propagate(
        initial_elements=initial_orbit,
        time_span=86400 * 30,  # 30 days
        ballistic_coefficient=ballistic_coefficient,
        include_j2=True,
        include_drag=True,
        include_thrust=False,
        time_step=1800  # 30 minutes
    )
    
    print("\nPropagating without atmospheric drag...")
    results_no_drag = propagator.propagate(
        initial_elements=initial_orbit,
        time_span=86400 * 30,  # 30 days
        include_j2=True,
        include_drag=False,
        include_thrust=False,
        time_step=1800  # 30 minutes
    )
    
    if results_drag['success'] and results_no_drag['success']:
        altitude_loss = (results_drag['altitudes'][0] - results_drag['altitudes'][-1]) / 1000
        print(f"\nDrag Effects over 30 days:")
        print(f"Altitude loss due to drag: {altitude_loss:.1f} km")
        print(f"Final altitude: {results_drag['altitudes'][-1]/1000:.1f} km")
    
    return results_drag, results_no_drag


def test_thrust_maneuver():
    """Test thrust maneuver for orbit raising."""
    
    print("\n" + "="*60)
    print("TEST 4: THRUST MANEUVER (ORBIT RAISING)")
    print("="*60)
    
    # Create initial orbit
    initial_orbit = OrbitalElements(
        semi_major_axis=6.78e6,  # ~400 km altitude
        eccentricity=0.0001,
        inclination=51.6,
        raan=0.0,
        arg_perigee=0.0,
        true_anomaly=0.0
    )
    
    print(f"Initial orbit:\n{initial_orbit}")
    
    # Spacecraft parameters
    spacecraft_mass = 500.0  # kg
    thrust_magnitude = 1.0   # N (small electric thruster)
    specific_impulse = 2000.0  # s (electric propulsion)
    ballistic_coefficient = 100.0  # kg/m²
    
    print(f"Spacecraft mass: {spacecraft_mass} kg")
    print(f"Thrust: {thrust_magnitude} N")
    print(f"Specific impulse: {specific_impulse} s")
    
    propagator = OrbitalPropagator()
    
    # Thrust in prograde direction (orbit raising)
    thrust_direction = np.array([1, 0, 0])  # Will be updated to velocity direction
    
    print("\nPropagating with prograde thrust...")
    results_thrust = propagator.propagate_with_mass(
        initial_elements=initial_orbit,
        spacecraft_mass=spacecraft_mass,
        time_span=86400 * 10,  # 10 days
        ballistic_coefficient=ballistic_coefficient,
        thrust_magnitude=thrust_magnitude,
        specific_impulse=specific_impulse,
        thrust_direction=None,  # Prograde (velocity direction)
        include_j2=True,
        include_drag=True,
        include_thrust=True,
        time_step=1800  # 30 minutes
    )
    
    if results_thrust['success']:
        altitude_gain = (results_thrust['altitudes'][-1] - results_thrust['altitudes'][0]) / 1000
        fuel_consumed = results_thrust['fuel_consumed']
        fuel_fraction = fuel_consumed / spacecraft_mass * 100
        
        print(f"\nThrust Maneuver Results (10 days):")
        print(f"Altitude gain: {altitude_gain:.1f} km")
        print(f"Fuel consumed: {fuel_consumed:.2f} kg ({fuel_fraction:.1f}% of initial mass)")
        print(f"Final mass: {results_thrust['final_mass']:.1f} kg")
    
    return results_thrust


def test_deorbit_maneuver():
    """Test retrograde thrust for deorbiting."""
    
    print("\n" + "="*60)
    print("TEST 5: DEORBIT MANEUVER (RETROGRADE THRUST)")
    print("="*60)
    
    # Create higher orbit for deorbiting
    initial_orbit = OrbitalElements(
        semi_major_axis=6.98e6,  # ~600 km altitude
        eccentricity=0.001,
        inclination=98.0,
        raan=0.0,
        arg_perigee=0.0,
        true_anomaly=0.0
    )
    
    print(f"Initial orbit:\n{initial_orbit}")
    
    # Spacecraft parameters
    spacecraft_mass = 200.0   # kg
    thrust_magnitude = 5.0    # N
    specific_impulse = 240.0  # s (chemical propulsion)
    ballistic_coefficient = 75.0  # kg/m²
    
    print(f"Spacecraft mass: {spacecraft_mass} kg")
    print(f"Thrust: {thrust_magnitude} N")
    print(f"Specific impulse: {specific_impulse} s")
    
    propagator = OrbitalPropagator()
    
    # Set up event to stop when altitude drops below 100 km
    def altitude_event(t, y):
        r = y[:3]
        r_mag = np.linalg.norm(r)
        altitude = r_mag - EARTH_CONSTANTS['radius']
        return altitude - 100000  # 100 km
    altitude_event.terminal = True
    altitude_event.direction = -1
    
    print("\nPropagating with retrograde thrust for deorbiting...")
    results_deorbit = propagator.propagate_with_mass(
        initial_elements=initial_orbit,
        spacecraft_mass=spacecraft_mass,
        time_span=86400 * 365,  # 1 year max
        ballistic_coefficient=ballistic_coefficient,
        thrust_magnitude=thrust_magnitude,
        specific_impulse=specific_impulse,
        thrust_direction=None,  # Retrograde (anti-velocity)
        include_j2=True,
        include_drag=True,
        include_thrust=True,
        time_step=1800,  # 30 minutes
        events=altitude_event
    )
    
    if results_deorbit['success']:
        deorbit_time = results_deorbit['times'][-1] / 86400  # days
        fuel_consumed = results_deorbit['fuel_consumed']
        fuel_fraction = fuel_consumed / spacecraft_mass * 100
        
        print(f"\nDeorbit Maneuver Results:")
        print(f"Time to 100 km altitude: {deorbit_time:.1f} days")
        print(f"Fuel consumed: {fuel_consumed:.2f} kg ({fuel_fraction:.1f}% of initial mass)")
        print(f"Final altitude: {results_deorbit['altitudes'][-1]/1000:.1f} km")
    
    return results_deorbit


def plot_comparison_results(results_list, labels, title, save_path=None):
    """Plot comparison of multiple propagation results."""
    
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Altitude vs Time
    plt.subplot(2, 3, 1)
    for results, label in zip(results_list, labels):
        if results['success']:
            times_days = results['times'] / 86400
            altitudes_km = results['altitudes'] / 1000
            plt.plot(times_days, altitudes_km, label=label, linewidth=2)
    
    plt.xlabel('Time (days)')
    plt.ylabel('Perigee Altitude (km)')
    plt.title('Altitude Evolution')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 2: Semi-major axis
    plt.subplot(2, 3, 2)
    for results, label in zip(results_list, labels):
        if results['success']:
            times_days = results['times'] / 86400
            sma_km = [elem.a/1000 for elem in results['orbital_elements']]
            plt.plot(times_days, sma_km, label=label, linewidth=2)
    
    plt.xlabel('Time (days)')
    plt.ylabel('Semi-major Axis (km)')
    plt.title('Orbital Energy')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 3: Eccentricity
    plt.subplot(2, 3, 3)
    for results, label in zip(results_list, labels):
        if results['success']:
            times_days = results['times'] / 86400
            eccentricity = [elem.e for elem in results['orbital_elements']]
            plt.plot(times_days, eccentricity, label=label, linewidth=2)
    
    plt.xlabel('Time (days)')
    plt.ylabel('Eccentricity')
    plt.title('Orbital Shape')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 4: RAAN
    plt.subplot(2, 3, 4)
    for results, label in zip(results_list, labels):
        if results['success']:
            times_days = results['times'] / 86400
            raan_deg = [elem.raan * CONVERSIONS['rad_to_deg'] for elem in results['orbital_elements']]
            plt.plot(times_days, raan_deg, label=label, linewidth=2)
    
    plt.xlabel('Time (days)')
    plt.ylabel('RAAN (degrees)')
    plt.title('Nodal Regression')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 5: Argument of Perigee
    plt.subplot(2, 3, 5)
    for results, label in zip(results_list, labels):
        if results['success']:
            times_days = results['times'] / 86400
            argp_deg = [elem.w * CONVERSIONS['rad_to_deg'] for elem in results['orbital_elements']]
            plt.plot(times_days, argp_deg, label=label, linewidth=2)
    
    plt.xlabel('Time (days)')
    plt.ylabel('Arg. of Perigee (degrees)')
    plt.title('Apsidal Precession')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 6: 3D Trajectory (if available)
    plt.subplot(2, 3, 6)
    for results, label in zip(results_list, labels):
        if results['success']:
            positions = results['positions'] / 1000  # km
            plt.plot(positions[0, :], positions[1, :], label=label, alpha=0.7)
    
    # Earth circle
    theta = np.linspace(0, 2*np.pi, 100)
    earth_radius_km = EARTH_CONSTANTS['radius'] / 1000
    plt.plot(earth_radius_km * np.cos(theta), earth_radius_km * np.sin(theta), 
             'brown', linewidth=2, label='Earth')
    
    plt.xlabel('X (km)')
    plt.ylabel('Y (km)')
    plt.title('Orbital Trajectory (X-Y Plane)')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def main():
    """Run all tests and demonstrations."""
    
    print("🛰️  ORBITAL PROPAGATOR TEST SUITE")
    print("=" * 60)
    print("Testing orbital propagator with various perturbations...")
    
    # Test 1: Basic propagation
    try:
        results_basic = test_basic_propagation()
    except Exception as e:
        print(f"Test 1 failed: {e}")
        results_basic = None
    
    # Test 2: J2 perturbations
    try:
        results_j2, results_no_j2 = test_j2_perturbation()
    except Exception as e:
        print(f"Test 2 failed: {e}")
        results_j2, results_no_j2 = None, None
    
    # Test 3: Atmospheric drag
    try:
        results_drag, results_no_drag = test_atmospheric_drag()
    except Exception as e:
        print(f"Test 3 failed: {e}")
        results_drag, results_no_drag = None, None
    
    # Test 4: Thrust maneuver
    try:
        results_thrust = test_thrust_maneuver()
    except Exception as e:
        print(f"Test 4 failed: {e}")
        results_thrust = None
    
    # Test 5: Deorbit maneuver
    try:
        results_deorbit = test_deorbit_maneuver()
    except Exception as e:
        print(f"Test 5 failed: {e}")
        results_deorbit = None
    
    print("\n" + "="*60)
    print("🎯 ALL TESTS COMPLETED")
    print("="*60)
    
    # Plot comparisons if results are available
    if results_j2 and results_no_j2:
        print("\nGenerating J2 comparison plots...")
        plot_comparison_results(
            [results_no_j2, results_j2],
            ['No J2', 'With J2'],
            'J2 Perturbation Effects Comparison'
        )
    
    if results_drag and results_no_drag:
        print("\nGenerating drag comparison plots...")
        plot_comparison_results(
            [results_no_drag, results_drag],
            ['No Drag', 'With Drag'],
            'Atmospheric Drag Effects Comparison'
        )


if __name__ == "__main__":
    main()
