#!/usr/bin/env python3
"""
Test script for the Harris-Priester atmospheric model.

This script demonstrates the key features and usage of the Harris-Priester
atmospheric density model implementation.
"""

import numpy as np
import matplotlib.pyplot as plt
from harris_priester import HarrisPriesterAtmosphere

def test_basic_functionality():
    """Test basic functionality of the Harris-Priester model."""
    print("Testing basic Harris-Priester functionality...")
    
    # Create model instance
    atm = HarrisPriesterAtmosphere()
    
    # Test single altitude
    altitude = 400e3  # 400 km
    density = atm.density(altitude)
    print(f"Density at {altitude/1000:.0f} km: {density:.2e} kg/m³")
    
    # Test multiple altitudes
    altitudes = np.linspace(200e3, 800e3, 10)
    densities = atm.density(altitudes)
    print(f"Density range: {densities.min():.2e} to {densities.max():.2e} kg/m³")
    
    # Test with different conditions
    density_night = atm.density(400e3, local_solar_time=0)  # midnight
    density_day = atm.density(400e3, local_solar_time=12)   # noon
    print(f"Day/night ratio: {density_day/density_night:.2f}")
    
    print("Basic functionality test passed!")
    return True

def test_solar_activity_effects():
    """Test solar activity scaling."""
    print("\nTesting solar activity effects...")
    
    altitude = 400e3
    f107_values = [70, 100, 150, 200, 300]
    densities = []
    
    for f107 in f107_values:
        atm = HarrisPriesterAtmosphere(solar_flux_f107=f107)
        density = atm.density(altitude)
        densities.append(density)
        print(f"F10.7 = {f107:3d}: ρ = {density:.2e} kg/m³")
    
    # Check that density increases with solar activity
    assert all(densities[i] <= densities[i+1] for i in range(len(densities)-1))
    print("Solar activity scaling test passed!")
    return True

def test_diurnal_variation():
    """Test diurnal (day/night) variation."""
    print("\nTesting diurnal variation...")
    
    atm = HarrisPriesterAtmosphere()
    altitude = 400e3
    times = np.linspace(0, 24, 25)  # Every hour
    densities = []
    
    for time in times:
        density = atm.density(altitude, local_solar_time=time)
        densities.append(density)
    
    densities = np.array(densities)
    
    # Find min and max
    min_density = densities.min()
    max_density = densities.max()
    diurnal_ratio = max_density / min_density
    
    print(f"Minimum density: {min_density:.2e} kg/m³")
    print(f"Maximum density: {max_density:.2e} kg/m³") 
    print(f"Diurnal ratio: {diurnal_ratio:.2f}")
    
    # Check that there is significant diurnal variation
    assert diurnal_ratio > 1.2  # At least 20% variation
    print("Diurnal variation test passed!")
    return True

def test_altitude_scaling():
    """Test altitude scaling and extrapolation."""
    print("\nTesting altitude scaling...")
    
    atm = HarrisPriesterAtmosphere()
    
    # Test within model range
    alt_low = 150e3   # 150 km
    alt_high = 800e3  # 800 km
    
    rho_low = atm.density(alt_low)
    rho_high = atm.density(alt_high)
    
    print(f"Density at {alt_low/1000:.0f} km: {rho_low:.2e} kg/m³")
    print(f"Density at {alt_high/1000:.0f} km: {rho_high:.2e} kg/m³")
    
    # Density should decrease with altitude
    assert rho_low > rho_high
    
    # Test scale height calculation
    scale_height = atm.scale_height(400e3)
    print(f"Scale height at 400 km: {scale_height/1000:.1f} km")
    
    # Scale height should be reasonable (20-100 km range)
    assert 20e3 < scale_height < 100e3
    
    print("Altitude scaling test passed!")
    return True

def plot_density_profiles():
    """Create plots showing density profiles and variations."""
    print("\nCreating density profile plots...")
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Density vs altitude for different solar conditions
    altitudes = np.linspace(150e3, 800e3, 100)
    
    for f107 in [70, 150, 300]:
        atm = HarrisPriesterAtmosphere(solar_flux_f107=f107)
        densities = atm.density(altitudes)
        ax1.semilogy(altitudes/1000, densities, label=f'F10.7 = {f107}')
    
    ax1.set_xlabel('Altitude (km)')
    ax1.set_ylabel('Density (kg/m³)')
    ax1.set_title('Density vs Altitude (Different Solar Activity)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Diurnal variation at different altitudes
    times = np.linspace(0, 24, 25)
    atm = HarrisPriesterAtmosphere()
    
    for alt_km in [200, 400, 600]:
        densities = []
        for time in times:
            density = atm.density(alt_km * 1000, local_solar_time=time)
            densities.append(density)
        ax2.plot(times, densities, label=f'{alt_km} km')
    
    ax2.set_xlabel('Local Solar Time (hours)')
    ax2.set_ylabel('Density (kg/m³)')
    ax2.set_title('Diurnal Variation')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Annual variation
    days = np.linspace(1, 365, 50)
    densities_eq = []  # Equator
    densities_60 = []  # 60° latitude
    
    for day in days:
        rho_eq = atm.density(400e3, latitude=0, day_of_year=int(day))
        rho_60 = atm.density(400e3, latitude=60, day_of_year=int(day))
        densities_eq.append(rho_eq)
        densities_60.append(rho_60)
    
    ax3.plot(days, densities_eq, label='Equator')
    ax3.plot(days, densities_60, label='60° Latitude')
    ax3.set_xlabel('Day of Year')
    ax3.set_ylabel('Density (kg/m³)')
    ax3.set_title('Annual Variation at 400 km')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Solar activity scaling
    f107_range = np.linspace(70, 300, 50)
    densities_solar = []
    
    for f107 in f107_range:
        atm_temp = HarrisPriesterAtmosphere(solar_flux_f107=f107)
        density = atm_temp.density(400e3)
        densities_solar.append(density)
    
    ax4.plot(f107_range, densities_solar)
    ax4.set_xlabel('F10.7 Solar Flux')
    ax4.set_ylabel('Density (kg/m³)')
    ax4.set_title('Solar Activity Scaling at 400 km')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('harris_priester_analysis.png', dpi=300, bbox_inches='tight')
    print("Plots saved as 'harris_priester_analysis.png'")
    
    # Show plots if possible
    try:
        plt.show()
    except:
        pass

def run_comprehensive_test():
    """Run comprehensive test suite."""
    print("Harris-Priester Atmospheric Model - Comprehensive Test")
    print("=" * 60)
    
    tests = [
        test_basic_functionality,
        test_solar_activity_effects, 
        test_diurnal_variation,
        test_altitude_scaling
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"Test failed: {e}")
    
    print(f"\nTest Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("All tests passed! ✅")
        
        # Create plots
        try:
            plot_density_profiles()
        except Exception as e:
            print(f"Plot generation failed: {e}")
            
    else:
        print("Some tests failed! ❌")
    
    return passed == total

if __name__ == "__main__":
    run_comprehensive_test()
