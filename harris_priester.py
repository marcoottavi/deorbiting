"""
Harris-Priester Atmospheric Density Model

This module implements the Harris-Priester atmospheric density model, which is
a semi-empirical model used for satellite orbit propagation. The model provides
atmospheric density as a function of altitude, geographic location, and time.

The Harris-Priester model features:
- Tabulated density values at discrete altitude levels (100-1000 km)
- Interpolation between altitude levels using exponential functions
- Diurnal (day/night) density variations based on solar heating
- Annual (seasonal) density variations
- Solar activity scaling based on F10.7 solar flux index

References:
- Harris, I. and Priester, W. (1962). "Time-dependent structure of the upper atmosphere"
- Montenbruck, O. and Gill, E. (2000). "Satellite Orbits: Models, Methods and Applications"
- Vallado, D. A. (2013). "Fundamentals of Astrodynamics and Applications"

Author: PhD Design Research
Date: June 2025
"""

import numpy as np
from scipy.interpolate import interp1d
import warnings

try:
    from .constants import HARRIS_PRIESTER, EARTH_CONSTANTS, PHYSICAL_CONSTANTS, CONVERSIONS
except ImportError:
    from constants import HARRIS_PRIESTER, EARTH_CONSTANTS, PHYSICAL_CONSTANTS, CONVERSIONS


class HarrisPriesterAtmosphere:
    """
    Harris-Priester atmospheric density model implementation.
    
    This class computes atmospheric density using the Harris-Priester model,
    which accounts for diurnal and seasonal variations in atmospheric density
    due to solar heating effects.
    
    Parameters:
    -----------
    solar_flux_f107 : float, optional
        Solar flux F10.7 index (10.7 cm radio flux). Default is 150 (average).
        Range: 70 (solar minimum) to 300+ (solar maximum)
    exospheric_temperature : float, optional
        Exospheric temperature in Kelvin. Default is 900 K.
        If None, temperature is calculated from F10.7 flux.
    include_diurnal : bool, optional
        Include diurnal (day/night) variations. Default is True.
    include_annual : bool, optional
        Include annual (seasonal) variations. Default is True.
        
    Attributes:
    -----------
    f107 : float
        Current F10.7 solar flux index
    temp_exo : float
        Current exospheric temperature [K]
    include_diurnal : bool
        Whether diurnal variations are included
    include_annual : bool
        Whether annual variations are included
    altitudes : np.ndarray
        Tabulated altitude levels [m]
    rho_min : np.ndarray
        Minimum density values [kg/m³]
    rho_max : np.ndarray
        Maximum density values [kg/m³]
    """
    
    def __init__(self, solar_flux_f107=150.0, exospheric_temperature=None,
                 include_diurnal=True, include_annual=True):
        """Initialize the Harris-Priester atmospheric model."""
        
        # Store configuration
        self.f107 = float(solar_flux_f107)
        self.include_diurnal = include_diurnal
        self.include_annual = include_annual
        
        # Load model data
        self.altitudes = HARRIS_PRIESTER['altitudes'].copy()
        self.rho_min = HARRIS_PRIESTER['rho_min'].copy()
        self.rho_max = HARRIS_PRIESTER['rho_max'].copy()
        self.n_diurnal = HARRIS_PRIESTER['n_diurnal']
        self.m_annual = HARRIS_PRIESTER['m_annual']
        self.lag_angle = HARRIS_PRIESTER['lag_angle'] * CONVERSIONS['deg_to_rad']
        
        # Set temperature
        if exospheric_temperature is None:
            self.temp_exo = self._calculate_temperature_from_f107()
        else:
            self.temp_exo = float(exospheric_temperature)
        
        # Validate inputs
        self._validate_inputs()
        
        # Setup interpolation functions
        self._setup_interpolators()
        
        # Calculate solar activity scaling
        self._update_solar_scaling()
        
        # Cache for performance
        self._cache = {}
    
    def _validate_inputs(self):
        """Validate input parameters."""
        if self.f107 < 50.0:
            warnings.warn(f"F10.7 flux ({self.f107}) is very low. Minimum realistic value is ~70.")
        elif self.f107 > 500.0:
            warnings.warn(f"F10.7 flux ({self.f107}) is very high. Typical maximum is ~400.")
        
        if self.temp_exo < 500.0 or self.temp_exo > 2000.0:
            warnings.warn(f"Exospheric temperature ({self.temp_exo:.0f} K) is outside typical range (600-1500 K).")
    
    def _calculate_temperature_from_f107(self):
        """Calculate exospheric temperature from F10.7 flux using empirical relation."""
        f107_ref = HARRIS_PRIESTER['f107_ref']
        temp_ref = HARRIS_PRIESTER['temp_exo_ref']
        
        # Empirical relationship: T = T_ref * (1 + α * (F10.7 - F10.7_ref) / F10.7_ref)
        alpha = 0.3  # Scaling factor
        temp_factor = 1.0 + alpha * (self.f107 - f107_ref) / f107_ref
        
        return temp_ref * temp_factor
    
    def _setup_interpolators(self):
        """Setup interpolation functions for density tables."""
        # Ensure data is sorted by altitude
        sort_indices = np.argsort(self.altitudes)
        alt_sorted = self.altitudes[sort_indices]
        rho_min_sorted = self.rho_min[sort_indices]
        rho_max_sorted = self.rho_max[sort_indices]
        
        # Create interpolators using logarithmic densities for better extrapolation
        self.interp_min = interp1d(
            alt_sorted, np.log(rho_min_sorted),
            kind='linear', 
            fill_value='extrapolate',
            bounds_error=False
        )
        
        self.interp_max = interp1d(
            alt_sorted, np.log(rho_max_sorted),
            kind='linear',
            fill_value='extrapolate', 
            bounds_error=False
        )
        
        # Store altitude bounds
        self.min_altitude = alt_sorted[0]
        self.max_altitude = alt_sorted[-1]
    
    def _update_solar_scaling(self):
        """Update solar activity scaling factor."""
        f107_ref = HARRIS_PRIESTER['f107_ref']
        
        # Solar activity scaling (empirical)
        # Density scales approximately as sqrt(F10.7)
        self.solar_scale = np.sqrt(self.f107 / f107_ref)
    
    def set_solar_conditions(self, f107_flux, exospheric_temp=None):
        """
        Update solar activity conditions.
        
        Parameters:
        -----------
        f107_flux : float
            F10.7 solar flux index
        exospheric_temp : float, optional
            Exospheric temperature [K]. If None, calculated from F10.7.
        """
        self.f107 = float(f107_flux)
        
        if exospheric_temp is not None:
            self.temp_exo = float(exospheric_temp)
        else:
            self.temp_exo = self._calculate_temperature_from_f107()
        
        self._validate_inputs()
        self._update_solar_scaling()
        
        # Clear cache since conditions changed
        self._cache.clear()
    
    def density(self, altitude, latitude=0.0, local_solar_time=12.0, day_of_year=80):
        """
        Calculate atmospheric density using Harris-Priester model.
        
        This is the main method that computes atmospheric density at given
        conditions, accounting for altitude, geographic location, and time.
        
        Parameters:
        -----------
        altitude : float or array-like
            Altitude above Earth's surface [m]
        latitude : float or array-like, optional
            Geodetic latitude [degrees], range: -90 to +90. Default is 0 (equator).
        local_solar_time : float or array-like, optional  
            Local solar time [hours], range: 0-24. Default is 12 (noon).
        day_of_year : int or array-like, optional
            Day of year, range: 1-365. Default is 80 (around spring equinox).
            
        Returns:
        --------
        float or np.ndarray
            Atmospheric density [kg/m³]
            
        Notes:
        ------
        The density calculation involves several steps:
        1. Interpolate base densities from tabulated values
        2. Calculate diurnal variation factor from solar elevation
        3. Calculate annual variation factor from seasonal effects  
        4. Apply solar activity scaling
        5. Apply altitude-dependent corrections
        
        Examples:
        ---------
        >>> atm = HarrisPriesterAtmosphere()
        >>> # Density at 400 km altitude, equator, noon
        >>> rho = atm.density(400e3)
        >>> print(f"Density: {rho:.2e} kg/m³")
        
        >>> # Density at multiple altitudes
        >>> altitudes = np.array([300, 400, 500, 600]) * 1000  # km to m
        >>> densities = atm.density(altitudes)
        """
        
        # Convert inputs to numpy arrays for vectorized operations
        altitude = np.atleast_1d(altitude)
        latitude = np.atleast_1d(latitude)
        local_solar_time = np.atleast_1d(local_solar_time)
        day_of_year = np.atleast_1d(day_of_year)
        
        # Broadcast all arrays to same shape
        altitude, latitude, local_solar_time, day_of_year = np.broadcast_arrays(
            altitude, latitude, local_solar_time, day_of_year
        )
        
        # Check altitude bounds
        self._check_altitude_bounds(altitude)
        
        # Get base densities from interpolation
        log_rho_min = self.interp_min(altitude)
        log_rho_max = self.interp_max(altitude)
        rho_min = np.exp(log_rho_min)
        rho_max = np.exp(log_rho_max)
        
        # Start with geometric mean density
        rho_base = np.sqrt(rho_min * rho_max)
        
        # Initialize result with base density
        density_result = rho_base.copy()
        
        # Apply diurnal variation
        if self.include_diurnal:
            diurnal_factor = self._compute_diurnal_factor(
                latitude, local_solar_time, day_of_year
            )
            # Interpolate between min and max densities
            density_result = rho_min + (rho_max - rho_min) * diurnal_factor
        
        # Apply annual variation
        if self.include_annual:
            annual_factor = self._compute_annual_factor(day_of_year, latitude)
            density_result *= annual_factor
        
        # Apply solar activity scaling
        density_result *= self.solar_scale
        
        # Apply altitude-dependent corrections
        alt_correction = self._altitude_correction(altitude)
        density_result *= alt_correction
        
        # Return scalar if input was scalar
        if np.isscalar(density_result) or density_result.shape == ():
            return float(density_result)
        elif density_result.shape == (1,):
            return float(density_result[0])
        else:
            return density_result
    
    def _check_altitude_bounds(self, altitude):
        """Check and warn about altitude bounds."""
        min_alt = np.min(altitude)
        max_alt = np.max(altitude)
        
        if min_alt < self.min_altitude:
            deficit = (self.min_altitude - min_alt) / 1000
            if deficit > 20:  # Only warn for significant extrapolation
                warnings.warn(f"Altitude {min_alt/1000:.1f} km is {deficit:.1f} km below "
                             f"model range ({self.min_altitude/1000:.0f} km). Extrapolation used.")
        
        if max_alt > self.max_altitude:
            excess = (max_alt - self.max_altitude) / 1000
            warnings.warn(f"Altitude {max_alt/1000:.1f} km is {excess:.1f} km above "
                         f"model range ({self.max_altitude/1000:.0f} km). Extrapolation used.")
    
    def _compute_diurnal_factor(self, latitude, local_solar_time, day_of_year):
        """
        Compute diurnal variation factor from solar illumination.
        
        The diurnal factor varies from 0 (night, minimum density) to 1 (day, maximum density)
        based on the solar elevation angle above the local horizon.
        
        Parameters:
        -----------
        latitude : np.ndarray
            Latitude [degrees]
        local_solar_time : np.ndarray
            Local solar time [hours]
        day_of_year : np.ndarray
            Day of year [1-365]
            
        Returns:
        --------
        np.ndarray
            Diurnal factor [0-1]
        """
        # Convert latitude to radians
        lat_rad = latitude * CONVERSIONS['deg_to_rad']
        
        # Solar declination angle (seasonal variation)
        declination = 23.45 * CONVERSIONS['deg_to_rad'] * np.sin(
            2 * np.pi * (day_of_year - 81) / 365.25
        )
        
        # Hour angle from local solar time
        hour_angle = (local_solar_time - 12.0) * 15.0 * CONVERSIONS['deg_to_rad']
        
        # Solar elevation angle
        sin_elevation = (np.sin(lat_rad) * np.sin(declination) + 
                        np.cos(lat_rad) * np.cos(declination) * np.cos(hour_angle))
        
        # Clamp to valid range
        sin_elevation = np.clip(sin_elevation, -1.0, 1.0)
        
        # Convert to cosine of zenith angle (complement of elevation)
        cos_zenith = sin_elevation
        
        # Only use positive values (sun above horizon)
        cos_zenith_pos = np.maximum(cos_zenith, 0.0)
        
        # Harris-Priester diurnal factor: 0.5 * (1 + cos^n(zenith))
        diurnal_factor = 0.5 * (1.0 + cos_zenith_pos ** self.n_diurnal)
        
        return diurnal_factor
    
    def _compute_annual_factor(self, day_of_year, latitude):
        """
        Compute annual (seasonal) variation factor.
        
        Parameters:
        -----------
        day_of_year : np.ndarray
            Day of year [1-365]
        latitude : np.ndarray
            Latitude [degrees]
            
        Returns:
        --------
        np.ndarray
            Annual variation factor
        """
        # Convert latitude to radians
        lat_rad = latitude * CONVERSIONS['deg_to_rad']
        
        # Annual phase angle with lag
        annual_phase = 2 * np.pi * (day_of_year - 1) / 365.25 + self.lag_angle
        
        # Latitude-dependent amplitude (stronger at higher latitudes)
        amplitude = 0.15 * (1.0 + 0.5 * np.cos(2 * lat_rad))
        
        # Annual variation factor
        annual_factor = 1.0 + amplitude * np.cos(annual_phase) ** self.m_annual
        
        return annual_factor
    
    def _altitude_correction(self, altitude):
        """
        Apply empirical altitude-dependent corrections.
        
        These corrections account for model limitations at very low altitudes
        where the Harris-Priester model may not be as accurate.
        
        Parameters:
        -----------
        altitude : np.ndarray
            Altitude [m]
            
        Returns:
        --------
        np.ndarray
            Altitude correction factor
        """
        # Reference altitude for scaling (120 km)
        h_ref = 120000.0
        
        # Initialize correction factor
        correction = np.ones_like(altitude)
        
        # Apply correction for altitudes below reference
        low_alt_mask = altitude < h_ref
        if np.any(low_alt_mask):
            h_ratio = altitude[low_alt_mask] / h_ref
            # Exponential correction for low altitudes
            correction[low_alt_mask] = np.exp(0.5 * (h_ratio - 1.0))
        
        return correction
    
    def scale_height(self, altitude):
        """
        Calculate atmospheric scale height at given altitude.
        
        Scale height is the altitude interval over which atmospheric density
        decreases by a factor of e (≈2.718).
        
        Parameters:
        -----------
        altitude : float or array-like
            Altitude [m]
            
        Returns:
        --------
        float or np.ndarray
            Scale height [m]
            
        Notes:
        ------
        Scale height H = kT/(mg) where:
        - k is Boltzmann constant
        - T is temperature  
        - m is mean molecular mass
        - g is gravitational acceleration
        """
        altitude = np.atleast_1d(altitude)
        
        # Mean molecular mass in upper atmosphere (≈27 g/mol for atomic oxygen dominated)
        mean_molecular_mass = 27e-3  # kg/mol
        
        # Gravitational acceleration at altitude
        r = EARTH_CONSTANTS['radius'] + altitude
        g_alt = EARTH_CONSTANTS['mu'] / r**2
        
        # Temperature profile (simplified - constant exospheric temperature)
        # In reality, temperature varies with altitude, but for scale height
        # calculation, we use the exospheric temperature
        temperature = self.temp_exo
        
        # Scale height calculation
        scale_height = (PHYSICAL_CONSTANTS['k_boltzmann'] * temperature * 
                       PHYSICAL_CONSTANTS['avogadro'] / 
                       (mean_molecular_mass * PHYSICAL_CONSTANTS['avogadro'] * g_alt))
        
        # Return scalar if input was scalar
        if np.isscalar(scale_height) or scale_height.shape == ():
            return float(scale_height)
        elif scale_height.shape == (1,):
            return float(scale_height[0])
        else:
            return scale_height
    
    def get_model_info(self):
        """
        Get information about the current model configuration.
        
        Returns:
        --------
        dict
            Model configuration and parameters
        """
        return {
            'model_name': 'Harris-Priester Atmospheric Model',
            'altitude_range': f"{self.min_altitude/1000:.0f} - {self.max_altitude/1000:.0f} km",
            'f107_flux': self.f107,
            'exospheric_temperature': f"{self.temp_exo:.0f} K",
            'solar_scale_factor': f"{self.solar_scale:.3f}",
            'include_diurnal': self.include_diurnal,
            'include_annual': self.include_annual,
            'diurnal_exponent': self.n_diurnal,
            'annual_exponent': self.m_annual,
            'lag_angle': f"{self.lag_angle * CONVERSIONS['rad_to_deg']:.1f}°"
        }
    
    def __repr__(self):
        """String representation of the Harris-Priester model."""
        return (f"HarrisPriesterAtmosphere(F10.7={self.f107:.1f}, "
                f"T_exo={self.temp_exo:.0f}K, "
                f"diurnal={self.include_diurnal}, annual={self.include_annual})")


def demo_harris_priester():
    """
    Demonstration of Harris-Priester atmospheric model usage.
    
    This function shows various ways to use the Harris-Priester model
    and demonstrates its key features.
    """
    print("Harris-Priester Atmospheric Model Demo")
    print("=" * 50)
    
    # Create model instance
    atm = HarrisPriesterAtmosphere(solar_flux_f107=150.0)
    print(f"Model: {atm}")
    print()
    
    # Model information
    info = atm.get_model_info()
    print("Model Configuration:")
    for key, value in info.items():
        print(f"  {key.replace('_', ' ').title()}: {value}")
    print()
    
    # Single point calculation
    altitude = 400e3  # 400 km
    rho = atm.density(altitude)
    print(f"Density at {altitude/1000:.0f} km: {rho:.2e} kg/m³")
    print()
    
    # Multiple altitudes
    altitudes = np.array([200, 300, 400, 500, 600, 800]) * 1000  # km to m
    densities = atm.density(altitudes)
    
    print("Density vs Altitude:")
    print("Altitude [km]  Density [kg/m³]")
    print("-" * 30)
    for alt, rho in zip(altitudes/1000, densities):
        print(f"{alt:8.0f}      {rho:.2e}")
    print()
    
    # Diurnal variation
    alt_test = 400e3
    times = np.array([0, 6, 12, 18])  # hours
    densities_diurnal = atm.density(alt_test, local_solar_time=times)
    
    print(f"Diurnal Variation at {alt_test/1000:.0f} km:")
    print("Time [h]  Density [kg/m³]")
    print("-" * 25)
    for t, rho in zip(times, densities_diurnal):
        print(f"{t:6.0f}    {rho:.2e}")
    print()
    
    # Solar activity comparison
    print("Solar Activity Effects:")
    print("F10.7     Density [kg/m³] at 400 km")
    print("-" * 35)
    for f107 in [70, 100, 150, 200, 300]:
        atm.set_solar_conditions(f107)
        rho = atm.density(400e3)
        print(f"{f107:5.0f}     {rho:.2e}")
    
    print("\nDemo completed!")


if __name__ == "__main__":
    demo_harris_priester()
