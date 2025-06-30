"""
Orbital Elements Module for Astrodynamics Calculations

This module provides orbital elements representation and conversions between
coordinate systems for satellite orbit propagation.

Author: PhD Design Research
Date: June 2025
"""

import numpy as np
import warnings
from constants import EARTH_CONSTANTS, CONVERSIONS


class OrbitalElements:
    """
    Classical orbital elements representation and calculations.
    
    Parameters:
    -----------
    semi_major_axis : float
        Semi-major axis in meters
    eccentricity : float
        Orbital eccentricity (0 ≤ e < 1 for elliptical orbits)
    inclination : float
        Orbital inclination in degrees
    raan : float
        Right Ascension of Ascending Node in degrees
    arg_perigee : float
        Argument of perigee in degrees
    true_anomaly : float
        True anomaly in degrees
    epoch : float, optional
        Epoch time for the elements (seconds since J2000). Default is 0.
    mu : float, optional
        Central body gravitational parameter. Default is Earth's mu.
    """
    
    def __init__(self, semi_major_axis, eccentricity, inclination, 
                 raan, arg_perigee, true_anomaly, epoch=0.0, 
                 mu=EARTH_CONSTANTS['mu']):
        """Initialize orbital elements."""
        
        # Store orbital elements (convert angles to radians)
        self.a = float(semi_major_axis)
        self.e = float(eccentricity)
        self.i = float(inclination) * CONVERSIONS['deg_to_rad']
        self.raan = float(raan) * CONVERSIONS['deg_to_rad']
        self.w = float(arg_perigee) * CONVERSIONS['deg_to_rad']
        self.nu = float(true_anomaly) * CONVERSIONS['deg_to_rad']
        self.epoch = float(epoch)
        self.mu = float(mu)
        
        # Validate orbital elements
        self._validate_elements()
    
    def _validate_elements(self):
        """Validate orbital elements for physical consistency."""
        
        # Check semi-major axis
        if self.a <= 0:
            raise ValueError(f"Semi-major axis must be positive, got {self.a}")
        
        # Check eccentricity
        if self.e < 0:
            raise ValueError(f"Eccentricity must be non-negative, got {self.e}")
        
        if self.e >= 1.0:
            warnings.warn(f"Eccentricity {self.e:.4f} ≥ 1 indicates non-elliptical orbit")
        
        # Check inclination
        if not (0 <= self.i <= np.pi):
            warnings.warn(f"Inclination {self.i * CONVERSIONS['rad_to_deg']:.1f}° "
                         f"outside normal range [0°, 180°]")
        
        # Check if orbit intersects Earth
        perigee_radius = self.a * (1 - self.e)
        if perigee_radius < EARTH_CONSTANTS['radius']:
            altitude_deficit = (EARTH_CONSTANTS['radius'] - perigee_radius) / 1000  # km
            if altitude_deficit > 50:  # Only warn if more than 50 km below surface
                warnings.warn(f"Perigee radius {perigee_radius/1000:.1f} km is {altitude_deficit:.1f} km "
                             f"below Earth's surface - orbit intersects Earth")
    
    @property
    def perigee_radius(self):
        """Perigee radius in meters."""
        return self.a * (1 - self.e)
    
    @property
    def apogee_radius(self):
        """Apogee radius in meters."""
        return self.a * (1 + self.e)
    
    @property
    def perigee_altitude(self):
        """Perigee altitude above Earth's surface in meters."""
        return self.perigee_radius - EARTH_CONSTANTS['radius']
    
    @property
    def apogee_altitude(self):
        """Apogee altitude above Earth's surface in meters."""
        return self.apogee_radius - EARTH_CONSTANTS['radius']
    
    @property
    def period(self):
        """Orbital period in seconds."""
        return 2 * np.pi * np.sqrt(self.a**3 / self.mu)
    
    @property
    def mean_motion(self):
        """Mean motion in rad/s."""
        return np.sqrt(self.mu / self.a**3)
    
    def to_cartesian(self):
        """
        Convert orbital elements to Cartesian position and velocity vectors.
        
        Returns:
        --------
        tuple
            (position, velocity) vectors in ECI coordinates [m, m/s]
        """
        
        # Semi-latus rectum
        p = self.a * (1 - self.e**2)
        
        # Current radius
        r_mag = p / (1 + self.e * np.cos(self.nu))
        
        # Position and velocity in perifocal coordinates
        r_pqw = r_mag * np.array([np.cos(self.nu), np.sin(self.nu), 0])
        
        v_pqw = np.sqrt(self.mu / p) * np.array([
            -np.sin(self.nu),
            self.e + np.cos(self.nu),
            0
        ])
        
        # Transformation matrix from perifocal to ECI
        R = self._rotation_matrix_pqw_to_eci()
        
        # Transform to ECI coordinates
        r_eci = R @ r_pqw
        v_eci = R @ v_pqw
        
        return r_eci, v_eci
    
    def _rotation_matrix_pqw_to_eci(self):
        """Calculate rotation matrix from perifocal to ECI coordinates."""
        
        cos_raan = np.cos(self.raan)
        sin_raan = np.sin(self.raan)
        cos_i = np.cos(self.i)
        sin_i = np.sin(self.i)
        cos_w = np.cos(self.w)
        sin_w = np.sin(self.w)
        
        # Rotation matrix elements
        R11 = cos_raan * cos_w - sin_raan * sin_w * cos_i
        R12 = -cos_raan * sin_w - sin_raan * cos_w * cos_i
        R13 = sin_raan * sin_i
        
        R21 = sin_raan * cos_w + cos_raan * sin_w * cos_i
        R22 = -sin_raan * sin_w + cos_raan * cos_w * cos_i
        R23 = -cos_raan * sin_i
        
        R31 = sin_w * sin_i
        R32 = cos_w * sin_i
        R33 = cos_i
        
        return np.array([
            [R11, R12, R13],
            [R21, R22, R23],
            [R31, R32, R33]
        ])
    
    @classmethod
    def from_cartesian(cls, position, velocity, mu=EARTH_CONSTANTS['mu'], epoch=0.0):
        """
        Create orbital elements from Cartesian position and velocity vectors.
        
        Parameters:
        -----------
        position : array-like
            Position vector in ECI coordinates [m]
        velocity : array-like
            Velocity vector in ECI coordinates [m/s]
        mu : float, optional
            Gravitational parameter [m³/s²]
        epoch : float, optional
            Epoch time [s]
            
        Returns:
        --------
        OrbitalElements
            Orbital elements object
        """
        
        r = np.array(position, dtype=float)
        v = np.array(velocity, dtype=float)
        
        r_mag = np.linalg.norm(r)
        v_mag = np.linalg.norm(v)
        
        # Angular momentum vector
        h = np.cross(r, v)
        h_mag = np.linalg.norm(h)
        
        # Node vector
        n = np.cross([0, 0, 1], h)
        n_mag = np.linalg.norm(n)
        
        # Eccentricity vector
        e_vec = ((v_mag**2 - mu/r_mag) * r - np.dot(r, v) * v) / mu
        e = np.linalg.norm(e_vec)
        
        # Semi-major axis
        energy = v_mag**2/2 - mu/r_mag
        if abs(energy) > 1e-10:  # Not parabolic
            a = -mu / (2 * energy)
        else:
            raise ValueError("Parabolic orbit detected")
        
        # Inclination
        i = np.arccos(h[2] / h_mag)
        
        # Right ascension of ascending node
        if n_mag > 1e-10:
            raan = np.arccos(n[0] / n_mag)
            if n[1] < 0:
                raan = 2*np.pi - raan
        else:
            raan = 0.0  # Equatorial orbit
        
        # Argument of perigee
        if n_mag > 1e-10 and e > 1e-10:
            w = np.arccos(np.dot(n, e_vec) / (n_mag * e))
            if e_vec[2] < 0:
                w = 2*np.pi - w
        else:
            w = 0.0
        
        # True anomaly
        if e > 1e-10:
            nu = np.arccos(np.dot(e_vec, r) / (e * r_mag))
            if np.dot(r, v) < 0:
                nu = 2*np.pi - nu
        else:
            # Circular orbit
            if n_mag > 1e-10:
                nu = np.arccos(np.dot(n, r) / (n_mag * r_mag))
                if r[2] < 0:
                    nu = 2*np.pi - nu
            else:
                nu = np.arccos(r[0] / r_mag)
                if r[1] < 0:
                    nu = 2*np.pi - nu
        
        # Convert to degrees
        i_deg = i * CONVERSIONS['rad_to_deg']
        raan_deg = raan * CONVERSIONS['rad_to_deg']
        w_deg = w * CONVERSIONS['rad_to_deg']
        nu_deg = nu * CONVERSIONS['rad_to_deg']
        
        return cls(a, e, i_deg, raan_deg, w_deg, nu_deg, epoch, mu)
    
    def __str__(self):
        """String representation of orbital elements."""
        return f"""Orbital Elements:
  Semi-major axis: {self.a/1000:.1f} km
  Eccentricity: {self.e:.6f}
  Inclination: {self.i * CONVERSIONS['rad_to_deg']:.3f}°
  RAAN: {self.raan * CONVERSIONS['rad_to_deg']:.3f}°
  Arg. of Perigee: {self.w * CONVERSIONS['rad_to_deg']:.3f}°
  True Anomaly: {self.nu * CONVERSIONS['rad_to_deg']:.3f}°
  Perigee Alt: {self.perigee_altitude/1000:.1f} km
  Apogee Alt: {self.apogee_altitude/1000:.1f} km
  Period: {self.period/3600:.2f} hours"""
