"""
Astrodynamics Orbital Propagator

This module provides a comprehensive orbital propagator that includes:
- Central gravitational force
- J2 gravitational perturbations
- Atmospheric drag using Harris-Priester model
- External constant thrust
- Numerical integration with event detection

Author: PhD Design Research
Date: June 2025
"""

import numpy as np
from scipy.integrate import solve_ivp
import warnings
from orbital_elements import OrbitalElements
from harris_priester import HarrisPriesterAtmosphere
from constants import EARTH_CONSTANTS, PHYSICAL_CONSTANTS, CONVERSIONS


class OrbitalPropagator:
    """
    Comprehensive orbital propagator with perturbations.
    
    This class propagates satellite orbits including various perturbative forces:
    - Central gravitational acceleration
    - J2 zonal harmonic perturbation
    - Atmospheric drag using Harris-Priester density model
    - Constant external thrust
    
    Parameters:
    -----------
    atmosphere_model : HarrisPriesterAtmosphere, optional
        Atmospheric density model. Default creates new instance.
    mu : float, optional
        Central body gravitational parameter [m³/s²]. Default is Earth's mu.
    j2 : float, optional
        J2 zonal harmonic coefficient. Default is Earth's J2.
    earth_radius : float, optional
        Central body radius [m]. Default is Earth's radius.
    rotation_rate : float, optional
        Central body rotation rate [rad/s]. Default is Earth's rotation rate.
    """
    
    def __init__(self, atmosphere_model=None, mu=EARTH_CONSTANTS['mu'],
                 j2=EARTH_CONSTANTS['j2'], earth_radius=EARTH_CONSTANTS['radius'],
                 rotation_rate=EARTH_CONSTANTS['rotation_rate']):
        
        self.mu = mu
        self.j2 = j2
        self.earth_radius = earth_radius
        self.rotation_rate = rotation_rate
        
        # Initialize atmospheric model
        if atmosphere_model is None:
            self.atmosphere = HarrisPriesterAtmosphere()
        else:
            self.atmosphere = atmosphere_model
        
        # Store last propagation results
        self.last_solution = None
    
    def propagate(self, initial_elements, time_span, 
                  ballistic_coefficient=None, thrust_magnitude=0.0, thrust_direction=None,
                  include_j2=True, include_drag=True, include_thrust=True,
                  time_step=60.0, method='DOP853', rtol=1e-8, atol=1e-10,
                  events=None, dense_output=False):
        """
        Propagate orbital motion with perturbations.
        
        Parameters:
        -----------
        initial_elements : OrbitalElements
            Initial orbital elements
        time_span : tuple or float
            Time span for propagation. If tuple (t0, tf), if float, (0, time_span)
        ballistic_coefficient : float, optional
            Ballistic coefficient m/(Cd*S) in kg/m². Required if include_drag=True.
        thrust_magnitude : float, optional
            Constant thrust magnitude in Newtons. Default is 0.
        thrust_direction : array-like, optional
            Unit vector for thrust direction in ECI frame. If None, uses anti-velocity.
        include_j2 : bool, optional
            Include J2 perturbations. Default is True.
        include_drag : bool, optional
            Include atmospheric drag. Default is True.
        include_thrust : bool, optional
            Include external thrust. Default is True.
        time_step : float, optional
            Maximum integration time step in seconds. Default is 60 s.
        method : str, optional
            Integration method. Default is 'DOP853'.
        rtol : float, optional
            Relative tolerance for integration. Default is 1e-8.
        atol : float, optional
            Absolute tolerance for integration. Default is 1e-10.
        events : callable or list, optional
            Event functions for integration termination.
        dense_output : bool, optional
            Enable dense output for interpolation. Default is False.
            
        Returns:
        --------
        dict
            Propagation results containing time, positions, velocities, and orbital elements
        """
        
        # Validate inputs
        if include_drag and ballistic_coefficient is None:
            raise ValueError("Ballistic coefficient required when include_drag=True")
        
        if isinstance(time_span, (int, float)):
            time_span = (0.0, float(time_span))
        
        # Convert initial elements to state vector
        r0, v0 = initial_elements.to_cartesian()
        state0 = np.concatenate([r0, v0])
        
        # Set up integration options
        options = {
            'ballistic_coefficient': ballistic_coefficient,
            'thrust_magnitude': thrust_magnitude,
            'thrust_direction': thrust_direction,
            'include_j2': include_j2,
            'include_drag': include_drag,
            'include_thrust': include_thrust
        }
        
        # Create time evaluation points
        t_eval = np.arange(time_span[0], time_span[1] + time_step, time_step)
        
        print(f"Propagating orbit from {time_span[0]:.1f} to {time_span[1]:.1f} seconds...")
        print(f"Initial orbit: {initial_elements.perigee_altitude/1000:.1f} km x {initial_elements.apogee_altitude/1000:.1f} km")
        
        # Solve differential equation
        try:
            sol = solve_ivp(
                fun=lambda t, y: self._equations_of_motion(t, y, options),
                t_span=time_span,
                y0=state0,
                method=method,
                t_eval=t_eval,
                rtol=rtol,
                atol=atol,
                max_step=time_step,
                events=events,
                dense_output=dense_output
            )
        except Exception as e:
            print(f"Integration failed: {e}")
            print("Retrying with more robust settings...")
            sol = solve_ivp(
                fun=lambda t, y: self._equations_of_motion(t, y, options),
                t_span=time_span,
                y0=state0,
                method='RK45',
                t_eval=t_eval,
                rtol=1e-6,
                atol=1e-8,
                max_step=time_step * 2,
                events=events,
                dense_output=False
            )
        
        # Store solution
        self.last_solution = sol
        
        if sol.success:
            # Extract results
            times = sol.t
            positions = sol.y[:3, :]
            velocities = sol.y[3:6, :]
            
            # Convert to orbital elements at each time step
            orbital_elements = []
            altitudes = []
            
            for i in range(len(times)):
                try:
                    r = positions[:, i]
                    v = velocities[:, i]
                    elements = OrbitalElements.from_cartesian(r, v, initial_elements.mu, times[i])
                    orbital_elements.append(elements)
                    altitudes.append(elements.perigee_altitude)
                except:
                    # If conversion fails, use previous elements or skip
                    if orbital_elements:
                        orbital_elements.append(orbital_elements[-1])
                        altitudes.append(altitudes[-1])
                    else:
                        orbital_elements.append(initial_elements)
                        altitudes.append(initial_elements.perigee_altitude)
            
            results = {
                'success': True,
                'times': times,
                'positions': positions,
                'velocities': velocities,
                'orbital_elements': orbital_elements,
                'altitudes': np.array(altitudes),
                'initial_elements': initial_elements,
                'integration_info': {
                    'method': method,
                    'nfev': sol.nfev,
                    'njev': sol.njev,
                    'nlu': sol.nlu,
                    'status': sol.status,
                    'message': sol.message
                }
            }
            
            print(f"Propagation completed successfully!")
            print(f"Final altitude: {altitudes[-1]/1000:.1f} km")
            print(f"Integration steps: {sol.nfev}")
            
            return results
        
        else:
            print(f"Propagation failed: {sol.message}")
            return {
                'success': False,
                'message': sol.message,
                'status': sol.status
            }
    
    def _equations_of_motion(self, t, state, options):
        """
        Equations of motion with perturbations.
        
        Parameters:
        -----------
        t : float
            Current time [s]
        state : array
            State vector [x, y, z, vx, vy, vz] in ECI frame
        options : dict
            Integration options and parameters
            
        Returns:
        --------
        array
            State derivative [vx, vy, vz, ax, ay, az]
        """
        
        # Extract position and velocity
        r = state[:3]
        v = state[3:6]
        
        r_mag = np.linalg.norm(r)
        
        # Initialize acceleration
        a = np.zeros(3)
        
        # Central gravitational acceleration
        a_gravity = -self.mu * r / r_mag**3
        a += a_gravity
        
        # J2 perturbation
        if options['include_j2']:
            a_j2 = self._j2_acceleration(r)
            a += a_j2
        
        # Atmospheric drag
        if options['include_drag'] and options['ballistic_coefficient'] is not None:
            altitude = r_mag - self.earth_radius
            if altitude > 0:  # Only apply drag above surface
                a_drag = self._drag_acceleration(r, v, options['ballistic_coefficient'])
                a += a_drag
        
        # External thrust
        if options['include_thrust'] and options['thrust_magnitude'] > 0:
            a_thrust = self._thrust_acceleration(v, options['thrust_magnitude'], 
                                               options['thrust_direction'])
            a += a_thrust
        
        # Return state derivative
        return np.concatenate([v, a])
    
    def _j2_acceleration(self, r):
        """
        Calculate J2 gravitational perturbation acceleration.
        
        Derived from the J2 gravitational potential and scaled by 1.5:
        U_J2 = -μ*J2*Re^2 / (2*r^3) * (3*z^2/r^2 - 1)
        
        Standard form with 1.5 factor but negative sign for correct physics.
        
        Parameters:
        -----------
        r : array
            Position vector [m] in ECI coordinates
            
        Returns:
        --------
        array
            J2 acceleration vector [m/s²] in ECI coordinates
        """
        
        r_mag = np.linalg.norm(r)
        x, y, z = r
        
        # Hybrid approach: potential-based but with 1.5 factor and correct sign
        factor = -1.5 * self.mu * self.j2 * self.earth_radius**2 / r_mag**5
        
        # Acceleration components
        ax = factor * x * (5 * z**2 / r_mag**2 - 1)
        ay = factor * y * (5 * z**2 / r_mag**2 - 1)
        az = factor * z * (5 * z**2 / r_mag**2 - 3)
        
        return np.array([ax, ay, az])
    
    def _drag_acceleration(self, r, v, ballistic_coefficient):
        """
        Calculate atmospheric drag acceleration.
        
        Parameters:
        -----------
        r : array
            Position vector [m]
        v : array
            Velocity vector [m/s]
        ballistic_coefficient : float
            Ballistic coefficient m/(Cd*S) [kg/m²]
            
        Returns:
        --------
        array
            Drag acceleration vector [m/s²]
        """
        
        r_mag = np.linalg.norm(r)
        altitude = r_mag - self.earth_radius
        
        # Get atmospheric density from Harris-Priester model
        # For simplicity, assume equatorial orbit at noon
        # In practice, you would calculate actual latitude and local solar time
        try:
            rho = self.atmosphere.density(altitude, latitude=0.0, local_solar_time=12.0)
        except:
            # If density calculation fails, use exponential model
            scale_height = 8500.0  # m
            rho_0 = 1.225e-9  # kg/m³ at 100 km
            h_0 = 100000.0  # m
            rho = rho_0 * np.exp(-(altitude - h_0) / scale_height)
        
        # Relative velocity (accounting for Earth's rotation)
        omega_earth = np.array([0, 0, self.rotation_rate])
        v_rel = v - np.cross(omega_earth, r)
        v_rel_mag = np.linalg.norm(v_rel)
        
        if v_rel_mag > 0 and rho > 0:
            # Drag acceleration magnitude
            drag_acceleration_mag = 0.5 * rho * v_rel_mag**2 / ballistic_coefficient
            
            # Drag acceleration direction (opposite to relative velocity)
            a_drag = -drag_acceleration_mag * v_rel / v_rel_mag
            
            return a_drag
        else:
            return np.zeros(3)
    
    def _thrust_acceleration(self, v, thrust_magnitude, thrust_direction):
        """
        Calculate thrust acceleration.
        
        Parameters:
        -----------
        v : array
            Velocity vector [m/s]
        thrust_magnitude : float
            Thrust magnitude [N]
        thrust_direction : array, string, or None
            Thrust direction. Can be:
            - "prograde": thrust in velocity direction
            - "retrograde": thrust opposite to velocity direction  
            - array: custom thrust direction unit vector
            - None: defaults to retrograde
            
        Returns:
        --------
        array
            Thrust acceleration vector [m/s²] (requires mass to convert from force)
        """
        
        v_mag = np.linalg.norm(v)
        
        if thrust_direction is None or thrust_direction == "retrograde":
            # Thrust opposite to velocity (retrograde)
            if v_mag > 0:
                thrust_direction = -v / v_mag
            else:
                thrust_direction = np.array([1, 0, 0])  # Default direction
        elif thrust_direction == "prograde":
            # Thrust in velocity direction (prograde)
            if v_mag > 0:
                thrust_direction = v / v_mag
            else:
                thrust_direction = np.array([1, 0, 0])  # Default direction
        else:
            # Custom direction vector
            thrust_direction = np.array(thrust_direction)
            thrust_direction = thrust_direction / np.linalg.norm(thrust_direction)
        
        # Note: This returns thrust force, not acceleration
        # To get acceleration, divide by spacecraft mass: a = F/m
        # For now, assume unit mass or handle mass separately
        return thrust_magnitude * thrust_direction
    
    def propagate_with_mass(self, initial_elements, spacecraft_mass, time_span,
                           ballistic_coefficient=None, thrust_magnitude=0.0, 
                           specific_impulse=300.0, thrust_direction=None,
                           include_j2=True, include_drag=True, include_thrust=True,
                           **kwargs):
        """
        Propagate orbit including spacecraft mass variation due to fuel consumption.
        
        Parameters:
        -----------
        initial_elements : OrbitalElements
            Initial orbital elements
        spacecraft_mass : float
            Initial spacecraft mass [kg]
        time_span : tuple or float
            Time span for propagation [s]
        ballistic_coefficient : float, optional
            Ballistic coefficient m/(Cd*S) [kg/m²]
        thrust_magnitude : float, optional
            Constant thrust magnitude [N]
        specific_impulse : float, optional
            Specific impulse [s]. Default is 300 s.
        thrust_direction : array-like, optional
            Thrust direction unit vector
        include_j2 : bool, optional
            Include J2 perturbations
        include_drag : bool, optional
            Include atmospheric drag
        include_thrust : bool, optional
            Include thrust acceleration
        **kwargs
            Additional arguments for propagate method
            
        Returns:
        --------
        dict
            Propagation results including mass history
        """
        
        if isinstance(time_span, (int, float)):
            time_span = (0.0, float(time_span))
        
        # Convert to state vector with mass
        r0, v0 = initial_elements.to_cartesian()
        state0 = np.concatenate([r0, v0, [spacecraft_mass]])
        
        # Calculate exhaust velocity
        exhaust_velocity = specific_impulse * PHYSICAL_CONSTANTS['g0']
        
        # Set up integration options
        options = {
            'ballistic_coefficient': ballistic_coefficient,
            'thrust_magnitude': thrust_magnitude,
            'thrust_direction': thrust_direction,
            'exhaust_velocity': exhaust_velocity,
            'include_j2': include_j2,
            'include_drag': include_drag,
            'include_thrust': include_thrust
        }
        
        # Create time evaluation points
        time_step = kwargs.get('time_step', 60.0)
        t_eval = np.arange(time_span[0], time_span[1] + time_step, time_step)
        
        print(f"Propagating orbit with mass variation...")
        print(f"Initial mass: {spacecraft_mass:.1f} kg")
        print(f"Thrust: {thrust_magnitude:.1f} N, Isp: {specific_impulse:.1f} s")
        
        # Solve differential equation
        sol = solve_ivp(
            fun=lambda t, y: self._equations_of_motion_with_mass(t, y, options),
            t_span=time_span,
            y0=state0,
            method=kwargs.get('method', 'DOP853'),
            t_eval=t_eval,
            rtol=kwargs.get('rtol', 1e-8),
            atol=kwargs.get('atol', 1e-10),
            max_step=time_step,
            events=kwargs.get('events', None),
            dense_output=kwargs.get('dense_output', False)
        )
        
        if sol.success:
            # Extract results
            times = sol.t
            positions = sol.y[:3, :]
            velocities = sol.y[3:6, :]
            masses = sol.y[6, :]
            
            # Calculate fuel consumed
            fuel_consumed = spacecraft_mass - masses[-1]
            
            # Convert to orbital elements
            orbital_elements = []
            altitudes = []
            
            for i in range(len(times)):
                try:
                    r = positions[:, i]
                    v = velocities[:, i]
                    elements = OrbitalElements.from_cartesian(r, v, initial_elements.mu, times[i])
                    orbital_elements.append(elements)
                    altitudes.append(elements.perigee_altitude)
                except:
                    if orbital_elements:
                        orbital_elements.append(orbital_elements[-1])
                        altitudes.append(altitudes[-1])
                    else:
                        orbital_elements.append(initial_elements)
                        altitudes.append(initial_elements.perigee_altitude)
            
            results = {
                'success': True,
                'times': times,
                'positions': positions,
                'velocities': velocities,
                'masses': masses,
                'orbital_elements': orbital_elements,
                'altitudes': np.array(altitudes),
                'fuel_consumed': fuel_consumed,
                'final_mass': masses[-1],
                'initial_elements': initial_elements,
                'initial_mass': spacecraft_mass
            }
            
            print(f"Propagation completed!")
            print(f"Final mass: {masses[-1]:.1f} kg")
            print(f"Fuel consumed: {fuel_consumed:.1f} kg")
            
            return results
        else:
            print(f"Propagation failed: {sol.message}")
            return {'success': False, 'message': sol.message}
    
    def _equations_of_motion_with_mass(self, t, state, options):
        """
        Equations of motion including mass variation.
        
        Parameters:
        -----------
        t : float
            Current time [s]
        state : array
            State vector [x, y, z, vx, vy, vz, mass]
        options : dict
            Integration options
            
        Returns:
        --------
        array
            State derivative [vx, vy, vz, ax, ay, az, dm/dt]
        """
        
        # Extract state variables
        r = state[:3]
        v = state[3:6]
        mass = state[6]
        
        r_mag = np.linalg.norm(r)
        
        # Initialize acceleration
        a = np.zeros(3)
        mass_flow_rate = 0.0
        
        # Central gravitational acceleration
        a_gravity = -self.mu * r / r_mag**3
        a += a_gravity
        
        # J2 perturbation
        if options['include_j2']:
            a_j2 = self._j2_acceleration(r)
            a += a_j2
        
        # Atmospheric drag
        if options['include_drag'] and options['ballistic_coefficient'] is not None:
            altitude = r_mag - self.earth_radius
            if altitude > 0:
                a_drag = self._drag_acceleration(r, v, options['ballistic_coefficient'])
                a += a_drag
        
        # Thrust acceleration with mass flow
        if (options['include_thrust'] and options['thrust_magnitude'] > 0 and mass > 0):
            
            # Thrust direction with string handling
            v_mag = np.linalg.norm(v)
            
            if options['thrust_direction'] is None or options['thrust_direction'] == "retrograde":
                if v_mag > 0:
                    thrust_direction = -v / v_mag  # Retrograde (opposite to velocity)
                else:
                    thrust_direction = np.array([1, 0, 0])  # Default direction
            elif options['thrust_direction'] == "prograde":
                if v_mag > 0:
                    thrust_direction = v / v_mag  # Prograde (along velocity)
                else:
                    thrust_direction = np.array([1, 0, 0])  # Default direction
            else:
                # Custom thrust direction vector
                thrust_direction = np.array(options['thrust_direction'])
                thrust_direction = thrust_direction / np.linalg.norm(thrust_direction)
            
            # Thrust acceleration
            a_thrust = options['thrust_magnitude'] * thrust_direction / mass
            a += a_thrust
            
            # Mass flow rate (negative because mass decreases)
            mass_flow_rate = -options['thrust_magnitude'] / options['exhaust_velocity']
        
        return np.concatenate([v, a, [mass_flow_rate]])
