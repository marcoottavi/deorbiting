"""
Orbital propagation simulation framework.

Edit the parameters below to configure your simulation, then run the script.

IMPORTANT NOTE ABOUT TRAJECTORY PLOTS:
The orbital trajectory may appear elliptical in the X-Y plane plot even for 
nearly circular orbits. This is a geometric projection effect caused by the 
orbital inclination (orbital plane tilted relative to X-Y plane), NOT an 
indication of orbital eccentricity. The actual orbital shape is determined 
by the eccentricity parameter and the radius variation shown in the plot.

Author: PhD Design Research
Date: June 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.offline as pyo
from datetime import datetime
import os

from orbital_elements import OrbitalElements
from orbital_propagator import OrbitalPropagator
from constants import EARTH_CONSTANTS, CONVERSIONS

# ============================================================================
# SIMULATION PARAMETERS - EDIT THESE VALUES TO CONFIGURE YOUR SIMULATION
# ============================================================================
os.system('clear')

# 📍 INITIAL ORBITAL ELEMENTS
altitude = 400.0          # km above Earth's surface
eccentricity = 0.001      # orbital eccentricity (0 = circular, <1 = elliptical)
inclination = 51.6        # degrees (0 = equatorial, 90 = polar)
raan = 45.0               # degrees (Right Ascension of Ascending Node)
arg_perigee = 45.0        # degrees (Argument of Perigee)
true_anomaly = 0.0        # degrees (True Anomaly)

# ⏱️ SIMULATION PARAMETERS
duration_days = 3      # simulation duration in days
time_step_minutes = 5.0  # integration time step in minutes

# 🌍 PERTURBATIONS (True/False)
include_j2 = True        # include J2 gravitational perturbations
include_drag = True     # include atmospheric drag
include_thrust = True   # include constant thrust

# 🚀 SPACECRAFT PARAMETERS
spacecraft_mass = 500.0         # kg
ballistic_coefficient = 100.0   # kg/m² (mass/drag_area)
solar_flux = 150.0              # F10.7 solar flux index

# 🔥 THRUST PARAMETERS (only used if include_thrust = True)
thrust_magnitude = 0.01          # N (Newtons)
specific_impulse = 5000.0       # s (seconds)
thrust_direction = "retrograde"   # "prograde" or "retrograde"

# 📊 OUTPUT OPTIONS
generate_3d_plot = True         # generate interactive 3D plot with Plotly

# 🎯 EVENT DETECTION (only used if include_thrust = True)
enable_altitude_event = True   # stop simulation when altitude threshold reached
altitude_threshold = 100.0      # km

# ============================================================================


def setup_simulation_parameters():
    """Convert user parameters to simulation format."""
    
    # Convert altitude to semi-major axis
    semi_major_axis = altitude * 1000 + EARTH_CONSTANTS['radius']
    
    # Convert time parameters
    time_span = duration_days * 86400
    time_step = time_step_minutes * 60
    
    return {
        'semi_major_axis': semi_major_axis,
        'eccentricity': eccentricity,
        'inclination': inclination,
        'raan': raan,
        'arg_perigee': arg_perigee,
        'true_anomaly': true_anomaly,
        'time_span': time_span,
        'time_step': time_step,
        'include_j2': include_j2,
        'include_drag': include_drag,
        'include_thrust': include_thrust,
        'spacecraft_mass': spacecraft_mass,
        'ballistic_coefficient': ballistic_coefficient,
        'solar_flux': solar_flux,
        'thrust_magnitude': thrust_magnitude,
        'specific_impulse': specific_impulse,
        'thrust_direction': thrust_direction,
        'enable_altitude_event': enable_altitude_event,
        'altitude_threshold': altitude_threshold,
        'generate_3d_plot': generate_3d_plot
    }


def run_simulation(params):
    """Run the orbital simulation with given parameters."""
    
    print("\n🛰️  STARTING SIMULATION")
    print("=" * 50)
    
    # Create initial orbital elements
    initial_orbit = OrbitalElements(
        semi_major_axis=params['semi_major_axis'],
        eccentricity=params['eccentricity'],
        inclination=params['inclination'],
        raan=params['raan'],
        arg_perigee=params['arg_perigee'],
        true_anomaly=params['true_anomaly']
    )
    
    # Display initial conditions
    print(f"Initial orbit:")
    print(f"  Altitude: {(initial_orbit.a - EARTH_CONSTANTS['radius'])/1000:.1f} km")
    print(f"  Eccentricity: {initial_orbit.e:.6f}")
    print(f"  Inclination: {initial_orbit.i * CONVERSIONS['rad_to_deg']:.1f}°")
    print(f"  Period: {initial_orbit.period/3600:.2f} hours")
    
    print(f"\nSimulation parameters:")
    print(f"  Duration: {params['time_span']/86400:.2f} days")
    print(f"  Time step: {params['time_step']/60:.1f} minutes")
    print(f"  J2 perturbations: {'Yes' if params['include_j2'] else 'No'}")
    print(f"  Atmospheric drag: {'Yes' if params['include_drag'] else 'No'}")
    print(f"  Thrust: {'Yes' if params['include_thrust'] else 'No'}")
    
    # Create propagator
    propagator = OrbitalPropagator()
    
    # Setup atmosphere if needed
    if params['include_drag']:
        propagator.atmosphere.set_solar_conditions(f107_flux=params['solar_flux'])
        print(f"  Solar flux (F10.7): {params['solar_flux']}")
    
    if params['include_thrust']:
        print(f"  Spacecraft mass: {params['spacecraft_mass']} kg")
        print(f"  Thrust: {params['thrust_magnitude']} N ({params['thrust_direction']})")
        print(f"  Specific impulse: {params['specific_impulse']} s")
    
    # Setup events
    events = None
    if params['enable_altitude_event']:
        def altitude_event(t, y):
            r = y[:3]
            r_mag = np.linalg.norm(r)
            altitude = r_mag - EARTH_CONSTANTS['radius']
            return altitude - params['altitude_threshold'] * 1000
        
        altitude_event.terminal = True
        altitude_event.direction = -1
        events = altitude_event
        print(f"  Altitude threshold: {params['altitude_threshold']} km")
    
    print("\nRunning simulation...")
    
    try:
        if params['include_thrust']:
            # Pass thrust direction string directly to propagator
            thrust_dir = params['thrust_direction']
            
            # Run simulation with mass variation
            results = propagator.propagate_with_mass(
                initial_elements=initial_orbit,
                spacecraft_mass=params['spacecraft_mass'],
                time_span=params['time_span'],
                ballistic_coefficient=params['ballistic_coefficient'],
                thrust_magnitude=params['thrust_magnitude'],
                specific_impulse=params['specific_impulse'],
                thrust_direction=thrust_dir,
                include_j2=params['include_j2'],
                include_drag=params['include_drag'],
                include_thrust=params['include_thrust'],
                time_step=params['time_step'],
                events=events
            )
        else:
            # Run simulation without mass variation
            results = propagator.propagate(
                initial_elements=initial_orbit,
                time_span=params['time_span'],
                ballistic_coefficient=params['ballistic_coefficient'],
                include_j2=params['include_j2'],
                include_drag=params['include_drag'],
                include_thrust=False,
                time_step=params['time_step'],
                events=events
            )
        
        if results['success']:
            print_results(results, params)
            plot_results(results, params)
        else:
            print("❌ Simulation failed!")
            if 'message' in results:
                print(f"Error: {results['message']}")
            return False
            
    except Exception as e:
        print(f"❌ Simulation error: {e}")
        return False
    
    return True


def print_results(results, params):
    """Print simulation results."""
    
    print("\n🎯 SIMULATION COMPLETED SUCCESSFULLY")
    print("=" * 50)
    
    # Basic results
    final_altitude = results['altitudes'][-1] / 1000
    initial_altitude = results['altitudes'][0] / 1000
    altitude_change = final_altitude - initial_altitude
    simulation_time = results['times'][-1] / 86400
    
    print(f"Simulation time: {simulation_time:.2f} days")
    print(f"Initial altitude: {initial_altitude:.1f} km")
    print(f"Final altitude: {final_altitude:.1f} km")
    print(f"Altitude change: {altitude_change:+.3f} km")
    
    # Orbital element changes
    initial_elements = results['orbital_elements'][0]
    final_elements = results['orbital_elements'][-1]
    
    raan_change = (final_elements.raan - initial_elements.raan) * CONVERSIONS['rad_to_deg']
    argp_change = (final_elements.w - initial_elements.w) * CONVERSIONS['rad_to_deg']
    
    print(f"\nOrbital element changes:")
    print(f"  RAAN drift: {raan_change:+.3f}°")
    print(f"  Arg. perigee drift: {argp_change:+.3f}°")
    print(f"  Eccentricity change: {final_elements.e - initial_elements.e:+.6f}")
    
    # Thrust-specific results
    if params['include_thrust'] and 'fuel_consumed' in results:
        fuel_consumed = results['fuel_consumed']
        fuel_fraction = fuel_consumed / params['spacecraft_mass'] * 100
        print(f"\nPropulsion results:")
        print(f"  Fuel consumed: {fuel_consumed:.2f} kg ({fuel_fraction:.1f}% of initial mass)")
        print(f"  Final mass: {results['final_mass']:.1f} kg")
    
    # Event results
    if params['enable_altitude_event'] and simulation_time < params['time_span']/86400:
        print(f"\n🎯 Altitude threshold ({params['altitude_threshold']} km) reached!")
        print(f"  Time to threshold: {simulation_time:.2f} days")


def plot_results(results, params):
    """Generate and display plots."""
    
    # Create output directory
    output_dir = "simulation_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create timestamp for file names
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Prepare parameter string for titles
    param_str = f"Alt: {params['semi_major_axis']/1000 - EARTH_CONSTANTS['radius']/1000:.0f}km, " + \
                f"e: {params['eccentricity']:.3f}, " + \
                f"i: {params['inclination']:.1f}°"
    
    effects_str = []
    if params['include_j2']:
        effects_str.append("J2")
    if params['include_drag']:
        effects_str.append("Drag")
    if params['include_thrust']:
        effects_str.append(f"Thrust({params['thrust_magnitude']:.1f}N)")
    
    effects_summary = ", ".join(effects_str) if effects_str else "Keplerian"
    
    # Create plots with high resolution
    fig, axes = plt.subplots(2, 3, figsize=(19.2, 10.8))  # 1920x1080 at 100 DPI
    
    times_days = results['times'] / 86400
    
    # Plot 1: Altitude
    axes[0, 0].plot(times_days, results['altitudes'] / 1000, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Time (days)')
    axes[0, 0].set_ylabel('Altitude (km)')
    axes[0, 0].set_title(f'Altitude Evolution\n{param_str}\nEffects: {effects_summary}')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Semi-major axis
    sma_km = [elem.a/1000 for elem in results['orbital_elements']]
    axes[0, 1].plot(times_days, sma_km, 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Time (days)')
    axes[0, 1].set_ylabel('Semi-major Axis (km)')
    axes[0, 1].set_title(f'Orbital Energy (Semi-major Axis)\n{param_str}\nEffects: {effects_summary}')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Eccentricity
    eccentricity = [elem.e for elem in results['orbital_elements']]
    axes[0, 2].plot(times_days, eccentricity, 'r-', linewidth=2)
    axes[0, 2].set_xlabel('Time (days)')
    axes[0, 2].set_ylabel('Eccentricity')
    axes[0, 2].set_title(f'Orbital Shape (Eccentricity)\n{param_str}\nEffects: {effects_summary}')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: RAAN
    raan_deg = [elem.raan * CONVERSIONS['rad_to_deg'] for elem in results['orbital_elements']]
    axes[1, 0].plot(times_days, raan_deg, 'm-', linewidth=2)
    axes[1, 0].set_xlabel('Time (days)')
    axes[1, 0].set_ylabel('RAAN (degrees)')
    axes[1, 0].set_title(f'RAAN Evolution\n{param_str}\nEffects: {effects_summary}')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Argument of Perigee
    argp_deg = [elem.w * CONVERSIONS['rad_to_deg'] for elem in results['orbital_elements']]
    axes[1, 1].plot(times_days, argp_deg, 'c-', linewidth=2)
    axes[1, 1].set_xlabel('Time (days)')
    axes[1, 1].set_ylabel('Argument of Perigee (degrees)')
    axes[1, 1].set_title(f'Argument of Perigee Evolution\n{param_str}\nEffects: {effects_summary}')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: 3D trajectory
    positions = results['positions'] / 1000  # Convert to km
    
    # Calculate orbital radii to check if orbit is actually circular
    radii = np.sqrt(positions[0,:]**2 + positions[1,:]**2 + positions[2,:]**2)
    
    axes[1, 2].plot(positions[0, :], positions[1, :], 'b-', alpha=0.7, linewidth=1)
    
    # Add Earth
    theta = np.linspace(0, 2*np.pi, 100)
    earth_radius_km = EARTH_CONSTANTS['radius'] / 1000
    axes[1, 2].plot(earth_radius_km * np.cos(theta), 
                   earth_radius_km * np.sin(theta), 
                   'brown', linewidth=2, label='Earth')
    
    axes[1, 2].set_xlabel('X (km)')
    axes[1, 2].set_ylabel('Y (km)')
    axes[1, 2].set_title(f'Orbital Trajectory (X-Y Projection)\n{param_str}\nEffects: {effects_summary}\n*Elliptical appearance due to orbital inclination*')
    axes[1, 2].axis('equal')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].legend()
    
    # Add text showing actual orbital parameters
    radius_range = (np.max(radii) - np.min(radii))
    axes[1, 2].text(0.02, 0.98, f'Actual radius variation: {radius_range:.1f} km', 
                   transform=axes[1, 2].transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Create comprehensive title
    duration_str = f"{params['time_span']/86400:.1f} days"
    if params['include_thrust']:
        title = f'Orbital Simulation Results - {duration_str} - {effects_summary} - Mass: {params["spacecraft_mass"]:.0f}kg'
    else:
        title = f'Orbital Simulation Results - {duration_str} - {effects_summary}'
    
    plt.suptitle(title, fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for suptitle
    
    plot_filename = os.path.join(output_dir, f'simulation_results_{timestamp}.png')
    plt.savefig(plot_filename, dpi=100, bbox_inches='tight')  # 100 DPI for 1920x1080
    
    print(f"\n📊 Plots saved to {plot_filename}")
    
    # Create interactive 3D plot with Plotly
    if params.get('generate_3d_plot', True):
        plot_3d_trajectory_plotly(results, params, output_dir, timestamp)
    
    # Try to show matplotlib plot if possible
    try:
        plt.show()
    except:
        pass


def plot_3d_trajectory_plotly(results, params, output_dir, timestamp):
    """Create an interactive 3D trajectory plot using Plotly."""
    
    positions = results['positions'] / 1000  # Convert to km
    
    # Calculate orbital radii
    radii = np.sqrt(positions[0,:]**2 + positions[1,:]**2 + positions[2,:]**2)
    
    # Create Earth sphere
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    earth_radius_km = EARTH_CONSTANTS['radius'] / 1000
    
    x_earth = earth_radius_km * np.outer(np.cos(u), np.sin(v))
    y_earth = earth_radius_km * np.outer(np.sin(u), np.sin(v))
    z_earth = earth_radius_km * np.outer(np.ones(np.size(u)), np.cos(v))
    
    # Create the figure
    fig = go.Figure()
    
    # Add Earth surface
    fig.add_trace(go.Surface(
        x=x_earth, y=y_earth, z=z_earth,
        colorscale='Earth',
        showscale=False,
        name='Earth',
        opacity=0.8
    ))
    
    # Add orbital trajectory
    fig.add_trace(go.Scatter3d(
        x=positions[0, :],
        y=positions[1, :],
        z=positions[2, :],
        mode='lines',
        line=dict(color='red', width=4),
        name='Orbital Trajectory'
    ))
    
    # Add start and end points
    fig.add_trace(go.Scatter3d(
        x=[positions[0, 0]],
        y=[positions[1, 0]],
        z=[positions[2, 0]],
        mode='markers',
        marker=dict(size=8, color='green'),
        name='Start'
    ))
    
    fig.add_trace(go.Scatter3d(
        x=[positions[0, -1]],
        y=[positions[1, -1]],
        z=[positions[2, -1]],
        mode='markers',
        marker=dict(size=8, color='orange'),
        name='End'
    ))
    
    # Update layout
    fig.update_layout(
        title=f'Interactive 3D Orbital Trajectory<br>' +
              f'Altitude: {(np.mean(radii) - earth_radius_km):.1f} km, ' +
              f'Inclination: {params["inclination"]:.1f}°, ' +
              f'Eccentricity: {params["eccentricity"]:.6f}<br>' +
              f'Duration: {params["time_span"]/86400:.1f} days, ' +
              f'Effects: {("J2, " if params["include_j2"] else "") + ("Drag, " if params["include_drag"] else "") + (f"Thrust({params["thrust_magnitude"]:.1f}N)" if params["include_thrust"] else "")}'.rstrip(', ') or 'Keplerian',
        scene=dict(
            xaxis_title='X (km)',
            yaxis_title='Y (km)',
            zaxis_title='Z (km)',
            aspectmode='cube',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=1920,  # High resolution
        height=1080
    )
    
    # Save as HTML file
    html_filename = os.path.join(output_dir, f'3d_trajectory_{timestamp}.html')
    pyo.plot(fig, filename=html_filename, auto_open=False)
    
    print(f"🌍 Interactive 3D plot saved to {html_filename}")
    
    return fig


def main():
    """Main entry point."""
    
    print("🛰️  ORBITAL SIMULATION")
    print("=" * 50)
    print("Running simulation with parameters defined in the script...")
    print()
    
    try:
        # Get simulation parameters from script configuration
        params = setup_simulation_parameters()
        
        # Run simulation
        success = run_simulation(params)
        
        if success:
            print("\n✅ Simulation completed successfully!")
        else:
            print("\n❌ Simulation failed!")
            
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted. Goodbye! 👋")
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()
