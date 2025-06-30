"""Quick test of the 3D plotting functionality"""

from main import setup_simulation_parameters, run_simulation

# Test with default parameters but shorter simulation
params = setup_simulation_parameters()
params['duration_days'] = 0.1  # 2.4 hours only
params['include_j2'] = False   # No perturbations for faster test
params['include_drag'] = False

print("Testing 3D plotting with quick simulation...")
success = run_simulation(params)
