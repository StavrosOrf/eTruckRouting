"""
Test Gurobi optimal solver with a small scenario.
"""

import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from truck_env.models.event_driven_env import EventDrivenTruckEnv
from truck_env.utils.utils import load_config
from truck_env.optimization.gurobi_solver import GurobiOptimalPlanner

# Small test configuration
CONFIG_FILE = "truck_env/config_files/config.yaml"
NUM_TRUCKS = 2
NUM_STOPS = 3
SEED = 42
GUROBI_TIME_LIMIT = 300  # 5 minutes

def main():
    print("\n" + "="*80)
    print(" "*20 + "TESTING GUROBI OPTIMAL SOLVER")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Trucks: {NUM_TRUCKS}")
    print(f"  Stops per truck: {NUM_STOPS}")
    print(f"  Seed: {SEED}")
    print(f"  Time limit: {GUROBI_TIME_LIMIT}s")
    
    # Load configuration
    config = load_config(CONFIG_FILE)
    config["environment"]["num_trucks"] = NUM_TRUCKS
    config["environment"]["num_stops"] = NUM_STOPS
    
    # Create environment
    env = EventDrivenTruckEnv(config=copy.deepcopy(config), verbose=False, enable_plotting=False)
    obs, info = env.reset(seed=SEED)
    
    print(f"\nEnvironment created:")
    print(f"  Charging nodes: {len(env.charging_nodes)}")
    print(f"  Total graph nodes: {env.transport_graph.num_nodes}")
    
    # Create solver
    solver = GurobiOptimalPlanner(env, config, time_limit=GUROBI_TIME_LIMIT, verbose=True)
    
    # Build model
    solver.build_model()
    
    # Solve
    success = solver.solve()
    
    if success:
        print("\n" + "="*80)
        print(" "*30 + "SUCCESS!")
        print("="*80)
        
        # Print solution
        solver.print_solution_summary()
        
        # Validate
        is_valid, message = solver.validate_solution(env)
        print(f"\nValidation: {message}")
        
        if is_valid:
            print("\n✓ Solution is valid and optimal!")
        else:
            print("\n✗ Solution validation failed")
    else:
        print("\n" + "="*80)
        print(" "*30 + "FAILED")
        print("="*80)
        print("\n✗ Failed to find optimal solution")
    
    env.close()

if __name__ == "__main__":
    main()
