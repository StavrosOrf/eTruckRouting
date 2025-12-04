"""
Simple test to verify action mask alignment between env.mask_fn() and GNN state space.
"""

import random
import numpy as np
from truck_env.models.event_driven_env import EventDrivenTruckEnv
from truck_env.state.gnn_state_space import GNNStateSpace
from truck_env.utils.utils import load_config


def test_mask_alignment():
    """Test that env.mask_fn() produces the same mask as GNN state space."""
    
    # Load config
    config_file = "truck_env/config_files/config.yaml"
    config = load_config(config_file)
    config['environment'].update({
        "num_trucks": 2,
        "num_stops": 2,
        "max_time": 100.0,
    })
    
    # Create environment
    env = EventDrivenTruckEnv(
        config=config,
        run_id="test_mask",
        verbose=False,
        enable_plotting=False,
    )
    
    # Create GNN state space
    gnn_state = GNNStateSpace(
        num_trucks=2,
        num_stops=2,
        max_time=100.0,
        num_charging_nodes=len(env.charging_nodes),
        device="cpu",
        verbose=False,
    )
    
    print("Testing action mask alignment...")
    print("-" * 60)
    
    # Reset environment
    obs, info = env.reset(seed=42)
    
    # Test initial state
    print(f"\nInitial state - Active truck: {env.active_truck_id}")
    
    # Get masks
    mask_env = env.mask_fn()
    gnn_graph = gnn_state.get_state_GNN(env)
    mask_gnn = gnn_graph.feasible_action_mask.cpu().numpy()
    
    print(f"Mask from env.mask_fn():  shape={mask_env.shape}, sum={mask_env.sum()}")
    print(f"Mask from GNN state:      shape={mask_gnn.shape}, sum={mask_gnn.sum()}")
    
    # Compare
    if np.array_equal(mask_env, mask_gnn):
        print("✓ Masks are identical!")
    else:
        print("✗ Masks differ!")
        diff_idx = np.where(mask_env != mask_gnn)[0]
        print(f"  Differing indices: {diff_idx}")
        print(f"  env.mask_fn()[diff]:  {mask_env[diff_idx]}")
        print(f"  GNN mask[diff]:       {mask_gnn[diff_idx]}")
        
    return np.array_equal(mask_env, mask_gnn)


if __name__ == "__main__":
    success = test_mask_alignment()
    print("\n" + "="*60)
    if success:
        print("TEST PASSED: Action masks are aligned!")
    else:
        print("TEST FAILED: Action masks differ!")
    print("="*60)
