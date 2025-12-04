"""
Verification script to compare action masks between env.mask_fn() and GNN state space.

This script runs multiple episodes and at each decision point compares:
1. The feasibility mask from env.mask_fn() 
2. The feasibility mask from GNN state space (data.feasible_action_mask)

It reports any discrepancies found.
"""

import random
import numpy as np
from truck_env.models.event_driven_env import EventDrivenTruckEnv
from truck_env.state.gnn_state_space import GNNStateSpace
from truck_env.utils.utils import load_config


def verify_action_masks(num_episodes: int = 10, seed: int = 42):
    """
    Run multiple episodes and verify that action masks match between methods.
    
    Args:
        num_episodes: Number of episodes to test
        seed: Random seed for reproducibility
    """
    config_file = "truck_env/config_files/config.yaml"
    config_overwrite = {
        "num_trucks": 5,
        "num_stops": 3,
        "max_time": 200.0,
    }

    config = load_config(config_file)
    config['environment'].update(config_overwrite)
    
    # Create environment
    env = EventDrivenTruckEnv(
        config=config,
        run_id="mask_verification",
        verbose=False,
        enable_plotting=False,
    )

    # Create GNN state space
    gnn_state_space = GNNStateSpace(
        num_trucks=config['environment']["num_trucks"],
        num_stops=config['environment']["num_stops"],
        max_time=config['environment']["max_time"],
        num_charging_nodes=len(env.charging_nodes),
        device="cpu",
        verbose=False,
    )

    total_checks = 0
    total_mismatches = 0
    mismatch_details = []

    print("="*80)
    print("ACTION MASK VERIFICATION")
    print("="*80)
    print(f"Testing {num_episodes} episodes with seed {seed}")
    print()

    for episode in range(num_episodes):
        episode_seed = seed + episode
        obs, info = env.reset(seed=episode_seed)
        env.action_space.seed(episode_seed)
        random.seed(episode_seed)
        
        episode_checks = 0
        episode_mismatches = 0
        
        while True:
            # Get mask from env.mask_fn()
            mask_from_env = env.mask_fn()
            
            # Get GNN state and extract mask
            gnn_graph = gnn_state_space.get_state_GNN(env)
            mask_from_gnn = gnn_graph.feasible_action_mask.cpu().numpy()
            
            # Compare masks
            episode_checks += 1
            total_checks += 1
            
            if not np.array_equal(mask_from_env, mask_from_gnn):
                episode_mismatches += 1
                total_mismatches += 1
                
                # Find which actions differ
                diff_indices = np.where(mask_from_env != mask_from_gnn)[0]
                
                mismatch_info = {
                    'episode': episode,
                    'step': episode_checks,
                    'truck_id': env.active_truck_id,
                    'diff_indices': diff_indices.tolist(),
                    'env_mask': mask_from_env[diff_indices].tolist(),
                    'gnn_mask': mask_from_gnn[diff_indices].tolist(),
                }
                mismatch_details.append(mismatch_info)
                
                print(f"⚠️  MISMATCH in Episode {episode}, Step {episode_checks}")
                print(f"   Truck ID: {env.active_truck_id}")
                print(f"   Differing action indices: {diff_indices.tolist()}")
                print(f"   env.mask_fn():  {mask_from_env[diff_indices]}")
                print(f"   GNN state mask: {mask_from_gnn[diff_indices]}")
                print()
            
            # Select a random feasible action from GNN mask
            feasible_indices = [i for i, is_feasible in enumerate(mask_from_gnn) if is_feasible]
            
            if not feasible_indices:
                print(f"Episode {episode}: No feasible actions available")
                break
            
            # Select random feasible action using GNN format
            selected_idx = random.choice(feasible_indices)
            node_id, is_charging = gnn_graph.action_to_node_map[selected_idx]
            charge_duration = gnn_graph.action_charge_durations[selected_idx].item()
            action = (node_id, charge_duration, is_charging)
            
            # Step environment
            try:
                obs, reward, done, truncated, info = env.step(action)
            except ValueError as e:
                print(f"   Skipping step due to error: {e}")
                break
            
            if done or truncated:
                break
        
        # Print episode summary
        if episode_mismatches == 0:
            print(f"✓ Episode {episode}: All {episode_checks} checks passed")
        else:
            print(f"✗ Episode {episode}: {episode_mismatches}/{episode_checks} mismatches")
        print()

    # Print final summary
    print("="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)
    print(f"Total checks performed: {total_checks}")
    print(f"Total mismatches found: {total_mismatches}")
    
    if total_mismatches == 0:
        print("\n✓ SUCCESS: All action masks match perfectly!")
    else:
        print(f"\n✗ FAILURE: {total_mismatches} mismatches found ({100*total_mismatches/total_checks:.2f}%)")
        print("\nMismatch details saved for debugging.")
    
    print("="*80)
    
    return total_checks, total_mismatches, mismatch_details


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify action mask consistency")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to test")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    total_checks, total_mismatches, details = verify_action_masks(
        num_episodes=args.episodes,
        seed=args.seed
    )
    
    if total_mismatches > 0 and args.verbose:
        print("\nDetailed mismatch information:")
        for i, detail in enumerate(details):
            print(f"\nMismatch {i+1}:")
            print(f"  Episode: {detail['episode']}, Step: {detail['step']}")
            print(f"  Truck: {detail['truck_id']}")
            print(f"  Differing actions: {detail['diff_indices']}")
            print(f"  env.mask_fn():  {detail['env_mask']}")
            print(f"  GNN state mask: {detail['gnn_mask']}")
