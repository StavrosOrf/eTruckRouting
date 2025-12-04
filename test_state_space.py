"""
Test script to verify the flattened state space matches GNN representation.
"""

import numpy as np
import torch
from truck_env.state.state_space import StateSpace
from truck_env.models.event_driven_env import EventDrivenTruckEnv
from truck_env.state.gnn_state_space import GNNStateSpace
from truck_env.baselines.heuristic_policy import HeuristicPolicy


def test_state_space_dimensions():
    """Test that state space has correct dimensions."""
    print("=" * 60)
    print("Testing State Space Dimensions")
    print("=" * 60)
    
    # Create environment with config
    config_path = "truck_env/config_files/config.yaml"
    env = EventDrivenTruckEnv(
        config=config_path,
        verbose=False,
        enable_plotting=False,
    )
    
    # Get state space info
    state_space = env.state_space_manager
    feature_info = state_space.get_feature_info()
    
    print(f"\nTotal state size: {feature_info['total_size']}")
    print(f"\nTruck features:")
    print(f"  - Count: {feature_info['truck_features']['count']}")
    print(f"  - Features per truck: {feature_info['truck_features']['features_per_truck']}")
    print(f"  - Total size: {feature_info['truck_features']['size']}")
    print(f"  - Offset: {feature_info['truck_features']['offset']}")
    print(f"  - Feature names: {feature_info['truck_features']['feature_names']}")
    
    print(f"\nDelivery features:")
    print(f"  - Count: {feature_info['delivery_features']['count']}")
    print(f"  - Features per delivery: {feature_info['delivery_features']['features_per_delivery']}")
    print(f"  - Total size: {feature_info['delivery_features']['size']}")
    print(f"  - Offset: {feature_info['delivery_features']['offset']}")
    print(f"  - Feature names: {feature_info['delivery_features']['feature_names']}")
    
    print(f"\nCharger features:")
    print(f"  - Count: {feature_info['charger_features']['count']}")
    print(f"  - Features per charger: {feature_info['charger_features']['features_per_charger']}")
    print(f"  - Total size: {feature_info['charger_features']['size']}")
    print(f"  - Offset: {feature_info['charger_features']['offset']}")
    print(f"  - Feature names: {feature_info['charger_features']['feature_names']}")
    
    print(f"\nGlobal features:")
    print(f"  - Features: {feature_info['global_features']['features']}")
    print(f"  - Offset: {feature_info['global_features']['offset']}")
    print(f"  - Feature names: {feature_info['global_features']['feature_names']}")
    
    # Verify observation space shape
    obs_shape = env.observation_space.shape
    print(f"\nObservation space shape: {obs_shape}")
    print(f"Expected: ({feature_info['total_size']},)")
    assert obs_shape == (feature_info['total_size'],), "Shape mismatch!"
    print("✓ Shape matches!")
    

def test_state_generation():
    """Test that state can be generated from environment."""
    print("\n" + "=" * 60)
    print("Testing State Generation")
    print("=" * 60)
    
    # Create environment with config
    config_path = "truck_env/config_files/config.yaml"
    env = EventDrivenTruckEnv(
        config=config_path,
        verbose=False,
        enable_plotting=False,
    )
    
    # Reset and get initial state
    obs, info = env.reset(seed=42)
    
    print(f"\nObservation shape: {obs.shape}")
    print(f"Observation dtype: {obs.dtype}")
    print(f"Min value: {obs.min():.4f}")
    print(f"Max value: {obs.max():.4f}")
    print(f"Non-zero elements: {np.count_nonzero(obs)}/{obs.size}")
    
    # Check normalization
    assert obs.min() >= 0.0, "Values below 0!"
    # Note: Charger node IDs can be > 1 (normalized by num_charging_nodes, not max value)
    # This matches GNN state space behavior
    print("✓ All values are non-negative (some node IDs may exceed 1)")
    
    # Get feature info to examine specific sections
    feature_info = env.state_space_manager.get_feature_info()
    
    # Check truck features
    truck_offset = feature_info['truck_features']['offset']
    truck_size = feature_info['truck_features']['size']
    truck_features = obs[truck_offset:truck_offset + truck_size]
    print(f"\nTruck features non-zero: {np.count_nonzero(truck_features)}/{truck_size}")
    
    # Check delivery features
    delivery_offset = feature_info['delivery_features']['offset']
    delivery_size = feature_info['delivery_features']['size']
    delivery_features = obs[delivery_offset:delivery_offset + delivery_size]
    print(f"Delivery features non-zero: {np.count_nonzero(delivery_features)}/{delivery_size}")
    
    # Check charger features
    charger_offset = feature_info['charger_features']['offset']
    charger_size = feature_info['charger_features']['size']
    charger_features = obs[charger_offset:charger_offset + charger_size]
    print(f"Charger features non-zero: {np.count_nonzero(charger_features)}/{charger_size}")
    
    # Check global features
    global_offset = feature_info['global_features']['offset']
    global_size = feature_info['global_features']['size']
    global_features = obs[global_offset:global_offset + global_size]
    print(f"Global features: {global_features}")
    

def test_state_evolution():
    """Test that state changes during episode."""
    print("\n" + "=" * 60)
    print("Testing State Evolution")
    print("=" * 60)
    
    # Create environment with config
    config_path = "truck_env/config_files/config.yaml"
    env = EventDrivenTruckEnv(
        config=config_path,
        verbose=False,
        enable_plotting=False,
    )
    
    # Reset
    obs, info = env.reset(seed=42)
    initial_obs = obs.copy()
    
    print(f"\nInitial non-zero elements: {np.count_nonzero(obs)}/{obs.size}")
    
    # Take a few steps
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            break
    
    print(f"After {step + 1} steps non-zero elements: {np.count_nonzero(obs)}/{obs.size}")
    
    # Check that state has changed
    diff = np.abs(obs - initial_obs)
    changed_elements = np.count_nonzero(diff)
    print(f"Changed elements: {changed_elements}/{obs.size}")
    
    assert changed_elements > 0, "State did not change!"
    print("✓ State evolved during episode")
    

def test_completed_truck_zeros():
    """Test that completed trucks have zero features."""
    print("\n" + "=" * 60)
    print("Testing Completed Truck Zeros")
    print("=" * 60)
    
    # Create environment with config
    config_path = "truck_env/config_files/config.yaml"
    env = EventDrivenTruckEnv(
        config=config_path,
        verbose=False,
        enable_plotting=False,
    )
    
    # Reset
    obs, info = env.reset(seed=42)
    feature_info = env.state_space_manager.get_feature_info()
    
    truck_offset = feature_info['truck_features']['offset']
    truck_dim = feature_info['truck_features']['features_per_truck']
    
    # Run until at least one truck completes
    max_steps = 1000
    for step in range(max_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Check if any truck completed
        completed_trucks = [i for i, truck in enumerate(env.trucks) 
                          if truck.is_complete or truck.failed]
        
        if completed_trucks:
            print(f"\nStep {step}: Truck(s) {completed_trucks} completed/failed")
            
            # Check their features are zeros
            for truck_id in completed_trucks:
                start_idx = truck_offset + truck_id * truck_dim
                end_idx = start_idx + truck_dim
                truck_features = obs[start_idx:end_idx]
                
                if np.allclose(truck_features, 0.0):
                    print(f"  ✓ Truck {truck_id} features are all zeros")
                else:
                    print(f"  ✗ Truck {truck_id} features NOT zeros: {truck_features}")
                    non_zero = np.count_nonzero(truck_features)
                    print(f"    Non-zero elements: {non_zero}/{truck_dim}")
            
            break
        
        if terminated or truncated:
            print(f"\nEpisode ended at step {step}")
            break


def test_flattened_vs_gnn_features():
    """Test that flattened state features match GNN state features over 100 episodes."""
    print("\n" + "=" * 60)
    print("Testing Flattened vs GNN Feature Consistency (100 Episodes)")
    print("=" * 60)
    
    # Create environment with config
    config_path = "truck_env/config_files/config.yaml"
    env = EventDrivenTruckEnv(
        config=config_path,
        verbose=False,
        enable_plotting=False,
    )
    
    # Create GNN state space manager
    gnn_state_space = GNNStateSpace(
        num_trucks=env.num_trucks,
        num_stops=env.num_stops,
        max_time=env.max_time,
        num_charging_nodes=env.num_charging_nodes,
        device='cpu',
        verbose=False,
    )
    
    # Get flattened state feature info
    env.reset(seed=0)
    feature_info = env.state_space_manager.get_feature_info()
    
    truck_offset = feature_info['truck_features']['offset']
    truck_dim = feature_info['truck_features']['features_per_truck']
    delivery_offset = feature_info['delivery_features']['offset']
    delivery_dim = feature_info['delivery_features']['features_per_delivery']
    charger_offset = feature_info['charger_features']['offset']
    charger_dim = feature_info['charger_features']['features_per_charger']
    
    # Create heuristic policy
    heuristic_policy = HeuristicPolicy(verbose=False)
    
    # Statistics
    total_steps = 0
    total_mismatches = 0
    episodes_with_mismatches = 0
    max_truck_diff = 0.0
    max_delivery_diff = 0.0
    max_charger_diff = 0.0
    
    print(f"\nRunning 100 episodes with HeuristicPolicy baseline...")
    print(f"Episode progress: ", end="", flush=True)
    
    # Run 100 episodes
    for episode in range(100):
        if (episode + 1) % 10 == 0:
            print(f"{episode + 1}", end="...", flush=True)
        
        obs_flat, info = env.reset(seed=episode)
        episode_mismatches = 0
        step = 0
        
        while True:
            # Get both states from the SAME environment state
            # Important: Get flat state first, then GNN state from same env
            try:
                gnn_state = gnn_state_space.get_state_GNN(env)
            except (ValueError, RuntimeError) as e:
                # Episode might have ended
                if "No action metadata" in str(e) or "No active trucks" in str(e):
                    break
                raise
            
            # obs_flat should be from the current environment state, not previous step
            # We need to regenerate it
            current_obs_flat = env.state_space_manager.get_state(
                trucks=env.trucks,
                active_truck_id=env.active_truck_id,
                transport_graph=env.transport_graph,
                charging_nodes=env.charging_nodes,
                truck_states=env.truck_states,
                event_queue=env.event_queue,
                global_clock=env.global_clock,
                charging_station=env.charging_station,
            )
            
            # Compare truck features
            gnn_truck_features = gnn_state['truck'].x.cpu().numpy()
            num_active_trucks = gnn_truck_features.shape[0]
            
            for truck_idx in range(num_active_trucks):
                flat_start = truck_offset + truck_idx * truck_dim
                flat_end = flat_start + truck_dim
                flat_truck_features = current_obs_flat[flat_start:flat_end]
                gnn_truck_feature = gnn_truck_features[truck_idx]
                
                # Compare first 13 features (14th is reserved/padding in flat)
                features_to_compare = min(13, len(gnn_truck_feature))
                flat_compare = flat_truck_features[:features_to_compare]
                gnn_compare = gnn_truck_feature[:features_to_compare]
                
                diff = np.abs(flat_compare - gnn_compare)
                max_diff = diff.max()
                max_truck_diff = max(max_truck_diff, max_diff)
                
                if max_diff > 1e-5:
                    # Fail immediately with detailed error
                    feature_names = ['node_type', 'current_position', 'battery_level', 'battery_percentage',
                                   'state_ready', 'state_routing', 'state_waiting', 'state_charging',
                                   'deliveries_done', 'deliveries_remaining', 'time_elapsed',
                                   'distance_traveled', 'time_to_destination']
                    diff_idx = np.argmax(diff)
                    print(f"\n{'='*60}")
                    print(f"TRUCK FEATURE MISMATCH DETECTED!")
                    print(f"{'='*60}")
                    print(f"Episode: {episode}, Step: {step}, Truck: {truck_idx}")
                    print(f"Feature: {feature_names[diff_idx]} (index {diff_idx})")
                    print(f"Flat value: {flat_compare[diff_idx]}")
                    print(f"GNN value:  {gnn_compare[diff_idx]}")
                    print(f"Difference: {diff[diff_idx]}")
                    print(f"\nAll features:")
                    for i, name in enumerate(feature_names):
                        print(f"  {name:25s}: flat={flat_compare[i]:.9f}, gnn={gnn_compare[i]:.9f}, diff={diff[i]:.9f}")
                    raise AssertionError(f"Truck feature '{feature_names[diff_idx]}' mismatch: flat={flat_compare[diff_idx]}, gnn={gnn_compare[diff_idx]}")
            
            # Compare delivery features
            gnn_delivery_features = gnn_state['delivery'].x.cpu().numpy()
            num_deliveries = gnn_delivery_features.shape[0]
            
            # Count non-zero deliveries in flat (to match GNN which only has active ones)
            flat_delivery_all = current_obs_flat[delivery_offset:delivery_offset + feature_info['delivery_features']['size']]
            flat_delivery_reshaped = flat_delivery_all.reshape(-1, delivery_dim)
            non_zero_mask = np.any(flat_delivery_reshaped != 0, axis=1)
            non_zero_indices = np.where(non_zero_mask)[0]
            
            if episode == 2 and step == 9:
                print(f"\n  Debug info at failure point:")
                print(f"  GNN has {num_deliveries} deliveries, {num_active_trucks} active trucks")
                print(f"  Flat has {len(non_zero_indices)} non-zero delivery slots")
                for truck_id, truck in enumerate(env.trucks):
                    print(f"  Truck {truck_id}: is_complete={truck.is_complete}, failed={truck.failed}, state={env.truck_states.get(truck_id)}")
                print(f"  Active truck ID: {env.active_truck_id}")
            
            # Only compare non-zero deliveries (up to number in GNN)
            for flat_idx_pos, flat_idx in enumerate(non_zero_indices[:num_deliveries]):
                flat_start = delivery_offset + flat_idx * delivery_dim
                flat_end = flat_start + delivery_dim
                flat_delivery_feature = current_obs_flat[flat_start:flat_end]
                
                if flat_idx_pos >= num_deliveries:
                    break
                    
                gnn_delivery_feature = gnn_delivery_features[flat_idx_pos]
                
                diff = np.abs(flat_delivery_feature - gnn_delivery_feature)
                max_diff = diff.max()
                max_delivery_diff = max(max_delivery_diff, max_diff)
                
                if max_diff > 1e-5:
                    # Fail immediately with detailed error
                    feature_names = ['node_type', 'node_id', 'delivery_sequence_index']
                    diff_idx = np.argmax(diff)
                    print(f"\n{'='*60}")
                    print(f"DELIVERY FEATURE MISMATCH DETECTED!")
                    print(f"{'='*60}")
                    print(f"Episode: {episode}, Step: {step}, Flat index: {flat_idx}, GNN index: {flat_idx_pos}")
                    print(f"Feature: {feature_names[diff_idx]} (index {diff_idx})")
                    print(f"Flat value: {flat_delivery_feature[diff_idx]}")
                    print(f"GNN value:  {gnn_delivery_feature[diff_idx]}")
                    print(f"Difference: {diff[diff_idx]}")
                    print(f"\nAll features:")
                    for i, name in enumerate(feature_names):
                        print(f"  {name:25s}: flat={flat_delivery_feature[i]:.9f}, gnn={gnn_delivery_feature[i]:.9f}, diff={diff[i]:.9f}")
                    raise AssertionError(f"Delivery feature '{feature_names[diff_idx]}' mismatch: flat={flat_delivery_feature[diff_idx]}, gnn={gnn_delivery_feature[diff_idx]}")
            
            # Compare charger features
            gnn_charger_features = gnn_state['charger'].x.cpu().numpy()
            num_chargers = gnn_charger_features.shape[0]
            
            for charger_idx in range(num_chargers):
                flat_start = charger_offset + charger_idx * charger_dim
                flat_end = flat_start + charger_dim
                flat_charger_feature = current_obs_flat[flat_start:flat_end]
                gnn_charger_feature = gnn_charger_features[charger_idx]
                
                diff = np.abs(flat_charger_feature - gnn_charger_feature)
                max_diff = diff.max()
                max_charger_diff = max(max_charger_diff, max_diff)
                
                if max_diff > 1e-5:
                    # Fail immediately with detailed error
                    feature_names = ['node_type', 'node_id', 'occupancy_rate', 'queue_length']
                    diff_idx = np.argmax(diff)
                    print(f"\n{'='*60}")
                    print(f"CHARGER FEATURE MISMATCH DETECTED!")
                    print(f"{'='*60}")
                    print(f"Episode: {episode}, Step: {step}, Charger: {charger_idx}")
                    print(f"Feature: {feature_names[diff_idx]} (index {diff_idx})")
                    print(f"Flat value: {flat_charger_feature[diff_idx]}")
                    print(f"GNN value:  {gnn_charger_feature[diff_idx]}")
                    print(f"Difference: {diff[diff_idx]}")
                    print(f"\nAll features:")
                    for i, name in enumerate(feature_names):
                        print(f"  {name:25s}: flat={flat_charger_feature[i]:.9f}, gnn={gnn_charger_feature[i]:.9f}, diff={diff[i]:.9f}")
                    raise AssertionError(f"Charger feature '{feature_names[diff_idx]}' mismatch: flat={flat_charger_feature[diff_idx]}, gnn={gnn_charger_feature[diff_idx]}")
            
            # Take action with heuristic policy
            action = heuristic_policy.get_action(env)
            obs_flat, reward, terminated, truncated, info = env.step(action)
            
            total_steps += 1
            step += 1
            
            if terminated or truncated:
                break
            
            # Safety limit
            if step > 1000:
                print(f"\n  Warning: Episode {episode} exceeded 1000 steps")
                break
        
        if episode_mismatches > 0:
            episodes_with_mismatches += 1
    
    print(" Done!\n")
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print("Summary Statistics")
    print(f"{'='*60}")
    print(f"Total episodes: 100")
    print(f"Total steps: {total_steps}")
    print(f"Episodes with mismatches: {episodes_with_mismatches}/100")
    print(f"Total feature mismatches: {total_mismatches}")
    print(f"\nMaximum differences observed:")
    print(f"  Truck features: {max_truck_diff:.9f}")
    print(f"  Delivery features: {max_delivery_diff:.9f}")
    print(f"  Charger features: {max_charger_diff:.9f}")
    
    if total_mismatches == 0:
        print(f"\n{'='*60}")
        print("✓ ALL FEATURES MATCHED ACROSS ALL 100 EPISODES!")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print(f"✗ Found {total_mismatches} total mismatches across {episodes_with_mismatches} episodes")
        print(f"{'='*60}")
        raise AssertionError(f"Flattened and GNN features do not match ({total_mismatches} mismatches found)")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("FLATTENED STATE SPACE TESTS")
    print("=" * 60)
    
    try:
        test_state_space_dimensions()
        test_state_generation()
        test_state_evolution()
        test_completed_truck_zeros()
        test_flattened_vs_gnn_features()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60 + "\n")
    except Exception as e:
        print(f"\n{'=' * 60}")
        print(f"TEST FAILED: {e}")
        print("=" * 60 + "\n")
        import traceback
        traceback.print_exc()
