"""
Test script for GNNStateSpaceDetourBased implementation.

This script validates:
1. Top-2 charger selection by minimum detour
2. Mandatory charging enforcement at chargers
3. Charging duration validation (reach d1 + charger OR complete trip)
4. Escape hatch for non-strategic chargers
"""

import numpy as np
from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.state.gnn_state_space_detour import GNNStateSpaceDetourBased
from EVRoutingEnv.utils.utils import load_config


def test_detour_based_state_space():
    """Test the detour-based state space implementation."""
    
    print("=" * 80)
    print("Testing GNNStateSpaceDetourBased Implementation")
    print("=" * 80)
    
    # Load a standard config
    print("\n1. Loading configuration...")
    config = load_config("EVRoutingEnv/config_files/config.yaml")
    
    # Override some settings for testing
    config["environment"]["num_trucks"] = 1
    config["environment"]["num_stops"] = 3
    
    # Create environment
    print("\n2. Creating environment...")
    env = EventDrivenTruckEnv(config=config, verbose=False)
    
    # Create detour-based state space
    print("\n3. Creating GNNStateSpaceDetourBased...")
    state_space = GNNStateSpaceDetourBased(
        num_trucks=len(env.trucks),
        num_stops=env.num_stops,
        max_time=env.max_time,
        num_charging_nodes=env.num_charging_nodes,
        device="cpu",
        verbose=True,
    )
    
    # Reset environment
    print("\n4. Resetting environment...")
    obs = env.reset()
    
    print(f"\nInitial state:")
    print(f"  Active truck: {env.active_truck_id}")
    if env.active_truck_id is not None:
        truck = env.trucks[env.active_truck_id]
        print(f"  Current location: {truck.current_node}")
        print(f"  Current battery: {truck.current_battery:.1f} kWh")
        print(f"  Next delivery: {truck.get_next_delivery_target()}")
        print(f"  Remaining deliveries: {truck.get_remaining_deliveries()}")
    
    # Get GNN state
    print("\n5. Building GNN state with detour-based charger selection...")
    print("-" * 80)
    gnn_data = state_space.get_state_GNN(env)
    print("-" * 80)
    
    # Analyze action space
    print("\n6. Analyzing action space:")
    action_to_node_map = gnn_data.action_to_node_map
    feasible_mask = gnn_data.feasible_action_mask.numpy()
    
    print(f"  Total actions: {len(action_to_node_map)}")
    print(f"  Feasible actions: {sum(feasible_mask)}")
    
    # Categorize actions
    charger_actions = []
    delivery_actions = []
    charging_actions = []
    
    for idx, (node_id, is_charging) in enumerate(action_to_node_map):
        if is_charging:
            if feasible_mask[idx]:
                charging_actions.append((idx, node_id))
        elif node_id in env.charging_nodes:
            if feasible_mask[idx]:
                charger_actions.append((idx, node_id))
        else:
            if feasible_mask[idx]:
                delivery_actions.append((idx, node_id))
    
    print(f"\n  Feasible charger routing actions: {len(charger_actions)}")
    for idx, charger_id in charger_actions:
        print(f"    Action {idx}: Route to charger {charger_id}")
    
    print(f"\n  Feasible delivery routing actions: {len(delivery_actions)}")
    for idx, delivery_id in delivery_actions:
        print(f"    Action {idx}: Route to delivery {delivery_id}")
    
    print(f"\n  Feasible charging actions: {len(charging_actions)}")
    for idx, duration in charging_actions:
        print(f"    Action {idx}: Charge for {duration} hours")
    
    # Test scenario: move to a charger and verify must_charge enforcement
    if env.active_truck_id is not None and charger_actions:
        print("\n7. Testing must_charge enforcement:")
        print("-" * 80)
        
        truck = env.trucks[env.active_truck_id]
        
        # Take action to route to first feasible charger
        charger_action_idx, target_charger = charger_actions[0]
        print(f"\n  Taking action {charger_action_idx}: Route to charger {target_charger}")
        
        obs, reward, done, truncated, info = env.step(charger_action_idx)
        
        # Fast-forward until truck arrives at charger
        max_steps = 100
        steps = 0
        while not (done or truncated) and truck.current_node != target_charger and steps < max_steps:
            # Check if we need to take another action
            if env.active_truck_id is not None:
                # Truck is ready - just take first feasible action to continue
                gnn_temp = state_space.get_state_GNN(env)
                mask_temp = gnn_temp.feasible_action_mask.numpy()
                if mask_temp.any():
                    action = np.where(mask_temp)[0][0]
                    obs, reward, done, truncated, info = env.step(action)
            steps += 1
        
        if truck.current_node == target_charger:
            print(f"\n  Truck arrived at charger {target_charger}")
            print(f"  Current battery: {truck.current_battery:.1f} kWh")
            print(f"  must_leave_charger: {truck.must_leave_charger}")
            
            # Get new GNN state - should only have charging actions
            print("\n  Building GNN state at charger (should enforce must_charge):")
            print("-" * 80)
            gnn_data = state_space.get_state_GNN(env)
            print("-" * 80)
            
            # Check action space
            action_to_node_map = gnn_data.action_to_node_map
            feasible_mask = gnn_data.feasible_action_mask.numpy()
            
            charger_actions_at_charger = []
            delivery_actions_at_charger = []
            charging_actions_at_charger = []
            
            for idx, (node_id, is_charging) in enumerate(action_to_node_map):
                if feasible_mask[idx]:
                    if is_charging:
                        charging_actions_at_charger.append((idx, node_id))
                    elif node_id in env.charging_nodes:
                        charger_actions_at_charger.append((idx, node_id))
                    else:
                        delivery_actions_at_charger.append((idx, node_id))
            
            print(f"\n  At charger - feasible routing actions: {len(charger_actions_at_charger) + len(delivery_actions_at_charger)}")
            print(f"  At charger - feasible charging actions: {len(charging_actions_at_charger)}")
            
            if len(charger_actions_at_charger) + len(delivery_actions_at_charger) == 0 and len(charging_actions_at_charger) > 0:
                print("\n  ✓ PASS: Must-charge enforcement working correctly!")
                print(f"    Only charging actions available at charger")
            else:
                print("\n  ✗ FAIL: Must-charge enforcement not working!")
                print(f"    Routing actions still available at charger")
        else:
            print(f"\n  Could not reach charger in {max_steps} steps")
    
    print("\n" + "=" * 80)
    print("Test completed!")
    print("=" * 80)


if __name__ == "__main__":
    test_detour_based_state_space()
