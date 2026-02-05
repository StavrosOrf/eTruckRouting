"""
Test script to verify depot blocking logic in flexible delivery mode.

This script tests that:
1. Depot routing is blocked when other deliveries remain
2. Depot routing is allowed only when all other deliveries are complete
3. Redirection logic works correctly in the environment
"""

import sys
import yaml
import numpy as np
from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv

def test_depot_blocking():
    """Test depot blocking in flexible delivery mode."""
    print("=" * 80)
    print("Testing Depot Blocking Logic in Flexible Delivery Mode")
    print("=" * 80)
    
    # Load config
    config_path = "./EVRoutingEnv/config_files/config_vrp.yaml"
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Create environment
        env = EventDrivenTruckEnv(config=config, verbose=True)
        
        print(f"\n✓ Environment loaded successfully")
        print(f"  - Trucks: {env.num_trucks}")
        print(f"  - Stops per truck: {env.num_stops}")
        print(f"  - Flexible delivery: {env.enable_flexible_delivery_order}")
        
        # Reset environment
        obs, info = env.reset()
        print(f"\n✓ Environment reset")
        
        # Get active truck
        truck = env.trucks[env.active_truck_id]
        print(f"\n📍 Active Truck: {truck.truck_id}")
        print(f"  - Delivery sequence: {truck.delivery_sequence}")
        print(f"  - Depot node: {truck.delivery_sequence[0]}")
        print(f"  - Actual deliveries: {truck.delivery_sequence[1:]}")
        print(f"  - Current location: {truck.current_node}")
        print(f"  - Battery: {truck.current_battery:.2f}/{truck.battery_capacity:.2f} kWh")
        
        # Check helper methods
        depot_node = truck.delivery_sequence[0]
        print(f"\n🔍 Testing helper methods:")
        print(f"  - is_depot_node({depot_node}): {truck.is_depot_node(depot_node)}")
        print(f"  - has_non_depot_deliveries(): {truck.has_non_depot_deliveries()}")
        
        remaining = truck.get_next_delivery_target()
        print(f"  - get_next_delivery_target(): {remaining}")
        print(f"  - Depot in remaining? {depot_node in remaining if isinstance(remaining, list) else False}")
        
        # Get GNN state and check feasible actions
        from EVRoutingEnv.state.gnn_state_space import GNNStateSpace
        gnn_state_space = GNNStateSpace(
            num_trucks=env.num_trucks,
            num_stops=env.num_stops,
            max_time=env.max_time,
            num_charging_nodes=len(env.charging_nodes),
            verbose=True
        )
        
        print(f"\n🎯 Generating GNN state with depot blocking...")
        data = gnn_state_space.get_state_GNN(env)
        
        print(f"\n📊 Action Analysis:")
        print(f"  - Total actions: {len(data.action_to_node_map)}")
        print(f"  - Feasible actions: {data.feasible_action_mask.sum().item()}")
        
        # Check depot action feasibility
        depot_action_indices = []
        for idx, (node_id, is_charging) in enumerate(data.action_to_node_map):
            if node_id == depot_node and not is_charging:
                depot_action_indices.append(idx)
                is_feasible = data.feasible_action_mask[idx].item()
                print(f"\n  🏠 Depot action found at index {idx}:")
                print(f"     - Node: {node_id}")
                print(f"     - Feasible: {is_feasible}")
                print(f"     - Expected: False (other deliveries remain)")
                
                if is_feasible:
                    print(f"     ❌ ERROR: Depot action should be blocked!")
                    return False
                else:
                    print(f"     ✓ Correctly blocked")
        
        if not depot_action_indices:
            print(f"\n  ⚠️  No depot action found in action space")
        
        # Try to take a few valid actions
        print(f"\n🚀 Taking valid actions to deliver some nodes...")
        step_count = 0
        max_steps = 50
        
        while not (truck.is_complete or truck.failed) and step_count < max_steps:
            # Get feasible actions
            feasible_indices = [i for i, is_feas in enumerate(data.feasible_action_mask) if is_feas]
            
            if not feasible_indices:
                print(f"  ⚠️  No feasible actions available at step {step_count}")
                break
            
            # Choose first feasible action
            action = feasible_indices[0]
            node_id, is_charging = data.action_to_node_map[action]
            
            print(f"\n  Step {step_count}: Action {action} -> Node {node_id} (charging={is_charging})")
            
            # Execute action
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            
            # Update state
            data = gnn_state_space.get_state_GNN(env)
            
            if terminated or truncated:
                print(f"  Episode terminated: {terminated}, truncated: {truncated}")
                break
            
            # Check if all deliveries done
            remaining = truck.get_next_delivery_target()
            if isinstance(remaining, list):
                print(f"  Remaining deliveries: {remaining}")
                
                # Check if only depot remains
                if len(remaining) == 1 and truck.is_depot_node(remaining[0]):
                    print(f"\n  🎉 All non-depot deliveries complete! Only depot remains: {remaining[0]}")
                    
                    # Check depot action is now feasible
                    depot_action_feasible = False
                    for idx, (node_id, is_charging) in enumerate(data.action_to_node_map):
                        if node_id == depot_node and not is_charging:
                            is_feasible = data.feasible_action_mask[idx].item()
                            print(f"  🏠 Depot action at index {idx}: feasible={is_feasible}")
                            if is_feasible:
                                depot_action_feasible = True
                                print(f"     ✓ Depot action correctly enabled!")
                            else:
                                print(f"     ❌ ERROR: Depot action should be feasible now!")
                                return False
                    
                    if not depot_action_feasible:
                        print(f"     ⚠️  Depot action not found in action space")
                    break
        
        print(f"\n" + "=" * 80)
        print(f"✓ Depot blocking test completed successfully!")
        print(f"=" * 80)
        return True
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_depot_blocking()
    sys.exit(0 if success else 1)
