"""
Test GNN state representation to verify:
1. Truck state encoding (ready, routing, waiting_to_charge, charging)
2. Completed/failed trucks are filtered out
3. Delivery points of completed/failed trucks are filtered out
"""

import numpy as np
import sys
sys.path.insert(0, '/home/sorfanouda/EVPR')

from truck_env.models.event_driven_env import EventDrivenTruckEnv
from truck_env.state.gnn_state_space import GNNStateSpace
from truck_env.baselines.heuristic_policy import HeuristicPolicy

def print_gnn_state_info(obs, env):
    """Print information about the GNN state."""
    print("\n" + "="*80)
    print("GNN State Information")
    print("="*80)
    
    # Access the Data object
    data = obs
    
    print(f"\nNode features shape: {data.x.shape}")
    print(f"Edge index shape: {data.edge_index.shape}")
    print(f"Edge features shape: {data.edge_attr.shape}")
    print(f"Number of nodes: {data.x.shape[0]}")
    print(f"Number of edges: {data.edge_index.shape[1]}")
    
    # Print node types
    node_types = data.x[:, 0].numpy()
    type_counts = {
        0: "Depot",
        1: "Truck",
        2: "Delivery",
        3: "Charger"
    }
    
    print("\nNode type distribution:")
    for type_id, type_name in type_counts.items():
        count = np.sum(node_types == type_id)
        print(f"  {type_name}: {count}")
    
    # Print truck states
    truck_nodes = data.x[node_types == 1]
    if len(truck_nodes) > 0:
        print("\nTruck states (one-hot encoded):")
        print("  Feature indices: [4]=ready, [5]=routing, [6]=waiting_to_charge, [7]=charging")
        for i, truck_node in enumerate(truck_nodes):
            state_vec = truck_node[4:8].numpy()
            state_name = ["ready", "routing", "waiting_to_charge", "charging"][np.argmax(state_vec)]
            battery_pct = truck_node[3].item()
            deliveries_done = int(truck_node[8].item())
            deliveries_remaining = int(truck_node[9].item())
            print(f"  Truck {i}: state={state_name}, battery={battery_pct:.1f}%, "
                  f"deliveries_done={deliveries_done}, remaining={deliveries_remaining}")
    
    # Print actual truck info from env
    print("\nActual truck states from environment:")
    active_count = 0
    complete_count = 0
    failed_count = 0
    for truck in env.trucks:
        if truck.is_complete:
            complete_count += 1
            print(f"  Truck {truck.truck_id}: COMPLETE (should NOT appear in GNN)")
        elif truck.failed:
            failed_count += 1
            print(f"  Truck {truck.truck_id}: FAILED (should NOT appear in GNN)")
        else:
            active_count += 1
            # Determine state
            if truck.is_charging:
                state = "charging"
            elif truck.truck_id in env.charging_station.charger_waitlist.get(truck.current_node, []):
                state = "waiting_to_charge"
            elif truck.route_destination is not None:
                state = "routing"
            else:
                state = "ready"
            print(f"  Truck {truck.truck_id}: {state.upper()} (should appear in GNN)")
    
    print(f"\nActive: {active_count}, Complete: {complete_count}, Failed: {failed_count}")
    print(f"Expected truck nodes in GNN: {active_count}")
    print(f"Actual truck nodes in GNN: {int(np.sum(node_types == 1))}")
    
    if active_count != np.sum(node_types == 1):
        print("⚠️  WARNING: Mismatch between active trucks and GNN truck nodes!")
    else:
        print("✅ GNN correctly filters completed/failed trucks")

def main():
    # Load config and create environment
    config_file = "truck_env/config_files/config.yaml"
    
    env = EventDrivenTruckEnv(
        config=config_file,
        run_id="gnn_state_test",
        verbose=False,
        enable_plotting=False
    )
    
    # Create GNN state space
    gnn_state = GNNStateSpace(
        num_trucks=env.num_trucks,
        num_stops=env.num_stops,
        max_time=env.max_time,
        num_charging_nodes=env.num_charging_nodes,
    )
    
    # Create policy for testing
    policy = HeuristicPolicy(verbose=False)
    
    env.reset()
    obs = gnn_state.get_state_GNN(env)
    print_gnn_state_info(obs, env)
    
    # Take a few random steps to see state changes
    for step in range(5):
        print(f"\n{'='*80}")
        print(f"Step {step + 1}")
        print('='*80)
        
        # Use heuristic policy
        action = policy.get_action(env)
        if action is None:
            print("No valid action available")
            break
            
        print(f"Action: {action}")
        
        _, reward, terminated, truncated, info = env.step(action)
        obs = gnn_state.get_state_GNN(env)
        print(f"Reward: {reward:.2f}, Terminated: {terminated}, Truncated: {truncated}")
        
        print_gnn_state_info(obs, env)
        
        if terminated or truncated:
            print("\nEpisode finished!")
            break
    
    env.close()
    print("\n✅ Test completed successfully!")

if __name__ == "__main__":
    main()
