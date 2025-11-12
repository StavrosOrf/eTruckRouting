"""
Test to verify truck is always connected to its current location with correct edge weights.
"""

import numpy as np
import sys
sys.path.insert(0, '/home/sorfanouda/EVPR')

from truck_env.models.event_driven_env import EventDrivenTruckEnv
from truck_env.state.gnn_state_space import GNNStateSpace
from truck_env.baselines.heuristic_policy import HeuristicPolicy


def analyze_truck_edges(data, env):
    """Analyze the edges connected to truck nodes."""
    print("\n" + "="*80)
    print("TRUCK EDGE ANALYSIS")
    print("="*80)
    
    # Find truck nodes (NODE_TYPE_TRUCK = 0 after removal of depot)
    node_types = data.x[:, 0].numpy()
    truck_indices = np.where(node_types == 0)[0]
    
    if len(truck_indices) == 0:
        print("No truck nodes in graph (all trucks completed/failed)")
        return
    
    edge_index = data.edge_index.numpy()
    edge_attr = data.edge_attr.numpy()
    
    # Analyze each truck
    for truck_idx in truck_indices:
        # Find the actual truck from node_list
        if hasattr(data, 'node_list'):
            node_type, truck_id = data.node_list[truck_idx]
            truck = env.trucks[truck_id]
            
            print(f"\n📍 Truck {truck_id}:")
            print(f"   Current Node: {truck.current_node}")
            print(f"   State: ", end="")
            if truck.is_charging:
                print("CHARGING")
            elif truck.truck_id in env.charging_station.charger_waitlist.get(truck.current_node, []):
                print("WAITING_TO_CHARGE")
            elif truck.route_destination is not None:
                print(f"ROUTING to {truck.route_destination}")
            else:
                print("READY")
            
            # Find edges connected to this truck
            outgoing = np.where(edge_index[0] == truck_idx)[0]
            incoming = np.where(edge_index[1] == truck_idx)[0]
            
            print(f"\n   Outgoing edges ({len(outgoing)}):")
            for edge_id in outgoing[:5]:  # Show first 5
                target_idx = edge_index[1, edge_id]
                energy, time = edge_attr[edge_id]
                if hasattr(data, 'node_list'):
                    target_type, target_id = data.node_list[target_idx]
                    # Updated type names after depot removal: Truck=0, Delivery=1, Charger=2
                    type_name = {0: "Truck", 1: "Delivery", 2: "Charger"}.get(int(node_types[target_idx]), "Unknown")
                    print(f"      → {type_name} (id {target_id}, idx {target_idx}): energy={energy:.2f} kWh, time={time:.2f} h")
                    
                    # Check if this is the current location
                    if target_id == truck.current_node:
                        print(f"         ✅ THIS IS CURRENT LOCATION (energy should be 0)")
                        if energy != 0.0:
                            print(f"         ⚠️  ERROR: Energy is {energy}, should be 0!")
            
            if len(outgoing) > 5:
                print(f"      ... and {len(outgoing) - 5} more")
            
            print(f"\n   Incoming edges ({len(incoming)}):")
            for edge_id in incoming[:5]:  # Show first 5
                source_idx = edge_index[0, edge_id]
                energy, time = edge_attr[edge_id]
                if hasattr(data, 'node_list'):
                    source_type, source_id = data.node_list[source_idx]
                    # Updated type names after depot removal: Truck=0, Delivery=1, Charger=2
                    type_name = {0: "Truck", 1: "Delivery", 2: "Charger"}.get(int(node_types[source_idx]), "Unknown")
                    print(f"      {type_name} (id {source_id}, idx {source_idx}) →: energy={energy:.2f} kWh, time={time:.2f} h")
                    
                    # Check if this is the current location
                    if source_id == truck.current_node:
                        print(f"         ✅ THIS IS CURRENT LOCATION (energy should be 0)")
                        if energy != 0.0:
                            print(f"         ⚠️  ERROR: Energy is {energy}, should be 0!")
            
            if len(incoming) > 5:
                print(f"      ... and {len(incoming) - 5} more")
            
            # Check if truck is connected to current location
            connected_to_current = False
            for edge_id in list(outgoing) + list(incoming):
                if edge_id in outgoing:
                    other_idx = edge_index[1, edge_id]
                else:
                    other_idx = edge_index[0, edge_id]
                
                if hasattr(data, 'node_list'):
                    _, other_id = data.node_list[other_idx]
                    if other_id == truck.current_node:
                        connected_to_current = True
                        energy, time = edge_attr[edge_id]
                        if energy == 0.0 and time == 0.0:
                            print(f"\n   ✅ CORRECT: Truck connected to current location (node {truck.current_node}) with 0 energy/time")
                        else:
                            print(f"\n   ⚠️  ERROR: Truck connected to current location but energy={energy}, time={time} (should be 0)")
                        break
            
            if not connected_to_current:
                # Check if truck is at depot (which we removed from graph)
                is_at_depot = truck.current_node == truck.delivery_sequence[0]
                if is_at_depot:
                    print(f"\n   ℹ️  Truck at depot (node {truck.current_node}) - depot nodes removed from graph per new design")
                else:
                    print(f"\n   ❌ ERROR: Truck NOT connected to its current location (node {truck.current_node})!")


def main():
    config_file = "truck_env/config_files/config.yaml"
    
    env = EventDrivenTruckEnv(
        config=config_file,
        run_id="truck_edge_test",
        verbose=False,
        enable_plotting=False
    )
    
    gnn_state = GNNStateSpace(
        num_trucks=env.num_trucks,
        num_stops=env.num_stops,
        max_time=env.max_time,
        num_charging_nodes=env.num_charging_nodes,
    )
    
    policy = HeuristicPolicy(verbose=False)
    
    env.reset()
    
    # Test initial state
    print("\n" + "="*80)
    print("TEST 1: INITIAL STATE (truck at depot)")
    print("="*80)
    obs = gnn_state.get_state_GNN(env)
    analyze_truck_edges(obs, env)
    
    # Take a step (deliver first package)
    print("\n\n" + "="*80)
    print("TEST 2: AFTER FIRST DELIVERY")
    print("="*80)
    action = policy.get_action(env)
    env.step(action)
    obs = gnn_state.get_state_GNN(env)
    analyze_truck_edges(obs, env)
    
    # Take another step
    print("\n\n" + "="*80)
    print("TEST 3: AFTER SECOND DELIVERY")
    print("="*80)
    action = policy.get_action(env)
    env.step(action)
    obs = gnn_state.get_state_GNN(env)
    analyze_truck_edges(obs, env)
    
    # Go to charger
    print("\n\n" + "="*80)
    print("TEST 4: AT CHARGER")
    print("="*80)
    action = policy.get_action(env)
    env.step(action)
    obs = gnn_state.get_state_GNN(env)
    analyze_truck_edges(obs, env)
    
    env.close()
    print("\n\n✅ Test completed!")


if __name__ == "__main__":
    main()
