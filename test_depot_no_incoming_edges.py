"""
Test script to verify that depot nodes have no incoming edges.
"""

import sys
sys.path.insert(0, "/home/sorfanouda/EVPR")

from truck_env.models.event_driven_env import EventDrivenTruckEnv
from truck_env.models.gnn_state_space import GNNStateSpace


def test_depot_no_incoming_edges():
    """Test that depot nodes have no incoming edges."""
    print("\n" + "=" * 80)
    print("Testing: Depot Nodes Have No Incoming Edges")
    print("=" * 80)
    
    # Initialize environment
    config_path = "/home/sorfanouda/EVPR/truck_env/config_files/config.yaml"
    env = EventDrivenTruckEnv(config_path, verbose=False, enable_plotting=False)
    
    # Initialize GNN state
    gnn_state = GNNStateSpace(
        num_trucks=env.num_trucks,
        num_stops=env.num_stops,
        max_time=env.max_time,
        num_charging_nodes=env.num_charging_nodes,
    )
    
    # Reset environment
    obs, info = env.reset(seed=0)
    
    # Get initial GNN state
    data = gnn_state.get_state_GNN(env)
    
    print(f"\nInitial Graph State:")
    print(f"  Total nodes: {data.num_nodes}")
    print(f"  Total edges: {data.num_edges}")
    print(f"  Node types: {[data.node_list[i][0] for i in range(min(10, data.num_nodes))]}")
    
    # Extract node list and identify depot indices
    node_list = data.node_list
    depot_indices = []
    for idx, (node_type, node_id) in enumerate(node_list):
        if node_type == "depot":
            depot_indices.append((idx, node_id))
    
    print(f"\nDepots found: {len(depot_indices)}")
    for idx, node_id in depot_indices:
        print(f"  - Depot {node_id} (node index: {idx})")
    
    # Check for incoming edges to depots
    edge_index = data.edge_index.cpu().numpy()
    incoming_edges_per_depot = {idx: 0 for idx, node_id in depot_indices}
    
    print(f"\nChecking incoming edges...")
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        
        # Check if destination is a depot
        for depot_idx, depot_id in depot_indices:
            if dst == depot_idx:
                incoming_edges_per_depot[depot_idx] += 1
                src_type, src_id = node_list[src]
                print(f"  ⚠ Found incoming edge: {src_type} {src_id} → Depot {depot_id}")
    
    print(f"\nIncoming edges per depot:")
    all_clear = True
    for idx, node_id in depot_indices:
        count = incoming_edges_per_depot[idx]
        status = "✓" if count == 0 else "✗"
        print(f"  {status} Depot {node_id}: {count} incoming edges")
        if count > 0:
            all_clear = False
    
    print("\n" + "=" * 80)
    if all_clear:
        print("✓ SUCCESS: All depots have NO incoming edges!")
    else:
        print("✗ FAILURE: Some depots have incoming edges!")
    print("=" * 80 + "\n")
    
    return all_clear


if __name__ == "__main__":
    success = test_depot_no_incoming_edges()
    sys.exit(0 if success else 1)
