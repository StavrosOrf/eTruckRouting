"""
Utility functions for the simple truck environment.
"""

import os
import pickle
import json
import networkx as nx
from typing import Dict, Optional, Any
import yaml

def load_config(config: Optional[Any] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file or dict.
    
    Args:
        config: Path to config file, dict with config, or None for default config.yaml
        
    Returns:
        Dictionary containing configuration parameters
    """
    # If already a dict, return it
    if isinstance(config, dict):
        return config
    
    config_path = config
    if config_path is None:
        # Use default config in same directory as this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config.yaml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return config_dict

def read_file(filename: str) -> Any:
    """Read a pickle file."""
    with open(filename, "rb") as f:
        return pickle.load(f)


def map_charger_type(charger_type_str: str) -> str:
    """Normalize a source charger label without erasing its technology class."""
    normalized = "".join(
        character
        for character in str(charger_type_str).strip().lower()
        if character.isalnum()
    )
    if normalized in {"level2", "l2", "aclevel2", "ac"}:
        return "Level2"
    if normalized in {"dcfast", "dcfc", "fast", "level3", "l3"}:
        return "DCFast"
    raise ValueError(f"unsupported charger type: {charger_type_str!r}")


def check_navigation_feasibility(
    truck,
    target_node: int,
    discharge: float,
    transport_graph,
    charging_nodes: list,
    energy_safety_factor: float = 1.0,
    verbose: bool = False
) -> bool:
    """
    Check if navigating to a delivery point will leave truck with feasible actions.
    Should only be called for navigation to non-terminal delivery points.
    
    Args:
        truck: Truck object
        target_node: Delivery node to navigate to
        discharge: Energy needed to reach target_node (actual energy with uncertainty)
        transport_graph: TransportationGraph object
        charging_nodes: List of charging station node IDs
        energy_safety_factor: Safety factor for worst-case energy (e.g., 1.20 for 20% uncertainty)
        verbose: Whether to print debug information
        
    Returns:
        True if truck will have at least one feasible action after arrival, False otherwise
    """
    # Check if this is the last delivery
    remaining_after_target = [d for d in truck.get_remaining_deliveries() if d != target_node]
    
    if not remaining_after_target:
        # This is the last delivery - always feasible
        return True
    
    # Simulate battery state after arrival
    battery_after_arrival = truck.current_battery - discharge
    
    # Check 1: Can reach any charger from target with remaining battery?
    can_reach_charger = False
    for charger_node in charging_nodes:
        energy_to_charger = transport_graph.get_path_energy(target_node, int(charger_node))
        max_energy_to_charger = energy_to_charger * energy_safety_factor
        if energy_to_charger != float('inf') and max_energy_to_charger <= battery_after_arrival:
            can_reach_charger = True
            break
    
    if can_reach_charger:
        return True
    
    # Check 2: Can complete all remaining deliveries from target?
    temp_battery = battery_after_arrival
    temp_node = target_node
    can_complete_remaining = True
    
    for delivery_node in remaining_after_target:
        energy_needed = transport_graph.get_path_energy(temp_node, delivery_node)
        max_energy_needed = energy_needed * energy_safety_factor
        if energy_needed == float('inf') or max_energy_needed > temp_battery:
            can_complete_remaining = False
            break
        temp_battery -= max_energy_needed
        temp_node = delivery_node
    
    if can_complete_remaining:
        return True
    
    # Check 3: Can reach next delivery and then a charger?
    if remaining_after_target:
        next_delivery = remaining_after_target[0]
        energy_to_next = transport_graph.get_path_energy(target_node, next_delivery)
        max_energy_to_next = energy_to_next * energy_safety_factor
        
        if energy_to_next != float('inf') and max_energy_to_next <= battery_after_arrival:
            battery_after_next = battery_after_arrival - max_energy_to_next
            
            for charger_node in charging_nodes:
                energy_to_charger = transport_graph.get_path_energy(next_delivery, int(charger_node))
                max_energy_to_charger = energy_to_charger * energy_safety_factor
                if energy_to_charger != float('inf') and max_energy_to_charger <= battery_after_next:
                    return True
    
    # No feasible action found
    if verbose:
        print(f"  ERROR: Navigation to delivery {target_node} will leave truck with no feasible actions")
        print(f"    Battery after arrival: {battery_after_arrival:.1f} kWh")
        print(f"    Remaining deliveries: {remaining_after_target}")
    
    return False


def get_graph(config: Optional[Dict[str, Any]] = None) -> nx.DiGraph:
    """
    Load and build the road network graph from JSON files.

    Args:
        config: Configuration dictionary. If None, uses default config.

    Returns:
        NetworkX directed graph with nodes, edges, and charging station information
    """
    # Load config if not provided
    if config is None:
        config = load_config()
    
    # Get network configuration
    network_config = config["network"]
    data_path = network_config["data_path"]
    energy_file = network_config["shortest_path_energy_file"]
    time_file = network_config["shortest_path_time_file"]
    station_file = network_config["station_info_file"]

    # Get the directory where this file is located (EVRoutingEnv/utils)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Navigate up from utils to EVRoutingEnv
    EVRoutingEnv_dir = os.path.dirname(current_dir)
    
    # Construct data directory path
    # If data_path starts with "EVRoutingEnv/", use it as-is relative to project root
    if data_path.startswith("EVRoutingEnv/"):
        # Relative to project root
        project_root = os.path.dirname(EVRoutingEnv_dir)
        data_dir = os.path.join(project_root, data_path.rstrip('/'))
    else:
        # Relative to EVRoutingEnv directory
        data_dir = os.path.join(EVRoutingEnv_dir, data_path.rstrip('/'))

    edge_distance_file = os.path.join(data_dir, energy_file)
    edge_time_file = os.path.join(data_dir, time_file)
    chargers_file = os.path.join(data_dir, station_file)

    # Load edge distance data from JSON
    with open(edge_distance_file, 'r') as f:
        edge_distance_raw = json.load(f)
    # Convert string keys back to tuples (keys are stored as '(u, v)')
    edge_distance = {}
    for k_str, distance_val in edge_distance_raw.items():
        # Parse string representation of tuple: '(u, v)' -> (u, v)
        k_str = k_str.strip('()')
        u, v = map(int, k_str.split(', '))
        edge_distance[(u, v)] = distance_val

    # Load edge time data from JSON
    with open(edge_time_file, 'r') as f:
        edge_time_raw = json.load(f)
    # Convert string keys back to tuples
    edge_time = {}
    for k_str, time_val in edge_time_raw.items():
        # Parse string representation of tuple: '(u, v)' -> (u, v)
        k_str = k_str.strip('()')
        u, v = map(int, k_str.split(', '))
        edge_time[(u, v)] = time_val

    # Load charger data from JSON
    with open(chargers_file, 'r') as f:
        chargers_raw = json.load(f)
    # Convert string keys to integers
    chargers = {int(k): v for k, v in chargers_raw.items()}

    # Build road network graph
    G = nx.DiGraph()
    all_nodes = set()

    # Collect all nodes from edges and chargers
    for u, v in edge_distance.keys():
        all_nodes.add(u)
        all_nodes.add(v)
    all_nodes.update(chargers.keys())

    # Create sorted list of nodes and mappings (for consistent ordering)
    node_list = sorted(all_nodes)
    node_to_index = {node: idx for idx, node in enumerate(node_list)}
    index_to_node = {idx: node for node, idx in node_to_index.items()}

    # Process chargers - handle multiple types per node
    charger_aggregated = {}
    for node, info in chargers.items():
        # Handle single charger type per node
        if "station_type" in info:
            mapped_type = map_charger_type(info["station_type"])
            count = int(info["total_capacity"])
            idx = node_to_index[node]
            charger_aggregated.setdefault(idx, {})[mapped_type] = count

        # Handle multiple charger types per node
        elif "chargers" in info:
            for charger in info["chargers"]:
                mapped_type = map_charger_type(charger["station_type"])
                count = int(charger["total_capacity"])
                idx = node_to_index[node]
                charger_aggregated.setdefault(idx, {})[mapped_type] = (
                    charger_aggregated.get(idx, {}).get(mapped_type, 0) + count
                )

    # Add nodes with properties using indexes
    for idx in range(len(node_list)):
        if idx in charger_aggregated:
            props = {
                "has_charger": True,
                "charger_type": charger_aggregated[idx],
                "original_id": index_to_node[
                    idx
                ],  # Store original node ID for reference
            }
        else:
            props = {
                "has_charger": False,
                "charger_type": None,
                "original_id": index_to_node[
                    idx
                ],  # Store original node ID for reference
            }
        G.add_node(idx, **props)

    # Add edges with attributes using indexes
    for u_orig, v_orig in edge_distance.keys():
        u_idx = node_to_index[u_orig]
        v_idx = node_to_index[v_orig]
        distance = edge_distance[(u_orig, v_orig)]
        time_val = edge_time.get((u_orig, v_orig), 0)
        G.add_edge(u_idx, v_idx, distance=distance, time=time_val, terrain_factor=1.0)

    return G
