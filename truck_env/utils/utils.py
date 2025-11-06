"""
Utility functions for the simple truck environment.
"""
import os
import pickle
import networkx as nx
from typing import Dict, Tuple, Any


def read_file(filename: str) -> Any:
    """Read a pickle file."""
    with open(filename, "rb") as f:
        return pickle.load(f)


def map_charger_type(charger_type_str: str) -> str:
    """Map charger type strings to standardized names."""
    charger_type_lower = charger_type_str.lower()
    if "level2" in charger_type_lower or "level 2" in charger_type_lower:
        return "Level2"
    elif "dcfast" in charger_type_lower or "dc fast" in charger_type_lower or "dc_fast" in charger_type_lower:
        return "DCFast"
    else:
        return charger_type_str


def get_graph() -> nx.DiGraph:
    """
    Load and build the road network graph from pickle files.
    
    Returns:
        NetworkX directed graph with nodes, edges, and charging station information
    """
    # Get the directory where this file is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, 'data')
    
    edge_distance_file = os.path.join(data_dir, 'shortest_path_energy_dict.pkl')
    edge_time_file = os.path.join(data_dir, 'shortest_path_time_dict.pkl')
    chargers_file = os.path.join(data_dir, 'station_info_dict.pkl')
    
    # Load data files
    edge_distance = read_file(edge_distance_file)
    edge_time = read_file(edge_time_file)
    chargers = read_file(chargers_file)

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
                "original_id": index_to_node[idx],  # Store original node ID for reference
            }
        else:
            props = {
                "has_charger": False,
                "charger_type": None,
                "original_id": index_to_node[idx],  # Store original node ID for reference
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


def discharge_function(current_charge: float, distance: float = 1.0) -> float:
    """
    Standard discharge function for battery consumption based on distance.
    
    Args:
        current_charge: Current battery level (unused, for compatibility)
        distance: Distance traveled in km
        
    Returns:
        Battery consumed in kWh
    """
    return 0.2 * distance


def charge_function(current_charge: float, time: float = 1.0) -> float:
    """
    Standard charge function for battery charging based on time.
    
    Args:
        current_charge: Current battery level (unused, for compatibility)
        time: Charging time in hours
        
    Returns:
        Battery charged in kWh
    """
    return 0.8 * time
