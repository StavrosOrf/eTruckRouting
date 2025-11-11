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
    """Map charger type strings to standardized names."""
    charger_type_lower = charger_type_str.lower()
    if "level2" in charger_type_lower or "level 2" in charger_type_lower:
        return "Level2"
    elif (
        "dcfast" in charger_type_lower
        or "dc fast" in charger_type_lower
        or "dc_fast" in charger_type_lower
    ):
        return "DCFast"
    else:
        return charger_type_str


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
    network_config = config.get("network", {})
    data_path = network_config.get("data_path", "data/")
    energy_file = network_config.get("shortest_path_energy_file", "shortest_path_energy_dict.json")
    time_file = network_config.get("shortest_path_time_file", "shortest_path_time_dict.json")
    station_file = network_config.get("station_info_file", "station_info_dict.json")

    # Get the directory where this file is located (truck_env/utils)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Navigate up from utils to truck_env
    truck_env_dir = os.path.dirname(current_dir)
    
    # Construct data directory path
    # If data_path starts with "truck_env/", use it as-is relative to project root
    if data_path.startswith("truck_env/"):
        # Relative to project root
        project_root = os.path.dirname(truck_env_dir)
        data_dir = os.path.join(project_root, data_path.rstrip('/'))
    else:
        # Relative to truck_env directory
        data_dir = os.path.join(truck_env_dir, data_path.rstrip('/'))

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
