"""Loaders for trucks and other entities in the environment."""

import numpy as np
from truck_env.models.truck import Truck


def create_truck(
    truck_id: int,
    transport_graph,
    config: dict,
    num_stops: int,
    min_hop_distance: float,
    max_hop_distance: float,
    charging_nodes: list,
) -> tuple:
    """
    Create a new truck with random delivery sequence.

    Args:
        truck_id: Unique identifier for the truck
        transport_graph: Transportation graph with nodes and edges
        config: Configuration dictionary with truck settings
        num_stops: Number of delivery stops
        min_hop_distance: Minimum distance between stops
        max_hop_distance: Maximum distance between stops
        charging_nodes: List of charging station nodes

    Returns:
        Tuple of (truck, delivery_sequence, start_node)
    """
    # Select random start node (avoid charging nodes and sink nodes)
    all_nodes = transport_graph.get_all_nodes()
    
    # Filter out charging nodes
    non_charging_nodes = [n for n in all_nodes if n not in charging_nodes]
    
    # Filter out sink nodes (nodes with no outgoing edges)
    graph = transport_graph.graph
    valid_start_nodes = [n for n in non_charging_nodes if graph.out_degree(n) > 0]
    
    # If no valid nodes found, use any non-charging node as fallback
    if not valid_start_nodes:
        valid_start_nodes = non_charging_nodes
    
    if not valid_start_nodes:
        # Last resort: use any node
        valid_start_nodes = all_nodes
    
    start_node = np.random.choice(valid_start_nodes)

    # Generate delivery sequence
    delivery_sequence = transport_graph.generate_delivery_sequence(
        start_node=start_node,
        num_stops=num_stops,
        min_hop_distance=min_hop_distance,
        max_hop_distance=max_hop_distance,
        exclude_charging_nodes=True,
    )

    # Get truck specifications (single type)
    truck_config = config["truck"]
    battery_capacity = truck_config["battery_capacity"]
    base_speed = truck_config["base_speed"]
    # Determine initial battery
    initial_battery_setting = truck_config["initial_battery"]
    if initial_battery_setting == "full":
        initial_battery = battery_capacity
    elif initial_battery_setting == "random":
        initial_battery = np.random.uniform(0.3, 1.0) * battery_capacity
    elif isinstance(initial_battery_setting, (int, float)):
        initial_battery = (initial_battery_setting / 100.0) * battery_capacity
    else:
        initial_battery = battery_capacity

    # Create truck
    truck = Truck(
        truck_id=truck_id,
        truck_type="electric",  # Single truck type
        delivery_sequence=delivery_sequence,
        initial_battery=initial_battery,
        battery_capacity=battery_capacity,
        base_speed=base_speed,
    )

    return truck, delivery_sequence, start_node
