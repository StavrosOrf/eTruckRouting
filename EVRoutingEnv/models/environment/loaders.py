"""Loaders for trucks and other entities in the environment."""

import numpy as np
from EVRoutingEnv.models.core.truck import Truck


def _validate_delivery_sequence_feasibility(
    delivery_sequence: list,
    battery_capacity: float,
    transport_graph,
    charging_nodes: list,
) -> bool:
    """
    Validate that a delivery sequence is feasible given battery capacity.
    
    Simulates the actual delivery sequence with battery consumption to ensure:
    1. Each leg can be completed with available battery
    2. From any point, truck can reach either next delivery or a charger
    3. Cumulative battery consumption doesn't create dead-ends
    
    Args:
        delivery_sequence: List of nodes in delivery order [start, del1, del2, ...]
        battery_capacity: Maximum battery capacity in kWh
        transport_graph: Transportation graph
        charging_nodes: List of available charging station nodes
        
    Returns:
        True if sequence is feasible, False otherwise
    """
    # Simulate the delivery sequence with cumulative battery tracking
    current_battery = battery_capacity  # Start with full battery
    
    for i in range(len(delivery_sequence) - 1):
        from_node = int(delivery_sequence[i])
        to_node = int(delivery_sequence[i + 1])
        
        # Try to reach destination with current battery
        direct_energy = transport_graph.get_path_energy(from_node, to_node)
        
        # Keep 5% battery buffer for safety
        min_buffer = 0.05 * battery_capacity
        
        if direct_energy <= current_battery - min_buffer:
            # Can reach directly
            current_battery -= direct_energy
            continue
        
        # Cannot reach directly - need to charge
        # Find if we can reach a charger, charge, then reach destination
        can_reach_via_charging = False
        
        for charger in charging_nodes:
            charger_int = int(charger)
            energy_to_charger = transport_graph.get_path_energy(from_node, charger_int)
            
            # Can we reach this charger with current battery?
            if energy_to_charger > current_battery - min_buffer:
                continue
            
            # After charging to 95%, can we reach destination?
            energy_from_charger = transport_graph.get_path_energy(charger_int, to_node)
            charged_battery = battery_capacity * 0.95
            
            if energy_from_charger <= charged_battery - min_buffer:
                # Yes! This works
                current_battery = charged_battery - energy_from_charger
                can_reach_via_charging = True
                break
            
            # Try via second charger
            for charger2 in charging_nodes:
                if charger == charger2:
                    continue
                charger2_int = int(charger2)
                
                energy_between = transport_graph.get_path_energy(charger_int, charger2_int)
                if energy_between > charged_battery - min_buffer:
                    continue
                
                energy_to_dest = transport_graph.get_path_energy(charger2_int, to_node)
                if energy_to_dest <= charged_battery - min_buffer:
                    # Can reach via two chargers
                    current_battery = charged_battery - energy_to_dest
                    can_reach_via_charging = True
                    break
            
            if can_reach_via_charging:
                break
        
        if not can_reach_via_charging:
            # Cannot complete this leg even with charging
            return False
    
    return True


def _is_leg_feasible(
    from_node: int,
    to_node: int,
    battery_capacity: float,
    transport_graph,
    charging_nodes: list,
) -> bool:
    """
    Check if travel from from_node to to_node is feasible with given battery capacity.
    
    A leg is feasible if:
    1. Direct path exists and requires <= battery_capacity, OR
    2. There exists at least one charger that can be reached from from_node
       and from which to_node can be reached, both within battery_capacity
    
    Args:
        from_node: Starting node
        to_node: Destination node
        battery_capacity: Battery capacity in kWh
        transport_graph: Transportation graph
        charging_nodes: List of charging station nodes
        
    Returns:
        True if leg is feasible, False otherwise
    """
    # Check direct path
    direct_energy = transport_graph.get_path_energy(from_node, to_node)
    if direct_energy != float('inf') and direct_energy <= battery_capacity:
        return True
    
    # Check via single charger
    for charger in charging_nodes:
        energy_to_charger = transport_graph.get_path_energy(from_node, int(charger))
        energy_from_charger = transport_graph.get_path_energy(int(charger), to_node)
        
        # Both legs must be within battery capacity (assuming full charge at charger)
        if (energy_to_charger != float('inf') and energy_to_charger <= battery_capacity and
            energy_from_charger != float('inf') and energy_from_charger <= battery_capacity):
            return True
    
    # Check via two chargers (for very long distances)
    for charger1 in charging_nodes:
        energy_to_c1 = transport_graph.get_path_energy(from_node, int(charger1))
        if energy_to_c1 == float('inf') or energy_to_c1 > battery_capacity:
            continue
            
        for charger2 in charging_nodes:
            if charger1 == charger2:
                continue
                
            energy_c1_to_c2 = transport_graph.get_path_energy(int(charger1), int(charger2))
            energy_c2_to_dest = transport_graph.get_path_energy(int(charger2), to_node)
            
            if (energy_c1_to_c2 != float('inf') and energy_c1_to_c2 <= battery_capacity and
                energy_c2_to_dest != float('inf') and energy_c2_to_dest <= battery_capacity):
                return True
    
    # No feasible path found
    return False


def create_truck(
    truck_id: int,
    transport_graph,
    config: dict,
    num_stops: int,
    min_hop_distance: float,
    max_hop_distance: float,
    charging_nodes: list,
    enable_flexible_delivery_order: bool = False,
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
        enable_flexible_delivery_order: If True, allow flexible delivery order selection

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

    # Generate delivery sequence with strict feasibility checks
    truck_config = config["truck"]
    battery_capacity = truck_config["battery_capacity"]
    max_tries = 100
    
    for attempt in range(max_tries):
        delivery_sequence = transport_graph.generate_delivery_sequence(
            start_node=start_node,
            num_stops=num_stops,
            min_hop_distance=min_hop_distance,
            max_hop_distance=max_hop_distance,
            exclude_charging_nodes=True,
        )
        
        # Check if sequence is feasible: every leg must have a valid path
        feasible = _validate_delivery_sequence_feasibility(
            delivery_sequence=delivery_sequence,
            battery_capacity=battery_capacity,
            transport_graph=transport_graph,
            charging_nodes=charging_nodes,
        )
        
        if feasible:
            break
    else:
        raise ValueError(
            f"Failed to generate a feasible delivery sequence after {max_tries} attempts. "
            f"Start node: {start_node}, Battery capacity: {battery_capacity} kWh. "
            "Consider increasing battery_capacity or adjusting hop distance constraints."
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
        enable_flexible_delivery_order=enable_flexible_delivery_order,
    )

    return truck, delivery_sequence, start_node
