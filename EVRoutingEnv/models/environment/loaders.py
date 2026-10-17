"""Loaders for trucks and other entities in the environment."""


import math

import numpy as np

from EVRoutingEnv.models.core.truck import Truck


def _validate_delivery_sequence_feasibility(
    delivery_sequence: list,
    battery_capacity: float,
    transport_graph,
    charging_nodes: list,
    initial_battery: float | None = None,
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
        initial_battery: Starting energy; defaults to full capacity
        
    Returns:
        True if sequence is feasible, False otherwise
    """
    # Simulate the delivery sequence with cumulative battery tracking
    current_battery = battery_capacity if initial_battery is None else initial_battery

    def energy(origin, destination):
        value = float(transport_graph.get_path_energy(origin, destination))
        return value if math.isfinite(value) and value >= 0.0 else math.inf
    
    for i in range(len(delivery_sequence) - 1):
        from_node = int(delivery_sequence[i])
        to_node = int(delivery_sequence[i + 1])
        
        # Try to reach destination with current battery
        direct_energy = energy(from_node, to_node)
        
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
            energy_to_charger = energy(from_node, charger_int)
            
            # Can we reach this charger with current battery?
            if energy_to_charger > current_battery - min_buffer:
                continue
            
            # After charging to 95%, can we reach destination?
            energy_from_charger = energy(charger_int, to_node)
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
                
                energy_between = energy(charger_int, charger2_int)
                if energy_between > charged_battery - min_buffer:
                    continue
                
                energy_to_dest = energy(charger2_int, to_node)
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



def create_truck(
    truck_id: int,
    transport_graph,
    config: dict,
    num_stops: int,
    min_hop_distance: float,
    max_hop_distance: float,
    charging_nodes: list,
    enable_flexible_delivery_order: bool = False,
    rng: np.random.Generator | None = None,
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
        rng: Optional episode-scoped generator for reproducible instances

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
    if not valid_start_nodes:
        raise ValueError("cannot generate a delivery route on an empty graph")
    
    # Get truck specifications
    truck_config = config["truck"]
    battery_capacity = truck_config["battery_capacity"]
    initial_battery_setting = truck_config["initial_battery"]
    if initial_battery_setting == "full":
        initial_battery = battery_capacity
    elif initial_battery_setting == "random":
        random_source = np.random if rng is None else rng
        initial_battery = random_source.uniform(0.3, 1.0) * battery_capacity
    elif isinstance(initial_battery_setting, (int, float)) and not isinstance(
        initial_battery_setting, bool
    ):
        initial_battery = initial_battery_setting / 100.0 * battery_capacity
    else:
        raise ValueError("invalid truck.initial_battery setting")
    if not math.isfinite(initial_battery) or not 0 <= initial_battery <= battery_capacity:
        raise ValueError("truck.initial_battery must be in [0, 100]")

    def sequence_is_feasible(sequence):
        # Flexible routes also owe a depot return after the last service.
        route = sequence + [sequence[0]] if enable_flexible_delivery_order else sequence
        return _validate_delivery_sequence_feasibility(
            delivery_sequence=route,
            battery_capacity=battery_capacity,
            transport_graph=transport_graph,
            charging_nodes=charging_nodes,
            initial_battery=initial_battery,
        )
    
    # Progressively reduce max_hop_distance to guarantee finding a feasible sequence
    current_max_hop = max_hop_distance
    current_min_hop = min_hop_distance
    reduction_factor = 0.8
    min_acceptable_hop = 1.0  # Minimum hop distance in km (adjust as needed)
    
    # Shuffle valid start nodes to try different ones
    candidate_start_nodes = list(valid_start_nodes)
    if rng is None:
        np.random.shuffle(candidate_start_nodes)
    else:
        rng.shuffle(candidate_start_nodes)
    
    feasible = False
    delivery_sequence = None
    start_node = None
    
    while not feasible and current_max_hop >= min_acceptable_hop:
        # Try multiple start nodes with current hop distance constraints
        max_start_node_tries = 20
        max_sequence_tries_per_node = 5
        
        for start_node_attempt in range(max_start_node_tries):
            # Select a start node (cycle through candidates if we run out)
            start_node = candidate_start_nodes[start_node_attempt % len(candidate_start_nodes)]
            
            # Try to generate a feasible sequence from this start node
            for _sequence_attempt in range(max_sequence_tries_per_node):
                delivery_sequence = transport_graph.generate_delivery_sequence(
                    start_node=start_node,
                    num_stops=num_stops,
                    min_hop_distance=current_min_hop,
                    max_hop_distance=current_max_hop,
                    exclude_charging_nodes=True,
                    rng=rng,
                )
                
                # Check if sequence has the correct number of stops
                # delivery_sequence includes start node, so should have num_stops + 1 nodes
                if len(delivery_sequence) != num_stops + 1:
                    continue  # Try again
                
                # Check if sequence is feasible: every leg must have a valid path
                feasible = sequence_is_feasible(delivery_sequence)
                
                if feasible:
                    # Found a feasible sequence!
                    break
            
            if feasible:
                # Success - exit start node loop
                break
        
        if not feasible:
            # Reduce hop distance constraints and try again
            current_max_hop *= reduction_factor
            current_min_hop = min(current_min_hop, current_max_hop * 0.5)
            # Reshuffle start nodes for next iteration
            if rng is None:
                np.random.shuffle(candidate_start_nodes)
            else:
                rng.shuffle(candidate_start_nodes)
    
    if not feasible:
        # Last resort: generate any sequence with very small hops
        # This should almost always succeed as shorter distances are easier to charge for
        for start_node_attempt in range(len(candidate_start_nodes)):
            start_node = candidate_start_nodes[start_node_attempt]
            
            for _sequence_attempt in range(20):
                delivery_sequence = transport_graph.generate_delivery_sequence(
                    start_node=start_node,
                    num_stops=num_stops,
                    min_hop_distance=0.1,
                    max_hop_distance=min_acceptable_hop,
                    exclude_charging_nodes=True,
                    rng=rng,
                )
                
                # Check if sequence has the correct number of stops
                if len(delivery_sequence) != num_stops + 1:
                    continue  # Try again
                
                feasible = sequence_is_feasible(delivery_sequence)
                
                if feasible:
                    break
            
            if feasible:
                break

    if not feasible:
        raise ValueError(
            f"could not generate a battery-feasible route with {num_stops} stops"
        )

    # Get truck specifications (single type)
    truck_config = config["truck"]
    battery_capacity = truck_config["battery_capacity"]
    base_speed = truck_config["base_speed"]
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
