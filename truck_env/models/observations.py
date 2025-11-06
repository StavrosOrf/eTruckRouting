"""Observation generation for the truck routing environment."""

import numpy as np
from typing import Optional, Dict, Any


def get_observation(
    trucks: list,
    active_truck_id: Optional[int],
    transport_graph,
    charging_nodes: list,
    truck_states: Dict,
    event_queue: list,
    observation_space_shape: tuple,
    global_clock: float,
) -> np.ndarray:
    """
    Generate observation array for the active truck.

    Args:
        trucks: List of all truck objects
        active_truck_id: ID of the active truck
        transport_graph: Transportation graph with distance calculations
        charging_nodes: List of charging station nodes
        truck_states: Dictionary mapping truck_id to state
        event_queue: Priority queue of pending events
        observation_space_shape: Shape of the observation space
        global_clock: Current simulation time

    Returns:
        Observation array with shape matching observation_space_shape
    """
    if active_truck_id is None:
        # Return zeros if no active truck
        return np.zeros(observation_space_shape[0], dtype=np.float64)

    truck = trucks[active_truck_id]

    # Normalize node IDs
    max_node_id = float(transport_graph.num_nodes)
    current_node_norm = truck.current_node / max_node_id
    next_delivery = truck.get_next_delivery_target()
    next_delivery_norm = (
        (next_delivery / max_node_id) if next_delivery is not None else 0.0
    )

    # Find nearest charger
    nearest_charger_dist = min(
        transport_graph.get_distance(truck.current_node, charger)
        for charger in charging_nodes
    )

    # Check if can reach next delivery
    can_reach_next = 0.0
    if next_delivery is not None:
        dist_to_next = transport_graph.get_distance(
            truck.current_node, next_delivery
        )
        can_reach_next = (
            1.0 if truck.can_reach_node(next_delivery, dist_to_next) else 0.0
        )

    # Count active trucks
    active_trucks = sum(
        1
        for state in truck_states.values()
        if state not in ["complete", "failed"]
    )

    # Count pending events
    events_pending = len(event_queue)

    obs = np.array(
        [
            current_node_norm,
            next_delivery_norm,
            truck.current_battery,
            truck.get_battery_percentage(),
            float(truck.is_charging),
            float(len(truck.get_remaining_deliveries())),
            nearest_charger_dist,
            can_reach_next,
            truck.total_time_elapsed,
            truck.total_distance_traveled,
            global_clock,
            float(active_trucks),
            float(events_pending),
        ],
        dtype=np.float64,
    )

    return obs


def action_to_string(
    action: int,
    num_charging_nodes: int,
    num_navigation_actions: int,
    charging_nodes: list,
) -> str:
    """
    Convert action integer to human-readable string.

    Args:
        action: Action index
        num_charging_nodes: Number of charging stations
        num_navigation_actions: Number of navigation actions
        charging_nodes: List of charging station nodes

    Returns:
        Human-readable action description
    """
    if action < num_charging_nodes:
        node = charging_nodes[action]
        return f"Go to charger @ node {node}"
    elif action == num_charging_nodes:
        return "Go to next delivery"
    else:
        charge_idx = action - num_navigation_actions
        hours = charge_idx + 1
        return f"Charge for {hours}h"
