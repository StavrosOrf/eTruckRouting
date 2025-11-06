"""
State space definition and state observation for the truck routing environment.

Handles:
- State space initialization (observation_space)
- State computation from environment
- State representation and normalization
"""

import numpy as np
from gymnasium import spaces
from typing import Optional, Dict, Any


class StateSpace:
    """
    Manages state space definition and state computation for the environment.
    """

    def __init__(
        self,
        num_trucks: int,
        num_stops: int,
        max_time: float,
        num_charging_nodes: int,
    ):
        """
        Initialize state space.

        Args:
            num_trucks: Number of trucks in the environment
            num_stops: Maximum number of stops per truck
            max_time: Maximum simulation time (hours)
            num_charging_nodes: Number of charging stations
        """
        self.num_trucks = num_trucks
        self.num_stops = num_stops
        self.max_time = max_time
        self.num_charging_nodes = num_charging_nodes

        # Define observation space - Box for current active truck + global state
        # Observation: [truck_state (10), global_time (1), active_trucks (1), events_pending (1)]
        self.observation_space = spaces.Box(
            low=np.array(
                [
                    0.0,  # current_node (normalized)
                    0.0,  # next_delivery_node (normalized)
                    0.0,  # battery_level
                    0.0,  # battery_percentage
                    0.0,  # is_charging
                    0.0,  # deliveries_remaining
                    0.0,  # nearest_charger_distance
                    0.0,  # can_reach_next_delivery
                    0.0,  # time_elapsed (truck)
                    0.0,  # distance_traveled
                    0.0,  # global_time
                    0.0,  # active_trucks
                    0.0,  # events_pending
                ]
            ),
            high=np.array(
                [
                    1.0,  # current_node (normalized)
                    1.0,  # next_delivery_node (normalized)
                    500.0,  # battery_level (kWh)
                    100.0,  # battery_percentage
                    1.0,  # is_charging
                    float(num_stops),  # deliveries_remaining
                    1000.0,  # nearest_charger_distance (km)
                    1.0,  # can_reach_next_delivery
                    1000.0,  # time_elapsed (hours)
                    5000.0,  # distance_traveled (km)
                    max_time,  # global_time
                    float(num_trucks),  # active_trucks
                    100.0,  # events_pending
                ]
            ),
            dtype=np.float64,
        )

    def get_state(
        self,
        trucks: list,
        active_truck_id: Optional[int],
        transport_graph,
        charging_nodes: list,
        truck_states: Dict,
        event_queue: list,
        global_clock: float,
    ) -> np.ndarray:
        """
        Generate observation/state array for the active truck.

        Args:
            trucks: List of all truck objects
            active_truck_id: ID of the active truck (None if episode over)
            transport_graph: Transportation graph with distance calculations
            charging_nodes: List of charging station nodes
            truck_states: Dictionary mapping truck_id to state
            event_queue: Priority queue of pending events
            global_clock: Current simulation time

        Returns:
            State array with shape matching observation_space shape
        """
        if active_truck_id is None:
            # Return zeros if no active truck
            return np.zeros(self.observation_space.shape[0], dtype=np.float64)

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

        state = np.array(
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

        return state

    def get_state_dict(
        self,
        trucks: list,
        active_truck_id: Optional[int],
        transport_graph,
        charging_nodes: list,
        truck_states: Dict,
        event_queue: list,
        global_clock: float,
    ) -> Dict[str, Any]:
        """
        Generate detailed state dictionary with named components.

        Args:
            trucks: List of all truck objects
            active_truck_id: ID of the active truck
            transport_graph: Transportation graph
            charging_nodes: List of charging stations
            truck_states: Truck state dictionary
            event_queue: Pending events queue
            global_clock: Current simulation time

        Returns:
            Dictionary with named state components
        """
        if active_truck_id is None:
            return {
                "current_node": 0.0,
                "next_delivery": 0.0,
                "battery": 0.0,
                "battery_pct": 0.0,
                "is_charging": False,
                "deliveries_remaining": 0,
                "nearest_charger_dist": 0.0,
                "can_reach_next": False,
                "truck_time": 0.0,
                "truck_distance": 0.0,
                "global_time": global_clock,
                "active_trucks": 0,
                "pending_events": 0,
            }

        truck = trucks[active_truck_id]
        max_node_id = float(transport_graph.num_nodes)
        next_delivery = truck.get_next_delivery_target()

        nearest_charger_dist = min(
            transport_graph.get_distance(truck.current_node, charger)
            for charger in charging_nodes
        )

        can_reach_next = False
        if next_delivery is not None:
            dist_to_next = transport_graph.get_distance(
                truck.current_node, next_delivery
            )
            can_reach_next = truck.can_reach_node(next_delivery, dist_to_next)

        active_trucks = sum(
            1
            for state in truck_states.values()
            if state not in ["complete", "failed"]
        )

        return {
            "current_node": truck.current_node / max_node_id,
            "next_delivery": (next_delivery / max_node_id)
            if next_delivery is not None
            else 0.0,
            "battery": truck.current_battery,
            "battery_pct": truck.get_battery_percentage(),
            "is_charging": truck.is_charging,
            "deliveries_remaining": len(truck.get_remaining_deliveries()),
            "nearest_charger_dist": nearest_charger_dist,
            "can_reach_next": can_reach_next,
            "truck_time": truck.total_time_elapsed,
            "truck_distance": truck.total_distance_traveled,
            "global_time": global_clock,
            "active_trucks": active_trucks,
            "pending_events": len(event_queue),
        }

    @property
    def state_shape(self) -> tuple:
        """Get the shape of the state array."""
        return self.observation_space.shape

    @property
    def state_size(self) -> int:
        """Get the total size of the state vector."""
        return self.observation_space.shape[0]
