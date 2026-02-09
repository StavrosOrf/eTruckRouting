"""
State space definition and state observation for the truck routing environment.

Handles:
- State space initialization (observation_space)
- State computation from environment
- State representation and normalization

This flattened state space mirrors the GNN state representation,
including all truck, delivery, and charger features.
"""

import numpy as np
from gymnasium import spaces
from typing import Optional, Dict, Any, Set


class StateSpace:
    """
    Manages state space definition and state computation for the environment.
    
    State structure (flattened):
    - All trucks: num_trucks * 14 features (zeros for completed/failed trucks)
    - All delivery nodes: num_stops * 3 features (zeros for delivered nodes)
    - All chargers: num_charging_nodes * 4 features
    - Global features: 2 features (global_time, active_truck_id)
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

        # Feature dimensions matching GNN state space
        self._truck_feature_dim = 14  # Per truck
        self._delivery_feature_dim = 5  # Per delivery node (includes energy/time to reach)
        self._charger_feature_dim = 6  # Per charger (includes energy/time to reach)
        self._global_feature_dim = 2  # Global state
        
        # Total state size
        self._total_size = (
            num_trucks * self._truck_feature_dim +
            num_stops * self._delivery_feature_dim +
            num_charging_nodes * self._charger_feature_dim +
            self._global_feature_dim
        )

        # Define observation space
        # Most features are normalized between 0 and 1
        # However, charger node_id can be > 1 since it's normalized by num_charging_nodes
        # but the actual node IDs can be much larger (matching GNN behavior)
        self.observation_space = spaces.Box(
            low=0.0,
            high=np.inf,  # Allow values > 1 for node IDs
            shape=(self._total_size,),
            dtype=np.float32,
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
        charging_station=None,
    ) -> np.ndarray:
        """
        Generate flattened observation/state array matching GNN state representation.

        Args:
            trucks: List of all truck objects
            active_truck_id: ID of the active truck (None if episode over)
            transport_graph: Transportation graph with distance calculations
            charging_nodes: List of charging station nodes
            truck_states: Dictionary mapping truck_id to state
            event_queue: Priority queue of pending events
            global_clock: Current simulation time
            charging_station: ChargingStation object for charger features

        Returns:
            Flattened state array with all truck, delivery, and charger features
        """
        if active_truck_id is None:
            # Return zeros if no active truck
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)

        # Initialize state vector
        state = np.zeros(self._total_size, dtype=np.float32)
        
        # Calculate offsets for each section
        truck_offset = 0
        delivery_offset = truck_offset + self.num_trucks * self._truck_feature_dim
        charger_offset = delivery_offset + self.num_stops * self._delivery_feature_dim
        global_offset = charger_offset + self.num_charging_nodes * self._charger_feature_dim
        
        # --- 1. Truck Features ---
        # Track which delivery nodes have been delivered
        delivered_nodes = self._get_delivered_nodes(trucks)
        
        for truck_id, truck in enumerate(trucks):
            if truck_id >= self.num_trucks:
                break
                
            start_idx = truck_offset + truck_id * self._truck_feature_dim
            
            # Skip completed/failed trucks (leave as zeros)
            if truck.failed or truck.is_complete:
                continue
            
            # Get truck features (14 features)
            truck_features = self._get_truck_features(
                truck, transport_graph, truck_states, global_clock, charging_station
            )
            state[start_idx:start_idx + self._truck_feature_dim] = truck_features
        
        # --- 2. Delivery Node Features ---
        # Collect all unique delivery nodes across all trucks
        # NOTE: delivery_sequence includes depot, so skip first node
        # Also filter out charging nodes (they should not be in delivery features)
        all_delivery_nodes = set()
        for truck in trucks:
            # Skip depot (first node in sequence) and filter out chargers
            for node_id in truck.delivery_sequence[1:]:
                if node_id not in charging_nodes:
                    all_delivery_nodes.add(node_id)
        
        # Sort delivery nodes for consistent ordering
        sorted_delivery_nodes = sorted(all_delivery_nodes)[:self.num_stops]
        
        for idx, node_id in enumerate(sorted_delivery_nodes):
            if idx >= self.num_stops:
                break
                
            start_idx = delivery_offset + idx * self._delivery_feature_dim
            
            # Skip delivered nodes (leave as zeros)
            if node_id in delivered_nodes:
                continue
            
            # Get delivery features (5 features)
            active_truck = trucks[active_truck_id] if active_truck_id is not None else None
            delivery_features = self._get_delivery_features(
                node_id, trucks, transport_graph, active_truck
            )
            state[start_idx:start_idx + self._delivery_feature_dim] = delivery_features
        
        # --- 3. Charger Features ---
        for idx, charger_node_id in enumerate(sorted(charging_nodes)):
            if idx >= self.num_charging_nodes:
                break
                
            start_idx = charger_offset + idx * self._charger_feature_dim
            
            # Get charger features (6 features)
            active_truck = trucks[active_truck_id] if active_truck_id is not None else None
            charger_features = self._get_charger_features(
                charger_node_id, charging_station, transport_graph, active_truck
            )
            state[start_idx:start_idx + self._charger_feature_dim] = charger_features
        
        # --- 4. Global Features ---
        state[global_offset] = global_clock / self.max_time  # Normalized global time
        state[global_offset + 1] = active_truck_id / self.num_trucks  # Normalized active truck ID
        
        return state
    
    def _get_delivered_nodes(self, trucks: list) -> Set[int]:
        """
        Track which delivery nodes have been fully delivered.
        A node is delivered if no active trucks have it in remaining deliveries.
        """
        # Build mapping of node_id -> set of trucks that still need to deliver
        node_pending_trucks = {}
        for truck in trucks:
            if truck.failed or truck.is_complete:
                continue
            for delivery_node in truck.get_remaining_deliveries():
                if delivery_node not in node_pending_trucks:
                    node_pending_trucks[delivery_node] = set()
                node_pending_trucks[delivery_node].add(truck.truck_id)
        
        # Collect all delivery nodes ever assigned
        all_delivery_nodes_ever = set()
        for truck in trucks:
            all_delivery_nodes_ever.update(truck.delivery_sequence[1:])
        
        # Nodes not in node_pending_trucks are fully delivered
        delivered_nodes = set()
        for node_id in all_delivery_nodes_ever:
            if node_id not in node_pending_trucks:
                delivered_nodes.add(node_id)
        
        return delivered_nodes
    
    def _get_truck_features(
        self,
        truck,
        transport_graph,
        truck_states: Dict,
        global_clock: float,
        charging_station,
    ) -> np.ndarray:
        """
        Extract 14 features for a truck (matching GNN representation).
        
        Features (all normalized to [0, 1]):
        [0]: Node type (always 0/3 for truck)
        [1]: Current position (normalized by num_nodes)
        [2]: Battery level (normalized by capacity)
        [3]: Battery percentage (0-1)
        [4-7]: State one-hot (ready, routing, waiting_to_charge, charging)
        [8]: Deliveries completed (normalized by total)
        [9]: Deliveries remaining (normalized by total)
        [10]: Time elapsed (normalized by max_time)
        [11]: Distance traveled (normalized by 1000)
        [12]: Time to destination (normalized by max_time)
        """
        features = np.zeros(14, dtype=np.float32)
        
        # [0] Node type
        features[0] = 0.0 / 3.0  # Truck = 0, normalized by 3 node types
        
        # [1] Current position
        num_nodes = transport_graph.num_nodes
        features[1] = truck.current_node / num_nodes if num_nodes > 0 else 0.0
        
        # [2-3] Battery
        features[2] = truck.current_battery / truck.battery_capacity
        features[3] = truck.get_battery_percentage() / 100.0
        
        # [4-7] State one-hot (ready, routing, waiting_to_charge, charging)
        if truck.is_charging:
            features[7] = 1.0  # charging
        elif charging_station and truck.current_node in charging_station.charger_waitlist and \
             truck.truck_id in charging_station.charger_waitlist[truck.current_node]:
            features[6] = 1.0  # waiting_to_charge
        elif truck.route_destination is not None and truck.route_arrival_time is not None:
            features[5] = 1.0  # routing
        else:
            features[4] = 1.0  # ready
        
        # [8-9] Deliveries
        total_deliveries = len(truck.delivery_sequence) - 1  # Exclude depot
        deliveries_remaining = len(truck.get_remaining_deliveries())
        is_flexible = bool(getattr(truck, "enable_flexible_delivery_order", False))
        if is_flexible:
            deliveries_done = min(len(truck.delivered_nodes), total_deliveries)
        else:
            deliveries_done = truck.current_sequence_index
        features[8] = deliveries_done / total_deliveries if total_deliveries > 0 else 0.0
        features[9] = deliveries_remaining / total_deliveries if total_deliveries > 0 else 0.0
        
        # [10] Time elapsed
        features[10] = truck.total_time_elapsed / self.max_time
        
        # [11] Distance traveled - normalize by 1000 to match GNN
        features[11] = truck.total_distance_traveled / 1000.0
        
        # [12] Time to destination
        if not is_flexible and truck.route_arrival_time is not None and truck.route_destination is not None:
            features[12] = max(0.0, truck.route_arrival_time - global_clock) / self.max_time
        
        return features
    
    def _get_delivery_features(
        self,
        node_id: int,
        trucks: list,
        transport_graph,
        active_truck=None,
    ) -> np.ndarray:
        """
        Extract 5 features for a delivery node.
        
        Features (all normalized):
        [0]: Node type (always 1/3 for delivery)
        [1]: Node ID (normalized by num_nodes)
        [2]: Delivery sequence index (normalized by num_stops)
        [3]: Energy required to reach from active truck (normalized by 1000)
        [4]: Time required to reach from active truck (normalized by max_time)
        """
        features = np.zeros(5, dtype=np.float32)
        
        # [0] Node type
        features[0] = 1.0 / 3.0  # Delivery = 1, normalized by 3 node types
        
        # [1] Node ID
        num_nodes = transport_graph.num_nodes
        features[1] = node_id / num_nodes if num_nodes > 0 else 0.0
        
        # [2] Delivery sequence index - minimum position across all active trucks
        min_index = float('inf')
        for truck in trucks:
            if truck.failed or truck.is_complete:
                continue
            remaining_deliveries = truck.get_remaining_deliveries()
            if node_id in remaining_deliveries:
                position = remaining_deliveries.index(node_id) + 1
                min_index = min(min_index, position)
        
        delivery_sequence_index = int(min_index) if min_index != float('inf') else 0
        features[2] = delivery_sequence_index / self.num_stops if self.num_stops > 0 else 0.0
        
        # [3-4] Energy and time from active truck (if available)
        if active_truck is not None:
            energy = transport_graph.get_path_energy(active_truck.current_node, node_id)
            time = transport_graph.get_time_distance(active_truck.current_node, node_id)
            features[3] = energy / 1000.0 if not np.isinf(energy) else 0.0
            features[4] = time / self.max_time if not np.isinf(time) else 0.0
        
        return features
    
    def _get_charger_features(
        self,
        node_id: int,
        charging_station,
        transport_graph=None,
        active_truck=None,
    ) -> np.ndarray:
        """
        Extract 6 features for a charger node.
        
        Features (all normalized):
        [0]: Node type (always 2/3 for charger)
        [1]: Node ID (normalized by num_charging_nodes - matches GNN)
        [2]: Occupancy rate (current/capacity)
        [3]: Queue length (normalized by num_trucks)
        [4]: Energy required to reach from active truck (normalized by 1000)
        [5]: Time required to reach from active truck (normalized by max_time)
        """
        features = np.zeros(6, dtype=np.float32)
        
        # [0] Node type
        features[0] = 2.0 / 3.0  # Charger = 2, normalized by 3 node types
        
        # [1] Node ID - normalize by num_charging_nodes to match GNN state space
        features[1] = node_id / self.num_charging_nodes if self.num_charging_nodes > 0 else 0.0
        
        # [2-3] Charger state
        if charging_station:
            charger_capacity = charging_station.charger_capacity.get(node_id, 1)
            charger_occupancy_list = charging_station.charger_occupancy.get(node_id, [])
            charger_queue = charging_station.charger_waitlist.get(node_id, [])
            
            features[2] = len(charger_occupancy_list) / charger_capacity if charger_capacity > 0 else 0.0
            features[3] = len(charger_queue) / self.num_trucks if self.num_trucks > 0 else 0.0
        
        # [4-5] Energy and time from active truck (if available)
        if active_truck is not None and transport_graph is not None:
            energy = transport_graph.get_path_energy(active_truck.current_node, node_id)
            time = transport_graph.get_time_distance(active_truck.current_node, node_id)
            features[4] = energy / 1000.0 if not np.isinf(energy) else 0.0
            features[5] = time / self.max_time if not np.isinf(time) else 0.0
        
        return features

    @property
    def state_shape(self) -> tuple:
        """Get the shape of the state array."""
        return self.observation_space.shape

    @property
    def state_size(self) -> int:
        """Get the total size of the state vector."""
        return self.observation_space.shape[0]
    
    def get_feature_info(self) -> Dict[str, Any]:
        """
        Get information about the state features and their positions.
        
        Returns:
            Dictionary with feature dimensions and offsets
        """
        truck_offset = 0
        delivery_offset = truck_offset + self.num_trucks * self._truck_feature_dim
        charger_offset = delivery_offset + self.num_stops * self._delivery_feature_dim
        global_offset = charger_offset + self.num_charging_nodes * self._charger_feature_dim
        
        return {
            'total_size': self._total_size,
            'truck_features': {
                'count': self.num_trucks,
                'features_per_truck': self._truck_feature_dim,
                'offset': truck_offset,
                'size': self.num_trucks * self._truck_feature_dim,
                'feature_names': [
                    'node_type', 'current_position', 'battery_level', 'battery_percentage',
                    'state_ready', 'state_routing', 'state_waiting', 'state_charging',
                    'deliveries_done', 'deliveries_remaining', 'time_elapsed',
                    'distance_traveled', 'time_to_destination', 'reserved'
                ],
            },
            'delivery_features': {
                'count': self.num_stops,
                'features_per_delivery': self._delivery_feature_dim,
                'offset': delivery_offset,
                'size': self.num_stops * self._delivery_feature_dim,
                'feature_names': ['node_type', 'node_id', 'delivery_sequence_index', 'energy_to_reach', 'time_to_reach'],
            },
            'charger_features': {
                'count': self.num_charging_nodes,
                'features_per_charger': self._charger_feature_dim,
                'offset': charger_offset,
                'size': self.num_charging_nodes * self._charger_feature_dim,
                'feature_names': ['node_type', 'node_id', 'occupancy_rate', 'queue_length', 'energy_to_reach', 'time_to_reach'],
            },
            'global_features': {
                'count': 1,
                'features': self._global_feature_dim,
                'offset': global_offset,
                'size': self._global_feature_dim,
                'feature_names': ['global_clock', 'active_truck_id'],
            },
        }


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
