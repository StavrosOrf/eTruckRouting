"""
GNN State Representation for Truck Routing Environment.

This module provides PyTorch Geometric graph representations of the truck routing
environment state, suitable for Graph Neural Network (GNN) agents.

Simplified Graph Construction:
- **Nodes**: 
  - Active Trucks (current position, battery, state)
  - Undelivered Delivery Nodes (position)
  - All Charging Stations (position, occupancy, queue)
  
- **Edges** (only feasible connections based on truck state and battery):
  - When truck is READY:
    * Truck ↔ Next Delivery (if energy < current battery)
    * Truck ↔ All Chargers (if energy < current battery)
  - When truck is WAITING_TO_CHARGE or CHARGING:
    * Truck ↔ Current Charger only
  - When truck is ROUTING:
    * Truck ↔ Destination (Charger or Delivery node being routed to)
  - Charger ↔ Charger (always, if energy feasible)
  - Charger ↔ Delivery (always, if energy feasible)
  - Delivery ↔ Delivery (always, if energy feasible)

Note: 
- No depot nodes
- No padding - each node type has different number of features
- Completed/failed trucks are excluded
- Delivered nodes are excluded
- Only feasible edges (energy < battery capacity)

This enables GNNs to learn routing policies by reasoning over feasible actions.
"""

import torch
import numpy as np
from typing import Optional, Dict, Tuple, Set

from torch_geometric.data import Data


class GNNStateSpace:
    """
    Manages GNN-based state space for truck routing environment.
    
    Converts the event-driven environment into PyTorch Geometric Data graphs
    that can be used with Graph Neural Networks.
    """

    def __init__(
        self,
        num_trucks: int,
        num_stops: int,
        max_time: float,
        num_charging_nodes: int,
        max_nodes_in_graph: int = 500,
        device: str = "cpu",
    ):
        """
        Initialize GNN state space.

        Args:
            num_trucks: Number of trucks
            num_stops: Maximum stops per truck
            max_time: Maximum simulation time
            num_charging_nodes: Number of charging stations
            max_nodes_in_graph: Maximum nodes allowed in graph representation
            device: torch device ("cpu" or "cuda")
        """
        self.num_trucks = num_trucks
        self.num_stops = num_stops
        self.max_time = max_time
        self.num_charging_nodes = num_charging_nodes
        self.max_nodes_in_graph = max_nodes_in_graph
        self.device = device

        # Node type constants (no depot)
        self.NODE_TYPE_TRUCK = 0
        self.NODE_TYPE_DELIVERY = 1
        self.NODE_TYPE_CHARGER = 2

    def get_state_GNN(self, env) -> Data:
        """
        Convert environment state to PyTorch Geometric Data graph.

        Simplified Graph Structure:
        - Nodes: Trucks + Undelivered Deliveries + All Chargers
        - Edges: Only feasible connections based on truck state and battery
        - No padding: each node type has different feature dimensions

        Args:
            env: EventDrivenTruckEnv instance

        Returns:
            torch_geometric.data.Data graph
        """

        # Build node list and features
        node_list = []
        node_features_list = []
        
        # Track which delivery nodes have been delivered
        delivered_nodes: Set[int] = set()
        for truck in env.trucks:
            all_deliveries = set(truck.delivery_sequence[1:])  # Skip depot at index 0
            remaining_deliveries = set(truck.get_remaining_deliveries())
            delivered = all_deliveries - remaining_deliveries
            delivered_nodes.update(delivered)

        # 1. Add all truck nodes (excluding failed and completed trucks)
        truck_id_to_node_idx = {}  # truck_id -> node_idx
        
        for truck in env.trucks:
            # Skip failed and completed trucks
            if truck.failed or truck.is_complete:
                truck_id_to_node_idx[truck.truck_id] = None
                continue
            
            node_idx = len(node_list)
            truck_id_to_node_idx[truck.truck_id] = node_idx
            node_list.append(("truck", truck.truck_id))
            
            features = self._get_truck_node_features(truck, env)
            node_features_list.append(features)

        # 2. Add undelivered delivery nodes
        delivery_node_to_idx = {}  # delivery_node_id -> node_idx
        
        # Collect all delivery nodes from active trucks
        all_delivery_nodes = set()
        for truck in env.trucks:
            if truck.failed or truck.is_complete:
                continue
            all_delivery_nodes.update(truck.delivery_sequence[1:])  # Skip depot
        
        # Add only undelivered nodes
        for delivery_node_id in sorted(all_delivery_nodes):
            if delivery_node_id not in delivered_nodes:
                node_idx = len(node_list)
                delivery_node_to_idx[delivery_node_id] = node_idx
                node_list.append(("delivery", delivery_node_id))
                
                features = self._get_delivery_node_features(delivery_node_id, env)
                node_features_list.append(features)

        # 3. Add all charging station nodes
        charger_node_to_idx = {}  # charger_node_id -> node_idx
        
        for charger_node_id in env.charging_nodes:
            node_idx = len(node_list)
            charger_node_to_idx[charger_node_id] = node_idx
            node_list.append(("charger", charger_node_id))
            
            features = self._get_charger_node_features(charger_node_id, env)
            node_features_list.append(features)

        # Convert node features to tensor
        # NOTE: PyTorch Geometric Data objects require uniform feature dimensions.
        # We pad to max length but store node_list so GNN can identify node types
        # and ignore padding. For truly heterogeneous features, use HeteroData instead.
        # 
        # Feature counts: Truck=13, Delivery=2, Charger=5
        # Padding is added with zeros but node type (first feature) indicates meaningful features
        
        if node_features_list:
            max_features = max(len(f) for f in node_features_list)
            padded_features = []
            for features in node_features_list:
                padded = np.pad(features, (0, max_features - len(features)), 
                               mode='constant', constant_values=0)
                padded_features.append(padded)
            # Convert to numpy array first to avoid warning
            padded_features_array = np.array(padded_features, dtype=np.float32)
            x = torch.tensor(padded_features_array, dtype=torch.float32, device=self.device)
        else:
            # No nodes in graph
            x = torch.zeros((0, 13), dtype=torch.float32, device=self.device)

        # Get max truck battery capacity for feasibility checks
        max_battery_capacity = max(truck.battery_capacity for truck in env.trucks)

        # Build edge list and edge features
        edge_index_list = []
        edge_features_list = []

        # 4. Add truck edges based on state
        for truck in env.trucks:
            if truck.failed or truck.is_complete:
                continue
            
            truck_idx = truck_id_to_node_idx[truck.truck_id]
            if truck_idx is None:
                continue
            
            current_location = truck.current_node
            current_battery = truck.current_battery
            
            # Determine truck state
            if truck.is_charging or truck.truck_id in env.charging_station.charger_waitlist.get(current_location, []):
                # CHARGING or WAITING_TO_CHARGE: only connect to current charger
                if current_location in charger_node_to_idx:
                    charger_idx = charger_node_to_idx[current_location]
                    # Bidirectional edge with 0 energy/time (at charger)
                    edge_index_list.append([truck_idx, charger_idx])
                    edge_features_list.append([0.0, 0.0])
                    edge_index_list.append([charger_idx, truck_idx])
                    edge_features_list.append([0.0, 0.0])
                    
            elif truck.route_destination is None:
                # READY: connect to next delivery and all feasible chargers
                
                # Connect to next delivery (if exists and feasible)
                next_delivery = truck.get_next_delivery_target()
                if next_delivery is not None and next_delivery in delivery_node_to_idx:
                    delivery_idx = delivery_node_to_idx[next_delivery]
                    
                    # Check if already at delivery location (0 energy/time)
                    if next_delivery == current_location:
                        energy, time = 0.0, 0.0
                    else:
                        energy = env.transport_graph.get_path_energy(current_location, next_delivery)
                        time = env.transport_graph.get_time_distance(current_location, next_delivery)
                    
                    # Only add edge if energy is feasible (< current battery)
                    if energy < current_battery and not np.isinf(energy):
                        edge_index_list.append([truck_idx, delivery_idx])
                        edge_features_list.append([energy, time])
                        edge_index_list.append([delivery_idx, truck_idx])
                        edge_features_list.append([energy, time])
                
                # Connect to all chargers (if feasible with current battery)
                for charger_id, charger_idx in charger_node_to_idx.items():
                    # Skip self-loop (truck already at this charger)
                    if charger_id == current_location:
                        # Add 0-weight edge to current location
                        edge_index_list.append([truck_idx, charger_idx])
                        edge_features_list.append([0.0, 0.0])
                        edge_index_list.append([charger_idx, truck_idx])
                        edge_features_list.append([0.0, 0.0])
                        continue
                    
                    energy = env.transport_graph.get_path_energy(current_location, charger_id)
                    time = env.transport_graph.get_time_distance(current_location, charger_id)
                    
                    # Only add edge if energy is feasible (< current battery)
                    if energy < current_battery and not np.isinf(energy):
                        edge_index_list.append([truck_idx, charger_idx])
                        edge_features_list.append([energy, time])
                        edge_index_list.append([charger_idx, truck_idx])
                        edge_features_list.append([energy, time])
            
            else:
                # ROUTING: connect only to destination node
                if truck.route_destination is not None:
                    destination = truck.route_destination
                    
                    # Check if destination is a delivery node
                    if destination in delivery_node_to_idx:
                        dest_idx = delivery_node_to_idx[destination]
                        # Calculate remaining energy/time to destination
                        time_remaining = max(0.0, truck.route_arrival_time - env.global_clock) if truck.route_arrival_time else 0.0
                        # Estimate energy based on distance already covered
                        total_energy = env.transport_graph.get_path_energy(truck.current_node, destination)
                        # For simplicity, use 0 energy/time since truck is in transit
                        edge_index_list.append([truck_idx, dest_idx])
                        edge_features_list.append([0.0, time_remaining])
                        edge_index_list.append([dest_idx, truck_idx])
                        edge_features_list.append([0.0, time_remaining])
                    
                    # Check if destination is a charger node
                    elif destination in charger_node_to_idx:
                        dest_idx = charger_node_to_idx[destination]
                        time_remaining = max(0.0, truck.route_arrival_time - env.global_clock) if truck.route_arrival_time else 0.0
                        edge_index_list.append([truck_idx, dest_idx])
                        edge_features_list.append([0.0, time_remaining])
                        edge_index_list.append([dest_idx, truck_idx])
                        edge_features_list.append([0.0, time_remaining])

        # 5. Add edges between chargers (always bidirectional if feasible)
        for i, charger1_id in enumerate(env.charging_nodes):
            if charger1_id not in charger_node_to_idx:
                continue
            charger1_idx = charger_node_to_idx[charger1_id]
            
            for charger2_id in env.charging_nodes[i+1:]:
                if charger2_id not in charger_node_to_idx:
                    continue
                charger2_idx = charger_node_to_idx[charger2_id]
                
                energy_dist = env.transport_graph.get_path_energy(charger1_id, charger2_id)
                time_to_traverse = env.transport_graph.get_time_distance(charger1_id, charger2_id)
                
                if energy_dist <= max_battery_capacity and not np.isinf(energy_dist):
                    edge_index_list.append([charger1_idx, charger2_idx])
                    edge_features_list.append([energy_dist, time_to_traverse])
                    
                    edge_index_list.append([charger2_idx, charger1_idx])
                    edge_features_list.append([energy_dist, time_to_traverse])

        # 6. Add edges between chargers and deliveries (bidirectional if feasible)
        for charger_id in env.charging_nodes:
            if charger_id not in charger_node_to_idx:
                continue
            charger_idx = charger_node_to_idx[charger_id]
            
            for delivery_id in delivery_node_to_idx.keys():
                delivery_idx = delivery_node_to_idx[delivery_id]
                
                # Charger → Delivery
                energy_dist = env.transport_graph.get_path_energy(charger_id, delivery_id)
                time_to_traverse = env.transport_graph.get_time_distance(charger_id, delivery_id)
                
                if energy_dist <= max_battery_capacity and not np.isinf(energy_dist):
                    edge_index_list.append([charger_idx, delivery_idx])
                    edge_features_list.append([energy_dist, time_to_traverse])
                
                # Delivery → Charger
                energy_dist = env.transport_graph.get_path_energy(delivery_id, charger_id)
                time_to_traverse = env.transport_graph.get_time_distance(delivery_id, charger_id)
                
                if energy_dist <= max_battery_capacity and not np.isinf(energy_dist):
                    edge_index_list.append([delivery_idx, charger_idx])
                    edge_features_list.append([energy_dist, time_to_traverse])

        # 7. Add edges between delivery nodes (bidirectional if feasible)
        delivery_ids = sorted(delivery_node_to_idx.keys())
        for i, delivery1_id in enumerate(delivery_ids):
            delivery1_idx = delivery_node_to_idx[delivery1_id]
            
            for delivery2_id in delivery_ids[i+1:]:
                delivery2_idx = delivery_node_to_idx[delivery2_id]
                
                energy_dist = env.transport_graph.get_path_energy(delivery1_id, delivery2_id)
                time_to_traverse = env.transport_graph.get_time_distance(delivery1_id, delivery2_id)
                
                if energy_dist <= max_battery_capacity and not np.isinf(energy_dist):
                    edge_index_list.append([delivery1_idx, delivery2_idx])
                    edge_features_list.append([energy_dist, time_to_traverse])
                    
                    edge_index_list.append([delivery2_idx, delivery1_idx])
                    energy_dist_back = env.transport_graph.get_path_energy(delivery2_id, delivery1_id)
                    time_to_traverse_back = env.transport_graph.get_time_distance(delivery2_id, delivery1_id)
                    edge_features_list.append([energy_dist_back, time_to_traverse_back])

        # Convert to tensors
        if edge_index_list:
            edge_index = torch.tensor(
                np.array(edge_index_list).T, dtype=torch.long, device=self.device
            )
            edge_attr = torch.tensor(
                edge_features_list, dtype=torch.float32, device=self.device
            )
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=self.device)
            edge_attr = torch.zeros((0, 2), dtype=torch.float32, device=self.device)

        # Create PyTorch Geometric Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=len(node_list),
        )

        # Add metadata
        data.active_truck_id = torch.tensor(
            [env.active_truck_id if env.active_truck_id is not None else -1],
            device=self.device,
        )
        data.global_clock = torch.tensor([env.global_clock], device=self.device)
        data.num_trucks = torch.tensor([env.num_trucks], device=self.device)
        
        # Store node information for visualization
        data.node_list = node_list

        return data

    # ==================== Node Feature Functions ====================

    def _get_truck_node_features(self, truck, env) -> list:
        """
        Get feature vector for a truck node.
        
        Features:
        - Node type (1)
        - Current position (normalized)
        - Battery level (kWh)
        - Battery percentage (0-100)
        - Truck state (one-hot encoded: ready, routing, waiting_to_charge, charging)
        - Deliveries completed
        - Deliveries remaining
        - Time elapsed
        - Distance traveled
        - Time to destination (hours, 0 if not on route)
        """
        num_nodes_norm = env.transport_graph.num_nodes
        current_node_norm = truck.current_node / num_nodes_norm

        deliveries_remaining = len(truck.get_remaining_deliveries())
        deliveries_done = truck.current_sequence_index
        
        # Calculate time to destination if truck is on route
        time_to_destination = 0.0
        if truck.route_arrival_time is not None and truck.route_destination is not None:
            time_to_destination = max(0.0, truck.route_arrival_time - env.global_clock)

        # Determine truck state (one-hot encoding)
        # States: ready, routing, waiting_to_charge, charging
        # Note: complete and failed trucks are already filtered out before feature extraction
        is_ready = 0.0
        is_routing = 0.0
        is_waiting_to_charge = 0.0
        is_charging = 0.0
        
        if truck.is_charging:
            is_charging = 1.0
        elif truck.truck_id in env.charging_station.charger_waitlist.get(truck.current_node, []):
            # Truck is in a charger queue, waiting to charge
            is_waiting_to_charge = 1.0
        elif truck.route_destination is not None and truck.route_arrival_time is not None:
            # Truck is actively routing to a destination
            is_routing = 1.0
        else:
            # Truck is ready for action (at a node, not charging, not routing)
            is_ready = 1.0

        return [
            float(self.NODE_TYPE_TRUCK),  # Node type
            current_node_norm,  # Position normalized
            truck.current_battery,  # Battery level (kWh)
            truck.get_battery_percentage(),  # Battery percentage (0-100)
            is_ready,  # State: ready (one-hot)
            is_routing,  # State: routing (one-hot)
            is_waiting_to_charge,  # State: waiting_to_charge (one-hot)
            is_charging,  # State: charging (one-hot)
            float(deliveries_done),  # Deliveries completed
            float(deliveries_remaining),  # Deliveries remaining
            truck.total_time_elapsed,  # Time elapsed (hours)
            truck.total_distance_traveled,  # Distance traveled (km)
            time_to_destination,  # Time to destination (hours)
        ]

    def _get_delivery_node_features(self, node_id: int, env) -> np.ndarray:
        """
        Delivery node features (2 features total, no padding).

        Features:
        [0]: node_type (1 = delivery)
        [1]: node_id (normalized by total number of nodes)

        Args:
            node_id: Delivery node ID
            env: EventDrivenTruckEnv instance

        Returns:
            Feature vector (2 features)
        """
        num_nodes = env.transport_graph.num_nodes
        node_id_norm = node_id / num_nodes if num_nodes > 0 else 0.0
        
        features = [
            self.NODE_TYPE_DELIVERY,
            node_id_norm,
        ]
        
        return np.array(features, dtype=np.float32)

    def _get_charger_node_features(self, node_id: int, env) -> np.ndarray:
        """
        Charger node features (5 features total, no padding).

        Features:
        [0]: node_type (2 = charger)
        [1]: node_id (normalized by total number of nodes)
        [2]: charger_occupancy_rate (current_occupancy / capacity)
        [3]: charger_queue_length (number of trucks waiting)
        [4]: charger_type (Level2=0, Level3=1, DC_Fast=2)

        Args:
            node_id: Charger node ID
            env: EventDrivenTruckEnv instance

        Returns:
            Feature vector (5 features)
        """
        num_nodes = env.transport_graph.num_nodes
        node_id_norm = node_id / num_nodes if num_nodes > 0 else 0.0
        
        # Get charger info from environment
        charger_capacity = env.charging_station.charger_capacity.get(node_id, 0)
        charger_occupancy_list = env.charging_station.charger_occupancy.get(node_id, [])
        charger_occupancy = len(charger_occupancy_list)  # Number of trucks currently charging
        charger_queue = env.charging_station.charger_waitlist.get(node_id, [])
        charger_type_str = env.charging_station.charger_type.get(node_id, "Level2")  # string
        
        # Normalize occupancy to [0, 1]
        occupancy_rate = charger_occupancy / charger_capacity if charger_capacity > 0 else 0.0
        queue_length = len(charger_queue)
        
        # Encode charger type as numeric value
        # Map known charger types to numeric codes
        charger_type_map = {
            "Level2": 0.0,
            "Level3": 1.0,
            "DC_Fast": 2.0,
        }
        charger_type_encoded = charger_type_map.get(charger_type_str, 0.0)
        
        features = [
            self.NODE_TYPE_CHARGER,
            node_id_norm,
            occupancy_rate,
            queue_length,
            charger_type_encoded,
        ]
        
        return np.array(features, dtype=np.float32)

    # ==================== Edge Feature Functions ====================

    def _get_edge_features(self, src_node_id: int, dst_node_id: int, env) -> list:
        """
        Get edge features between two nodes.
        
        Edge Features:
        - Energy distance (kWh)
        - Time to traverse (hours)
        """
        energy_dist = env.transport_graph.get_path_energy(src_node_id, dst_node_id)
        time_to_traverse = env.transport_graph.get_time_distance(src_node_id, dst_node_id)
        
        return [energy_dist, time_to_traverse]

    # ==================== Utility Functions ====================

    def get_state_dict_for_gnn(self, env) -> Dict:
        """
        Get complete state information for GNN as dictionary.

        Returns:
            Dictionary with environment state and metadata
        """
        return {
            "graph": self.get_state_GNN(env),
            "active_truck_id": env.active_truck_id,
            "global_clock": env.global_clock,
            "num_trucks": env.num_trucks,
            "num_deliveries_remaining": (
                len(env.trucks[env.active_truck_id].get_remaining_deliveries())
                if env.active_truck_id is not None
                else 0
            ),
            "max_time": env.max_time,
        }

    @staticmethod
    def graph_to_numpy(data: Data) -> Dict:
        """Convert PyTorch Geometric Data to numpy for inspection."""
        return {
            "x": data.x.cpu().numpy(),
            "edge_index": data.edge_index.cpu().numpy(),
            "edge_attr": data.edge_attr.cpu().numpy() if data.edge_attr is not None else None,
            "num_nodes": data.num_nodes,
        }

    @staticmethod
    def visualize_graph_info(data: Data):
        """Print information about the graph."""
        print(f"PyTorch Geometric Data Graph")
        print(f"  - Nodes: {data.num_nodes}")
        print(f"  - Node features: {data.x.shape}")
        print(f"  - Edges: {data.edge_index.shape[1]}")
        if data.edge_attr is not None:
            print(f"  - Edge features: {data.edge_attr.shape}")
        else:
            print(f"  - Edge features: None")
