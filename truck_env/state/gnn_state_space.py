"""
GNN State Representation for Truck Routing Environment.

This module provides PyTorch Geometric graph representations of the truck routing
environment state, suitable for Graph Neural Network (GNN) agents.

The graph is constructed as:
- **Nodes**: Active Trucks + Remaining Delivery Nodes + All Charging Stations
- **Edges**: 
  - Truck ↔ Current Destination (delivery or charger) with energy_distance and time
  - Charger ↔ Charger with energy_distance and time
  - Charger ↔ Delivery with energy_distance and time
  - Delivery ↔ Charger with energy_distance and time
  - Delivery ↔ Delivery with energy_distance and time

Note: Complete and failed trucks are excluded from the graph.

This enables GNNs to learn routing policies by reasoning over the spatial-temporal
structure of the problem.
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

        # Node type constants
        self.NODE_TYPE_DEPOT = 0
        self.NODE_TYPE_TRUCK = 1
        self.NODE_TYPE_DELIVERY = 2
        self.NODE_TYPE_CHARGER = 3

    def get_state_GNN(self, env) -> Data:
        """
        Convert environment state to PyTorch Geometric Data graph.

        Graph Structure:
        ================
        Nodes:
        - Depot (starting node): Starting location for all trucks
        - All Trucks: Current position, battery, delivery progress
        - All Delivery Nodes: Location (excluding already delivered)
        - All Charging Stations: Location, occupancy, queue length

        Edges (with weights):
        - Truck → Depot: [energy_distance, time_to_traverse] to every delivery and charger
        - Charger ↔ Charger: [energy_distance, time_to_traverse]
        - Charger ↔ Delivery: [energy_distance, time_to_traverse]
        - Delivery ↔ Charger: [energy_distance, time_to_traverse]

        Args:
            env: EventDrivenTruckEnv instance

        Returns:
            torch_geometric.data.Data graph
        """

        # Build node list and features
        node_list = []
        node_features_list = []
        node_index_map = {}  # node_idx -> (node_type, node_id)
        
        # Track which delivery nodes have been delivered by any truck
        delivered_nodes: Set[int] = set()
        for truck in env.trucks:
            # Get delivered deliveries (all except remaining)
            all_deliveries = set(truck.delivery_sequence[1:])  # Skip depot at index 0
            remaining_deliveries = set(truck.get_remaining_deliveries())
            delivered = all_deliveries - remaining_deliveries
            delivered_nodes.update(delivered)

        # Get unique starting nodes and filter to only active depots
        # A depot is active if at least one truck is:
        # 1. Not complete AND not failed (still has potential to visit it)
        # 2. Either at the depot, OR will return to it (considered as "pointing to it")
        # 
        # Key insight: If truck.current_node != starting_node, the truck has departed
        # and is no longer "pointing to" this depot. Only trucks that are:
        # - At the depot (current_node == starting_node), OR
        # - Will eventually return to collect route_destination data pointing back
        # should keep the depot alive in the graph.
        #
        # Simplest rule: Depot exists if at least one active truck is at it
        unique_starting_nodes = {}  # starting_node_id -> depot_idx
        
        for truck in env.trucks:
            starting_node = truck.delivery_sequence[0]
            
            # Skip if truck is complete or failed
            if truck.is_complete or truck.failed:
                continue
            
            # Skip if we've already created this depot
            if starting_node in unique_starting_nodes:
                continue
            
            # Create depot node ONLY if this truck is still at the depot
            # (not departed, even if might come back - that's handled by edges)
            if truck.current_node == starting_node:
                depot_idx = len(node_list)
                unique_starting_nodes[starting_node] = depot_idx
                node_list.append(("depot", starting_node))
                node_features_list.append(self._get_depot_node_features(starting_node, env))

        # 1. Add all truck nodes (excluding failed trucks)
        truck_nodes_start = len(node_list)
        truck_to_depot_idx = {}  # truck_id -> depot_idx0
        truck_id_to_node_idx = {}  # truck_id -> node_idx (for mapping during edge creation)
        
        for truck in env.trucks:
            # Skip failed and completed trucks - they should not appear in the graph
            if truck.failed or truck.is_complete:
                truck_to_depot_idx[truck.truck_id] = None
                truck_id_to_node_idx[truck.truck_id] = None
                continue
            
            node_idx = len(node_list)
            node_index_map[node_idx] = ("truck", truck.truck_id)
            node_list.append(("truck", truck.truck_id))
            truck_id_to_node_idx[truck.truck_id] = node_idx
            
            features = self._get_truck_node_features(truck, env)
            node_features_list.append(features)
            
            # Map truck to its corresponding depot (if depot exists for this truck)
            starting_node = truck.delivery_sequence[0]
            if starting_node in unique_starting_nodes:
                truck_to_depot_idx[truck.truck_id] = unique_starting_nodes[starting_node]
            else:
                # Depot not created because truck is complete/failed, so no mapping
                truck_to_depot_idx[truck.truck_id] = None

        truck_nodes_end = len(node_list)

        # 2. Add all delivery nodes (excluding already delivered ones and from failed trucks)
        delivery_nodes_start = len(node_list)
        delivery_node_to_idx = {}  # delivery_node_id -> node_idx
        
        # Collect all delivery nodes only from active (not failed, not complete) trucks
        all_delivery_nodes = set()
        for truck in env.trucks:
            # Skip failed and complete trucks - their deliveries shouldn't appear in graph
            if truck.failed or truck.is_complete:
                continue
            all_delivery_nodes.update(truck.delivery_sequence[1:])  # Skip depot
        
        for delivery_node_id in sorted(all_delivery_nodes):
            if delivery_node_id not in delivered_nodes:
                node_idx = len(node_list)
                delivery_node_to_idx[delivery_node_id] = node_idx
                node_index_map[node_idx] = ("delivery", delivery_node_id)
                node_list.append(("delivery", delivery_node_id))
                
                features = self._get_delivery_node_features(delivery_node_id, env)
                node_features_list.append(features)

        delivery_nodes_end = len(node_list)

        # 3. Add all charging station nodes
        charger_nodes_start = len(node_list)
        charger_node_to_idx = {}  # charger_node_id -> node_idx
        
        for charger_node_id in env.charging_nodes:
            node_idx = len(node_list)
            charger_node_to_idx[charger_node_id] = node_idx
            node_index_map[node_idx] = ("charger", charger_node_id)
            node_list.append(("charger", charger_node_id))
            
            features = self._get_charger_node_features(charger_node_id, env)
            node_features_list.append(features)

        charger_nodes_end = len(node_list)

        # Convert node features to tensor
        x = torch.tensor(node_features_list, dtype=torch.float32, device=self.device)

        # Get max truck battery capacity for feasibility checks
        max_battery_capacity = max(truck.battery_capacity for truck in env.trucks)

        # Build edge list and edge features
        edge_index_list = []
        edge_features_list = []

        # 4. Add edges: Truck ↔ Current Location (always connected to where it is)
        # The truck is always at some node - this creates the edge to that node
        # If truck is at the node, edge weight is 0 (no energy/time needed)
        for truck in env.trucks:
            # Skip failed and completed trucks
            if truck.failed or truck.is_complete:
                continue
            
            truck_idx = truck_id_to_node_idx[truck.truck_id]
            if truck_idx is None:
                continue
            
            current_location = truck.current_node
            location_idx = None
            
            # Find which node represents the current location
            # Check if it's the depot
            if current_location == truck.delivery_sequence[0]:
                location_idx = truck_to_depot_idx[truck.truck_id]
            # Check if it's a delivery node
            elif current_location in delivery_node_to_idx:
                location_idx = delivery_node_to_idx[current_location]
            # Check if it's a charger node
            elif current_location in charger_node_to_idx:
                location_idx = charger_node_to_idx[current_location]
            
            # Create bidirectional edge with 0 energy/time (truck is already there)
            if location_idx is not None:
                # Truck → Current Location (0 energy, 0 time - already there)
                edge_index_list.append([truck_idx, location_idx])
                edge_features_list.append([0.0, 0.0])
                
                # Current Location → Truck (0 energy, 0 time - already there)
                edge_index_list.append([location_idx, truck_idx])
                edge_features_list.append([0.0, 0.0])

        # 5. Add edges: Depot → All Delivery Nodes (only outgoing from depot)
        for depot_starting_node, depot_idx in unique_starting_nodes.items():
            for delivery_id in delivery_node_to_idx.keys():
                delivery_idx = delivery_node_to_idx[delivery_id]
                
                # Depot → Delivery (only outgoing)
                energy_dist = env.transport_graph.get_path_energy(depot_starting_node, delivery_id)
                time_to_traverse = env.transport_graph.get_time_distance(depot_starting_node, delivery_id)
                if energy_dist <= max_battery_capacity and not np.isinf(energy_dist):
                    edge_index_list.append([depot_idx, delivery_idx])
                    edge_features_list.append([energy_dist, time_to_traverse])

        # 6. Add edges: Depot → All Charger Nodes (only outgoing from depot)
        for depot_starting_node, depot_idx in unique_starting_nodes.items():
            for charger_id in env.charging_nodes:
                if charger_id not in charger_node_to_idx:
                    continue
                charger_idx = charger_node_to_idx[charger_id]
                
                # Depot → Charger (only outgoing)
                energy_dist = env.transport_graph.get_path_energy(depot_starting_node, charger_id)
                time_to_traverse = env.transport_graph.get_time_distance(depot_starting_node, charger_id)
                if energy_dist <= max_battery_capacity and not np.isinf(energy_dist):
                    edge_index_list.append([depot_idx, charger_idx])
                    edge_features_list.append([energy_dist, time_to_traverse])

        # 7. Add edges between chargers (charger → charger)
        for i, charger1_id in enumerate(env.charging_nodes):
            if charger1_id not in charger_node_to_idx:
                continue
            charger1_idx = charger_node_to_idx[charger1_id]
            
            for charger2_id in env.charging_nodes[i+1:]:
                if charger2_id not in charger_node_to_idx:
                    continue
                charger2_idx = charger_node_to_idx[charger2_id]
                
                # Add bidirectional edges
                energy_dist = env.transport_graph.get_path_energy(charger1_id, charger2_id)
                time_to_traverse = env.transport_graph.get_time_distance(charger1_id, charger2_id)
                
                if energy_dist <= max_battery_capacity and not np.isinf(energy_dist):
                    edge_index_list.append([charger1_idx, charger2_idx])
                    edge_features_list.append([energy_dist, time_to_traverse])
                    
                    edge_index_list.append([charger2_idx, charger1_idx])
                    edge_features_list.append([energy_dist, time_to_traverse])

                # 8. Add edges between chargers and deliveries (charger ↔ delivery)
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

        # 9. Add edges between delivery nodes
        delivery_ids = sorted(delivery_node_to_idx.keys())
        for i, delivery1_id in enumerate(delivery_ids):
            delivery1_idx = delivery_node_to_idx[delivery1_id]
            
            for delivery2_id in delivery_ids[i+1:]:
                delivery2_idx = delivery_node_to_idx[delivery2_id]
                
                # Bidirectional edges
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
        
        # Store node information for visualization (node_type, node_id)
        data.node_list = node_list

        return data

    # ==================== Node Feature Functions ====================

    def _get_depot_node_features(self, depot_node_id: int, env) -> list:
        """
        Get feature vector for a depot node.
        
        Features:
        - Node type (0)
        - Position (normalized)
        - Padding features for consistency with truck nodes (13 total features)
        """
        num_nodes_norm = env.transport_graph.num_nodes
        node_norm = depot_node_id / num_nodes_norm
        return [
            float(self.NODE_TYPE_DEPOT),  # Node type
            node_norm,  # Position normalized
            0.0,  # Padding
            0.0,  # Padding
            0.0,  # Padding
            0.0,  # Padding
            0.0,  # Padding
            0.0,  # Padding
            0.0,  # Padding
            0.0,  # Padding
            0.0,  # Padding
            0.0,  # Padding
            0.0,  # Padding (13 features total)
        ]

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

    def _get_delivery_node_features(self, delivery_node_id: int, env) -> list:
        """
        Get feature vector for a delivery node.
        
        Features:
        - Node type (2)
        - Position (normalized)
        - Delivery priority / demand
        - Assignment count (how many trucks have this in their list)
        - Padding features for consistency with truck nodes (13 total features)
        """
        num_nodes_norm = env.transport_graph.num_nodes
        node_norm = delivery_node_id / num_nodes_norm

        # Count how many trucks have this delivery
        assignment_count = sum(
            1 for truck in env.trucks 
            if delivery_node_id in truck.delivery_sequence
        )

        return [
            float(self.NODE_TYPE_DELIVERY),  # Node type
            node_norm,  # Position normalized
            float(assignment_count),  # How many trucks need this delivery
            0.0,  # Padding
            0.0,  # Padding
            0.0,  # Padding
            0.0,  # Padding
            0.0,  # Padding
            0.0,  # Padding
            0.0,  # Padding
            0.0,  # Padding
            0.0,  # Padding
            0.0,  # Padding (13 features total)
        ]

    def _get_charger_node_features(self, charger_node_id: int, env) -> list:
        """
        Get feature vector for a charging station node.
        
        Features:
        - Node type (3)
        - Position (normalized)
        - Occupancy rate
        - Queue length (normalized)
        - Charger type (Level2 vs DCFast)
        - Padding features for consistency with truck nodes (13 total features)
        """
        num_nodes_norm = env.transport_graph.num_nodes
        node_norm = charger_node_id / num_nodes_norm

        occupancy = len(env.charging_station.charger_occupancy[charger_node_id])
        capacity = env.charging_station.charger_capacity[charger_node_id]
        occupancy_rate = occupancy / capacity if capacity > 0 else 0.0

        queue_length = len(env.charging_station.charger_waitlist[charger_node_id])
        queue_norm = min(queue_length / 5.0, 1.0)  # Normalize to 5 trucks max

        charger_type = env.charging_station.charger_type[charger_node_id]
        charger_type_id = 1.0 if charger_type == "DCFast" else 0.0

        return [
            float(self.NODE_TYPE_CHARGER),  # Node type
            node_norm,  # Position normalized
            occupancy_rate,  # Occupancy (0-1)
            queue_norm,  # Queue length normalized
            charger_type_id,  # Charger type (0=Level2, 1=DCFast)
            0.0,  # Padding
            0.0,  # Padding
            0.0,  # Padding
            0.0,  # Padding
            0.0,  # Padding
            0.0,  # Padding
            0.0,  # Padding
            0.0,  # Padding (13 features total)
        ]

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
