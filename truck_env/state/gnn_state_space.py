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

from torch_geometric.data import Data, HeteroData


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
        self.node_type_order = ['truck', 'delivery', 'charger']
        self.node_type_to_code = {
            node_type: idx for idx, node_type in enumerate(self.node_type_order)
        }
        
        # Feature dimensions (calculated from feature extraction methods)
        # Truck: 13 features, Delivery: 3 features, Charger: 4 features, Edge: 2 features
        self._truck_feature_dim = 13
        self._delivery_feature_dim = 3
        self._charger_feature_dim = 4
        self._edge_feature_dim = 2

    def get_state_GNN(self, env) -> HeteroData:
        """
        Convert environment state to PyTorch Geometric HeteroData graph.

        Heterogeneous Graph Structure:
        - Node types: 'truck', 'delivery', 'charger' (no padding!)
        - Edge types: ('truck', 'to', 'delivery'), ('delivery', 'to', 'truck'), etc.
        - Edge features: [energy, time] for each edge type

        Args:
            env: EventDrivenTruckEnv instance

        Returns:
            torch_geometric.data.HeteroData graph
        """
        
        # Initialize HeteroData
        data = HeteroData()
        
        # Track node mappings for edge construction
        truck_id_to_idx = {}  # truck_id -> index in truck node list
        delivery_node_to_idx = {}  # delivery_node_id -> index in delivery node list
        charger_node_to_idx = {}  # charger_node_id -> index in charger node list
        
        # Track which delivery nodes have been delivered
        delivered_nodes: Set[int] = set()
        for truck in env.trucks:
            all_deliveries = set(truck.delivery_sequence[1:])  # Skip depot at index 0
            remaining_deliveries = set(truck.get_remaining_deliveries())
            delivered = all_deliveries - remaining_deliveries
            delivered_nodes.update(delivered)

        # 1. Build truck nodes (excluding failed and completed trucks)
        truck_features_list = []
        
        for truck in env.trucks:
            # Skip failed and completed trucks
            if truck.failed or truck.is_complete:
                truck_id_to_idx[truck.truck_id] = None
                continue
            
            idx = len(truck_features_list)
            truck_id_to_idx[truck.truck_id] = idx
            
            features = self._get_truck_node_features(truck, env)
            truck_features_list.append(features)
        
        if truck_features_list:
            # Convert to numpy array first to avoid warning
            truck_features_array = np.array(truck_features_list, dtype=np.float32)
            data['truck'].x = torch.tensor(truck_features_array, dtype=torch.float32, device=self.device)
            self._truck_feature_dim = truck_features_array.shape[1]
        else:
            data['truck'].x = torch.zeros((0, self._truck_feature_dim), dtype=torch.float32, device=self.device)
        
        # 2. Build delivery nodes (only undelivered)
        delivery_features_list = []
        
        # Collect all delivery nodes from active trucks
        all_delivery_nodes = set()
        for truck in env.trucks:
            if truck.failed or truck.is_complete:
                continue
            all_delivery_nodes.update(truck.delivery_sequence[1:])  # Skip depot
        
        # Add only undelivered nodes
        for delivery_node_id in sorted(all_delivery_nodes):
            if delivery_node_id not in delivered_nodes:
                idx = len(delivery_features_list)
                delivery_node_to_idx[delivery_node_id] = idx
                
                features = self._get_delivery_node_features(delivery_node_id, env)
                delivery_features_list.append(features)
        
        if delivery_features_list:
            # Convert to numpy array first to avoid warning
            delivery_features_array = np.array(delivery_features_list, dtype=np.float32)
            data['delivery'].x = torch.tensor(delivery_features_array, dtype=torch.float32, device=self.device)
            self._delivery_feature_dim = delivery_features_array.shape[1]
        else:
            data['delivery'].x = torch.zeros((0, self._delivery_feature_dim), dtype=torch.float32, device=self.device)
        
        # 3. Build charger nodes (all chargers)
        charger_features_list = []
        
        for charger_node_id in env.charging_nodes:
            idx = len(charger_features_list)
            charger_node_to_idx[charger_node_id] = idx
            
            features = self._get_charger_node_features(charger_node_id, env)
            charger_features_list.append(features)
        
        if charger_features_list:
            # Convert to numpy array first to avoid warning
            charger_features_array = np.array(charger_features_list, dtype=np.float32)
            data['charger'].x = torch.tensor(charger_features_array, dtype=torch.float32, device=self.device)
            self._charger_feature_dim = charger_features_array.shape[1]
        else:
            data['charger'].x = torch.zeros((0, self._charger_feature_dim), dtype=torch.float32, device=self.device)

        # Get max truck battery capacity for feasibility checks
        max_battery_capacity = max(truck.battery_capacity for truck in env.trucks)

        # Build edges by type
        # Edge types we'll create:
        # - ('truck', 'to', 'delivery')
        # - ('delivery', 'to', 'truck')
        # - ('truck', 'to', 'charger')
        # - ('charger', 'to', 'truck')
        # - ('truck', 'to', 'truck')
        # - ('charger', 'to', 'charger')
        # - ('charger', 'to', 'delivery')
        # - ('delivery', 'to', 'delivery')
        
        edge_dict = {
            ('truck', 'to', 'delivery'): {'edge_index': [], 'edge_attr': []},
            ('delivery', 'to', 'truck'): {'edge_index': [], 'edge_attr': []},
            ('truck', 'to', 'charger'): {'edge_index': [], 'edge_attr': []},
            ('charger', 'to', 'truck'): {'edge_index': [], 'edge_attr': []},
            ('truck', 'to', 'truck'): {'edge_index': [], 'edge_attr': []},
            ('charger', 'to', 'charger'): {'edge_index': [], 'edge_attr': []},
            ('charger', 'to', 'delivery'): {'edge_index': [], 'edge_attr': []},
            ('delivery', 'to', 'delivery'): {'edge_index': [], 'edge_attr': []},
        }

        # 4. Add truck edges based on state
        for truck in env.trucks:
            if truck.failed or truck.is_complete:
                continue
            
            truck_idx = truck_id_to_idx[truck.truck_id]
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
                    edge_dict[('truck', 'to', 'charger')]['edge_index'].append([truck_idx, charger_idx])
                    edge_dict[('truck', 'to', 'charger')]['edge_attr'].append([0.0, 0.0])
                    edge_dict[('charger', 'to', 'truck')]['edge_index'].append([charger_idx, truck_idx])
                    edge_dict[('charger', 'to', 'truck')]['edge_attr'].append([0.0, 0.0])
                    
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
                        # Normalize edge features
                        energy_norm = energy / 1000.0
                        time_norm = time / self.max_time
                        edge_dict[('truck', 'to', 'delivery')]['edge_index'].append([truck_idx, delivery_idx])
                        edge_dict[('truck', 'to', 'delivery')]['edge_attr'].append([energy_norm, time_norm])
                        edge_dict[('delivery', 'to', 'truck')]['edge_index'].append([delivery_idx, truck_idx])
                        edge_dict[('delivery', 'to', 'truck')]['edge_attr'].append([energy_norm, time_norm])
                
                # Connect to all chargers (if feasible with current battery)
                for charger_id, charger_idx in charger_node_to_idx.items():
                    # Skip self-loop (truck already at this charger)
                    if charger_id == current_location:
                        # Add 0-weight edge to current location
                        edge_dict[('truck', 'to', 'charger')]['edge_index'].append([truck_idx, charger_idx])
                        edge_dict[('truck', 'to', 'charger')]['edge_attr'].append([0.0, 0.0])
                        edge_dict[('charger', 'to', 'truck')]['edge_index'].append([charger_idx, truck_idx])
                        edge_dict[('charger', 'to', 'truck')]['edge_attr'].append([0.0, 0.0])
                        continue
                    
                    energy = env.transport_graph.get_path_energy(current_location, charger_id)
                    time = env.transport_graph.get_time_distance(current_location, charger_id)
                    
                    # Only add edge if energy is feasible (< current battery)
                    if energy < current_battery and not np.isinf(energy):
                        # Normalize edge features
                        energy_norm = energy / 1000.0
                        time_norm = time / self.max_time
                        edge_dict[('truck', 'to', 'charger')]['edge_index'].append([truck_idx, charger_idx])
                        edge_dict[('truck', 'to', 'charger')]['edge_attr'].append([energy_norm, time_norm])
                        edge_dict[('charger', 'to', 'truck')]['edge_index'].append([charger_idx, truck_idx])
                        edge_dict[('charger', 'to', 'truck')]['edge_attr'].append([energy_norm, time_norm])
            
            else:
                # ROUTING: connect only to destination node
                if truck.route_destination is not None:
                    destination = truck.route_destination
                    time_remaining = max(0.0, truck.route_arrival_time - env.global_clock) if truck.route_arrival_time else 0.0
                    time_remaining_norm = time_remaining / self.max_time
                    
                    # Check if destination is a delivery node
                    if destination in delivery_node_to_idx:
                        dest_idx = delivery_node_to_idx[destination]
                        edge_dict[('truck', 'to', 'delivery')]['edge_index'].append([truck_idx, dest_idx])
                        edge_dict[('truck', 'to', 'delivery')]['edge_attr'].append([0.0, time_remaining_norm])
                        edge_dict[('delivery', 'to', 'truck')]['edge_index'].append([dest_idx, truck_idx])
                        edge_dict[('delivery', 'to', 'truck')]['edge_attr'].append([0.0, time_remaining_norm])
                    
                    # Check if destination is a charger node
                    elif destination in charger_node_to_idx:
                        dest_idx = charger_node_to_idx[destination]
                        edge_dict[('truck', 'to', 'charger')]['edge_index'].append([truck_idx, dest_idx])
                        edge_dict[('truck', 'to', 'charger')]['edge_attr'].append([0.0, time_remaining_norm])
                        edge_dict[('charger', 'to', 'truck')]['edge_index'].append([dest_idx, truck_idx])
                        edge_dict[('charger', 'to', 'truck')]['edge_attr'].append([0.0, time_remaining_norm])

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
                    # Normalize edge features
                    energy_norm = energy_dist / 1000.0
                    time_norm = time_to_traverse / self.max_time
                    edge_dict[('charger', 'to', 'charger')]['edge_index'].append([charger1_idx, charger2_idx])
                    edge_dict[('charger', 'to', 'charger')]['edge_attr'].append([energy_norm, time_norm])
                    
                    edge_dict[('charger', 'to', 'charger')]['edge_index'].append([charger2_idx, charger1_idx])
                    edge_dict[('charger', 'to', 'charger')]['edge_attr'].append([energy_norm, time_norm])

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
                    # Normalize edge features
                    energy_norm = energy_dist / 1000.0
                    time_norm = time_to_traverse / self.max_time
                    edge_dict[('charger', 'to', 'delivery')]['edge_index'].append([charger_idx, delivery_idx])
                    edge_dict[('charger', 'to', 'delivery')]['edge_attr'].append([energy_norm, time_norm])
                
                # Delivery → Charger (reverse direction)
                # Note: This is delivery->charger but we don't have that edge type defined
                # We'll add it to charger->delivery with reversed indices
                # Actually, we should add ('delivery', 'to', 'charger') edge type
                # But to keep it simple, let's add the reverse edge to the existing type
                # NO - we need proper bidirectionality, so let's add the reverse edge properly

        # Add missing edge type for delivery->charger
        if ('delivery', 'to', 'charger') not in edge_dict:
            edge_dict[('delivery', 'to', 'charger')] = {'edge_index': [], 'edge_attr': []}
        
        for delivery_id in delivery_node_to_idx.keys():
            delivery_idx = delivery_node_to_idx[delivery_id]
            
            for charger_id in env.charging_nodes:
                if charger_id not in charger_node_to_idx:
                    continue
                charger_idx = charger_node_to_idx[charger_id]
                
                # Delivery → Charger
                energy_dist = env.transport_graph.get_path_energy(delivery_id, charger_id)
                time_to_traverse = env.transport_graph.get_time_distance(delivery_id, charger_id)
                
                if energy_dist <= max_battery_capacity and not np.isinf(energy_dist):
                    # Normalize edge features
                    energy_norm = energy_dist / 1000.0
                    time_norm = time_to_traverse / self.max_time
                    edge_dict[('delivery', 'to', 'charger')]['edge_index'].append([delivery_idx, charger_idx])
                    edge_dict[('delivery', 'to', 'charger')]['edge_attr'].append([energy_norm, time_norm])

        # 7. Add edges between delivery nodes (bidirectional if feasible)
        delivery_ids = sorted(delivery_node_to_idx.keys())
        for i, delivery1_id in enumerate(delivery_ids):
            delivery1_idx = delivery_node_to_idx[delivery1_id]
            
            for delivery2_id in delivery_ids[i+1:]:
                delivery2_idx = delivery_node_to_idx[delivery2_id]
                
                energy_dist = env.transport_graph.get_path_energy(delivery1_id, delivery2_id)
                time_to_traverse = env.transport_graph.get_time_distance(delivery1_id, delivery2_id)
                
                if energy_dist <= max_battery_capacity and not np.isinf(energy_dist):
                    # Normalize edge features
                    energy_norm = energy_dist / 1000.0
                    time_norm = time_to_traverse / self.max_time
                    edge_dict[('delivery', 'to', 'delivery')]['edge_index'].append([delivery1_idx, delivery2_idx])
                    edge_dict[('delivery', 'to', 'delivery')]['edge_attr'].append([energy_norm, time_norm])
                    
                    energy_dist_back = env.transport_graph.get_path_energy(delivery2_id, delivery1_id)
                    time_to_traverse_back = env.transport_graph.get_time_distance(delivery2_id, delivery1_id)
                    # Normalize edge features
                    energy_norm_back = energy_dist_back / 1000.0
                    time_norm_back = time_to_traverse_back / self.max_time
                    edge_dict[('delivery', 'to', 'delivery')]['edge_index'].append([delivery2_idx, delivery1_idx])
                    edge_dict[('delivery', 'to', 'delivery')]['edge_attr'].append([energy_norm_back, time_norm_back])

        # Convert edge lists to tensors and add to HeteroData
        for edge_type, edges in edge_dict.items():
            if len(edges['edge_index']) > 0:
                edge_index = torch.tensor(
                    np.array(edges['edge_index']).T, dtype=torch.long, device=self.device
                )
                edge_attr = torch.tensor(
                    edges['edge_attr'], dtype=torch.float32, device=self.device
                )
                data[edge_type].edge_index = edge_index
                data[edge_type].edge_attr = edge_attr
            else:
                # Empty edge type
                data[edge_type].edge_index = torch.zeros((2, 0), dtype=torch.long, device=self.device)
                data[edge_type].edge_attr = torch.zeros((0, self._edge_feature_dim), dtype=torch.float32, device=self.device)

        # Add metadata
        data['truck'].active_truck_id = torch.tensor(
            [env.active_truck_id if env.active_truck_id is not None else -1],
            device=self.device,
        )
        data.global_clock = torch.tensor([env.global_clock], device=self.device)
        data.num_trucks = torch.tensor([env.num_trucks], device=self.device)
        
        # Build discrete action space for active truck
        # Actions: [next_delivery, charger_0, charger_1, ..., charger_N, charge_here]
        # feasible_action_mask marks which actions are valid
        # action_to_node_map maps action_idx -> (node_id, is_charging_action)
        
        action_to_node_map = []  # List of (node_id, is_charging_action) tuples
        feasible_action_mask = []
        action_node_types = []  # Encoded node type per discrete action
        action_local_indices = []  # Local index inside node type tensor
        action_is_charging = []
        can_charge_here = False
        
        # Build node_id_to_type mapping: node_id -> (node_type_str, local_idx_in_type)
        node_id_to_type = {}
        
        # Map truck IDs to their local indices in truck features
        for truck_id, local_idx in truck_id_to_idx.items():
            if local_idx is not None:
                node_id_to_type[truck_id] = ('truck', local_idx)
        
        # Map delivery IDs to their local indices in delivery features
        for delivery_id, local_idx in delivery_node_to_idx.items():
            if local_idx is not None:
                node_id_to_type[delivery_id] = ('delivery', local_idx)
        
        # Map charger IDs to their local indices in charger features
        for charger_id, local_idx in charger_node_to_idx.items():
            if local_idx is not None:
                node_id_to_type[charger_id] = ('charger', local_idx)
        
        def _append_action_metadata(node_id: int, is_charging_action: bool):
            """Store metadata for action mapping to node embeddings."""
            if node_id == -1 or node_id not in node_id_to_type:
                action_node_types.append(-1)
                action_local_indices.append(-1)
            else:
                node_type, local_idx = node_id_to_type[node_id]
                node_type_code = self.node_type_to_code.get(node_type, -1)
                action_node_types.append(node_type_code)
                action_local_indices.append(local_idx)
            action_is_charging.append(bool(is_charging_action))

        if env.active_truck_id is not None and env.active_truck_id in truck_id_to_idx:
            active_truck_idx = truck_id_to_idx[env.active_truck_id]
            if active_truck_idx is not None:
                active_truck = env.trucks[env.active_truck_id]
                current_battery = active_truck.current_battery
                current_location = active_truck.current_node
                battery_pct = active_truck.get_battery_percentage()
                
                # Action 0: Go to next delivery (if exists and feasible)
                next_delivery = active_truck.get_next_delivery_target()
                if next_delivery is not None:
                    energy = env.transport_graph.get_path_energy(current_location, next_delivery)
                    is_feasible = energy < current_battery and not np.isinf(energy)
                    action_to_node_map.append((next_delivery, False))
                    feasible_action_mask.append(is_feasible)
                    _append_action_metadata(next_delivery, False)
                else:
                    # No delivery left - add dummy action
                    action_to_node_map.append((-1, False))
                    feasible_action_mask.append(False)
                    _append_action_metadata(-1, False)

                # Actions 1 to N: Go to charger i
                for charger_id in sorted(charger_node_to_idx.keys()):
                    if charger_id == current_location:
                        # Skip current location in routing actions
                        continue
                    energy = env.transport_graph.get_path_energy(current_location, charger_id)
                    is_feasible = energy < current_battery and not np.isinf(energy)
                    action_to_node_map.append((charger_id, False))
                    feasible_action_mask.append(is_feasible)
                    _append_action_metadata(charger_id, False)

                # Last action: Charge at current location (if at charger and battery not full)
                if current_location in charger_node_to_idx:
                    # Check if battery is not full (allow charging if < 95% to avoid precision issues)
                    can_charge = battery_pct < 95.0
                    can_charge_here = can_charge
                    action_to_node_map.append((current_location, True))
                    feasible_action_mask.append(can_charge)
                    _append_action_metadata(current_location, True)
                else:
                    # Not at charger - can't charge
                    action_to_node_map.append((-1, True))
                    feasible_action_mask.append(False)
                    _append_action_metadata(-1, True)

        # Convert to tensors
        data.action_to_node_map = action_to_node_map  # Keep as list for easy lookup
        data.feasible_action_mask = torch.tensor(feasible_action_mask, dtype=torch.bool, device=self.device)
        if action_node_types:
            data.action_node_type = torch.tensor(action_node_types, dtype=torch.long, device=self.device)
            data.action_local_index = torch.tensor(action_local_indices, dtype=torch.long, device=self.device)
            data.action_is_charging = torch.tensor(action_is_charging, dtype=torch.bool, device=self.device)
        else:
            data.action_node_type = torch.zeros((0,), dtype=torch.long, device=self.device)
            data.action_local_index = torch.zeros((0,), dtype=torch.long, device=self.device)
            data.action_is_charging = torch.zeros((0,), dtype=torch.bool, device=self.device)
        data.can_charge_here = can_charge_here
        data.node_id_to_type = node_id_to_type  # For Actor to map actions to node embeddings

        # Store metadata for debugging
        data.num_actions = torch.tensor(len(action_to_node_map), dtype=torch.long, device=self.device)
        
        # Store node type offsets for easy indexing
        data.node_type_offsets = {
            'truck': 0,
            'delivery': len(truck_features_list),
            'charger': len(truck_features_list) + len(delivery_features_list)
        }

        return data

    # ==================== Node Feature Functions ====================

    def _get_truck_node_features(self, truck, env) -> list:
        """
        Get feature vector for a truck node.
        
        Features (all normalized):
        - Node type (normalized by number of node types)
        - Current position (normalized by number of trucks)
        - Battery level (normalized by capacity, 0-1)
        - Battery percentage (normalized, 0-1)
        - Truck state (one-hot encoded: ready, routing, waiting_to_charge, charging)
        - Deliveries completed (normalized by total deliveries)
        - Deliveries remaining (normalized by total deliveries)
        - Time elapsed (normalized by max simulation time)
        - Distance traveled (normalized by dividing by 1000)
        - Time to destination (normalized by max simulation time)
        """
        # Normalize current position by total number of nodes in the graph
        num_nodes = env.transport_graph.num_nodes
        current_node_norm = truck.current_node / num_nodes if num_nodes > 0 else 0.0

        # Normalize deliveries by total number of deliveries for this truck
        total_deliveries = len(truck.delivery_sequence) - 1  # Exclude depot
        deliveries_remaining = len(truck.get_remaining_deliveries())
        deliveries_done = truck.current_sequence_index
        deliveries_done_norm = deliveries_done / total_deliveries if total_deliveries > 0 else 0.0
        deliveries_remaining_norm = deliveries_remaining / total_deliveries if total_deliveries > 0 else 0.0
        
        # Calculate time to destination if truck is on route (normalized)
        time_to_destination = 0.0
        if truck.route_arrival_time is not None and truck.route_destination is not None:
            time_to_destination = max(0.0, truck.route_arrival_time - env.global_clock) / self.max_time

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
            float(self.NODE_TYPE_TRUCK) / len(self.node_type_order),  # Node type
            current_node_norm,  # Position normalized by num_trucks
            truck.current_battery / truck.battery_capacity,  # Battery level normalized (0-1)
            truck.get_battery_percentage() / 100.0,  # Battery percentage normalized (0-1)
            is_ready,  # State: ready (one-hot)
            is_routing,  # State: routing (one-hot)
            is_waiting_to_charge,  # State: waiting_to_charge (one-hot)
            is_charging,  # State: charging (one-hot)
            deliveries_done_norm,  # Deliveries completed (normalized)
            deliveries_remaining_norm,  # Deliveries remaining (normalized)
            truck.total_time_elapsed / self.max_time,  # Time elapsed (normalized)
            truck.total_distance_traveled / 1000.0,  # Distance traveled (normalized by 1000)
            time_to_destination,  # Time to destination (normalized)
        ]

    def _get_delivery_node_features(self, node_id: int, env) -> np.ndarray:
        """
        Delivery node features (3 features total, no padding).

        Features (all normalized):
        [0]: node_type (normalized by number of node types)
        [1]: node_id (normalized by total number of nodes)
        [2]: delivery_sequence_index (relative position in remaining deliveries, normalized by max stops)

        Args:
            node_id: Delivery node ID
            env: EventDrivenTruckEnv instance

        Returns:
            Feature vector (3 features)
        """
        num_nodes = env.transport_graph.num_nodes
        node_id_norm = node_id / num_nodes if num_nodes > 0 else 0.0
        
        # Calculate delivery sequence index directly
        # Find the minimum position across all active trucks
        min_index = float('inf')
        
        for truck in env.trucks:
            # Skip failed and completed trucks
            if truck.failed or truck.is_complete:
                continue
                
            # Get remaining deliveries for this truck
            remaining_deliveries = truck.get_remaining_deliveries()
            
            # Check if node_id is in remaining deliveries
            if node_id in remaining_deliveries:
                # Find its position (1-based index)
                position = remaining_deliveries.index(node_id) + 1
                min_index = min(min_index, position)
        
        # Return 0 if node not found in any truck's sequence
        delivery_sequence_index = int(min_index) if min_index != float('inf') else 0
        
        # Normalize by max stops per truck
        delivery_sequence_index_norm = delivery_sequence_index / self.num_stops if self.num_stops > 0 else 0.0
        
        features = [
            self.NODE_TYPE_DELIVERY / len(self.node_type_order),
            node_id_norm,
            delivery_sequence_index_norm,
        ]
        
        return np.array(features, dtype=np.float32)

    def _get_charger_node_features(self, node_id: int, env) -> np.ndarray:
        """
        Charger node features (4 features total, no padding).

        Features (all normalized):
        [0]: node_type (normalized by number of node types)
        [1]: node_id (normalized by number of charging stations)
        [2]: charger_occupancy_rate (current_occupancy / capacity, 0-1)
        [3]: charger_queue_length (normalized by number of trucks)

        Args:
            node_id: Charger node ID
            env: EventDrivenTruckEnv instance

        Returns:
            Feature vector (4 features)
        """
        node_id_norm = node_id / self.num_charging_nodes if self.num_charging_nodes > 0 else 0.0
        
        # Get charger info from environment
        charger_capacity = env.charging_station.charger_capacity.get(node_id, 1)  # Default to 1 to avoid division by zero
        charger_occupancy_list = env.charging_station.charger_occupancy.get(node_id, [])
        charger_occupancy = len(charger_occupancy_list)  # Number of trucks currently charging
        charger_queue = env.charging_station.charger_waitlist.get(node_id, [])
        
        # Normalize occupancy to [0, 1]
        occupancy_rate = charger_occupancy / charger_capacity if charger_capacity > 0 else 0.0
        
        # Normalize queue length by number of trucks
        queue_length_norm = len(charger_queue) / self.num_trucks if self.num_trucks > 0 else 0.0
        
        features = [
            self.NODE_TYPE_CHARGER / len(self.node_type_order),  # Node type normalized
            node_id_norm,
            occupancy_rate,
            queue_length_norm,
        ]
        
        return np.array(features, dtype=np.float32)

    # ==================== Edge Feature Functions ====================

    def _get_edge_features(self, src_node_id: int, dst_node_id: int, env) -> list:
        """
        Get edge features between two nodes.
        
        Edge Features (normalized):
        - Energy distance (normalized by dividing by 1000)
        - Time to traverse (normalized by max simulation time)
        """
        energy_dist = env.transport_graph.get_path_energy(src_node_id, dst_node_id)
        time_to_traverse = env.transport_graph.get_time_distance(src_node_id, dst_node_id)
        
        # Normalize edge features
        energy_dist_norm = energy_dist / 1000.0
        time_to_traverse_norm = time_to_traverse / self.max_time
        
        return [energy_dist_norm, time_to_traverse_norm]

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
    def graph_to_numpy(data: HeteroData) -> Dict:
        """Convert PyTorch Geometric HeteroData to numpy for inspection."""
        result = {}
        
        # Node features
        for node_type in data.node_types:
            result[f'{node_type}_x'] = data[node_type].x.cpu().numpy()
        
        # Edge features
        for edge_type in data.edge_types:
            result[f'{edge_type}_edge_index'] = data[edge_type].edge_index.cpu().numpy()
            result[f'{edge_type}_edge_attr'] = data[edge_type].edge_attr.cpu().numpy()
        
        return result

    @staticmethod
    def visualize_graph_info(data: HeteroData):
        """Print information about the heterogeneous graph."""
        print(f"PyTorch Geometric HeteroData Graph")
        print(f"Node types:")
        for node_type in data.node_types:
            print(f"  - {node_type}: {data[node_type].x.shape[0]} nodes, {data[node_type].x.shape[1]} features")
        print(f"Edge types:")
        for edge_type in data.edge_types:
            num_edges = data[edge_type].edge_index.shape[1]
            print(f"  - {edge_type}: {num_edges} edges")