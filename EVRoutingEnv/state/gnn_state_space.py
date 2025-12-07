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
        verbose: bool = False,
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
        self.verbose = verbose

        # Node type constants (no depot)
        self.NODE_TYPE_TRUCK = 0
        self.NODE_TYPE_DELIVERY = 1
        self.NODE_TYPE_CHARGER = 2
        self.node_type_order = ['truck', 'delivery', 'charger']
        self.node_type_to_code = {
            node_type: idx for idx, node_type in enumerate(self.node_type_order)
        }
        
        # Feature dimensions (calculated from feature extraction methods)
        # Truck: 14 features (added must_leave_charger), Delivery: 3 features, Charger: 4 features, Edge: 2 features
        self._truck_feature_dim = 14
        self._delivery_feature_dim = 3
        self._charger_feature_dim = 4
        self._edge_feature_dim = 2
        
        self.BIDIRECTIONAL_EDGES = True

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
        
        # Track which delivery nodes have been delivered (per-truck tracking)
        # A node is only "delivered" if ALL trucks that need to visit it have completed that visit
        delivered_nodes: Set[int] = set()
        
        # Build a mapping of node_id -> set of trucks that still need to deliver there
        node_pending_trucks = {}
        for truck in env.trucks:
            if truck.failed or truck.is_complete:
                continue
            for delivery_node in truck.get_remaining_deliveries():
                if delivery_node not in node_pending_trucks:
                    node_pending_trucks[delivery_node] = set()
                node_pending_trucks[delivery_node].add(truck.truck_id)
        
        # A node is "delivered" if no active trucks have it in remaining deliveries
        # (We don't add it to delivered_nodes if any truck still needs it)
        # For nodes not in node_pending_trucks, check if any truck ever had it
        all_delivery_nodes_ever = set()
        for truck in env.trucks:
            all_delivery_nodes_ever.update(truck.delivery_sequence[1:])
        
        for node_id in all_delivery_nodes_ever:
            if node_id not in node_pending_trucks:
                # No active truck needs this node anymore - it's fully delivered
                delivered_nodes.add(node_id)

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
            raise ValueError("No active trucks found")
            data['truck'].x = torch.zeros((0, self._truck_feature_dim), dtype=torch.float32, device=self.device)
        
        # 2. Build delivery nodes (only undelivered)
        delivery_features_list = []
        
        # Collect all delivery nodes from active trucks
        all_delivery_nodes = set()
        for truck in env.trucks:
            if truck.failed or truck.is_complete:
                continue
            remaining = truck.get_remaining_deliveries()
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
            # No delivery nodes - this should only happen if episode should have terminated
            # Check if all trucks are complete or failed
            all_terminal = all(truck.failed or truck.is_complete for truck in env.trucks)
            
            # Detailed debugging for each truck
            truck_debug_info = []
            for truck in env.trucks:
                remaining = truck.get_remaining_deliveries()
                next_del = truck.get_next_delivery_target()
                truck_debug_info.append(
                    f"Truck {truck.truck_id}: "
                    f"seq_idx={truck.current_sequence_index}/{len(truck.delivery_sequence)-1}, "
                    f"is_complete={truck.is_complete}, "
                    f"failed={truck.failed}, "
                    f"state={env.truck_states.get(truck.truck_id)}, "
                    f"next_target={next_del}, "
                    f"remaining={len(remaining)}"
                )
            
            debug_details = "\n    ".join(truck_debug_info)
            
            if all_terminal:
                raise ValueError(
                    f"No delivery nodes found and all trucks are in terminal state. "
                    f"Episode should have terminated but didn't.\n"
                    f"  Active truck: {env.active_truck_id}\n"
                    f"  Truck states: {env.truck_states}\n"
                    f"  Truck details:\n    {debug_details}"
                )
            else:
                raise ValueError(
                    f"No delivery nodes found but not all trucks are terminal.\n"
                    f"  Active truck: {env.active_truck_id}\n"
                    f"  Truck states: {env.truck_states}\n"
                    f"  Truck details:\n    {debug_details}"
                )
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
            raise ValueError("No charger features found")
            # data['charger'].x = torch.zeros((0, self._charger_feature_dim), dtype=torch.float32, device=self.device)

        # Get max truck battery capacity for feasibility checks
        max_battery_capacity = max(truck.battery_capacity for truck in env.trucks)
        
        # Get energy uncertainty factor for safety margin
        # If energy uncertainty is enabled, we need to account for worst-case energy consumption
        energy_safety_factor = 1.0
        if hasattr(env, 'traffic_config') and env.traffic_config['enable_traffic'] and env.traffic_config['enable_energy_uncertainty']:
            # Use max_energy_multiplier as safety factor (e.g., 1.20 = 20% higher energy consumption)
            energy_safety_factor = env.traffic_config['max_energy_multiplier']
   
        edge_dict = {
            ('truck', 'to', 'delivery'): {'edge_index': [], 'edge_attr': []},
            ('delivery', 'to', 'truck'): {'edge_index': [], 'edge_attr': []},
            ('truck', 'to', 'charger'): {'edge_index': [], 'edge_attr': []},
            ('charger', 'to', 'truck'): {'edge_index': [], 'edge_attr': []},
            ('truck', 'to', 'truck'): {'edge_index': [], 'edge_attr': []},
            ('charger', 'to', 'charger'): {'edge_index': [], 'edge_attr': []},
            ('charger', 'to', 'delivery'): {'edge_index': [], 'edge_attr': []},
            ('delivery', 'to', 'charger'): {'edge_index': [], 'edge_attr': []},
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
            charger_waitlist = env.charging_station.charger_waitlist[current_location] if current_location in env.charging_station.charger_waitlist else []
            if truck.is_charging or truck.truck_id in charger_waitlist:
                # CHARGING or WAITING_TO_CHARGE: only connect to current charger
                if current_location in charger_node_to_idx:
                    charger_idx = charger_node_to_idx[current_location]
                    # Bidirectional edge with 0 energy/time (at charger)
                    edge_dict[('truck', 'to', 'charger')]['edge_index'].append([truck_idx, charger_idx])
                    edge_dict[('truck', 'to', 'charger')]['edge_attr'].append([0.0, 0.0])
                    
                    if self.BIDIRECTIONAL_EDGES:                    
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
                        energy_inv = env.transport_graph.get_path_energy(next_delivery, current_location)             
                        
                        time = env.transport_graph.get_time_distance(current_location, next_delivery)
                        time_inv = env.transport_graph.get_time_distance(next_delivery, current_location)

                    # Only add edge if energy is feasible (< current battery with safety margin)
                    # Account for worst-case energy consumption due to uncertainty
                    max_energy_needed = energy * energy_safety_factor
                    if max_energy_needed < current_battery and not np.isinf(energy):
                        # Normalize edge features                        
                        edge_dict[('truck', 'to', 'delivery')]['edge_index'].append([truck_idx, delivery_idx])
                        edge_dict[('truck', 'to', 'delivery')]['edge_attr'].append([energy/1000.0, time/self.max_time])                        
                        
                        if self.BIDIRECTIONAL_EDGES:
                            edge_dict[('delivery', 'to', 'truck')]['edge_index'].append([delivery_idx, truck_idx])
                            edge_dict[('delivery', 'to', 'truck')]['edge_attr'].append([energy_inv/1000.0, time_inv/self.max_time])
                
                # Connect to all chargers (if feasible with current battery)
                for charger_id, charger_idx in charger_node_to_idx.items():
                    # Skip self-loop (truck already at this charger)
                    if charger_id == current_location:
                        # Add 0-weight edge to current location
                        edge_dict[('truck', 'to', 'charger')]['edge_index'].append([truck_idx, charger_idx])
                        edge_dict[('truck', 'to', 'charger')]['edge_attr'].append([0.0, 0.0])
                        
                        if self.BIDIRECTIONAL_EDGES:
                            edge_dict[('charger', 'to', 'truck')]['edge_index'].append([charger_idx, truck_idx])
                            edge_dict[('charger', 'to', 'truck')]['edge_attr'].append([0.0, 0.0])
                            
                        continue
                    
                    energy = env.transport_graph.get_path_energy(current_location, charger_id)
                    energy_inv = env.transport_graph.get_path_energy(charger_id, current_location)             
                    time = env.transport_graph.get_time_distance(current_location, charger_id)
                    time_inv = env.transport_graph.get_time_distance(charger_id, current_location)
                    
                    # Only add edge if energy is feasible (< current battery with safety margin)
                    # Account for worst-case energy consumption due to uncertainty
                    max_energy_needed = energy * energy_safety_factor
                    if max_energy_needed < current_battery and not np.isinf(energy):
                        # Normalize edge features
                        edge_dict[('truck', 'to', 'charger')]['edge_index'].append([truck_idx, charger_idx])
                        edge_dict[('truck', 'to', 'charger')]['edge_attr'].append([energy/1000.0, time/self.max_time])
                        
                        if self.BIDIRECTIONAL_EDGES:                        
                            edge_dict[('charger', 'to', 'truck')]['edge_index'].append([charger_idx, truck_idx])
                            edge_dict[('charger', 'to', 'truck')]['edge_attr'].append([energy_inv/1000.0, time_inv/self.max_time])
            
            else:
                # ROUTING: connect only to destination node
                if truck.route_destination is not None:
                    destination = truck.route_destination
                    time_remaining = max(0.0, truck.route_arrival_time - env.global_clock)
                    time_remaining_norm = time_remaining / self.max_time
                    
                    # Check if destination is a delivery node
                    if destination in delivery_node_to_idx:
                        dest_idx = delivery_node_to_idx[destination]
                        edge_dict[('truck', 'to', 'delivery')]['edge_index'].append([truck_idx, dest_idx])
                        edge_dict[('truck', 'to', 'delivery')]['edge_attr'].append([0.0, time_remaining_norm])
                        
                        if self.BIDIRECTIONAL_EDGES:
                            edge_dict[('delivery', 'to', 'truck')]['edge_index'].append([dest_idx, truck_idx])
                            edge_dict[('delivery', 'to', 'truck')]['edge_attr'].append([0.0, time_remaining_norm])
                    
                    # Check if destination is a charger node
                    elif destination in charger_node_to_idx:
                        dest_idx = charger_node_to_idx[destination]
                        edge_dict[('truck', 'to', 'charger')]['edge_index'].append([truck_idx, dest_idx])
                        edge_dict[('truck', 'to', 'charger')]['edge_attr'].append([0.0, time_remaining_norm])
                        
                        if self.BIDIRECTIONAL_EDGES:
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
                energy_dist_back = env.transport_graph.get_path_energy(charger2_id, charger1_id)            
                
                time_to_traverse = env.transport_graph.get_time_distance(charger1_id, charger2_id)
                time_to_traverse_back = env.transport_graph.get_time_distance(charger2_id, charger1_id)
                
                # Account for energy uncertainty in feasibility checks
                if energy_dist * energy_safety_factor <= max_battery_capacity and not np.isinf(energy_dist):
                    # Normalize edge features                    
                    edge_dict[('charger', 'to', 'charger')]['edge_index'].append([charger1_idx, charger2_idx])
                    edge_dict[('charger', 'to', 'charger')]['edge_attr'].append([energy_dist/1000.0, time_to_traverse/self.max_time])
                
                if energy_dist_back * energy_safety_factor <= max_battery_capacity and not np.isinf(energy_dist_back):
                    edge_dict[('charger', 'to', 'charger')]['edge_index'].append([charger2_idx, charger1_idx])
                    edge_dict[('charger', 'to', 'charger')]['edge_attr'].append([energy_dist_back/1000.0, time_to_traverse_back/self.max_time])

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
                
                # Account for energy uncertainty in feasibility checks
                if energy_dist * energy_safety_factor <= max_battery_capacity and not np.isinf(energy_dist):
                    # Normalize edge features
                    energy_norm = energy_dist / 1000.0
                    time_norm = time_to_traverse / self.max_time
                    edge_dict[('charger', 'to', 'delivery')]['edge_index'].append([charger_idx, delivery_idx])
                    edge_dict[('charger', 'to', 'delivery')]['edge_attr'].append([energy_norm, time_norm])
                
        for delivery_id in delivery_node_to_idx.keys():
            delivery_idx = delivery_node_to_idx[delivery_id]
            
            for charger_id in env.charging_nodes:
                if charger_id not in charger_node_to_idx:
                    continue
                charger_idx = charger_node_to_idx[charger_id]
                
                # Delivery → Charger
                energy_dist = env.transport_graph.get_path_energy(delivery_id, charger_id)
                time_to_traverse = env.transport_graph.get_time_distance(delivery_id, charger_id)
                
                # Account for energy uncertainty in feasibility checks
                if energy_dist * energy_safety_factor <= max_battery_capacity and not np.isinf(energy_dist):
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
                
                # Account for energy uncertainty in feasibility checks
                if energy_dist * energy_safety_factor <= max_battery_capacity and not np.isinf(energy_dist):
                    # Normalize edge features
                    energy_norm = energy_dist / 1000.0
                    time_norm = time_to_traverse / self.max_time
                    edge_dict[('delivery', 'to', 'delivery')]['edge_index'].append([delivery1_idx, delivery2_idx])
                    edge_dict[('delivery', 'to', 'delivery')]['edge_attr'].append([energy_norm, time_norm])
                    
                energy_dist_back = env.transport_graph.get_path_energy(delivery2_id, delivery1_id)
                time_to_traverse_back = env.transport_graph.get_time_distance(delivery2_id, delivery1_id)
                if energy_dist_back * energy_safety_factor <= max_battery_capacity and not np.isinf(energy_dist_back):                    
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
        
        
        # ============ BUILD DISCRETE ACTION SPACE METADATA ============
        # Actions: [next_delivery, charger_0, charger_1, ..., charger_N, charge_here]
        # feasible_action_mask marks which actions are valid
        # action_to_node_map maps action_idx -> (node_id, is_charging_action)
        # action_charge_durations aligns with action_to_node_map (0.0 for navigation)
        
        action_to_node_map = []  # List of (node_id, is_charging_action) tuples
        feasible_action_mask = []
        action_node_types = []  # Encoded node type per discrete action
        action_local_indices = []  # Local index inside node type tensor
        action_is_charging = []
        action_charge_durations = []
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
                node_type_code = self.node_type_to_code[node_type] if node_type in self.node_type_to_code else -1
                action_node_types.append(node_type_code)
                action_local_indices.append(local_idx)
            action_is_charging.append(bool(is_charging_action))

        # Validate active truck - GNN state should only be called when environment has valid active truck
        if env.active_truck_id is None:
            raise ValueError(
                "Cannot generate GNN state: active_truck_id is None. "
                "GNN state should only be called when environment is active with a valid truck.\n"
                f"Truck states: {env.truck_states}\n"
                f"Global clock: {env.global_clock:.2f}h"
            )
        
        if env.active_truck_id not in truck_id_to_idx:
            # Active truck not in mapping - likely completed or failed
            if env.active_truck_id >= len(env.trucks):
                raise ValueError(
                    f"Invalid active_truck_id {env.active_truck_id}: out of range (num_trucks={len(env.trucks)})"
                )
            
            active_truck = env.trucks[env.active_truck_id]
            truck_status = "complete" if active_truck.is_complete else ("failed" if active_truck.failed else "unknown")
            raise ValueError(
                f"Cannot generate GNN state: active truck {env.active_truck_id} is {truck_status}.\n"
                f"Active trucks are filtered out when complete/failed, but environment still has it as active.\n"
                f"This indicates a bug in the environment's event processing.\n"
                f"Truck details:\n"
                f"  - Status: {truck_status}\n"
                f"  - Current node: {active_truck.current_node}\n"
                f"  - Battery: {active_truck.current_battery:.2f}/{active_truck.battery_capacity:.2f} kWh\n"
                f"  - Deliveries: {active_truck.current_sequence_index}/{len(active_truck.delivery_sequence)-1}\n"
                f"  - Truck state: {env.truck_states.get(env.active_truck_id, 'unknown')}\n"
                f"  - Global clock: {env.global_clock:.2f}h"
            )
        
        active_truck_idx = truck_id_to_idx[env.active_truck_id]
        if active_truck_idx is None:
            # This should be unreachable after the check above, but keep as safety
            active_truck = env.trucks[env.active_truck_id]
            raise ValueError(
                f"Active truck {env.active_truck_id} has idx=None in truck_id_to_idx mapping. "
                f"Truck complete={active_truck.is_complete}, failed={active_truck.failed}"
            )
        
        # Generate actions for valid active truck
        if True:  # Keep indentation level
                active_truck = env.trucks[env.active_truck_id]
                current_battery = active_truck.current_battery
                current_location = active_truck.current_node

                charge_durations = env.charging_config['charge_durations']  # in hours
                
                # Check if truck must leave charger (after charging)
                must_leave = active_truck.must_leave_charger
                
                # Check if truck is at charger - if so, must charge (unless must_leave is True)
                at_charger = current_location in charger_node_to_idx
                must_charge_now = at_charger and not must_leave
                next_delivery = active_truck.get_next_delivery_target()
                if self.verbose:
                    print(f'\n-- at_charger: {at_charger}, must_leave: {must_leave}, must_charge_now: {must_charge_now}, next_delivery: {next_delivery}')

                # Actions 0 to N-1: Go to charger i (must match environment action order)
                # Note: Include current location to match environment's action indexing
                for charger_id in sorted(charger_node_to_idx.keys()):
                    if charger_id == current_location:
                        # Current location - routing here is always infeasible
                        action_to_node_map.append((charger_id, False))
                        feasible_action_mask.append(False)
                        _append_action_metadata(charger_id, False)
                        action_charge_durations.append(0.0)
                    else:
                        energy = env.transport_graph.get_path_energy(current_location, charger_id)
                        max_energy_needed = energy * energy_safety_factor
                        is_energy_feasible = max_energy_needed < current_battery and not np.isinf(energy)
                        # Disable routing if truck must charge now
                        is_feasible = is_energy_feasible and not must_charge_now
                        action_to_node_map.append((charger_id, False))
                        feasible_action_mask.append(is_feasible)
                        _append_action_metadata(charger_id, False)
                        action_charge_durations.append(0.0)

                # Action N: Go to next delivery (must come after all chargers)
                if next_delivery is not None:
                    energy_to_delivery = env.transport_graph.get_path_energy(current_location, next_delivery)
                    max_energy_to_delivery = energy_to_delivery * energy_safety_factor
                    is_energy_feasible = max_energy_to_delivery < current_battery
                    
                    if self.verbose:
                        print(f'[RoutingToDel] Energy to next delivery {next_delivery}: {energy_to_delivery:.2f} kWh (max: {max_energy_to_delivery:.2f} kWh), current battery: {current_battery:.2f} kWh, is_energy_feasible: {is_energy_feasible}')
                    
                    # Additional check: After reaching the delivery, can the truck reach ANY charger or next delivery?
                    # This prevents the truck from getting stranded after completing this delivery
                    can_continue_after_delivery = False
                    if is_energy_feasible:
                        # Use worst-case energy for battery projection
                        battery_after_delivery = current_battery - max_energy_to_delivery
                        
                        # Check if there are more deliveries after this one
                        remaining_after_this = active_truck.get_remaining_deliveries()
                        has_more_deliveries = len(remaining_after_this) > 1  # More than just this delivery
                        if self.verbose:
                            print(f'  remaining_after_this: {remaining_after_this}, has_more_deliveries: {has_more_deliveries}')
                        
                        
                        #check if it can reach any charger after delivery
                        if self.verbose:
                            print(f'  Checking if can reach any charger after delivery from node {next_delivery}')
                        
                        if not has_more_deliveries:
                            can_continue_after_delivery = True
                        else:
                            for charger_id in charger_node_to_idx.keys():
                                energy_to_charger = env.transport_graph.get_path_energy(next_delivery, charger_id)
                                max_energy_to_charger = energy_to_charger * energy_safety_factor
                                if battery_after_delivery > max_energy_to_charger:
                                    can_continue_after_delivery = True
                                    break
                        
                        if self.verbose:
                            print(f'  can_continue_after_delivery: {can_continue_after_delivery} (battery after delivery: {battery_after_delivery:.2f} kWh)')
                        
                    # Disable routing if truck must charge now OR if truck would be stranded after delivery
                    is_feasible = is_energy_feasible and not must_charge_now and can_continue_after_delivery
                    action_to_node_map.append((next_delivery, False))
                    feasible_action_mask.append(is_feasible)
                    _append_action_metadata(next_delivery, False)
                    action_charge_durations.append(0.0)
                else:
                    raise ValueError("No next delivery found for active truck")

                # Last actions: Charge at current location (if at charger)
                if current_location in charger_node_to_idx:
                    if self.verbose:
                        print(f"\n[DEBUG] Truck {active_truck.truck_id} at charger {current_location}, must_leave={must_leave}")
                    
                    # If truck must leave charger, disable all charging actions
                    if must_leave:
                        can_charge_here = False
                        for charge_hours in charge_durations:
                            action_to_node_map.append((current_location, True))
                            feasible_action_mask.append(False)  # Force leaving, no charging allowed
                            _append_action_metadata(current_location, True)
                            action_charge_durations.append(float(charge_hours))
                        if self.verbose:
                            print(f"  Forcing truck to leave")
                            
                            #print bottom 5 ditances to chargers and the delivery
                            distances = []
                            for charger_id in charger_node_to_idx.keys():
                                energy_to_charger = env.transport_graph.get_path_energy(current_location, charger_id)
                                distances.append((charger_id, energy_to_charger))
                            distances.sort(key=lambda x: x[1])
                            print("  Closest chargers and distances:")
                            for charger_id, dist in distances[:5]:
                                print(f"    Charger {charger_id}: {dist:.2f} kWh")                        
                            
                    else:                                                
                        
                        # Calculate minimum energy needed to leave charger
                        # Truck must be able to reach at least ONE feasible destination
                        deliveries_left = active_truck.get_remaining_deliveries()
                        
                        # Start with energy to next delivery
                        energy_to_delivery = env.transport_graph.get_path_energy(current_location, next_delivery)
                        
                        # Try to find feasible destinations with original safety factor
                        # If none found, reduce safety factor progressively until feasible destinations exist
                        # Keep reducing even below 1.0 to allow risky actions (truck will fail later if route is truly impossible)
                        adjusted_safety_factor = energy_safety_factor
                        feasible_destinations = []
                        min_safety_factor = 0.5  # Allow safety factor down to 0.5 (50% margin in reverse)
                        
                        while not feasible_destinations and adjusted_safety_factor >= min_safety_factor:
                            # Check if next delivery is feasible
                            if energy_to_delivery * adjusted_safety_factor <= active_truck.battery_capacity:
                                feasible_destinations.append(energy_to_delivery)
                            
                            # Check all other chargers
                            for charger_id in charger_node_to_idx.keys():
                                if charger_id != current_location:
                                    energy_to_charger = env.transport_graph.get_path_energy(current_location, charger_id)
                                    # Only consider if feasible with full battery
                                    if energy_to_charger * adjusted_safety_factor <= active_truck.battery_capacity:
                                        feasible_destinations.append(energy_to_charger)
                            
                            if not feasible_destinations:
                                # Reduce safety factor and try again
                                adjusted_safety_factor -= 0.05
                                if self.verbose and adjusted_safety_factor >= min_safety_factor:
                                    print(f"  No feasible destinations with safety factor {adjusted_safety_factor + 0.05:.2f}, trying {adjusted_safety_factor:.2f}")
                        
                        if not feasible_destinations:
                            # Even with minimum safety factor (0.5), no feasible destinations
                            # This means the truck truly cannot reach ANY destination even with optimistic assumptions
                            # Fail the truck in the environment - it will handle the failure properly
                            print(f"\n[CRITICAL] Truck {active_truck.truck_id} at charger {current_location} has NO FEASIBLE DESTINATIONS!")
                            print(f"  Energy to next delivery {next_delivery}: {energy_to_delivery:.2f} kWh")
                            print(f"  Battery capacity: {active_truck.battery_capacity:.2f} kWh")
                            print(f"  Tried safety factors down to {min_safety_factor:.2f}, still no feasible destinations.")
                            print(f"  FAILING TRUCK - environment will handle the failure.")
                            
                            # Fail the truck in the environment
                            active_truck.failed = True
                            env.truck_states[active_truck.truck_id] = "failed"
                            
                            # Raise error to stop state generation and trigger environment termination check
                            raise ValueError(
                                f"Truck {active_truck.truck_id} at charger {current_location} cannot reach any destination "
                                f"even with reduced safety factor (tried down to {min_safety_factor:.2f}).\n"
                                f"Next delivery {next_delivery} requires {energy_to_delivery:.2f} kWh base energy.\n"
                                f"Battery capacity: {active_truck.battery_capacity:.2f} kWh.\n"
                                f"Delivery sequence is impossible. Truck has been marked as failed."
                            )
                        else:
                            # Use minimum feasible destination
                            min_energy_to_leave = min(feasible_destinations)
                            if adjusted_safety_factor < energy_safety_factor and self.verbose:
                                print(f"  Reduced safety factor from {energy_safety_factor:.2f} to {adjusted_safety_factor:.2f} to find feasible destinations")
                        
                        if self.verbose:
                            print(f"min_energy_to_leave: {min_energy_to_leave:.2f} kWh (feasible destinations: {len(feasible_destinations)})")
                            print(f'deliveries_left: {deliveries_left}')
                            
                        # Get charger configuration for charge rate calculation
                        charger_type = env.charging_station.charger_type.get(current_location, "Level2")
                        charging_config = env.config["charging"]
                        if charger_type == "DCFast":
                            charger_config = charging_config["dcfast"]
                        else:
                            charger_config = charging_config["level2"]
                        charge_rate = charger_config["charge_rate"]  # kW
                        efficiency = charger_config["efficiency"]
                        
                        # Debug: Print charging feasibility information
                        if self.verbose:
                            print(f"\n[CHARGING FEASIBILITY DEBUG] Truck {active_truck.truck_id}")
                            print(f"  Location: charger_{current_location} (type: {charger_type})")
                            print(f"  Current battery: {current_battery:.2f} / {active_truck.battery_capacity:.2f} kWh")
                            print(f"  Charge rate: {charge_rate:.2f} kW, Efficiency: {efficiency:.2f}")
                            print(f"  Next delivery target: {next_delivery}")                        
                                                        
                            # print(f"  Energy to delivery: {energy_to_delivery:.2f} kWh")
                            print(f"  Min energy to leave (any destination): {min_energy_to_leave:.2f} kWh")
                            print(f"  Evaluating charge durations: {charge_durations}")
                        
                        can_charge_here = False
                        
                        # Add global use_realistic_curve flag to charger config
                        charger_config_with_curve = charger_config.copy()
                        charger_config_with_curve["use_realistic_curve"] = charging_config["use_realistic_curve"]
                        
                        for charge_hours in charge_durations:
                            # Calculate resulting battery after this charge duration using curve model
                            # Clamp to [0.0, 1.0] to handle any floating point precision issues
                            initial_soc = min(1.0, max(0.0, active_truck.get_battery_percentage() / 100.0))
                            charge_amount, _ = env.charging_curve_model.calculate_charge(
                                initial_soc=initial_soc,
                                charge_hours=charge_hours,
                                battery_capacity=active_truck.battery_capacity,
                                charger_config=charger_config_with_curve,
                                charger_type=charger_type
                            )
                            resulting_battery = min(active_truck.battery_capacity, current_battery + charge_amount)
                            
                            # Check if resulting battery is enough to leave
                            # Must account for adjusted safety factor (may be reduced if original was too strict)
                            # Allow charging only if it provides enough energy to reach another location
                            min_energy_with_safety = min_energy_to_leave * adjusted_safety_factor
                            is_feasible = resulting_battery >= min_energy_with_safety
                            
                            if self.verbose:
                                # Debug: Print each charge duration evaluation
                                print(f"    Charge {charge_hours}h: +{charge_amount:.2f} kWh → {resulting_battery:.2f} kWh total | "
                                      f"Need: {min_energy_with_safety:.2f} kWh (base: {min_energy_to_leave:.2f} × {adjusted_safety_factor:.2f}) | "
                                      f"Feasible: {is_feasible}")
                            
                            action_to_node_map.append((current_location, True))
                            feasible_action_mask.append(is_feasible)
                            _append_action_metadata(current_location, True)
                            action_charge_durations.append(float(charge_hours))
                            
                            if is_feasible:
                                can_charge_here = True
                        
                        # Debug: Print summary
                        if self.verbose:
                            print(f"  Result: can_charge_here = {can_charge_here}")
                            if not can_charge_here:
                                print(f"  WARNING: NO FEASIBLE CHARGING ACTIONS! All durations insufficient.")
                else:
                    # Not at charger - can't charge
                    can_charge_here = False
                    for charge_hours in charge_durations or [0.0]:
                        action_to_node_map.append((-1, True))
                        feasible_action_mask.append(False)
                        _append_action_metadata(-1, True)
                        action_charge_durations.append(float(charge_hours))

        # Convert to tensors
        data.action_to_node_map = action_to_node_map  # Keep as list for easy lookup
        data.feasible_action_mask = torch.tensor(feasible_action_mask, dtype=torch.bool, device=self.device)
        
        if action_node_types:
            data.action_node_type = torch.tensor(action_node_types, dtype=torch.long, device=self.device)
            data.action_local_index = torch.tensor(action_local_indices, dtype=torch.long, device=self.device)
            data.action_is_charging = torch.tensor(action_is_charging, dtype=torch.bool, device=self.device)
            data.action_charge_durations = torch.tensor(action_charge_durations, dtype=torch.float32, device=self.device)
        else:
            if self.verbose:
                # Debug: Print action generation summary
                print(f"\n[ACTION GENERATION ERROR]")
                print(f"  Active truck: {env.active_truck_id}")
                print(f"  Total actions generated: {len(action_to_node_map)}")
                print(f"  Feasible actions: {sum(feasible_action_mask)}")
                print(f"  action_node_types length: {len(action_node_types)}")
                print(f"  action_to_node_map: {action_to_node_map}")
                print(f"  feasible_action_mask: {feasible_action_mask}")
                env.charging_station.print_queues()
            raise ValueError("No action metadata found for active truck")
            # data.action_node_type = torch.zeros((0,), dtype=torch.long, device=self.device)
            # data.action_local_index = torch.zeros((0,), dtype=torch.long, device=self.device)
            # data.action_is_charging = torch.zeros((0,), dtype=torch.bool, device=self.device)
            # data.action_charge_durations = torch.zeros((0,), dtype=torch.float32, device=self.device)
            
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

        # Build action graph features: [normalized_action_type, resulting_soc]
        # Only include features for feasible actions
        feasible_indices = [i for i, is_feasible in enumerate(feasible_action_mask) if is_feasible]
        feasible_action_to_node_map = [action_to_node_map[i] for i in feasible_indices]
        feasible_action_is_charging = [action_is_charging[i] for i in feasible_indices]
        feasible_action_durations = [action_charge_durations[i] for i in feasible_indices]
        
        # Strict validation: Raise error if all actions are infeasible
        if not feasible_action_to_node_map:
            active_truck = env.trucks[env.active_truck_id]
            diagnostics = self._get_action_feasibility_diagnostics(env, active_truck, charger_node_to_idx)
            
            # Build detailed infeasibility breakdown
            infeasibility_details = []
            infeasibility_details.append(f"\nAction Infeasibility Breakdown (Total: {len(action_to_node_map)} actions):")
            
            # Count action types and their feasibility
            routing_to_charger = sum(1 for (node_id, is_chg) in action_to_node_map if not is_chg and node_id in charger_node_to_idx)
            routing_to_delivery = sum(1 for (node_id, is_chg) in action_to_node_map if not is_chg and node_id not in charger_node_to_idx and node_id >= 0)
            charging_actions = sum(1 for (node_id, is_chg) in action_to_node_map if is_chg)
            
            infeasibility_details.append(f"  - Routing to chargers: {routing_to_charger} actions (all infeasible)")
            infeasibility_details.append(f"  - Routing to delivery: {routing_to_delivery} actions (all infeasible)")
            infeasibility_details.append(f"  - Charging actions: {charging_actions} actions (all infeasible)")
            
            # Explain likely causes
            infeasibility_details.append(f"\nLikely causes:")
            if active_truck.current_battery < 50.0:
                infeasibility_details.append(f"  ⚠ Low battery ({active_truck.current_battery:.2f} kWh) - may not reach any destination")
            if active_truck.must_leave_charger:
                infeasibility_details.append(f"  ⚠ Truck must leave charger but cannot reach any destination with current battery")
            at_charger = active_truck.current_node in charger_node_to_idx
            if at_charger and not active_truck.must_leave_charger:
                infeasibility_details.append(f"  ⚠ At charger but all charging durations insufficient to reach any destination")
            
            raise ValueError(
                f"Cannot generate GNN state: ALL ACTIONS ARE INFEASIBLE for truck {env.active_truck_id}.\n"
                f"This indicates the truck is in an unrecoverable state (e.g., stranded with insufficient battery).\n\n"
                f"{diagnostics}\n"
                f"{''.join(infeasibility_details)}\n\n"
                f"Full action details:\n"
                f"  action_to_node_map: {action_to_node_map}\n"
                f"  feasible_action_mask: {feasible_action_mask}\n"
                f"  action_is_charging: {action_is_charging}\n"
                f"  action_charge_durations: {action_charge_durations}"
            )
        
        data.action_graph_features = self._build_action_graph_features(
            env,
            feasible_action_to_node_map,
            feasible_action_is_charging,
            feasible_action_durations,
            active_truck_idx if env.active_truck_id in truck_id_to_idx else None,
        )
        
        # print(f'action_local_index: {data.action_local_index}')
        # print(f'action_node_type: {data.action_node_type}')
        # print(f'action_is_charging: {data.action_is_charging}')
        # print(f'action_charge_durations: {data.action_charge_durations}')
        # print(f'can_charge_here: {data.can_charge_here}')
        # print(f'num_actions: {data.num_actions}')
        
        # print("Feasible action mask:", data.feasible_action_mask)
        # print("feasible_indices", feasible_indices)
        # print("feasible_action_to_node_map", feasible_action_to_node_map)
        # print("feasible_action_is_charging", feasible_action_is_charging)
        # print("feasible_action_durations", feasible_action_durations)
                    
        return data

    # ==================== Node Feature Functions ====================

    def _get_action_feasibility_diagnostics(self, env, active_truck, charger_node_to_idx) -> str:
        """Generate detailed diagnostics about why actions may be infeasible."""
        diagnostics = []
        diagnostics.append(f"Active Truck {active_truck.truck_id} State:")
        diagnostics.append(f"  Location: node {active_truck.current_node}")
        diagnostics.append(f"  Battery: {active_truck.current_battery:.2f}/{active_truck.battery_capacity:.2f} kWh ({active_truck.get_battery_percentage():.1f}%)")
        diagnostics.append(f"  Must leave charger: {active_truck.must_leave_charger}")
        diagnostics.append(f"  Is charging: {active_truck.is_charging}")
        diagnostics.append(f"  Route destination: {active_truck.route_destination}")
        
        next_delivery = active_truck.get_next_delivery_target()
        diagnostics.append(f"  Next delivery: {next_delivery}")
        
        if next_delivery is not None:
            energy_to_delivery = env.transport_graph.get_path_energy(active_truck.current_node, next_delivery)
            diagnostics.append(f"  Energy to next delivery: {energy_to_delivery:.2f} kWh")
        
        # Get energy safety factor
        energy_safety_factor = 1.0
        if hasattr(env, 'traffic_config') and env.traffic_config['enable_traffic'] and env.traffic_config['enable_energy_uncertainty']:
            energy_safety_factor = env.traffic_config['max_energy_multiplier']
        diagnostics.append(f"  Energy safety factor: {energy_safety_factor:.2f}x")
        
        # Check nearest chargers
        if charger_node_to_idx:
            charger_distances = []
            for charger_id in sorted(charger_node_to_idx.keys())[:5]:  # First 5 chargers
                energy = env.transport_graph.get_path_energy(active_truck.current_node, charger_id)
                max_energy = energy * energy_safety_factor
                feasible = max_energy < active_truck.current_battery and not np.isinf(energy)
                charger_distances.append(f"    Charger {charger_id}: {energy:.2f} kWh (×{energy_safety_factor:.2f}={max_energy:.2f}) - {'✓ feasible' if feasible else '✗ infeasible'}")
            diagnostics.append(f"  Nearest chargers:")
            diagnostics.extend(charger_distances)
        
        # Check at charger status
        at_charger = active_truck.current_node in charger_node_to_idx
        diagnostics.append(f"  At charger: {at_charger}")
        if at_charger:
            charger_id = active_truck.current_node
            occupancy = len(env.charging_station.charger_occupancy.get(charger_id, []))
            capacity = env.charging_station.charger_capacity.get(charger_id, 1)
            queue_len = len(env.charging_station.charger_waitlist.get(charger_id, []))
            diagnostics.append(f"    Occupancy: {occupancy}/{capacity}, Queue: {queue_len}")
        
        return "\n".join(diagnostics)

    def _get_truck_node_features(self, truck, env) -> list:
        """
        Get feature vector for a truck node.
        
        Features (all normalized):
        - Node type (normalized by number of node types)
        - Current position (normalized by number of trucks)
        - Battery level (normalized by capacity, 0-1)
        - Battery percentage (normalized, 0-1)
        - Truck state (one-hot encoded: ready, routing, waiting_to_charge, charging)
        - Must leave charger (binary: 1 if must leave, 0 otherwise)
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
        elif truck.current_node in env.charging_station.charger_waitlist and truck.truck_id in env.charging_station.charger_waitlist[truck.current_node]:
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
            # float(truck.must_leave_charger),  # Must leave charger flag (binary)
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
        charger_capacity = env.charging_station.charger_capacity[node_id] if node_id in env.charging_station.charger_capacity else 1
        charger_occupancy_list = env.charging_station.charger_occupancy[node_id] if node_id in env.charging_station.charger_occupancy else []
        charger_occupancy = len(charger_occupancy_list)  # Number of trucks currently charging
        charger_queue = env.charging_station.charger_waitlist[node_id] if node_id in env.charging_station.charger_waitlist else []
        
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

    def _build_action_graph_features(
        self,
        env,
        action_to_node_map: list,
        action_is_charging: list,
        action_charge_durations: list,
        active_truck_idx: int,
    ) -> torch.Tensor:
        """
        Build action graph features: [normalized_action_type, resulting_soc, charge_duration_norm]
        
        Action types:
        - 1/3: routing to delivery node
        - 2/3: routing to charger node
        - 3/3: charging at current location
        
        Resulting SOC: battery level after taking the action (normalized 0-1)
        """
        if not action_to_node_map:
            raise ValueError(
                "INTERNAL ERROR: _build_action_graph_features called with empty action_to_node_map. "
                "This should have been caught earlier in get_state_GNN. "
                f"Active truck: {env.active_truck_id if hasattr(env, 'active_truck_id') else 'unknown'}"
            )
        
        # Get active truck info
        current_battery = 0.0
        battery_capacity = 1.0
        current_location = -1
        
        assert env.active_truck_id is not None and env.active_truck_id < len(env.trucks), "Active truck ID is invalid"
            
        active_truck = env.trucks[env.active_truck_id]
        current_battery = active_truck.current_battery
        battery_capacity = active_truck.battery_capacity
        current_location = active_truck.current_node
        
        max_charge_duration = max(env.charging_config["charge_durations"])
        features = []
        for action_idx, (node_id, is_charging) in enumerate(action_to_node_map):
            charge_duration = action_charge_durations[action_idx]
            charge_duration_norm = (
                float(charge_duration) / float(max_charge_duration) if max_charge_duration > 0 else 0.0
            )
            # Determine action type
            if action_is_charging[action_idx]:
                action_type_norm = 3.0 / 3.0  # Charging action
                # After charging, battery will depend on charge duration (using curve model)
                charger_type = env.charging_station.charger_type[current_location]
                charging_config = env.config["charging"]
                if charger_type == "DCFast":
                    charger_config_type = charging_config["dcfast"]
                else:
                    charger_config_type = charging_config["level2"]
                
                # Add global use_realistic_curve flag to charger config
                charger_config_with_curve = charger_config_type.copy()
                charger_config_with_curve["use_realistic_curve"] = charging_config["use_realistic_curve"]
                
                # Calculate resulting SOC using curve model
                # Clamp to [0.0, 1.0] to handle any floating point precision issues
                initial_soc = min(1.0, max(0.0, current_battery / battery_capacity)) if battery_capacity > 0 else 0.0
                charge_amount, _ = env.charging_curve_model.calculate_charge(
                    initial_soc=initial_soc,
                    charge_hours=charge_duration,
                    battery_capacity=battery_capacity,
                    charger_config=charger_config_with_curve,
                    charger_type=charger_type
                )
                resulting_soc = min(1.0, (current_battery + charge_amount) / battery_capacity) if battery_capacity > 0 else 0.0
                # print(f'Action {action_idx}: Charging for {charge_duration} hours at node {node_id}, resulting_soc: {resulting_soc:.2f}')
            else:
                # Check if node is a delivery or charger
                is_delivery_node = node_id not in env.charging_nodes if node_id >= 0 else False
                
                if is_delivery_node:
                    action_type_norm = 1.0 / 3.0  # Routing to delivery
                else:
                    action_type_norm = 2.0 / 3.0  # Routing to charger
                
                # Calculate resulting SOC after routing
                if node_id >= 0 and current_location >= 0:
                    energy_consumed = env.transport_graph.get_path_energy(current_location, node_id)
                    if battery_capacity > 0:
                        resulting_soc = max(0.0, (current_battery - energy_consumed) / battery_capacity)
                    else:
                        resulting_soc = 0.0
                    # print(f'Action {action_idx}: Routing to node {node_id} from {current_location}, energy_consumed: {energy_consumed:.2f}, resulting_soc: {resulting_soc:.2f}')
                    
                else:
                    raise ValueError("Invalid node_id or current_location for routing action")
                    # # Invalid action or no path
                    # resulting_soc = 0.0
            
            features.append([action_type_norm, resulting_soc, charge_duration_norm])
        
        return torch.tensor(features, dtype=torch.float32, device=self.device)

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
