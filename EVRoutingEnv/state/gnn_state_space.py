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

from EVRoutingEnv.state.gnn_utils import feasible_mask_to_numpy

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
        # Action graph feature dimension: [action_type_norm, resulting_soc, charge_duration_norm]
        self.action_feature_dim = 3
        
        self.BIDIRECTIONAL_EDGES = True
        self.FILTER_CHARGERS = False  # Filter to top 3 chargers based on fitness

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
                # READY: connect to delivery/deliveries and all feasible chargers
                
                # Get delivery target(s) - handle both sequential and flexible modes
                next_delivery = truck.get_next_delivery_target()
                
                if truck.enable_flexible_delivery_order:
                    # Flexible mode: connect to all remaining deliveries (top-charger-style openness)
                    remaining_deliveries = next_delivery if isinstance(next_delivery, list) else []
                    
                    for delivery_node in remaining_deliveries:
                        if delivery_node not in delivery_node_to_idx:
                            continue
                        
                        delivery_idx = delivery_node_to_idx[delivery_node]
                        
                        # Check if already at delivery location (0 energy/time)
                        if delivery_node == current_location:
                            energy, time = 0.0, 0.0
                            energy_inv, time_inv = 0.0, 0.0
                        else:
                            energy = env.transport_graph.get_path_energy(current_location, delivery_node)
                            energy_inv = env.transport_graph.get_path_energy(delivery_node, current_location)
                            time = env.transport_graph.get_time_distance(current_location, delivery_node)
                            time_inv = env.transport_graph.get_time_distance(delivery_node, current_location)
                            
                            # Debug: Check for asymmetric paths
                            if np.isinf(energy) and not np.isinf(energy_inv):
                                print(f"[GNN State] ERROR: Asymmetric path! {current_location}->{delivery_node} is inf but reverse is {energy_inv}")
                            if np.isinf(energy_inv) and not np.isinf(energy):
                                print(f"[GNN State] ERROR: Asymmetric path! {delivery_node}->{current_location} is inf but forward path exists")
                                print(f"  Forward: energy={energy}, time={time}")
                                print(f"  Reverse: energy_inv={energy_inv}, time_inv={time_inv}")
                                print(f"  This suggests a directed graph issue or missing edges in transport_graph")
                        
                        # Only add edge if energy is feasible
                        max_energy_needed = energy * energy_safety_factor
                        if max_energy_needed < current_battery and not np.isinf(energy):
                            edge_dict[('truck', 'to', 'delivery')]['edge_index'].append([truck_idx, delivery_idx])
                            edge_dict[('truck', 'to', 'delivery')]['edge_attr'].append([energy/1000.0, time/self.max_time])
                            
                            if self.BIDIRECTIONAL_EDGES:
                                # Only add reverse edge if it's valid (not inf/nan)
                                # Inf means no path exists, so we shouldn't add the edge
                                if not (np.isnan(energy_inv) or np.isinf(energy_inv) or 
                                       np.isnan(time_inv) or np.isinf(time_inv)):
                                    energy_norm = energy_inv / 1000.0
                                    time_norm = time_inv / self.max_time if self.max_time > 0 else 0.0
                                    edge_dict[('delivery', 'to', 'truck')]['edge_index'].append([delivery_idx, truck_idx])
                                    edge_dict[('delivery', 'to', 'truck')]['edge_attr'].append([energy_norm, time_norm])
                                else:
                                    # Log this for debugging
                                    if not (np.isinf(energy) or np.isinf(time)):
                                        print(f"[GNN State] WARNING: Skipping reverse edge {delivery_node}->{current_location} (inf) while forward exists")
                else:
                    # Sequential mode: connect to next delivery only
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
                            
                            # # Debug: Check for asymmetric paths
                            # if np.isinf(energy_inv) and not np.isinf(energy):
                            #     print(f"[GNN State] ERROR: Asymmetric path (sequential)! {next_delivery}->{current_location} is inf but forward path exists")
                            #     print(f"  Forward: energy={energy}, time={time}")
                            #     print(f"  Reverse: energy_inv={energy_inv}, time_inv={time_inv}")

                        # Only add edge if energy is feasible (< current battery with safety margin)
                        # Account for worst-case energy consumption due to uncertainty
                        max_energy_needed = energy * energy_safety_factor
                        if max_energy_needed < current_battery and not np.isinf(energy):
                            edge_dict[('truck', 'to', 'delivery')]['edge_index'].append([truck_idx, delivery_idx])
                            edge_dict[('truck', 'to', 'delivery')]['edge_attr'].append([energy/1000.0, time/self.max_time])
                            
                            if self.BIDIRECTIONAL_EDGES:
                                # Only add reverse edge if it's valid (not inf/nan)
                                # Inf means no path exists, so we shouldn't add the edge
                                if not (np.isnan(energy_inv) or np.isinf(energy_inv) or 
                                       np.isnan(time_inv) or np.isinf(time_inv)):
                                    energy_norm = energy_inv / 1000.0
                                    time_norm = time_inv / self.max_time if self.max_time > 0 else 0.0
                                    edge_dict[('delivery', 'to', 'truck')]['edge_index'].append([delivery_idx, truck_idx])
                                    edge_dict[('delivery', 'to', 'truck')]['edge_attr'].append([energy_norm, time_norm])
                                # else:
                                #     # Log this for debugging
                                #     if not (np.isinf(energy) or np.isinf(time)):
                                #         print(f"[GNN State] WARNING: Skipping reverse edge {next_delivery}->{current_location} (inf) while forward exists")                # Connect to all chargers (if feasible with current battery)
                # Apply charger filtering if enabled and in sequential delivery mode
                chargers_to_connect = list(charger_node_to_idx.items())
                
                if self.FILTER_CHARGERS and not truck.enable_flexible_delivery_order and next_delivery is not None:
                    # Calculate fitness for each charger based on whether it's an intermediate stop
                    charger_fitness = []
                    
                    # Get direct distance from current location to next delivery
                    direct_distance = env.transport_graph.get_path_energy(current_location, next_delivery)
                    
                    for charger_id, charger_idx in charger_node_to_idx.items():
                        if charger_id == current_location:
                            # Current location gets highest priority (0 distance)
                            charger_fitness.append((charger_id, charger_idx, -1.0))  # Negative for highest priority
                            continue
                        
                        # Calculate detour: distance via charger vs direct distance
                        dist_to_charger = env.transport_graph.get_path_energy(current_location, charger_id)
                        dist_from_charger = env.transport_graph.get_path_energy(charger_id, next_delivery)
                        
                        # Skip if any distance is infinite
                        if np.isinf(dist_to_charger) or np.isinf(dist_from_charger) or np.isinf(direct_distance):
                            continue
                        
                        # Calculate total distance via charger
                        total_via_charger = dist_to_charger + dist_from_charger
                        
                        # Fitness metric: extra distance taken by going via this charger
                        # Lower is better (chargers on the way to delivery have lower detour)
                        detour = total_via_charger - direct_distance
                        
                        # Check if charger is feasible with current battery
                        max_energy_needed = dist_to_charger * energy_safety_factor
                        if max_energy_needed < current_battery:
                            charger_fitness.append((charger_id, charger_idx, detour))
                    
                    # Sort by fitness (detour) and keep top 3
                    charger_fitness.sort(key=lambda x: x[2])
                    chargers_to_connect = [(cid, cidx) for cid, cidx, _ in charger_fitness[:3]]
                    
                    if self.verbose and len(charger_fitness) > 3:
                        print(f"[GNN State] Filtered chargers for truck {truck.truck_id}: {len(charger_fitness)} -> {len(chargers_to_connect)}")
                        print(f"  Top 3 chargers: {[(cid, f'{detour:.1f}') for cid, _, detour in charger_fitness[:3]]}")
                
                # Add edges for selected chargers
                for charger_id, charger_idx in chargers_to_connect:
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

                # Progressively reduce safety factor for routing actions if needed
                # This allows risky routing when truck has limited options
                routing_safety_factor = energy_safety_factor
                min_routing_safety = 0.5  # Minimum safety factor for routing
                if active_truck.enable_flexible_delivery_order:
                    remaining_count = len(next_delivery) if isinstance(next_delivery, list) else (1 if next_delivery is not None else 0)
                    if remaining_count <= 2:
                        # Late-stage: allow a lower floor to avoid dead-ends near completion
                        min_routing_safety = 0.35
                
                # Check if any routing action would be feasible with current safety factor
                def check_routing_feasible(safety_factor):
                    """Check if at least one routing action is feasible with given safety factor."""
                    # Check chargers
                    for cid in charger_node_to_idx.keys():
                        if cid != current_location:
                            energy = env.transport_graph.get_path_energy(current_location, cid)
                            if energy * safety_factor < current_battery and not np.isinf(energy):
                                return True
                    # Check delivery/deliveries (handle both sequential and flexible modes)
                    if next_delivery is not None:
                        # Handle flexible mode (list of deliveries)
                        if isinstance(next_delivery, list):
                            for delivery_node in next_delivery:
                                energy = env.transport_graph.get_path_energy(current_location, delivery_node)
                                if energy * safety_factor < current_battery and not np.isinf(energy):
                                    return True
                        else:
                            # Sequential mode (single delivery)
                            energy = env.transport_graph.get_path_energy(current_location, next_delivery)
                            if energy * safety_factor < current_battery and not np.isinf(energy):
                                return True
                    return False
                
                # If must_leave and no routing actions feasible, reduce safety factor
                if must_leave and not must_charge_now:
                    while not check_routing_feasible(routing_safety_factor) and routing_safety_factor >= min_routing_safety:
                        routing_safety_factor -= 0.05
                        if self.verbose and routing_safety_factor >= min_routing_safety:
                            print(f"  [Routing] No feasible routes with safety {routing_safety_factor + 0.05:.2f}, trying {routing_safety_factor:.2f}")
                    
                    if routing_safety_factor < energy_safety_factor and self.verbose:
                        print(f"  [Routing] Reduced safety factor from {energy_safety_factor:.2f} to {routing_safety_factor:.2f} for routing actions")

                navigation_entries = []  # (action_idx, node_id, is_charger)

                # Actions 0 to N-1: Go to charger i (must match environment action order)
                # Note: Include current location to match environment's action indexing
                progress_scores = []  # (action_idx, score)
                progress_raw = []  # track raw deltas for diagnostics

                for charger_id in sorted(charger_node_to_idx.keys()):
                    if charger_id == current_location:
                        # Current location - routing here is always infeasible
                        action_to_node_map.append((charger_id, False))
                        feasible_action_mask.append(False)
                        _append_action_metadata(charger_id, False)
                        action_charge_durations.append(0.0)
                        navigation_entries.append((len(action_to_node_map) - 1, charger_id, True))
                    else:
                        energy = env.transport_graph.get_path_energy(current_location, charger_id)
                        max_energy_needed = energy * routing_safety_factor
                        is_energy_feasible = max_energy_needed < current_battery and not np.isinf(energy)
                        # Disable routing if truck must charge now
                        is_feasible = is_energy_feasible and not must_charge_now

                        # Flexible mode: allow any charger only if it enables progress
                        if is_feasible and active_truck.enable_flexible_delivery_order:
                            # In flexible mode, allow routing to any reachable charger to recharge; avoid over-pruning
                            # Only apply progress gating when already at a charger (handled elsewhere)
                            pass

                        action_to_node_map.append((charger_id, False))
                        feasible_action_mask.append(is_feasible)
                        _append_action_metadata(charger_id, False)
                        action_charge_durations.append(0.0)
                        navigation_entries.append((len(action_to_node_map) - 1, charger_id, True))

                        # Simple progress score: how much this charger reduces min delivery energy
                        if active_truck.enable_flexible_delivery_order:
                            if isinstance(next_delivery, list) and next_delivery:
                                min_from_here = min(
                                    [env.transport_graph.get_path_energy(current_location, d) for d in next_delivery]
                                )
                                min_from_charger = min(
                                    [env.transport_graph.get_path_energy(charger_id, d) for d in next_delivery]
                                )
                                energy_to_charger = energy
                                detour = energy_to_charger + min_from_charger - min_from_here
                                delta = min_from_here - min_from_charger  # positive is progress
                                score = delta - max(detour, 0) * 0.25  # penalize detour if it grows distance
                                progress_scores.append((len(action_to_node_map) - 1, score))
                                progress_raw.append(score)
                        else:
                            if next_delivery is not None:
                                energy_here = env.transport_graph.get_path_energy(current_location, next_delivery)
                                energy_after = env.transport_graph.get_path_energy(charger_id, next_delivery)
                                delta = energy_here - energy_after
                                progress_scores.append((len(action_to_node_map) - 1, delta))
                                progress_raw.append(delta)

                # Delivery Actions: Handle both sequential and flexible modes
                if active_truck.enable_flexible_delivery_order:
                    # Flexible mode: Create action for each delivery in sequence (to match env action space)
                    # Actions are indexed by position in delivery_sequence (excluding depot)
                    remaining_deliveries = next_delivery if isinstance(next_delivery, list) else []
                    
                    if self.verbose:
                        print(f'[FlexibleMode] Remaining deliveries: {remaining_deliveries}')
                        print(f'[FlexibleMode] Delivery sequence: {active_truck.delivery_sequence}')
                    
                    # Create actions for each possible delivery slot (num_stops actions)
                    for i in range(env.num_stops):
                        # Map action index to delivery node in sequence (skip depot at index 0)
                        if i + 1 < len(active_truck.delivery_sequence):
                            delivery_node = active_truck.delivery_sequence[i + 1]
                            
                            # Check if this delivery is still remaining (not yet delivered)
                            if delivery_node in remaining_deliveries:
                                # Validate feasibility for this delivery
                                energy_to_delivery = env.transport_graph.get_path_energy(current_location, delivery_node)
                                max_energy_to_delivery = energy_to_delivery * routing_safety_factor
                                is_energy_feasible = max_energy_to_delivery < current_battery
                                
                                # Check if truck can continue after this delivery
                                can_continue_after_delivery = False
                                progress_guard = False
                                if is_energy_feasible:
                                    battery_after_delivery = current_battery - max_energy_to_delivery
                                    
                                    # Get remaining deliveries after this one
                                    other_remaining = [d for d in remaining_deliveries if d != delivery_node]
                                    has_more_deliveries = len(other_remaining) > 0
                                    
                                    if not has_more_deliveries:
                                        can_continue_after_delivery = True
                                    else:
                                        # Check if can reach any charger after delivery
                                        for charger_id in charger_node_to_idx.keys():
                                            energy_to_charger = env.transport_graph.get_path_energy(delivery_node, charger_id)
                                            max_energy_to_charger = energy_to_charger * routing_safety_factor
                                            if battery_after_delivery > max_energy_to_charger:
                                                can_continue_after_delivery = True
                                                break

                                    # In flexible mode while leaving a charger, allow deliveries that either keep you able
                                    # to continue afterward or pass a greedy completion check; avoid over-pruning must_leave
                                    if at_charger and must_leave:
                                        progress_guard = can_continue_after_delivery or (
                                            has_more_deliveries
                                            and self._can_complete_route_from(
                                                env,
                                                delivery_node,
                                                battery_after_delivery,
                                                other_remaining,
                                                active_truck.battery_capacity,
                                                routing_safety_factor,
                                            )
                                        )
                                    else:
                                        progress_guard = can_continue_after_delivery or must_leave

                                is_feasible = is_energy_feasible and not must_charge_now and progress_guard
                                action_to_node_map.append((delivery_node, False))
                                feasible_action_mask.append(is_feasible)
                                _append_action_metadata(delivery_node, False)
                                action_charge_durations.append(0.0)
                                navigation_entries.append((len(action_to_node_map) - 1, delivery_node, False))

                                if active_truck.enable_flexible_delivery_order and is_feasible:
                                    # Delivery progress score: how many deliveries left and min charger reachability after
                                    other_remaining = [d for d in remaining_deliveries if d != delivery_node]
                                    score = len(remaining_deliveries) - len(other_remaining)
                                    progress_scores.append((len(action_to_node_map) - 1, score))
                                    progress_raw.append(score)
                                
                                if self.verbose:
                                    print(f'  Delivery action {i}: node {delivery_node}, energy {energy_to_delivery:.2f} kWh, feasible: {is_feasible}')
                            else:
                                # Delivery already completed - action is infeasible
                                action_to_node_map.append((delivery_node, False))
                                feasible_action_mask.append(False)
                                _append_action_metadata(delivery_node, False)
                                action_charge_durations.append(0.0)
                                navigation_entries.append((len(action_to_node_map) - 1, delivery_node, False))
                                
                                if self.verbose:
                                    print(f'  Delivery action {i}: node {delivery_node} already delivered, infeasible')
                        else:
                            # No delivery at this position (fewer deliveries than num_stops)
                            action_to_node_map.append((-1, False))
                            feasible_action_mask.append(False)
                            _append_action_metadata(-1, False)
                            action_charge_durations.append(0.0)
                            navigation_entries.append((len(action_to_node_map) - 1, -1, False))
                            
                            if self.verbose:
                                print(f'  Delivery action {i}: no delivery at this position, infeasible')
                else:
                    # Sequential mode: Single action for next delivery
                    if next_delivery is not None:
                        energy_to_delivery = env.transport_graph.get_path_energy(current_location, next_delivery)
                        max_energy_to_delivery = energy_to_delivery * routing_safety_factor
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
                                    max_energy_to_charger = energy_to_charger * routing_safety_factor
                                    if battery_after_delivery > max_energy_to_charger:
                                        can_continue_after_delivery = True
                                        break
                            
                            if self.verbose:
                                print(f'  can_continue_after_delivery: {can_continue_after_delivery} (battery after delivery: {battery_after_delivery:.2f} kWh)')
                            
                        # Disable routing if truck must charge now
                        # OR if truck would be stranded after delivery (UNLESS truck must leave charger)
                        # If must_leave=True, allow risky routing since truck has no choice but to leave
                        is_feasible = is_energy_feasible and not must_charge_now and (can_continue_after_delivery or must_leave)
                        action_to_node_map.append((next_delivery, False))
                        feasible_action_mask.append(is_feasible)
                        _append_action_metadata(next_delivery, False)
                        action_charge_durations.append(0.0)
                        navigation_entries.append((len(action_to_node_map) - 1, next_delivery, False))
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
                        # In flexible mode, next_delivery is a list - use closest delivery for min energy calculation
                        if isinstance(next_delivery, list):
                            # Flexible mode: calculate energy to closest delivery
                            energy_to_delivery = min([env.transport_graph.get_path_energy(current_location, d) for d in next_delivery]) if next_delivery else float('inf')
                            next_delivery_str = str(next_delivery)
                        else:
                            # Sequential mode: single delivery
                            energy_to_delivery = env.transport_graph.get_path_energy(current_location, next_delivery)
                            next_delivery_str = str(next_delivery)
                        
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
                            print(f"  Energy to next delivery {next_delivery_str}: {energy_to_delivery:.2f} kWh")
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
                                f"Next delivery {next_delivery_str} requires {energy_to_delivery:.2f} kWh base energy.\n"
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

                # If flexible delivery order and charging is impossible here, unlock navigation after charger
                if (
                    active_truck.enable_flexible_delivery_order
                    and at_charger
                    and not must_leave
                    and not can_charge_here
                ):
                    # Re-enable navigation actions using same routing safety factor but without mandatory charge
                    remaining_deliveries = next_delivery if isinstance(next_delivery, list) else []

                    for nav_idx, nav_node_id, nav_is_charger in navigation_entries:
                        if nav_is_charger:
                            if nav_node_id == current_location:
                                feasible_action_mask[nav_idx] = False
                                continue
                            energy = env.transport_graph.get_path_energy(current_location, nav_node_id)
                            max_energy_needed = energy * routing_safety_factor
                            is_energy_feasible = max_energy_needed < current_battery and not np.isinf(energy)
                            if is_energy_feasible:
                                if active_truck.enable_flexible_delivery_order:
                                    feasible_action_mask[nav_idx] = self._can_progress_via_charger(
                                        env=env,
                                        charger_id=nav_node_id,
                                        deliveries_after=remaining_deliveries,
                                        battery_after_travel=current_battery - max_energy_needed,
                                        energy_safety_factor=routing_safety_factor,
                                    )
                                else:
                                    feasible_action_mask[nav_idx] = True
                            else:
                                feasible_action_mask[nav_idx] = False
                        else:
                            # Delivery navigation entry
                            if nav_node_id == -1 or nav_node_id not in remaining_deliveries:
                                feasible_action_mask[nav_idx] = False
                                continue
                            energy_to_delivery = env.transport_graph.get_path_energy(current_location, nav_node_id)
                            max_energy_to_delivery = energy_to_delivery * routing_safety_factor
                            is_energy_feasible = max_energy_to_delivery < current_battery and not np.isinf(energy_to_delivery)
                            can_continue_after_delivery = False
                            if is_energy_feasible:
                                battery_after_delivery = current_battery - max_energy_to_delivery
                                other_remaining = [d for d in remaining_deliveries if d != nav_node_id]
                                if not other_remaining:
                                    can_continue_after_delivery = True
                                else:
                                    for charger_id in charger_node_to_idx.keys():
                                        energy_to_charger = env.transport_graph.get_path_energy(nav_node_id, charger_id)
                                        max_energy_to_charger = energy_to_charger * routing_safety_factor
                                        if battery_after_delivery > max_energy_to_charger:
                                            can_continue_after_delivery = True
                                            break
                            feasible_action_mask[nav_idx] = is_energy_feasible and (can_continue_after_delivery or must_leave)

                # Progress-aware pruning: keep top actions by score when multiple feasible
                if active_truck.enable_flexible_delivery_order and any(feasible_action_mask):
                    scored = [(idx, score) for idx, score in progress_scores if feasible_action_mask[idx]]
                    if scored:
                        # Penalize negative progress hard
                        for idx, score in scored:
                            if score < 0:
                                feasible_action_mask[idx] = False

                        # Recompute with remaining feasible
                        scored = [(idx, score) for idx, score in scored if feasible_action_mask[idx]]
                        if scored:
                            scores_only = [s for _, s in scored]
                            rng = np.random.default_rng()
                            rng.shuffle(scored)  # randomize tie handling
                            # Keep top 60% by score (drop bottom 40%) for stronger bias
                            cutoff = np.percentile(scores_only, 40)
                            kept_any = False
                            for idx, score in scored:
                                if score >= cutoff:
                                    kept_any = True
                                else:
                                    feasible_action_mask[idx] = False
                            # Ensure at least one action remains
                            if not kept_any:
                                best_idx, _ = max(scored, key=lambda x: x[1])
                                feasible_action_mask[best_idx] = True

                # If still no feasible actions in flexible mode (not at charger), retry navigation with relaxed safety
                if (
                    active_truck.enable_flexible_delivery_order
                    and not at_charger
                    and not any(feasible_action_mask)
                ):
                    relaxed_safety = max(min_routing_safety, routing_safety_factor * 0.7)
                    remaining_deliveries = next_delivery if isinstance(next_delivery, list) else []

                    for nav_idx, nav_node_id, nav_is_charger in navigation_entries:
                        if nav_is_charger:
                            if nav_node_id == current_location:
                                feasible_action_mask[nav_idx] = False
                                continue
                            energy = env.transport_graph.get_path_energy(current_location, nav_node_id)
                            max_energy_needed = energy * relaxed_safety
                            is_energy_feasible = max_energy_needed < current_battery and not np.isinf(energy)
                            feasible_action_mask[nav_idx] = is_energy_feasible
                        else:
                            if nav_node_id == -1 or nav_node_id not in remaining_deliveries:
                                feasible_action_mask[nav_idx] = False
                                continue
                            energy_to_delivery = env.transport_graph.get_path_energy(current_location, nav_node_id)
                            max_energy_to_delivery = energy_to_delivery * relaxed_safety
                            is_energy_feasible = max_energy_to_delivery < current_battery and not np.isinf(energy_to_delivery)
                            can_continue_after_delivery = False
                            if is_energy_feasible:
                                battery_after_delivery = current_battery - max_energy_to_delivery
                                other_remaining = [d for d in remaining_deliveries if d != nav_node_id]
                                if not other_remaining:
                                    can_continue_after_delivery = True
                                else:
                                    for charger_id in charger_node_to_idx.keys():
                                        energy_to_charger = env.transport_graph.get_path_energy(nav_node_id, charger_id)
                                        max_energy_to_charger = energy_to_charger * relaxed_safety
                                        if battery_after_delivery > max_energy_to_charger:
                                            can_continue_after_delivery = True
                                            break
                            feasible_action_mask[nav_idx] = is_energy_feasible and (can_continue_after_delivery or must_leave)

                # Escape hatch: at charger with must_leave in flexible mode and no actions — allow any energy-feasible nav
                if (
                    active_truck.enable_flexible_delivery_order
                    and at_charger
                    and must_leave
                    and not any(feasible_action_mask)
                ):
                    relaxed_safety = max(min_routing_safety, routing_safety_factor * 0.7)
                    remaining_deliveries = next_delivery if isinstance(next_delivery, list) else []

                    for nav_idx, nav_node_id, nav_is_charger in navigation_entries:
                        if nav_is_charger:
                            if nav_node_id == current_location:
                                feasible_action_mask[nav_idx] = False
                                continue
                            energy = env.transport_graph.get_path_energy(current_location, nav_node_id)
                            max_energy_needed = energy * relaxed_safety
                            feasible_action_mask[nav_idx] = max_energy_needed < current_battery and not np.isinf(energy)
                        else:
                            if nav_node_id == -1 or nav_node_id not in remaining_deliveries:
                                feasible_action_mask[nav_idx] = False
                                continue
                            energy_to_delivery = env.transport_graph.get_path_energy(current_location, nav_node_id)
                            max_energy_to_delivery = energy_to_delivery * relaxed_safety
                            is_energy_feasible = max_energy_to_delivery < current_battery and not np.isinf(energy_to_delivery)
                            feasible_action_mask[nav_idx] = is_energy_feasible

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
            # Handle flexible order: next_delivery can be a list
            if isinstance(next_delivery, list):
                if next_delivery:
                    energies = [env.transport_graph.get_path_energy(active_truck.current_node, d) for d in next_delivery]
                    min_energy = min(energies) if energies else float('inf')
                    diagnostics.append(
                        f"  Energy to next deliveries: min {min_energy:.2f} kWh across {len(next_delivery)} options"
                    )
                else:
                    diagnostics.append("  Energy to next deliveries: none (empty list)")
            else:
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

    def _can_progress_via_charger(
        self,
        env,
        charger_id: int,
        deliveries_after: list,
        battery_after_travel: float,
        energy_safety_factor: float,
    ) -> bool:
        """Check that routing to a charger keeps the truck viable in flexible mode.

        A charger is allowed if, after arriving with ``battery_after_travel``:
        - the truck can finish all remaining deliveries directly, or
        - it can reach any charger (with safety), or
        - it can reach some delivery and then a charger afterwards.
        """

        # No remaining deliveries: any reachable charger is fine
        if not deliveries_after:
            return True

        # Case 1: full route feasible after reaching this charger (best case)
        battery_cap = env.trucks[env.active_truck_id].battery_capacity
        if self._can_complete_route_from(
            env,
            charger_id,
            battery_after_travel,
            deliveries_after,
            battery_cap,
            energy_safety_factor,
        ):
            return True

        # Case 2: can leave charger to any other charger (avoid getting stuck)
        for next_charger in env.charging_nodes:
            if next_charger == charger_id:
                continue
            energy = env.transport_graph.get_path_energy(charger_id, next_charger)
            if np.isinf(energy):
                continue
            if battery_after_travel > energy * energy_safety_factor:
                return True

        # Case 3: can reach some delivery and then a charger
        for delivery_id in deliveries_after:
            energy_to_delivery = env.transport_graph.get_path_energy(charger_id, delivery_id)
            if np.isinf(energy_to_delivery):
                continue
            battery_after_delivery = battery_after_travel - energy_to_delivery * energy_safety_factor
            if battery_after_delivery <= 0:
                continue
            for next_charger in env.charging_nodes:
                energy_to_charger = env.transport_graph.get_path_energy(delivery_id, next_charger)
                if np.isinf(energy_to_charger):
                    continue
                if battery_after_delivery > energy_to_charger * energy_safety_factor:
                    return True

        return False

    def _can_complete_route_from(
        self,
        env,
        start_location: int,
        battery_at_start: float,
        remaining_deliveries: list,
        battery_capacity: float,
        energy_safety_factor: float,
    ) -> bool:
        """Greedy feasibility check to see if remaining deliveries are completable.

        Tries to reach each delivery; if not enough battery, attempts a detour
        via any charger (with safety factor) and assumes full charge there.
        """

        if not remaining_deliveries:
            return True

        current_loc = start_location
        current_battery = battery_at_start

        for delivery in remaining_deliveries:
            energy_to_delivery = env.transport_graph.get_path_energy(current_loc, delivery)
            if np.isinf(energy_to_delivery):
                return False

            needed = energy_to_delivery * energy_safety_factor
            if current_battery >= needed:
                current_battery -= needed
                current_loc = delivery
                continue

            # Need a charger; see if any reachable charger allows continuing
            can_charge = False
            for charger in env.charging_nodes:
                energy_to_charger = env.transport_graph.get_path_energy(current_loc, charger)
                if np.isinf(energy_to_charger):
                    continue

                if current_battery < energy_to_charger * energy_safety_factor:
                    continue

                energy_charger_to_delivery = env.transport_graph.get_path_energy(charger, delivery)
                if np.isinf(energy_charger_to_delivery):
                    continue

                if battery_capacity >= energy_charger_to_delivery * energy_safety_factor:
                    # Assume full charge then continue
                    current_battery = battery_capacity - energy_charger_to_delivery * energy_safety_factor
                    current_loc = delivery
                    can_charge = True
                    break

            if not can_charge:
                return False

        return True

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

    def get_action_graph(self, env) -> Dict:
        """Convenience wrapper returning action graph metadata.

        Returns:
            Dict with keys:
            - data: full HeteroData graph
            - feasible_action_mask: numpy boolean mask aligned to env action space
            - action_to_node_map: list of (node_id, is_charging_action)
        """
        data = self.get_state_GNN(env)
        return {
            "data": data,
            "feasible_action_mask": feasible_mask_to_numpy(getattr(data, "feasible_action_mask", None)),
            "action_to_node_map": getattr(data, "action_to_node_map", []),
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
