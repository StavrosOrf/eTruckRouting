"""
GNN State Representation for flexible-delivery (VRP-style) control.

Use this when ``enable_flexible_delivery_order`` is enabled; otherwise prefer
``GNNStateSpaceNonFlex`` (with or without detour masking) for sequential runs.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch

from EVRoutingEnv.state.gnn_state_space import GNNStateSpace
from EVRoutingEnv.state.gnn_utils import extract_action_graph


class GNNStateSpaceVRP(GNNStateSpace):
    """GNN state space specialized for flexible delivery ordering (VRP)."""

    def __init__(
        self,
        num_trucks: int,
        num_stops: int,
        max_time: float,
        num_charging_nodes: int,
        max_nodes_in_graph: int = 500,
        vrp_top_k_deliveries: int = 3,
        device: str = "cpu",
        verbose: bool = False,
    ):
        super().__init__(
            num_trucks=num_trucks,
            num_stops=num_stops,
            max_time=max_time,
            num_charging_nodes=num_charging_nodes,
            max_nodes_in_graph=max_nodes_in_graph,
            vrp_top_k_deliveries=vrp_top_k_deliveries,
            device=device,
            verbose=verbose,
        )
        if "depot" not in self.node_type_order:
            self.NODE_TYPE_DEPOT = len(self.node_type_order)
            self.node_type_order = self.node_type_order + ["depot"]
            self.node_type_to_code = {
                node_type: idx for idx, node_type in enumerate(self.node_type_order)
            }

    def get_state_GNN(self, env):  # type: ignore[override]
        data = super().get_state_GNN(env)
        data, depot_node_to_idx = self._add_depot_nodes_and_edges(env, data)
        data = self._update_depot_action_metadata(env, data, depot_node_to_idx)

        active_truck = None
        if getattr(env, "active_truck_id", None) is not None and env.active_truck_id < len(env.trucks):
            active_truck = env.trucks[env.active_truck_id]

        is_flexible_env = bool(getattr(env, "enable_flexible_delivery_order", False))
        is_flexible_truck = bool(active_truck and getattr(active_truck, "enable_flexible_delivery_order", False))
        if not (is_flexible_env or is_flexible_truck):
            raise ValueError(
                "GNNStateSpaceVRP requires flexible delivery ordering. "
                "Use GNNStateSpaceNonFlex for sequential runs."
            )

        expected_actions: Optional[int] = getattr(getattr(env, "action_space", None), "n", None)
        action_count = len(getattr(data, "action_to_node_map", []))
        if expected_actions is not None and action_count != expected_actions:
            raise ValueError(
                f"VRP action map size {action_count} does not match environment action space {expected_actions}."
            )

        # Attach small bits of metadata for downstream debuggers/agents.
        data.delivery_mode = "flexible"
        data.action_space_n = expected_actions
        return data

    def get_action_graph(self, env):
        return extract_action_graph(self, env)

    def _add_depot_nodes_and_edges(self, env, data):
        depot_ids = self._get_depot_ids(env)
        depot_node_to_idx = {node_id: idx for idx, node_id in enumerate(sorted(depot_ids))}
        depot_features_list = [self._get_depot_node_features(node_id, env) for node_id in sorted(depot_ids)]
        if depot_features_list:
            depot_features_array = np.array(depot_features_list, dtype=np.float32)
            data["depot"].x = torch.tensor(depot_features_array, dtype=torch.float32, device=self.device)
        else:
            data["depot"].x = torch.zeros((0, self._delivery_feature_dim), dtype=torch.float32, device=self.device)

        node_id_to_type: Dict[int, tuple] = dict(getattr(data, "node_id_to_type", {}))
        for depot_id, local_idx in depot_node_to_idx.items():
            node_id_to_type[depot_id] = ("depot", local_idx)
        data.node_id_to_type = node_id_to_type

        truck_count = int(data["truck"].x.shape[0]) if "truck" in data.node_types else 0
        delivery_count = int(data["delivery"].x.shape[0]) if "delivery" in data.node_types else 0
        charger_count = int(data["charger"].x.shape[0]) if "charger" in data.node_types else 0
        data.node_type_offsets = {
            "truck": 0,
            "delivery": truck_count,
            "charger": truck_count + delivery_count,
            "depot": truck_count + delivery_count + charger_count,
        }

        if not depot_node_to_idx:
            return data, depot_node_to_idx

        delivery_node_to_idx = {
            node_id: local_idx
            for node_id, (node_type, local_idx) in node_id_to_type.items()
            if node_type == "delivery"
        }
        charger_node_to_idx = {
            node_id: local_idx
            for node_id, (node_type, local_idx) in node_id_to_type.items()
            if node_type == "charger"
        }
        truck_id_to_idx = {
            node_id: local_idx
            for node_id, (node_type, local_idx) in node_id_to_type.items()
            if node_type == "truck"
        }

        max_battery_capacity = max(truck.battery_capacity for truck in env.trucks) if env.trucks else 0.0
        energy_safety_factor = 1.0
        if (
            hasattr(env, "traffic_config")
            and env.traffic_config["enable_traffic"]
            and env.traffic_config["enable_energy_uncertainty"]
        ):
            energy_safety_factor = env.traffic_config["max_energy_multiplier"]

        def _append_edges(edge_type, edge_index_list, edge_attr_list):
            if not edge_index_list:
                return
            edge_index = torch.tensor(
                np.array(edge_index_list).T, dtype=torch.long, device=self.device
            )
            edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32, device=self.device)
            if edge_type in data.edge_types and data[edge_type].edge_index.numel() > 0:
                data[edge_type].edge_index = torch.cat(
                    [data[edge_type].edge_index, edge_index], dim=1
                )
                data[edge_type].edge_attr = torch.cat(
                    [data[edge_type].edge_attr, edge_attr], dim=0
                )
            else:
                data[edge_type].edge_index = edge_index
                data[edge_type].edge_attr = edge_attr

        # Truck <-> Depot edges (READY/ROUTING)
        truck_to_depot_edges = []
        truck_to_depot_attrs = []
        depot_to_truck_edges = []
        depot_to_truck_attrs = []

        for truck in env.trucks:
            if truck.failed or truck.is_complete:
                continue
            truck_idx = truck_id_to_idx.get(truck.truck_id)
            if truck_idx is None:
                continue
            current_location = truck.current_node
            current_battery = truck.current_battery
            charger_waitlist = env.charging_station.charger_waitlist.get(current_location, [])
            if truck.is_charging or truck.truck_id in charger_waitlist:
                continue

            depot_id = truck.delivery_sequence[0] if truck.delivery_sequence else None
            if depot_id is None or depot_id not in depot_node_to_idx:
                continue
            depot_idx = depot_node_to_idx[depot_id]

            if truck.route_destination is None:
                energy = env.transport_graph.get_path_energy(current_location, depot_id)
                time = env.transport_graph.get_time_distance(current_location, depot_id)
                max_energy_needed = energy * energy_safety_factor
                if max_energy_needed < current_battery and not np.isinf(energy):
                    truck_to_depot_edges.append([truck_idx, depot_idx])
                    truck_to_depot_attrs.append([energy / 1000.0, time / self.max_time])
                    if self.BIDIRECTIONAL_EDGES:
                        energy_back = env.transport_graph.get_path_energy(depot_id, current_location)
                        time_back = env.transport_graph.get_time_distance(depot_id, current_location)
                        if not (
                            np.isnan(energy_back)
                            or np.isinf(energy_back)
                            or np.isnan(time_back)
                            or np.isinf(time_back)
                        ):
                            depot_to_truck_edges.append([depot_idx, truck_idx])
                            depot_to_truck_attrs.append([
                                energy_back / 1000.0,
                                time_back / self.max_time,
                            ])
            else:
                if truck.route_destination == depot_id:
                    time_remaining = max(0.0, truck.route_arrival_time - env.global_clock)
                    time_remaining_norm = time_remaining / self.max_time
                    truck_to_depot_edges.append([truck_idx, depot_idx])
                    truck_to_depot_attrs.append([0.0, time_remaining_norm])
                    if self.BIDIRECTIONAL_EDGES:
                        depot_to_truck_edges.append([depot_idx, truck_idx])
                        depot_to_truck_attrs.append([0.0, time_remaining_norm])

        _append_edges(("truck", "to", "depot"), truck_to_depot_edges, truck_to_depot_attrs)
        _append_edges(("depot", "to", "truck"), depot_to_truck_edges, depot_to_truck_attrs)

        # Charger <-> Depot edges
        charger_to_depot_edges = []
        charger_to_depot_attrs = []
        depot_to_charger_edges = []
        depot_to_charger_attrs = []

        for charger_id, charger_idx in charger_node_to_idx.items():
            for depot_id, depot_idx in depot_node_to_idx.items():
                energy = env.transport_graph.get_path_energy(charger_id, depot_id)
                time = env.transport_graph.get_time_distance(charger_id, depot_id)
                if energy * energy_safety_factor <= max_battery_capacity and not np.isinf(energy):
                    charger_to_depot_edges.append([charger_idx, depot_idx])
                    charger_to_depot_attrs.append([energy / 1000.0, time / self.max_time])

                energy_back = env.transport_graph.get_path_energy(depot_id, charger_id)
                time_back = env.transport_graph.get_time_distance(depot_id, charger_id)
                if energy_back * energy_safety_factor <= max_battery_capacity and not np.isinf(energy_back):
                    depot_to_charger_edges.append([depot_idx, charger_idx])
                    depot_to_charger_attrs.append([energy_back / 1000.0, time_back / self.max_time])

        _append_edges(("charger", "to", "depot"), charger_to_depot_edges, charger_to_depot_attrs)
        _append_edges(("depot", "to", "charger"), depot_to_charger_edges, depot_to_charger_attrs)

        # Depot <-> Delivery edges
        depot_to_delivery_edges = []
        depot_to_delivery_attrs = []
        delivery_to_depot_edges = []
        delivery_to_depot_attrs = []

        for depot_id, depot_idx in depot_node_to_idx.items():
            for delivery_id, delivery_idx in delivery_node_to_idx.items():
                energy = env.transport_graph.get_path_energy(depot_id, delivery_id)
                time = env.transport_graph.get_time_distance(depot_id, delivery_id)
                if energy * energy_safety_factor <= max_battery_capacity and not np.isinf(energy):
                    depot_to_delivery_edges.append([depot_idx, delivery_idx])
                    depot_to_delivery_attrs.append([energy / 1000.0, time / self.max_time])

                energy_back = env.transport_graph.get_path_energy(delivery_id, depot_id)
                time_back = env.transport_graph.get_time_distance(delivery_id, depot_id)
                if energy_back * energy_safety_factor <= max_battery_capacity and not np.isinf(energy_back):
                    delivery_to_depot_edges.append([delivery_idx, depot_idx])
                    delivery_to_depot_attrs.append([energy_back / 1000.0, time_back / self.max_time])

        _append_edges(("depot", "to", "delivery"), depot_to_delivery_edges, depot_to_delivery_attrs)
        _append_edges(("delivery", "to", "depot"), delivery_to_depot_edges, delivery_to_depot_attrs)

        # Depot <-> Depot edges
        depot_ids_sorted = sorted(depot_node_to_idx.keys())
        depot_to_depot_edges = []
        depot_to_depot_attrs = []
        for i, depot1_id in enumerate(depot_ids_sorted):
            depot1_idx = depot_node_to_idx[depot1_id]
            for depot2_id in depot_ids_sorted[i + 1 :]:
                depot2_idx = depot_node_to_idx[depot2_id]
                energy = env.transport_graph.get_path_energy(depot1_id, depot2_id)
                time = env.transport_graph.get_time_distance(depot1_id, depot2_id)
                if energy * energy_safety_factor <= max_battery_capacity and not np.isinf(energy):
                    depot_to_depot_edges.append([depot1_idx, depot2_idx])
                    depot_to_depot_attrs.append([energy / 1000.0, time / self.max_time])

                energy_back = env.transport_graph.get_path_energy(depot2_id, depot1_id)
                time_back = env.transport_graph.get_time_distance(depot2_id, depot1_id)
                if energy_back * energy_safety_factor <= max_battery_capacity and not np.isinf(energy_back):
                    depot_to_depot_edges.append([depot2_idx, depot1_idx])
                    depot_to_depot_attrs.append([energy_back / 1000.0, time_back / self.max_time])

        _append_edges(("depot", "to", "depot"), depot_to_depot_edges, depot_to_depot_attrs)

        return data, depot_node_to_idx

    def _update_depot_action_metadata(self, env, data, depot_node_to_idx):
        if not depot_node_to_idx:
            return data

        action_to_node_map = getattr(data, "action_to_node_map", [])
        if not action_to_node_map:
            return data

        action_node_type = getattr(data, "action_node_type", None)
        action_local_index = getattr(data, "action_local_index", None)
        action_is_charging = getattr(data, "action_is_charging", None)
        action_charge_durations = getattr(data, "action_charge_durations", None)
        feasible_action_mask = getattr(data, "feasible_action_mask", None)

        if action_node_type is None or action_local_index is None or action_is_charging is None:
            return data

        action_node_type = action_node_type.clone()
        action_local_index = action_local_index.clone()
        action_is_charging_list = action_is_charging.tolist()

        for idx, (node_id, is_charging) in enumerate(action_to_node_map):
            if is_charging:
                continue
            if node_id in depot_node_to_idx:
                action_node_type[idx] = self.node_type_to_code.get("depot", -1)
                action_local_index[idx] = depot_node_to_idx[node_id]

        data.action_node_type = action_node_type
        data.action_local_index = action_local_index

        active_truck = None
        if getattr(env, "active_truck_id", None) is not None and env.active_truck_id < len(env.trucks):
            active_truck = env.trucks[env.active_truck_id]
        allow_depot_action = bool(getattr(active_truck, "return_to_depot_pending", False))

        if feasible_action_mask is not None and not allow_depot_action:
            feasible_action_mask = feasible_action_mask.clone()
            for idx, (node_id, is_charging) in enumerate(action_to_node_map):
                if is_charging:
                    continue
                if node_id in depot_node_to_idx:
                    feasible_action_mask[idx] = False
            data.feasible_action_mask = feasible_action_mask

            feasible_indices = [i for i, is_ok in enumerate(feasible_action_mask.tolist()) if is_ok]
            feasible_action_to_node_map = [action_to_node_map[i] for i in feasible_indices]
            feasible_action_is_charging = [action_is_charging_list[i] for i in feasible_indices]
            feasible_action_durations = (
                [action_charge_durations[i].item() for i in feasible_indices]
                if action_charge_durations is not None
                else []
            )
            if feasible_action_to_node_map:
                active_truck_idx = None
                node_id_to_type = getattr(data, "node_id_to_type", {})
                if getattr(env, "active_truck_id", None) in node_id_to_type:
                    node_type, local_idx = node_id_to_type[env.active_truck_id]
                    if node_type == "truck":
                        active_truck_idx = local_idx
                data.action_graph_features = self._build_action_graph_features(
                    env,
                    feasible_action_to_node_map,
                    feasible_action_is_charging,
                    feasible_action_durations,
                    active_truck_idx,
                )

        return data

    def _get_depot_ids(self, env):
        depot_ids = set()
        for truck in env.trucks:
            if not truck.delivery_sequence:
                continue
            depot_ids.add(truck.delivery_sequence[0])
        return depot_ids

    def _get_depot_node_features(self, node_id: int, env):
        num_nodes = env.transport_graph.num_nodes
        node_id_norm = node_id / num_nodes if num_nodes > 0 else 0.0
        seq_index_norm = 0.0
        return np.array(
            [self.NODE_TYPE_DEPOT / len(self.node_type_order), node_id_norm, seq_index_norm],
            dtype=np.float32,
        )


def get_action_graph(env, state_space: Optional[GNNStateSpaceVRP] = None):
    """Module-level helper for convenience when only an env is available."""
    space = state_space
    if space is None:
        space = GNNStateSpaceVRP(
            num_trucks=env.num_trucks,
            num_stops=env.num_stops,
            max_time=env.max_time,
            num_charging_nodes=len(env.charging_nodes),
            device=getattr(env, "device", "cpu"),
        )
    return extract_action_graph(space, env)
