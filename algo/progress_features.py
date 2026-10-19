"""Observable charging connectivity to eligible remaining work.

The fixed California station graph is cached in the policy configuration.
Its edge costs and the dynamic customer costs are already present in the
canonical observation. No witness, realized future draw or demonstrator is used.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from EVRoutingEnv.state.feasibility import ActionKind
from EVRoutingEnv.state.features import (
    ACTION_FEATURES,
    CHARGER_FEATURES,
    CUSTOMER_FEATURES,
    GLOBAL_FEATURES,
    TRUCK_FEATURES,
)


PROGRESS_FEATURE_DIM = 6


def station_progress_config(env):
    ids = sorted(env.charging_nodes)
    capacity = float(env.trucks[0].battery_capacity)
    factor = 1.0
    if (
        env.traffic_config["enable_traffic"]
        and env.traffic_config["enable_energy_uncertainty"]
    ):
        factor = max(1.0, float(env.traffic_config["max_energy_multiplier"]))
    hops = np.full((len(ids), len(ids)), len(ids) + 1, dtype=float)
    for i, source in enumerate(ids):
        for j, target in enumerate(ids):
            if i == j:
                hops[i, j] = 0
            elif (
                env.transport_graph.get_path_energy(source, target) * factor
                <= capacity + 1e-8
            ):
                hops[i, j] = 1
    for middle in range(len(ids)):
        hops = np.minimum(hops, hops[:, middle, None] + hops[None, middle, :])
    return {
        "progress_station_ids": tuple(ids),
        "progress_station_hops": tuple(tuple(row) for row in hops.tolist()),
        "progress_energy_multiplier": factor,
    }


class ProgressActionFeatures(nn.Module):
    """Six bounded features per candidate, derived without trainable weights.

    Columns: fraction of eligible tasks reachable after charging, station hops
    to work/home, direct progress flag, required energy ratio, nominal work/home
    travel hours, and improvement in station-hop distance. Hops assume the
    fixed map and reference battery capacity; outages make them optimistic.
    """

    def __init__(self, station_ids, station_hops, factor):
        super().__init__()
        self.register_buffer("station_ids", torch.tensor(station_ids, dtype=torch.long))
        self.register_buffer(
            "station_hops", torch.tensor(station_hops, dtype=torch.float32)
        )
        self.factor = factor

    @torch.no_grad()
    def forward(self, tensors):
        trucks, customers, chargers = (
            tensors.nodes[key] for key in ("truck", "customer", "charger")
        )
        active = trucks[..., TRUCK_FEATURES.index("is_active")]
        active_truck = (trucks * active.unsqueeze(-1)).sum(1)
        capacity = active_truck[
            :, TRUCK_FEATURES.index("battery_capacity_kwh")
        ].clamp_min(1)
        battery = active_truck[:, TRUCK_FEATURES.index("battery_kwh")]
        payload = active_truck[:, TRUCK_FEATURES.index("remaining_payload")]
        active_id = tensors.global_features[:, GLOBAL_FEATURES.index("active_truck_id")]
        owner = customers[..., CUSTOMER_FEATURES.index("preassigned_to")]
        eligible = customers[..., CUSTOMER_FEATURES.index("status_code")] == 0
        eligible &= (owner < 0) | (owner == active_id.unsqueeze(-1))
        eligible &= (
            customers[..., CUSTOMER_FEATURES.index("demand")]
            <= payload.unsqueeze(-1) + 1e-6
        )
        eligible &= tensors.node_masks["customer"]
        has_work = eligible.any(-1)
        available = (
            chargers[..., CHARGER_FEATURES.index("available")] > 0.5
        ) & tensors.node_masks["charger"]
        customer_station = tensors.pairwise[("customer", "charger")]
        escape = (
            customer_station[..., 0]
            .masked_fill(
                ~(available.unsqueeze(1) & (customer_station[..., 2] > 0.5)), torch.inf
            )
            .amin(-1)
        )
        depot_energy = customers[..., CUSTOMER_FEATURES.index("depot_energy_kwh")]
        depot_reachable = (
            customers[..., CUSTOMER_FEATURES.index("depot_reachable")] > 0.5
        )
        escape = torch.minimum(
            escape, depot_energy.masked_fill(~depot_reachable, torch.inf)
        )
        station_customer = tensors.pairwise[("charger", "customer")]
        work_energy = self.factor * (station_customer[..., 0] + escape.unsqueeze(1))
        usable = eligible.unsqueeze(1) & (station_customer[..., 2] > 0.5)
        work_energy = work_energy.masked_fill(~usable, torch.inf)
        home_energy = (
            self.factor * chargers[..., CHARGER_FEATURES.index("depot_energy_kwh")]
        )
        home_energy = home_energy.masked_fill(
            chargers[..., CHARGER_FEATURES.index("depot_reachable")] < 0.5, torch.inf
        )
        direct_work = work_energy <= capacity[:, None, None] + 1e-6
        direct_home = home_energy <= capacity.unsqueeze(-1) + 1e-6
        goal = (
            torch.where(has_work.unsqueeze(-1), direct_work.any(-1), direct_home)
            & available
        )
        observed_ids = chargers[..., CHARGER_FEATURES.index("node_id")].long()
        matches = observed_ids.unsqueeze(-1) == self.station_ids
        if bool((tensors.node_masks["charger"] & ~matches.any(-1)).any().item()):
            raise ValueError(
                "progress feature map does not match observed charger identities"
            )
        order = matches.long().argmax(-1)
        hops = self.station_hops[order.unsqueeze(-1), order.unsqueeze(1)]
        unreachable = float(self.station_ids.numel() + 1)
        distance = (
            (hops + (~goal).unsqueeze(1) * unreachable).amin(-1).clamp_max(unreachable)
        )
        fraction = direct_work.sum(-1) / eligible.sum(-1).clamp_min(1).unsqueeze(-1)
        fraction = torch.where(has_work.unsqueeze(-1), fraction, direct_home.float())
        minimum_energy = torch.where(
            has_work.unsqueeze(-1), work_energy.amin(-1), home_energy
        )
        hours = station_customer[..., 1].masked_fill(~usable, torch.inf).amin(-1)
        hours = torch.where(
            has_work.unsqueeze(-1),
            hours,
            chargers[..., CHARGER_FEATURES.index("depot_travel_hours")],
        )
        horizon = tensors.global_features[
            :, GLOBAL_FEATURES.index("max_time")
        ].clamp_min(1)
        table = torch.stack(
            (
                fraction,
                (distance / 4).clamp_max(2),
                goal.float(),
                (minimum_energy / capacity.unsqueeze(-1)).clamp_max(2),
                (hours / horizon.unsqueeze(-1)).clamp_max(2),
            ),
            dim=-1,
        )
        action_targets = tensors.action[..., ACTION_FEATURES.index("target_node")]
        target_matches = (
            action_targets.unsqueeze(-1) == observed_ids.unsqueeze(1)
        ) & tensors.node_masks["charger"].unsqueeze(1)
        features = torch.bmm(target_matches.float(), table)
        target_distance = torch.bmm(
            target_matches.float(), distance.unsqueeze(-1)
        ).squeeze(-1)
        current_node = active_truck[:, TRUCK_FEATURES.index("current_node")]
        at_station = (current_node.unsqueeze(-1) == observed_ids) & tensors.node_masks[
            "charger"
        ]
        truck_station = (
            tensors.pairwise[("truck", "charger")] * active[:, :, None, None]
        ).sum(1)
        can_reach_station = (
            (self.factor * truck_station[..., 0] <= battery.unsqueeze(-1) + 1e-6)
            & (truck_station[..., 2] > 0.5)
            & available
        )
        current_distance = (
            (distance + 1).masked_fill(~can_reach_station, unreachable).amin(-1)
        )
        current_distance = torch.where(
            at_station.any(-1), (distance * at_station).sum(-1), current_distance
        )
        truck_customer = (
            tensors.pairwise[("truck", "customer")] * active[:, :, None, None]
        ).sum(1)
        direct_now = (
            (
                self.factor * (truck_customer[..., 0] + escape)
                <= battery.unsqueeze(-1) + 1e-6
            )
            & (truck_customer[..., 2] > 0.5)
            & eligible
        ).any(-1)
        current_distance = torch.where(has_work & direct_now, 0.0, current_distance)
        improvement = ((current_distance.unsqueeze(-1) - target_distance) / 4).clamp(
            -2, 2
        )
        features = torch.cat((features, improvement.unsqueeze(-1)), dim=-1)
        kinds = tensors.action[..., ACTION_FEATURES.index("kind_code")].long()
        customer_kind = kinds == list(ActionKind).index(ActionKind.CUSTOMER)
        depot_kind = kinds == list(ActionKind).index(ActionKind.DEPOT)
        navigation_kind = customer_kind | depot_kind
        features[..., 0] = torch.where(customer_kind, 1.0, features[..., 0])
        features[..., 1] = torch.where(navigation_kind, 0.0, features[..., 1])
        features[..., 2] = torch.where(navigation_kind, 1.0, features[..., 2])
        features[..., 3] = torch.where(
            navigation_kind,
            (
                tensors.action[..., ACTION_FEATURES.index("required_energy")]
                / capacity.unsqueeze(-1)
            ).clamp(0, 2),
            features[..., 3],
        )
        features[..., 4] = torch.where(
            navigation_kind,
            (
                tensors.action[..., ACTION_FEATURES.index("leg_travel_hours")]
                / horizon.unsqueeze(-1)
            ).clamp(0, 2),
            features[..., 4],
        )
        features[..., 5] = torch.where(navigation_kind, 1.0, features[..., 5])
        features = torch.nan_to_num(features, nan=0, posinf=2, neginf=-2)
        return features * tensors.action_padding_mask.unsqueeze(-1)
