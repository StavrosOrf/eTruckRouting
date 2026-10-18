"""Label class and extend_label implementation (forward labeling) -- TD-EVRPTW.

This version implements a REF closer to the paper:
- Rule 1 (MBR) at departure and after processing the arrival node
- time-dependent travel & energy using utils.get_travel_time_and_consumption
- waiting function at STATIONS applied on arrival (ω_j(t+τ_ij(t)))
- customer service or STATION recharge (full-recharge, PWL inverse)
- checks of time windows and global horizon T
- elementarity handled outside (pricing), but we still update visited set
- reduced-cost update: c_ij - π_j (π only for customers)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, FrozenSet
import numpy as np

from utils import PiecewiseLinearFunction, get_travel_time_and_consumption

# =============================== Class Label ===================================

@dataclass(slots=True, order=False)
class Label:
    # Reduced cost accumulated so far
    cost: float
    # Current (departure) time at the node
    time: float
    # Battery available at the node (state of charge)
    battery: float
    # Total demand already served (delivery model)
    served_demand: float

    # Topology / identification
    current_node_id: int
    num_visited_customers: int
    visited_customers: FrozenSet[int]
    last_customer_id: int

    # Back pointer for path reconstruction
    parent: Optional["Label"] = None

    # Deterministic tie-breaking for heap ordering (best-first)
    def __lt__(self, other: "Label") -> bool:
        if not isinstance(other, Label):
            return NotImplemented
        # best-first: lower reduced cost, then earlier time, then higher battery,
        # then less served demand; tie-break by node ids and size of visited set
        return (
            (self.cost, self.time, -self.battery, self.served_demand,
             self.current_node_id, self.last_customer_id, len(self.visited_customers))
            <
            (other.cost, other.time, -other.battery, other.served_demand,
             other.current_node_id, other.last_customer_id, len(other.visited_customers))
        )

# =============================== Extend Label ===================================

def extend_label(label: Label, next_node_id: int, instance: dict, duals: dict) -> Optional[Label]:
    """
    Forward REF with:
      - Rule 1 (MBR) at departure
      - time-dependent travel & energy
      - TW check upon arrival
      - station waiting ω_j upon arrival, before any operation at j
      - customer service OR station recharge to full (PWL inverse)
      - TW and horizon checks after service/recharge
      - Rule 1 (MBR) at the arrival node (unless depot)
      - reduced-cost update: c_ij - π_j (π_j on customers only)
    """
    params = instance['parameters']
    depot_id = instance['depot_id']
    from_node_id = label.current_node_id
    to_node = instance['nodes_by_id'][next_node_id]

    # Heuristic relaxations flags (used by heuristic pricing variants)
    relax_b = instance.get('_relax_b', False)

    # ---- Rule 1 on departure (unless relaxed): need enough battery to be able to leave node
    if not relax_b and 'mbr' in instance:
        if label.battery + 1e-9 < float(instance['mbr'][from_node_id]):
            return None

    # Definitions
    T = float(params["planning_horizon_T"])
    speed_defs = instance["time_dependent_definitions"]["speed_profiles"]
    tb = np.array(speed_defs["time_intervals_factors"], dtype=float) * T
    profile_name = instance["time_dependent_assignments"]["arc_speed_profiles"][from_node_id][next_node_id]
    multipliers = np.array(speed_defs["profiles"].get(profile_name, [1.0]), dtype=float)

    # 1) travel & consumption (time dependent)
    total_distance = float(instance["distance_matrix"][from_node_id, next_node_id])
    departure_time = label.time
    travel_time, consumption = get_travel_time_and_consumption(departure_time, total_distance, tb, multipliers, params)
    if travel_time == float('inf'):
        return None

    arrival_time = departure_time + travel_time
    arrival_battery = label.battery - consumption
    if arrival_battery < -1e-6:
        return None

    # 2) TW/horizon on arrival instant
    if arrival_time > to_node['time_window']['end'] + 1e-6 or arrival_time > T + 1e-6:
        return None

    # 3) station waiting on arrival
    effective_arrival_time = arrival_time
    if to_node['type'] == 1:
        wf = instance["time_dependent_definitions"].get("waiting_function", {})
        if wf and wf.get("points"):
            wait_pwl = PiecewiseLinearFunction(wf["points"])
            extra_wait = float(wait_pwl.evaluate(arrival_time))
            if extra_wait > 0:
                effective_arrival_time += extra_wait

    # 4) passive wait to meet earliest TW.start
    effective_arrival_time = max(effective_arrival_time, to_node['time_window']['start'])

    # 5) capacity (delivery model): total served cannot exceed vehicle capacity
    new_served_demand = label.served_demand + (to_node.get('demand', 0.0) if to_node['type'] == 0 else 0.0)
    if new_served_demand > params["vehicle_capacity"] + 1e-9:
        return None

    # 6) service or recharge and resulting (time, battery, visited set)
    next_time = effective_arrival_time
    next_battery = arrival_battery
    num_visited = label.num_visited_customers
    if to_node['type'] == 1:  # station
        # Full recharge to capacity using station mode PWL inverse
        recharge_modes = instance['time_dependent_definitions'].get('recharge_functions', {})
        station_modes = instance['time_dependent_assignments'].get('station_recharge_modes', {})
        mode = station_modes.get(str(next_node_id))
        recharge_pwl_data = recharge_modes.get(mode, [[0, 0], [params['battery_capacity'], params['battery_capacity']]])
        recharge_pwl = PiecewiseLinearFunction(recharge_pwl_data)
        try:
            t_start = recharge_pwl.evaluate_inverse(max(0.0, arrival_battery))
            t_end = recharge_pwl.domain()[1]
            time_to_full = max(0.0, (t_end - t_start))
        except Exception:
            time_to_full = 0.0
        next_time = effective_arrival_time + time_to_full
        next_battery = params['battery_capacity']
        new_visited_set = label.visited_customers
        new_last_customer = label.last_customer_id

    elif to_node['type'] == 0:  # customer
        service_time = to_node.get('service_time', 0.0)
        next_time = effective_arrival_time + service_time
        next_battery = arrival_battery
        # elementarity is enforced in the caller (pricing); we still update state
        new_visited_set = label.visited_customers.union({next_node_id})
        new_last_customer = next_node_id
        num_visited += 1
    else:
        # depot or other type: simply leave immediately
        service_time = to_node.get('service_time', 0.0)
        next_time = effective_arrival_time + service_time
        next_battery = arrival_battery
        new_visited_set = label.visited_customers
        new_last_customer = label.last_customer_id

    # 7) TW/horizon after service/recharge
    if next_time > to_node['time_window']['end'] + 1e-6 or next_time > T + 1e-6:
        return None

    # 8) Rule 1 at the arrival node (except depot) unless relaxed
    if not relax_b and 'mbr' in instance and next_node_id != depot_id:
        if next_battery + 1e-9 < float(instance['mbr'][next_node_id]):
            return None

    # 9) reduced-cost update (only customers carry a dual)
    arc_cost = float(instance["distance_matrix"][from_node_id][next_node_id])
    dual_value = duals.get(next_node_id, 0.0) if to_node['type'] == 0 else 0.0
    new_cost = label.cost + arc_cost - dual_value

    return Label(
        new_cost, next_time, next_battery, new_served_demand,
        next_node_id, num_visited, frozenset(new_visited_set), new_last_customer, label
    )


