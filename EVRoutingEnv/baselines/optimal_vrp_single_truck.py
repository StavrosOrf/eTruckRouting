"""
Single-truck VRP solver (flexible delivery order) using Gurobi MILP.

Objective: minimize total completion time (travel + charging + waiting).
Assumptions:
- Single truck, flexible delivery order, required return to depot.
- At most one charger visit between consecutive deliveries.
- Deterministic travel times (shortest-path) and linear charging model.
- Energy feasibility uses a fixed 1.1 safety factor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import gurobipy as gp
from gurobipy import GRB


@dataclass
class PlanStep:
    """A single planned action for a truck."""

    kind: str  # "nav_delivery", "nav_charger", or "charge"
    target: Optional[int] = None  # delivery/depot or charger node id
    duration: Optional[float] = None  # hours for charge


class OptimalVRPSingleTruckPolicy:
    """
    MILP-based single-truck VRP solver with charging decisions.

    The model chooses a delivery order and, for each leg, either:
    - direct travel, or
    - detour to a single charger, then charge and continue.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._plans: Dict[int, List[PlanStep]] = {}
        self._cursors: Dict[int, int] = {}
        self.energy_safety_factor = 1.1

    def get_action(self, env) -> int:
        if env.active_truck_id is None:
            return env.action_space.sample()

        truck_id = env.active_truck_id
        truck = env.trucks[truck_id]

        if truck.is_complete or truck.failed:
            return env.action_space.sample()

        needs_plan = (
            truck_id not in self._plans
            or self._cursors.get(truck_id, 0) >= len(self._plans[truck_id])
        )
        if needs_plan:
            try:
                plan = self._solve_truck(truck=truck, env=env)
            except Exception as exc:
                print(f"[VRP Optimal] Error solving for truck {truck_id}: {exc}")
                if self.verbose:
                    print(f"[VRP Optimal] Solver failed for truck {truck_id}: {exc}")
                plan = self._create_emergency_plan(truck, env)
            self._plans[truck_id] = plan
            self._cursors[truck_id] = 0

        if not self._plans[truck_id]:
            return env.action_space.sample()

        step_idx = self._cursors[truck_id]
        step = self._plans[truck_id][step_idx]
        self._cursors[truck_id] = step_idx + 1
        return self._to_env_action(step, env, truck)

    def _solve_truck(self, truck, env) -> List[PlanStep]:
        start_node = int(truck.current_node)
        depot_node = int(truck.delivery_sequence[0])
        remaining = [int(n) for n in truck.get_remaining_deliveries() if int(n) != depot_node]

        if not remaining:
            if start_node == depot_node:
                return []
            return self._plan_single_leg(start_node, depot_node, truck, env)

        # Build visit list: start + deliveries + depot return
        visit_nodes = [start_node] + remaining + [depot_node]
        num_deliveries = len(remaining)
        start_idx = 0
        end_idx = num_deliveries + 1

        battery_cap = float(truck.battery_capacity)
        init_battery = float(truck.current_battery)
        min_battery_buffer = 0.05 * battery_cap

        max_charge_hours = 24.0
        graph = env.transport_graph

        wait_time_by_charger = self._compute_wait_times(env)

        # Precompute arc options
        arc_data: Dict[Tuple[int, int], Dict] = {}
        arcs = []
        for i_idx, i_node in enumerate(visit_nodes):
            if i_idx == end_idx:
                continue
            for j_idx, j_node in enumerate(visit_nodes):
                if j_idx == start_idx or i_idx == j_idx:
                    continue
                if i_idx == end_idx:
                    continue
                if j_idx == start_idx:
                    continue

                arcs.append((i_idx, j_idx))

                try:
                    direct_time = graph.get_time_distance(i_node, j_node)
                except Exception:
                    direct_time = float("inf")
                direct_energy = graph.get_path_energy(i_node, j_node)
                if math.isfinite(direct_energy):
                    direct_energy *= self.energy_safety_factor

                charger_options = []
                for charger in env.charging_nodes:
                    try:
                        to_time = graph.get_time_distance(i_node, charger)
                        from_time = graph.get_time_distance(charger, j_node)
                    except Exception:
                        continue
                    to_energy = graph.get_path_energy(i_node, charger)
                    from_energy = graph.get_path_energy(charger, j_node)
                    if not math.isfinite(to_energy) or not math.isfinite(from_energy):
                        continue
                    to_energy *= self.energy_safety_factor
                    from_energy *= self.energy_safety_factor
                    rate, eff = self._charger_profile(env, charger)
                    wait_time = float(wait_time_by_charger.get(charger, 0.0))
                    charger_options.append(
                        {
                            "node": int(charger),
                            "to_time": float(to_time),
                            "from_time": float(from_time),
                            "to_energy": float(to_energy),
                            "from_energy": float(from_energy),
                            "rate": float(rate),
                            "efficiency": float(eff),
                            "wait_time": wait_time,
                        }
                    )

                arc_data[(i_idx, j_idx)] = {
                    "direct_time": float(direct_time),
                    "direct_energy": float(direct_energy),
                    "chargers": charger_options,
                }

        # Quick feasibility: at least one outgoing arc from start
        if not any(i == start_idx for i, _ in arcs):
            raise RuntimeError("No feasible outgoing arcs from start")

        max_energy_leg = max(
            [
                arc_data[a]["direct_energy"]
                for a in arcs
                if math.isfinite(arc_data[a]["direct_energy"])
            ]
            + [
                opt["to_energy"] + opt["from_energy"]
                for a in arcs
                for opt in arc_data[a]["chargers"]
            ],
            default=battery_cap,
        )
        max_rate = max(
            env.charging_config["level2"]["charge_rate"]
            * env.charging_config["level2"]["efficiency"],
            env.charging_config["dcfast"]["charge_rate"]
            * env.charging_config["dcfast"]["efficiency"],
        )
        max_charge_possible = max_charge_hours * max_rate
        big_m = battery_cap + max_energy_leg + max_charge_possible + 1.0

        model = gp.Model("vrp_single_truck")
        model.Params.OutputFlag = 0
        model.Params.TimeLimit = 180.0
        model.Params.MIPGap = 0.01
        model.Params.MIPFocus = 1

        x = model.addVars(arcs, vtype=GRB.BINARY, name="x")
        direct = model.addVars(arcs, vtype=GRB.BINARY, name="direct")
        battery = model.addVars(len(visit_nodes), lb=min_battery_buffer, ub=battery_cap, name="battery")

        use_charger: Dict[Tuple[int, int, int], gp.Var] = {}
        charge_time: Dict[Tuple[int, int, int], gp.Var] = {}

        # Flow constraints
        model.addConstr(gp.quicksum(x[start_idx, j] for _, j in arcs if _ == start_idx) == 1)
        model.addConstr(gp.quicksum(x[i, end_idx] for i, _ in arcs if _ == end_idx) == 1)

        for k in range(1, num_deliveries + 1):
            model.addConstr(gp.quicksum(x[i, k] for i, j in arcs if j == k) == 1)
            model.addConstr(gp.quicksum(x[k, j] for i, j in arcs if i == k) == 1)

        # Start and end degree restrictions
        model.addConstr(gp.quicksum(x[i, start_idx] for i, j in arcs if j == start_idx) == 0)
        model.addConstr(gp.quicksum(x[end_idx, j] for i, j in arcs if i == end_idx) == 0)

        model.addConstr(battery[start_idx] == init_battery)

        objective_terms = []

        for i, j in arcs:
            seg = arc_data[(i, j)]
            chargers = seg["chargers"]

            for c_idx, opt in enumerate(chargers):
                use_charger[(i, j, c_idx)] = model.addVar(vtype=GRB.BINARY, name=f"use_{i}_{j}_{opt['node']}")
                charge_time[(i, j, c_idx)] = model.addVar(lb=0.0, ub=max_charge_hours, name=f"charge_{i}_{j}_{opt['node']}")
                model.addConstr(charge_time[(i, j, c_idx)] <= max_charge_hours * use_charger[(i, j, c_idx)])

            choice_vars = [direct[i, j]] + [use_charger[(i, j, c_idx)] for c_idx in range(len(chargers))]
            model.addConstr(gp.quicksum(choice_vars) == x[i, j])

            # Direct arc constraints
            if math.isfinite(seg["direct_energy"]):
                energy = seg["direct_energy"]
                model.addConstr(battery[i] - energy >= -big_m * (1 - direct[i, j]))
                model.addConstr(battery[j] >= battery[i] - energy - big_m * (1 - direct[i, j]))
                model.addConstr(battery[j] <= battery[i] - energy + big_m * (1 - direct[i, j]))
                if math.isfinite(seg["direct_time"]):
                    objective_terms.append(direct[i, j] * seg["direct_time"])
            else:
                model.addConstr(direct[i, j] == 0)

            # Charger detour constraints
            for c_idx, opt in enumerate(chargers):
                use_var = use_charger[(i, j, c_idx)]
                ct_var = charge_time[(i, j, c_idx)]

                added_energy = opt["rate"] * opt["efficiency"] * ct_var

                model.addConstr(battery[i] - opt["to_energy"] >= -big_m * (1 - use_var))
                model.addConstr(
                    battery[i] - opt["to_energy"] + added_energy
                    <= battery_cap + big_m * (1 - use_var)
                )
                model.addConstr(
                    battery[j]
                    >= battery[i] - opt["to_energy"] - opt["from_energy"] + added_energy - big_m * (1 - use_var)
                )
                model.addConstr(
                    battery[j]
                    <= battery[i] - opt["to_energy"] - opt["from_energy"] + added_energy + big_m * (1 - use_var)
                )

                objective_terms.append(
                    use_var * (opt["to_time"] + opt["from_time"] + opt["wait_time"]) + ct_var
                )

        # MTZ subtour elimination for deliveries only
        order = model.addVars(num_deliveries + 1, lb=1.0, ub=float(num_deliveries), name="order")
        for i in range(1, num_deliveries + 1):
            for j in range(1, num_deliveries + 1):
                if i == j:
                    continue
                model.addConstr(order[i] - order[j] + num_deliveries * x[i, j] <= num_deliveries - 1)

        model.setObjective(gp.quicksum(objective_terms), GRB.MINIMIZE)
        model.optimize()

        if model.Status == GRB.INFEASIBLE:
            raise RuntimeError("VRP model infeasible")
        if model.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
            if model.SolCount == 0:
                raise RuntimeError(f"VRP solver failed with status {model.Status}")

        # Extract route
        plan: List[PlanStep] = []
        current = start_idx
        visited = set([start_idx])
        while current != end_idx:
            next_idx = None
            for _, j in arcs:
                if _ == current and x[current, j].X > 0.5:
                    next_idx = j
                    break
            if next_idx is None:
                raise RuntimeError("Failed to extract route")

            seg = arc_data[(current, next_idx)]
            if direct[current, next_idx].X > 0.5:
                plan.append(PlanStep(kind="nav_delivery", target=int(visit_nodes[next_idx])))
            else:
                chargers = seg["chargers"]
                chosen_idx = next(
                    c_idx for c_idx in range(len(chargers)) if use_charger[(current, next_idx, c_idx)].X > 0.5
                )
                chosen = chargers[chosen_idx]
                plan.append(PlanStep(kind="nav_charger", target=int(chosen["node"])))
                selected_duration = float(charge_time[(current, next_idx, chosen_idx)].X)
                if selected_duration > 1e-6:
                    plan.append(PlanStep(kind="charge", duration=selected_duration))
                plan.append(PlanStep(kind="nav_delivery", target=int(visit_nodes[next_idx])))

            if next_idx in visited:
                break
            visited.add(next_idx)
            current = next_idx

        return plan

    def _plan_single_leg(self, start: int, target: int, truck, env) -> List[PlanStep]:
        graph = env.transport_graph
        battery = float(truck.current_battery)
        battery_cap = float(truck.battery_capacity)
        best = None

        direct_energy = graph.get_path_energy(start, target)
        if math.isfinite(direct_energy):
            direct_energy *= self.energy_safety_factor
            direct_time = graph.get_time_distance(start, target)
            if direct_energy <= battery:
                best = (direct_time, [PlanStep(kind="nav_delivery", target=target)])

        wait_time_by_charger = self._compute_wait_times(env)

        for charger in env.charging_nodes:
            to_energy = graph.get_path_energy(start, charger)
            from_energy = graph.get_path_energy(charger, target)
            if not math.isfinite(to_energy) or not math.isfinite(from_energy):
                continue
            to_energy *= self.energy_safety_factor
            from_energy *= self.energy_safety_factor
            if to_energy > battery:
                continue

            to_time = graph.get_time_distance(start, charger)
            from_time = graph.get_time_distance(charger, target)
            rate, eff = self._charger_profile(env, charger)

            available = max(0.0, battery - to_energy)
            needed = max(0.0, from_energy - available)
            if available + needed > battery_cap + 1e-6:
                continue

            charge_time = needed / max(rate * eff, 1e-6)
            wait_time = float(wait_time_by_charger.get(charger, 0.0))
            total_time = to_time + from_time + charge_time + wait_time

            steps = [
                PlanStep(kind="nav_charger", target=int(charger)),
                PlanStep(kind="charge", duration=charge_time),
                PlanStep(kind="nav_delivery", target=target),
            ]
            if best is None or total_time < best[0]:
                best = (total_time, steps)

        if best is None:
            raise RuntimeError("No feasible single-leg plan")
        return best[1]

    def _compute_wait_times(self, env) -> Dict[int, float]:
        wait_times = {}
        if not hasattr(env, "charging_station"):
            return wait_times

        for charger in env.charging_nodes:
            info = env.charging_station.get_charger_info(charger, env.global_clock)
            capacity = max(1, int(info.get("capacity", 1)))
            occupancy = int(info.get("current_occupancy", 0))
            utilization = min(0.95, max(0.05, occupancy / float(capacity)))
            wait_times[charger] = env.charging_station.get_waiting_time(charger, utilization)

        return wait_times

    def _charger_profile(self, env, node: int) -> Tuple[float, float]:
        if hasattr(env, "charging_station"):
            ctype = env.charging_station.charger_type.get(node, "Level2")
        else:
            ctype = env.transport_graph.get_charger_type(node) or "Level2"
        key = "dcfast" if str(ctype).lower() == "dcfast" else "level2"
        cfg = env.charging_config[key]
        return float(cfg["charge_rate"]), float(cfg["efficiency"])

    def _create_emergency_plan(self, truck, env) -> List[PlanStep]:
        plan = []
        current_node = int(truck.current_node)
        current_battery = float(truck.current_battery)

        min_energy = float("inf")
        best_charger = None
        for charger_node in env.charging_nodes:
            try:
                energy_needed = env.transport_graph.get_path_energy(current_node, int(charger_node))
                energy_needed *= self.energy_safety_factor
                if energy_needed < current_battery and energy_needed < min_energy:
                    min_energy = energy_needed
                    best_charger = int(charger_node)
            except Exception:
                continue

        if best_charger is not None:
            plan.append(PlanStep(kind="nav_charger", target=best_charger))
            rate, eff = self._charger_profile(env, best_charger)
            charge_needed = max(0.0, truck.battery_capacity - (current_battery - min_energy))
            charge_hours = charge_needed / max(rate * eff, 1e-6)
            plan.append(PlanStep(kind="charge", duration=charge_hours))

        remaining = [int(n) for n in truck.get_remaining_deliveries() if int(n) != truck.delivery_sequence[0]]
        if remaining:
            plan.append(PlanStep(kind="nav_delivery", target=remaining[0]))

        return plan

    def _to_env_action(self, step: PlanStep, env, truck) -> int:
        if step.kind == "nav_charger":
            try:
                idx = env.charging_nodes.index(step.target)
                return idx
            except ValueError:
                return env.action_space.sample()

        if step.kind == "nav_delivery":
            if env.enable_flexible_delivery_order:
                depot_node = int(truck.delivery_sequence[0])
                if step.target == depot_node:
                    delivery_idx = len(truck.delivery_sequence) - 1
                    return env.num_charging_nodes + delivery_idx

                try:
                    pos = truck.delivery_sequence.index(step.target)
                    delivery_idx = pos - 1
                    if delivery_idx >= 0:
                        return env.num_charging_nodes + delivery_idx
                except ValueError:
                    return env.action_space.sample()

            return env.num_charging_nodes

        if step.kind == "charge":
            durations = sorted([float(d) for d in env.charging_config["charge_durations"]])
            requested = max(0.0, float(step.duration or 0.0))
            requested_with_buffer = requested * 1.10
            candidate = None
            for d in durations:
                if d >= requested_with_buffer - 1e-6:
                    candidate = d
                    break
            if candidate is None:
                candidate = durations[-1]
            dur_idx = env.charging_config["charge_durations"].index(candidate)
            return env.num_navigation_actions + dur_idx

        return env.action_space.sample()
