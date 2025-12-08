"""
Simplified optimal charging planner built with Gurobi with stochastic considerations.

Key features:
- Applies energy safety margin based on config settings to account for stochastic variations
- Uses nominal (base) travel times without traffic variations
- Ignores charger queues and contention (assumes always available)
- Conservative charging: always rounds up to ensure sufficient charge
- Focuses on 100% feasibility over optimality in stochastic environments

This model should maintain high success rate even with stochastic energy consumption.
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
    target: Optional[int] = None  # charger node id for nav_charger
    duration: Optional[float] = None  # hours for charge


class OptimalGurobiSimplePolicy:
    """
    Simplified per-truck optimal policy using MILP with stochastic robustness.
    
    Maintains high success rate in stochastic environments by:
    - Applying energy safety margin based on config to account for stochastic consumption
    - Rounding charging durations up
    - Adding minimum battery buffers
    - Using conservative planning to handle uncertainty
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._plans: Dict[int, List[PlanStep]] = {}
        self._cursors: Dict[int, int] = {}
        self._energy_safety_factor = None  # Will be set from env config

    def get_action(self, env) -> int:
        """Return the next action for the currently active truck."""
        if env.active_truck_id is None:
            return env.action_space.sample()

        # Initialize energy safety factor from config on first call
        if self._energy_safety_factor is None:
            self._energy_safety_factor = self._compute_energy_safety_factor(env)
            if self.verbose:
                print(f"[Simple Optimal] Energy safety factor set to {self._energy_safety_factor:.2f}")

        truck_id = env.active_truck_id
        truck = env.trucks[truck_id]

        # If truck already complete/failed, no meaningful action
        if truck.is_complete or truck.failed:
            return env.action_space.sample()

        # Replan if needed (new episode, cursor exhausted, or plan missing)
        needs_plan = (
            truck_id not in self._plans
            or self._cursors.get(truck_id, 0) >= len(self._plans[truck_id])
        )
        if needs_plan:
            try:
                plan = self._solve_truck(truck=truck, env=env)
            except Exception as e:
                if self.verbose:
                    print(f"[Simple Optimal] Solver failed for truck {truck_id}: {e}")
                # Emergency fallback: go to nearest charger and charge to full
                plan = self._create_emergency_plan(truck, env)
            self._plans[truck_id] = plan
            self._cursors[truck_id] = 0

        # If no actions are required, default to next-delivery action
        if not self._plans[truck_id]:
            return env.num_charging_nodes

        step_idx = self._cursors[truck_id]
        step = self._plans[truck_id][step_idx]
        self._cursors[truck_id] = step_idx + 1
        return self._to_env_action(step, env)

    def _solve_truck(self, truck, env) -> List[PlanStep]:
        """Build and solve the MILP for a single truck."""
        deliveries = [int(truck.current_node)] + [
            int(n) for n in truck.get_remaining_deliveries()
        ]
        if len(deliveries) <= 1:
            return []

        segments = self._build_segments(deliveries, env)
        battery_cap = float(truck.battery_capacity)
        init_battery = float(truck.current_battery)
        max_charge_hours = 24.0

        # Quick feasibility check
        for seg in segments:
            direct_ok = math.isfinite(seg["direct_energy"])
            charger_ok = len(seg["chargers"]) > 0
            if not direct_ok and not charger_ok:
                raise ValueError(
                    f"No feasible path from node {seg['start']} to delivery {seg['end']}"
                )

        # Big-M calculation
        max_energy_leg = max(
            [seg["direct_energy"] for seg in segments if math.isfinite(seg["direct_energy"])]
            + [opt["to_energy"] + opt["from_energy"] for seg in segments for opt in seg["chargers"]],
            default=battery_cap,
        )
        max_rate = max(
            env.charging_config["level2"]["charge_rate"] * env.charging_config["level2"]["efficiency"],
            env.charging_config["dcfast"]["charge_rate"] * env.charging_config["dcfast"]["efficiency"],
        )
        max_charge_possible = max_charge_hours * max_rate
        big_m = battery_cap + max_energy_leg + max_charge_possible + 1.0

        model = gp.Model("simple_optimal_truck")
        model.Params.OutputFlag = 0
        model.Params.TimeLimit = 120.0  # 2 minute timeout
        model.Params.MIPGap = 0.10  # Accept 10% gap from optimal
        model.Params.MIPFocus = 1  # Focus on finding feasible solutions

        num_segments = len(segments)
        # Add 10% minimum battery buffer to avoid running too close to empty
        min_battery_buffer = 0.10 * battery_cap
        battery = model.addVars(
            num_segments + 1, lb=min_battery_buffer, ub=battery_cap, name="battery"
        )
        direct = model.addVars(num_segments, vtype=GRB.BINARY, name="direct")
        use_charger: Dict[Tuple[int, int], gp.Var] = {}
        charge_time: Dict[Tuple[int, int], gp.Var] = {}

        # Get max charging duration for bounds
        available_durations = sorted([float(d) for d in env.charging_config["charge_durations"]])
        max_discrete_duration = max(available_durations)
        
        # Optional initial charge if starting node has a charger
        # Use CONTINUOUS charging time (will round to discrete when converting to action)
        start_charge_time = None
        start_rate = start_eff = None
        if deliveries[0] in env.charging_nodes:
            start_charge_time = model.addVar(lb=0.0, ub=max_discrete_duration, name="start_charge_time")
            start_rate, start_eff = self._charger_profile(env, deliveries[0])
            model.addConstr(battery[0] == init_battery + start_rate * start_eff * start_charge_time)
        else:
            model.addConstr(battery[0] == init_battery)

        objective_terms = []
        if start_charge_time is not None:
            objective_terms.append(start_charge_time)
        
        for i, seg in enumerate(segments):
            chargers = seg["chargers"]
            for c_idx, opt in enumerate(chargers):
                use_charger[(i, c_idx)] = model.addVar(vtype=GRB.BINARY, name=f"use_{i}_{opt['node']}")
                # Use CONTINUOUS charging time (will round to discrete when converting to action)
                charge_time[(i, c_idx)] = model.addVar(lb=0.0, ub=max_discrete_duration, name=f"charge_{i}_{opt['node']}")
                # If charger not chosen, force zero charge time
                model.addConstr(charge_time[(i, c_idx)] <= max_discrete_duration * use_charger[(i, c_idx)])

            # Exactly one choice per segment: direct or via one charger
            choice_vars = [direct[i]] + [use_charger[(i, c_idx)] for c_idx in range(len(chargers))]
            model.addConstr(gp.quicksum(choice_vars) == 1)

            # Direct travel constraints
            if math.isfinite(seg["direct_energy"]):
                energy = seg["direct_energy"]
                model.addConstr(battery[i] - energy >= -big_m * (1 - direct[i]))
                model.addConstr(battery[i + 1] >= battery[i] - energy - big_m * (1 - direct[i]))
                model.addConstr(battery[i + 1] <= battery[i] - energy + big_m * (1 - direct[i]))
                if math.isfinite(seg["direct_time"]):
                    objective_terms.append(direct[i] * seg["direct_time"])
            else:
                model.addConstr(direct[i] == 0)

            # Charger detour constraints
            for c_idx, opt in enumerate(chargers):
                use_var = use_charger[(i, c_idx)]
                ct_var = charge_time[(i, c_idx)]
                
                # Use simple linear charging model (no realistic curves)
                added_energy = opt["rate"] * opt["efficiency"] * ct_var
                
                # Battery constraints for charger route
                model.addConstr(battery[i] - opt["to_energy"] >= -big_m * (1 - use_var))
                model.addConstr(
                    battery[i] - opt["to_energy"] + added_energy <= battery_cap + big_m * (1 - use_var)
                )
                model.addConstr(
                    battery[i + 1] >= battery[i] - opt["to_energy"] - opt["from_energy"] + added_energy - big_m * (1 - use_var)
                )
                model.addConstr(
                    battery[i + 1] <= battery[i] - opt["to_energy"] - opt["from_energy"] + added_energy + big_m * (1 - use_var)
                )

                objective_terms.append(use_var * (opt["to_time"] + opt["from_time"]) + ct_var)

        model.setObjective(gp.quicksum(objective_terms), GRB.MINIMIZE)
        model.optimize()

        if model.Status == GRB.INFEASIBLE:
            if self.verbose:
                print(f"[Simple Optimal] Model is INFEASIBLE")
            raise RuntimeError(f"Gurobi model is INFEASIBLE")
        elif model.Status == GRB.TIME_LIMIT:
            if self.verbose:
                print(f"[Simple Optimal] Time limit reached, using best solution found")
            if model.SolCount == 0:
                raise RuntimeError(f"No solution found within time limit")
        elif model.Status != GRB.OPTIMAL:
            if model.SolCount > 0:
                if self.verbose:
                    print(f"[Simple Optimal] Using suboptimal solution (status {model.Status})")
            else:
                raise RuntimeError(f"Gurobi solver failed with status {model.Status}")

        # Extract solution
        plan: List[PlanStep] = []
        if start_charge_time is not None and start_charge_time.X > 1e-6:
            plan.append(PlanStep(kind="charge", duration=float(start_charge_time.X)))

        for i, seg in enumerate(segments):
            if direct[i].X > 0.5:
                plan.append(PlanStep(kind="nav_delivery"))
                continue

            chargers = seg["chargers"]
            chosen_idx = next(
                c_idx for c_idx in range(len(chargers)) if use_charger[(i, c_idx)].X > 0.5
            )
            chosen = chargers[chosen_idx]
            plan.append(PlanStep(kind="nav_charger", target=chosen["node"]))
            selected_duration = float(charge_time[(i, chosen_idx)].X)
            if selected_duration and selected_duration > 1e-6:
                plan.append(PlanStep(kind="charge", duration=selected_duration))
            plan.append(PlanStep(kind="nav_delivery"))

        return plan

    def _compute_energy_safety_factor(self, env) -> float:
        """
        Compute energy safety factor from environment config.
        
        Returns:
            Safety factor (multiplier) based on energy uncertainty settings.
            If energy uncertainty is disabled, returns 1.0 (no safety margin).
            If enabled, returns max_energy_multiplier from config.
        """
        traffic_config = getattr(env, 'traffic_config', {})
        
        # Check if energy uncertainty is enabled
        enable_energy_uncertainty = traffic_config.get('enable_energy_uncertainty', False)
        
        if not enable_energy_uncertainty:
            # No energy uncertainty - use deterministic values
            return 1.0
        
        # Energy uncertainty enabled - use max multiplier from config as safety factor
        max_energy_multiplier = traffic_config.get('max_energy_multiplier', 1.20)
        
        return float(max_energy_multiplier)

    def _build_segments(self, deliveries: Sequence[int], env) -> List[Dict]:
        """Precompute travel options for each leg with energy safety margin from config."""
        segments = []
        graph = env.transport_graph
        
        # Use cached safety factor computed from config
        energy_safety_factor = self._energy_safety_factor if self._energy_safety_factor is not None else 1.0
        
        for idx in range(len(deliveries) - 1):
            start = deliveries[idx]
            target = deliveries[idx + 1]
            
            # Use base values without traffic/uncertainty multipliers
            try:
                direct_time = graph.get_time_distance(start, target)
            except Exception:
                direct_time = float("inf")
            
            direct_energy = graph.get_path_energy(start, target) * energy_safety_factor

            charger_options = []
            for charger in env.charging_nodes:
                try:
                    to_time = graph.get_time_distance(start, charger)
                    from_time = graph.get_time_distance(charger, target)
                except Exception:
                    continue
                    
                to_energy = graph.get_path_energy(start, charger) * energy_safety_factor
                from_energy = graph.get_path_energy(charger, target) * energy_safety_factor
                
                if not math.isfinite(to_energy) or not math.isfinite(from_energy):
                    continue
                    
                rate, eff = self._charger_profile(env, charger)
                charger_options.append({
                    "node": charger,
                    "to_time": float(to_time),
                    "from_time": float(from_time),
                    "to_energy": float(to_energy),
                    "from_energy": float(from_energy),
                    "rate": float(rate),
                    "efficiency": float(eff),
                })

            segments.append({
                "start": start,
                "end": target,
                "direct_time": float(direct_time),
                "direct_energy": float(direct_energy),
                "chargers": charger_options,
            })
            
        return segments

    def _charger_profile(self, env, node: int) -> Tuple[float, float]:
        """Return (rate, efficiency) for a charger node."""
        if hasattr(env, "charging_station"):
            ctype = env.charging_station.charger_type.get(node, "Level2")
        else:
            ctype = env.transport_graph.get_charger_type(node) or "Level2"
        key = "dcfast" if str(ctype).lower() == "dcfast" else "level2"
        cfg = env.charging_config[key]
        return float(cfg["charge_rate"]), float(cfg["efficiency"])

    def _create_emergency_plan(self, truck, env) -> List[PlanStep]:
        """Create emergency plan: go to nearest charger and charge to full."""
        plan = []
        current_node = truck.current_node
        current_battery = truck.current_battery
        battery_cap = truck.battery_capacity
        
        # Find nearest reachable charger
        min_energy = float('inf')
        best_charger = None
        
        for charger_node in env.charging_nodes:
            try:
                energy_needed = env.transport_graph.get_path_energy(current_node, int(charger_node))
                if energy_needed < current_battery and energy_needed < min_energy:
                    min_energy = energy_needed
                    best_charger = charger_node
            except Exception:
                continue
        
        if best_charger is not None:
            # Navigate to nearest charger
            plan.append(PlanStep(kind="nav_charger", target=best_charger))
            
            # Charge to 95% capacity
            battery_after_travel = current_battery - min_energy
            soc_after_travel = battery_after_travel / battery_cap
            target_soc = 0.95
            charge_needed = (target_soc - soc_after_travel) * battery_cap
            
            # Get charger rate and efficiency
            rate, eff = self._charger_profile(env, best_charger)
            charge_hours_needed = charge_needed / (rate * eff)
            
            # Round up to nearest available duration with 30% buffer
            plan.append(PlanStep(kind="charge", duration=charge_hours_needed * 1.30))
        
        # Navigate to next delivery
        plan.append(PlanStep(kind="nav_delivery"))
        
        if self.verbose:
            print(f"[Simple Optimal] Emergency plan created: {len(plan)} steps")
        
        return plan

    def _to_env_action(self, step: PlanStep, env) -> int:
        """Map a PlanStep to the environment's discrete action index."""
        if step.kind == "nav_delivery":
            return env.num_charging_nodes
            
        if step.kind == "nav_charger":
            try:
                idx = env.charging_nodes.index(step.target)
                return idx
            except ValueError:
                return env.action_space.sample()
                
        if step.kind == "charge":
            durations = sorted([float(d) for d in env.charging_config["charge_durations"]])
            requested = max(0.0, float(step.duration or 0.0))
            
            # Add 10% safety buffer when rounding up
            requested_with_buffer = requested * 1.10
            
            # Pick the smallest available duration that is >= requested_with_buffer
            candidate = None
            for d in durations:
                if d >= requested_with_buffer - 1e-6:
                    candidate = d
                    break
                    
            if candidate is None:
                # If no duration is large enough, use maximum available
                candidate = durations[-1]
                if self.verbose:
                    print(f"[Simple Optimal] Warning: requested {requested:.2f}h (with buffer: {requested_with_buffer:.2f}h) exceeds max duration {candidate:.2f}h")
                
            dur_idx = env.charging_config["charge_durations"].index(candidate)
            return env.num_navigation_actions + dur_idx
            
        return env.action_space.sample()
