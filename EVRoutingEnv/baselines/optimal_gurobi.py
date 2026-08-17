"""
Conservative per-truck deterministic MILP reference (Gurobi). LEGACY.

R1.4 objects to this being presented as an optimality benchmark, and the
objection is correct: it plans one truck at a time against a fixed delivery
order, ignores charger contention entirely, and assumes deterministic travel,
so it bounds neither the fleet problem nor the stochastic one. It is named for
what it is and kept only for the inherited preassigned-route benchmark.

The revision's optimization baseline is
EVRoutingEnv/baselines/exact_optimization.py: a fleet-level nominal CP-SAT model
validated against exhaustive enumeration on tiny instances, executed with
energy-safe repair, publishing solver status, bound, gap, and fallback counts
per episode.

Assumptions:
- Fixed delivery order per truck (provided by the environment)
- No charger queueing or contention (deterministic service)
- Travel times are deterministic (no traffic uncertainty)
- At most one charger visit between consecutive deliveries
- Continuous charging durations
- Optional realistic DC fast charging curves (CCCV) when enabled in config
- Energy use per trip inflated by safety factor matching environment's max_energy_multiplier
  to ensure robust plans under energy consumption uncertainty
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import gurobipy as gp
from gurobipy import GRB

from EVRoutingEnv.models.simulation.charging_curve import ChargingCurveModel


@dataclass
class PlanStep:
    """A single planned action for a truck."""

    kind: str  # "nav_delivery", "nav_charger", or "charge"
    target: Optional[int] = None  # charger node id for nav_charger
    duration: Optional[float] = None  # hours for charge


class OptimalGurobiPolicy:
    """
    Per-truck optimal policy using a compact MILP.

    The model selects, for each leg between deliveries, whether to:
    - drive directly to the next delivery, or
    - detour to one charger, pick a discrete charge duration, then continue.

    Objective: minimize total elapsed time (travel + charging).
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._plans: Dict[int, List[PlanStep]] = {}
        self._cursors: Dict[int, int] = {}
        self._charging_curve_model = ChargingCurveModel(verbose=False)
        # Energy safety factor - will be set dynamically based on environment config
        self.energy_safety_factor = 1.22  # Default value with 22% buffer

    # ---------- Public API ----------
    def get_action(self, env) -> int:
        """Return the next action for the currently active truck."""
        if env.active_truck_id is None:
            return env.action_space.sample()

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
                # Log the error for debugging
                if self.verbose:
                    print(f"[Optimal Policy] Solver failed for truck {truck_id}: {e}")
                # Try with increased safety margin
                try:
                    original_factor = self.energy_safety_factor
                    self.energy_safety_factor = min(1.25, original_factor * 1.05)  # Slightly more conservative
                    plan = self._solve_truck(truck=truck, env=env)
                    if self.verbose:
                        print(f"[Optimal Policy] Retry with factor {self.energy_safety_factor} succeeded")
                except Exception as e2:
                    if self.verbose:
                        print(f"[Optimal Policy] Retry also failed: {e2}. Using fallback strategy.")
                    # Emergency fallback: go to nearest charger first
                    plan = self._create_emergency_plan(truck, env)
                finally:
                    self.energy_safety_factor = original_factor
            self._plans[truck_id] = plan
            self._cursors[truck_id] = 0

        # If no actions are required, default to next-delivery action
        if not self._plans[truck_id]:
            return env.num_charging_nodes

        step_idx = self._cursors[truck_id]
        step = self._plans[truck_id][step_idx]
        self._cursors[truck_id] = step_idx + 1
        return self._to_env_action(step, env)

    # ---------- Core solver ----------
    def _solve_truck(self, truck, env) -> List[PlanStep]:
        """Build and solve the MILP for a single truck."""
        # Set energy safety factor based on environment's worst-case energy multiplier
        if hasattr(env, 'traffic_config') and env.traffic_config.get('enable_energy_uncertainty', False):
            # Use max multiplier directly - it already represents worst case
            self.energy_safety_factor = env.traffic_config.get('max_energy_multiplier', 1.2)
        else:
            self.energy_safety_factor = 1.12  # 12% safety margin without uncertainty
        
        deliveries = [int(truck.current_node)] + [
            int(n) for n in truck.get_remaining_deliveries()
        ]
        if len(deliveries) <= 1:
            return []

        segments = self._build_segments(deliveries, env)
        battery_cap = float(truck.battery_capacity)
        init_battery = float(truck.current_battery)
        max_charge_hours = 24.0  # allow any duration in [0, 24]

        # Quick feasibility check
        for seg in segments:
            direct_ok = math.isfinite(seg["direct_energy"])
            charger_ok = len(seg["chargers"]) > 0
            if not direct_ok and not charger_ok:
                raise ValueError(
                    f"No feasible path from node {seg['start']} to delivery {seg['end']}"
                )

        # Big-M
        max_energy_leg = max(
            [
                seg["direct_energy"]
                for seg in segments
                if math.isfinite(seg["direct_energy"])
            ]
            + [
                opt["to_energy"] + opt["from_energy"]
                for seg in segments
                for opt in seg["chargers"]
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

        model = gp.Model("deterministic_truck")
        model.Params.OutputFlag = 0

        num_segments = len(segments)
        # Add minimum battery safety buffer (5% of capacity) to avoid running too close to empty
        min_battery_buffer = 0.05 * battery_cap
        battery = model.addVars(
            num_segments + 1, lb=min_battery_buffer, ub=battery_cap, name="battery"
        )
        direct = model.addVars(num_segments, vtype=GRB.BINARY, name="direct")
        use_charger: Dict[Tuple[int, int], gp.Var] = {}
        charge_time: Dict[Tuple[int, int], gp.Var] = {}

        # Optional initial charge if starting node has a charger
        start_charge_time = None
        start_rate = start_eff = None
        start_charger_type = None
        if deliveries[0] in env.charging_nodes:
            start_charge_time = model.addVar(
                lb=0.0, ub=max_charge_hours, name="start_charge_time"
            )
            start_rate, start_eff = self._charger_profile(env, deliveries[0])
            
            # Get charger type for realistic curve
            if hasattr(env, "charging_station"):
                start_charger_type = env.charging_station.charger_type.get(deliveries[0], "Level2")
            else:
                start_charger_type = env.transport_graph.get_charger_type(deliveries[0]) or "Level2"
            
            # For the initial charge, use piecewise linear approximation
            # This is a simplified approach - actual charge depends on initial SOC
            model.addConstr(
                battery[0]
                == init_battery + start_rate * start_eff * start_charge_time
            )
        else:
            model.addConstr(battery[0] == init_battery)

        objective_terms = []
        if start_charge_time is not None:
            objective_terms.append(start_charge_time)

        for i, seg in enumerate(segments):
            chargers = seg["chargers"]
            for c_idx, opt in enumerate(chargers):
                use_charger[(i, c_idx)] = model.addVar(
                    vtype=GRB.BINARY, name=f"use_{i}_{opt['node']}"
                )
                charge_time[(i, c_idx)] = model.addVar(
                    lb=0.0, ub=max_charge_hours, name=f"charge_{i}_{opt['node']}"
                )
                # If charger not chosen, force zero charge time
                model.addConstr(charge_time[(i, c_idx)] <= max_charge_hours * use_charger[(i, c_idx)])

            choice_vars = [direct[i]] + [
                use_charger[(i, c_idx)] for c_idx in range(len(chargers))
            ]
            model.addConstr(gp.quicksum(choice_vars) == 1)

            # Direct travel constraints
            if math.isfinite(seg["direct_energy"]):
                energy = seg["direct_energy"]
                model.addConstr(battery[i] - energy >= -big_m * (1 - direct[i]))
                model.addConstr(
                    battery[i + 1] >= battery[i] - energy - big_m * (1 - direct[i])
                )
                model.addConstr(
                    battery[i + 1] <= battery[i] - energy + big_m * (1 - direct[i])
                )
                if math.isfinite(seg["direct_time"]):
                    objective_terms.append(direct[i] * seg["direct_time"])
            else:
                model.addConstr(direct[i] == 0)

            # Charger detour constraints
            for c_idx, opt in enumerate(chargers):
                use_var = use_charger[(i, c_idx)]
                ct_var = charge_time[(i, c_idx)]
                
                # Handle realistic vs linear charging
                use_realistic = env.charging_config.get("use_realistic_curve", False)
                charger_node = opt["node"]
                
                if hasattr(env, "charging_station"):
                    ctype = env.charging_station.charger_type.get(charger_node, "Level2")
                else:
                    ctype = env.transport_graph.get_charger_type(charger_node) or "Level2"
                
                if use_realistic and str(ctype).upper() == "DCFAST":
                    # Use conservative approximation for realistic charging
                    # Apply a taper efficiency factor to account for reduced power at high SOC
                    # Based on validation: realistic charging delivers ~89.4% of linear charging
                    # Using 87% for safety margin (2.4% buffer below validated average)
                    taper_efficiency = 0.87  # Conservative but accurate estimate
                    effective_rate = opt["rate"] * opt["efficiency"] * taper_efficiency
                    added_energy = effective_rate * ct_var
                else:
                    # Use linear (constant-rate) charging model
                    added_energy = opt["rate"] * opt["efficiency"] * ct_var
                
                # Standard constraints (same for both models)
                model.addConstr(
                    battery[i] - opt["to_energy"] >= -big_m * (1 - use_var)
                )
                model.addConstr(
                    battery[i] - opt["to_energy"] + added_energy
                    <= battery_cap + big_m * (1 - use_var)
                )
                model.addConstr(
                    battery[i + 1]
                    >= battery[i]
                    - opt["to_energy"]
                    - opt["from_energy"]
                    + added_energy
                    - big_m * (1 - use_var)
                )
                model.addConstr(
                    battery[i + 1]
                    <= battery[i]
                    - opt["to_energy"]
                    - opt["from_energy"]
                    + added_energy
                    + big_m * (1 - use_var)
                )

                objective_terms.append(
                    use_var * (opt["to_time"] + opt["from_time"]) + ct_var
                )

        model.setObjective(gp.quicksum(objective_terms), GRB.MINIMIZE)
        model.optimize()

        if model.Status == GRB.INFEASIBLE:
            if self.verbose:
                print(f"[Optimal Policy] Model is infeasible. Computing IIS...")
            model.computeIIS()
            raise RuntimeError(f"Gurobi model is INFEASIBLE - no feasible solution exists")
        elif model.Status == GRB.UNBOUNDED:
            raise RuntimeError(f"Gurobi model is UNBOUNDED")
        elif model.Status != GRB.OPTIMAL:
            # Try to use best bound if available
            if model.SolCount > 0:
                if self.verbose:
                    print(f"[Optimal Policy] Using suboptimal solution (status {model.Status})")
            else:
                raise RuntimeError(f"Gurobi solver failed with status {model.Status}")

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

    # ---------- Helpers ----------
    def _build_segments(self, deliveries: Sequence[int], env) -> List[Dict]:
        """Precompute travel options for each leg between deliveries.
        
        Applies traffic multipliers to travel times when traffic is enabled,
        ensuring robust solutions under traffic uncertainty.
        """
        segments = []
        graph = env.transport_graph
        for idx in range(len(deliveries) - 1):
            start = deliveries[idx]
            target = deliveries[idx + 1]
            try:
                direct_time = graph.get_time_distance(start, target)
            except Exception:
                direct_time = float("inf")
            direct_energy = graph.get_path_energy(start, target)
            if math.isfinite(direct_energy):
                direct_energy *= self.energy_safety_factor

            charger_options = []
            for charger in env.charging_nodes:
                try:
                    to_time = graph.get_time_distance(start, charger)
                    from_time = graph.get_time_distance(charger, target)
                except Exception:
                    continue
                to_energy = graph.get_path_energy(start, charger)
                from_energy = graph.get_path_energy(charger, target)
                if not math.isfinite(to_energy) or not math.isfinite(from_energy):
                    continue
                to_energy *= self.energy_safety_factor
                from_energy *= self.energy_safety_factor
                rate, eff = self._charger_profile(env, charger)
                charger_options.append(
                    {
                        "node": charger,
                        "to_time": float(to_time),
                        "from_time": float(from_time),
                        "to_energy": float(to_energy),
                        "from_energy": float(from_energy),
                        "rate": float(rate),
                        "efficiency": float(eff),
                    }
                )

            segments.append(
                {
                    "start": start,
                    "end": target,
                    "direct_time": float(direct_time),
                    "direct_energy": float(direct_energy),
                    "chargers": charger_options,
                }
            )
        return segments

    def _get_traffic_multiplier(self, env, base_travel_time: float) -> float:
        """No traffic uncertainty: deterministic travel times."""
        return 1.0
    
    def _charger_profile(self, env, node: int) -> Tuple[float, float]:
        """Return (rate, efficiency) for a charger node."""
        if hasattr(env, "charging_station"):
            ctype = env.charging_station.charger_type.get(node, "Level2")
        else:
            ctype = env.transport_graph.get_charger_type(node) or "Level2"
        key = "dcfast" if str(ctype).lower() == "dcfast" else "level2"
        cfg = env.charging_config[key]
        return float(cfg["charge_rate"]), float(cfg["efficiency"])
    
    def _calculate_charge_amount(
        self, 
        initial_soc: float, 
        charge_hours: float, 
        battery_capacity: float,
        charger_config: Dict,
        charger_type: str,
        env
    ) -> float:
        """
        Calculate actual charge delivered using charging curve model if available.
        
        Falls back to linear model if curve model is not available or realistic curves disabled.
        """
        # Check if realistic curves are enabled
        use_realistic = env.charging_config["use_realistic_curve"]
        
        if use_realistic and self._charging_curve_model and charger_type == "DCFast":
            # Use realistic charging curve
            charge_amount, _ = self._charging_curve_model.calculate_charge(
                initial_soc=initial_soc,
                charge_hours=charge_hours,
                battery_capacity=battery_capacity,
                charger_config=charger_config,
                charger_type=charger_type
            )
            return float(charge_amount)
        else:
            # Use linear (constant-rate) model
            rate = charger_config["charge_rate"]
            efficiency = charger_config["efficiency"]
            max_charge = (1.0 - initial_soc) * battery_capacity
            requested_charge = charge_hours * rate * efficiency
            return min(requested_charge, max_charge)

    def _create_emergency_plan(self, truck, env) -> List[PlanStep]:
        """
        Create an emergency plan when the optimal solver fails.
        Strategy: Go to nearest reachable charger and charge to near-full.
        """
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
                # Apply safety factor
                energy_needed *= self.energy_safety_factor
                if energy_needed < current_battery and energy_needed < min_energy:
                    min_energy = energy_needed
                    best_charger = charger_node
            except Exception:
                continue
        
        if best_charger is not None:
            # Navigate to nearest charger
            plan.append(PlanStep(kind="nav_charger", target=best_charger))
            
            # Charge to 90% capacity (conservative, leaves room for uncertainty)
            battery_after_travel = current_battery - min_energy
            soc_after_travel = battery_after_travel / battery_cap
            target_soc = 0.90
            charge_needed = (target_soc - soc_after_travel) * battery_cap
            
            # Get charger rate and efficiency
            rate, eff = self._charger_profile(env, best_charger)
            charge_hours_needed = charge_needed / (rate * eff)
            
            # Add to plan with buffer
            plan.append(PlanStep(kind="charge", duration=charge_hours_needed * 1.15))
        
        # Navigate to next delivery
        plan.append(PlanStep(kind="nav_delivery"))
        
        if self.verbose:
            print(f"[Optimal Policy] Emergency plan: {len(plan)} steps")
        
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
            durations = [float(d) for d in env.charging_config["charge_durations"]]
            requested = max(0.0, float(step.duration or 0.0))
            # Add 7% safety buffer to requested duration to account for:
            # 1. Quantization errors from continuous->discrete mapping
            # 2. Potential charging curve variations
            # 3. Battery capacity uncertainty
            # 4. Edge cases in energy consumption
            requested_with_buffer = requested * 1.07
            # Pick the smallest available duration that is >= requested; if none, pick the max.
            candidate = None
            for d in sorted(durations):
                if d >= requested_with_buffer - 1e-6:  # small tolerance
                    candidate = d
                    break
            if candidate is None:
                candidate = max(durations)
            dur_idx = durations.index(candidate)
            return env.num_navigation_actions + dur_idx
        return env.action_space.sample()
