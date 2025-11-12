"""
Heuristic Policy for Truck Routing Environment.

This module provides a greedy heuristic algorithm that ensures trucks complete
all deliveries by intelligently deciding when to navigate to deliveries vs chargers.

The heuristic uses a feasibility check to determine if the truck can reach:
1. The next delivery
2. The next delivery AND the nearest charger from there
3. If neither is feasible, it navigates to the nearest charger

This guarantees no stranded trucks (unless impossible from the start).
"""

import numpy as np
from typing import Optional, Tuple, List
import heapq


class HeuristicPolicy:
    """Greedy heuristic policy for truck routing decisions.

    Simplified charger selection (no queue or wait-time awareness):
    - Choose the reachable charger requiring the least energy from current node
    - Always ensure energy for: current node → next delivery → nearest charger from that delivery
    - When at a charger, select minimal charge duration to satisfy required energy
    """

    def __init__(self, verbose: bool = False, buffer_frac: float = 0.05, min_target_soc: float = 0.6, multi_delivery_depth: int = 2):
        """
        Initialize the heuristic policy.

        Args:
            verbose: Print decision reasoning
            buffer_frac: Extra safety buffer on required energy (e.g., 0.25 => +25%)
            min_target_soc: Minimum SOC target (fraction of capacity) when requirement becomes large
            multi_delivery_depth: Number of upcoming deliveries (with chargers in between) to plan energy for
        """
        self.verbose = verbose
        self.decision_history = []
        self.buffer_frac = max(0.0, float(buffer_frac))
        self.min_target_soc = float(min(1.0, max(0.0, min_target_soc)))
        self.multi_delivery_depth = max(1, int(multi_delivery_depth))

    # ---- Energy planning helpers ----
    def _compute_span_energy(self, env, truck) -> Tuple[float, int]:
        """Compute required energy to service up to multi_delivery_depth deliveries
        from current state, assuming we recharge at the nearest charger after
        each delivery except possibly the final one in the span.

        Returns (required_energy, actual_depth_used).
        """
        graph = env.transport_graph
        remaining = list(truck.get_remaining_deliveries())
        if not remaining:
            return 0.0, 0
        start_node = int(truck.current_node)
        total_energy = 0.0
        depth_used = 0
        for i, delivery_node in enumerate(remaining):
            if depth_used >= self.multi_delivery_depth:
                break
            delivery_node = int(delivery_node)
            e_seg = graph.get_path_energy(start_node, delivery_node)
            if e_seg == float('inf'):
                return float('inf'), depth_used
            total_energy += e_seg
            is_last_in_span = (i == self.multi_delivery_depth - 1) or (i == len(remaining) - 1)
            # After delivery, plan to go to nearest charger unless last planned delivery (to maintain buffer ability)
            if not is_last_in_span:
                nearest_charger, e_to_charger = graph.get_nearest_charging_node(delivery_node)
                if nearest_charger is None or e_to_charger == float('inf'):
                    return float('inf'), depth_used
                total_energy += e_to_charger
                start_node = int(nearest_charger)
            else:
                start_node = delivery_node  # end span at delivery location
            depth_used += 1
        return total_energy, depth_used

    def get_action(self, env) -> int:
        """Return best action index for current state.

        Args:
            env: The environment instance (NOT the observation array)

        Delegates to the unified decision maker and logs the decision.
        
        Note: This policy requires the full environment object, not just the observation.
        """
        # Defensive check to ensure we received an environment, not an observation
        if isinstance(env, np.ndarray):
            raise TypeError(
                "HeuristicPolicy.get_action() expects the environment object, not the observation array.\n"
                "Usage: action = policy.get_action(env)  # NOT policy.get_action(obs)"
            )
        
        if not hasattr(env, 'active_truck_id'):
            raise TypeError(
                f"HeuristicPolicy.get_action() expects an environment object with 'active_truck_id' attribute.\n"
                f"Received object of type: {type(env).__name__}"
            )
        
        action, explanation = self._decide_action(env)
        truck_id = env.active_truck_id
        if truck_id is not None:
            self.log_decision(truck_id, action, explanation)
        if self.verbose and explanation:
            print(explanation)
        return action

    def _navigate_to_delivery(self, delivery_node: int, env) -> int:
        """
        Get the action to navigate to the next delivery.

        Args:
            delivery_node: Target delivery node
            env: The environment

        Returns:
            Action index for "go to next delivery"
        """
        return env.num_charging_nodes  # Special action for next delivery

    def _navigate_to_charger(self, charger_node: int, env) -> int:
        """
        Get the action to navigate to a specific charger.

        Args:
            charger_node: Target charger node
            env: The environment

        Returns:
            Action index for navigating to the charger
        """
        try:
            charger_idx = env.charging_nodes.index(charger_node)
            return charger_idx
        except ValueError:
            raise ValueError(f"Node {charger_node} is not a valid charging node")

    def _navigate_to_best_charger(self, current_node: int, env, must_be_reachable: bool = True, target_node: Optional[int] = None) -> int:
        """Select charger minimizing total time-to-ready for delivery+post-delivery charger.
        Scoring:
          - travel feasibility with current battery
          - charge_time at that charger to meet requirement from charger (delivery -> nearest charger)
        Ignores queue/wait times entirely. If must_be_reachable=True, restrict to chargers within current battery.
        """
        graph = env.transport_graph
        charging_nodes = env.charging_nodes
        current_battery = env.trucks[env.active_truck_id].current_battery if env.active_truck_id is not None else float('inf')
        best_idx = None
        best_score = float('inf')
        truck = env.trucks[env.active_truck_id]
        battery_capacity = truck.battery_capacity
        for node in charging_nodes:
            energy_to_charger = graph.get_path_energy(current_node, node)
            if energy_to_charger == float('inf'):
                continue
            if must_be_reachable and energy_to_charger > current_battery:
                continue
            # From this charger, compute requirement to target (delivery) and then to nearest charger
            if target_node is None:
                # Fallback: minimize energy to reach only
                score = energy_to_charger
            else:
                e_chg_to_deliv = graph.get_path_energy(node, int(target_node))
                if e_chg_to_deliv == float('inf'):
                    # Can't go from charger to delivery; skip
                    continue
                nearest_after, e_deliv_to_next = graph.get_nearest_charging_node(int(target_node))
                if nearest_after is None or e_deliv_to_next == float('inf'):
                    # No charger reachable after delivery; skip
                    continue
                required_from_charger = (e_chg_to_deliv + e_deliv_to_next) * (1.0 + self.buffer_frac)
                # Battery upon arrival at charger:
                battery_on_arrival = max(0.0, current_battery - energy_to_charger)
                # Charging rate at this charger
                ctype = env.charger_type.get(node, "level2") if hasattr(env, 'charger_type') else "level2"
                cfg = env.charging_config["dcfast"] if ctype == "DCFast" else env.charging_config["level2"]
                rate = cfg["charge_rate"] * cfg["efficiency"]
                deficit = max(0.0, min(required_from_charger, battery_capacity) - battery_on_arrival)
                # Time to charge needed energy (in hours)
                charge_time = deficit / max(rate, 1e-6)
                # Scoring: prefer lower charge_time; tie-break with lower energy_to_charger
                score = charge_time + 1e-6 * energy_to_charger
            if score < best_score - 1e-9:
                best_score = score
                best_idx = charging_nodes.index(node)
        if best_idx is None:
            raise ValueError(
                "No reachable charger with current battery. Cannot navigate to any charger without violating battery constraint."
            )
        return best_idx

    def _get_charge_duration(self, env, truck, required_energy: float) -> int:
        """Compute minimal charge duration index to satisfy required_energy from current node."""
        durations = env.charging_config["charge_durations"]
        node = int(truck.current_node)
        charger_type = env.charger_type.get(node, "level2") if hasattr(env, 'charger_type') else "level2"
        if charger_type == "DCFast":
            cfg = env.charging_config["dcfast"]
        else:
            cfg = env.charging_config["level2"]
        rate = cfg["charge_rate"] * cfg["efficiency"]  # kWh per hour
        deficit = max(0.0, required_energy - truck.current_battery)
        max_add = max(0.0, truck.battery_capacity - truck.current_battery)
        deficit = min(deficit, max_add)
        if deficit <= 1e-6:
            return 0
        needed_hours = deficit / max(rate, 1e-6)
        for idx, h in enumerate(durations):
            if h + 1e-9 >= needed_hours:
                return idx
        return len(durations) - 1

    def get_action_with_explanations(self, env) -> Tuple[int, str]:
        """Return (action, explanation) and log it."""
        action, explanation = self._decide_action(env)
        truck_id = env.active_truck_id
        if truck_id is not None:
            self.log_decision(truck_id, action, explanation)
        return action, explanation

    def log_decision(self, truck_id: int, action: int, explanation: str):
        """
        Log a decision for analysis.

        Args:
            truck_id: The truck making the decision
            action: The action chosen
            explanation: Explanation of the decision
        """
        self.decision_history.append(
            {
                "truck_id": truck_id,
                "action": action,
                "explanation": explanation,
            }
        )

    # ---- Internal helpers ----
    def _decide_action(self, env) -> Tuple[int, str]:
        """Core decision logic shared by get_action and get_action_with_explanations."""
        truck_id = env.active_truck_id
        if truck_id is None:
            return env.action_space.sample(), "No active truck"

        truck = env.trucks[truck_id]
        current_node = int(truck.current_node)
        current_battery = truck.current_battery
        battery_capacity = truck.battery_capacity

        next_delivery = truck.get_next_delivery_target()
        if next_delivery is None:
            # No more deliveries - truck should be complete
            # Return a safe action (try to go to next delivery which will be handled by env)
            return env.num_charging_nodes, "No more deliveries - truck complete"

        next_delivery = int(next_delivery)
        graph = env.transport_graph
        charging_nodes = env.charging_nodes

        # Energy to next delivery and span info (span used only for informative logging)
        span_energy, depth_used = self._compute_span_energy(env, truck)
        energy_to_delivery = graph.get_path_energy(current_node, next_delivery)
        expl = [
            f"[Heuristic] Truck {truck_id}:",
            f"  - Node {current_node}, Battery {current_battery:.1f}/{battery_capacity:.1f} kWh",
            f"  - Span energy (depth={depth_used}) requirement: {span_energy if span_energy!=float('inf') else 'inf'} kWh",
            f"  - Energy to first delivery @{next_delivery}: {energy_to_delivery:.1f} kWh",
            f"  - Buffer fraction: {self.buffer_frac:.2f}",
        ]

        # Log energy distance to every charger from current node
        charger_energies: List[Tuple[int, float]] = []
        for cn in charging_nodes:
            e = graph.get_path_energy(current_node, int(cn))
            charger_energies.append((int(cn), e))
        # Sort by finite energy first, then by energy value
        charger_energies.sort(key=lambda t: (np.isinf(t[1]), t[1]))
        expl.append(f"  - Energy to chargers from node {current_node}:")
        for cn, e in charger_energies:
            e_str = f"{e:.1f} kWh" if e != float('inf') else "inf"
            expl.append(f"      • charger {cn}: {e_str}")

        if energy_to_delivery == float('inf'):
            # Hard error: No path from current node to delivery
            raise ValueError(
                f"Infeasible routing: no path from node {current_node} to delivery {next_delivery}."
            )

        can_reach_delivery = energy_to_delivery <= current_battery
        remaining = truck.get_remaining_deliveries()
        is_final = len(remaining) == 1
        nearest_after, energy_deliv_to_chg = graph.get_nearest_charging_node(next_delivery)
        if nearest_after is None or energy_deliv_to_chg == float('inf'):
            # Hard error: No charger reachable from delivery
            raise ValueError(
                f"Infeasible routing: no charger reachable from delivery {next_delivery}."
            )

        # Feasibility check with full battery (starting at current node with full charge)
        full_required = energy_to_delivery + energy_deliv_to_chg
        
        # If current routing is infeasible, try to find an alternative charger
        if full_required > battery_capacity + 1e-6:
            # This route from current node is infeasible
            # If we're at a charger, try to find a better charger that CAN reach the delivery
            if current_node in charging_nodes:
                # Find a charger that can reach the delivery
                best_charger = None
                best_charger_energy = float('inf')
                
                for charger_node in charging_nodes:
                    if charger_node == current_node:
                        continue  # Already know this one doesn't work
                    
                    # Energy from this charger to delivery
                    e_to_delivery_from_charger = graph.get_path_energy(charger_node, next_delivery)
                    if e_to_delivery_from_charger == float('inf'):
                        continue
                    
                    # Total energy needed from this charger
                    total_from_charger = e_to_delivery_from_charger + energy_deliv_to_chg
                    
                    if total_from_charger <= battery_capacity + 1e-6:
                        # This charger can reach the delivery! Find the closest one
                        e_to_charger = graph.get_path_energy(current_node, charger_node)
                        if e_to_charger < best_charger_energy:
                            best_charger = charger_node
                            best_charger_energy = e_to_charger
                
                if best_charger is not None:
                    # Navigate to the better charger
                    action = self._navigate_to_charger(best_charger, env)
                    expl.append(f"  - Current charger cannot reach delivery {next_delivery}")
                    expl.append(f"  - Navigating to alternative charger @ node {best_charger} that CAN reach delivery")
                    return action, "\n".join(expl)
            
            # No alternative found - this is truly infeasible
            raise ValueError(
                "Infeasible routing: energy required (current→delivery→nearest charger) exceeds battery capacity. "
                f"required={full_required:.2f} kWh, capacity={battery_capacity:.2f} kWh, current_node={current_node}, delivery={next_delivery}"
            )
        
        battery_after_delivery = current_battery - energy_to_delivery
        can_reach_charger_after_delivery = battery_after_delivery >= energy_deliv_to_chg

        # Only proceed if we can reach delivery AND a charger after delivery (no exception for final)
        if can_reach_delivery and can_reach_charger_after_delivery:
            expl.append("  - Requirement satisfied (delivery + post-delivery charger): proceed to delivery")
            return self._navigate_to_delivery(next_delivery, env), "\n".join(expl)

        # Need to charge first (or move to charger)
        # Need to charge: decide target energy (buffered single-step requirement)
        required_total = energy_to_delivery + energy_deliv_to_chg
        buffered_total = required_total * (1.0 + self.buffer_frac) if required_total != float('inf') else float('inf')
        # Enforce feasibility vs capacity using unbuffered requirement (physical guarantee)
        if required_total > battery_capacity + 1e-6:
            raise ValueError(
                "Infeasible routing: even from full battery, requirement (current→delivery→nearest charger) exceeds capacity. "
                f"required={required_total:.2f} kWh, capacity={battery_capacity:.2f} kWh, node={current_node}, delivery={next_delivery}"
            )
        # Target buffered energy but not above capacity
        target_energy = min(battery_capacity, buffered_total) if required_total != float('inf') else (self.min_target_soc * battery_capacity)
        if current_node in charging_nodes:
            dur_idx = self._get_charge_duration(env, truck, target_energy)
            action = env.num_navigation_actions + dur_idx
            expl.append(
                f"  - Charging to meet buffered requirement. target_energy={target_energy:.1f} kWh (buffer={self.buffer_frac*100:.0f}%). "
                f"Charge {env.charging_config['charge_durations'][dur_idx]}h"
            )
            return action, "\n".join(expl)
        action = self._navigate_to_best_charger(current_node, env, must_be_reachable=True, target_node=next_delivery)
        expl.append("  - Navigate to charger (insufficient energy for delivery + next charger)")
        return action, "\n".join(expl)

    def print_statistics(self):
        """Print summary statistics about heuristic decisions."""
        if not self.decision_history:
            print("[Heuristic] No decisions logged yet")
            return

        print(f"\n[Heuristic] Decision Statistics: {len(self.decision_history)} decisions")
        truck_ids = set(d["truck_id"] for d in self.decision_history)
        print(f"  - Trucks: {sorted(truck_ids)}")
        print(f"  - Decisions per truck: {len(self.decision_history) / len(truck_ids):.1f} avg")
