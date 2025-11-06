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
    """Greedy heuristic policy for truck routing decisions."""

    def __init__(self, verbose: bool = False):
        """
        Initialize the heuristic policy.

        Args:
            verbose: Print decision reasoning
        """
        self.verbose = verbose
        self.decision_history = []

    def get_action(self, env) -> int:
        """
        Compute the best action for the active truck using heuristic rules.

        Decision Logic:
        1. If truck can reach next delivery AND has battery for next charger after: GO TO DELIVERY
        2. Else if truck can reach nearest charger: GO TO CHARGER
        3. Else: GO TO DELIVERY (best attempt, may fail but indicates impossible situation)

        Args:
            env: The EventDrivenTruckEnv environment object

        Returns:
            Action index (0 to action_space.n - 1)
        """
        # Get active truck
        truck_id = env.active_truck_id
        if truck_id is None:
            return env.action_space.sample()

        truck = env.trucks[truck_id]
        current_node = int(truck.current_node)
        current_battery = truck.current_battery
        battery_capacity = truck.battery_capacity

        # Get next delivery target
        next_delivery = truck.get_next_delivery_target()
        if next_delivery is None:
            # No more deliveries - we're done or need to wait
            return None

        next_delivery = int(next_delivery)

        # Get transportation graph for distance queries
        graph = env.transport_graph
        charging_nodes = env.charging_nodes

        # Calculate energy needed to reach next delivery
        energy_to_delivery = graph.get_path_energy(current_node, next_delivery)

        if energy_to_delivery == float('inf'):
            # Can't reach delivery at all - go to nearest charger
            action = self._navigate_to_nearest_charger(
                current_node, charging_nodes, graph, env
            )
            if self.verbose:
                print(f"[Heuristic] Truck {truck_id}: Cannot reach delivery @ {next_delivery}")
                print(f"  → Action: Navigate to nearest charger")
            return action

        # Check if we can reach the delivery
        can_reach_delivery = energy_to_delivery <= current_battery

        if can_reach_delivery:
            # Check if we can reach a charger from the delivery location
            nearest_charger, distance_to_charger = graph.get_nearest_charging_node(
                next_delivery
            )

            if nearest_charger is not None and distance_to_charger != float('inf'):
                # Check if we have enough battery to reach delivery AND charger
                total_energy = energy_to_delivery + distance_to_charger
                can_reach_charger_after_delivery = total_energy <= battery_capacity

                if can_reach_charger_after_delivery:
                    # Safe to go to delivery - we can reach a charger after
                    action = self._navigate_to_delivery(next_delivery, env)
                    if self.verbose:
                        print(f"[Heuristic] Truck {truck_id}: Safe path to delivery @ {next_delivery}")
                        print(f"  - Energy to delivery: {energy_to_delivery:.1f} kWh")
                        print(f"  - Can reach charger after (dist={distance_to_charger:.1f})")
                        print(f"  → Action: Navigate to delivery")
                    return action
                else:
                    # Can reach delivery but not charger after - need to charge first
                    action = self._navigate_to_nearest_charger(
                        current_node, charging_nodes, graph, env
                    )
                    if self.verbose:
                        print(f"[Heuristic] Truck {truck_id}: Cannot reach charger after delivery")
                        print(f"  - Energy to delivery: {energy_to_delivery:.1f}")
                        print(f"  - Total needed: {total_energy:.1f} > capacity {battery_capacity:.1f}")
                        print(f"  → Action: Charge first at current location")
                    return action
            else:
                # No charger reachable from delivery - risky but go anyway
                action = self._navigate_to_delivery(next_delivery, env)
                if self.verbose:
                    print(f"[Heuristic] Truck {truck_id}: No charger from delivery @ {next_delivery}")
                    print(f"  → Action: Navigate to delivery (risky)")
                return action
        else:
            # Can't even reach the delivery - must charge first
            action = self._navigate_to_nearest_charger(
                current_node, charging_nodes, graph, env
            )
            if self.verbose:
                print(f"[Heuristic] Truck {truck_id}: Insufficient battery for delivery")
                print(f"  - Current battery: {current_battery:.1f} kWh")
                print(f"  - Energy needed: {energy_to_delivery:.1f} kWh")
                print(f"  → Action: Navigate to nearest charger")
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

    def _navigate_to_nearest_charger(
        self, current_node: int, charging_nodes: List[int], graph, env
    ) -> int:
        """
        Find and navigate to the nearest charger.

        Args:
            current_node: Current truck position
            charging_nodes: List of all charging station nodes
            graph: Transportation graph
            env: The environment

        Returns:
            Action index for the nearest charger
        """
        # Find nearest reachable charger
        min_distance = float('inf')
        nearest_charger = None
        nearest_charger_action = 0

        for charger_node in charging_nodes:
            distance = graph.get_path_energy(current_node, charger_node)
            if distance < min_distance and distance != float('inf'):
                min_distance = distance
                nearest_charger = charger_node
                nearest_charger_action = charging_nodes.index(charger_node)

        if nearest_charger is None:
            # No reachable charger - this is bad, return random valid action
            if self.verbose:
                print(f"[Heuristic] WARNING: No reachable charger from {current_node}!")
            return 0

        return nearest_charger_action

    def _get_charge_duration(self, truck, battery_capacity: float) -> int:
        """
        Get recommended charge duration (action index for charge duration).

        Charges to near-full capacity.

        Args:
            truck: The truck object
            battery_capacity: Full battery capacity

        Returns:
            Charge action index (0-3 typically for 1-4 hours)
        """
        # Simple strategy: charge for the maximum available time
        # This can be customized based on battery level
        return 0  # First charge option (typically 1 hour)

    def get_action_with_explanations(self, env) -> Tuple[int, str]:
        """
        Get action and return explanation string.

        Args:
            env: The environment

        Returns:
            Tuple of (action, explanation_string)
        """
        truck_id = env.active_truck_id
        if truck_id is None:
            return env.action_space.sample(), "No active truck"

        truck = env.trucks[truck_id]
        current_node = int(truck.current_node)
        current_battery = truck.current_battery
        battery_capacity = truck.battery_capacity

        next_delivery = truck.get_next_delivery_target()
        if next_delivery is None:
            return None, "No more deliveries"

        next_delivery = int(next_delivery)

        graph = env.transport_graph
        charging_nodes = env.charging_nodes

        # Calculate distances
        energy_to_delivery = graph.get_path_energy(current_node, next_delivery)

        explanation = f"Truck {truck_id} @ node {current_node} (battery: {current_battery:.1f}/{battery_capacity:.1f})\n"

        if energy_to_delivery == float('inf'):
            explanation += f"Cannot reach delivery @ {next_delivery} - navigating to charger"
            action = self._navigate_to_nearest_charger(
                current_node, charging_nodes, graph, env
            )
            return action, explanation

        can_reach_delivery = energy_to_delivery <= current_battery
        explanation += f"Distance to next delivery @ {next_delivery}: {energy_to_delivery:.1f} kWh\n"

        if can_reach_delivery:
            nearest_charger, distance_to_charger = graph.get_nearest_charging_node(
                next_delivery
            )

            if nearest_charger is not None and distance_to_charger != float('inf'):
                total_energy = energy_to_delivery + distance_to_charger
                can_reach_charger_after = total_energy <= battery_capacity

                if can_reach_charger_after:
                    explanation += f"Can reach delivery + charger ({distance_to_charger:.1f} kWh) → GO TO DELIVERY"
                    action = self._navigate_to_delivery(next_delivery, env)
                else:
                    explanation += (
                        f"Cannot reach charger after delivery ({total_energy:.1f} > {battery_capacity:.1f}) "
                        f"→ CHARGE FIRST"
                    )
                    action = self._navigate_to_nearest_charger(
                        current_node, charging_nodes, graph, env
                    )
            else:
                explanation += "No charger reachable from delivery → GO TO DELIVERY (risky)"
                action = self._navigate_to_delivery(next_delivery, env)
        else:
            explanation += f"Insufficient battery ({current_battery:.1f} < {energy_to_delivery:.1f}) → CHARGE"
            action = self._navigate_to_nearest_charger(
                current_node, charging_nodes, graph, env
            )

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

    def print_statistics(self):
        """Print summary statistics about heuristic decisions."""
        if not self.decision_history:
            print("[Heuristic] No decisions logged yet")
            return

        print(f"\n[Heuristic] Decision Statistics: {len(self.decision_history)} decisions")
        truck_ids = set(d["truck_id"] for d in self.decision_history)
        print(f"  - Trucks: {sorted(truck_ids)}")
        print(f"  - Decisions per truck: {len(self.decision_history) / len(truck_ids):.1f} avg")
