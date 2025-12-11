"""
GNN State Representation with Single Strategic Charger Selection.

This module provides a simplified GNN action space where only ONE charging station
is considered between the current position and the next delivery. The charger is
selected strategically to minimize energy cost while ensuring all deliveries can
be completed.

Key Differences from Standard GNN State Space:
- Only one charger action is available (or none if can reach delivery directly)
- Charger selection algorithm ensures feasibility for all remaining deliveries
- Reduced action space complexity: from N chargers to 1 charger
- Action space: [strategic_charger, next_delivery, charge_1h, ..., charge_Nh]

Strategic Charger Selection Algorithm:
1. Check if truck can reach next delivery directly
   - If yes AND can continue after (reach charger or finish), no charger needed
   - If no, must charge first
   
2. Find the optimal charger that minimizes: energy_to_charger + energy_to_delivery
   - Must be reachable with current battery
   - Must allow reaching delivery after charging
   - Must allow continuing after delivery (reach another charger or complete)
   
3. Verify charger enables completion of ALL remaining deliveries
   - Check if total energy for remaining route is feasible
   - Account for need to reach chargers between deliveries

Raises:
    ValueError: If no feasible actions exist (truck is stranded with no way forward)
"""

import torch
import numpy as np
from typing import Optional, Dict, Tuple, Set, List

from torch_geometric.data import HeteroData
from EVRoutingEnv.state.gnn_state_space import GNNStateSpace


class GNNStateSpaceSingleCharger(GNNStateSpace):
    """
    GNN State Space with single strategic charger selection.
    
    Inherits from GNNStateSpace but modifies edge construction to only
    include one strategically selected charger.
    
    Raises:
        ValueError: When no feasible actions exist for the active truck
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
        """Initialize single-charger GNN state space."""
        super().__init__(
            num_trucks=num_trucks,
            num_stops=num_stops,
            max_time=max_time,
            num_charging_nodes=num_charging_nodes,
            max_nodes_in_graph=max_nodes_in_graph,
            device=device,
            verbose=verbose,
        )
        # Override charger filtering flag
        self.FILTER_CHARGERS = True
        self.NUM_CHARGERS_TO_KEEP = 2  # Keep top-2 chargers by minimum detour
    
    def get_state_GNN(self, env) -> HeteroData:
        """
        Override parent to build action mask with single strategic charger.
        
        Builds the full graph structure using parent's logic, then completely
        rebuilds the action mask from scratch with single-charger restriction.
        
        Uses backwards planning from the last delivery to ensure global feasibility.
        """
        # Build graph structure using parent (nodes, edges, features)
        # This will also build the action_to_node_map which we need
        data = super().get_state_GNN(env)
        
        # Get active truck
        if env.active_truck_id is None:
            return data
        
        active_truck = env.trucks[env.active_truck_id]
        current_location = active_truck.current_node
        current_battery = active_truck.current_battery
        battery_capacity = active_truck.battery_capacity
        next_delivery = active_truck.get_next_delivery_target()
        remaining_deliveries = active_truck.get_remaining_deliveries()
        
        # Skip single-charger logic if flexible delivery mode
        if active_truck.enable_flexible_delivery_order:
            if self.verbose:
                print("[Single-Charger] Skipping - flexible delivery mode not supported")
            return data
        
        # Skip if no next delivery
        if next_delivery is None:
            return data
        
        # Build NEW action mask from scratch
        # Use action_to_node_map to understand the action space structure
        action_to_node_map = data.action_to_node_map
        new_feasible_mask = [False] * len(action_to_node_map)
        
        # Get energy safety factor
        energy_safety_factor = 1.0
        if hasattr(env, 'traffic_config') and env.traffic_config.get('enable_traffic', False):
            if env.traffic_config.get('enable_energy_uncertainty', False):
                energy_safety_factor = env.traffic_config.get('max_energy_multiplier', 1.0)
        
        # Determine truck state
        at_charger = current_location in env.charging_nodes
        must_leave = active_truck.must_leave_charger
        must_charge_now = at_charger and not must_leave
        
        # Compute globally feasible plan using backwards planning
        strategic_chargers = []  # List of (charger_id, min_charge_needed) tuples
        strategic_charger_dict = {}  # Map charger_id -> min_charge_needed
        
        try:
            strategic_chargers, strategic_charger_dict = self.plan_globally_feasible_route(
                env=env,
                truck=active_truck,
                current_location=current_location,
                current_battery=current_battery,
                remaining_deliveries=remaining_deliveries,
                energy_safety_factor=energy_safety_factor,
            )
            if self.verbose:
                if strategic_chargers:
                    charger_info = ", ".join([f"{c_id} ({strategic_charger_dict[c_id]:.1f} kWh)" for c_id in strategic_chargers])
                    print(f"[Single-Charger] Strategic chargers (top-2): {charger_info}")
                else:
                    print(f"[Single-Charger] No charger needed - can complete route directly")
        except ValueError as e:
            # No feasible plan exists - truck is stranded
            if self.verbose:
                print(f"[Single-Charger] No globally feasible plan exists")
                print(f"  Error: {e}")
            strategic_chargers = []
            strategic_charger_dict = {}
        
        # If at a charger that's NOT one of the strategic ones, recompute min charge for current location
        min_charge_at_current = None
        route_infeasible_from_here = False
        if at_charger and not must_leave and current_location not in strategic_chargers:
            # Recompute minimum charge needed at THIS charger
            min_charge_at_current = self._calculate_min_charge_to_complete_route(
                env=env,
                charger_location=current_location,
                next_delivery=next_delivery,
                battery_at_charger=current_battery,
                remaining_deliveries=remaining_deliveries,
                battery_capacity=active_truck.battery_capacity,
                energy_safety_factor=energy_safety_factor,
            )
            if self.verbose:
                if min_charge_at_current is not None:
                    print(f"[Single-Charger] At non-strategic charger {current_location}, "
                          f"recomputed min charge: {min_charge_at_current:.1f} kWh")
                else:
                    print(f"[Single-Charger] WARNING: At charger {current_location} but route is INFEASIBLE from here!")
                    print(f"  This charger cannot enable completing remaining deliveries: {remaining_deliveries}")
                    route_infeasible_from_here = True
        
        # Process each action
        for action_idx, (node_id, is_charging_action) in enumerate(action_to_node_map):
            if is_charging_action:
                # Charging action - only feasible if at charger and not must_leave
                if not at_charger:
                    # Not at charger - can't charge
                    new_feasible_mask[action_idx] = False
                elif must_leave:
                    # Just charged, must leave now - no more charging
                    new_feasible_mask[action_idx] = False
                elif current_location not in strategic_chargers and min_charge_at_current is None:
                    # At non-strategic charger but can't complete route even with full charge
                    new_feasible_mask[action_idx] = False
                elif current_location in strategic_chargers and current_location not in strategic_charger_dict:
                    # At strategic charger but no charge requirement (shouldn't happen)
                    new_feasible_mask[action_idx] = False
                else:
                    # At charger and haven't charged yet - check if this is a strategic duration
                    charge_duration = data.action_charge_durations[action_idx].item()
                    
                    # Compute strategic durations for this charger
                    if current_location in strategic_chargers:
                        strategic_durations = self.compute_strategic_charging_durations(
                            env=env,
                            truck=active_truck,
                            charger_location=current_location,
                            current_battery=current_battery,
                            remaining_deliveries=remaining_deliveries,
                            energy_safety_factor=energy_safety_factor,
                        )
                    else:
                        # Non-strategic charger - use traditional min charge validation
                        strategic_durations = None
                    
                    if strategic_durations is not None:
                        # Check if charge_duration is in strategic durations
                        is_feasible = charge_duration in strategic_durations
                    else:
                        # Fall back to min charge validation
                        min_charge_to_check = min_charge_at_current if current_location not in strategic_chargers else strategic_charger_dict.get(current_location, 0.0)
                        is_feasible = self._validate_charging_duration_global(
                            env=env,
                            truck=active_truck,
                            current_location=current_location,
                            current_battery=current_battery,
                            charge_duration=charge_duration,
                            min_charge_needed=min_charge_to_check,
                            energy_safety_factor=energy_safety_factor,
                        )
                    new_feasible_mask[action_idx] = is_feasible
            elif node_id == -1:
                # Invalid action
                new_feasible_mask[action_idx] = False
            elif node_id in env.charging_nodes:
                # Routing to charger action
                if node_id == current_location:
                    # Already at this charger - can't route to current location
                    new_feasible_mask[action_idx] = False
                elif route_infeasible_from_here:
                    # Current charger cannot complete route - allow routing to ANY reachable charger
                    # as last resort (override must_charge_now to enable escape)
                    # This is better than being stranded with no actions
                    energy = env.transport_graph.get_path_energy(current_location, node_id)
                    max_energy_needed = energy * energy_safety_factor
                    is_reachable = max_energy_needed < current_battery and not np.isinf(energy)
                    new_feasible_mask[action_idx] = is_reachable
                elif must_charge_now:
                    # Must charge, can't route anywhere
                    new_feasible_mask[action_idx] = False
                elif must_leave:
                    # Just charged and must leave - allow routing to reachable chargers
                    # that enable reaching at least the next delivery
                    energy = env.transport_graph.get_path_energy(current_location, node_id)
                    max_energy_needed = energy * energy_safety_factor
                    is_reachable = max_energy_needed < current_battery and not np.isinf(energy)
                    
                    if is_reachable:
                        # Validate that from this charger, we can reach the next delivery with full battery
                        energy_to_next_del = env.transport_graph.get_path_energy(node_id, next_delivery)
                        max_energy_to_next_del = energy_to_next_del * energy_safety_factor
                        can_help = battery_capacity >= max_energy_to_next_del and not np.isinf(energy_to_next_del)
                        new_feasible_mask[action_idx] = can_help
                    else:
                        new_feasible_mask[action_idx] = False
                elif node_id in strategic_chargers:
                    # This is one of the strategic chargers - check if reachable
                    energy = env.transport_graph.get_path_energy(current_location, node_id)
                    max_energy_needed = energy * energy_safety_factor
                    is_reachable = max_energy_needed < current_battery and not np.isinf(energy)
                    new_feasible_mask[action_idx] = is_reachable
                else:
                    # Not a strategic charger and not must_leave - infeasible
                    new_feasible_mask[action_idx] = False
            else:
                # Routing to delivery action
                if must_charge_now:
                    # At charger but haven't charged yet - must charge first (mandatory charging)
                    new_feasible_mask[action_idx] = False
                else:
                    # Use parent's logic for delivery feasibility
                    # The strategic charger selection already ensures the route is viable
                    new_feasible_mask[action_idx] = data.feasible_action_mask[action_idx].item()
        
        # Update the mask
        data.feasible_action_mask = torch.tensor(new_feasible_mask, dtype=torch.bool, device=self.device)
        
        if self.verbose:
            num_feasible = sum(new_feasible_mask)
            num_charger_actions = len([1 for nid, is_chg in action_to_node_map 
                                      if not is_chg and nid in env.charging_nodes])
            feasible_chargers = [nid for idx, (nid, is_chg) in enumerate(action_to_node_map) 
                                if not is_chg and nid in env.charging_nodes and new_feasible_mask[idx]]
            print(f"[Single-Charger] Rebuilt mask: {num_feasible} feasible actions total")
            print(f"[Single-Charger] Charger actions: {len(feasible_chargers)} feasible of {num_charger_actions} total")
            print(f"[Single-Charger] Feasible chargers: {feasible_chargers}")
        
        return data
    
    def plan_globally_feasible_route(
        self,
        env,
        truck,
        current_location: int,
        current_battery: float,
        remaining_deliveries: List[int],
        energy_safety_factor: float,
    ) -> Tuple[List[int], Dict[int, float]]:
        """
        Plan a globally feasible route using backwards planning from the last delivery.
        
        Algorithm:
        1. Start from the last delivery and work backwards
        2. For each delivery segment, determine if charging is needed
        3. If charging is needed, find the top-2 chargers by minimum detour that:
           - Are reachable from current position
           - Enable reaching the delivery
           - Provide enough energy to complete remaining route
        4. Calculate minimum charge needed at each charger
        
        Args:
            env: Environment instance
            truck: Active truck
            current_location: Current truck location
            current_battery: Current battery level
            remaining_deliveries: List of remaining deliveries in order
            energy_safety_factor: Safety factor for energy calculations
            
        Returns:
            (strategic_chargers, charger_to_min_charge) tuple:
            - strategic_chargers: List of top-2 charger IDs (empty if can go direct)
            - charger_to_min_charge: Dict mapping charger_id -> min_charge_needed
            
        Raises:
            ValueError: If no globally feasible route exists
        """
        if not remaining_deliveries:
            return None, None
        
        battery_capacity = truck.battery_capacity
        next_delivery = remaining_deliveries[0]
        
        # Calculate energy needed for the full route from next delivery onwards
        # Work backwards to determine charging requirements
        
        # Case 1: Can we reach next delivery directly?
        energy_to_next_delivery = env.transport_graph.get_path_energy(current_location, next_delivery)
        if np.isinf(energy_to_next_delivery):
            raise ValueError(f"Cannot reach next delivery {next_delivery} from {current_location}")
        
        max_energy_to_next = energy_to_next_delivery * energy_safety_factor
        
        # Check if we can complete the route directly (no charging)
        if self._can_complete_route_from(
            env, next_delivery, current_battery - max_energy_to_next,
            remaining_deliveries[1:], battery_capacity, energy_safety_factor
        ):
            # Can complete route without charging
            if current_battery >= max_energy_to_next:
                return [], {}
        
        # Special case: already at a charger - include it as first candidate
        charger_candidates = []
        
        if current_location in env.charging_nodes:
            min_charge = self._calculate_min_charge_to_complete_route(
                env, current_location, next_delivery, current_battery,
                remaining_deliveries, battery_capacity, energy_safety_factor
            )
            if min_charge is not None:
                # Add current charger with zero detour (already here)
                charger_candidates.append({
                    'id': current_location,
                    'min_charge': min_charge,
                    'detour': 0.0,
                })
        
        # Find feasible chargers using multi-step lookahead
        # The charger must enable completing the entire remaining route
        
        # Look ahead 2-3 deliveries to ensure charger enables route completion
        lookahead_deliveries = remaining_deliveries[:min(3, len(remaining_deliveries))]
        
        for charger_id in env.charging_nodes:
            # Skip if already at this charger (already handled above)
            if charger_id == current_location:
                continue
            
            # Can we reach this charger?
            energy_to_charger = env.transport_graph.get_path_energy(current_location, charger_id)
            if np.isinf(energy_to_charger):
                continue
            
            max_energy_to_charger = energy_to_charger * energy_safety_factor
            if current_battery < max_energy_to_charger:
                continue  # Can't reach this charger
            
            # From charger, can we reach next delivery?
            energy_charger_to_delivery = env.transport_graph.get_path_energy(charger_id, next_delivery)
            if np.isinf(energy_charger_to_delivery):
                continue
            
            max_energy_charger_to_delivery = energy_charger_to_delivery * energy_safety_factor
            
            # Calculate minimum charge needed at this charger to complete the FULL route
            # (not just reach the next delivery)
            battery_at_charger = current_battery - max_energy_to_charger
            
            # Key change: verify charger enables completing ALL remaining deliveries
            min_charge = self._calculate_min_charge_to_complete_route(
                env, charger_id, next_delivery, battery_at_charger,
                remaining_deliveries, battery_capacity, energy_safety_factor
            )
            
            if min_charge is None:
                # Cannot complete route even with full charge at this charger
                if self.verbose:
                    print(f"    [Charger {charger_id}] Cannot complete route with lookahead deliveries: {lookahead_deliveries}")
                continue
            
            # Additional validation: simulate charging and verify we can actually progress
            # through at least the first 2-3 deliveries
            battery_after_charge = min(battery_capacity, battery_at_charger + min_charge)
            if not self._can_complete_route_from(
                env, charger_id, battery_after_charge,
                lookahead_deliveries, battery_capacity, energy_safety_factor
            ):
                if self.verbose:
                    print(f"    [Charger {charger_id}] Cannot complete lookahead deliveries: {lookahead_deliveries}")
                continue
            
            # Calculate detour for this charger
            detour = energy_to_charger + energy_charger_to_delivery - energy_to_next_delivery
            
            # Add to candidates
            charger_candidates.append({
                'id': charger_id,
                'min_charge': min_charge,
                'detour': detour,
            })
        
        if not charger_candidates:
            raise ValueError(
                f"No globally feasible charger found. "
                f"Location: {current_location}, Battery: {current_battery:.1f} kWh, "
                f"Next delivery: {next_delivery}, Remaining: {len(remaining_deliveries)}"
            )
        
        # Sort by detour (ascending) and select top-2
        charger_candidates.sort(key=lambda x: (x['detour'], x['min_charge']))
        top_chargers = charger_candidates[:self.NUM_CHARGERS_TO_KEEP]
        
        # Build return values
        strategic_chargers = [c['id'] for c in top_chargers]
        charger_to_min_charge = {c['id']: c['min_charge'] for c in top_chargers}
        
        return strategic_chargers, charger_to_min_charge
    
    def select_strategic_charger(
        self,
        env,
        truck,
        current_location: int,
        next_delivery: int,
        remaining_deliveries: List[int],
        charger_node_to_idx: Dict[int, int],
    ) -> Optional[int]:
        """
        Select the single best charger between current location and next delivery.
        
        Selection criteria (in priority order):
        1. Must be reachable with current battery
        2. Must enable reaching next delivery after charging
        3. Must enable completing all remaining deliveries
        4. Minimize total detour: (energy_to_charger + energy_charger_to_delivery)
        
        Args:
            env: Environment instance
            truck: Active truck
            current_location: Current truck location
            next_delivery: Next delivery node
            remaining_deliveries: List of all remaining deliveries
            charger_node_to_idx: Mapping of charger nodes to indices
            
        Returns:
            Selected charger node ID, or None if no charger needed/feasible
            
        Raises:
            ValueError: If charger is required but none are feasible
        """
        current_battery = truck.current_battery
        battery_capacity = truck.battery_capacity
        
        # Get energy safety factor
        energy_safety_factor = 1.0
        if hasattr(env, 'traffic_config') and env.traffic_config.get('enable_traffic', False) and env.traffic_config.get('enable_energy_uncertainty', False):
            energy_safety_factor = env.traffic_config.get('max_energy_multiplier', 1.0)
        
        # Calculate direct energy to next delivery
        energy_direct = env.transport_graph.get_path_energy(current_location, next_delivery)
        
        # Check if we can reach delivery directly and continue after
        can_reach_directly = energy_direct * energy_safety_factor < current_battery
        
        if can_reach_directly:
            # Can reach delivery, but can we continue after?
            battery_after_delivery = current_battery - (energy_direct * energy_safety_factor)
            
            # If this is the last delivery, no charger needed
            if len(remaining_deliveries) == 1:
                if self.verbose:
                    print(f"  [Strategic Charger] No charger needed - last delivery reachable directly")
                return None
            
            # Check if we can reach any charger from delivery
            can_reach_charger_from_delivery = False
            for charger_id in env.charging_nodes:
                energy_to_charger = env.transport_graph.get_path_energy(next_delivery, charger_id)
                if not np.isinf(energy_to_charger) and battery_after_delivery > (energy_to_charger * energy_safety_factor):
                    can_reach_charger_from_delivery = True
                    break
            
            if can_reach_charger_from_delivery:
                if self.verbose:
                    print(f"  [Strategic Charger] No charger needed - can reach delivery and continue")
                return None
        
        # Need to charge - find best charger
        candidate_chargers = []
        
        for charger_id in env.charging_nodes:
            # Skip if already at this charger
            if charger_id == current_location:
                # Special case: already at charger, can use it
                candidate_chargers.append({
                    'id': charger_id,
                    'energy_to_charger': 0.0,
                    'energy_to_delivery': env.transport_graph.get_path_energy(charger_id, next_delivery),
                    'total_detour': 0.0 - energy_direct,  # Negative detour (we're already here)
                })
                continue
            
            # Check if reachable with current battery
            energy_to_charger = env.transport_graph.get_path_energy(current_location, charger_id)
            if np.isinf(energy_to_charger) or energy_to_charger * energy_safety_factor >= current_battery:
                continue  # Not reachable
            
            # Check energy from charger to next delivery
            energy_charger_to_delivery = env.transport_graph.get_path_energy(charger_id, next_delivery)
            if np.isinf(energy_charger_to_delivery):
                continue  # Can't reach delivery from this charger
            
            # Calculate minimum charge needed to reach delivery and continue
            # Need: energy to reach delivery + energy to reach a charger from delivery
            min_energy_needed_from_charger = energy_charger_to_delivery
            
            # If not last delivery, also need energy to reach a charger from delivery
            if len(remaining_deliveries) > 1:
                min_energy_to_next_charger = float('inf')
                for next_charger_id in env.charging_nodes:
                    energy_del_to_charger = env.transport_graph.get_path_energy(next_delivery, next_charger_id)
                    if energy_del_to_charger < min_energy_to_next_charger:
                        min_energy_to_next_charger = energy_del_to_charger
                
                if not np.isinf(min_energy_to_next_charger):
                    min_energy_needed_from_charger += min_energy_to_next_charger
            
            # Apply safety factor
            min_energy_needed_from_charger *= energy_safety_factor
            
            # Check if we can charge enough at this station to proceed
            # Get charger configuration to estimate max possible charge
            charger_type = env.charging_station.charger_type.get(charger_id, "DCFast")
            charging_config = env.config["charging"]
            if charger_type == "DCFast":
                charger_config_type = charging_config["dcfast"]
            else:
                charger_config_type = charging_config["level2"]
            
            # Calculate battery at arrival at charger
            battery_at_charger = current_battery - (energy_to_charger * energy_safety_factor)
            
            # Check if we can charge enough (using longest charge duration)
            max_charge_hours = max(env.charging_config['charge_durations'])
            
            # Calculate charge amount using charging curve
            initial_soc = max(0.0, min(1.0, battery_at_charger / battery_capacity))
            charger_config_with_curve = charger_config_type.copy()
            charger_config_with_curve["use_realistic_curve"] = charging_config.get("use_realistic_curve", False)
            
            max_charge_amount, _ = env.charging_curve_model.calculate_charge(
                initial_soc=initial_soc,
                charge_hours=max_charge_hours,
                battery_capacity=battery_capacity,
                charger_config=charger_config_with_curve,
                charger_type=charger_type
            )
            
            battery_after_max_charge = min(battery_capacity, battery_at_charger + max_charge_amount)
            
            if battery_after_max_charge < min_energy_needed_from_charger:
                continue  # Can't charge enough at this station
            
            # Calculate total detour vs direct route
            total_distance_via_charger = energy_to_charger + energy_charger_to_delivery
            detour = total_distance_via_charger - energy_direct
            
            candidate_chargers.append({
                'id': charger_id,
                'energy_to_charger': energy_to_charger,
                'energy_to_delivery': energy_charger_to_delivery,
                'total_detour': detour,
                'battery_at_arrival': battery_at_charger,
                'min_energy_needed': min_energy_needed_from_charger,
            })
        
        if not candidate_chargers:
            # No feasible charger found - this is a critical error
            error_msg = (
                f"No feasible charger found for truck {truck.truck_id}!\n"
                f"  Current location: {current_location}\n"
                f"  Current battery: {current_battery:.1f} kWh\n"
                f"  Next delivery: {next_delivery}\n"
                f"  Energy to delivery: {energy_direct:.1f} kWh\n"
                f"  Can reach directly: {can_reach_directly}\n"
                f"  Remaining deliveries: {len(remaining_deliveries)}\n"
                f"  Available chargers: {len(env.charging_nodes)}"
            )
            raise ValueError(error_msg)
        
        # Select charger with minimum detour
        best_charger = min(candidate_chargers, key=lambda x: x['total_detour'])
        
        if self.verbose:
            print(f"  [Strategic Charger] Selected charger {best_charger['id']} "
                  f"(detour: {best_charger['total_detour']:.1f} kWh, "
                  f"candidates: {len(candidate_chargers)})")
        
        return best_charger['id']
    
    def _validate_charging_duration(
        self,
        env,
        truck,
        current_location: int,
        current_battery: float,
        charge_duration: float,
        next_delivery: int,
        remaining_deliveries: List[int],
        energy_safety_factor: float,
    ) -> bool:
        """
        Validate that charging for the given duration enables future progress.
        
        A charging duration is feasible if after charging:
        1. Truck can reach the next delivery
        2. Truck can continue after the delivery (reach next charger or complete trip)
        
        Args:
            env: Environment instance
            truck: Active truck
            current_location: Current truck location (must be a charger)
            current_battery: Current battery level
            charge_duration: Charging duration in hours
            next_delivery: Next delivery node
            remaining_deliveries: List of all remaining deliveries
            energy_safety_factor: Safety factor for energy calculations
            
        Returns:
            True if this charging duration enables future progress, False otherwise
        """
        battery_capacity = truck.battery_capacity
        
        # Get charger configuration
        charger_type = env.charging_station.charger_type.get(current_location, "DCFast")
        charging_config = env.config["charging"]
        if charger_type == "DCFast":
            charger_config_type = charging_config["dcfast"]
        else:
            charger_config_type = charging_config["level2"]
        
        # Calculate battery after charging
        initial_soc = max(0.0, min(1.0, current_battery / battery_capacity))
        charger_config_with_curve = charger_config_type.copy()
        charger_config_with_curve["use_realistic_curve"] = charging_config.get("use_realistic_curve", False)
        
        charge_amount, _ = env.charging_curve_model.calculate_charge(
            initial_soc=initial_soc,
            charge_hours=charge_duration,
            battery_capacity=battery_capacity,
            charger_config=charger_config_with_curve,
            charger_type=charger_type
        )
        
        battery_after_charging = min(battery_capacity, current_battery + charge_amount)
        
        # Check if can reach next delivery
        energy_to_delivery = env.transport_graph.get_path_energy(current_location, next_delivery)
        if np.isinf(energy_to_delivery):
            return False
        
        max_energy_to_delivery = energy_to_delivery * energy_safety_factor
        if battery_after_charging < max_energy_to_delivery:
            # Can't reach next delivery
            return False
        
        # Battery after reaching delivery
        battery_after_delivery = battery_after_charging - max_energy_to_delivery
        
        # Check if can continue after delivery
        if len(remaining_deliveries) == 1:
            # Last delivery - no need to continue
            return True
        
        # Check if can reach any charger from delivery (for next segment)
        can_reach_next_charger = False
        for charger_id in env.charging_nodes:
            energy_to_charger = env.transport_graph.get_path_energy(next_delivery, charger_id)
            if not np.isinf(energy_to_charger):
                max_energy_to_charger = energy_to_charger * energy_safety_factor
                if battery_after_delivery >= max_energy_to_charger:
                    can_reach_next_charger = True
                    break
        
        return can_reach_next_charger
    
    def _can_complete_route_from(
        self,
        env,
        start_location: int,
        battery_at_start: float,
        remaining_deliveries: List[int],
        battery_capacity: float,
        energy_safety_factor: float,
    ) -> bool:
        """
        Check if truck can complete remaining deliveries from given location and battery.
        
        Uses greedy approach: at each delivery, if can't reach next, try to reach nearest charger,
        charge to full, and continue.
        """
        if not remaining_deliveries:
            return True
        
        current_loc = start_location
        current_battery = battery_at_start
        
        if self.verbose:
            print(f"        [CanComplete] start_loc={start_location}, battery={battery_at_start:.1f}, "
                  f"remaining={remaining_deliveries}")
        
        for i, delivery in enumerate(remaining_deliveries):
            energy_to_delivery = env.transport_graph.get_path_energy(current_loc, delivery)
            if np.isinf(energy_to_delivery):
                if self.verbose:
                    print(f"        [CanComplete] ✗ No path from {current_loc} to delivery {delivery}")
                return False
            
            max_energy_needed = energy_to_delivery * energy_safety_factor
            
            if current_battery >= max_energy_needed:
                # Can reach this delivery
                current_battery -= max_energy_needed
                current_loc = delivery
                if self.verbose:
                    print(f"        [CanComplete] Step {i+1}: Direct to delivery {delivery} "
                          f"(energy={max_energy_needed:.1f}, battery_after={current_battery:.1f})")
            else:
                # Need to charge - check if we can reach any charger
                if self.verbose:
                    print(f"        [CanComplete] Step {i+1}: Need charger (battery={current_battery:.1f}, "
                          f"need={max_energy_needed:.1f} to reach {delivery})")
                can_charge = False
                for charger_id in env.charging_nodes:
                    energy_to_charger = env.transport_graph.get_path_energy(current_loc, charger_id)
                    if np.isinf(energy_to_charger):
                        continue
                    
                    max_energy_to_charger = energy_to_charger * energy_safety_factor
                    if current_battery < max_energy_to_charger:
                        continue
                    
                    # Can reach charger, charge to full, then check if can reach delivery
                    energy_charger_to_delivery = env.transport_graph.get_path_energy(charger_id, delivery)
                    if np.isinf(energy_charger_to_delivery):
                        continue
                    
                    max_energy_charger_to_delivery = energy_charger_to_delivery * energy_safety_factor
                    if battery_capacity >= max_energy_charger_to_delivery:
                        # Found a viable charger
                        current_battery = battery_capacity - max_energy_charger_to_delivery
                        current_loc = delivery
                        can_charge = True
                        if self.verbose:
                            print(f"          [CanComplete] ✓ Via charger {charger_id} to delivery {delivery} "
                                  f"(battery_after={current_battery:.1f})")
                        break
                
                if not can_charge:
                    if self.verbose:
                        print(f"        [CanComplete] ✗ Cannot reach delivery {delivery} from {current_loc}")
                    return False
        
        if self.verbose:
            print(f"        [CanComplete] ✓ Route completable")
        return True
    
    def _calculate_min_charge_to_complete_route(
        self,
        env,
        charger_location: int,
        next_delivery: int,
        battery_at_charger: float,
        remaining_deliveries: List[int],
        battery_capacity: float,
        energy_safety_factor: float,
    ) -> Optional[float]:
        """
        Calculate minimum charge needed at a charger to complete the remaining route.
        
        Uses binary search to find the minimum charge amount.
        """
        if not remaining_deliveries:
            return 0.0
        
        if self.verbose:
            print(f"    [MinCharge] charger={charger_location}, battery_at_charger={battery_at_charger:.1f}, "
                  f"remaining={len(remaining_deliveries)} deliveries")
        
        # Get charger configuration for realistic charging calculation
        charger_type = env.charging_station.charger_type.get(charger_location, "DCFast")
        charging_config = env.config["charging"]
        if charger_type == "DCFast":
            charger_config_type = charging_config["dcfast"]
        else:
            charger_config_type = charging_config["level2"]
        
        charger_config_with_curve = charger_config_type.copy()
        charger_config_with_curve["use_realistic_curve"] = charging_config.get("use_realistic_curve", False)
        
        # Try charge durations from shortest to longest
        charge_durations = sorted(env.charging_config['charge_durations'])
        if self.verbose:
            print(f"    [MinCharge] Trying {len(charge_durations)} charge durations: {charge_durations}")
        
        for charge_hours in charge_durations:
            initial_soc = max(0.0, min(1.0, battery_at_charger / battery_capacity))
            
            charge_amount, _ = env.charging_curve_model.calculate_charge(
                initial_soc=initial_soc,
                charge_hours=charge_hours,
                battery_capacity=battery_capacity,
                charger_config=charger_config_with_curve,
                charger_type=charger_type
            )
            
            battery_after_charging = min(battery_capacity, battery_at_charger + charge_amount)
            
            if self.verbose:
                print(f"      [MinCharge] charge_hours={charge_hours:.1f}, charge_amount={charge_amount:.1f}, "
                      f"battery_after={battery_after_charging:.1f}")
            
            # Check if this charge amount enables completing the route
            if self._can_complete_route_from(
                env, charger_location, battery_after_charging,
                remaining_deliveries, battery_capacity, energy_safety_factor
            ):
                if self.verbose:
                    print(f"      [MinCharge] ✓ Found feasible charge: {charge_amount:.1f} kWh")
                return charge_amount
            elif self.verbose:
                print(f"      [MinCharge] ✗ Cannot complete route with this charge")
        
        # Even full charge doesn't help
        if self.verbose:
            print(f"    [MinCharge] ✗ No feasible charge duration found")
        return None
    
    def _validate_charging_duration_global(
        self,
        env,
        truck,
        current_location: int,
        current_battery: float,
        charge_duration: float,
        min_charge_needed: float,
        energy_safety_factor: float,
    ) -> bool:
        """
        Validate that charging for the given duration provides at least the minimum charge needed.
        """
        battery_capacity = truck.battery_capacity
        
        # Get charger configuration
        charger_type = env.charging_station.charger_type.get(current_location, "DCFast")
        charging_config = env.config["charging"]
        if charger_type == "DCFast":
            charger_config_type = charging_config["dcfast"]
        else:
            charger_config_type = charging_config["level2"]
        
        # Calculate battery after charging
        initial_soc = max(0.0, min(1.0, current_battery / battery_capacity))
        charger_config_with_curve = charger_config_type.copy()
        charger_config_with_curve["use_realistic_curve"] = charging_config.get("use_realistic_curve", False)
        
        charge_amount, _ = env.charging_curve_model.calculate_charge(
            initial_soc=initial_soc,
            charge_hours=charge_duration,
            battery_capacity=battery_capacity,
            charger_config=charger_config_with_curve,
            charger_type=charger_type
        )
        
        # Check if charge amount meets minimum requirement
        return charge_amount >= min_charge_needed
    
    def compute_strategic_charging_durations(
        self,
        env,
        truck,
        charger_location: int,
        current_battery: float,
        remaining_deliveries: List[int],
        energy_safety_factor: float,
    ) -> Optional[Set[float]]:
        """
        Compute strategic charging durations that enable completing key route segments.
        
        Returns a set of 3 strategic durations:
        1. h_min^{d1+c}: Minimum to reach d1 and a charger from d1
        2. h_min^{d1+d2+complete}: Minimum to reach d1, d2, and continue/complete
        3. h_max: Maximum charging duration (12 hours typically)
        
        Args:
            env: Environment instance
            truck: Active truck
            charger_location: Current charger location
            current_battery: Current battery level
            remaining_deliveries: List of remaining deliveries
            energy_safety_factor: Safety factor for energy calculations
            
        Returns:
            Set of strategic charging durations (hours), or None if no feasible durations
        """
        if not remaining_deliveries:
            return None
        
        battery_capacity = truck.battery_capacity
        d1 = remaining_deliveries[0]
        charge_durations = sorted(env.charging_config['charge_durations'])
        h_max = max(charge_durations)
        
        # Get charger configuration
        charger_type = env.charging_station.charger_type.get(charger_location, "DCFast")
        charging_config = env.config["charging"]
        if charger_type == "DCFast":
            charger_config_type = charging_config["dcfast"]
        else:
            charger_config_type = charging_config["level2"]
        
        charger_config_with_curve = charger_config_type.copy()
        charger_config_with_curve["use_realistic_curve"] = charging_config.get("use_realistic_curve", False)
        initial_soc = max(0.0, min(1.0, current_battery / battery_capacity))
        
        # Calculate energy requirements
        energy_to_d1 = env.transport_graph.get_path_energy(charger_location, d1)
        if np.isinf(energy_to_d1):
            return None
        
        # 1. Find h_min^{d1+c}: reach d1 and a charger from d1
        h_min_d1_c = None
        min_energy_d1_c = energy_to_d1 * energy_safety_factor
        
        # Find minimum energy to reach a charger from d1
        min_energy_d1_to_charger = float('inf')
        for c_id in env.charging_nodes:
            energy_d1_to_c = env.transport_graph.get_path_energy(d1, c_id)
            if not np.isinf(energy_d1_to_c):
                min_energy_d1_to_charger = min(min_energy_d1_to_charger, energy_d1_to_c)
        
        if not np.isinf(min_energy_d1_to_charger):
            min_energy_d1_c += min_energy_d1_to_charger * energy_safety_factor
        
        # Find minimum charge duration that provides this energy
        for h in charge_durations:
            charge_amount, _ = env.charging_curve_model.calculate_charge(
                initial_soc=initial_soc,
                charge_hours=h,
                battery_capacity=battery_capacity,
                charger_config=charger_config_with_curve,
                charger_type=charger_type
            )
            if charge_amount >= min_energy_d1_c - current_battery:
                h_min_d1_c = h
                break
        
        # 2. Find h_min^{d1+d2+complete}: reach d1, d2, and continue/complete
        h_min_d1_d2_complete = None
        
        if len(remaining_deliveries) >= 2:
            d2 = remaining_deliveries[1]
            energy_d1_to_d2 = env.transport_graph.get_path_energy(d1, d2)
            
            if not np.isinf(energy_d1_to_d2):
                # Calculate energy requirement
                min_energy_d1_d2 = (energy_to_d1 + energy_d1_to_d2) * energy_safety_factor
                
                # If more deliveries remain, need to reach a charger from d2
                if len(remaining_deliveries) > 2:
                    min_energy_d2_to_charger = float('inf')
                    for c_id in env.charging_nodes:
                        energy_d2_to_c = env.transport_graph.get_path_energy(d2, c_id)
                        if not np.isinf(energy_d2_to_c):
                            min_energy_d2_to_charger = min(min_energy_d2_to_charger, energy_d2_to_c)
                    
                    if not np.isinf(min_energy_d2_to_charger):
                        min_energy_d1_d2 += min_energy_d2_to_charger * energy_safety_factor
                
                # Find minimum charge duration
                for h in charge_durations:
                    charge_amount, _ = env.charging_curve_model.calculate_charge(
                        initial_soc=initial_soc,
                        charge_hours=h,
                        battery_capacity=battery_capacity,
                        charger_config=charger_config_with_curve,
                        charger_type=charger_type
                    )
                    if charge_amount >= min_energy_d1_d2 - current_battery:
                        h_min_d1_d2_complete = h
                        break
        elif len(remaining_deliveries) == 1:
            # Only d1 remains - same as h_min_d1_c but checking if it's the last
            h_min_d1_d2_complete = h_min_d1_c
        
        # Build strategic duration set
        strategic_durations = set()
        
        if h_min_d1_c is not None:
            strategic_durations.add(h_min_d1_c)
        
        if h_min_d1_d2_complete is not None:
            strategic_durations.add(h_min_d1_d2_complete)
        
        strategic_durations.add(h_max)
        
        if self.verbose:
            print(f"    [StrategicDurations] charger={charger_location}, d1={d1}, "
                  f"durations={sorted(strategic_durations)}")
        
        return strategic_durations if strategic_durations else None
    
    def _build_edges(
        self,
        env,
        truck_id_to_idx: Dict[int, Optional[int]],
        delivery_node_to_idx: Dict[int, int],
        charger_node_to_idx: Dict[int, int],
    ) -> Dict[Tuple[str, str, str], Dict[str, List]]:
        """
        Build edges with single strategic charger selection.
        
        Overrides parent method to only include one charger per truck.
        
        Raises:
            ValueError: If no feasible actions exist for the active truck
        """
        # Get energy safety factor
        energy_safety_factor = 1.0
        if hasattr(env, 'traffic_config') and env.traffic_config.get('enable_traffic', False) and env.traffic_config.get('enable_energy_uncertainty', False):
            energy_safety_factor = env.traffic_config.get('max_energy_multiplier', 1.0)
        
        # Initialize edge dictionary
        edge_dict = {
            ('truck', 'to', 'delivery'): {'edge_index': [], 'edge_attr': []},
            ('delivery', 'to', 'truck'): {'edge_index': [], 'edge_attr': []},
            ('truck', 'to', 'charger'): {'edge_index': [], 'edge_attr': []},
            ('charger', 'to', 'truck'): {'edge_index': [], 'edge_attr': []},
            ('charger', 'to', 'delivery'): {'edge_index': [], 'edge_attr': []},
            ('delivery', 'to', 'charger'): {'edge_index': [], 'edge_attr': []},
            ('charger', 'to', 'charger'): {'edge_index': [], 'edge_attr': []},
            ('delivery', 'to', 'delivery'): {'edge_index': [], 'edge_attr': []},
        }
        
        # Build truck-specific edges
        for truck in env.trucks:
            if truck.failed or truck.is_complete:
                continue
            
            truck_idx = truck_id_to_idx[truck.truck_id]
            if truck_idx is None:
                continue
            
            current_location = truck.current_node
            current_battery = truck.current_battery
            truck_state = env.truck_states.get(truck.truck_id, "unknown")
            
            # Get next delivery
            next_delivery = truck.get_next_delivery_target()
            if truck.enable_flexible_delivery_order and isinstance(next_delivery, list):
                remaining_deliveries = next_delivery
                # Select closest as "next"
                if remaining_deliveries:
                    min_energy = float('inf')
                    next_delivery_node = None
                    for del_node in remaining_deliveries:
                        energy = env.transport_graph.get_path_energy(current_location, del_node)
                        if energy < min_energy:
                            min_energy = energy
                            next_delivery_node = del_node
                    next_delivery = next_delivery_node
                else:
                    next_delivery = None
            else:
                remaining_deliveries = truck.get_remaining_deliveries()
            
            # Track if any feasible action exists
            has_feasible_action = False
            
            # READY state: connect to next delivery and strategic charger
            if truck_state == "ready":
                # Connect to next delivery
                if next_delivery is not None and next_delivery in delivery_node_to_idx:
                    dest_idx = delivery_node_to_idx[next_delivery]
                    energy = env.transport_graph.get_path_energy(current_location, next_delivery)
                    time = env.transport_graph.get_time_distance(current_location, next_delivery)
                    
                    # Check feasibility
                    max_energy_needed = energy * energy_safety_factor
                    if max_energy_needed < current_battery and not np.isinf(energy):
                        edge_dict[('truck', 'to', 'delivery')]['edge_index'].append([truck_idx, dest_idx])
                        edge_dict[('truck', 'to', 'delivery')]['edge_attr'].append([energy/1000.0, time/self.max_time])
                        has_feasible_action = True
                        
                        if self.BIDIRECTIONAL_EDGES:
                            energy_inv = env.transport_graph.get_path_energy(next_delivery, current_location)
                            time_inv = env.transport_graph.get_time_distance(next_delivery, current_location)
                            edge_dict[('delivery', 'to', 'truck')]['edge_index'].append([dest_idx, truck_idx])
                            edge_dict[('delivery', 'to', 'truck')]['edge_attr'].append([energy_inv/1000.0, time_inv/self.max_time])
                
                # Select strategic charger
                if next_delivery is not None:
                    try:
                        strategic_charger = self.select_strategic_charger(
                            env=env,
                            truck=truck,
                            current_location=current_location,
                            next_delivery=next_delivery,
                            remaining_deliveries=remaining_deliveries,
                            charger_node_to_idx=charger_node_to_idx,
                        )
                    except ValueError as e:
                        # Re-raise if truck is active and has no feasible actions
                        if not has_feasible_action:
                            raise
                        # Otherwise, log warning and continue
                        if self.verbose:
                            print(f"  [WARNING] {str(e)}")
                        strategic_charger = None
                    
                    if strategic_charger is not None and strategic_charger in charger_node_to_idx:
                        charger_idx = charger_node_to_idx[strategic_charger]
                        
                        if strategic_charger == current_location:
                            # Already at charger
                            edge_dict[('truck', 'to', 'charger')]['edge_index'].append([truck_idx, charger_idx])
                            edge_dict[('truck', 'to', 'charger')]['edge_attr'].append([0.0, 0.0])
                            has_feasible_action = True
                            
                            if self.BIDIRECTIONAL_EDGES:
                                edge_dict[('charger', 'to', 'truck')]['edge_index'].append([charger_idx, truck_idx])
                                edge_dict[('charger', 'to', 'truck')]['edge_attr'].append([0.0, 0.0])
                        else:
                            # Connect to strategic charger
                            energy = env.transport_graph.get_path_energy(current_location, strategic_charger)
                            time = env.transport_graph.get_time_distance(current_location, strategic_charger)
                            
                            # Verify reachability
                            max_energy_needed = energy * energy_safety_factor
                            if max_energy_needed < current_battery and not np.isinf(energy):
                                edge_dict[('truck', 'to', 'charger')]['edge_index'].append([truck_idx, charger_idx])
                                edge_dict[('truck', 'to', 'charger')]['edge_attr'].append([energy/1000.0, time/self.max_time])
                                has_feasible_action = True
                                
                                if self.BIDIRECTIONAL_EDGES:
                                    energy_inv = env.transport_graph.get_path_energy(strategic_charger, current_location)
                                    time_inv = env.transport_graph.get_time_distance(strategic_charger, current_location)
                                    edge_dict[('charger', 'to', 'truck')]['edge_index'].append([charger_idx, truck_idx])
                                    edge_dict[('charger', 'to', 'truck')]['edge_attr'].append([energy_inv/1000.0, time_inv/self.max_time])
                
                # Check if truck has any feasible action
                if not has_feasible_action and truck_state == "ready":
                    error_msg = (
                        f"No feasible actions for truck {truck.truck_id} in READY state!\n"
                        f"  Current location: {current_location}\n"
                        f"  Current battery: {current_battery:.1f} kWh\n"
                        f"  Battery capacity: {truck.battery_capacity:.1f} kWh\n"
                        f"  Next delivery: {next_delivery}\n"
                        f"  Remaining deliveries: {len(remaining_deliveries)}\n"
                        f"  Truck is stranded - cannot reach any destination."
                    )
                    raise ValueError(error_msg)
            
            elif truck_state in ["waiting_to_charge", "charging"]:
                # Connect only to current charger
                if current_location in charger_node_to_idx:
                    charger_idx = charger_node_to_idx[current_location]
                    edge_dict[('truck', 'to', 'charger')]['edge_index'].append([truck_idx, charger_idx])
                    edge_dict[('truck', 'to', 'charger')]['edge_attr'].append([0.0, 0.0])
                    
                    if self.BIDIRECTIONAL_EDGES:
                        edge_dict[('charger', 'to', 'truck')]['edge_index'].append([charger_idx, truck_idx])
                        edge_dict[('charger', 'to', 'truck')]['edge_attr'].append([0.0, 0.0])
            
            elif truck_state == "routing":
                # Connect to destination
                if truck.route_destination is not None:
                    destination = truck.route_destination
                    time_remaining = max(0.0, truck.route_arrival_time - env.global_clock)
                    time_remaining_norm = time_remaining / self.max_time
                    
                    if destination in delivery_node_to_idx:
                        dest_idx = delivery_node_to_idx[destination]
                        edge_dict[('truck', 'to', 'delivery')]['edge_index'].append([truck_idx, dest_idx])
                        edge_dict[('truck', 'to', 'delivery')]['edge_attr'].append([0.0, time_remaining_norm])
                        
                        if self.BIDIRECTIONAL_EDGES:
                            edge_dict[('delivery', 'to', 'truck')]['edge_index'].append([dest_idx, truck_idx])
                            edge_dict[('delivery', 'to', 'truck')]['edge_attr'].append([0.0, time_remaining_norm])
                    
                    elif destination in charger_node_to_idx:
                        dest_idx = charger_node_to_idx[destination]
                        edge_dict[('truck', 'to', 'charger')]['edge_index'].append([truck_idx, dest_idx])
                        edge_dict[('truck', 'to', 'charger')]['edge_attr'].append([0.0, time_remaining_norm])
                        
                        if self.BIDIRECTIONAL_EDGES:
                            edge_dict[('charger', 'to', 'truck')]['edge_index'].append([dest_idx, truck_idx])
                            edge_dict[('charger', 'to', 'truck')]['edge_attr'].append([0.0, time_remaining_norm])
        
        # Build charger-to-delivery and charger-to-charger edges (structural)
        for charger_id, charger_idx in charger_node_to_idx.items():
            # Charger to deliveries
            for delivery_id, delivery_idx in delivery_node_to_idx.items():
                energy = env.transport_graph.get_path_energy(charger_id, delivery_id)
                time = env.transport_graph.get_time_distance(charger_id, delivery_id)
                
                if not np.isinf(energy):
                    edge_dict[('charger', 'to', 'delivery')]['edge_index'].append([charger_idx, delivery_idx])
                    edge_dict[('charger', 'to', 'delivery')]['edge_attr'].append([energy/1000.0, time/self.max_time])
                    
                    if self.BIDIRECTIONAL_EDGES:
                        energy_inv = env.transport_graph.get_path_energy(delivery_id, charger_id)
                        time_inv = env.transport_graph.get_time_distance(delivery_id, charger_id)
                        edge_dict[('delivery', 'to', 'charger')]['edge_index'].append([delivery_idx, charger_idx])
                        edge_dict[('delivery', 'to', 'charger')]['edge_attr'].append([energy_inv/1000.0, time_inv/self.max_time])
            
            # Charger to other chargers
            for other_charger_id, other_charger_idx in charger_node_to_idx.items():
                if charger_id == other_charger_id:
                    continue
                
                energy = env.transport_graph.get_path_energy(charger_id, other_charger_id)
                time = env.transport_graph.get_time_distance(charger_id, other_charger_id)
                
                if not np.isinf(energy):
                    edge_dict[('charger', 'to', 'charger')]['edge_index'].append([charger_idx, other_charger_idx])
                    edge_dict[('charger', 'to', 'charger')]['edge_attr'].append([energy/1000.0, time/self.max_time])
        
        # Build delivery-to-delivery edges (structural)
        for delivery_id_1, delivery_idx_1 in delivery_node_to_idx.items():
            for delivery_id_2, delivery_idx_2 in delivery_node_to_idx.items():
                if delivery_id_1 == delivery_id_2:
                    continue
                
                energy = env.transport_graph.get_path_energy(delivery_id_1, delivery_id_2)
                time = env.transport_graph.get_time_distance(delivery_id_1, delivery_id_2)
                
                if not np.isinf(energy):
                    edge_dict[('delivery', 'to', 'delivery')]['edge_index'].append([delivery_idx_1, delivery_idx_2])
                    edge_dict[('delivery', 'to', 'delivery')]['edge_attr'].append([energy/1000.0, time/self.max_time])
        
        return edge_dict
