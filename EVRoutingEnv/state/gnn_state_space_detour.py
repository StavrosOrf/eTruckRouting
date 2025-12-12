"""
GNN State Representation with Top-2 Detour-Based Charger Selection (sequential order only).

This module provides a simplified GNN action space for **non-flexible** delivery ordering
where only the TOP-2 charging stations by minimum detour are considered between the
current position and the next delivery. The charger selection is purely based on detour
minimization without global feasibility checks. Use ``GNNStateSpaceVRP`` for flexible
delivery ordering.

Key Differences from Other GNN State Spaces:
- Only top-2 chargers by minimum detour are available for routing
- When at a charger, truck MUST charge (cannot route away until charging completes)
- Charging durations must provide energy to EITHER:
  1. Reach next delivery + reach ANY charger from delivery, OR
  2. Reach next delivery + complete all remaining deliveries
- Action space: [top_2_chargers, next_delivery, charge_1h, ..., charge_Nh]

Detour Calculation:
- detour = energy_to_charger + energy_charger_to_delivery - energy_direct_to_delivery
- Lower detour = less energy wasted on charging detour
- Chargers sorted by ascending detour, top-2 selected

Charging Enforcement at Chargers:
- If at_charger and not must_leave_charger: ONLY charging actions allowed
- If at_charger and must_leave_charger: ONLY routing actions allowed
- Charging durations validated to ensure progress toward completion

Raises:
    ValueError: If no feasible actions exist (truck is stranded with no way forward)
"""

import torch
import numpy as np
from typing import Optional, Dict, Tuple, Set, List

from torch_geometric.data import HeteroData
from EVRoutingEnv.state.gnn_state_space import GNNStateSpace
from EVRoutingEnv.utils.utils import check_navigation_feasibility


class GNNStateSpaceDetourBased(GNNStateSpace):
    """
    GNN State Space with top-2 detour-based charger selection.
    
    Inherits from GNNStateSpace but modifies action masking to only
    include the two chargers with minimum detour to next delivery.
    
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
        route_delivery_after_charge_only: bool = True,
    ):
        """Initialize detour-based GNN state space."""
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
        # Optional routing restriction: after charging, try delivery first
        self.route_delivery_after_charge_only = route_delivery_after_charge_only
    
    def get_state_GNN(self, env) -> HeteroData:
        """
        Override parent to build action mask with top-2 detour-based charger selection.
        
        Builds the full graph structure using parent's logic, then completely
        rebuilds the action mask from scratch with detour-based restriction.
        
        Enforces mandatory charging at chargers:
        - If at charger and not must_leave: only charging actions enabled
        - If at charger and must_leave: only routing actions enabled
        """
        # Build graph structure using parent (nodes, edges, features)
        # This will also build the action_to_node_map which we need
        data = super().get_state_GNN(env)
        
        # Get active truck
        if env.active_truck_id is None:
            return data
        
        active_truck = env.trucks[env.active_truck_id]
        if getattr(env, "enable_flexible_delivery_order", False) or active_truck.enable_flexible_delivery_order:
            raise ValueError(
                "GNNStateSpaceDetourBased is only for sequential delivery order. "
                "Use GNNStateSpaceVRP when flexible ordering is enabled."
            )
        current_location = active_truck.current_node
        current_battery = active_truck.current_battery
        battery_capacity = active_truck.battery_capacity
        next_delivery = active_truck.get_next_delivery_target()
        remaining_deliveries = active_truck.get_remaining_deliveries()
        
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

        # Flag: after charging, prioritize routing directly to delivery; allow chargers only if needed
        route_delivery_first = bool(
            self.route_delivery_after_charge_only
            or env.config.get('gnn', {}).get('route_delivery_after_charge_only', False)
            or env.config.get('environment', {}).get('route_delivery_after_charge_only', False)
        )
        
        # Determine truck state
        at_charger = current_location in env.charging_nodes
        must_leave = active_truck.must_leave_charger
        must_charge_now = at_charger and not must_leave

        # Pre-compute whether routing directly to next delivery is feasible (used for route_delivery_first)
        delivery_feasible = False
        if next_delivery is not None and not must_charge_now:
            energy_to_delivery = env.transport_graph.get_path_energy(current_location, next_delivery)
            if not np.isinf(energy_to_delivery):
                max_energy = energy_to_delivery * energy_safety_factor
                if current_battery >= max_energy:
                    delivery_feasible = check_navigation_feasibility(
                        truck=active_truck,
                        target_node=next_delivery,
                        discharge=max_energy,
                        transport_graph=env.transport_graph,
                        charging_nodes=env.charging_nodes,
                        energy_safety_factor=energy_safety_factor,
                        verbose=self.verbose,
                    )
        
        # Select top-2 chargers by detour
        top_chargers_by_detour = []
        no_feasible_chargers = False
        try:
            top_chargers_by_detour = self.select_top_2_chargers_by_detour(
                env=env,
                current_location=current_location,
                current_battery=current_battery,
                next_delivery=next_delivery,
                battery_capacity=battery_capacity,
                energy_safety_factor=energy_safety_factor,
            )
            if self.verbose:
                print(f"[Detour] Top-2 chargers by detour: {[(c['id'], c['detour']) for c in top_chargers_by_detour]}")
        except ValueError as e:
            # No feasible chargers found - fall back to escape routing (any reachable charger)
            no_feasible_chargers = True
            if self.verbose:
                print(f"[Detour] No feasible chargers found: {e}. Enabling escape routing to any reachable charger.")
            top_chargers_by_detour = []
        
        # Pre-validate all charging durations to determine feasibility and backup strategy
        valid_charging_durations = set()
        allow_escape_routing = False
        use_max_duration_fallback = False
        
        if at_charger and not must_leave:
            # First time at charger - MUST charge, validate charging options
            if self.verbose:
                print(f"[Detour] At charger {current_location}, must charge - checking charging durations...")
            
            # Get actual charging durations from action space (not from config!)
            # action_to_node_map stores the charger_id for charging actions; the
            # actual duration is in action_charge_durations at the same index.
            charge_durations_in_action_space = sorted({
                float(data.action_charge_durations[idx].item())
                for idx, (_, is_charging) in enumerate(action_to_node_map)
                if is_charging
            })
            
            if self.verbose:
                print(f"[Detour] Charging durations in action space: {charge_durations_in_action_space}")
            
            # First pass: check if ANY duration satisfies primary criteria
            for charge_duration in charge_durations_in_action_space:
                if self.validate_charging_duration_detour_based(
                    env, active_truck, current_location, current_battery,
                    charge_duration, next_delivery, remaining_deliveries,
                    battery_capacity, energy_safety_factor
                ):
                    valid_charging_durations.add(charge_duration)
            
            if valid_charging_durations:
                # At least one duration satisfies primary criteria
                if self.verbose:
                    print(f"[Detour] Valid charging durations: {sorted(valid_charging_durations)}")
            else:
                # No duration satisfies primary criteria - check if max duration at least reaches next delivery
                max_duration = max(charge_durations_in_action_space)
                
                # Calculate battery after max charging
                charger_type = env.charging_station.charger_type.get(current_location, "DCFast")
                charging_config = env.config["charging"]
                
                if charger_type == "DCFast":
                    charger_config_type = charging_config["dcfast"]
                else:
                    charger_config_type = charging_config["level2"]
                
                initial_soc = max(0.0, min(1.0, current_battery / battery_capacity))
                charger_config_with_curve = charger_config_type.copy()
                charger_config_with_curve["use_realistic_curve"] = charging_config.get("use_realistic_curve", False)
                
                charge_amount, _ = env.charging_curve_model.calculate_charge(
                    initial_soc=initial_soc,
                    charge_hours=max_duration,
                    battery_capacity=battery_capacity,
                    charger_config=charger_config_with_curve,
                    charger_type=charger_type
                )
                
                battery_after_max_charge = min(battery_capacity, current_battery + charge_amount)
                energy_to_delivery = env.transport_graph.get_path_energy(current_location, next_delivery)
                max_energy_to_delivery = energy_to_delivery * energy_safety_factor
                
                if not np.isinf(energy_to_delivery) and battery_after_max_charge >= max_energy_to_delivery:
                    # Max duration at least reaches next delivery - use as fallback
                    use_max_duration_fallback = True
                    valid_charging_durations.add(max_duration)
                    if self.verbose:
                        print(f"[Detour] No duration satisfies primary criteria, using max duration {max_duration}h as fallback")
                else:
                    # Even max charging won't help - allow escape routing to other chargers
                    allow_escape_routing = True
                    if self.verbose:
                        print(f"[Detour] No charging duration helps - enabling escape routing to all reachable chargers")
        
        # If we couldn't find any charger that both is reachable now and can reach the delivery on a full charge,
        # allow escape routing so the agent can chain chargers instead of getting stranded with zero actions.
        if no_feasible_chargers:
            allow_escape_routing = True

        # Process each action
        for action_idx, (node_id, is_charging_action) in enumerate(action_to_node_map):
            if is_charging_action:
                # Charging action
                charge_duration = float(data.action_charge_durations[action_idx].item())
                
                if not at_charger:
                    # Not at charger - cannot charge
                    new_feasible_mask[action_idx] = False
                elif must_leave:
                    # Must leave charger - cannot charge again
                    new_feasible_mask[action_idx] = False
                else:
                    # At charger and haven't charged yet
                    if allow_escape_routing:
                        # Escape mode: allow charging to build enough energy to reach another charger
                        new_feasible_mask[action_idx] = True
                    else:
                        # Normal mode: only allow durations that ensure progress toward delivery
                        new_feasible_mask[action_idx] = charge_duration in valid_charging_durations
                    
            elif node_id == -1:
                # Invalid action marker
                new_feasible_mask[action_idx] = False
                
            elif node_id in env.charging_nodes:
                # Routing to charger action
                
                # NEVER allow routing to current location (regardless of whether at charger or not)
                if node_id == current_location:
                    new_feasible_mask[action_idx] = False
                    if self.verbose:
                        print(f"[Detour] Blocked routing to current location (charger {node_id})")
                elif must_charge_now and not allow_escape_routing:
                    # Must charge now - cannot route away
                    new_feasible_mask[action_idx] = False
                elif route_delivery_first and must_leave and delivery_feasible and not allow_escape_routing:
                    # After charging: prefer direct delivery; suppress charger routing if delivery is feasible
                    new_feasible_mask[action_idx] = False
                else:
                    # Check if this charger is in top-2
                    top_charger_ids = [c['id'] for c in top_chargers_by_detour]
                    
                    if allow_escape_routing:
                        # Escape mode - allow routing to ANY reachable charger (except current)
                        energy_to_charger = env.transport_graph.get_path_energy(current_location, node_id)
                        if not np.isinf(energy_to_charger):
                            max_energy = energy_to_charger * energy_safety_factor
                            new_feasible_mask[action_idx] = current_battery >= max_energy
                        else:
                            new_feasible_mask[action_idx] = False
                    elif node_id in top_charger_ids:
                        # In top-2 - check reachability
                        energy_to_charger = env.transport_graph.get_path_energy(current_location, node_id)
                        if not np.isinf(energy_to_charger):
                            max_energy = energy_to_charger * energy_safety_factor
                            new_feasible_mask[action_idx] = current_battery >= max_energy
                        else:
                            new_feasible_mask[action_idx] = False
                    else:
                        # Not in top-2 - disable
                        new_feasible_mask[action_idx] = False
                        
            else:
                # Routing to delivery action
                
                # NEVER allow routing to current location (should not happen for deliveries but be safe)
                if node_id == current_location:
                    new_feasible_mask[action_idx] = False
                    if self.verbose:
                        print(f"[Detour] Blocked routing to current location (delivery {node_id})")
                elif must_charge_now and not allow_escape_routing:
                    # Must charge now - cannot route away
                    new_feasible_mask[action_idx] = False
                else:
                    # Check if can reach delivery
                    energy_to_delivery = env.transport_graph.get_path_energy(current_location, node_id)
                    if not np.isinf(energy_to_delivery):
                        max_energy = energy_to_delivery * energy_safety_factor
                        if current_battery >= max_energy:
                            # Mirror env check_navigation_feasibility so masks never allow stranded moves
                            new_feasible_mask[action_idx] = check_navigation_feasibility(
                                truck=active_truck,
                                target_node=node_id,
                                discharge=max_energy,
                                transport_graph=env.transport_graph,
                                charging_nodes=env.charging_nodes,
                                energy_safety_factor=energy_safety_factor,
                                verbose=self.verbose,
                            )
                        else:
                            new_feasible_mask[action_idx] = False
                    else:
                        new_feasible_mask[action_idx] = False

                    # If we wanted delivery-only but it is infeasible, keep chargers enabled elsewhere
        
        # Update the mask
        # If no feasible actions remain, provide a safe fallback to avoid empty masks
        if not any(new_feasible_mask):
            # Prefer enabling direct delivery if it exists in the action map
            delivery_idx = None
            for idx, (nid, is_chg) in enumerate(action_to_node_map):
                if not is_chg and nid == next_delivery:
                    delivery_idx = idx
                    break
            if delivery_idx is not None:
                new_feasible_mask[delivery_idx] = True
            else:
                # Otherwise enable the first charger action that is not current location
                charger_idx = None
                for idx, (nid, is_chg) in enumerate(action_to_node_map):
                    if not is_chg and nid in env.charging_nodes and nid != current_location:
                        charger_idx = idx
                        break
                if charger_idx is not None:
                    new_feasible_mask[charger_idx] = True
                elif action_to_node_map:
                    # Last resort: enable the first action
                    new_feasible_mask[0] = True

        data.feasible_action_mask = torch.tensor(new_feasible_mask, dtype=torch.bool, device=self.device)

        # Rebuild action_graph_features to match the updated feasible set
        feasible_indices = [i for i, is_feas in enumerate(new_feasible_mask) if is_feas]
        if feasible_indices:
            action_is_charging_list = data.action_is_charging.tolist() if hasattr(data, 'action_is_charging') else []
            action_charge_durations_list = [float(data.action_charge_durations[i].item()) for i in feasible_indices]
            feasible_action_to_node_map = [action_to_node_map[i] for i in feasible_indices]
            feasible_action_is_charging = [action_is_charging_list[i] for i in feasible_indices]
            data.action_graph_features = self._build_action_graph_features(
                env,
                feasible_action_to_node_map,
                feasible_action_is_charging,
                action_charge_durations_list,
                env.active_truck_id,
            )
        else:
            # Empty feasible set should be handled upstream, but keep tensor aligned
            data.action_graph_features = torch.zeros((0, self.action_feature_dim), dtype=torch.float32, device=self.device)
        
        if self.verbose:
            num_feasible = sum(new_feasible_mask)
            num_charger_actions = len([1 for nid, is_chg in action_to_node_map 
                                      if not is_chg and nid in env.charging_nodes])
            feasible_chargers = [nid for idx, (nid, is_chg) in enumerate(action_to_node_map) 
                                if not is_chg and nid in env.charging_nodes and new_feasible_mask[idx]]
            print(f"[Detour] Rebuilt mask: {num_feasible} feasible actions total")
            print(f"[Detour] Charger actions: {len(feasible_chargers)} feasible of {num_charger_actions} total")
            print(f"[Detour] Feasible chargers: {feasible_chargers}")
            print(f"[Detour] Must charge now: {must_charge_now}, Allow escape: {allow_escape_routing}")
        
        return data
    
    def select_top_2_chargers_by_detour(
        self,
        env,
        current_location: int,
        current_battery: float,
        next_delivery: int,
        battery_capacity: float,
        energy_safety_factor: float,
    ) -> List[Dict]:
        """
        Select top-2 chargers based purely on minimum detour to next delivery.
        
        Detour is calculated as:
            detour = energy_to_charger + energy_charger_to_delivery - energy_direct_to_delivery
        
        Selection criteria:
        1. Must be reachable with current battery
        2. Must enable reaching next delivery from charger (with full charge)
        3. Minimize detour energy
        
        Args:
            env: Environment instance
            current_location: Current truck location
            current_battery: Current battery level
            next_delivery: Next delivery node
            battery_capacity: Truck battery capacity
            energy_safety_factor: Safety factor for energy calculations
            
        Returns:
            List of top-2 charger dicts with keys: 'id', 'detour', 'energy_to_charger', 'energy_to_delivery'
            Sorted by ascending detour.
            
        Raises:
            ValueError: If no chargers are reachable and feasible
        """
        if self.verbose:
            print(f"[Detour] Selecting top-2 chargers: location={current_location}, battery={current_battery:.1f}, next_delivery={next_delivery}")
        
        # Calculate direct energy to next delivery (baseline)
        energy_direct = env.transport_graph.get_path_energy(current_location, next_delivery)
        
        if np.isinf(energy_direct):
            raise ValueError(f"Cannot reach next delivery {next_delivery} from {current_location}")
        
        charger_candidates = []
        
        for charger_id in env.charging_nodes:
            # Can we reach this charger with current battery?
            energy_to_charger = env.transport_graph.get_path_energy(current_location, charger_id)
            
            if np.isinf(energy_to_charger):
                continue  # Not connected
            
            max_energy_to_charger = energy_to_charger * energy_safety_factor
            
            # Special case: already at this charger
            if charger_id == current_location:
                energy_to_charger = 0.0
                max_energy_to_charger = 0.0
            elif current_battery < max_energy_to_charger:
                continue  # Cannot reach with current battery
            
            # Can we reach next delivery from this charger (assuming full charge)?
            energy_charger_to_delivery = env.transport_graph.get_path_energy(charger_id, next_delivery)
            
            if np.isinf(energy_charger_to_delivery):
                continue  # Cannot reach delivery from charger
            
            max_energy_charger_to_delivery = energy_charger_to_delivery * energy_safety_factor
            
            # Check if even with full battery we can reach delivery from charger
            if battery_capacity < max_energy_charger_to_delivery:
                continue  # Even full battery insufficient
            
            # Calculate detour
            detour = energy_to_charger + energy_charger_to_delivery - energy_direct
            
            charger_candidates.append({
                'id': charger_id,
                'detour': detour,
                'energy_to_charger': energy_to_charger,
                'energy_to_delivery': energy_charger_to_delivery,
            })
        
        if not charger_candidates:
            raise ValueError(
                f"No feasible chargers found. "
                f"Location: {current_location}, Battery: {current_battery:.1f} kWh, "
                f"Next delivery: {next_delivery}"
            )
        
        # Sort by detour (ascending) and select top-2
        charger_candidates.sort(key=lambda x: (x['detour'], x['energy_to_charger']))
        top_chargers = charger_candidates[:self.NUM_CHARGERS_TO_KEEP]
        
        if self.verbose:
            print(f"[Detour] Found {len(charger_candidates)} feasible chargers, selected top-{len(top_chargers)}")
            for c in top_chargers:
                print(f"[Detour]   Charger {c['id']}: detour={c['detour']:.1f} kWh")
        
        return top_chargers
    
    def validate_charging_duration_detour_based(
        self,
        env,
        truck,
        current_location: int,
        current_battery: float,
        charge_duration: float,
        next_delivery: int,
        remaining_deliveries: List[int],
        battery_capacity: float,
        energy_safety_factor: float,
    ) -> bool:
        """
        Validate that charging for given duration enables progress toward trip completion.
        
        A charging duration is feasible if after charging, truck can:
        OPTION 1: Reach next delivery AND reach ANY charger from delivery, OR
        OPTION 2: Reach next delivery AND complete all remaining deliveries
        
        Args:
            env: Environment instance
            truck: Active truck
            current_location: Current truck location (must be a charger)
            current_battery: Current battery level
            charge_duration: Charging duration in hours
            next_delivery: Next delivery node
            remaining_deliveries: List of all remaining deliveries
            battery_capacity: Truck battery capacity
            energy_safety_factor: Safety factor for energy calculations
            
        Returns:
            True if this charging duration enables progress, False otherwise
        """
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
            return False  # Cannot reach delivery
        
        max_energy_to_delivery = energy_to_delivery * energy_safety_factor
        
        if battery_after_charging < max_energy_to_delivery:
            if self.verbose and charge_duration <= 2:  # Debug first 2 durations
                print(f"[Detour] Charge {charge_duration}h: {battery_after_charging:.1f} < {max_energy_to_delivery:.1f} - cannot reach delivery")
            return False  # Cannot reach delivery even after charging
        
        # Battery after reaching delivery
        battery_at_delivery = battery_after_charging - max_energy_to_delivery
        
        if self.verbose and charge_duration <= 2:  # Debug first 2 durations
            print(f"[Detour] Charge {charge_duration}h: battery after charging = {battery_after_charging:.1f}, at delivery = {battery_at_delivery:.1f}")
        
        # Check if this is the last delivery
        if len(remaining_deliveries) <= 1:
            # This is the last delivery - just need to reach it
            if self.verbose:
                print(f"[Detour] Charge {charge_duration}h valid: can reach last delivery d{next_delivery}")
            return True
        
        # OPTION 1: Can reach ANY charger from delivery?
        can_reach_charger_from_delivery = False
        for charger_id in env.charging_nodes:
            energy_delivery_to_charger = env.transport_graph.get_path_energy(next_delivery, charger_id)
            
            if not np.isinf(energy_delivery_to_charger):
                max_energy_to_charger = energy_delivery_to_charger * energy_safety_factor
                
                if self.verbose and charge_duration <= 2:  # Debug first 2 durations
                    print(f"[Detour]   Checking charger {charger_id}: needs {energy_delivery_to_charger:.1f} kWh (with safety: {max_energy_to_charger:.1f}), have {battery_at_delivery:.1f}")
                
                if battery_at_delivery >= max_energy_to_charger:
                    # Can reach delivery + charger from delivery
                    if self.verbose:
                        print(f"[Detour] Charge {charge_duration}h valid: can reach d{next_delivery} + charger {charger_id}")
                    return True
                    
                # Track if at least one charger exists (even if not reachable with current battery)
                can_reach_charger_from_delivery = True
        
        # OPTION 2: Can complete remaining deliveries from delivery WITHOUT additional charging?
        # NOTE: _can_complete_route_from allows charging at intermediate chargers,
        # which is too permissive. We should only validate if truck can reach next delivery + charger.
        # Commenting out Option 2 for now - only Option 1 (reach delivery + charger) is required.
        # if self._can_complete_route_from(
        #     env, next_delivery, battery_at_delivery,
        #     remaining_deliveries[1:], battery_capacity, energy_safety_factor
        # ):
        #     if self.verbose:
        #         print(f"[Detour] Charge {charge_duration}h valid: can reach d{next_delivery} + complete remaining deliveries")
        #     return True
        
        # Neither option satisfied
        if self.verbose:
            print(f"[Detour] Charge {charge_duration}h INVALID: reaches d{next_delivery} but cannot reach any charger from there (battery at delivery: {battery_at_delivery:.1f} kWh)")
        return False
    
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
        
        Uses greedy simulation: at each delivery, if can't reach next, try to reach
        nearest charger, charge to full, and continue.
        
        Args:
            env: Environment instance
            start_location: Starting location
            battery_at_start: Battery level at start
            remaining_deliveries: List of remaining deliveries to complete
            battery_capacity: Truck battery capacity
            energy_safety_factor: Safety factor for energy calculations
            
        Returns:
            True if can complete all remaining deliveries, False otherwise
        """
        if not remaining_deliveries:
            return True  # No deliveries left
        
        current_loc = start_location
        current_battery = battery_at_start
        
        if self.verbose:
            print(f"[Detour] Simulating route from {start_location} with battery {battery_at_start:.1f}")
        
        for i, delivery in enumerate(remaining_deliveries):
            # Try to reach this delivery directly
            energy_to_delivery = env.transport_graph.get_path_energy(current_loc, delivery)
            
            if np.isinf(energy_to_delivery):
                if self.verbose:
                    print(f"[Detour]   Cannot reach delivery {delivery} from {current_loc}")
                return False
            
            max_energy_to_delivery = energy_to_delivery * energy_safety_factor
            
            if current_battery >= max_energy_to_delivery:
                # Can reach delivery directly
                current_battery -= max_energy_to_delivery
                current_loc = delivery
                
                if self.verbose:
                    print(f"[Detour]   Reached delivery {delivery}, battery now {current_battery:.1f}")
                
                continue
            
            # Cannot reach delivery directly - need to charge first
            # Find nearest reachable charger
            best_charger = None
            min_energy_to_charger = float('inf')
            
            for charger_id in env.charging_nodes:
                energy_to_charger = env.transport_graph.get_path_energy(current_loc, charger_id)
                
                if np.isinf(energy_to_charger):
                    continue
                
                max_energy_to_charger = energy_to_charger * energy_safety_factor
                
                if current_battery >= max_energy_to_charger:
                    # Can reach this charger
                    if energy_to_charger < min_energy_to_charger:
                        min_energy_to_charger = energy_to_charger
                        best_charger = charger_id
            
            if best_charger is None:
                # Cannot reach any charger
                if self.verbose:
                    print(f"[Detour]   Cannot reach any charger from {current_loc}")
                return False
            
            # Go to charger and charge to full
            current_battery -= min_energy_to_charger * energy_safety_factor
            current_battery = battery_capacity  # Charge to full
            current_loc = best_charger
            
            if self.verbose:
                print(f"[Detour]   Charged at {best_charger}, battery now {current_battery:.1f}")
            
            # Now try to reach delivery again
            energy_to_delivery = env.transport_graph.get_path_energy(current_loc, delivery)
            
            if np.isinf(energy_to_delivery):
                if self.verbose:
                    print(f"[Detour]   Cannot reach delivery {delivery} from charger {current_loc}")
                return False
            
            max_energy_to_delivery = energy_to_delivery * energy_safety_factor
            
            if current_battery < max_energy_to_delivery:
                # Even with full charge cannot reach delivery
                if self.verbose:
                    print(f"[Detour]   Full battery insufficient to reach delivery {delivery}")
                return False
            
            current_battery -= max_energy_to_delivery
            current_loc = delivery
            
            if self.verbose:
                print(f"[Detour]   Reached delivery {delivery} after charging, battery now {current_battery:.1f}")
        
        if self.verbose:
            print(f"[Detour] Successfully completed all deliveries")
        return True
