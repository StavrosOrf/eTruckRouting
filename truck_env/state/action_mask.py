"""
Action feasibility mask generation for the truck routing environment.

This module provides functionality to determine which actions are feasible
for the active truck based on battery constraints, location, and state.
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from truck_env.models.event_driven_env import EventDrivenTruckEnv


def get_action_mask(env: "EventDrivenTruckEnv") -> np.ndarray:
    """
    Generate feasibility mask for actions using the same logic as GNN state space.
    
    Args:
        env: EventDrivenTruckEnv instance
    
    Returns:
        np.ndarray: Boolean array where True indicates feasible actions.
                   Shape: (action_space.n,)
                   Order: [charger_0, ..., charger_N-1, next_delivery, charge_1h, ..., charge_4h]
    """
    # Initialize all actions as infeasible
    feasible_mask = np.zeros(env.action_space.n, dtype=bool)
    
    # If no active truck, all actions are infeasible
    if env.active_truck_id is None:
        return feasible_mask
    
    active_truck = env.trucks[env.active_truck_id]
    
    # Skip if truck is completed or failed
    if active_truck.failed or active_truck.is_complete:
        return feasible_mask
    
    current_battery = active_truck.current_battery
    current_location = active_truck.current_node
    charge_durations = env.charging_config['charge_durations']
    
    # Check if truck must leave charger (after charging)
    must_leave = active_truck.must_leave_charger
    
    # Check if truck is at charger - if so, must charge (unless must_leave is True)
    at_charger = current_location in env.charging_nodes
    must_charge_now = at_charger and not must_leave
    
    next_delivery = active_truck.get_next_delivery_target()
    
    # Build charger node to index mapping
    charger_node_to_idx = {node: idx for idx, node in enumerate(sorted(env.charging_nodes))}
    
    # --- Navigation Actions to Chargers ---
    # Actions 0 to num_charging_nodes-1: Go to charger i
    for action_idx, charger_id in enumerate(sorted(env.charging_nodes)):
        if charger_id == current_location:
            # Current location - routing here is always infeasible
            feasible_mask[action_idx] = False
        else:
            energy = env.transport_graph.get_path_energy(current_location, charger_id)
            is_energy_feasible = energy < current_battery and not np.isinf(energy)
            # Disable routing if truck must charge now
            is_feasible = is_energy_feasible and not must_charge_now
            feasible_mask[action_idx] = is_feasible
    
    # --- Navigation Action to Next Delivery ---
    # Action num_charging_nodes: Go to next delivery
    delivery_action_idx = env.num_charging_nodes
    
    if next_delivery is not None:
        energy_to_delivery = env.transport_graph.get_path_energy(current_location, next_delivery)
        is_energy_feasible = energy_to_delivery < current_battery
        
        # Additional check: After reaching delivery, can truck reach ANY charger or next delivery?
        can_continue_after_delivery = False
        if is_energy_feasible:
            battery_after_delivery = current_battery - energy_to_delivery
            
            # Check if there are more deliveries after this one
            remaining_after_this = active_truck.get_remaining_deliveries()
            has_more_deliveries = len(remaining_after_this) > 1
            
            if not has_more_deliveries:
                can_continue_after_delivery = True
            else:
                # Check if can reach any charger after delivery
                for charger_id in env.charging_nodes:
                    energy_to_charger = env.transport_graph.get_path_energy(next_delivery, charger_id)
                    if battery_after_delivery > energy_to_charger:
                        can_continue_after_delivery = True
                        break
        
        # Disable routing if truck must charge now OR if truck would be stranded after delivery
        is_feasible = is_energy_feasible and not must_charge_now and can_continue_after_delivery
        feasible_mask[delivery_action_idx] = is_feasible
    else:
        feasible_mask[delivery_action_idx] = False
    
    # --- Charging Actions at Current Location ---
    # Actions (num_charging_nodes+1) to end: Charge for 1-4 hours
    charge_action_start_idx = env.num_navigation_actions
    
    if at_charger:
        # If truck must leave charger, disable all charging actions
        if must_leave:
            for i, charge_hours in enumerate(charge_durations):
                feasible_mask[charge_action_start_idx + i] = False
        else:
            # Determine minimum energy needed to leave charger
            deliveries_left = active_truck.get_remaining_deliveries()
            min_energy_to_leave = env.transport_graph.get_path_energy(current_location, next_delivery)
            
            if not (len(deliveries_left) == 1 and min_energy_to_leave < active_truck.battery_capacity):
                # Find closest charger from current location (excluding current)
                closest_charger_energy = float('inf')
                for charger_id in env.charging_nodes:
                    if charger_id != current_location:
                        energy_to_charger = env.transport_graph.get_path_energy(current_location, charger_id)
                        if energy_to_charger < closest_charger_energy:
                            closest_charger_energy = energy_to_charger
                
                min_energy_to_leave = closest_charger_energy
            
            # Get charger configuration for charge rate calculation
            charger_type = env.charging_station.charger_type.get(current_location, "Level2")
            charging_config = env.config["charging"]
            if charger_type == "DCFast":
                charger_config = charging_config["dcfast"]
            else:
                charger_config = charging_config["level2"]
            charge_rate = charger_config["charge_rate"]  # kW
            efficiency = charger_config["efficiency"]
            
            # Evaluate each charge duration
            for i, charge_hours in enumerate(charge_durations):
                charge_amount = charge_hours * charge_rate * efficiency
                resulting_battery = min(active_truck.battery_capacity, current_battery + charge_amount)
                
                # Check if resulting battery is enough to leave
                is_feasible = resulting_battery >= min_energy_to_leave
                feasible_mask[charge_action_start_idx + i] = is_feasible
    else:
        # Not at charger - can't charge
        for i in range(len(charge_durations)):
            feasible_mask[charge_action_start_idx + i] = False
    
    return feasible_mask
