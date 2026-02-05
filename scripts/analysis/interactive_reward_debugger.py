#!/usr/bin/env python3
"""
Interactive reward debugger for the EventDrivenTruckEnv.

At every environment step the script:
  - Prints detailed environment / truck statistics.
  - Lists every discrete action with feasibility diagnostics.
  - Simulates each action on a cloned environment to preview reward + termination.
  - Highlights the suggestion from a simple heuristic policy.
  - Lets the user pick which action to actually execute.

This is intended for manual reward debugging on the fully discrete action space.
"""

import argparse
import copy
import random
from typing import Dict, List, Optional

import numpy as np

from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.state.gnn_utils import create_default_gnn_space
from EVRoutingEnv.utils.utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive reward debugger for EVPR.")
    parser.add_argument("--config", type=str, default="EVRoutingEnv/config_files/config_vrp.yaml",
    # parser.add_argument("--config", type=str, default="EVRoutingEnv/config_files/config.yaml",
                        help="Path to the environment config file.")
    parser.add_argument("--seed", type=int, default=1, help="Environment RNG seed.")
    parser.add_argument("--num-trucks", type=int, default=1,
                        help="Override number of trucks from the config.")
    parser.add_argument("--num-stops", type=int, default=3,
                        help="Override number of delivery stops from the config.")
    parser.add_argument("--max-time", type=float, default=None,
                        help="Override maximum simulation time from the config.")
    parser.add_argument("--enable-traffic", action="store_true",
                        help="Enable stochastic traffic delays.")
    parser.add_argument("--max-steps", type=int, default=10_000,
                        help="Upper bound on interactive steps before exiting.")
    parser.add_argument("--auto-accept-heuristic", action="store_true",
                        help="Automatically take the heuristic action each step (non-interactive).")
    parser.add_argument("--env-verbose", action="store_true",
                        help="Also print the environment's internal verbose logs (very noisy).")
    return parser.parse_args()


def clone_env(env: EventDrivenTruckEnv) -> EventDrivenTruckEnv:
    """Deep copy the environment so we can simulate hypothetical actions safely."""
    return copy.deepcopy(env)


def compute_navigation_stats(env: EventDrivenTruckEnv, src: int, dst: Optional[int]) -> Dict[str, Optional[float]]:
    """Compute travel energy (kWh) and time (h) from src to dst if defined."""
    if dst is None:
        return {"energy": None, "time": None}
    # If src and dst are the same, no travel needed
    if src == dst:
        return {"energy": 0.0, "time": 0.0}
    energy = env.transport_graph.get_path_energy(src, dst)
    travel_time = env.transport_graph.get_time_distance(src, dst)
    if np.isinf(energy):
        energy = None
    if np.isinf(travel_time):
        travel_time = None
    return {"energy": energy, "time": travel_time}


def describe_actions(env: EventDrivenTruckEnv, gnn_state_space=None, gnn_single_charger=None) -> List[Dict]:
    """Build metadata for every discrete action."""
    if env.active_truck_id is None:
        return []

    truck = env.trucks[env.active_truck_id]
    current_node = int(truck.current_node)
    charge_durations = env.charging_config["charge_durations"]
    current_time = env.global_clock
    actions = []
    
    # Get action mask (ground truth feasibility from environment)
    env_action_mask = env.mask_fn()
    
    # Get GNN feasibility if available
    gnn_feasible_mask = None
    if gnn_state_space is not None:
        try:
            gnn_state = gnn_state_space.get_state_GNN(env)
            gnn_feasible_mask = gnn_state.feasible_action_mask.cpu().numpy()
        except Exception as e:
            print(f"Warning: Could not compute GNN feasibility: {e}")
            gnn_feasible_mask = None
    
    # Get single-charger GNN feasibility if available
    gnn_single_feasible_mask = None
    if gnn_single_charger is not None:
        try:
            gnn_single_state = gnn_single_charger.get_state_GNN(env)
            gnn_single_feasible_mask = gnn_single_state.feasible_action_mask.cpu().numpy()
        except Exception as e:
            print(f"Warning: Could not compute single-charger GNN feasibility: {e}")
            gnn_single_feasible_mask = None
    
    # Get next delivery info for additional metrics
    next_delivery = truck.get_next_delivery_target()
    if truck.enable_flexible_delivery_order:
        remaining_deliveries = truck.get_remaining_deliveries()
        next_delivery_node = remaining_deliveries[0] if remaining_deliveries else None
    else:
        next_delivery_node = next_delivery

    for action_idx in range(env.action_space.n):
        if action_idx < env.num_charging_nodes:
            target_node = int(env.charging_nodes[action_idx])
            nav_stats = compute_navigation_stats(env, current_node, target_node)
            
            # Use actual action mask from environment for accurate feasibility
            feasible = bool(env_action_mask[action_idx])
            if not feasible:
                if nav_stats["energy"] is None or np.isinf(nav_stats["energy"]):
                    reason = "Unreachable path"
                elif nav_stats["energy"] >= truck.current_battery:
                    energy_safety = 1.0
                    if hasattr(env, 'traffic_config') and env.traffic_config.get('enable_traffic') and env.traffic_config.get('enable_energy_uncertainty'):
                        energy_safety = env.traffic_config.get('max_energy_multiplier', 1.0)
                    reason = f"Insufficient battery (need {nav_stats['energy'] * energy_safety:.1f}kWh with safety factor, have {truck.current_battery:.1f}kWh)"
                elif truck.current_node in env.charging_nodes and not truck.must_leave_charger:
                    reason = "Must charge now (at charger)"
                else:
                    reason = "Insufficient battery with safety margin"
            else:
                reason = None
            action_type = "route:charger"
            charge_hours = None
            
            # Estimate arrival time and potential wait
            arrival_time = current_time + nav_stats["time"] if nav_stats["time"] is not None else None
            queue_length = len(env.charging_station.charger_waitlist.get(target_node, []))
            occupancy = len(env.charging_station.charger_occupancy.get(target_node, []))
            capacity = env.charging_station.charger_capacity.get(target_node, 1)
            available_ports = capacity - occupancy
            
            # Estimate wait time (rough approximation)
            if available_ports > 0:
                estimated_wait = 0.0
            else:
                # Assume average charging session is the median charge duration
                avg_charge_time = np.median(charge_durations) if charge_durations else 1.0
                estimated_wait = (queue_length + 1) * avg_charge_time / capacity
            
            # Calculate energy from charger to next delivery
            energy_to_next_delivery = None
            if next_delivery_node is not None:
                energy_to_next_delivery = env.transport_graph.get_path_energy(target_node, next_delivery_node)
                if np.isinf(energy_to_next_delivery):
                    energy_to_next_delivery = None
            
            # Calculate battery at arrival (after navigation)
            battery_at_arrival = None
            if nav_stats["energy"] is not None:
                battery_at_arrival = truck.current_battery - nav_stats["energy"]
            
        elif action_idx < env.num_navigation_actions:
            # Delivery action(s) - handle both sequential and flexible modes
            if truck.enable_flexible_delivery_order:
                # Flexible mode: decode which delivery from action index
                delivery_idx = action_idx - env.num_charging_nodes
                num_delivery_slots = len(truck.delivery_sequence)  # includes depot at [0]
                
                # Check if this is the depot return action (last slot)
                if delivery_idx == num_delivery_slots - 1:
                    # Depot return action
                    target_node = truck.delivery_sequence[0]  # Depot is at index 0
                    nav_stats = compute_navigation_stats(env, current_node, target_node)
                    feasible = bool(env_action_mask[action_idx])
                    
                    if not feasible:
                        if not getattr(truck, 'all_deliveries_done', False):
                            reason = "Not all deliveries completed yet"
                        elif nav_stats["energy"] is None or np.isinf(nav_stats["energy"]):
                            reason = "Unreachable path"
                        elif nav_stats["energy"] >= truck.current_battery:
                            reason = "Insufficient battery"
                        else:
                            reason = "Depot return not available"
                    else:
                        reason = None
                    action_type = "route:depot"
                elif delivery_idx < len(truck.delivery_sequence) - 1:
                    target_node = truck.delivery_sequence[delivery_idx + 1]
                    # Check if this delivery is still remaining
                    remaining_deliveries = truck.get_next_delivery_target()
                    if isinstance(remaining_deliveries, list) and target_node not in remaining_deliveries:
                        # Already delivered
                        nav_stats = {"energy": None, "time": None}
                        feasible = False
                        reason = f"Delivery at node {target_node} already completed"
                    else:
                        nav_stats = compute_navigation_stats(env, current_node, target_node)
                        # Use actual action mask from environment for accurate feasibility
                        feasible = bool(env_action_mask[action_idx])
                        if not feasible:
                            if nav_stats["energy"] is None or np.isinf(nav_stats["energy"]):
                                reason = "Unreachable path"
                            elif nav_stats["energy"] >= truck.current_battery:
                                reason = "Insufficient battery"
                            else:
                                # Check if it's a stranding issue
                                energy_safety = 1.0
                                if hasattr(env, 'traffic_config') and env.traffic_config.get('enable_traffic') and env.traffic_config.get('enable_energy_uncertainty'):
                                    energy_safety = env.traffic_config.get('max_energy_multiplier', 1.0)
                                
                                battery_after = truck.current_battery - (nav_stats["energy"] * energy_safety)
                                
                                # Check if can reach any charger from delivery
                                can_reach_charger = False
                                min_charger_dist = float('inf')
                                for charger_id in env.charging_nodes:
                                    charger_energy = env.transport_graph.get_path_energy(target_node, charger_id)
                                    if charger_energy < min_charger_dist:
                                        min_charger_dist = charger_energy
                                    if battery_after > charger_energy * energy_safety:
                                        can_reach_charger = True
                                        break
                                
                                if not can_reach_charger and min_charger_dist != float('inf'):
                                    reason = f"Would be stranded (need {min_charger_dist * energy_safety:.1f}kWh to nearest charger, have {battery_after:.1f}kWh after)"
                                elif truck.current_node in env.charging_nodes and not truck.must_leave_charger:
                                    reason = "Must charge now (at charger)"
                                else:
                                    reason = "Would be stranded after delivery"
                        else:
                            reason = None
                    action_type = f"route:delivery[{delivery_idx}]"
                else:
                    # No delivery at this position
                    target_node = None
                    nav_stats = {"energy": None, "time": None}
                    feasible = False
                    reason = "No delivery at this sequence position"
                    action_type = f"route:delivery[{delivery_idx}]"
            else:
                # Sequential mode: single next delivery
                target_node = truck.get_next_delivery_target()
                nav_stats = compute_navigation_stats(env, current_node, target_node)
                # Use actual action mask from environment for accurate feasibility
                feasible = bool(env_action_mask[action_idx])
                if not feasible:
                    if target_node is None:
                        reason = "No deliveries left"
                    elif nav_stats["energy"] is None or np.isinf(nav_stats["energy"]):
                        reason = "Unreachable path"
                    elif nav_stats["energy"] >= truck.current_battery:
                        reason = "Insufficient battery"
                    else:
                        reason = "Would be stranded after delivery or must charge now"
                else:
                    reason = None
                action_type = "route:delivery"
            
            charge_hours = None
            arrival_time = current_time + nav_stats["time"] if nav_stats["time"] is not None else None
            queue_length = None
            estimated_wait = None
            
            # Calculate distance to closest charger from this delivery node
            energy_to_next_delivery = None
            if target_node is not None and target_node not in env.charging_nodes:
                # Find closest charger from this delivery
                min_charger_energy = float('inf')
                for charger_id in env.charging_nodes:
                    charger_energy = env.transport_graph.get_path_energy(target_node, charger_id)
                    if charger_energy < min_charger_energy:
                        min_charger_energy = charger_energy
                if not np.isinf(min_charger_energy):
                    energy_to_next_delivery = min_charger_energy
                else:
                    energy_to_next_delivery = None
            
            # Calculate battery at arrival (after navigation)
            battery_at_arrival = None
            if nav_stats["energy"] is not None:
                battery_at_arrival = truck.current_battery - nav_stats["energy"]
            
        else:
            target_node = current_node
            nav_stats = {"energy": None, "time": None}
            charge_idx = action_idx - env.num_navigation_actions
            charge_hours = charge_durations[charge_idx]
            can_charge_here = target_node in env.charging_nodes
            not_full = (truck.current_battery + 1e-5) < truck.battery_capacity
            
            # Use action mask for accurate feasibility
            feasible = bool(env_action_mask[action_idx])
            
            reason = None
            if not can_charge_here:
                reason = "Not at a charger"
            elif not not_full:
                reason = "Battery already full"
            
            action_type = f"charge({charge_hours}h)"
            arrival_time = None
            queue_length = None
            estimated_wait = None
            energy_to_next_delivery = None
            
            # Calculate battery after charging
            battery_at_arrival = None
            if can_charge_here:
                # Get charging rate and calculate energy added
                charger_type = env.charging_station.charger_type.get(target_node, "DCFast")
                charging_config = env.config["charging"]
                if charger_type == "DCFast":
                    charger_config = charging_config["dcfast"]
                else:
                    charger_config = charging_config["level2"]
                
                charging_rate = charger_config["charge_rate"] * charger_config["efficiency"]
                energy_added = charging_rate * charge_hours
                battery_at_arrival = min(truck.current_battery + energy_added, truck.battery_capacity)

        # Generate label based on action type
        if action_type.startswith("charge"):
            label = f"Charge {charge_hours}h"
        elif action_type.startswith("route:charger"):
            label = f"→Charger {target_node}"
        elif action_type.startswith("route:delivery"):
            if truck.enable_flexible_delivery_order:
                if target_node is not None:
                    label = f"→Delivery {target_node}"
                else:
                    label = "No delivery"
            else:
                label = "→Next delivery"
        elif action_type.startswith("route:depot") or "depot" in action_type.lower():
            label = f"→DEPOT {target_node} (return)"
        else:
            label = env._action_to_string(action_idx)

        # Get GNN feasibility for this action
        gnn_feasible = None
        if gnn_feasible_mask is not None and action_idx < len(gnn_feasible_mask):
            gnn_feasible = bool(gnn_feasible_mask[action_idx])
        
        # Get single-charger GNN feasibility for this action
        gnn_single_feasible = None
        if gnn_single_feasible_mask is not None and action_idx < len(gnn_single_feasible_mask):
            gnn_single_feasible = bool(gnn_single_feasible_mask[action_idx])
        
        actions.append({
            "index": action_idx,
            "label": label,
            "type": action_type,
            "target": target_node,
            "charge_hours": charge_hours,
            "navigation": nav_stats,
            "feasible": feasible,
            "gnn_feasible": gnn_feasible,
            "gnn_single_feasible": gnn_single_feasible,
            "infeasible_reason": reason,
            "arrival_time": arrival_time,
            "queue_length": queue_length,
            "estimated_wait": estimated_wait,
            "energy_to_next_delivery": energy_to_next_delivery,
            "battery_at_arrival": battery_at_arrival,
        })

    return actions


def simulate_action(env: EventDrivenTruckEnv, action_idx: int) -> Dict:
    """
    Simulate one action on a cloned environment.
    Restores RNG state so the real environment stays deterministic.
    """
    np_state = np.random.get_state()
    py_state = random.getstate()
    try:
        sim_env = clone_env(env)
        obs, reward, done, truncated, info = sim_env.step(action_idx)
        summary = {
            "reward": reward,
            "done": done,
            "truncated": truncated,
            "info": {
                "all_complete": info.get("all_complete", False),
                "any_failed": info.get("any_failed", False),
                "global_clock": info.get("global_clock", None),
                "active_truck_id": info.get("active_truck_id", None),
            },
            "active_truck_post": info.get("active_truck_id", None),
        }
    except Exception as exc:
        summary = {"error": str(exc)}
    finally:
        np.random.set_state(np_state)
        random.setstate(py_state)
    return summary


def heuristic_action(env: EventDrivenTruckEnv, actions: List[Dict]) -> Optional[int]:
    """Simple heuristic: finish deliveries if plenty of charge, otherwise charge."""
    if env.active_truck_id is None:
        return None
    truck = env.trucks[env.active_truck_id]
    battery_pct = truck.get_battery_percentage()

    # Prefer charging if battery critically low or at station.
    if battery_pct < 25 or (truck.current_node in env.charging_nodes and battery_pct < 90):
        for action in actions:
            if action["type"].startswith("charge") and action["feasible"]:
                return action["index"]

        # If no feasible charging action, fall back to nearest charger.
        charger_actions = [
            (action["navigation"]["energy"], action["index"])
            for action in actions
            if action["type"].startswith("route:charger") and action["feasible"]
        ]
        if charger_actions:
            charger_actions.sort()
            return charger_actions[0][1]

    # Otherwise try to deliver - handle both sequential and flexible modes
    delivery_actions = [
        (action["navigation"]["energy"], action["index"])
        for action in actions
        if action["type"].startswith("route:delivery") and action["feasible"]
    ]
    if delivery_actions:
        # In flexible mode, choose closest delivery
        delivery_actions.sort()
        return delivery_actions[0][1]

    # Fall back to any feasible charger navigation.
    for action in actions:
        if action["feasible"]:
            return action["index"]

    return None


def evaluate_actions(env: EventDrivenTruckEnv, actions: List[Dict]) -> None:
    """Attach simulation results to each action dict."""
    for action in actions:
        action["simulation"] = simulate_action(env, action["index"])


def print_state_statistics(env: EventDrivenTruckEnv) -> None:
    """Pretty-print key environment and truck details."""
    if env.active_truck_id is None:
        print("\nNo active truck. Environment most likely finished.\n")
        return

    info = env._get_info()

    print("\n" + "=" * 120)
    print(f"Global clock: {info['global_clock']:.2f}h | Episode reward: {info['episode_reward']:.2f}")
    print(f"Events queued: {info['events_pending']} | Active truck: {info['active_truck_id']}")
    print("-" * 120)
    print("Truck status:")
    for truck_state in info["trucks"]:
        truck_id = truck_state['truck_id']
        truck = env.trucks[truck_id]
        
        # Get delivery info based on mode
        if truck.enable_flexible_delivery_order:
            remaining = truck.get_remaining_deliveries()
            # Delivery nodes are delivery_sequence[1:] (excluding depot at index 0)
            total_deliveries = len(truck.delivery_sequence) - 1
            completed = total_deliveries - len(remaining)
            all_done_flag = " | ALL_DONE→depot" if getattr(truck, 'all_deliveries_done', False) else ""
            depot_node = truck.delivery_sequence[0]
            next_del_info = f"depot={depot_node} | deliveries: {completed}/{total_deliveries}{all_done_flag}"
            if remaining:
                next_del_info += f" rem={remaining}"
        else:
            next_del = truck_state.get('next_delivery_target')
            next_del_info = f"next={next_del}"
        
        flag = ""
        time_info = ""
        
        if truck_state["failed"]:
            flag = "FAILED"
        elif truck_state["is_complete"]:
            flag = "COMPLETE"
        elif truck_state["is_charging"]:
            flag = "CHARGING"
            charge_end_time = env.charging_station.truck_charge_end_time.get(truck_id)
            if charge_end_time is not None:
                time_remaining = charge_end_time - env.global_clock
                time_info = f"(finishes in {time_remaining:.2f}h at {charge_end_time:.2f}h)"
        elif truck.route_destination is not None and truck.route_arrival_time is not None:
            flag = "ROUTING"
            time_to_arrival = truck.route_arrival_time - env.global_clock
            dest_type = "charger" if truck.route_destination in env.charging_nodes else "delivery"
            time_info = f"(to {dest_type} node {truck.route_destination}, arrives in {time_to_arrival:.2f}h at {truck.route_arrival_time:.2f}h)"
        else:
            flag = "READY"

        print(
            f"  Truck {truck_state['truck_id']:2d} | {flag:<8} {time_info:<55} | node={truck_state['current_node']:4d} "
            f"| {next_del_info:<40} "
            f"| battery={truck_state['current_battery']:.1f}/{truck_state['battery_capacity']:.1f} "
            f"({truck_state['battery_percentage']:.1f}%)"
        )

    print("-" * 120)
    print("Charger queues (waitlist lengths & occupancy):")
    for node in env.charging_nodes:
        node = int(node)
        wait_len = len(env.charging_station.charger_waitlist.get(node, []))
        occ = len(env.charging_station.charger_occupancy.get(node, []))
        cap = env.charging_station.charger_capacity.get(node, 1)
        available = cap - occ
        
        # List trucks in queue
        queue_trucks = env.charging_station.charger_waitlist.get(node, [])
        queue_str = f"queue: [{', '.join(str(t) for t in queue_trucks)}]" if queue_trucks else "queue: []"
        
        print(f"  Charger {node:4d} | available={available}/{cap} | waiting={wait_len} | {queue_str}")
    print("=" * 120 + "\n")


def format_sim_result(action: Dict) -> str:
    """Format simulation outcome for table display."""
    sim = action.get("simulation", {})
    if not sim:
        return "N/A"
    if "error" in sim:
        return "ERROR"
    reward = sim["reward"]
    return f"{reward:+7.2f}"


def print_actions_table(actions: List[Dict], heuristic_idx: Optional[int]) -> None:
    """Display all actions with diagnostics."""
    if not actions:
        print("No actions available.")
        return

    # ANSI color codes
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

    # Two-line header with units on second line
    header_line1 = f"{'Idx':<4} {'H':<2} {'Feas':<5} {'GNN':<4} {'1Chr':<4} {'Description':<24} {'ΔEnergy':<7} {'→NextDel':<8} {'Total':<7} {'TravelTime':<11} {'BatteryAtArr':<12} {'Reason':<20} {'SimReward':<10}"
    header_line2 = f"{'':4} {'':2} {'':5} {'':4} {'':4} {'':24} {'(kWh)':<7} {'(kWh)':<8} {'(kWh)':<7} {'(h)':<11} {'(kWh)':<12} {'':20} {'':<10}"
    print(header_line1)
    print(header_line2)
    print("-" * len(header_line1))

    for action in actions:
        nav_energy = action["navigation"]["energy"]
        nav_time = action["navigation"]["time"]
        energy_str = f"{nav_energy:>5.1f}" if nav_energy is not None else "  N/A"
        time_str = f"{nav_time:>6.2f}h" if nav_time is not None else "   N/A"
        
        # Energy to next delivery (for chargers) or closest charger (for deliveries)
        next_del_energy = action.get("energy_to_next_delivery")
        next_del_str = f"{next_del_energy:>5.1f}" if next_del_energy is not None else "  N/A"
        
        # Total energy (sum of nav_energy and next_del_energy)
        if nav_energy is not None and next_del_energy is not None:
            total_energy = nav_energy + next_del_energy
            total_str = f"{total_energy:>5.1f}"
        else:
            total_str = "  N/A"
        
        # Battery at arrival
        battery_at_arrival = action.get("battery_at_arrival")
        battery_str = f"{battery_at_arrival:>6.1f}" if battery_at_arrival is not None else "   N/A"
        
        # Shorten reason for display
        reason = action.get("infeasible_reason", "")
        if reason:
            # Abbreviate common reasons
            if "Insufficient battery" in reason:
                reason = "Low battery"
            elif "Unreachable" in reason:
                reason = "Unreachable"
            elif "already completed" in reason or "already full" in reason:
                reason = "Already done"
            elif "stranded" in reason:
                reason = "Would strand"
            elif "Must charge now" in reason:
                reason = "Must charge"
            elif "Not at" in reason:
                reason = "Wrong location"
            elif "No deliver" in reason:
                reason = "No delivery"
            # Truncate if still too long
            if len(reason) > 18:
                reason = reason[:17] + "…"
        reason_str = reason if reason else "-"
        
        feasible = action["feasible"]
        feasible_str = "yes" if feasible else "no"
        gnn_feas = action.get("gnn_feasible")
        gnn_feas_str = "yes" if gnn_feas else ("no" if gnn_feas is not None else "-")
        gnn_single_feas = action.get("gnn_single_feasible")
        gnn_single_feas_str = "yes" if gnn_single_feas else ("no" if gnn_single_feas is not None else "-")
        heur_mark = "*" if heuristic_idx is not None and action["index"] == heuristic_idx else ""
        desc = action["label"]
        sim_summary = format_sim_result(action)
        
        # Apply color based on feasibility
        if feasible:
            color = GREEN
        else:
            color = RED
        
        print(
            f"{color}{action['index']:<4} {heur_mark:<2} {feasible_str:<5} {gnn_feas_str:<4} {gnn_single_feas_str:<4} "
            f"{desc:<24} {energy_str:<7} {next_del_str:<8} {total_str:<7} {time_str:<11} {battery_str:<12} {reason_str:<20} {sim_summary:<10}{RESET}"
        )
        
        # Show queue info for charger navigation actions (still useful to keep)
        if action['type'] == 'route:charger' and action['queue_length'] is not None and action['queue_length'] > 0:
            print(f"      {YELLOW}↳ queue: {action['queue_length']} trucks waiting{RESET}")

    print()


def prompt_user_action(actions: List[Dict], heuristic_idx: Optional[int], auto_accept: bool) -> Optional[int]:
    """Ask the user which action to execute."""
    valid_indices = {str(action["index"]) for action in actions}
    if auto_accept:
        print(f"[auto] Taking heuristic action {heuristic_idx}")
        return heuristic_idx

    print("Choose an action index (enter number), 'h' for heuristic, 'q' to quit.")
    while True:
        choice = input("Your choice: ").strip().lower()
        if choice == "q":
            return None
        if choice in ("h", ""):
            if heuristic_idx is not None:
                return heuristic_idx
            print("Heuristic action unavailable. Please choose explicit index.")
            continue
        if choice in valid_indices:
            return int(choice)
        print(f"Invalid choice. Valid options: {sorted(valid_indices)}")


def interactive_loop(env: EventDrivenTruckEnv, max_steps: int, auto_accept: bool, seed: int) -> None:
    obs, info = env.reset(seed=seed)
    step = 0
    done = False
    truncated = False
    
    # Initialize GNN state spaces based on delivery mode
    flexible = getattr(env, "enable_flexible_delivery_order", False)
    mode = "vrp" if flexible else "nonflex"

    gnn_state_space = create_default_gnn_space(env, mode=mode, use_detour=False)
    env._default_gnn_state_space = gnn_state_space
    env.use_detour_mask = False

    gnn_single_charger = None if flexible else create_default_gnn_space(env, mode="nonflex", use_detour=True)

    while not (done or truncated):
        step += 1
        if step > max_steps:
            print(f"Reached max interactive steps ({max_steps}). Exiting loop.")
            break

        print_state_statistics(env)
        if env.active_truck_id is None:
            break

        actions = describe_actions(env, gnn_state_space, gnn_single_charger)
        heuristic_idx = heuristic_action(env, actions)
        evaluate_actions(env, actions)
        print_actions_table(actions, heuristic_idx)

        action_choice = prompt_user_action(actions, heuristic_idx, auto_accept)
        if action_choice is None:
            print("Exiting at user request.")
            break

        print(f"\nExecuting action {action_choice}: {env._action_to_string(action_choice)}")
        obs, reward, done, truncated, info = env.step(action_choice)
        print(f"→ Reward: {reward:.4f}, done={done}, truncated={truncated}")
        print(f"→ Info: active_truck={info.get('active_truck_id')}, global_clock={info.get('global_clock'):.2f}h, "
              f"episode_reward={info.get('episode_reward'):.2f}\n")

        if done or truncated:
            print("Episode finished.")
            break


if __name__ == "__main__":
    args = parse_args()

    config = load_config(args.config)
    if args.num_trucks is not None:
        config["environment"]["num_trucks"] = args.num_trucks
    if args.num_stops is not None:
        config["environment"]["num_stops"] = args.num_stops
    if args.max_time is not None:
        config["environment"]["max_time"] = args.max_time
    if args.enable_traffic:
        config["traffic"]["enable_traffic"] = True

    env_config = config.setdefault("environment", {})
    env_config.setdefault("verbose", False)

    # Respect flexible delivery flag to align masks/state space
    flexible = config.get("delivery", {}).get("enable_flexible_delivery_order", False)
    env_config["enable_flexible_delivery_order"] = flexible

    env = EventDrivenTruckEnv(
        config=config,
        verbose=args.env_verbose,
        enable_plotting=True,
        run_id="interactive_debugger",
    )
    env.use_detour_mask = False

    try:
        interactive_loop(env, args.max_steps, args.auto_accept_heuristic, args.seed)
    finally:
        env.close()
