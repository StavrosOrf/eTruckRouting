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

from truck_env.models.event_driven_env import EventDrivenTruckEnv
from truck_env.utils.utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive reward debugger for EVPR.")
    parser.add_argument("--config", type=str, default="truck_env/config_files/config.yaml",
                        help="Path to the environment config file.")
    parser.add_argument("--seed", type=int, default=1, help="Environment RNG seed.")
    parser.add_argument("--num-trucks", type=int, default=2,
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


def describe_actions(env: EventDrivenTruckEnv) -> List[Dict]:
    """Build metadata for every discrete action."""
    if env.active_truck_id is None:
        return []

    truck = env.trucks[env.active_truck_id]
    current_node = int(truck.current_node)
    charge_durations = env.charging_config["charge_durations"]
    current_time = env.global_clock
    actions = []

    for action_idx in range(env.action_space.n):
        if action_idx < env.num_charging_nodes:
            target_node = int(env.charging_nodes[action_idx])
            nav_stats = compute_navigation_stats(env, current_node, target_node)
            feasible = (
                nav_stats["energy"] is not None
                and nav_stats["energy"] < truck.current_battery
                and not truck.failed
            )
            reason = None if feasible else "Insufficient battery or unreachable path"
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
            
        elif action_idx == env.num_charging_nodes:
            target_node = truck.get_next_delivery_target()
            nav_stats = compute_navigation_stats(env, current_node, target_node)
            feasible = target_node is not None and nav_stats["energy"] is not None and nav_stats["energy"] < truck.current_battery
            reason = None if feasible else "No deliveries left or too expensive"
            action_type = "route:delivery"
            charge_hours = None
            arrival_time = current_time + nav_stats["time"] if nav_stats["time"] is not None else None
            queue_length = None
            estimated_wait = None
            
        else:
            target_node = current_node
            nav_stats = {"energy": None, "time": None}
            charge_idx = action_idx - env.num_navigation_actions
            charge_hours = charge_durations[charge_idx]
            can_charge_here = target_node in env.charging_nodes
            not_full = (truck.current_battery + 1e-5) < truck.battery_capacity
            feasible = can_charge_here and not_full
            reason = None
            if not can_charge_here:
                reason = "Not at a charger"
            elif not not_full:
                reason = "Battery already full"
            action_type = f"charge({charge_hours}h)"
            arrival_time = None
            queue_length = None
            estimated_wait = None

        label = env._action_to_string(action_idx)
        if charge_hours is not None:
            label = f"Charge for {charge_hours}h"

        actions.append({
            "index": action_idx,
            "label": label,
            "type": action_type,
            "target": target_node,
            "charge_hours": charge_hours,
            "navigation": nav_stats,
            "feasible": feasible,
            "infeasible_reason": reason,
            "arrival_time": arrival_time,
            "queue_length": queue_length,
            "estimated_wait": estimated_wait,
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

    # Otherwise try to deliver.
    for action in actions:
        if action["type"] == "route:delivery" and action["feasible"]:
            return action["index"]

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
        
        flag = ""
        time_info = ""
        
        if truck_state["failed"]:
            flag = "FAILED"
        elif truck_state["is_complete"]:
            flag = "COMPLETE"
        elif truck_state["is_charging"]:
            flag = "CHARGING"
            if truck.charge_end_time is not None:
                time_remaining = truck.charge_end_time - env.global_clock
                time_info = f"(finishes in {time_remaining:.2f}h at {truck.charge_end_time:.2f}h)"
        elif truck.route_destination is not None and truck.route_arrival_time is not None:
            flag = "ROUTING"
            time_to_arrival = truck.route_arrival_time - env.global_clock
            dest_type = "charger" if truck.route_destination in env.charging_nodes else "delivery"
            time_info = f"(to {dest_type} node {truck.route_destination}, arrives in {time_to_arrival:.2f}h at {truck.route_arrival_time:.2f}h)"
        else:
            flag = "READY"

        print(
            f"  Truck {truck_state['truck_id']:2d} | {flag:<8} {time_info:<55} | node={truck_state['current_node']:4d} "
            f"| next_delivery={truck_state['next_delivery_target']:4d} "
            f"| battery={truck_state['current_battery']:.1f}/{truck_state['battery_capacity']:.1f} "
            f"({truck_state['battery_percentage']:.1f}%) | remaining={truck_state['deliveries_remaining']}"
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
        return f"ERR: {sim['error']}"
    status = "done" if sim["done"] else ("trunc" if sim["truncated"] else "ongoing")
    reward = sim["reward"]
    return f"{reward:+8.2f} | {status}"


def print_actions_table(actions: List[Dict], heuristic_idx: Optional[int]) -> None:
    """Display all actions with diagnostics."""
    if not actions:
        print("No actions available.")
        return

    header = f"{'Idx':<4} {'H':<2} {'Feas':<5} {'Type':<16} {'Description':<28} {'Target':<8} {'ΔEnergy':<9} {'TravelTime':<11} {'ArrivalAt':<11} {'EstWait':<9} {'Sim reward/status'}"
    print(header)
    print("-" * len(header))

    for action in actions:
        nav_energy = action["navigation"]["energy"]
        nav_time = action["navigation"]["time"]
        energy_str = f"{nav_energy:>6.1f}kWh" if nav_energy is not None else "   N/A"
        time_str = f"{nav_time:>6.2f}h" if nav_time is not None else "   N/A"
        arrival_str = f"{action['arrival_time']:>6.2f}h" if action['arrival_time'] is not None else "   N/A"
        wait_str = f"{action['estimated_wait']:>5.2f}h" if action['estimated_wait'] is not None else "  N/A"
        
        target = action["target"] if action["target"] is not None else "-"
        feasible = "yes" if action["feasible"] else "no"
        heur_mark = "*" if heuristic_idx is not None and action["index"] == heuristic_idx else ""
        desc = action["label"]
        sim_summary = format_sim_result(action)
        
        print(
            f"{action['index']:<4} {heur_mark:<2} {feasible:<5} {action['type']:<16} "
            f"{desc:<28} {str(target):<8} {energy_str:<9} {time_str:<11} {arrival_str:<11} {wait_str:<9} {sim_summary}"
        )
        
        # Show additional context for infeasible actions
        if not action["feasible"] and action["infeasible_reason"]:
            print(f"      ↳ reason: {action['infeasible_reason']}")
        
        # Show queue info for charger navigation actions
        if action['type'] == 'route:charger' and action['queue_length'] is not None and action['queue_length'] > 0:
            print(f"      ↳ queue: {action['queue_length']} trucks waiting")

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

    while not (done or truncated):
        step += 1
        if step > max_steps:
            print(f"Reached max interactive steps ({max_steps}). Exiting loop.")
            break

        print_state_statistics(env)
        if env.active_truck_id is None:
            break

        actions = describe_actions(env)
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

    env = EventDrivenTruckEnv(
        config=config,
        verbose=args.env_verbose,
        enable_plotting=True,
        run_id="interactive_debugger",
    )

    try:
        interactive_loop(env, args.max_steps, args.auto_accept_heuristic, args.seed)
    finally:
        env.close()
