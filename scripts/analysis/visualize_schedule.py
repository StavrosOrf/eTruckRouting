"""
Visualize truck schedules (Gantt chart) for a single scenario.
"""

import copy
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from EVRoutingEnv.models.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.state.gnn_state_space import GNNStateSpace
from EVRoutingEnv.utils.utils import load_config
from EVRoutingEnv.utils.plotter import EnvironmentPlotter
from EVRoutingEnv.baselines.optimal_gurobi import OptimalGurobiPolicy
from scripts.training.train_PPO_Variable import compute_action_mask
from algo.policy_utils import load_policy

# ============ CONFIGURATION ============
# Three policies to compare
POLICIES = [
    ("saved_models/NewFeasibleSpace_FixedGraph_ppo-variable_steps=1024_epochs=5_ent=0.1_seed=0_gnnhd=32_mlphd=256/", "variable-ppo"),
    ("optimal", "optimal"),
    ("heuristic", "heuristic"),
]

CONFIG_FILE = "EVRoutingEnv/config_files/config.yaml"
NUM_TRUCKS = 10
NUM_STOPS = 3
MAX_TIME = 200.0
SEED = 1000
OUTPUT_DIR = "results/visualization"
# =======================================

class InstrumentedEnv(EventDrivenTruckEnv):
    """
    Subclass of EventDrivenTruckEnv that records state history for visualization.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history = []  # List of (t_start, t_end, truck_details)

    def _advance_to_next_decision(self):
        """
        Override to capture the time interval and states *before* advancing time.
        """
        t_start = self.global_clock
        
        # Capture detailed state for each truck (including start SoC)
        truck_details = {}
        for truck in self.trucks:
            state = self.truck_states.get(truck.truck_id, "unknown")
            details = {
                "state": state,
                "current_node": int(truck.current_node),
                "destination": int(truck.route_destination) if truck.route_destination is not None else None,
                "start_soc": truck.get_battery_percentage()  # Capture SoC at start of interval
            }
            truck_details[truck.truck_id] = details
            
        super()._advance_to_next_decision()
        
        t_end = self.global_clock
        
        # Record the interval if time actually passed
        if t_end > t_start:
            # Add end SoC to details
            for truck in self.trucks:
                if truck.truck_id in truck_details:
                    truck_details[truck.truck_id]["end_soc"] = truck.get_battery_percentage()

            self.history.append({
                "start": t_start,
                "end": t_end,
                "trucks": truck_details
            })

def run_scenario(policy_path, policy_type):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Setup
    config = load_config(CONFIG_FILE)
    config["environment"]["num_trucks"] = NUM_TRUCKS
    config["environment"]["num_stops"] = NUM_STOPS
    config["environment"]["max_time"] = MAX_TIME
    
    # Initialize State Space
    env_init = EventDrivenTruckEnv(config=config, verbose=False, enable_plotting=False)
    gnn_state_space = GNNStateSpace(
        num_trucks=NUM_TRUCKS,
        num_stops=NUM_STOPS,
        max_time=config["environment"]["max_time"],
        num_charging_nodes=env_init.num_charging_nodes,
    )
    env_init.close()

    # Load Policy
    print(f"Loading policy: {policy_path} (type: {policy_type})...")
    
    if policy_type == "optimal":
        try:
            policy = OptimalGurobiPolicy(verbose=False)
            active_policy_type = "optimal"
        except ImportError as exc:
            raise RuntimeError(
                "Optimal (Gurobi) policy requires gurobipy to be installed."
            ) from exc
    elif policy_type == "heuristic":
        from EVRoutingEnv.baselines.heuristic_policy import HeuristicPolicy
        policy = HeuristicPolicy()
        active_policy_type = "heuristic"
    else:
        policy, active_policy_type = load_policy(policy_path, policy_type, gnn_state_space, config, device="cuda")

    # Run Instrumented Environment
    # Enable plotting to populate truck_routes for accurate event timing
    env = InstrumentedEnv(config=copy.deepcopy(config), verbose=False, enable_plotting=True)
    
    print(f"Running scenario with seed {SEED}...")
    obs, info = env.reset(seed=SEED)
    done = truncated = False
    episode_steps = 0
    
    # Recreate optimal planner per episode to avoid stale plans
    episode_policy = OptimalGurobiPolicy(verbose=False) if active_policy_type == "optimal" else policy
    
    while not (done or truncated):
        if active_policy_type == "optimal" or active_policy_type == "heuristic":
            # Baseline policies don't need GNN state
            action = episode_policy.get_action(env)
        else:
            gnn_state = gnn_state_space.get_state_GNN(env)
            
            if active_policy_type == "ppo-variable" or active_policy_type == "variable-ppo":
                raw_action = policy.select_action(gnn_state, deterministic=True)
                action = policy.to_env_action(gnn_state, int(raw_action))
            else:
                mask = torch.tensor(compute_action_mask(env), dtype=torch.bool)
                raw_action = policy.select_action(gnn_state, deterministic=True, action_mask=mask)
                if isinstance(raw_action, tuple):
                    action = raw_action
                else:
                    action = int(raw_action) % env.action_space.n

        obs, reward, done, truncated, info = env.step(action)
        episode_steps += 1

    print(f"Scenario finished. Reward: {env.episode_reward:.2f}")
    print(f"Total Steps: {episode_steps}")
    print(f"Success: {'Yes' if info['all_complete'] else 'No'}")
    return (
        env.history,
        config["environment"]["max_time"],
        env.charging_nodes,
        episode_steps,
        env,
        active_policy_type,
    )

def process_truck_routes(env):
    """
    Process truck_routes from environment into timeline segments per truck.
    
    Combines truck_routes (for accurate routing timing) with history 
    (for charging and waiting states).
    
    truck_routes format: (node, time, event_type, soc_at_arrival)
    
    Args:
        env: EventDrivenTruckEnv with truck_routes and history populated
        
    Returns:
        dict: truck_id -> list of timeline segments
    """
    truck_timelines = {}
    
    # First, build routing segments from truck_routes (accurate timing)
    for truck_id, route_events in env.truck_routes.items():
        timeline = []
        prev_time = 0.0
        
        for i, event in enumerate(route_events):
            # Handle both old 3-tuple and new 4-tuple formats
            if len(event) == 4:
                node, time, event_type, soc = event
            else:
                node, time, event_type = event
                soc = None
            
            if event_type == "start":
                prev_time = time
                continue
            
            # Create routing segment from previous time to this arrival
            if i > 0 and time > prev_time:
                timeline.append({
                    "start": prev_time,
                    "end": time,
                    "state": "routing",
                    "meta": {"destination": node},
                    "end_soc": soc
                })
            
            prev_time = time
        
        truck_timelines[truck_id] = timeline
    
    # Now merge in charging and waiting segments from history
    # History has these states but with timing issues for routing
    # We only extract charging/waiting states
    if hasattr(env, 'history') and env.history:
        # First collect all charging/waiting segments per truck
        temp_segments = {}  # truck_id -> list of segments
        
        for entry in env.history:
            start = entry["start"]
            end = entry["end"]
            trucks_info = entry["trucks"]
            
            for truck_id, details in trucks_info.items():
                state = details.get("state", "unknown")
                
                # Only add charging and waiting states from history
                if state in ["charging", "waiting_to_charge"]:
                    meta = {}
                    if state in ["charging", "waiting_to_charge"]:
                        meta["current_node"] = details.get("current_node")
                    
                    start_soc = details.get("start_soc")
                    end_soc = details.get("end_soc")
                    
                    if truck_id not in temp_segments:
                        temp_segments[truck_id] = []
                    
                    temp_segments[truck_id].append({
                        "start": start,
                        "end": end,
                        "state": state,
                        "meta": meta,
                        "start_soc": start_soc,
                        "end_soc": end_soc
                    })
        
        # Now merge consecutive segments with same state and node
        for truck_id, segments in temp_segments.items():
            if not segments:
                continue
            
            # Sort by start time
            segments.sort(key=lambda x: x["start"])
            
            # Merge consecutive segments
            merged = []
            current = segments[0]
            
            for i in range(1, len(segments)):
                next_seg = segments[i]
                
                # Check if we should merge: same state, same node, consecutive time
                if (current["state"] == next_seg["state"] and 
                    current["meta"].get("current_node") == next_seg["meta"].get("current_node") and
                    abs(current["end"] - next_seg["start"]) < 0.01):
                    # Merge: extend end time and update end_soc
                    current["end"] = next_seg["end"]
                    current["end_soc"] = next_seg["end_soc"]
                else:
                    # Not mergeable, save current and move to next
                    merged.append(current)
                    current = next_seg
            
            # Don't forget the last segment
            merged.append(current)
            
            # Now insert merged segments into timeline
            timeline = truck_timelines.get(truck_id, [])
            for seg in merged:
                # Find where to insert (maintain chronological order)
                insert_idx = len(timeline)
                for idx, existing_seg in enumerate(timeline):
                    if existing_seg["start"] >= seg["start"]:
                        insert_idx = idx
                        break
                
                timeline.insert(insert_idx, seg)
            
            truck_timelines[truck_id] = timeline
    
    # Add final state if truck is complete or failed
    for truck_id in truck_timelines:
        truck = env.trucks[truck_id]
        timeline = truck_timelines[truck_id]
        
        if timeline:
            prev_time = timeline[-1]["end"]
        else:
            prev_time = 0.0
        
        if truck.is_complete:
            timeline.append({
                "start": prev_time,
                "end": prev_time,
                "state": "complete",
                "meta": {},
                "end_soc": truck.get_battery_percentage()
            })
        elif truck.failed:
            timeline.append({
                "start": prev_time,
                "end": prev_time,
                "state": "failed",
                "meta": {},
                "end_soc": truck.get_battery_percentage()
            })
    
    return truck_timelines

def process_history(history):
    """Process raw history into timeline segments per truck.
    
    DEPRECATED: This has timing issues due to global_clock advancing during
    event processing. Use process_truck_routes() instead for accurate timing.
    """
    truck_ids = sorted(list(history[0]["trucks"].keys()))
    truck_timelines = {tid: [] for tid in truck_ids}
    
    for entry in history:
        start = entry["start"]
        end = entry["end"]
        trucks_info = entry["trucks"]
        
        for tid in truck_ids:
            details = trucks_info.get(tid, {})
            state = details.get("state", "unknown")
            meta = {}
            if state == "routing":
                meta["destination"] = details.get("destination")
            elif state in ["charging", "waiting_to_charge"]:
                meta["current_node"] = details.get("current_node")
            
            # Merge segments
            if truck_timelines[tid]:
                last = truck_timelines[tid][-1]
                if last["state"] == state and last["meta"] == meta and abs(last["end"] - start) < 1e-6:
                    last["end"] = end
                    last["end_soc"] = details.get("end_soc")
                    continue
            
            truck_timelines[tid].append({
                "start": start,
                "end": end,
                "state": state,
                "meta": meta,
                "end_soc": details.get("end_soc")
            })
    return truck_timelines, truck_ids


def generate_schedule_log(histories, envs, policy_names, output_dir):
    """Generate detailed schedule log file comparing all policies."""
    
    log_path = os.path.join(output_dir, "schedule_comparison_log.txt")
    
    with open(log_path, 'w') as f:
        f.write("="*120 + "\n")
        f.write("TRUCK SCHEDULE COMPARISON LOG\n")
        f.write("="*120 + "\n")
        f.write(f"Configuration: {NUM_TRUCKS} trucks, {NUM_STOPS} stops, Seed: {SEED}\n")
        f.write(f"Policies: {', '.join(policy_names)}\n")
        f.write("="*120 + "\n\n")
        
        # Use truck_routes instead of history for accurate event times
        # truck_routes is populated directly by event handler with correct event.time
        all_schedules = {}
        for env, policy_name in zip(envs, policy_names):
            schedules = {}
            
            # Get truck_routes from environment
            truck_routes = env.truck_routes  # truck_id -> list of (node, time, event_type)
            
            for truck_id, route in truck_routes.items():
                schedule = []
                truck = env.trucks[truck_id]
                
                # Process route events (handle both 3-tuple and 4-tuple formats)
                for event in route:
                    if len(event) == 4:
                        node, time, event_type, soc = event
                    else:
                        node, time, event_type = event
                        # Fallback: get SoC from current state if this is the current node
                        soc = truck.get_battery_percentage() if truck.current_node == node else None
                    
                    if event_type == "start":
                        continue  # Skip start event
                    
                    if event_type in ["delivery", "charger", "travel"]:
                        schedule.append({
                            "time": time,
                            "node": node,
                            "soc": soc,
                            "event": "arrive"
                        })
                
                # TODO: Add charging events from history if needed
                # For now, just show arrivals which are the most important
                
                schedules[truck_id] = schedule
            all_schedules[policy_name] = schedules
        
        # Get all truck IDs
        truck_ids = sorted(list(all_schedules[policy_names[0]].keys()))
        
        # Write detailed comparison for each truck
        for truck_id in truck_ids:
            f.write("\n" + "="*120 + "\n")
            f.write(f"TRUCK {truck_id}\n")
            f.write("="*120 + "\n\n")
            
            for policy_name in policy_names:
                f.write(f"\n--- {policy_name} ---\n")
                f.write(f"{'Time (h)':<12} {'Event':<25} {'Node':<10} {'SoC (%)':<10}\n")
                f.write("-"*60 + "\n")
                
                schedule = all_schedules[policy_name].get(truck_id, [])
                
                if not schedule:
                    f.write("  No events recorded\n")
                else:
                    for event in schedule:
                        time_str = f"{event['time']:.2f}"
                        event_str = event['event'].replace('_', ' ').title()
                        node_str = str(event['node']) if event['node'] is not None else "N/A"
                        soc_str = f"{event['soc']:.1f}" if event['soc'] is not None else "N/A"
                        
                        f.write(f"{time_str:<12} {event_str:<25} {node_str:<10} {soc_str:<10}\n")
                
                # Summary statistics from environment
                env = envs[policy_names.index(policy_name)]
                truck = next((t for t in env.trucks if t.truck_id == truck_id), None)
                
                if truck and schedule:
                    total_time = schedule[-1]['time'] if schedule else 0
                    final_soc = schedule[-1]['soc'] if schedule and schedule[-1]['soc'] is not None else 0
                    num_arrivals = len([e for e in schedule if e['event'] == 'arrive'])
                    num_charges = len([e for e in schedule if e['event'] == 'start_charging'])
                    
                    f.write(f"\nSummary:\n")
                    f.write(f"  Total Time: {total_time:.2f} hours\n")
                    f.write(f"  Final SoC: {final_soc:.1f}%\n")
                    f.write(f"  Total Distance: {truck.total_distance_traveled:.2f} km\n")
                    f.write(f"  Total Charging Time: {truck.total_charging_time:.2f} hours\n")
                    f.write(f"  Node Arrivals: {num_arrivals}\n")
                    f.write(f"  Charging Sessions: {num_charges}\n")
                    f.write(f"  Deliveries Remaining: {len(truck.get_remaining_deliveries())}\n")
                    f.write(f"  Status: {'Complete' if truck.is_complete else 'Failed' if truck.failed else 'Incomplete'}\n")
        
        # Overall comparison summary
        f.write("\n\n" + "="*120 + "\n")
        f.write("OVERALL COMPARISON SUMMARY\n")
        f.write("="*120 + "\n\n")
        
        for env, policy_name in zip(envs, policy_names):
            f.write(f"\n{policy_name}:\n")
            f.write(f"  Episode Reward: {env.episode_reward:.2f}\n")
            f.write(f"  Completion Time: {env.global_clock:.2f} hours\n")
            
            # Calculate aggregate statistics
            all_complete = all(len(t.get_remaining_deliveries()) == 0 for t in env.trucks)
            total_distance = sum(truck.total_distance_traveled for truck in env.trucks)
            total_charging_time = sum(truck.total_charging_time for truck in env.trucks)
            total_deliveries = sum(len(t.delivery_sequence) - 1 - len(t.get_remaining_deliveries()) for t in env.trucks)
            num_failed = sum(1 for t in env.trucks if t.failed)
            
            f.write(f"  All Deliveries Complete: {'Yes' if all_complete else 'No'}\n")
            f.write(f"  Failed Trucks: {num_failed}/{len(env.trucks)}\n")
            f.write(f"  Total Distance: {total_distance:.2f} km\n")
            f.write(f"  Total Charging Time: {total_charging_time:.2f} hours\n")
            f.write(f"  Total Deliveries Completed: {total_deliveries}\n")
            f.write(f"  Avg Distance per Truck: {total_distance/len(env.trucks):.2f} km\n")
            f.write(f"  Avg Charging Time per Truck: {total_charging_time/len(env.trucks):.2f} hours\n")
    
    print(f"  ✓ Schedule comparison log saved to: {log_path}")
    return log_path

def plot_comparison(histories, envs, policy_names, max_time, charging_nodes):
    """Generate Gantt chart comparing multiple policies (3 policies as stacked lines per truck).
    
    Uses truck_routes from envs for accurate event timing instead of history which has interval issues.
    """
    
    colors = {
        "routing": "#3498db",      # Blue
        "charging": "#2ecc71",     # Green
        "waiting_to_charge": "#e74c3c", # Red
        "ready": "#95a5a6",        # Gray
        "complete": "#f1c40f",     # Gold
        "failed": "#34495e"        # Dark Blue/Black
    }
    
    # Process truck_routes from envs for accurate timing
    timelines_list = []
    truck_ids = None
    for env in envs:
        timeline = process_truck_routes(env)
        timelines_list.append(timeline)
        if truck_ids is None:
            truck_ids = sorted(timeline.keys())
    
    # Calculate actual maximum time needed across all policies
    actual_max_time = 0
    for timelines in timelines_list:
        for tid in truck_ids:
            for segment in timelines[tid]:
                actual_max_time = max(actual_max_time, segment["end"])
    
    # Add small padding (5%) for better visualization
    actual_max_time = actual_max_time * 1.05
    
    # Increase figure height to accommodate triple lines
    fig, ax = plt.subplots(figsize=(20, 14))
    
    # Y-axis positions: Truck ID i -> Policy 1 at i+0.2, Policy 2 at i, Policy 3 at i-0.2
    offsets = [0.2, 0, -0.2]
    
    for tid in truck_ids:
        # Draw background track
        ax.hlines(y=tid, xmin=0, xmax=actual_max_time, colors='gray', linestyles=':', alpha=0.1, linewidth=25)
        
        # Plot each policy in two passes: routing first, then charging/waiting on top
        for idx, (timelines, policy_name, offset) in enumerate(zip(timelines_list, policy_names, offsets)):
            y_pos = tid + offset
            
            # First pass: draw routing segments
            for segment in timelines[tid]:
                if segment["state"] == "routing":
                    _plot_segment(ax, segment, y_pos, colors, charging_nodes, label_soc=True)
            
            # Second pass: draw charging and waiting on top
            for segment in timelines[tid]:
                if segment["state"] in ["charging", "waiting_to_charge", "complete", "failed"]:
                    _plot_segment(ax, segment, y_pos, colors, charging_nodes, label_soc=True)
            
            # Add label for policy
            ax.text(-1.0, y_pos, policy_name, ha='right', va='center', fontsize=8, color='gray', weight='bold')

    # Formatting
    ax.set_xlabel("Simulation Time (hours)", fontsize=12)
    ax.set_ylabel("Truck ID", fontsize=12)
    ax.set_title(f"Schedule Comparison: {', '.join(policy_names)} - Seed {SEED}", fontsize=14)
    ax.set_xlim(0, actual_max_time)
    ax.set_yticks(truck_ids)
    ax.set_yticklabels([f"Truck {tid}" for tid in truck_ids])
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)
    
    # Legend
    legend_elements = [
        mpatches.Patch(color=colors["routing"], label='Routing'),
        mpatches.Patch(color=colors["charging"], label='Charging'),
        mpatches.Patch(color=colors["waiting_to_charge"], label='Waiting'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='white', markeredgecolor=colors["charging"], markersize=8, markeredgewidth=2, label='Charger Node'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='white', markeredgecolor=colors["routing"], markersize=8, markeredgewidth=2, label='Delivery Node'),
        plt.Line2D([0], [0], marker='*', color='gold', markerfacecolor='gold', markeredgecolor='black', markersize=12, label='Completed'),
        plt.Line2D([0], [0], marker='X', color='red', markerfacecolor='red', markeredgecolor='black', markersize=10, label='Failed'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.1, 1), fontsize=10)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "schedule_comparison_3policies.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved to {save_path}")

def _plot_segment(ax, segment, y_pos, colors, charging_nodes, label_soc=False):
    state = segment["state"]
    start = segment["start"]
    end = segment["end"]
    meta = segment["meta"]
    end_soc = segment.get("end_soc")
    start_soc = segment.get("start_soc")
    
    color = colors.get(state, "black")
    
    # Use different linewidths and z-orders for better visibility
    if state == "charging":
        linewidth = 6  # Thicker for charging
        zorder = 20  # Draw on top
    elif state == "waiting_to_charge":
        linewidth = 5  # Thick for waiting
        zorder = 19
    elif state == "routing":
        linewidth = 2.5
        zorder = 10
    else:
        linewidth = 1.0
        zorder = 5
    
    ax.hlines(y=y_pos, xmin=start, xmax=end, colors=color, linewidth=linewidth, zorder=zorder)
    
    # For charging: show duration above the bar and SoC at start and end
    if state == "charging":
        duration = end - start
        if duration > 0:
            # Show duration above the charging bar
            mid_point = (start + end) / 2
            ax.text(mid_point, y_pos + 0.12, f"{duration:.1f}h", ha='center', va='bottom', 
                    fontsize=7, color='black', weight='bold', bbox=dict(boxstyle='round,pad=0.3', 
                    facecolor='white', edgecolor='green', alpha=0.8))
            
            # Show SoC at start and end
            if label_soc:
                if start_soc is not None:
                    ax.text(start, y_pos - 0.08, f"{start_soc:.0f}%", ha='center', va='top', 
                            fontsize=6, color='green', alpha=0.9, weight='bold')
                if end_soc is not None:
                    ax.text(end, y_pos - 0.08, f"{end_soc:.0f}%", ha='center', va='top', 
                            fontsize=6, color='green', alpha=0.9, weight='bold')
    
    # For other states: show SoC only at the end (arrival points)
    elif label_soc and end_soc is not None and state not in ["waiting_to_charge", "complete", "failed"]:
        ax.text(end, y_pos - 0.08, f"{end_soc:.0f}%", ha='center', va='top', fontsize=5, color='black', alpha=0.7)

    if state == "routing":
        dest = meta.get("destination")
        if dest is not None:
            is_charger = dest in charging_nodes
            marker = 'o' if is_charger else 's'
            ax.plot(end, y_pos, marker=marker, markersize=6, 
                    markerfacecolor='white', markeredgecolor=color, markeredgewidth=1.5, zorder=10)
            ax.text(end, y_pos + 0.1, str(dest), ha='center', va='bottom', fontsize=6, fontweight='bold', color=color)
    
    elif state == "complete":
        ax.plot(start, y_pos, marker='*', markersize=10, 
                markerfacecolor='gold', markeredgecolor='black', markeredgewidth=0.5, zorder=15)
    elif state == "failed":
        ax.plot(start, y_pos, marker='X', markersize=8, 
                markerfacecolor='red', markeredgecolor='black', markeredgewidth=0.5, zorder=15)

def plot_charger_queue_comparison(envs, policy_names, max_time, output_dir):
    """Generate comparison of charger queue dynamics across multiple policies."""
    
    # Find all chargers that were visited by any policy
    all_visited_chargers = set()
    for env in envs:
        for node in env.charging_station.charging_nodes:
            history = env.charging_station.queue_history[node]
            if len(history["truck_events"]) > 0:
                all_visited_chargers.add(node)
    
    if not all_visited_chargers:
        print("  ⚠ No charging stations were visited during any simulation")
        return
    
    # Sort chargers by total activity across all policies
    charger_activity = {}
    for node in all_visited_chargers:
        total_activity = 0
        for env in envs:
            history = env.charging_station.queue_history[node]
            total_activity += len(history["truck_events"])
        charger_activity[node] = total_activity
    
    visited_chargers = sorted(charger_activity.keys(), key=lambda x: charger_activity[x], reverse=True)
    
    # Limit to top 6 most active chargers
    max_chargers_to_plot = 6
    if len(visited_chargers) > max_chargers_to_plot:
        print(f"  📊 Plotting top {max_chargers_to_plot} most active chargers (out of {len(visited_chargers)})")
        visited_chargers = visited_chargers[:max_chargers_to_plot]
    
    n_chargers = len(visited_chargers)
    n_policies = len(policy_names)
    
    # Create subplots: rows = chargers, columns = policies
    fig, axes = plt.subplots(n_chargers, n_policies, 
                            figsize=(7 * n_policies, 3.5 * n_chargers), 
                            dpi=150, squeeze=False)
    
    # Overall title
    fig.suptitle('Charging Station Queue Dynamics Comparison', 
                fontsize=18, fontweight='bold', y=0.998)
    
    # Colors for consistency
    colors = {
        'occupancy': '#2E86AB',
        'waitlist': '#F24236',
        'capacity': '#27AE60',
        'events': {'arrive': '#FF6B35', 'start': '#004E89', 'finish': '#9B59B6'}
    }
    
    for row_idx, charger_node in enumerate(visited_chargers):
        for col_idx, (env, policy_name) in enumerate(zip(envs, policy_names)):
            ax = axes[row_idx, col_idx]
            
            charging_station = env.charging_station
            history = charging_station.queue_history[charger_node]
            
            # Get charger info
            charger_type = charging_station.charger_type[charger_node]
            capacity = int(charging_station.charger_capacity[charger_node])
            stats = charging_station.charger_stats[charger_node]
            
            # Extract time series data
            times = np.array(history["times"])
            occupancy = np.array(history["occupancy"])
            waitlist = np.array(history["waitlist"])
            truck_events = history["truck_events"]
            
            # Process truck charging events - merge consecutive charging sessions
            merged_sessions = []
            if len(truck_events) > 0:
                charging_sessions = []  # (truck_id, start_time, end_time)
                truck_charging_state = {}  # truck_id -> start_time
                
                for event_time, truck_id, event_type in truck_events:
                    if event_type == 'start':
                        truck_charging_state[truck_id] = event_time
                    elif event_type == 'finish' and truck_id in truck_charging_state:
                        start_time = truck_charging_state[truck_id]
                        charging_sessions.append((truck_id, start_time, event_time))
                        del truck_charging_state[truck_id]
                
                # Sort charging sessions by start time
                charging_sessions.sort(key=lambda x: x[1])
                
                # Merge consecutive charging sessions for the same truck
                i = 0
                while i < len(charging_sessions):
                    truck_id, start, end = charging_sessions[i]
                    
                    # Look ahead to merge consecutive sessions from the same truck
                    j = i + 1
                    while j < len(charging_sessions):
                        next_truck, next_start, next_end = charging_sessions[j]
                        if next_truck == truck_id and abs(next_start - end) < 0.01:
                            # Merge this session
                            end = next_end
                            j += 1
                        else:
                            break
                    
                    merged_sessions.append((truck_id, start, end))
                    i = j
            
            if len(times) == 0:
                # No activity for this charger in this policy
                ax.text(0.5, 0.5, 'No Activity', transform=ax.transAxes,
                       ha='center', va='center', fontsize=14, color='gray')
                ax.set_xlim(0, max_time)
                ax.set_ylim(0, capacity + 1)
            else:
                # Calculate statistics
                max_occupancy = occupancy.max()
                max_waitlist = waitlist.max()
                avg_occupancy = occupancy.mean()
                utilization = stats["occupancy_time"] / max_time if max_time > 0 else 0
                
                # Plot stacked area (original visualization)
                ax.fill_between(times, 0, occupancy, alpha=0.4, 
                               color=colors['occupancy'], step='post', linewidth=0)
                ax.fill_between(times, occupancy, occupancy + waitlist, alpha=0.4, 
                               color=colors['waitlist'], step='post', linewidth=0)
                
                # Plot step lines
                ax.step(times, occupancy, where='post', color='#1A5276', linewidth=2)
                ax.step(times, occupancy + waitlist, where='post', 
                       color='#A93226', linewidth=2, linestyle='--')
                
                # Capacity line
                ax.axhline(y=capacity, color=colors['capacity'], 
                          linestyle=':', linewidth=2.5, alpha=0.8)
                
                # Set limits
                y_max = max(capacity + 1, max_occupancy + max_waitlist + 1)
                ax.set_xlim(-1, max_time + 1)
                ax.set_ylim(0, y_max * 1.1)
            
            # Titles and labels
            if row_idx == 0:
                ax.set_title(f'{policy_name}', fontsize=12, fontweight='bold', pad=8)
            
            if col_idx == 0:
                ax.set_ylabel(f'Charger {charger_node}\n({charger_type})\nTrucks', 
                             fontsize=10, fontweight='bold')
            
            if row_idx == n_chargers - 1:
                ax.set_xlabel('Time (hours)', fontsize=10)
            
            # Grid
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            ax.set_axisbelow(True)
            
            # Add statistics text
            if len(times) > 0:
                # Use merged sessions count for accurate statistics
                actual_sessions = len(merged_sessions)
                stats_text = (f'Util: {utilization*100:.0f}% | '
                             f'Peak Queue: {int(max_waitlist)} | '
                             f'Sessions: {actual_sessions}')
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                       fontsize=8, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # Add a common legend at the bottom
    legend_elements = [
        mpatches.Patch(color=colors['occupancy'], alpha=0.4, label='Actively Charging'),
        mpatches.Patch(color=colors['waitlist'], alpha=0.4, label='Waiting in Queue'),
        plt.Line2D([0], [0], color=colors['capacity'], linestyle=':', linewidth=2.5, label='Capacity Limit')
    ]
    fig.legend(handles=legend_elements, loc='lower center', 
              bbox_to_anchor=(0.5, -0.01), ncol=3, fontsize=11, framealpha=0.95)
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.995])
    filepath = os.path.join(output_dir, "charger_queue_comparison.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"  ✓ Charger queue comparison plot saved to: {filepath}")


def main():
    print("="*80)
    print(f"VISUALIZING TRUCK SCHEDULES - Comparing {len(POLICIES)} Policies")
    print("="*80)
    print(f"Configuration: {NUM_TRUCKS} trucks, {NUM_STOPS} stops, {MAX_TIME}h max time")
    print(f"Seed: {SEED}")
    print()
    
    histories = []
    policy_names = []
    envs = []
    max_time = None
    charging_nodes = None
    
    # Run each policy
    for idx, (policy_path, policy_type) in enumerate(POLICIES, 1):
        print(f"\n{'='*80}")
        print(f"[{idx}/{len(POLICIES)}] Running Policy: {policy_path if policy_path not in ['optimal', 'heuristic'] else policy_path.upper()}")
        print(f"{'='*80}")
        
        hist, mt, cn, steps, env, active_type = run_scenario(policy_path, policy_type)
        
        histories.append(hist)
        envs.append(env)
        
        # Generate readable policy name
        if policy_path == "heuristic":
            name = "Heuristic"
        elif policy_path == "optimal":
            name = "Optimal"
        else:
            # Extract short name from path
            base = os.path.basename(policy_path.rstrip('/'))
            if len(base) > 20:
                name = base[:20]
            else:
                name = base
        policy_names.append(name)
        
        if max_time is None:
            max_time = mt
            charging_nodes = cn
    
    print(f"\n{'='*80}")
    print("GENERATING COMPARISON PLOT")
    print(f"{'='*80}")
    
    # Plot Comparison (pass envs for accurate truck_routes data)
    plot_comparison(histories, envs, policy_names, max_time, charging_nodes)
    
    # Plot Queue Dynamics Comparison
    print(f"\nGenerating charger queue dynamics comparison...")
    plot_charger_queue_comparison(envs, policy_names, max_time, OUTPUT_DIR)
    
    # Generate Schedule Log
    print(f"\nGenerating detailed schedule comparison log...")
    generate_schedule_log(histories, envs, policy_names, OUTPUT_DIR)
    
    print(f"\n{'='*80}")
    print("VISUALIZATION COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {OUTPUT_DIR}/")
    print()

if __name__ == "__main__":
    main()
