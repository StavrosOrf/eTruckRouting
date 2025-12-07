"""
Visualize truck schedules (Gantt chart) using truck event logs.

This version uses the built-in truck event monitoring system to directly
extract accurate timelines without postprocessing event queues.
"""
import os
import sys
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shutil
import traceback
import torch

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.state.gnn_state_space import GNNStateSpace
from EVRoutingEnv.utils.utils import load_config
from EVRoutingEnv.baselines.optimal_gurobi import OptimalGurobiPolicy
from scripts.training.train_PPO_Variable import compute_action_mask
from algo.policy_utils import load_policy

# ============ CONFIGURATION ============
# Policies to compare
POLICIES = [
    ("saved_models/NewFeasibleSpace_FixedGraph_ppo-variable_steps=1024_epochs=5_ent=0.1_seed=0_gnnhd=32_mlphd=256/", "variable-ppo"),
    ("optimal", "optimal"),
    # ("heuristic", "heuristic"),
]

CONFIG_FILE = "EVRoutingEnv/config_files/config.yaml"
NUM_TRUCKS = 10
NUM_STOPS = 5
MAX_TIME = 200.0
SEED = 1005
OUTPUT_DIR = "results/visualization"
# =======================================


def run_scenario(policy_path, policy_type):
    """Run a single scenario with the given policy and collect truck event logs."""
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

    # Run Environment
    env = EventDrivenTruckEnv(config=copy.deepcopy(config), verbose=False, enable_plotting=False, run_id="visualization_temp")
    
    print(f"Running scenario with seed {SEED}...")
    obs, info = env.reset(seed=SEED)
    
    done = truncated = False
    episode_steps = 0
    
    # Recreate optimal planner per episode to avoid stale plans
    episode_policy = OptimalGurobiPolicy(verbose=False) if active_policy_type == "optimal" else policy
    
    while not (done or truncated):
        if active_policy_type == "optimal" or active_policy_type == "heuristic":
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
    
    # Clean up temporary output directory
    if hasattr(env, 'output_dir') and os.path.exists(env.output_dir):
        try:
            shutil.rmtree(env.output_dir)
        except Exception as e:
            print(f"Warning: Could not remove temporary directory {env.output_dir}: {e}")
    
    return env, active_policy_type


def extract_timeline_from_truck(truck):
    """
    Extract timeline segments directly from truck's event history.
    
    Returns:
        list: Timeline segments with start, end, state, and metadata
    """
    timeline = []
    events = truck.event_history
    
    # Pair up start/end events to create timeline segments
    i = 0
    while i < len(events):
        event = events[i]
        event_type = event["event_type"]
        
        # Map event types to timeline states
        if event_type.endswith("_START"):
            # Find matching end event
            activity_type = event_type.replace("_START", "")
            end_type = activity_type + "_END"
            
            # Search for matching end event
            end_event = None
            for j in range(i + 1, len(events)):
                if events[j]["event_type"] == end_type:
                    end_event = events[j]
                    break
            
            if end_event:
                # Create timeline segment
                state_map = {
                    "ROUTING": "routing",
                    "CHARGING": "charging",
                    "UNLOADING": "unloading",
                    "WAITING": "waiting_to_charge"
                }
                
                state = state_map.get(activity_type, activity_type.lower())
                
                segment = {
                    "start": event["timestamp"],
                    "end": end_event["timestamp"],
                    "state": state,
                    "start_soc": event["battery_soc"],
                    "end_soc": end_event["battery_soc"],
                    "location": event["location"],
                    "details": {**event["details"], **end_event["details"]}
                }
                
                timeline.append(segment)
        
        i += 1
    
    # Add terminal states (complete/failed)
    if truck.is_complete:
        last_time = timeline[-1]["end"] if timeline else 0.0
        timeline.append({
            "start": last_time,
            "end": last_time,
            "state": "complete",
            "start_soc": truck.get_battery_percentage(),
            "end_soc": truck.get_battery_percentage(),
            "location": truck.current_node,
            "details": {}
        })
    elif truck.failed:
        last_time = timeline[-1]["end"] if timeline else 0.0
        timeline.append({
            "start": last_time,
            "end": last_time,
            "state": "failed",
            "start_soc": truck.get_battery_percentage(),
            "end_soc": truck.get_battery_percentage(),
            "location": truck.current_node,
            "details": {}
        })
    
    return timeline


def plot_comparison(envs, policy_names, max_time):
    """
    Generate Gantt chart comparing multiple policies using truck event logs.
    """
    colors = {
        "routing": "#3498db",
        "charging": "#2ecc71",
        "waiting_to_charge": "#e74c3c",
        "unloading": "#ff8c00",
        "ready": "#95a5a6",
        "complete": "#f1c40f",
        "failed": "#34495e"
    }
    
    # Extract timelines from truck event logs
    timelines_list = []
    truck_ids = None
    charging_nodes = set()
    
    for env in envs:
        timelines = {}
        for truck in env.trucks:
            timelines[truck.truck_id] = extract_timeline_from_truck(truck)
        timelines_list.append(timelines)
        
        if truck_ids is None:
            truck_ids = sorted(timelines.keys())
        
        # Collect charging nodes
        if hasattr(env, 'charging_nodes'):
            charging_nodes.update(env.charging_nodes)
    
    # Calculate actual maximum time
    actual_max_time = 0
    for timelines in timelines_list:
        for tid in truck_ids:
            for segment in timelines[tid]:
                actual_max_time = max(actual_max_time, segment["end"])
    
    actual_max_time = actual_max_time * 1.05  # Add 5% padding
    
    # Create figure
    length = len(truck_ids)
    fig, ax = plt.subplots(figsize=(20, length))
    
    # Y-axis positions: offset for each policy (increased spacing to avoid overlap)
    offsets = [0.35, 0, -0.35][:len(policy_names)]
    
    for tid in truck_ids:
        # Draw background track
        ax.hlines(y=tid, xmin=0, xmax=actual_max_time, colors='gray', 
                 linestyles=':', alpha=0.1, linewidth=35)
        
        # Plot each policy
        for idx, (timelines, policy_name, offset) in enumerate(zip(timelines_list, policy_names, offsets)):
            y_pos = tid + offset
            
            # Draw segments (routing first, then others on top)
            for segment in timelines[tid]:
                if segment["state"] == "routing":
                    _plot_segment(ax, segment, y_pos, colors, charging_nodes)
            
            for segment in timelines[tid]:
                if segment["state"] in ["charging", "unloading", "waiting_to_charge", "complete", "failed"]:
                    _plot_segment(ax, segment, y_pos, colors, charging_nodes)
            
            # Add policy label
            ax.text(-1.0, y_pos, policy_name, ha='right', va='center', 
                   fontsize=8, color='gray', weight='bold')
    
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
        mpatches.Patch(color=colors["unloading"], label='Unloading'),
        mpatches.Patch(color=colors["charging"], label='Charging'),
        mpatches.Patch(color=colors["waiting_to_charge"], label='Waiting in Queue'),
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='#e74c3c', 
                  markeredgecolor='darkred', markersize=6, markeredgewidth=1.5, label='Queue Wait Point'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='white', 
                  markeredgecolor=colors["charging"], markersize=8, markeredgewidth=2, label='Charger Node'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='white', 
                  markeredgecolor=colors["routing"], markersize=8, markeredgewidth=2, label='Delivery Node'),
        plt.Line2D([0], [0], marker='*', color='gold', markerfacecolor='gold', 
                  markeredgecolor='black', markersize=12, label='Completed'),
        plt.Line2D([0], [0], marker='X', color='red', markerfacecolor='red', 
                  markeredgecolor='black', markersize=10, label='Failed'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.1, 1), fontsize=10)
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "schedule_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved to {save_path}")


def _plot_segment(ax, segment, y_pos, colors, charging_nodes):
    """Plot a single timeline segment."""
    state = segment["state"]
    start = segment["start"]
    end = segment["end"]
    start_soc = segment.get("start_soc")
    end_soc = segment.get("end_soc")
    details = segment.get("details", {})
    location = segment.get("location")
    
    color = colors.get(state, "black")
    
    # Set linewidth and z-order based on state
    if state == "charging":
        linewidth, zorder = 6, 8
    elif state == "unloading":
        linewidth, zorder = 5.5, 8
    elif state == "waiting_to_charge":
        linewidth, zorder = 5, 19
    elif state == "routing":
        linewidth, zorder = 2.5, 10
    else:
        linewidth, zorder = 1.0, 5
    
    # Draw the line
    ax.hlines(y=y_pos, xmin=start, xmax=end, colors=color, linewidth=linewidth, zorder=zorder)
    
    duration = end - start
    
    # Add duration labels and markers based on state
    if state == "charging" and duration > 0:
        mid_point = (start + end) / 2
        ax.text(mid_point, y_pos + 0.12, f"{duration:.1f}h", ha='center', va='bottom',
               fontsize=4.5, color='black', weight='bold',
               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='green', alpha=0.8))
        
        # Show SoC at end only
        if end_soc is not None:
            ax.text(end, y_pos - 0.08, f"{end_soc:.0f}%", ha='center', va='top',
                   fontsize=4.5, color='green', alpha=0.9, weight='bold')
    
    elif state == "unloading" and duration > 0:
        # Place label right next to event end, small and compact
        label_x = end + 0.15
        ax.text(label_x, y_pos + 0.05, f"{duration:.1f}h", ha='left', va='bottom',
               fontsize=3.5, color='black', weight='bold', rotation=25,
               bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='#ff8c00', alpha=0.8))
    
    elif state == "routing" and duration > 0.1:
        mid_point = (start + end) / 2
        ax.text(mid_point, y_pos + 0.12, f"{duration:.1f}h", ha='center', va='bottom',
               fontsize=4.5, color='black', weight='bold',
               bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='#3498db', alpha=0.7))
        
        # Show destination node (higher z-order to be on top)
        destination = details.get("destination")
        if destination is not None:
            is_charger = destination in charging_nodes
            marker = 'o' if is_charger else 's'
            ax.plot(end, y_pos, marker=marker, markersize=6,
                   markerfacecolor='white', markeredgecolor=color, markeredgewidth=1.5, zorder=25)
            ax.text(end, y_pos + 0.1, str(destination), ha='center', va='bottom',
                   fontsize=6, fontweight='bold', color=color)
        
        # Show SoC at end
        if end_soc is not None:
            ax.text(end, y_pos - 0.08, f"{end_soc:.0f}%", ha='center', va='top',
                   fontsize=4.5, color='black', alpha=0.7)
    
    elif state == "waiting_to_charge" and duration > 0.001:
        mid_point = (start + end) / 2
        wait_label = f"{duration:.1f}h"
        ax.text(mid_point, y_pos + 0.12, wait_label, ha='center', va='bottom',
               fontsize=4.5, color='darkred', weight='bold',
               bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='#e74c3c', alpha=0.9))
    
    elif state == "complete":
        ax.plot(start, y_pos, marker='*', markersize=10,
               markerfacecolor='gold', markeredgecolor='black', markeredgewidth=0.5, zorder=15)
    
    elif state == "failed":
        ax.plot(start, y_pos, marker='X', markersize=8,
               markerfacecolor='red', markeredgecolor='black', markeredgewidth=0.5, zorder=15)


def generate_schedule_log(envs, policy_names, output_dir):
    """Generate detailed schedule log from truck event histories."""
    log_path = os.path.join(output_dir, "schedule_comparison_log.txt")
    
    with open(log_path, 'w') as f:
        f.write("="*120 + "\n")
        f.write("TRUCK SCHEDULE COMPARISON LOG (from Event Logs)\n")
        f.write("="*120 + "\n")
        f.write(f"Configuration: {NUM_TRUCKS} trucks, {NUM_STOPS} stops, Seed: {SEED}\n")
        f.write(f"Policies: {', '.join(policy_names)}\n")
        f.write("="*120 + "\n\n")
        
        # Get all truck IDs from first environment
        truck_ids = sorted([t.truck_id for t in envs[0].trucks])
        
        # Write detailed comparison for each truck
        for truck_id in truck_ids:
            f.write("\n" + "="*120 + "\n")
            f.write(f"TRUCK {truck_id}\n")
            f.write("="*120 + "\n\n")
            
            for policy_name, env in zip(policy_names, envs):
                truck = next(t for t in env.trucks if t.truck_id == truck_id)
                
                f.write(f"\n--- {policy_name} ---\n")
                f.write(f"{'Time (h)':<12} {'Event':<25} {'Location':<10} {'SoC (%)':<10} {'Details':<50}\n")
                f.write("-"*110 + "\n")
                
                # Get event log
                event_log = truck.export_event_log(format="dict")
                events = event_log["events"]
                
                if not events:
                    f.write("  No events recorded\n")
                else:
                    for event in events:
                        time_str = f"{event['timestamp']:.2f}"
                        event_type = event['event_type'].replace('_', ' ').title()
                        location = event['location']
                        soc = event['battery_soc']
                        
                        # Format key details
                        details = event['details']
                        detail_parts = []
                        if 'reason' in details:
                            detail_parts.append(f"Reason: {details['reason']}")
                        if 'destination' in details:
                            detail_parts.append(f"→ Node {details['destination']}")
                        if 'distance_km' in details:
                            detail_parts.append(f"{details['distance_km']:.1f}km")
                        if 'charge_amount_kwh' in details:
                            detail_parts.append(f"+{details['charge_amount_kwh']:.1f}kWh")
                        if 'charge_duration_hours' in details:
                            detail_parts.append(f"{details['charge_duration_hours']:.2f}h")
                        if 'unloading_duration_hours' in details:
                            detail_parts.append(f"{details['unloading_duration_hours']:.2f}h")
                        if 'wait_duration_hours' in details:
                            detail_parts.append(f"Waited: {details['wait_duration_hours']:.2f}h")
                        
                        detail_str = ", ".join(detail_parts)[:50]
                        
                        f.write(f"{time_str:<12} {event_type:<25} {location:<10} {soc:<10.1f} {detail_str:<50}\n")
                
                # Summary statistics
                summary = event_log["episode_summary"]
                breakdown = event_log["activity_breakdown"]
                
                f.write(f"\nSummary:\n")
                f.write(f"  Status: {'Complete' if summary['is_complete'] else 'Failed' if summary['failed'] else 'Incomplete'}\n")
                f.write(f"  Total Time: {summary['total_time_hours']:.2f} hours\n")
                f.write(f"  Total Distance: {summary['total_distance_km']:.2f} km\n")
                f.write(f"  Total Charging Time: {summary['total_charging_time_hours']:.2f} hours\n")
                f.write(f"  Total Unloading Time: {summary['total_unloading_time_hours']:.2f} hours\n")
                f.write(f"  Total Waiting Time: {summary['total_waiting_time_hours']:.2f} hours\n")
                f.write(f"  Charging Sessions: {summary['num_charging_sessions']}\n")
                f.write(f"  Final Battery: {summary['final_battery_soc']:.1f}%\n")
                
                f.write(f"\nActivity Breakdown:\n")
                for activity, duration in breakdown.items():
                    if activity != "total":
                        pct = (duration / breakdown['total'] * 100) if breakdown['total'] > 0 else 0
                        f.write(f"  {activity:12s}: {duration:6.2f}h ({pct:5.1f}%)\n")
        
        # Overall comparison summary
        f.write("\n\n" + "="*120 + "\n")
        f.write("OVERALL COMPARISON SUMMARY\n")
        f.write("="*120 + "\n\n")
        
        for policy_name, env in zip(policy_names, envs):
            f.write(f"\n{policy_name}:\n")
            f.write(f"  Episode Reward: {env.episode_reward:.2f}\n")
            f.write(f"  Completion Time: {env.global_clock:.2f} hours\n")
            
            # Aggregate statistics
            all_complete = all(t.is_complete for t in env.trucks)
            total_distance = sum(t.total_distance_traveled for t in env.trucks)
            total_charging_time = sum(t.total_charging_time for t in env.trucks)
            total_unloading_time = sum(t.total_unloading_time for t in env.trucks)
            total_waiting_time = sum(t.waiting_time for t in env.trucks)
            num_failed = sum(1 for t in env.trucks if t.failed)
            
            f.write(f"  All Deliveries Complete: {'Yes' if all_complete else 'No'}\n")
            f.write(f"  Failed Trucks: {num_failed}/{len(env.trucks)}\n")
            f.write(f"  Total Distance: {total_distance:.2f} km\n")
            f.write(f"  Total Charging Time: {total_charging_time:.2f} hours\n")
            f.write(f"  Total Unloading Time: {total_unloading_time:.2f} hours\n")
            f.write(f"  Total Waiting Time: {total_waiting_time:.2f} hours\n")
            f.write(f"  Avg Distance per Truck: {total_distance/len(env.trucks):.2f} km\n")
            f.write(f"  Avg Charging Time per Truck: {total_charging_time/len(env.trucks):.2f} hours\n")
    
    print(f"  ✓ Schedule comparison log saved to: {log_path}")
    return log_path


def main():
    print("="*80)
    print(f"VISUALIZING TRUCK SCHEDULES - Comparing {len(POLICIES)} Policies")
    print("="*80)
    print(f"Configuration: {NUM_TRUCKS} trucks, {NUM_STOPS} stops, {MAX_TIME}h max time")
    print(f"Seed: {SEED}")
    print()
    
    envs = []
    policy_names = []
    
    # Run each policy
    for idx, (policy_path, policy_type) in enumerate(POLICIES, 1):
        print(f"\n{'='*80}")
        print(f"[{idx}/{len(POLICIES)}] Running Policy: {policy_path if policy_path not in ['optimal', 'heuristic'] else policy_path.upper()}")
        print(f"{'='*80}")
        
        env, active_type = run_scenario(policy_path, policy_type)
        envs.append(env)
        
        # Generate readable policy name
        if policy_path == "heuristic":
            name = "Heuristic"
        elif policy_path == "optimal":
            name = "Optimal"
        else:
            base = os.path.basename(policy_path.rstrip('/'))
            name = base[:20] if len(base) > 20 else base
        policy_names.append(name)
    
    print(f"\n{'='*80}")
    print("GENERATING COMPARISON VISUALIZATIONS")
    print(f"{'='*80}")
    
    # Plot Comparison
    plot_comparison(envs, policy_names, MAX_TIME)
    
    # Generate Schedule Log
    print(f"\nGenerating detailed schedule comparison log...")
    generate_schedule_log(envs, policy_names, OUTPUT_DIR)
    
    print(f"\n{'='*80}")
    print("VISUALIZATION COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {OUTPUT_DIR}/")
    print()


if __name__ == "__main__":
    main()
