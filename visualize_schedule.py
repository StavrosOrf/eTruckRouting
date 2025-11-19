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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from truck_env.models.event_driven_env import EventDrivenTruckEnv
from truck_env.state.gnn_state_space import GNNStateSpace
from truck_env.utils.utils import load_config
from truck_env.utils.plotter import EnvironmentPlotter
from train import compute_action_mask
from algo.policy_utils import load_policy

# ============ CONFIGURATION ============
# POLICY_PATH = "saved_models/ppo-variable_steps=128_epochs=10_ent=0.1_seed=0_gnnhd=64_mlphd=256"
POLICY_PATH = "saved_models/ppo-variable_steps=128_epochs=10_ent=0.1_seed=0_gnnhd=64_mlphd=64/"
POLICY_TYPE = "ppo-variable"  # or "heuristic" or "ppo"
# POLICY_TYPE = "heuristic"  # or "heuristic" or "ppo"
CONFIG_FILE = "truck_env/config_files/config.yaml"
NUM_TRUCKS = 10
NUM_STOPS = 3
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
        
        # Capture detailed state for each truck
        truck_details = {}
        for truck in self.trucks:
            state = self.truck_states.get(truck.truck_id, "unknown")
            details = {
                "state": state,
                "current_node": int(truck.current_node),
                "destination": int(truck.route_destination) if truck.route_destination is not None else None,
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

def run_scenario():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Setup
    config = load_config(CONFIG_FILE)
    config["environment"]["num_trucks"] = NUM_TRUCKS
    config["environment"]["num_stops"] = NUM_STOPS
    
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
    print(f"Loading policy: {POLICY_PATH} (requested {POLICY_TYPE})...")
    policy, active_policy_type = load_policy(POLICY_PATH, POLICY_TYPE, gnn_state_space, config)

    # Run Instrumented Environment
    env = InstrumentedEnv(config=copy.deepcopy(config), verbose=False, enable_plotting=False)
    
    print(f"Running scenario with seed {SEED}...")
    obs, info = env.reset(seed=SEED)
    done = truncated = False
    episode_steps = 0
    
    while not (done or truncated) and episode_steps < 200:
        gnn_state = gnn_state_space.get_state_GNN(env)

        if active_policy_type == "heuristic":
            action = policy.get_action(env)
        elif active_policy_type == "ppo-variable":
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
    return (
        env.history,
        config["environment"]["max_time"],
        env.charging_nodes,
        episode_steps,
        env,
        active_policy_type,
    )

def plot_gantt(history, max_time, charging_nodes, total_steps, policy_label):
    """Generate Timeline chart from history."""
    
    # Define colors for states
    colors = {
        "routing": "#3498db",      # Blue
        "charging": "#2ecc71",     # Green
        "waiting_to_charge": "#e74c3c", # Red
        "ready": "#95a5a6",        # Gray
        "complete": "#f1c40f",     # Gold
        "failed": "#34495e"        # Dark Blue/Black
    }
    
    # Prepare plot
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Get list of trucks
    truck_ids = sorted(list(history[0]["trucks"].keys()))
    
    # Process history per truck to merge segments
    truck_timelines = {tid: [] for tid in truck_ids}
    
    for entry in history:
        start = entry["start"]
        end = entry["end"]
        trucks_info = entry["trucks"]
        
        for tid in truck_ids:
            details = trucks_info.get(tid, {})
            state = details.get("state", "unknown")
            # relevant details for merging
            meta = {}
            if state == "routing":
                meta["destination"] = details.get("destination")
            elif state in ["charging", "waiting_to_charge"]:
                meta["current_node"] = details.get("current_node")
            
            # Check if we can merge with last segment
            if truck_timelines[tid]:
                last = truck_timelines[tid][-1]
                # Only merge if state and meta match. 
                # Note: We might lose intermediate SoC points if we merge. 
                # If we want SoC at every event finish, we should probably NOT merge if we want to show all points.
                # However, the user said "after every truck event finish".
                # If we merge, we only have the end of the merged segment.
                # Let's keep merging for visual cleanliness, but update the end_soc of the merged segment.
                if last["state"] == state and last["meta"] == meta and abs(last["end"] - start) < 1e-6:
                    last["end"] = end # Extend
                    last["end_soc"] = details.get("end_soc") # Update end SoC
                    continue
            
            # Add new segment
            truck_timelines[tid].append({
                "start": start,
                "end": end,
                "state": state,
                "meta": meta,
                "end_soc": details.get("end_soc")
            })

    # Plotting
    for tid in truck_ids:
        timeline = truck_timelines[tid]
        
        # Draw the base line (track) for the truck
        ax.hlines(y=tid, xmin=0, xmax=max_time, colors='gray', linestyles=':', alpha=0.2, linewidth=1)

        for segment in timeline:
            state = segment["state"]
            start = segment["start"]
            end = segment["end"]
            meta = segment["meta"]
            end_soc = segment.get("end_soc")
            
            color = colors.get(state, "black")
            # Thinner lines as requested
            linewidth = 3 if state in ["charging", "routing", "waiting_to_charge"] else 1.5
            
            # Plot line segment
            ax.hlines(y=tid, xmin=start, xmax=end, colors=color, linewidth=linewidth)
            
            # Plot SoC under the line at the end of the segment
            if end_soc is not None:
                ax.text(end, tid - 0.15, f"{end_soc:.0f}%", ha='center', va='top', fontsize=6, color='black', alpha=0.7)

            # Add markers/labels for specific events
            if state == "routing":
                # End of routing is an arrival
                dest = meta.get("destination")
                if dest is not None:
                    is_charger = dest in charging_nodes
                    marker = 'o' if is_charger else 's' # Circle for charger, Square for delivery
                    
                    # Plot marker at the end
                    ax.plot(end, tid, marker=marker, markersize=8, 
                            markerfacecolor='white', markeredgecolor=color, markeredgewidth=2, zorder=10)
                    
                    # Label the node
                    ax.text(end, tid + 0.2, str(dest), ha='center', va='bottom', fontsize=8, fontweight='bold', color=color)
            
            # Explicitly mark completion and failure
            elif state == "complete":
                # Mark the start of completion
                ax.plot(start, tid, marker='*', markersize=12, 
                        markerfacecolor='gold', markeredgecolor='black', markeredgewidth=1, zorder=15)
            elif state == "failed":
                # Mark the start of failure
                ax.plot(start, tid, marker='X', markersize=10, 
                        markerfacecolor='red', markeredgecolor='black', markeredgewidth=1, zorder=15)

    # Formatting
    ax.set_xlabel("Simulation Time (hours)")
    ax.set_ylabel("Truck ID")
    ax.set_title(f"Truck Schedule - {policy_label} (Seed {SEED}) - Total Steps: {total_steps}")
    ax.set_xlim(0, max_time)
    ax.set_yticks(truck_ids)
    ax.set_yticklabels([f"Truck {tid}" for tid in truck_ids])
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)
    
    # Custom Legend
    legend_elements = [
        mpatches.Patch(color=colors["routing"], label='Routing'),
        mpatches.Patch(color=colors["charging"], label='Charging'),
        mpatches.Patch(color=colors["waiting_to_charge"], label='Waiting'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='white', markeredgecolor=colors["charging"], markersize=8, markeredgewidth=2, label='Charger Node'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='white', markeredgecolor=colors["routing"], markersize=8, markeredgewidth=2, label='Delivery Node'),
        plt.Line2D([0], [0], marker='*', color='gold', markerfacecolor='gold', markeredgecolor='black', markersize=12, label='Completed'),
        plt.Line2D([0], [0], marker='X', color='red', markerfacecolor='red', markeredgecolor='black', markersize=10, label='Failed'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "schedule_timeline.png")
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to {save_path}")

def main():
    history, max_time, charging_nodes, total_steps, env, policy_label = run_scenario()
    plot_gantt(history, max_time, charging_nodes, total_steps, policy_label)
    
    # Plot queue dynamics
    print("Plotting queue dynamics...")
    plotter = EnvironmentPlotter(OUTPUT_DIR, verbose=False, use_osm=False)
    plotter.plot_charger_queue_dynamics(env.charging_station, env.transport_graph, max_time)

if __name__ == "__main__":
    main()
