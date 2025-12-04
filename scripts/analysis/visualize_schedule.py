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
from scripts.training.train import compute_action_mask
from algo.policy_utils import load_policy

# ============ CONFIGURATION ============
POLICY_PATH = "saved_models/NewFeasibleSpace_FixedGraph_ppo-variable_steps=512_epochs=5_ent=0.1_seed=0_gnnhd=32_mlphd=256/"
CONFIG_FILE = "EVRoutingEnv/config_files/config.yaml"
NUM_TRUCKS = 10
NUM_STOPS = 6
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

def run_scenario(policy_type):
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
    print(f"Loading policy: {POLICY_PATH} (requested {policy_type})...")
    policy, active_policy_type = load_policy(POLICY_PATH, policy_type, gnn_state_space, config, device="cuda")

    # Run Instrumented Environment
    env = InstrumentedEnv(config=copy.deepcopy(config), verbose=False, enable_plotting=False)
    
    print(f"Running scenario with seed {SEED}...")
    obs, info = env.reset(seed=SEED)
    done = truncated = False
    episode_steps = 0
    
    while not (done or truncated):
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

def process_history(history):
    """Process raw history into timeline segments per truck."""
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

def plot_comparison(history_heuristic, history_ppo, max_time, charging_nodes):
    """Generate Comparison Timeline chart."""
    
    colors = {
        "routing": "#3498db",      # Blue
        "charging": "#2ecc71",     # Green
        "waiting_to_charge": "#e74c3c", # Red
        "ready": "#95a5a6",        # Gray
        "complete": "#f1c40f",     # Gold
        "failed": "#34495e"        # Dark Blue/Black
    }
    
    timelines_heuristic, truck_ids = process_history(history_heuristic)
    timelines_ppo, _ = process_history(history_ppo)
    
    # Increase figure height to accommodate double lines
    fig, ax = plt.subplots(figsize=(18, 12))
    
    # Y-axis positions: Truck ID i -> Heuristic at i-0.15, PPO at i+0.15
    offset = 0.15
    
    for tid in truck_ids:
        # Draw background track
        ax.hlines(y=tid, xmin=0, xmax=max_time, colors='gray', linestyles=':', alpha=0.1, linewidth=20)
        
        # Plot Heuristic (Top line)
        y_pos_h = tid + offset
        for segment in timelines_heuristic[tid]:
            _plot_segment(ax, segment, y_pos_h, colors, charging_nodes, label_soc=True)
            
        # Plot PPO (Bottom line)
        y_pos_p = tid - offset
        for segment in timelines_ppo[tid]:
            _plot_segment(ax, segment, y_pos_p, colors, charging_nodes, label_soc=True)
            
        # Add labels for policies
        ax.text(-0.5, y_pos_h, "Heuristic", ha='right', va='center', fontsize=8, color='gray')
        ax.text(-0.5, y_pos_p, "PPO", ha='right', va='center', fontsize=8, color='gray')

    # Formatting
    ax.set_xlabel("Simulation Time (hours)")
    ax.set_ylabel("Truck ID")
    ax.set_title(f"Schedule Comparison: Heuristic (Top) vs PPO (Bottom) - Seed {SEED}")
    ax.set_xlim(0, max_time)
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
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.12, 1))

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "schedule_comparison.png")
    plt.savefig(save_path, dpi=300)
    print(f"Comparison plot saved to {save_path}")

def _plot_segment(ax, segment, y_pos, colors, charging_nodes, label_soc=False):
    state = segment["state"]
    start = segment["start"]
    end = segment["end"]
    meta = segment["meta"]
    end_soc = segment.get("end_soc")
    
    color = colors.get(state, "black")
    linewidth = 2.5 if state in ["charging", "routing", "waiting_to_charge"] else 1.0
    
    ax.hlines(y=y_pos, xmin=start, xmax=end, colors=color, linewidth=linewidth)
    
    if label_soc and end_soc is not None:
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

def main():
    # Run Heuristic
    print("\n--- Running Heuristic Policy ---")
    hist_h, max_time, charging_nodes, _, env_h, _ = run_scenario("heuristic")
    
    # Run PPO
    print("\n--- Running PPO Policy ---")
    hist_p, _, _, _, env_p, _ = run_scenario("ppo-variable")
    
    # Plot Comparison
    plot_comparison(hist_h, hist_p, max_time, charging_nodes)
    
    # Plot Queue Dynamics for PPO (most relevant)
    print("Plotting queue dynamics for PPO...")
    plotter = EnvironmentPlotter(OUTPUT_DIR, verbose=False, use_osm=False)
    plotter.plot_charger_queue_dynamics(env_p.charging_station, env_p.transport_graph, max_time)

if __name__ == "__main__":
    main()
