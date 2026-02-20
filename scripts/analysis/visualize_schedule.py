"""
Visualize truck schedules (Gantt chart) using truck event logs.

This version uses the built-in truck event monitoring system to directly
extract accurate timelines without postprocessing event queues.
"""
import os
import sys
import copy
import json
import pickle
import hashlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import shutil
import traceback
import torch
from stable_baselines3 import PPO, DQN
from sb3_contrib import MaskablePPO, QRDQN

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.state.gnn_state_space_nonflex import GNNStateSpaceNonFlex
from EVRoutingEnv.state.gnn_state_space_vrp import GNNStateSpaceVRP
from EVRoutingEnv.state.gnn_utils import create_default_gnn_space
from EVRoutingEnv.utils.utils import load_config
from EVRoutingEnv.baselines.optimal_gurobi import OptimalGurobiPolicy
from EVRoutingEnv.baselines.optimal_gurobi_simple import OptimalGurobiSimplePolicy
from EVRoutingEnv.baselines.optimal_vrp_single_truck import OptimalVRPSingleTruckPolicy
from EVRoutingEnv.state.action_mask import get_action_mask
from algo.policy_utils import load_policy

# ============ CONFIGURATION ============
# Policies to compare
POLICIES = [
    # # #10T3S
    # ("saved_models/ppov_seq_10T3S_spu256_ep5_ent0.1_g32_m256_vk5_ck5_s0_8197/", "variable-ppo", "detour"),    
    # ("saved_models/ppov_seq_10T3S_spu256_ep5_ent0.1_g32_m256_vk5_ck2_s0_8197/", "variable-ppo", "detour"),  
    # ("saved_models/10trucks_3stops/maskppo_seed0_20260212_223718/best_model.zip", "sb3-maskppo", "base"),
    # ("saved_models/10trucks_3stops/ppo_seed1_20260212_223935/best_model.zip", "sb3-ppo", "base"),
    #   ("optimal", "optimal"),
    #   ("optimal-simple", "optimal-simple"),
    
    #VRP models
    # ("saved_models/Top5del_NewStateppov_1T10S_spu256_ep5_ent0.1_seed0_505/", "variable-ppo", "vrp"),    
    # ("saved_models/1trucks_10stops/maskppo_seed0_20260209_164605/best_model.zip", "sb3-maskppo", "base"),
    

    # ("saved_models/ppov_vrp_1T20S_spu256_ep5_ent0.1_g32_m256_vk2_ck5_hl2_s0_8166/", "variable-ppo", "vrp"),        
    
    ("saved_models/ppov_vrp_1T30S_spu256_ep5_ent0.1_g32_m256_vk3_ck5_hl2_s0_8166/", "variable-ppo", "vrp"), 
    ("saved_models/ppov_vrp_1T10S_spu256_ep5_ent0.1_g32_m256_vk5_ck5_hl2_s0_8166/", "variable-ppo", "vrp"),        
    ("saved_models/ppov_vrp_1T10S_spu256_ep5_ent0.1_g32_m256_vk3_ck5_hl2_s0_8166/", "variable-ppo", "vrp"),
    ("saved_models/1trucks_30stops/maskppo_seed0_20260219_172612/best_model.zip", "sb3-maskppo", "base"),
    ("saved_models/1trucks_30stops/ppo_seed1_20260219_172612/best_model.zip", "sb3-ppo", "base"),
    ("savings", "savings", "base"),
    ("nn-2opt", "nn-2opt", "base"),
    ("optimal-vrp", "optimal-vrp"),
    
    # ("heuristic", "heuristic"),
]

# CONFIG_FILE = "EVRoutingEnv/config_files/config.yaml"
CONFIG_FILE = "EVRoutingEnv/config_files/config_vrp.yaml"
NUM_TRUCKS = 1
NUM_STOPS = 30
MAX_TIME = 200.0
SEED = 1000111112 #10001
OUTPUT_DIR = "results/visualization"
CACHE_DIR = os.path.join(OUTPUT_DIR, "cache")
CACHE_ENABLED = False
CACHE_VERSION = "v1"
# =======================================


def _safe_file_stamp(path: str):
    """Return lightweight file/dir stamp for cache invalidation."""
    if not isinstance(path, str):
        return None
    if os.path.isfile(path):
        stat = os.stat(path)
        return {"type": "file", "path": path, "mtime": stat.st_mtime, "size": stat.st_size}
    if os.path.isdir(path):
        stat = os.stat(path)
        return {"type": "dir", "path": path, "mtime": stat.st_mtime}
    return {"type": "virtual", "path": path}


def _build_scenario_cache_key(policy_path, policy_type, gnn_space_type):
    """Build deterministic cache key for one scenario run."""
    config_stamp = _safe_file_stamp(CONFIG_FILE)
    policy_stamp = _safe_file_stamp(policy_path)
    payload = {
        "cache_version": CACHE_VERSION,
        "seed": SEED,
        "max_time": MAX_TIME,
        "default_num_trucks": NUM_TRUCKS,
        "default_num_stops": NUM_STOPS,
        "config_file": CONFIG_FILE,
        "config_stamp": config_stamp,
        "policy_path": str(policy_path),
        "policy_type": str(policy_type),
        "gnn_space_type": str(gnn_space_type),
        "policy_stamp": policy_stamp,
    }
    serialized = json.dumps(payload, sort_keys=True, default=str)
    key = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:20]
    return key, payload


def _load_cached_scenario(cache_path):
    with open(cache_path, "rb") as f:
        return pickle.load(f)


def _save_cached_scenario(cache_path, payload):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def _extract_sb3_config(policy_path):
    """Return (num_trucks, num_stops) if encoded like '1trucks_10stops' in path."""
    import re

    for part in str(policy_path).rstrip("/").split("/"):
        match = re.search(r"(\d+)trucks_(\d+)stops", part)
        if match:
            return int(match.group(1)), int(match.group(2))
    return None


def _load_saved_gnn_state_config(policy_path: str) -> dict:
    if not isinstance(policy_path, str):
        return {}
    if policy_path in ("heuristic", "optimal", "optimal-simple", "optimal-vrp", "optimal_vrp"):
        return {}
    base_path = os.path.dirname(policy_path) if policy_path.endswith(".zip") else policy_path
    if not os.path.isdir(base_path):
        return {}
    config_path = os.path.join(base_path, "config.yaml")
    if not os.path.exists(config_path):
        return {}
    try:
        return load_config(config_path).get("gnn_state_space", {})
    except Exception:
        return {}


def _create_gnn_state_space(env_init, gnn_space_type: str, policy_path: str):
    """Instantiate the requested GNN state space; defaults to base if unknown."""
    gnn_space_type = (gnn_space_type or "base").lower()
    mode = "vrp" if gnn_space_type == "vrp" else "nonflex"
    use_detour = gnn_space_type == "detour"
    gnn_cfg = _load_saved_gnn_state_config(policy_path)
    vrp_top_k = int(gnn_cfg.get("vrp_top_k_deliveries", 5))
    detour_top_k = int(gnn_cfg.get("detour_top_k_chargers", 2))
    detour_hop_limit = int(gnn_cfg.get("detour_hop_limit", 2))
    return create_default_gnn_space(
        env_init,
        mode=mode,
        use_detour=use_detour,
        device="cpu",
        vrp_top_k_deliveries=vrp_top_k,
        detour_num_chargers_to_keep=detour_top_k,
        detour_hop_limit=detour_hop_limit,
    )


def run_scenario(policy_path, policy_type, gnn_space_type="base"):
    """Run a single scenario with the given policy and collect truck event logs."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Setup
    config = load_config(CONFIG_FILE)
    num_trucks = NUM_TRUCKS
    num_stops = NUM_STOPS

    # Align environment size with SB3 training config if encoded in path
    if str(policy_type).startswith("sb3-"):
        detected = _extract_sb3_config(policy_path)
        if detected:
            num_trucks, num_stops = detected
            print(f"Detected SB3 config: {num_trucks} trucks, {num_stops} stops")
        else:
            print("SB3 policy detected but config not encoded in path; using defaults.")

    config["environment"]["num_trucks"] = num_trucks
    config["environment"]["num_stops"] = num_stops
    config["environment"]["max_time"] = MAX_TIME
    
    # Initialize State Space
    env_init = EventDrivenTruckEnv(config=config, verbose=False, enable_plotting=False)
    gnn_state_space = _create_gnn_state_space(env_init, gnn_space_type, policy_path)
    env_init.close()

    # Load Policy
    print(f"Loading policy: {policy_path} (type: {policy_type})...")
    
    if str(policy_type).startswith("sb3-"):
        algo = policy_type.replace("sb3-", "")
        if algo == "ppo":
            policy = PPO.load(policy_path, device="cpu")
            active_policy_type = "sb3-ppo"
        elif algo == "maskppo":
            policy = MaskablePPO.load(policy_path, device="cpu")
            active_policy_type = "sb3-maskppo"
        elif algo == "dqn":
            policy = DQN.load(policy_path, device="cpu")
            active_policy_type = "sb3-dqn"
        elif algo == "qrdqn":
            policy = QRDQN.load(policy_path, device="cpu")
            active_policy_type = "sb3-qrdqn"
        else:
            raise ValueError(f"Unknown SB3 algorithm: {algo}")
        print(f"Loaded SB3 model ({algo.upper()})")
    elif policy_type == "optimal":
        try:
            policy = OptimalGurobiPolicy(verbose=False)
            active_policy_type = "optimal"
        except ImportError as exc:
            raise RuntimeError(
                "Optimal (Gurobi) policy requires gurobipy to be installed."
            ) from exc
    elif policy_type == "optimal-simple":
        try:
            policy = OptimalGurobiSimplePolicy(verbose=False)
            active_policy_type = "optimal-simple"
        except ImportError as exc:
            raise RuntimeError(
                "Optimal Simple (Gurobi) policy requires gurobipy to be installed."
            ) from exc
    elif policy_type in ("optimal-vrp", "optimal_vrp"):
        try:
            policy = OptimalVRPSingleTruckPolicy(verbose=False)
            active_policy_type = "optimal-vrp"
        except ImportError as exc:
            raise RuntimeError(
                "Optimal VRP (Gurobi) policy requires gurobipy to be installed."
            ) from exc
    elif policy_type == "heuristic":
        from EVRoutingEnv.baselines.heuristic_policy import HeuristicPolicy
        policy = HeuristicPolicy()
        active_policy_type = "heuristic"
    else:
        policy, active_policy_type = load_policy(policy_path, policy_type, gnn_state_space, config, device="cuda")

    # Run Environment with plotting enabled for route visualization
    env = EventDrivenTruckEnv(config=copy.deepcopy(config), verbose=False, enable_plotting=True, run_id="visualization_temp")
    env.use_detour_mask = gnn_space_type == "detour"
    env._default_gnn_state_space = gnn_state_space
    
    print(f"Running scenario with seed {SEED}...")
    obs, info = env.reset(seed=SEED)
    
    done = truncated = False
    episode_steps = 0
    
    # Recreate optimal planner per episode to avoid stale plans
    if active_policy_type == "optimal":
        episode_policy = OptimalGurobiPolicy(verbose=False)
    elif active_policy_type == "optimal-simple":
        episode_policy = OptimalGurobiSimplePolicy(verbose=False)
    elif active_policy_type == "optimal-vrp":
        episode_policy = OptimalVRPSingleTruckPolicy(verbose=False)
    else:
        episode_policy = policy
    
    while not (done or truncated):
        if active_policy_type.startswith("sb3-"):
            if active_policy_type == "sb3-maskppo":
                action_masks = get_action_mask(env)
                action, _states = policy.predict(obs, action_masks=action_masks, deterministic=True)
            else:
                action, _states = policy.predict(obs, deterministic=True)
        elif active_policy_type in ["optimal", "optimal-simple", "optimal-vrp", "heuristic", "savings", "nn-2opt"] or hasattr(episode_policy, "get_action"):
            action = episode_policy.get_action(env)
        else:
            gnn_state = gnn_state_space.get_state_GNN(env)
            
            if active_policy_type == "ppo-variable" or active_policy_type == "variable-ppo":
                mask = torch.tensor(get_action_mask(env), dtype=torch.bool)
                raw_action = policy.select_action(
                    gnn_state, deterministic=True, action_mask=mask
                )
                action = policy.to_env_action(gnn_state, int(raw_action))
            else:
                mask = torch.tensor(get_action_mask(env), dtype=torch.bool)
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
    
    # Map each truck to a vertically spaced track to reduce overlap between trucks
    spacing_scale = 1.6
    y_tracks = {tid: idx * spacing_scale for idx, tid in enumerate(truck_ids)}

    # Create figure with more height per truck
    length = len(truck_ids)
    fig, ax = plt.subplots(figsize=(24, length * 1.9))
    
    # Y-axis positions: offset for each policy (increased spacing to avoid overlap)
    if len(policy_names) == 1:
        offsets = [0.0]
    elif len(policy_names) == 2:
        offsets = [0.25, -0.25]
    elif len(policy_names) == 3:
        offsets = [0.35, 0.0, -0.35]
    else:
        step = 0.3
        start = step * (len(policy_names) - 1) / 2
        offsets = [start - i * step for i in range(len(policy_names))]
    
    for tid in truck_ids:
        y_base = y_tracks[tid]
        # Draw background track with more spacing
        ax.hlines(y=y_base, xmin=0, xmax=actual_max_time, colors='gray', 
                 linestyles=':', alpha=0.1, linewidth=42)
        
        # Plot each policy
        for idx, (timelines, policy_name, offset) in enumerate(zip(timelines_list, policy_names, offsets)):
            y_pos = y_base + offset
            
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
    ax.set_yticks([y_tracks[tid] for tid in truck_ids])
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
               markerfacecolor='gold', markeredgecolor='black', markeredgewidth=0.5, zorder=30)
    
    elif state == "failed":
        ax.plot(start, y_pos, marker='X', markersize=8,
               markerfacecolor='red', markeredgecolor='black', markeredgewidth=0.5, zorder=30)


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


def _add_cartodb_basemap(ax, all_coords):
    """Add a CartoDB Positron basemap focused on provided coordinates."""
    try:
        import contextily as ctx
    except ImportError:
        return

    if not all_coords:
        return

    lats = [lat for lat, _lon in all_coords]
    lons = [lon for _lat, lon in all_coords]
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)

    lat_pad = max((max_lat - min_lat) * 0.06, 0.01)
    lon_pad = max((max_lon - min_lon) * 0.06, 0.01)

    ax.set_xlim(min_lon - lon_pad, max_lon + lon_pad)
    ax.set_ylim(min_lat - lat_pad, max_lat + lat_pad)

    provider = ctx.providers.OpenStreetMap.Mapnik
    try:
        provider = ctx.providers.CartoDB.Positron
    except AttributeError:
        provider = ctx.providers.OpenStreetMap.Mapnik

    ctx.add_basemap(
        ax,
        crs="EPSG:4326",
        source=provider,
        zoom=7,
        alpha=1,
    )


def _collect_energy_edge_segments(transport_graph, node_coords, candidate_nodes):
    """Collect energy-colored edge segments between candidate nodes."""
    nodes = [n for n in candidate_nodes if n in node_coords]
    if len(nodes) < 2:
        return []

    edge_segments = []
    for i, src in enumerate(nodes):
        for dst in nodes[i + 1:]:
            energy = transport_graph.get_path_energy(src, dst)
            if np.isfinite(energy):
                edge_segments.append(
                    (
                        node_coords[src][1],
                        node_coords[src][0],
                        node_coords[dst][1],
                        node_coords[dst][0],
                        energy,
                    )
                )
    return edge_segments


def plot_route_maps(envs, policy_names, output_dir):
    """Generate one combined route-map figure with policy subplots."""

    panels = []
    for env, policy_name in zip(envs, policy_names):
        try:
            if not hasattr(env, 'plotter') or env.plotter is None:
                print(f"  ⚠ Skipping {policy_name}: No plotter available")
                continue

            plotter = env.plotter
            if not plotter.node_coords and not plotter.charger_coords:
                print(f"  ⚠ Skipping {policy_name}: No coordinate data available")
                continue

            node_coords = plotter._create_node_id_to_osm_map(env.transport_graph)
            charger_coords = plotter._create_charger_id_to_osm_map(env.transport_graph)
            if not node_coords:
                print(f"  ⚠ Skipping {policy_name}: No node coordinates found")
                continue

            per_truck_routes = []
            visited_delivery_nodes = set()
            visited_charger_nodes = set()
            depot_nodes = set()

            for truck in env.trucks:
                route_nodes = []
                depot_node = int(truck.delivery_sequence[0])
                depot_nodes.add(depot_node)
                route_nodes.append(depot_node)

                for event in truck.event_history:
                    if event["event_type"] == "ROUTING_END":
                        route_nodes.append(int(event["location"]))

                dedup_route_nodes = []
                for node in route_nodes:
                    if not dedup_route_nodes or dedup_route_nodes[-1] != node:
                        dedup_route_nodes.append(node)

                per_truck_routes.append((truck.truck_id, dedup_route_nodes))

                for node in dedup_route_nodes:
                    if node in env.charging_nodes:
                        visited_charger_nodes.add(node)
                    else:
                        visited_delivery_nodes.add(node)

            if not per_truck_routes:
                print(f"  ⚠ Skipping {policy_name}: No visited route nodes found")
                continue

            visited_delivery_coords = [
                node_coords[node]
                for node in visited_delivery_nodes
                if node in node_coords
            ]

            visited_charger_coords = []
            for node in visited_charger_nodes:
                if node in charger_coords:
                    visited_charger_coords.append(charger_coords[node])
                elif node in node_coords:
                    visited_charger_coords.append(node_coords[node])

            area_coords = visited_delivery_coords + visited_charger_coords
            if not area_coords:
                print(f"  ⚠ Skipping {policy_name}: No coordinates found for visited nodes")
                continue

            displayed_nodes = set(visited_delivery_nodes) | set(visited_charger_nodes)
            edge_segments = _collect_energy_edge_segments(
                env.transport_graph,
                node_coords,
                displayed_nodes,
            )

            panels.append(
                {
                    "env": env,
                    "policy_name": policy_name,
                    "node_coords": node_coords,
                    "charger_coords": charger_coords,
                    "per_truck_routes": per_truck_routes,
                    "visited_delivery_coords": visited_delivery_coords,
                    "visited_charger_coords": visited_charger_coords,
                    "depot_nodes": depot_nodes,
                    "area_coords": area_coords,
                    "edge_segments": edge_segments,
                }
            )

        except Exception as e:
            print(f"  ✗ Error preparing route map for {policy_name}: {e}")

    if not panels:
        print("  ⚠ No route maps generated: no valid policy panels")
        return

    n_panels = len(panels)
    n_cols = 2
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14 * n_cols, 9 * n_rows), dpi=300)
    axes = np.array(axes).reshape(-1)

    all_energies = []
    for panel in panels:
        all_energies.extend([seg[4] for seg in panel["edge_segments"]])

    if all_energies:
        vmin = min(all_energies)
        vmax = max(all_energies)
        if vmin == vmax:
            vmin -= 1.0
            vmax += 1.0
        edge_norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        edge_cmap = plt.cm.viridis
    else:
        edge_norm = None
        edge_cmap = None
        vmax = 0

    for panel_idx, panel in enumerate(panels):
        ax = axes[panel_idx]
        ax.set_facecolor("#f7f5f2")

        _add_cartodb_basemap(ax, panel["area_coords"])

        if edge_norm is not None:
            for x1, y1, x2, y2, energy in panel["edge_segments"]:
                ax.plot(
                    [x1, x2],
                    [y1, y2],
                    color=edge_cmap(edge_norm(energy)),
                    linewidth=0.9,
                    alpha=0.22,
                    zorder=2,
                )

        num_trucks = len(panel["env"].trucks)
        truck_colors = plt.cm.tab10(range(num_trucks))

        for truck_id, route_nodes in panel["per_truck_routes"]:
            truck_color = truck_colors[truck_id % len(truck_colors)]

            route_coords = []
            for node in route_nodes:
                if node in panel["node_coords"]:
                    route_coords.append((node, panel["node_coords"][node]))
                elif node in panel["charger_coords"]:
                    route_coords.append((node, panel["charger_coords"][node]))

            if len(route_coords) > 1:
                lats = [coord[1][0] for coord in route_coords]
                lons = [coord[1][1] for coord in route_coords]
                ax.plot(lons, lats, c=truck_color, linewidth=4.0, alpha=0.78, zorder=5)

                for i in range(len(route_coords) - 1):
                    start_lat, start_lon = route_coords[i][1]
                    end_lat, end_lon = route_coords[i + 1][1]

                    dx = end_lon - start_lon
                    dy = end_lat - start_lat
                    seg_len = (dx * dx + dy * dy) ** 0.5
                    if seg_len < 1e-9:
                        continue

                    mid_lon = (start_lon + end_lon) * 0.5
                    mid_lat = (start_lat + end_lat) * 0.5
                    arrow_scale = 0.18
                    vec_lon = dx * arrow_scale
                    vec_lat = dy * arrow_scale

                    ax.annotate(
                        "",
                        xy=(mid_lon + vec_lon * 0.5, mid_lat + vec_lat * 0.5),
                        xytext=(mid_lon - vec_lon * 0.5, mid_lat - vec_lat * 0.5),
                        arrowprops=dict(
                            arrowstyle="-|>",
                            color=truck_color,
                            lw=1.3,
                            alpha=0.95,
                            mutation_scale=11,
                            shrinkA=0,
                            shrinkB=0,
                        ),
                        zorder=6,
                    )

        if panel["visited_delivery_coords"]:
            delivery_lats = [lat for lat, _lon in panel["visited_delivery_coords"]]
            delivery_lons = [lon for _lat, lon in panel["visited_delivery_coords"]]
            ax.scatter(
                delivery_lons,
                delivery_lats,
                s=58,
                c="#d95f02",
                alpha=0.9,
                linewidths=0.4,
                edgecolors="#4a2a00",
                zorder=6,
            )

        if panel["visited_charger_coords"]:
            charger_lats = [lat for lat, _lon in panel["visited_charger_coords"]]
            charger_lons = [lon for _lat, lon in panel["visited_charger_coords"]]
            ax.scatter(
                charger_lons,
                charger_lats,
                s=75,
                c="#419B0C",
                marker="s",
                alpha=0.9,
                linewidths=0.5,
                edgecolors="#0b3d2e",
                zorder=7,
            )

        for depot_node in panel["depot_nodes"]:
            if depot_node in panel["node_coords"]:
                depot_lat, depot_lon = panel["node_coords"][depot_node]
                ax.scatter(
                    depot_lon,
                    depot_lat,
                    c="black",
                    s=95,
                    marker="^",
                    alpha=0.95,
                    edgecolors="white",
                    linewidths=0.8,
                    zorder=8,
                )

        all_interest_coords = panel["visited_delivery_coords"] + panel["visited_charger_coords"]
        if all_interest_coords:
            lats = [lat for lat, _ in all_interest_coords]
            lons = [lon for _, lon in all_interest_coords]
            min_lat, max_lat = min(lats), max(lats)
            min_lon, max_lon = min(lons), max(lons)
            lat_pad = max((max_lat - min_lat) * 0.06, 0.01)
            lon_pad = max((max_lon - min_lon) * 0.06, 0.01)
            ax.set_xlim(min_lon - lon_pad, max_lon + lon_pad)
            ax.set_ylim(min_lat - lat_pad, max_lat + lat_pad)

        ax.set_xlabel("Longitude", fontsize=11)
        ax.set_ylabel("Latitude", fontsize=11)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, color="#c7c7c7", alpha=0.7)

        panel_letter = chr(ord('a') + panel_idx)
        ax.text(
            0.5,
            -0.14,
            f"{panel_letter}. {panel['policy_name']}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=12,
            fontweight="bold",
        )

    for ax in axes[n_panels:]:
        ax.set_visible(False)

    if edge_norm is not None:
        sm = plt.cm.ScalarMappable(norm=edge_norm, cmap=edge_cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes[:n_panels], orientation="horizontal", fraction=0.03, pad=0.08, location="top")
        tick_interval = 50
        max_tick = int(np.ceil(vmax / tick_interval) * tick_interval)
        ticks = np.arange(0, max_tick + tick_interval, tick_interval)
        cbar.set_ticks(ticks)
        cbar.ax.tick_params(pad=6)
        cbar.set_label("Average Energy Needed (kWh)")

    common_handles = [
        plt.Line2D([0], [0], marker='o', color='none', markerfacecolor="#d95f02", markeredgecolor="#4a2a00", markersize=8, label="Visited Delivery Nodes"),
        plt.Line2D([0], [0], marker='s', color='none', markerfacecolor="#419B0C", markeredgecolor="#0b3d2e", markersize=8, label="Visited Charging Stations"),
        plt.Line2D([0], [0], marker='^', color='none', markerfacecolor="black", markeredgecolor="white", markersize=8, label="Depot"),
        plt.Line2D([0], [0], color="#444444", linewidth=4.0, label="Truck Path"),
    ]
    fig.legend(
        handles=common_handles,
        loc="lower center",
        ncol=4,
        frameon=True,
        framealpha=0.9,
        bbox_to_anchor=(0.5, 0.01),
    )

    fig.suptitle(f"Route Map Comparison (Seed {SEED})", fontsize=16, y=0.98)
    fig.tight_layout(rect=[0.02, 0.06, 0.98, 0.94])
    save_path = os.path.join(output_dir, "route_maps_comparison.png")
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Combined route map figure saved: {save_path}")


def plot_charging_queue_dynamics(envs, policy_names, output_dir):
    """Plot charging queue dynamics for top 10 busiest chargers."""
    # Collect charging events from all environments
    charger_usage = {}  # charger_node -> total_time_occupied
    
    for env in envs:
        for truck in env.trucks:
            for event in truck.event_history:
                if event['event_type'] == 'CHARGING_START':
                    charger_node = event['location']
                    # Find corresponding end event
                    for future_event in truck.event_history:
                        if (future_event['event_type'] == 'CHARGING_END' and 
                            future_event['location'] == charger_node and
                            future_event['timestamp'] > event['timestamp']):
                            duration = future_event['timestamp'] - event['timestamp']
                            charger_usage[charger_node] = charger_usage.get(charger_node, 0) + duration
                            break
    
    # Get top 10 chargers by usage
    top_chargers = sorted(charger_usage.items(), key=lambda x: x[1], reverse=True)[:10]
    if not top_chargers:
        print("  ⚠ No charging events found to visualize")
        return
    
    top_charger_nodes = [c[0] for c in top_chargers]
    
    # Create subplots for each charger
    n_chargers = len(top_charger_nodes)
    fig, axes = plt.subplots(n_chargers, 1, figsize=(20, 3 * n_chargers))
    if n_chargers == 1:
        axes = [axes]
    
    colors_policy = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6'][:len(policy_names)]
    
    for idx, charger_node in enumerate(top_charger_nodes):
        ax = axes[idx]
        
        for env, policy_name, color in zip(envs, policy_names, colors_policy):
            # Collect queue events for this charger
            queue_timeline = []  # [(time, queue_length)]
            charging_timeline = []  # [(start, end, truck_id)]
            waiting_timeline = []  # [(start, end, truck_id)]
            
            for truck in env.trucks:
                for i, event in enumerate(truck.event_history):
                    # Track charging at this node
                    if event['event_type'] == 'CHARGING_START' and event['location'] == charger_node:
                        start_time = event['timestamp']
                        # Find end event
                        for j in range(i+1, len(truck.event_history)):
                            if truck.event_history[j]['event_type'] == 'CHARGING_END':
                                end_time = truck.event_history[j]['timestamp']
                                charging_timeline.append((start_time, end_time, truck.truck_id))
                                break
                    
                    # Track waiting at this node
                    if event['event_type'] == 'WAITING_START' and event['location'] == charger_node:
                        start_time = event['timestamp']
                        # Find end event
                        for j in range(i+1, len(truck.event_history)):
                            if truck.event_history[j]['event_type'] == 'WAITING_END':
                                end_time = truck.event_history[j]['timestamp']
                                waiting_timeline.append((start_time, end_time, truck.truck_id))
                                break
            
            # Calculate queue length over time
            all_times = set()
            for start, end, _ in charging_timeline + waiting_timeline:
                all_times.add(start)
                all_times.add(end)
            
            if all_times:
                all_times = sorted(all_times)
                queue_lengths = []
                
                for t in all_times:
                    # Count trucks charging or waiting at this time
                    charging_count = sum(1 for start, end, _ in charging_timeline if start <= t < end)
                    waiting_count = sum(1 for start, end, _ in waiting_timeline if start <= t < end)
                    queue_lengths.append(waiting_count)
                
                # Plot queue length
                ax.step(all_times, queue_lengths, where='post', label=policy_name, 
                       color=color, linewidth=2, alpha=0.8)
        
        # Get charger capacity
        capacity = env.charging_station.charger_capacity.get(charger_node, 1)
        ax.axhline(y=capacity, color='red', linestyle='--', linewidth=1, alpha=0.5, label=f'Capacity ({capacity})')
        
        ax.set_ylabel('Queue Length', fontsize=10)
        ax.set_title(f'Charger Node {charger_node} (Total Usage: {charger_usage[charger_node]:.1f}h)', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)
        
        if idx == n_chargers - 1:
            ax.set_xlabel('Time (hours)', fontsize=11)
    
    plt.suptitle(f'Charging Queue Dynamics - Top {n_chargers} Busiest Chargers', fontsize=14, y=0.995)
    plt.tight_layout()
    save_path = os.path.join(output_dir, "charging_queue_dynamics.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Charging queue dynamics saved: {save_path}")


def main():
    print("="*80)
    print(f"VISUALIZING TRUCK SCHEDULES - Comparing {len(POLICIES)} Policies")
    print("="*80)
    print(f"Configuration: {NUM_TRUCKS} trucks, {NUM_STOPS} stops, {MAX_TIME}h max time (defaults; SB3 entries auto-detect)")
    print(f"Seed: {SEED}")
    if CACHE_ENABLED:
        print(f"Cache: enabled ({CACHE_DIR})")
    else:
        print("Cache: disabled")
    print()
    
    envs = []
    policy_names = []
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if CACHE_ENABLED:
        os.makedirs(CACHE_DIR, exist_ok=True)
    
    # Run each policy
    for idx, policy_entry in enumerate(POLICIES, 1):
        gnn_space_type = "base"
        label_override = None

        if len(policy_entry) == 2:
            policy_path, policy_type = policy_entry
        elif len(policy_entry) == 3:
            policy_path, policy_type, third = policy_entry
            if str(third).lower() in {"base", "detour", "vrp"}:
                gnn_space_type = str(third)
            else:
                label_override = third
        elif len(policy_entry) == 4:
            policy_path, policy_type, gnn_space_type, label_override = policy_entry
        else:
            raise ValueError(f"POLICIES entry must have 2 to 4 elements, got {len(policy_entry)}")
        print(f"\n{'='*80}")
        print(f"[{idx}/{len(POLICIES)}] Running Policy: {policy_path if policy_path not in ['optimal', 'heuristic'] else policy_path.upper()}")
        print(f"{'='*80}")

        cache_key, cache_meta = _build_scenario_cache_key(policy_path, policy_type, gnn_space_type)
        cache_path = os.path.join(CACHE_DIR, f"scenario_{cache_key}.pkl")

        env = None
        active_type = None
        loaded_from_cache = False

        if CACHE_ENABLED and os.path.exists(cache_path):
            try:
                cached = _load_cached_scenario(cache_path)
                env = cached["env"]
                active_type = cached.get("active_policy_type")
                loaded_from_cache = True
                print(f"Loaded cached scenario: {os.path.basename(cache_path)}")
            except Exception as cache_exc:
                print(f"Cache load failed, rerunning scenario ({cache_exc})")

        if not loaded_from_cache:
            env, active_type = run_scenario(policy_path, policy_type, gnn_space_type)
            if CACHE_ENABLED:
                try:
                    _save_cached_scenario(
                        cache_path,
                        {
                            "env": env,
                            "active_policy_type": active_type,
                            "cache_meta": cache_meta,
                        },
                    )
                    print(f"Saved scenario cache: {os.path.basename(cache_path)}")
                except Exception as cache_exc:
                    print(f"Warning: Could not save scenario cache ({cache_exc})")

        envs.append(env)
        
        # Generate readable policy name
        if label_override:
            name = label_override
        elif policy_path == "heuristic":
            name = "Heuristic"
        elif policy_path == "optimal":
            name = "Optimal"
        elif policy_path == "optimal-simple":
            name = "MP Robust"
        elif str(policy_type).startswith("sb3-"):
            dir_path = os.path.dirname(policy_path) if str(policy_path).endswith('.zip') else policy_path
            base = os.path.basename(dir_path.rstrip('/'))
            base = base[:40]
            name = f"SB3-{base}"
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
    
    # Plot Route Maps
    print(f"\nGenerating route visualization maps...")
    plot_route_maps(envs, policy_names, OUTPUT_DIR)
    
    # Plot Charging Queue Dynamics
    print(f"\nGenerating charging queue dynamics...")
    plot_charging_queue_dynamics(envs, policy_names, OUTPUT_DIR)
    
    print(f"\n{'='*80}")
    print("VISUALIZATION COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {OUTPUT_DIR}/")

    for env in envs:
        try:
            env.close()
        except Exception:
            pass
    print()


if __name__ == "__main__":
    main()
