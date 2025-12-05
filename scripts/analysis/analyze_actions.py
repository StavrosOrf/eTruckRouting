"""
Analyze and visualize action distributions for different policies.
"""

import copy
import os
import sys
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from EVRoutingEnv.models.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.state.gnn_state_space import GNNStateSpace
from EVRoutingEnv.utils.utils import load_config
from scripts.training.train_PPO_Variable import compute_action_mask
from algo.policy_utils import load_policy

# ============ CONFIGURATION ============
POLICIES = [
    ("saved_models/ppo-variable_steps=128_epochs=10_ent=0.1_seed=0_gnnhd=64_mlphd=256", "ppo-variable"),
    ("saved_models/ppo-variable_steps=128_epochs=10_ent=0.1_seed=0_gnnhd=64_mlphd=64", "ppo-variable"),
    ("heuristic", "heuristic"),
]
CONFIG_FILE = "EVRoutingEnv/config_files/config.yaml"
NUM_TRUCKS = 10
NUM_STOPS = 3
NUM_EVAL_SCENARIOS = 10
SEED = 1000
OUTPUT_DIR = "results/analysis"
# =======================================

def categorize_action(action, env):
    """
    Convert an environment action (int or tuple) into a standardized category.
    Returns: (category_string, duration_float)
    """
    # Tuple Action (GNN)
    if isinstance(action, tuple):
        node_id, duration, is_charging = action
        if is_charging:
            return "Charge", float(duration)
        else:
            # Navigation
            if node_id in env.charging_nodes:
                return "Route to Charger", 0.0
            else:
                return "Route to Delivery", 0.0

    # Integer Action (Legacy/Heuristic)
    else:
        action = int(action)
        if action < env.num_charging_nodes:
            return "Route to Charger", 0.0
        elif action == env.num_charging_nodes:
            return "Route to Delivery", 0.0
        else:
            # Charge action
            charge_idx = action - env.num_navigation_actions
            durations = env.charging_config["charge_durations"]
            if 0 <= charge_idx < len(durations):
                return "Charge", float(durations[charge_idx])
            else:
                return "Charge (Unknown)", 0.0

def collect_actions(env, policy, gnn_state_space, policy_type, num_episodes, seed, policy_name):
    """Run episodes and collect all actions."""
    data = []
    
    for episode in tqdm(range(num_episodes), desc=f"Collecting {policy_name}", leave=False):
        obs, info = env.reset(seed=seed + episode)
        episode_steps = 0
        done = truncated = False

        while not (done or truncated):
            gnn_state = gnn_state_space.get_state_GNN(env)

            # Select Action
            if policy_type == "heuristic":
                action = policy.get_action(env)
            elif policy_type == "ppo-variable":
                raw_action = policy.select_action(gnn_state, deterministic=True)
                action = policy.to_env_action(gnn_state, int(raw_action))
            else:  # ppo standard
                mask = torch.tensor(compute_action_mask(env), dtype=torch.bool)
                raw_action = policy.select_action(gnn_state, deterministic=True, action_mask=mask)
                if isinstance(raw_action, tuple):
                    action = raw_action
                else:
                    action = int(raw_action) % env.action_space.n

            # Categorize and Store
            cat, duration = categorize_action(action, env)
            data.append({
                "Policy": policy_name,
                "Episode": episode,
                "Step": episode_steps,
                "Action Type": cat,
                "Duration": duration if cat == "Charge" else None
            })

            # Step Env
            obs, reward, done, truncated, info = env.step(action)
            episode_steps += 1

    return data

def plot_analysis(df, output_dir):
    """Generate and save plots."""
    sns.set_theme(style="whitegrid")
    
    # 1. Action Type Distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x="Policy", hue="Action Type", palette="viridis")
    plt.title("Action Distribution by Policy")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "action_distribution.png"))
    plt.close()

    # 2. Charging Duration Distribution
    charge_df = df[df["Action Type"] == "Charge"].copy()
    if not charge_df.empty:
        plt.figure(figsize=(10, 6))
        sns.histplot(data=charge_df, x="Duration", hue="Policy", element="step", stat="density", common_norm=False, palette="magma")
        plt.title("Charging Duration Distribution")
        plt.xlabel("Duration (hours)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "charging_duration_dist.png"))
        plt.close()
        
        # Boxplot for durations
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=charge_df, x="Policy", y="Duration", palette="magma")
        plt.title("Charging Duration Boxplot")
        plt.ylabel("Duration (hours)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "charging_duration_box.png"))
        plt.close()

    print(f"Plots saved to {output_dir}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Setup Env & State
    config = load_config(CONFIG_FILE)
    config["environment"]["num_trucks"] = NUM_TRUCKS
    config["environment"]["num_stops"] = NUM_STOPS
    
    env_init = EventDrivenTruckEnv(config=config, verbose=False, enable_plotting=False)
    gnn_state_space = GNNStateSpace(
        num_trucks=NUM_TRUCKS,
        num_stops=NUM_STOPS,
        max_time=config["environment"]["max_time"],
        num_charging_nodes=env_init.num_charging_nodes,
    )
    env_init.close()

    # Load Policies
    policies = {}
    for policy_path, policy_type in POLICIES:
        print(f"Loading: {policy_path} ({policy_type})...")
        policy, resolved_type = load_policy(policy_path, policy_type, gnn_state_space, config)
        name = os.path.basename(policy_path) if policy_path != "heuristic" else "Heuristic"
        policies[name] = {"policy": policy, "type": resolved_type}

    # Collect Data
    all_data = []
    eval_env = EventDrivenTruckEnv(config=copy.deepcopy(config), verbose=False, enable_plotting=False)
    
    print(f"\n{'='*60}")
    print(f"Analyzing Actions for {len(policies)} policies")
    print(f"{'='*60}\n")

    for name, info in policies.items():
        policy_data = collect_actions(
            eval_env, 
            info["policy"], 
            gnn_state_space, 
            info["type"], 
            NUM_EVAL_SCENARIOS, 
            SEED,
            name
        )
        all_data.extend(policy_data)
    
    eval_env.close()

    # Analyze
    df = pd.DataFrame(all_data)
    print("\nSummary Statistics:")
    print(df.groupby(["Policy", "Action Type"]).size().unstack(fill_value=0))
    
    plot_analysis(df, OUTPUT_DIR)

if __name__ == "__main__":
    main()
