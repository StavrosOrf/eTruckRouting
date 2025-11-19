"""Evaluate and compare different policies on multiple scenarios."""

import copy
import os
import sys
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from truck_env.models.event_driven_env import EventDrivenTruckEnv
from truck_env.state.gnn_state_space import GNNStateSpace
from truck_env.utils.utils import load_config

# Import compute_action_mask from train module
from train import compute_action_mask
from algo.policy_utils import load_policy

# ============ HARDCODED PARAMETERS ============
POLICIES = [
    # ("saved_models/ppo-variable_steps=128_epochs=10_ent=0.1_seed=0_gnnhd=64_mlphd=256/", "variable-ppo"),
    ("saved_models/ppo-variable_steps=128_epochs=10_ent=0.1_seed=0_gnnhd=64_mlphd=64/", "variable-ppo"),
    # ("saved_models/debug_ppo_var_eval", "ppo-variable"),
    ("heuristic", "heuristic"),
]
CONFIG_FILE = "truck_env/config_files/config.yaml"
NUM_TRUCKS = 10
NUM_STOPS = 3
NUM_EVAL_SCENARIOS = 20
MAX_EPISODE_STEPS = 200
SEED = 1000
# =============================================


def evaluate_policy(
    env, policy, gnn_state_space, policy_type, num_episodes, seed, max_steps
):
    """Evaluate a policy over multiple episodes."""
    rewards, successes, distances, charging_times = [], [], [], []

    for episode in tqdm(range(num_episodes), desc="Evaluating", leave=False):
        obs, info = env.reset(seed=seed + episode)
        episode_reward, episode_steps = 0.0, 0
        done = truncated = False

        while not (done or truncated) and episode_steps < max_steps:
            gnn_state = gnn_state_space.get_state_GNN(env)

            if policy_type == "heuristic":
                action = policy.get_action(env)
            elif policy_type == "ppo-variable":
                # Use deterministic evaluation (greedy) to match train.py
                raw_action = policy.select_action(gnn_state, deterministic=True)
                action = policy.to_env_action(gnn_state, int(raw_action))
            else:  # ppo
                # Use deterministic evaluation (greedy) to match train.py
                mask = torch.tensor(compute_action_mask(env), dtype=torch.bool)
                raw_action = policy.select_action(gnn_state, deterministic=True, action_mask=mask)
                if isinstance(raw_action, tuple):
                    action = raw_action
                else:
                    action = int(raw_action) % env.action_space.n

            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            episode_steps += 1

        rewards.append(episode_reward)
        successes.append(1.0 if info.get("all_complete", False) else 0.0)

        total_dist = sum(t.get("total_distance", 0.0) for t in info.get("trucks", []))
        total_charge = sum(
            t.get("total_charging_time", 0.0) for t in info.get("trucks", [])
        )
        distances.append(total_dist)
        charging_times.append(total_charge)

    return {
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "success_rate": np.mean(successes),
        "mean_total_distance": np.mean(distances),
        "std_total_distance": np.std(distances),
        "mean_charging_time": np.mean(charging_times),
        "std_charging_time": np.std(charging_times),
    }


def main():
    """Evaluate policies with hardcoded parameters."""

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

    # Load policies
    policies = {}
    for policy_path, policy_type in POLICIES:
        print(f"Loading: {policy_path} ({policy_type})...")
        policy, resolved_type = load_policy(policy_path, policy_type, gnn_state_space, config)
        name = (
            os.path.basename(policy_path) if policy_path != "heuristic" else "Heuristic"
        )
        policies[name] = {"policy": policy, "type": resolved_type}

    # Evaluate all policies
    eval_env = EventDrivenTruckEnv(
        config=copy.deepcopy(config), verbose=False, enable_plotting=False
    )

    print(f"\n{'='*90}")
    print(f"Evaluating {len(policies)} policies over {NUM_EVAL_SCENARIOS} scenarios")
    print(f"Environment: {NUM_TRUCKS} trucks, {NUM_STOPS} stops\n")

    results = {}
    for policy_name, policy_info in tqdm(policies.items(), desc="Policies", position=0):
        results[policy_name] = evaluate_policy(
            eval_env,
            policy_info["policy"],
            gnn_state_space,
            policy_info["type"],
            NUM_EVAL_SCENARIOS,
            SEED,
            MAX_EPISODE_STEPS,
        )

    eval_env.close()

    # Print results table
    print(f"\n{'='*90}\nRESULTS (averaged over {NUM_EVAL_SCENARIOS} scenarios)\n")
    print(
        f"{'Policy':<50} {'Reward':<20} {'Success %':<15} {'Distance (km)':<18} {'Charging (h)':<15}"
    )
    print("-" * 90)

    for name in sorted(results.keys()):
        r = results[name]
        print(
            f"{name:<50} {r['mean_reward']:8.0f} ±{r['std_reward']:6.0f} {r['success_rate']*100:>6.1f}%      "
            f"{r['mean_total_distance']:>9.0f} ±{r['std_total_distance']:<6.0f} {r['mean_charging_time']:>7.1f} ±{r['std_charging_time']:<6.1f}"
        )

    print(f"{'='*90}\n")


if __name__ == "__main__":
    main()
