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
from truck_env.optimization import HAS_GUROBI

# Import compute_action_mask from train module
from train import compute_action_mask
from algo.policy_utils import load_policy

# ============ HARDCODED PARAMETERS ============
POLICIES = [
    # ("saved_models/ppo-variable_steps=128_epochs=10_ent=0.1_seed=0_gnnhd=64_mlphd=256/", "variable-ppo"),
    # ("saved_models/ppo-variable_steps=128_epochs=10_ent=0.1_seed=0_gnnhd=64_mlphd=64/", "variable-ppo"),
    # ("saved_models/ppo-variable_steps=128_epochs=10_ent=0.1_seed=0_gnnhd=32_mlphd=256/", "variable-ppo"),
    ("saved_models/NewSoCReward_new_action_step_ppo-variable_steps=128_epochs=10_ent=0.1_seed=0_gnn=32_mlphd=256/", "variable-ppo"),
    ("saved_models/new_action_step_ppo-variable_steps=128_epochs=10_ent=0.1_seed=0_gnn=32_mlphd=256/", "variable-ppo"),
    # ("saved_models/debug_ppo_var_eval", "ppo-variable"),
    ("heuristic", "heuristic"),
]

if HAS_GUROBI:
    POLICIES.append(("optimal", "optimal"))

CONFIG_FILE = "truck_env/config_files/config.yaml"
NUM_TRUCKS = 10
NUM_STOPS = 3
NUM_EVAL_SCENARIOS = 10
MAX_EPISODE_STEPS = 200
SEED = 1000
# =============================================


def _truncate_name(name: str, width: int = 20) -> str:
    return name if len(name) <= width else f"{name[: width - 3]}..."


def evaluate_policy(
    env, policy, gnn_state_space, policy_type, num_episodes, seed, max_steps
):
    """Evaluate a policy over multiple episodes."""
    rewards, successes, distances, charging_times = [], [], [], []
    total_times, makespans, global_times = [], [], []
    step_counts = []
    optimal_charge_durations = []

    for episode in tqdm(range(num_episodes), desc="Evaluating", leave=False):
        obs, info = env.reset(seed=seed + episode)
        if hasattr(policy, "start_episode"):
            policy.start_episode(env)
        episode_reward, episode_steps = 0.0, 0
        done = truncated = False

        while not (done or truncated) and episode_steps < max_steps:
            if policy_type in ("heuristic", "optimal"):
                action = policy.get_action(env)
                if (
                    policy_type == "optimal"
                    and isinstance(action, tuple)
                    and len(action) == 3
                    and action[2]
                ):
                    optimal_charge_durations.append(float(action[1]))
            elif policy_type == "ppo-variable":
                gnn_state = gnn_state_space.get_state_GNN(env)
                # Use deterministic evaluation (greedy) to match train.py
                raw_action = policy.select_action(gnn_state, deterministic=True)
                action = policy.to_env_action(gnn_state, int(raw_action))
            else:  # ppo
                gnn_state = gnn_state_space.get_state_GNN(env)
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
        step_counts.append(episode_steps)

        total_dist = sum(t.get("total_distance", 0.0) for t in info.get("trucks", []))
        total_charge = sum(
            t.get("total_charging_time", 0.0) for t in info.get("trucks", [])
        )
        distances.append(total_dist)
        charging_times.append(total_charge)
        truck_times = [t.get("total_time", 0.0) for t in info.get("trucks", [])]
        total_times.append(sum(truck_times))
        makespans.append(max(truck_times) if truck_times else 0.0)
        global_times.append(info.get("global_clock", env.global_clock))

    metrics = {
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "success_rate": np.mean(successes),
        "mean_total_distance": np.mean(distances),
        "std_total_distance": np.std(distances),
        "mean_charging_time": np.mean(charging_times),
        "std_charging_time": np.std(charging_times),
        "mean_total_time": np.mean(total_times),
        "std_total_time": np.std(total_times),
        "mean_makespan": np.mean(makespans),
        "std_makespan": np.std(makespans),
        "mean_global_clock": np.mean(global_times),
        "mean_steps": np.mean(step_counts),
    }
    if policy_type == "optimal" and optimal_charge_durations:
        sample = ", ".join(f"{d:.2f}" for d in optimal_charge_durations[:10])
        more = len(optimal_charge_durations) - 10
        if more > 0:
            sample += f", ... (+{more} more)"
        print(
            f"[Optimal] Charge durations selected (hours): {sample}\n"
            f"[Optimal] Total charging actions logged: {len(optimal_charge_durations)}"
        )
    return metrics


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
    policy_name_counts = {}
    for policy_path, policy_type in POLICIES:
        print(f"Loading: {policy_path} ({policy_type})...")
        policy, resolved_type = load_policy(policy_path, policy_type, gnn_state_space, config, device="cpu")
        if policy_path == "heuristic":
            name = "Heuristic"
        else:
            base_name = os.path.basename(policy_path.rstrip("/"))
            # Track duplicate names and append counter if needed
            if base_name in policy_name_counts:
                policy_name_counts[base_name] += 1
                name = f"{base_name} (#{policy_name_counts[base_name]})"
            else:
                policy_name_counts[base_name] = 1
                name = base_name
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
        f"{'Policy':<20} {'Reward':<18} {'Success %':<10} {'Total Time (h)':<18} "
        f"{'Makespan (h)':<18} {'Distance (km)':<18} {'Charging (h)':<15} "
        f"{'GlobalClk (h)':<15} {'Steps':<7}"
    )
    print("-" * 90)

    for name in sorted(results.keys()):
        r = results[name]
        display_name = _truncate_name(name, 20)
        reward_str = f"{r['mean_reward']:8.0f} ±{r['std_reward']:6.0f}"
        success_str = f"{r['success_rate']*100:6.1f}%"
        total_time_str = f"{r['mean_total_time']:8.1f} ±{r['std_total_time']:<6.1f}"
        makespan_str = f"{r['mean_makespan']:8.1f} ±{r['std_makespan']:<6.1f}"
        distance_str = f"{r['mean_total_distance']:8.1f} ±{r['std_total_distance']:<6.1f}"
        charging_str = f"{r['mean_charging_time']:6.1f} ±{r['std_charging_time']:<5.1f}"
        global_clock_str = f"{r['mean_global_clock']:6.1f}"
        steps_str = f"{r['mean_steps']:6.1f}"
        print(
            f"{display_name:<20} {reward_str:<18} {success_str:<10} {total_time_str:<18} "
            f"{makespan_str:<18} {distance_str:<18} {charging_str:<15} "
            f"{global_clock_str:<15} {steps_str:<7}"
        )

    print(f"{'='*90}\n")


if __name__ == "__main__":
    main()
