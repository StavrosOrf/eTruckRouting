"""Evaluate and compare different policies on multiple scenarios."""

import copy
import os
import sys
import numpy as np
import torch
from tqdm import tqdm

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from EVRoutingEnv.baselines.optimal_gurobi import OptimalGurobiPolicy
from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.state.gnn_state_space import GNNStateSpace
from EVRoutingEnv.utils.utils import load_config
from EVRoutingEnv.state.action_mask import get_action_mask

# Import compute_action_mask from train module
from scripts.training.train_PPO_Variable import compute_action_mask
from algo.policy_utils import load_policy

# Import SB3 algorithms
from stable_baselines3 import PPO, DQN
from sb3_contrib import MaskablePPO, QRDQN

# ============ HARDCODED PARAMETERS ============
POLICIES = [
    # GNN-based policies
    # ("saved_models/NewFeasibleSpace_FixedGraph_ppo-variable_steps=1024_epochs=5_ent=0.1_seed=0_gnnhd=32_mlphd=256/", "variable-ppo"),
    # ("saved_models/curriculum_staged_seed0/", "variable-ppo"),
    ("saved_models/NewActions_Traffic_CCCV_steps=128_epochs=10_ent=0.1_seed=0_gnnhd=32_mlphd=256/", "variable-ppo"),
    ("saved_models/NewActions_Traffic_CCCV_steps=512_epochs=10_ent=0.1_seed=0_gnnhd=32_mlphd=256/", "variable-ppo"),
    # ("saved_models/curriculum_mixed_seed0/", "variable-ppo"),
    # ("saved_models/curriculum_uniform_seed0/", "variable-ppo"),
    # SB3 policies
    # ("saved_models/10trucks_3stops/maskppo_seed0_20251204_202440/best_model.zip", "sb3-maskppo"),
    # ("saved_models/10trucks_3stops/ppo_seed0_20251204_202437/best_model.zip", "sb3-ppo"),
    # ("saved_models/10trucks_3stops/dqn_seed0_20251204_202435/best_model.zip", "sb3-dqn"),
    # ("saved_models/10trucks_3stops/qrdqn_seed0_20251204_202442/best_model.zip", "sb3-qrdqn"),
    # Baselines
    ("optimal", "optimal"),  # Gurobi-based optimal MILP solver
    # ("heuristic", "heuristic"),
]
CONFIG_FILE = "EVRoutingEnv/config_files/config.yaml"
NUM_TRUCKS = 10  # Must match the configuration used during training
NUM_STOPS = 5
NUM_EVAL_SCENARIOS = 10
SEED = 1000
# =============================================


def evaluate_policy(
    env, policy, gnn_state_space, policy_type, num_episodes, seed, config
):
    """Evaluate a policy over multiple episodes."""
    rewards, successes, distances, charging_times, steps, completion_times, total_deliveries = [], [], [], [], [], [], []
    
    # Check if this is an SB3 policy
    is_sb3_policy = policy_type.startswith("sb3-")

    for episode in tqdm(range(num_episodes), desc="Evaluating", leave=False):
        obs, info = env.reset(seed=seed + episode)
        episode_reward, episode_steps = 0.0, 0
        done = truncated = False
        # Recreate optimal planner per episode to avoid stale plans across seeds
        episode_policy = OptimalGurobiPolicy(verbose=False) if policy_type == "optimal" else policy

        while not (done or truncated):
            if is_sb3_policy:
                # SB3 policies use observation directly
                if policy_type == "sb3-maskppo":
                    # MaskablePPO requires action masks
                    action_masks = get_action_mask(env)
                    action, _states = policy.predict(obs, action_masks=action_masks, deterministic=True)
                else:
                    # Standard SB3 policies (PPO, DQN, QRDQN)
                    action, _states = policy.predict(obs, deterministic=True)
            else:
                # Custom policies
                if policy_type == "optimal":
                    action = episode_policy.get_action(env)
                elif policy_type == "heuristic":
                    action = policy.get_action(env)
                elif policy_type == "ppo-variable" or policy_type == "variable-ppo":
                    gnn_state = gnn_state_space.get_state_GNN(env)
                    raw_action = policy.select_action(gnn_state, deterministic=True)
                    action = policy.to_env_action(gnn_state, int(raw_action))
                else:  # ppo
                    gnn_state = gnn_state_space.get_state_GNN(env)
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
        successes.append(1.0 if info["all_complete"] else 0.0)
        steps.append(episode_steps)
        completion_times.append(env.global_clock)

        # Extract metrics from truck info
        trucks_info = info["trucks"]
        total_dist = sum(t["total_distance"] for t in trucks_info)
        total_charge = sum(t["total_charging_time"] for t in trucks_info)
        
        # Count deliveries completed: total sequence length - remaining deliveries - 1 (for start)
        num_deliveries = 0
        for t in trucks_info:
            total_stops = len(t["delivery_sequence"])
            remaining = t["deliveries_remaining"]
            # Deliveries made = total stops - start node - remaining deliveries
            if total_stops > 0:
                num_deliveries += max(0, total_stops - 1 - remaining)
        
        distances.append(total_dist)
        charging_times.append(total_charge)
        total_deliveries.append(num_deliveries)

    return {
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "success_rate": np.mean(successes),
        "mean_total_distance": np.mean(distances),
        "std_total_distance": np.std(distances),
        "mean_charging_time": np.mean(charging_times),
        "std_charging_time": np.std(charging_times),
        "mean_steps": np.mean(steps),
        "std_steps": np.std(steps),
        "mean_completion_time": np.mean(completion_times),
        "std_completion_time": np.std(completion_times),
        "mean_deliveries": np.mean(total_deliveries),
        "std_deliveries": np.std(total_deliveries),
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
    policy_counter = {}  # Track duplicate names
    for policy_path, policy_type in POLICIES:
        print(f"Loading: {policy_path} ({policy_type})...")
        
        # Handle SB3 policies differently
        if policy_type.startswith("sb3-"):
            # Load SB3 model
            algo_name = policy_type.replace("sb3-", "")
            if algo_name == "ppo":
                policy = PPO.load(policy_path)
            elif algo_name == "maskppo":
                policy = MaskablePPO.load(policy_path)
            elif algo_name == "dqn":
                policy = DQN.load(policy_path)
            elif algo_name == "qrdqn":
                policy = QRDQN.load(policy_path)
            else:
                raise ValueError(f"Unknown SB3 algorithm: {algo_name}")
            resolved_type = policy_type
            print(f"  Loaded SB3 {algo_name.upper()} model")
        elif policy_type == "optimal":
            try:
                policy = OptimalGurobiPolicy(verbose=False)
            except ImportError as exc:
                raise RuntimeError(
                    "Optimal (Gurobi) policy requires gurobipy to be installed."
                ) from exc
            resolved_type = "optimal"
        else:
            # Load GNN-based policies using existing function
            policy, resolved_type = load_policy(policy_path, policy_type, gnn_state_space, config)
        
        # Generate unique name for each policy
        if policy_path == "heuristic":
            name = "Heuristic"
        elif policy_path == "optimal":
            name = "Optimal (Gurobi)"
        elif policy_type.startswith("sb3-"):
            # For SB3 models, create readable name from directory path
            dir_path = os.path.dirname(policy_path) if policy_path.endswith('.zip') else policy_path
            base_name = os.path.basename(dir_path.rstrip('/'))
            base_name = base_name[:40]  # Truncate
            name = f"SB3-{base_name}"
        else:
            base_name = os.path.basename(policy_path.rstrip('/'))
            # Truncate name to first 30 characters
            base_name = base_name[:30]
            # Handle duplicate names by appending counter
            if base_name in policy_counter:
                policy_counter[base_name] += 1
                name = f"{base_name}_v{policy_counter[base_name]}"
            else:
                policy_counter[base_name] = 1
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
            config,
        )

    eval_env.close()

    # Print results table
    print(f"\n{'='*160}")
    print(f"RESULTS (averaged over {NUM_EVAL_SCENARIOS} scenarios)\n")
    print(
        f"{'Policy':<50} {'Reward':<18} {'Success':<12} {'Deliveries':<15} {'Steps':<15} "
        f"{'Time (h)':<18} {'Distance (km)':<18} {'Charging (h)':<18}"
    )
    print("-" * 160)

    for name in sorted(results.keys()):
        r = results[name]
        print(
            f"{name:<50} "
            f"{r['mean_reward']:>7.0f}±{r['std_reward']:<7.0f}  "
            f"{r['success_rate']*100:>5.1f}%      "
            f"{r['mean_deliveries']:>6.1f}±{r['std_deliveries']:<5.1f}  "
            f"{r['mean_steps']:>6.1f}±{r['std_steps']:<5.1f}  "
            f"{r['mean_completion_time']:>7.1f}±{r['std_completion_time']:<7.1f}  "
            f"{r['mean_total_distance']:>7.0f}±{r['std_total_distance']:<7.0f}  "
            f"{r['mean_charging_time']:>7.1f}±{r['std_charging_time']:<7.1f}"
        )

    print(f"{'='*160}\n")


if __name__ == "__main__":
    main()
