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
from truck_env.optimization.gurobi_solver import GurobiOptimalPlanner

# Import compute_action_mask from train module
from train import compute_action_mask
from algo.policy_utils import load_policy

# ============ HARDCODED PARAMETERS ============
POLICIES = [
    # ("saved_models/ppo-variable_steps=128_epochs=10_ent=0.1_seed=0_gnnhd=64_mlphd=256/", "variable-ppo"),
    # ("saved_models/SanityCheck_ppo-variable_steps=128_epochs=5_ent=0.1_seed=0_gnnhd=32_mlphd=256/", "variable-ppo"),
    # ("saved_models/NewFeasibleSpace_FixedGraph_ppo-variable_steps=128_epochs=5_ent=0.1_seed=0_gnnhd=32_mlphd=256/", "variable-ppo"),
    # ("saved_models/NewFeasibleSpace_FixedGraph_ppo-variable_steps=512_epochs=5_ent=0.1_seed=0_gnnhd=32_mlphd=256/", "variable-ppo"),
    ("optimal", "optimal"),  # Gurobi-based optimal MILP solver
    # ("saved_models/debug_ppo_var_eval", "ppo-variable"),
    ("heuristic", "heuristic"),
]
CONFIG_FILE = "truck_env/config_files/config.yaml"
NUM_TRUCKS = 2
NUM_STOPS = 3
NUM_EVAL_SCENARIOS = 10
MAX_EPISODE_STEPS = 200
SEED = 1000
# =============================================


def evaluate_policy(
    env, policy, gnn_state_space, policy_type, num_episodes, seed, max_steps, config
):
    """Evaluate a policy over multiple episodes."""
    rewards, successes, distances, charging_times, steps, completion_times, total_deliveries = [], [], [], [], [], [], []

    for episode in tqdm(range(num_episodes), desc="Evaluating", leave=False):
        obs, info = env.reset(seed=seed + episode)
        episode_reward, episode_steps = 0.0, 0
        done = truncated = False

        # For optimal policy, solve once and execute the solution
        if policy_type == "optimal":
            # Solve optimal plan for this scenario
            solver = GurobiOptimalPlanner(env,
                                          config,
                                          time_limit=180,
                                          verbose=False)
            solver.build_model()
            success = solver.solve()
            
            if not success:
                # Optimal solver failure indicates a problem with the model
                raise RuntimeError(
                    f"Optimal solver failed for episode {episode} (seed={seed + episode}). "
                    "This indicates an issue with the problem formulation or scenario feasibility. "
                    "Check the Gurobi logs for details."
                )
            
            # Execute the optimal solution (step through environment)
            # For now, just extract final metrics from the solution
            episode_reward = solver.solution['objective']  # Use negative for reward
            episode_steps = sum(len(solver.solution['routes'][k]) - 1 for k in solver.solution['routes'])
            
            # Mark episode as complete
            done = True
            
            # Get final info (need to step through or use solution metrics)
            # For simplicity, extract from solution
            completion_time = solver.solution['makespan']
            total_dist = sum(
                sum(
                    env.transport_graph.get_path_energy(solver.solution['routes'][k][i], 
                                                        solver.solution['routes'][k][i+1])
                    for i in range(len(solver.solution['routes'][k]) - 1)
                )
                for k in solver.solution['routes']
            )
            total_charge = sum(
                sum(info['duration'] for info in solver.solution['charging'][k].values())
                for k in solver.solution['charging']
            )
            num_deliveries = sum(len(env.trucks[k].delivery_sequence) - 1 for k in range(len(env.trucks)))
            
            rewards.append(-completion_time)  # Negative time as reward
            successes.append(1.0)
            steps.append(episode_steps)
            completion_times.append(completion_time)
            distances.append(total_dist)
            charging_times.append(total_charge)
            total_deliveries.append(num_deliveries)
            continue

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
        policy, resolved_type = load_policy(policy_path, policy_type, gnn_state_space, config)
        
        # Generate unique name for each policy
        if policy_path == "heuristic":
            name = "Heuristic"
        elif policy_path == "optimal":
            name = "Optimal (Gurobi)"
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
            MAX_EPISODE_STEPS,
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
