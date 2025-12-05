"""Grid evaluation: Compare policies across different environment configurations."""

import copy
import os
import sys
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from EVRoutingEnv.models.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.state.gnn_state_space import GNNStateSpace
from EVRoutingEnv.utils.utils import load_config
from EVRoutingEnv.baselines.optimal_gurobi import OptimalGurobiPolicy

# Import compute_action_mask from train module
from scripts.training.train import compute_action_mask
from EVRoutingEnv.state.action_mask import get_action_mask
from algo.policy_utils import load_policy

# SB3 imports
from stable_baselines3 import PPO, DQN
from sb3_contrib import MaskablePPO, QRDQN
from sb3_contrib.common.maskable.utils import get_action_masks

# ============ HARDCODED PARAMETERS ============
POLICIES = [
    # GNN-based policies
    ("saved_models/NewFeasibleSpace_FixedGraph_ppo-variable_steps=1024_epochs=5_ent=0.1_seed=0_gnnhd=32_mlphd=256/", "variable-ppo"),    
    # SB3 policies
    ("saved_models/10trucks_3stops/maskppo_seed0_20251204_202440/best_model.zip", "sb3-maskppo"),
    # ("saved_models/10trucks_3stops/ppo_seed0_20251204_202437/best_model.zip", "sb3-ppo"),
    # ("saved_models/10trucks_3stops/dqn_seed0_20251204_202435/best_model.zip", "sb3-dqn"),
    # ("saved_models/10trucks_3stops/qrdqn_seed0_20251204_202442/best_model.zip", "sb3-qrdqn"),
    # Baselines
    # ("optimal", "optimal"),  # Gurobi-based optimal MILP solver
    ("heuristic", "heuristic"),
]

# Grid parameters
NUM_TRUCKS_GRID = [20, 100]
NUM_STOPS_GRID = [3, 10]

CONFIG_FILE = "EVRoutingEnv/config_files/config.yaml"
NUM_EVAL_SCENARIOS = 2
SEED = 1000

# Parallel processing
USE_PARALLEL = True  # Run policies in parallel (each on GPU), configs sequential
NUM_PARALLEL_POLICIES = 4  # Number of policies to evaluate in parallel (adjust based on GPU memory)
# =============================================


def evaluate_policy_single_config(
    env, gnn_state_space, policy, resolved_type,
    num_episodes, seed,
    sb3_model=None,
    sb3_type=None
):
    """
    Evaluate a single policy on a single configuration.
    Environment and policy are passed in (already initialized).
    """
    
    # Evaluate
    rewards, successes, distances, charging_times, steps, completion_times, total_deliveries = [], [], [], [], [], [], []

    for episode in range(num_episodes):
        obs, info = env.reset(seed=seed + episode)
        episode_reward, episode_steps = 0.0, 0
        done = truncated = False
        
        # Recreate optimal planner per episode to avoid stale plans across seeds
        episode_policy = OptimalGurobiPolicy(verbose=False) if resolved_type == "optimal" else policy

        while not (done or truncated):
            if sb3_model is not None:
                # SB3 policy: use raw observation
                # (don't compute GNN state for SB3 models)
                pass
            elif resolved_type == "optimal" or resolved_type == "heuristic":
                # Optimal and heuristic policies don't need GNN state
                pass
            else:
                # Custom policies: compute GNN state
                gnn_state = gnn_state_space.get_state_GNN(env)

            if sb3_model is not None:
                # SB3 policy
                if sb3_type == "maskppo":
                    # MaskablePPO: use action masks from environment
                    action_masks = get_action_mask(env)
                    action, _ = sb3_model.predict(obs, action_masks=action_masks, deterministic=True)
                else:
                    # PPO, DQN, QRDQN
                    action, _ = sb3_model.predict(obs, deterministic=True)
            else:
                # Custom policies
                if resolved_type == "optimal":
                    action = episode_policy.get_action(env)
                elif resolved_type == "heuristic":
                    action = episode_policy.get_action(env)
                elif resolved_type == "ppo-variable" or resolved_type == "variable-ppo":
                    raw_action = policy.select_action(gnn_state, deterministic=True)
                    action = policy.to_env_action(gnn_state, int(raw_action))
                else:  # ppo
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


def format_cell(mean, std):
    """Format a table cell with mean ± std."""
    return f"{mean:.1f} ±{std:.1f}"


def format_cell_int(mean, std):
    """Format a table cell with mean ± std (integers)."""
    return f"{mean:.0f} ±{std:.0f}"


def format_cell_percent(rate):
    """Format a success rate as percentage."""
    return f"{rate*100:.1f}%"


def create_policy_mapping(results_dict):
    """Create mapping of short names to full policy names."""
    policy_mapping = {}
    for policy_name in sorted(results_dict.keys()):
        if len(policy_name) > 30:
            short_name = policy_name[:30]
            # Ensure uniqueness by appending number if needed
            base_short = short_name
            counter = 1
            while short_name in policy_mapping:
                short_name = f"{base_short[:-2]}{counter:02d}"
                counter += 1
            policy_mapping[short_name] = policy_name
        else:
            policy_mapping[policy_name] = policy_name
    return policy_mapping


def print_policy_legend(policy_mapping):
    """Print legend for shortened policy names."""
    if any(short != full for short, full in policy_mapping.items()):
        print("\nPolicy Legend:")
        for short_name, full_name in sorted(policy_mapping.items()):
            if short_name != full_name:
                print(f"  {short_name} = {full_name}")
        print()


def print_metric_table(results_dict, metric_mean, metric_std, title, formatter=format_cell, policy_mapping=None, sb3_config_lookup=None):
    """
    Print a table for a single metric across all policies and configurations.
    
    Args:
        results_dict: Dict[policy_name][config_tuple] = results
        metric_mean: Key for the mean value (e.g., 'mean_reward')
        metric_std: Key for the std value (e.g., 'std_reward') or None
        title: Title of the table
        formatter: Function to format the cell value
        policy_mapping: Optional pre-computed mapping of short -> full names
        sb3_config_lookup: Optional dict mapping policy_name to (num_trucks, num_stops)
    """
    print(f"\n{'='*120}")
    print(f"{title}")
    print(f"{'='*120}\n")
    
    # Get all configurations
    configs = sorted(set(
        config for policy_results in results_dict.values() 
        for config in policy_results.keys()
    ))
    
    # Use provided policy mapping or create new one
    if policy_mapping is None:
        policy_mapping = create_policy_mapping(results_dict)
    
    # Header
    header = f"{'Policy':<30} |"
    for num_trucks, num_stops in configs:
        header += f" T={num_trucks:3d}, S={num_stops:3d} |" #order it left to right        
    print(header)
    print("-" * len(header))
    
    # Rows for each policy
    for short_name, full_name in sorted(policy_mapping.items()):
        row = f"{short_name:<30} |"
        for config in configs:
            dash = False
            if sb3_config_lookup and full_name in sb3_config_lookup:
                allowed_config = sb3_config_lookup[full_name]
                if config != allowed_config:
                    dash = True
            if dash:
                row += f" {'-':^12} |"
            elif config in results_dict[full_name]:
                r = results_dict[full_name][config]
                if metric_std and metric_std in r:
                    cell = formatter(r[metric_mean], r[metric_std])
                else:
                    cell = format_cell_percent(r[metric_mean])
                row += f" {cell:^12} |"
            else:
                row += f" {'N/A':^12} |"
        print(row)
    
    print()


def evaluate_single_policy_all_configs(policy_spec):
    """Evaluate a single policy across all configurations (runs in separate process with GPU)."""
    policy_path, policy_type = policy_spec

    # Generate policy name (keep full name)
    if policy_path == "heuristic":
        policy_name = "Heuristic"
    elif policy_path == "optimal":
        policy_name = "Optimal (Gurobi)"
    else:
        base_name = os.path.basename(policy_path.rstrip('/'))
        # For SB3 policies, extract algorithm name from folder (e.g., "ppo_seed0_20251204_202437" -> "PPO")
        if policy_type.startswith("sb3-"):
            # Get the algorithm folder name
            algo_folder = os.path.basename(os.path.dirname(policy_path.rstrip('/')))
            algo_name = algo_folder.split('_')[0].upper()  # Extract "ppo", "dqn", etc. and uppercase
            policy_name = algo_name
        else:
            policy_name = base_name

    # Load config
    config = load_config(CONFIG_FILE)

    # Detect SB3 policy type and trained config
    sb3_model = None
    sb3_type = None
    sb3_trained_config = None
    if policy_type.startswith("sb3-"):
        # Extract config from path (e.g. .../10trucks_3stops/algorithm_name/...)
        path_parts = policy_path.rstrip('/').split('/')
        import re
        for part in path_parts:
            m = re.search(r"(\d+)trucks_(\d+)stops", part)
            if m:
                sb3_trained_config = (int(m.group(1)), int(m.group(2)))
                break
        
        # Map type to SB3 class
        sb3_type = policy_type.replace("sb3-", "")
        if sb3_type == "ppo":
            sb3_model = PPO.load(policy_path, device="cpu")
        elif sb3_type == "maskppo":
            sb3_model = MaskablePPO.load(policy_path, device="cpu")
        elif sb3_type == "dqn":
            sb3_model = DQN.load(policy_path, device="cpu")
        elif sb3_type == "qrdqn":
            sb3_model = QRDQN.load(policy_path, device="cpu")
        else:
            raise ValueError(f"Unknown SB3 type: {sb3_type}")
        resolved_type = None
        policy = None
    elif policy_type == "optimal":
        # Optimal (Gurobi) policy
        try:
            policy = OptimalGurobiPolicy(verbose=False)
        except ImportError as exc:
            raise RuntimeError(
                "Optimal (Gurobi) policy requires gurobipy to be installed."
            ) from exc
        resolved_type = "optimal"
    else:
        # Custom policies
        config_temp = copy.deepcopy(config)
        config_temp["environment"]["num_trucks"] = NUM_TRUCKS_GRID[0]
        config_temp["environment"]["num_stops"] = NUM_STOPS_GRID[0]

        env_temp = EventDrivenTruckEnv(config=config_temp, verbose=False, enable_plotting=False)
        gnn_state_space_temp = GNNStateSpace(
            num_trucks=NUM_TRUCKS_GRID[0],
            num_stops=NUM_STOPS_GRID[0],
            max_time=config_temp["environment"]["max_time"],
            num_charging_nodes=env_temp.num_charging_nodes,
            verbose=False,
        )
        policy, resolved_type = load_policy(policy_path, policy_type, gnn_state_space_temp, config_temp, device="cuda")
        env_temp.close()

    # Create environments for each configuration
    environments = {}
    for num_trucks in NUM_TRUCKS_GRID:
        for num_stops in NUM_STOPS_GRID:
            config_copy = copy.deepcopy(config)
            config_copy["environment"]["num_trucks"] = num_trucks
            config_copy["environment"]["num_stops"] = num_stops

            env = EventDrivenTruckEnv(config=config_copy, verbose=False, enable_plotting=False)
            gnn_state_space = GNNStateSpace(
                num_trucks=num_trucks,
                num_stops=num_stops,
                max_time=config_copy["environment"]["max_time"],
                num_charging_nodes=env.num_charging_nodes,
                verbose=False,
            )

            environments[(num_trucks, num_stops)] = {
                "env": env,
                "gnn_state_space": gnn_state_space
            }

    # Evaluate across all configurations
    policy_results = {}
    for num_trucks in NUM_TRUCKS_GRID:
        for num_stops in NUM_STOPS_GRID:
            config_key = (num_trucks, num_stops)
            
            # Skip SB3 policies if not their trained config
            if sb3_model is not None and sb3_trained_config is not None:
                if config_key != sb3_trained_config:
                    continue
            
            env_info = environments[config_key]

            result = evaluate_policy_single_config(
                env=env_info["env"],
                gnn_state_space=env_info["gnn_state_space"],
                policy=policy,
                resolved_type=resolved_type,
                num_episodes=NUM_EVAL_SCENARIOS,
                seed=SEED,
                sb3_model=sb3_model,
                sb3_type=sb3_type
            )

            result["num_trucks"] = num_trucks
            result["num_stops"] = num_stops
            policy_results[config_key] = result

    # Cleanup
    for env_info in environments.values():
        env_info["env"].close()

    return policy_name, policy_results


def main():
    """Run grid evaluation across multiple policies and configurations."""
    
    print("="*120)
    print(f"GRID EVALUATION")
    print("="*120)
    print(f"\nPolicies: {len(POLICIES)}")
    for policy_path, policy_type in POLICIES:
        if policy_path == "heuristic":
            print(f"  - Heuristic")
        else:
            print(f"  - {os.path.basename(policy_path.rstrip('/'))} ({policy_type})")
    
    print(f"\nGrid configurations: {len(NUM_TRUCKS_GRID)} × {len(NUM_STOPS_GRID)} = {len(NUM_TRUCKS_GRID) * len(NUM_STOPS_GRID)} configs")
    print(f"  - Trucks: {NUM_TRUCKS_GRID}")
    print(f"  - Stops: {NUM_STOPS_GRID}")
    print(f"\nTotal evaluations: {len(POLICIES) * len(NUM_TRUCKS_GRID) * len(NUM_STOPS_GRID)} (each with {NUM_EVAL_SCENARIOS} episodes)")
    
    if USE_PARALLEL:
        print(f"Parallel mode: {NUM_PARALLEL_POLICIES} policies in parallel (each on GPU)")
    else:
        print("Sequential mode (single GPU)")
    print()
    
    # Run evaluations
    results = {}
    
    if USE_PARALLEL:
        # Parallel execution: each policy runs in its own process with GPU
        mp.set_start_method('spawn', force=True)
        with ProcessPoolExecutor(max_workers=NUM_PARALLEL_POLICIES) as executor:
            futures = {executor.submit(evaluate_single_policy_all_configs, policy_spec): policy_spec 
                      for policy_spec in POLICIES}
            
            with tqdm(total=len(POLICIES), desc="Policies") as pbar:
                for future in as_completed(futures):
                    try:
                        policy_name, policy_results = future.result()
                        results[policy_name] = policy_results
                        pbar.set_description(f"Completed {policy_name[:20]}")
                        pbar.update(1)
                    except Exception as e:
                        policy_spec = futures[future]
                        print(f"\nError evaluating {policy_spec[0]}: {e}")
                        import traceback
                        traceback.print_exc()
                        pbar.update(1)
    else:
        # Sequential execution
        with tqdm(total=len(POLICIES), desc="Policies") as pbar:
            for policy_spec in POLICIES:
                pbar.set_description(f"Evaluating {policy_spec[0][:20]}")
                policy_name, policy_results = evaluate_single_policy_all_configs(policy_spec)
                results[policy_name] = policy_results
                pbar.update(1)
    
    print("\nDone!")
    
    print("\n" + "="*120)
    print("RESULTS")
    print("="*120)
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("results", "grid_eval", f"grid_eval_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nSaving results to: {output_dir}\n")
    
    # Create policy name mapping once
    policy_mapping = create_policy_mapping(results)
    
    # Print legend once at the top
    print_policy_legend(policy_mapping)
    
    # Build SB3 config lookup: {policy_name: (num_trucks, num_stops)}
    sb3_config_lookup = {}
    for policy_path, policy_type in POLICIES:
        if policy_type.startswith("sb3-"):
            # Try to extract config from path (e.g. .../10trucks_3stops/...)
            base = os.path.basename(os.path.dirname(policy_path.rstrip('/')))
            import re
            m = re.search(r"(\d+)trucks_(\d+)stops", base)
            if m:
                num_trucks = int(m.group(1))
                num_stops = int(m.group(2))
                policy_name = os.path.basename(policy_path.rstrip('/'))
                sb3_config_lookup[policy_name] = (num_trucks, num_stops)

    # Print tables for each metric (passing the same mapping)
    print_metric_table(
        results, "mean_reward", "std_reward", 
        "REWARD", format_cell_int, policy_mapping, sb3_config_lookup
    )
    print_metric_table(
        results, "success_rate", None,
        "SUCCESS RATE", format_cell_percent, policy_mapping, sb3_config_lookup
    )
    print_metric_table(
        results, "mean_deliveries", "std_deliveries",
        "TOTAL DELIVERIES", format_cell, policy_mapping, sb3_config_lookup
    )
    print_metric_table(
        results, "mean_steps", "std_steps",
        "STEPS", format_cell, policy_mapping, sb3_config_lookup
    )
    print_metric_table(
        results, "mean_completion_time", "std_completion_time",
        "COMPLETION TIME (hours)", format_cell, policy_mapping, sb3_config_lookup
    )
    print_metric_table(
        results, "mean_total_distance", "std_total_distance",
        "TOTAL DISTANCE (km)", format_cell_int, policy_mapping, sb3_config_lookup
    )
    print_metric_table(
        results, "mean_charging_time", "std_charging_time",
        "CHARGING TIME (hours)", format_cell, policy_mapping, sb3_config_lookup
    )
    
    # Save results to CSV
    print("\n" + "="*120)
    print("SAVING RESULTS")
    print("="*120)
    
    # Flatten results for DataFrame
    rows = []
    for policy_name, policy_results in results.items():
        for (num_trucks, num_stops), metrics in policy_results.items():
            row = {
                "policy": policy_name,
                "num_trucks": num_trucks,
                "num_stops": num_stops,
                **metrics
            }
            rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Sort by policy, num_trucks, num_stops
    df = df.sort_values(["policy", "num_trucks", "num_stops"])
    
    # Save to CSV
    output_file = os.path.join(output_dir, "grid_evaluation_results.csv")
    df.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to: {output_file}")
    
    # Also save formatted tables to text file
    import sys
    from io import StringIO
    
    output_txt = os.path.join(output_dir, "grid_evaluation_results.txt")
    with open(output_txt, "w") as f:
        # Redirect stdout to file
        old_stdout = sys.stdout
        sys.stdout = f
        
        print("="*120)
        print("GRID EVALUATION RESULTS")
        print("="*120)
        print(f"\nPolicies: {len(POLICIES)}")
        for policy_path, policy_type in POLICIES:
            if policy_path == "heuristic":
                print(f"  - Heuristic")
            else:
                print(f"  - {os.path.basename(policy_path.rstrip('/'))} ({policy_type})")
        
        print(f"\nGrid: {NUM_TRUCKS_GRID} trucks × {NUM_STOPS_GRID} stops")
        print(f"Episodes per config: {NUM_EVAL_SCENARIOS}")
        print(f"Seed: {SEED}")
        
        # Print legend once
        print_policy_legend(policy_mapping)
        
        # Print all tables
        print_metric_table(results, "mean_reward", "std_reward", "REWARD", format_cell_int, policy_mapping)
        print_metric_table(results, "success_rate", None, "SUCCESS RATE", format_cell_percent, policy_mapping)
        print_metric_table(results, "mean_deliveries", "std_deliveries", "TOTAL DELIVERIES", format_cell, policy_mapping)
        print_metric_table(results, "mean_steps", "std_steps", "STEPS", format_cell, policy_mapping)
        print_metric_table(results, "mean_completion_time", "std_completion_time", "COMPLETION TIME (hours)", format_cell, policy_mapping)
        print_metric_table(results, "mean_total_distance", "std_total_distance", "TOTAL DISTANCE (km)", format_cell_int, policy_mapping)
        print_metric_table(results, "mean_charging_time", "std_charging_time", "CHARGING TIME (hours)", format_cell, policy_mapping)
        
        # Restore stdout
        sys.stdout = old_stdout
    
    print(f"✓ Formatted tables saved to: {output_txt}")
    print()


if __name__ == "__main__":
    main()
