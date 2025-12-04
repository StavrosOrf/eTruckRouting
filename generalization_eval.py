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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from truck_env.models.event_driven_env import EventDrivenTruckEnv
from truck_env.state.gnn_state_space import GNNStateSpace
from truck_env.utils.utils import load_config

# Import compute_action_mask from train module
from train import compute_action_mask
from algo.policy_utils import load_policy

# ============ HARDCODED PARAMETERS ============
POLICIES = [    
    ("saved_models/NewFeasibleSpace_FixedGraph_ppo-variable_steps=512_epochs=5_ent=0.1_seed=0_gnnhd=32_mlphd=256/", "variable-ppo"),
    # ("saved_models/NewFeasibleSpace_FixedGraph_ppo-variable_steps=64_epochs=5_ent=0.1_seed=0_gnnhd=32_mlphd=256/", "variable-ppo"),
    # ("heuristic", "heuristic"),
]

# Grid parameters
NUM_TRUCKS_GRID = [2, 10, 40]
NUM_STOPS_GRID = [2, 3, 12]

CONFIG_FILE = "truck_env/config_files/config.yaml"
NUM_EVAL_SCENARIOS = 2
MAX_EPISODE_STEPS = 1000
SEED = 1000

# Parallel processing
USE_PARALLEL = True  # Run policies in parallel (each on GPU), configs sequential
NUM_PARALLEL_POLICIES = 4  # Number of policies to evaluate in parallel (adjust based on GPU memory)
# =============================================


def evaluate_policy_single_config(
    env, gnn_state_space, policy, resolved_type,
    num_episodes, seed, max_steps
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

        while not (done or truncated) and episode_steps < max_steps:
            gnn_state = gnn_state_space.get_state_GNN(env)

            if resolved_type == "heuristic":
                action = policy.get_action(env)
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


def print_metric_table(results_dict, metric_mean, metric_std, title, formatter=format_cell, policy_mapping=None):
    """
    Print a table for a single metric across all policies and configurations.
    
    Args:
        results_dict: Dict[policy_name][config_tuple] = results
        metric_mean: Key for the mean value (e.g., 'mean_reward')
        metric_std: Key for the std value (e.g., 'std_reward') or None
        title: Title of the table
        formatter: Function to format the cell value
        policy_mapping: Optional pre-computed mapping of short -> full names
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
            if config in results_dict[full_name]:
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
    else:
        base_name = os.path.basename(policy_path.rstrip('/'))
        policy_name = base_name
    
    # Load config
    config = load_config(CONFIG_FILE)
    
    # Load policy once
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
            env_info = environments[config_key]
            
            result = evaluate_policy_single_config(
                env=env_info["env"],
                gnn_state_space=env_info["gnn_state_space"],
                policy=policy,
                resolved_type=resolved_type,
                num_episodes=NUM_EVAL_SCENARIOS,
                seed=SEED,
                max_steps=MAX_EPISODE_STEPS,
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
    print(f"Max steps per episode: {MAX_EPISODE_STEPS}")
    
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
    
    # Create policy name mapping once
    policy_mapping = create_policy_mapping(results)
    
    # Print legend once at the top
    print_policy_legend(policy_mapping)
    
    # Print tables for each metric (passing the same mapping)
    print_metric_table(
        results, "mean_reward", "std_reward", 
        "REWARD", format_cell_int, policy_mapping
    )
    
    print_metric_table(
        results, "success_rate", None,
        "SUCCESS RATE", format_cell_percent, policy_mapping
    )
    
    print_metric_table(
        results, "mean_deliveries", "std_deliveries",
        "TOTAL DELIVERIES", format_cell, policy_mapping
    )
    
    print_metric_table(
        results, "mean_steps", "std_steps",
        "STEPS", format_cell, policy_mapping
    )
    
    print_metric_table(
        results, "mean_completion_time", "std_completion_time",
        "COMPLETION TIME (hours)", format_cell, policy_mapping
    )
    
    print_metric_table(
        results, "mean_total_distance", "std_total_distance",
        "TOTAL DISTANCE (km)", format_cell_int, policy_mapping
    )
    
    print_metric_table(
        results, "mean_charging_time", "std_charging_time",
        "CHARGING TIME (hours)", format_cell, policy_mapping
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
    output_file = "grid_evaluation_results.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to: {output_file}")
    
    # Also save formatted tables to text file
    import sys
    from io import StringIO
    
    output_txt = "grid_evaluation_results.txt"
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
