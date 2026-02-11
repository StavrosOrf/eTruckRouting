"""Grid evaluation: Compare policies across different environment configurations."""

import atexit
import copy
import os
import re
import sys
import time
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

from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.state.gnn_utils import create_default_gnn_space
from EVRoutingEnv.utils.utils import load_config
from EVRoutingEnv.baselines.optimal_gurobi import OptimalGurobiPolicy
from EVRoutingEnv.baselines.optimal_gurobi_simple import OptimalGurobiSimplePolicy
from EVRoutingEnv.baselines.optimal_vrp_single_truck import OptimalVRPSingleTruckPolicy
from EVRoutingEnv.state.action_mask import get_action_mask
from algo.policy_utils import load_policy

# SB3 imports
from stable_baselines3 import PPO, DQN
from sb3_contrib import MaskablePPO, QRDQN
# ============ HARDCODED PARAMETERS ============
POLICIES = [
    #Trained models on Electric Truck Routing 
    # ("saved_models/ppov_seq_1T3S_spu256_ep5_ent0.1_g32_m256_vk5_ck2_s0_8197/", "variable-ppo", "detour"),       
    # ("saved_models/ppov_seq_5T3S_spu256_ep5_ent0.1_g32_m256_vk5_ck3_s1_8197/", "variable-ppo", "detour"),    
    # ("saved_models/ppov_seq_10T3S_spu256_ep5_ent0.1_g32_m256_vk5_ck5_s0_8197/", "variable-ppo", "detour"),    
    # ("saved_models/ppov_seq_10T3S_spu256_ep5_ent0.1_g32_m256_vk5_ck2_s0_8197/", "variable-ppo", "detour"),     
    
    
    # ("optimal", "optimal", "base"),  # Gurobi-based optimal MILP solver
    #("optimal-simple", "optimal-simple", "base"),  # MP Robust - Gurobi solver with 20% energy safety margin
    
    # ("heuristic", "heuristic", "base"),    
        
    # eVRP Single TRUCK
    ("saved_models/Top5del_NewStateppov_1T10S_spu256_ep5_ent0.1_seed0_505/", "variable-ppo", "vrp"),        
    ("saved_models/1trucks_10stops/maskppo_seed0_20260209_164605/best_model.zip", "sb3-maskppo", "base"),
        
    # ("savings", "savings", "base"),
    # ("nn-2opt", "nn-2opt", "base"),
    # ("optimal-vrp", "optimal-vrp", "vrp"),
    
    # Baselines


]
# Grid parameters
NUM_TRUCKS_GRID = [1]
NUM_STOPS_GRID = [5, 10, 20, 30, 50]

# CONFIG_FILE = "EVRoutingEnv/config_files/config.yaml"
CONFIG_FILE = "EVRoutingEnv/config_files/config_vrp.yaml"
NUM_EVAL_SCENARIOS = 5
SEED = 1000

# Parallel processing
USE_PARALLEL = True  # Run per-episode tasks in parallel
NUM_WORKERS = 10
GPU_DEVICES = (0, 1, 2)
GPU_POLICY_TYPES = ("variable-ppo", "ppo-variable")
# =============================================

_WORKER_CACHE = {
    "gnn_state_space": {},
    "policies": {},
}


def _cleanup_children():
    for child in mp.active_children():
        try:
            child.terminate()
            child.join(timeout=2)
        except Exception:
            pass


def _extract_sb3_config(policy_path):
    """Return (num_trucks, num_stops) if encoded in path like '1trucks_10stops'."""
    for part in policy_path.rstrip("/").split("/"):
        match = re.search(r"(\d+)trucks_(\d+)stops", part)
        if match:
            return int(match.group(1)), int(match.group(2))
    return None


def _load_saved_gnn_state_config(policy_path):
    if not isinstance(policy_path, str):
        return {}
    if policy_path in (
        "heuristic",
        "optimal",
        "optimal_simple",
        "optimal-vrp",
        "optimal_vrp",
        "savings",
        "nn-2opt",
    ):
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


def _get_gnn_state_space(space, policy_path, env_init):
    mode = "vrp" if space == "vrp" else "nonflex"
    use_detour = space == "detour"
    gnn_cfg = _load_saved_gnn_state_config(policy_path)
    vrp_top_k = int(gnn_cfg.get("vrp_top_k_deliveries", 5))
    detour_top_k = int(gnn_cfg.get("detour_top_k_chargers", 2))
    cache_key = (space, vrp_top_k, detour_top_k)
    if cache_key not in _WORKER_CACHE["gnn_state_space"]:
        _WORKER_CACHE["gnn_state_space"][cache_key] = create_default_gnn_space(
            env_init,
            mode=mode,
            use_detour=use_detour,
            device="cpu",
            vrp_top_k_deliveries=vrp_top_k,
            detour_num_chargers_to_keep=detour_top_k,
        )
    return _WORKER_CACHE["gnn_state_space"][cache_key]


def _should_use_gpu(policy_type):
    if policy_type.startswith("sb3-"):
        return True
    return policy_type in GPU_POLICY_TYPES


def _load_policy_cached(policy_path, policy_type, gnn_state_space, config, device):
    if policy_type in ("optimal", "optimal_simple", "optimal-vrp", "optimal_vrp"):
        return None, policy_type
    cache_key = (policy_path, policy_type, device)
    if cache_key in _WORKER_CACHE["policies"]:
        return _WORKER_CACHE["policies"][cache_key]

    if policy_type.startswith("sb3-"):
        algo_name = policy_type.replace("sb3-", "")
        if algo_name == "ppo":
            policy = PPO.load(policy_path, device=device)
        elif algo_name == "maskppo":
            policy = MaskablePPO.load(policy_path, device=device)
        elif algo_name == "dqn":
            policy = DQN.load(policy_path, device=device)
        elif algo_name == "qrdqn":
            policy = QRDQN.load(policy_path, device=device)
        else:
            raise ValueError(f"Unknown SB3 algorithm: {algo_name}")
        resolved_type = policy_type
    else:
        policy, resolved_type = load_policy(policy_path, policy_type, gnn_state_space, config, device=device)

    _WORKER_CACHE["policies"][cache_key] = (policy, resolved_type)
    return policy, resolved_type


def _run_episode_task(task):
    (
        policy_name,
        policy_path,
        policy_type,
        gnn_space_type,
        num_trucks,
        num_stops,
        episode_idx,
        seed,
        config,
        device,
    ) = task

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(int(device.split(":")[-1]))
    else:
        device = "cpu"

    local_config = copy.deepcopy(config)
    local_config["environment"]["num_trucks"] = num_trucks
    local_config["environment"]["num_stops"] = num_stops

    env = EventDrivenTruckEnv(config=local_config, verbose=False, enable_plotting=False)
    try:
        gnn_state_space = _get_gnn_state_space(gnn_space_type, policy_path, env)
        env._default_gnn_state_space = gnn_state_space
        env.use_detour_mask = getattr(gnn_state_space, "use_detour", False)

        policy, resolved_type = _load_policy_cached(
            policy_path, policy_type, gnn_state_space, local_config, device
        )

        start_time = time.perf_counter()
        obs, info = env.reset(seed=seed + episode_idx)
        episode_reward, episode_steps = 0.0, 0
        done = truncated = False

        if resolved_type == "optimal":
            episode_policy = OptimalGurobiPolicy(verbose=False)
        elif resolved_type == "optimal_simple":
            episode_policy = OptimalGurobiSimplePolicy(verbose=False)
        elif resolved_type in ("optimal-vrp", "optimal_vrp"):
            episode_policy = OptimalVRPSingleTruckPolicy(verbose=False)
        else:
            episode_policy = policy

        while not (done or truncated):
            if resolved_type.startswith("sb3-"):
                if resolved_type == "sb3-maskppo":
                    action_masks = get_action_mask(env)
                    action, _states = episode_policy.predict(
                        obs, action_masks=action_masks, deterministic=True
                    )
                else:
                    action, _states = episode_policy.predict(obs, deterministic=True)
            else:
                if resolved_type in ("optimal", "optimal_simple", "optimal-vrp", "optimal_vrp"):
                    action = episode_policy.get_action(env)
                elif resolved_type in ("heuristic", "savings", "nn-2opt"):
                    action = episode_policy.get_action(env)
                elif resolved_type in ("ppo-variable", "variable-ppo"):
                    gnn_state = gnn_state_space.get_state_GNN(env)
                    raw_action = episode_policy.select_action(gnn_state, deterministic=True)
                    action = episode_policy.to_env_action(gnn_state, int(raw_action))
                else:
                    gnn_state = gnn_state_space.get_state_GNN(env)
                    mask = torch.tensor(get_action_mask(env), dtype=torch.bool)
                    raw_action = episode_policy.select_action(
                        gnn_state, deterministic=True, action_mask=mask
                    )
                    if isinstance(raw_action, tuple):
                        action = raw_action
                    else:
                        action = int(raw_action) % env.action_space.n

            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            episode_steps += 1

        exec_time = time.perf_counter() - start_time

        max_time_reached = 1.0 if truncated and env.global_clock >= env.max_time else 0.0
        max_steps_reached = 1.0 if truncated and episode_steps >= env.max_episode_steps else 0.0

        trucks_info = info["trucks"]
        total_dist = 0.0
        total_charge = 0.0
        total_sessions = 0
        total_waiting = 0.0
        total_routing = 0.0
        total_unloading = 0.0
        num_failed = 0
        num_deliveries = 0

        for t in trucks_info:
            total_dist += t.get("total_distance", 0.0)
            total_charge += t.get("total_charging_time", 0.0)
            total_sessions += t.get("num_charging_sessions", 0)
            total_waiting += t.get("waiting_time", 0.0)
            total_unloading += t.get("total_unloading_time", 0.0)

            if t.get("failed", False):
                num_failed += 1

            total_stops = len(t.get("delivery_sequence", []))
            remaining = t.get("deliveries_remaining", 0)
            if total_stops > 0:
                num_deliveries += max(0, total_stops - 1 - remaining)

            truck_total_time = t.get("total_time", 0.0)
            truck_charging_time = t.get("total_charging_time", 0.0)
            truck_unloading_time = t.get("total_unloading_time", 0.0)
            truck_waiting_time = t.get("waiting_time", 0.0)
            truck_routing_time = truck_total_time - truck_charging_time - truck_unloading_time - truck_waiting_time
            total_routing += max(0.0, truck_routing_time)

        if trucks_info:
            avg_soc = float(np.mean([t.get("battery_percentage", 0.0) for t in trucks_info]))
        else:
            avg_soc = 0.0

        return {
            "policy_name": policy_name,
            "num_trucks": num_trucks,
            "num_stops": num_stops,
            "episode_idx": episode_idx,
            "reward": episode_reward,
            "success": 1.0 if info["all_complete"] else 0.0,
            "distance": total_dist,
            "charging_time": total_charge,
            "steps": episode_steps,
            "completion_time": env.global_clock,
            "deliveries": num_deliveries,
            "charging_sessions": total_sessions,
            "waiting_time": total_waiting,
            "routing_time": total_routing,
            "unloading_time": total_unloading,
            "failures": num_failed,
            "avg_completion_soc": avg_soc,
            "exec_time": exec_time,
            "max_time_termination": max_time_reached,
            "max_steps_termination": max_steps_reached,
            "truncated": 1.0 if truncated else 0.0,
        }
    finally:
        env.close()


def _aggregate_episode_results(episode_results):
    rewards = []
    successes = []
    distances = []
    charging_times = []
    steps = []
    completion_times = []
    total_deliveries = []
    num_charging_sessions = []
    waiting_times = []
    routing_times = []
    unloading_times = []
    failures = []
    avg_completion_soc = []
    exec_times = []
    truncated_flags = []
    max_time_terminations = []
    max_steps_terminations = []

    for result in episode_results:
        rewards.append(result["reward"])
        successes.append(result["success"])
        distances.append(result["distance"])
        charging_times.append(result["charging_time"])
        steps.append(result["steps"])
        completion_times.append(result["completion_time"])
        total_deliveries.append(result["deliveries"])
        num_charging_sessions.append(result["charging_sessions"])
        waiting_times.append(result["waiting_time"])
        routing_times.append(result["routing_time"])
        unloading_times.append(result["unloading_time"])
        failures.append(result["failures"])
        avg_completion_soc.append(result["avg_completion_soc"])
        exec_times.append(result["exec_time"])
        truncated_flags.append(result["truncated"])
        max_time_terminations.append(result["max_time_termination"])
        max_steps_terminations.append(result["max_steps_termination"])

    return {
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "episode_rewards": rewards,
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
        "mean_charging_sessions": np.mean(num_charging_sessions),
        "std_charging_sessions": np.std(num_charging_sessions),
        "mean_waiting_time": np.mean(waiting_times),
        "std_waiting_time": np.std(waiting_times),
        "mean_routing_time": np.mean(routing_times),
        "std_routing_time": np.std(routing_times),
        "mean_unloading_time": np.mean(unloading_times),
        "std_unloading_time": np.std(unloading_times),
        "mean_failures": np.mean(failures),
        "std_failures": np.std(failures),
        "mean_completion_soc": np.mean(avg_completion_soc),
        "std_completion_soc": np.std(avg_completion_soc),
        "mean_exec_time": np.mean(exec_times),
        "std_exec_time": np.std(exec_times),
        "mean_truncated": np.mean(truncated_flags),
        "std_truncated": np.std(truncated_flags),
        "max_time_terminations": np.sum(max_time_terminations),
        "max_steps_terminations": np.sum(max_steps_terminations),
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


def _build_policy_name(policy_path, policy_type, policy_counter):
    if policy_path == "heuristic":
        return "Heuristic"
    if policy_path == "optimal":
        return "Optimal (Gurobi)"
    if policy_path in ("optimal_simple", "optimal-simple"):
        return "Optimal (Simple)"
    if policy_path in ("optimal-vrp", "optimal_vrp"):
        return "Optimal VRP"
    if policy_type.startswith("sb3-"):
        dir_path = os.path.dirname(policy_path) if policy_path.endswith(".zip") else policy_path
        base_name = os.path.basename(dir_path.rstrip("/"))
        return f"SB3-{base_name}"

    base_name = os.path.basename(policy_path.rstrip("/"))
    if base_name in policy_counter:
        policy_counter[base_name] += 1
        return f"{base_name}_v{policy_counter[base_name]}"
    policy_counter[base_name] = 1
    return base_name


def main():
    """Run grid evaluation across multiple policies and configurations."""

    atexit.register(_cleanup_children)

    print("="*120)
    print(f"GRID EVALUATION")
    print("="*120)
    print(f"\nPolicies: {len(POLICIES)}")
    for policy_entry in POLICIES:
        policy_path, policy_type = policy_entry[0], policy_entry[1]
        if policy_path == "heuristic":
            print("  - Heuristic")
        else:
            print(f"  - {os.path.basename(policy_path.rstrip('/'))} ({policy_type})")
    
    print(f"\nGrid configurations: {len(NUM_TRUCKS_GRID)} × {len(NUM_STOPS_GRID)} = {len(NUM_TRUCKS_GRID) * len(NUM_STOPS_GRID)} configs")
    print(f"  - Trucks: {NUM_TRUCKS_GRID}")
    print(f"  - Stops: {NUM_STOPS_GRID}")
    print(f"\nTotal evaluations: {len(POLICIES) * len(NUM_TRUCKS_GRID) * len(NUM_STOPS_GRID)} (each with {NUM_EVAL_SCENARIOS} episodes)")
    
    if USE_PARALLEL:
        print(f"Parallel mode: {NUM_WORKERS} workers (per-episode tasks)")
    else:
        print("Sequential mode (single GPU)")
    print()
    
    config = load_config(CONFIG_FILE)

    if any((len(entry) > 2 and entry[2] == "detour") for entry in POLICIES):
        if config["delivery"].get("enable_flexible_delivery_order", False):
            print(
                "Detected detour policies; forcing sequential delivery order (enable_flexible_delivery_order=False)."
            )
            config["delivery"]["enable_flexible_delivery_order"] = False

    # Build policy entries with names
    policies = {}
    policy_counter = {}
    for policy_entry in POLICIES:
        policy_path, policy_type = policy_entry[0], policy_entry[1]
        gnn_space_type = policy_entry[2] if len(policy_entry) > 2 else "base"
        name = _build_policy_name(policy_path, policy_type, policy_counter)
        policies[name] = {
            "path": policy_path,
            "type": policy_type,
            "gnn_space": gnn_space_type,
        }

    # Run evaluations
    results = {}
    completed_policies = 0

    print(f"\n{'='*120}")
    print("STARTING EVALUATIONS")
    print(f"{'='*120}\n")

    tasks = []
    expected_counts = {}
    gpu_counter = 0
    for policy_name, policy_info in policies.items():
        sb3_trained_config = None
        if policy_info["type"].startswith("sb3-"):
            sb3_trained_config = _extract_sb3_config(policy_info["path"])
        for num_trucks in NUM_TRUCKS_GRID:
            for num_stops in NUM_STOPS_GRID:
                config_key = (num_trucks, num_stops)
                if sb3_trained_config is not None and config_key != sb3_trained_config:
                    continue
                expected_counts[(policy_name, config_key)] = NUM_EVAL_SCENARIOS
                for episode_idx in range(NUM_EVAL_SCENARIOS):
                    if _should_use_gpu(policy_info["type"]):
                        gpu_id = GPU_DEVICES[gpu_counter % len(GPU_DEVICES)]
                        device = f"cuda:{gpu_id}"
                        gpu_counter += 1
                    else:
                        device = "cpu"
                    tasks.append(
                        (
                            policy_name,
                            policy_info["path"],
                            policy_info["type"],
                            policy_info["gnn_space"],
                            num_trucks,
                            num_stops,
                            episode_idx,
                            SEED,
                            config,
                            device,
                        )
                    )

    episode_results = {
        key: [None for _ in range(NUM_EVAL_SCENARIOS)] for key in expected_counts.keys()
    }

    try:
        if USE_PARALLEL:
            mp_context = mp.get_context("spawn")
            executor = ProcessPoolExecutor(max_workers=NUM_WORKERS, mp_context=mp_context)
            try:
                futures = [executor.submit(_run_episode_task, task) for task in tasks]
                for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating", leave=False):
                    result = future.result()
                    policy_name = result["policy_name"]
                    config_key = (result["num_trucks"], result["num_stops"])
                    episode_idx = result["episode_idx"]
                    episode_results[(policy_name, config_key)][episode_idx] = result
            finally:
                executor.shutdown(wait=True, cancel_futures=True)
                _cleanup_children()
        else:
            for task in tqdm(tasks, desc="Evaluating", leave=False):
                result = _run_episode_task(task)
                policy_name = result["policy_name"]
                config_key = (result["num_trucks"], result["num_stops"])
                episode_idx = result["episode_idx"]
                episode_results[(policy_name, config_key)][episode_idx] = result
    finally:
        _cleanup_children()

    for (policy_name, config_key), episodes in episode_results.items():
        if any(result is None for result in episodes):
            missing = [i for i, r in enumerate(episodes) if r is None]
            raise RuntimeError(f"Missing episode results for {policy_name} {config_key}: {missing}")
        results.setdefault(policy_name, {})[config_key] = _aggregate_episode_results(episodes)

    for policy_name, policy_results in results.items():
        completed_policies += 1
        success_rates = [r["success_rate"] for r in policy_results.values()]
        avg_success = np.mean(success_rates) * 100 if success_rates else 0
        print(
            f"\n✓ [{completed_policies}/{len(policies)}] {policy_name} completed | "
            f"Avg success: {avg_success:.1f}% across {len(policy_results)} configs"
        )
    
    print(f"\n{'='*120}")
    print(f"ALL EVALUATIONS COMPLETED! ({completed_policies}/{len(POLICIES)} policies)")
    print(f"{'='*120}")
    
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
    for policy_name, policy_info in policies.items():
        if policy_info["type"].startswith("sb3-"):
            detected = _extract_sb3_config(policy_info["path"])
            if detected:
                sb3_config_lookup[policy_name] = detected

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
    print_metric_table(
        results, "mean_charging_sessions", "std_charging_sessions",
        "CHARGING SESSIONS", format_cell, policy_mapping, sb3_config_lookup
    )
    print_metric_table(
        results, "mean_waiting_time", "std_waiting_time",
        "WAITING TIME (hours)", format_cell, policy_mapping, sb3_config_lookup
    )
    print_metric_table(
        results, "mean_routing_time", "std_routing_time",
        "ROUTING TIME (hours)", format_cell, policy_mapping, sb3_config_lookup
    )
    print_metric_table(
        results, "mean_unloading_time", "std_unloading_time",
        "UNLOADING TIME (hours)", format_cell, policy_mapping, sb3_config_lookup
    )
    print_metric_table(
        results, "mean_completion_soc", "std_completion_soc",
        "AVG COMPLETION SOC (%)", format_cell, policy_mapping, sb3_config_lookup
    )
    print_metric_table(
        results, "mean_exec_time", "std_exec_time",
        "EXEC TIME (s)", format_cell, policy_mapping, sb3_config_lookup
    )
    print_metric_table(
        results, "mean_failures", "std_failures",
        "FAILURES", format_cell, policy_mapping, sb3_config_lookup
    )
    print_metric_table(
        results, "max_time_terminations", None,
        "MAX TIME REACHED", format_cell_int, policy_mapping, sb3_config_lookup
    )
    print_metric_table(
        results, "max_steps_terminations", None,
        "MAX STEPS REACHED", format_cell_int, policy_mapping, sb3_config_lookup
    )
    
    # Save results to CSV
    print("\n" + "="*120)
    print("SAVING RESULTS")
    print("="*120)
    
    # Flatten results for DataFrame
    rows = []
    for policy_name, policy_results in results.items():
        for (num_trucks, num_stops), metrics in policy_results.items():
            metrics_filtered = {k: v for k, v in metrics.items() if k != "episode_rewards"}
            row = {
                "policy": policy_name,
                "num_trucks": num_trucks,
                "num_stops": num_stops,
                **metrics_filtered
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
        for policy_entry in POLICIES:
            policy_path, policy_type = policy_entry[0], policy_entry[1]
            if policy_path == "heuristic":
                print("  - Heuristic")
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
        print_metric_table(results, "mean_charging_sessions", "std_charging_sessions", "CHARGING SESSIONS", format_cell, policy_mapping)
        print_metric_table(results, "mean_waiting_time", "std_waiting_time", "WAITING TIME (hours)", format_cell, policy_mapping)
        print_metric_table(results, "mean_routing_time", "std_routing_time", "ROUTING TIME (hours)", format_cell, policy_mapping)
        print_metric_table(results, "mean_unloading_time", "std_unloading_time", "UNLOADING TIME (hours)", format_cell, policy_mapping)
        print_metric_table(results, "mean_completion_soc", "std_completion_soc", "AVG COMPLETION SOC (%)", format_cell, policy_mapping)
        print_metric_table(results, "mean_exec_time", "std_exec_time", "EXEC TIME (s)", format_cell, policy_mapping)
        print_metric_table(results, "mean_failures", "std_failures", "FAILURES", format_cell, policy_mapping)
        print_metric_table(results, "max_time_terminations", None, "MAX TIME REACHED", format_cell_int, policy_mapping)
        print_metric_table(results, "max_steps_terminations", None, "MAX STEPS REACHED", format_cell_int, policy_mapping)
        
        # Restore stdout
        sys.stdout = old_stdout
    
    print(f"✓ Formatted tables saved to: {output_txt}")
    print()


if __name__ == "__main__":
    main()
