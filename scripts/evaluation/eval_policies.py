"""Evaluate and compare different policies on multiple scenarios."""

import copy
import os
import re
import sys
import numpy as np
import torch
from tqdm import tqdm

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from EVRoutingEnv.baselines.optimal_gurobi import OptimalGurobiPolicy
from EVRoutingEnv.baselines.optimal_gurobi_simple import OptimalGurobiSimplePolicy
from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.state.gnn_utils import create_default_gnn_space
from EVRoutingEnv.utils.utils import load_config
from EVRoutingEnv.state.action_mask import get_action_mask
from algo.policy_utils import load_policy

# Import SB3 algorithms
from stable_baselines3 import PPO, DQN
from sb3_contrib import MaskablePPO, QRDQN

# ============ HARDCODED PARAMETERS ============
POLICIES = [
    # GNN-based policies: (path, policy_type, gnn_state_space)
    # ("saved_models/Base_r=500_updatedDelivery_steps=256_epochs=5_ent=0.01_seed=0_gnnhd=32_mlphd=256_6343/", "variable-ppo", "detour"),
    # ("saved_models/OneChargePerDelivery_Base_r=500_updatedDelivery_steps=512_epochs=5_ent=0.1_seed=0_gnnhd=32_mlphd=256_9306/", "variable-ppo", "detour"),
    # ("saved_models/Top5Charger_Fallback_OneChargePerDelivery_Base_r=500_updatedDelivery_steps=256_epochs=5_ent=0.1_seed=0_gnnhd=32_mlphd=256_7597/", "variable-ppo", "detour"),
    # ("saved_models/OneChargePerDelivery_Base_r=500_updatedDelivery_steps=256_epochs=5_ent=0.1_seed=0_gnnhd=32_mlphd=256_9303/", "variable-ppo", "detour"),
    # ("saved_models/Top1Charger_Fallback_OneChargePerDelivery_Base_r=500_updatedDelivery_steps=256_epochs=5_ent=0.1_seed=0_gnnhd=32_mlphd=256_9652/", "variable-ppo", "detour"),
    ("saved_models/ppov_1T10S_spu256_ep5_ent0.1_seed0_1447/", "variable-ppo", "vrp"),
    # SB3 policies
    # ("saved_models/1trucks_10stops/maskppo_seed0_20251212_070042/best_model.zip", "sb3-maskppo", "base"),
    # ("saved_models/10trucks_3stops/ppo_seed0_20251204_202437/best_model.zip", "sb3-ppo", "base"),
    # ("saved_models/10trucks_3stops/dqn_seed0_20251204_202435/best_model.zip", "sb3-dqn", "base"),
    # ("saved_models/10trucks_3stops/qrdqn_seed0_20251204_202442/best_model.zip", "sb3-qrdqn", "base"),
    # Baselines
    # ("optimal", "optimal", "base"),  # Gurobi-based optimal MILP solver
    # ("optimal-simple", "optimal-simple", "base"),  # MP Robust - Gurobi solver with 20% energy safety margin
    # ("heuristic", "heuristic", "base"),
]
# CONFIG_FILE = "EVRoutingEnv/config_files/config.yaml"
CONFIG_FILE = "EVRoutingEnv/config_files/config_vrp.yaml"
NUM_TRUCKS = 1  # Must match the configuration used during training
NUM_STOPS = 10
NUM_EVAL_SCENARIOS = 100
SEED = 1000
# =============================================

def _extract_sb3_config(policy_path):
    """Return (num_trucks, num_stops) if encoded in path like '1trucks_10stops'."""
    for part in policy_path.rstrip("/").split("/"):
        match = re.search(r"(\d+)trucks_(\d+)stops", part)
        if match:
            return int(match.group(1)), int(match.group(2))
    return None


def evaluate_policy(
    env, policy, gnn_state_space, policy_type, num_episodes, seed, config
):
    """Evaluate a policy over multiple episodes."""
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
    max_time_terminations = []
    max_steps_terminations = []
    
    # Ensure action_mask uses the provided GNN state space
    env._default_gnn_state_space = gnn_state_space
    env.use_detour_mask = getattr(gnn_state_space, "use_detour", False)

    # Check if this is an SB3 policy
    is_sb3_policy = policy_type.startswith("sb3-")

    for episode in tqdm(range(num_episodes), desc="Evaluating", leave=False):
        obs, info = env.reset(seed=seed + episode)
        episode_reward, episode_steps = 0.0, 0
        done = truncated = False
        # Recreate optimal planners per episode to avoid stale plans across seeds
        if policy_type == "optimal":
            episode_policy = OptimalGurobiPolicy(verbose=False)
        elif policy_type == "optimal-simple":
            episode_policy = OptimalGurobiSimplePolicy(verbose=False)
        else:
            episode_policy = policy

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
                if policy_type == "optimal" or policy_type == "optimal-simple":
                    action = episode_policy.get_action(env)
                elif policy_type == "heuristic":
                    action = policy.get_action(env)
                elif policy_type == "ppo-variable" or policy_type == "variable-ppo":
                    gnn_state = gnn_state_space.get_state_GNN(env)
                    raw_action = policy.select_action(gnn_state, deterministic=True)
                    action = policy.to_env_action(gnn_state, int(raw_action))
                else:  # ppo
                    gnn_state = gnn_state_space.get_state_GNN(env)
                    mask = torch.tensor(get_action_mask(env), dtype=torch.bool)
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
        
        # Track termination reasons
        max_time_reached = 1.0 if truncated and env.global_clock >= env.max_time else 0.0
        max_steps_reached = 1.0 if truncated and episode_steps >= env.max_episode_steps else 0.0
        max_time_terminations.append(max_time_reached)
        max_steps_terminations.append(max_steps_reached)

        # Extract detailed metrics from truck info
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
            total_dist += t["total_distance"]
            total_charge += t["total_charging_time"]
            total_sessions += t["num_charging_sessions"]
            total_waiting += t["waiting_time"]
            total_unloading += t["total_unloading_time"]
            
            # Count failures
            if t["failed"]:
                num_failed += 1
            
            # Count deliveries completed
            total_stops = len(t["delivery_sequence"])
            remaining = t["deliveries_remaining"]
            if total_stops > 0:
                num_deliveries += max(0, total_stops - 1 - remaining)
            
            # Calculate routing time: total_time - charging - unloading - waiting
            truck_total_time = t["total_time"]
            truck_charging_time = t["total_charging_time"]
            truck_unloading_time = t["total_unloading_time"]
            truck_waiting_time = t["waiting_time"]
            truck_routing_time = truck_total_time - truck_charging_time - truck_unloading_time - truck_waiting_time
            total_routing += max(0.0, truck_routing_time)
        
        distances.append(total_dist)
        charging_times.append(total_charge)
        total_deliveries.append(num_deliveries)
        num_charging_sessions.append(total_sessions)
        waiting_times.append(total_waiting)
        routing_times.append(total_routing)
        unloading_times.append(total_unloading)
        failures.append(num_failed)

    return {
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "episode_rewards": rewards,  # Store individual episode rewards for comparison
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
        "max_time_terminations": np.sum(max_time_terminations),
        "max_steps_terminations": np.sum(max_steps_terminations),
    }


def main():
    """Evaluate policies with hardcoded parameters."""

    config = load_config(CONFIG_FILE)

    # Detour GNN space requires sequential deliveries; disable flexible ordering if needed
    if any((len(entry) > 2 and entry[2] == "detour") for entry in POLICIES):
        if config["delivery"].get("enable_flexible_delivery_order", False):
            print(
                "Detected detour policies; forcing sequential delivery order (enable_flexible_delivery_order=False)."
            )
            config["delivery"]["enable_flexible_delivery_order"] = False

    # Detect SB3-trained configuration from policy paths (e.g., "1trucks_10stops")
    sb3_present = any(len(entry) >= 2 and entry[1].startswith("sb3-") for entry in POLICIES)
    sb3_configs = []
    for entry in POLICIES:
        if len(entry) < 2:
            continue
        if entry[1].startswith("sb3-"):
            detected = _extract_sb3_config(entry[0])
            if detected:
                sb3_configs.append(detected)
    unique_sb3_configs = set(sb3_configs)

    eval_num_trucks = NUM_TRUCKS
    eval_num_stops = NUM_STOPS
    if unique_sb3_configs:
        if len(unique_sb3_configs) > 1:
            raise ValueError(
                f"SB3 policies must share the same training config, found: {sorted(unique_sb3_configs)}"
            )
        eval_num_trucks, eval_num_stops = next(iter(unique_sb3_configs))
        print(
            f"Detected SB3 training config from path: {eval_num_trucks} trucks, {eval_num_stops} stops"
        )
    elif sb3_present:
        print(
            "SB3 policy detected but training config not encoded in path; using default constants (may mismatch)."
        )
    else:
        print(
            f"Using default config constants: {eval_num_trucks} trucks, {eval_num_stops} stops"
        )

    config["environment"]["num_trucks"] = eval_num_trucks
    config["environment"]["num_stops"] = eval_num_stops

    env_init = EventDrivenTruckEnv(config=config, verbose=False, enable_plotting=False)
    # Build the required GNN state spaces once, based on policy list
    requested_spaces = set(entry[2] if len(entry) > 2 else "base" for entry in POLICIES)
    gnn_state_spaces = {}
    for space in requested_spaces:
        mode = "vrp" if space == "vrp" else "nonflex"
        use_detour = space == "detour"
        gnn_state_spaces[space] = create_default_gnn_space(
            env_init,
            mode=mode,
            use_detour=use_detour,
            device="cpu",
        )
    env_init.close()

    # Load policies
    policies = {}
    policy_counter = {}  # Track duplicate names
    for policy_entry in POLICIES:
        policy_path, policy_type = policy_entry[0], policy_entry[1]
        gnn_space_type = policy_entry[2] if len(policy_entry) > 2 else "base"
        print(f"Loading: {policy_path} ({policy_type})...")
        
        # Handle SB3 policies differently
        if policy_type.startswith("sb3-"):
            # Load SB3 model
            algo_name = policy_type.replace("sb3-", "")
            if algo_name == "ppo":
                policy = PPO.load(policy_path, device="cpu")
            elif algo_name == "maskppo":
                policy = MaskablePPO.load(policy_path, device="cpu")
            elif algo_name == "dqn":
                policy = DQN.load(policy_path, device="cpu")
            elif algo_name == "qrdqn":
                policy = QRDQN.load(policy_path, device="cpu")
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
        elif policy_type == "optimal-simple":
            try:
                policy = OptimalGurobiSimplePolicy(verbose=False)
            except ImportError as exc:
                raise RuntimeError(
                    "Optimal Simple (Gurobi) policy requires gurobipy to be installed."
                ) from exc
            resolved_type = "optimal-simple"
        else:
            # Load GNN-based policies using existing function
            policy, resolved_type = load_policy(policy_path, policy_type, gnn_state_spaces[gnn_space_type], config)
        
        # Generate unique name for each policy
        if policy_path == "heuristic":
            name = "Heuristic"
        elif policy_path == "optimal":
            name = "Optimal (Gurobi)"
        elif policy_path == "optimal-simple":
            name = "MP Robust"
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
        
        policies[name] = {"policy": policy, "type": resolved_type, "gnn_space_type": gnn_space_type}

    # Evaluate all policies
    eval_env = EventDrivenTruckEnv(
        config=copy.deepcopy(config), verbose=False, enable_plotting=False
    )

    print(f"\n{'='*90}")
    print(f"Evaluating {len(policies)} policies over {NUM_EVAL_SCENARIOS} scenarios")
    print(f"Environment: {eval_num_trucks} trucks, {eval_num_stops} stops\n")

    results = {}
    for policy_name, policy_info in tqdm(policies.items(), desc="Policies", position=0):
        results[policy_name] = evaluate_policy(
            eval_env,
            policy_info["policy"],
            gnn_state_spaces[policy_info["gnn_space_type"]],
            policy_info["type"],
            NUM_EVAL_SCENARIOS,
            SEED,
            config,
        )

    eval_env.close()

    # Calculate reward gap vs optimal-simple baseline if it exists
    baseline_name = None
    for name in results.keys():
        if "MP Robust" in name or "optimal-simple" in name.lower():
            baseline_name = name
            break
    
    if baseline_name:
        baseline_rewards = np.array(results[baseline_name]["episode_rewards"])
        baseline_mean = results[baseline_name]["mean_reward"]
        print(f"\n{'='*90}")
        print(f"Episode-by-Episode Win Rate Analysis (vs {baseline_name})")
        print(f"{'='*90}")
        print(f"Baseline Mean Reward: {baseline_mean:.0f}\n")
        
        # Calculate win rate for each policy (% of episodes where policy beats baseline)
        for name in sorted(results.keys()):
            if name == baseline_name:
                continue
            policy_rewards = np.array(results[name]["episode_rewards"])
            policy_mean = results[name]["mean_reward"]
            
            # Count episodes where policy beats baseline
            wins = np.sum(policy_rewards > baseline_rewards)
            ties = np.sum(policy_rewards == baseline_rewards)
            losses = np.sum(policy_rewards < baseline_rewards)
            win_rate = (wins / len(policy_rewards)) * 100
            
            # Calculate mean reward difference
            mean_diff = policy_mean - baseline_mean
            diff_str = f"+{mean_diff:.0f}" if mean_diff >= 0 else f"{mean_diff:.0f}"
            
            print(f"  {name:40s}: Win Rate: {win_rate:5.1f}% ({wins}W/{ties}T/{losses}L)  Δ Reward: {diff_str}")
        print(f"{'='*90}\n")
        
        # Store win rate in results for later display
        for name in results.keys():
            if name == baseline_name:
                results[name]["win_rate_vs_baseline"] = 50.0  # Baseline against itself
            else:
                policy_rewards = np.array(results[name]["episode_rewards"])
                wins = np.sum(policy_rewards > baseline_rewards)
                results[name]["win_rate_vs_baseline"] = (wins / len(policy_rewards)) * 100

    # Print results in vertical format with policies side-by-side
    def wrap_name(name, width=20):
        """Wrap long policy names into multiple lines."""
        if len(name) <= width:
            return [name.ljust(width)]
        words = name.replace('_', ' ').replace('-', ' ').split()
        lines = []
        current_line = ""
        for word in words:
            if len(current_line) + len(word) + 1 <= width:
                current_line += (" " if current_line else "") + word
            else:
                if current_line:
                    lines.append(current_line.ljust(width))
                current_line = word
        if current_line:
            lines.append(current_line.ljust(width))
        return lines if lines else [name[:width].ljust(width)]
    
    sorted_names = sorted(results.keys())
    col_width = 22
    metric_col_width = 25
    # Calculate total width including vertical separators
    separator_width = metric_col_width + 1 + (col_width + 1) * len(sorted_names) + 1
    
    print(f"\n{'='*separator_width}")
    print(f"RESULTS (averaged over {NUM_EVAL_SCENARIOS} scenarios)")
    print(f"Environment: {eval_num_trucks} trucks, {eval_num_stops} stops")
    print(f"{'='*separator_width}\n")
    
    # Print policy names (wrapped if needed)
    name_lines = [wrap_name(name, col_width) for name in sorted_names]
    max_name_lines = max(len(lines) for lines in name_lines)
    
    print("Metric".ljust(metric_col_width), end="")
    print("|", end="")
    for i in range(max_name_lines):
        for j, lines in enumerate(name_lines):
            if i < len(lines):
                print(f" {lines[i]}", end="")
            else:
                print(f" {' '*col_width}", end="")
            if j < len(name_lines) - 1:
                print("|", end="")
        if i < max_name_lines - 1:
            print()
            print(" " * metric_col_width + "|", end="")
    print(" |")
    print("-" * separator_width)
    
    # Define metrics to display
    metrics = [
        ("Reward", "mean_reward", "std_reward", ".0f"),
    ]
    
    # Add win rate if baseline exists
    if baseline_name:
        metrics.append(("Win Rate vs Baseline (%)", "win_rate_vs_baseline", None, ".1f"))
    
    metrics.extend([
        ("Success Rate (%)", "success_rate", None, ".1f", 100),
        ("Deliveries", "mean_deliveries", "std_deliveries", ".1f"),
        ("Steps", "mean_steps", "std_steps", ".1f"),
        ("Total Time (h)", "mean_completion_time", "std_completion_time", ".1f"),
        ("Distance (km)", "mean_total_distance", "std_total_distance", ".0f"),
        ("Charging Time (h)", "mean_charging_time", "std_charging_time", ".1f"),
        ("Charging Sessions", "mean_charging_sessions", "std_charging_sessions", ".1f"),
        ("Waiting Time (h)", "mean_waiting_time", "std_waiting_time", ".1f"),
        ("Routing Time (h)", "mean_routing_time", "std_routing_time", ".1f"),
        ("Unloading Time (h)", "mean_unloading_time", "std_unloading_time", ".1f"),
        ("Failures", "mean_failures", "std_failures", ".1f"),
        ("Max Time Reached", "max_time_terminations", None, ".0f"),
        ("Max Steps Reached", "max_steps_terminations", None, ".0f"),
    ])
    
    for metric_info in metrics:
        label = metric_info[0]
        mean_key = metric_info[1]
        std_key = metric_info[2] if len(metric_info) > 2 else None
        fmt = metric_info[3] if len(metric_info) > 3 else ".1f"
        multiplier = metric_info[4] if len(metric_info) > 4 else 1
        
        print(f"{label:<{metric_col_width}}", end="")
        print("|", end="")
        for idx, name in enumerate(sorted_names):
            r = results[name]
            mean_val = r[mean_key] * multiplier
            if std_key and std_key in r:
                std_val = r[std_key] * multiplier
                value_str = f"{mean_val:{fmt}} ±{std_val:{fmt}}"
            else:
                value_str = f"{mean_val:{fmt}}"
            print(f" {value_str:>{col_width}}", end="")
            if idx < len(sorted_names) - 1:
                print(" |", end="")
        print(" |")
    
    print(f"{'='*separator_width}\n")


if __name__ == "__main__":
    main()
