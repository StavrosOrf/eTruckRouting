"""Evaluate and compare different policies on multiple scenarios (parallel)."""

import atexit
import copy
import os
import re
import sys
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
from tqdm import tqdm

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

from EVRoutingEnv.baselines.optimal_gurobi import OptimalGurobiPolicy
from EVRoutingEnv.baselines.optimal_gurobi_simple import OptimalGurobiSimplePolicy
from EVRoutingEnv.baselines.optimal_vrp_single_truck import OptimalVRPSingleTruckPolicy
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
    #Trained models on Electric Truck Routing     
    
    #1T3S
    ("saved_models/ppov_seq_1T3S_spu256_ep5_ent0.1_g32_m256_vk5_ck2_s0_8197/", "variable-ppo", "detour"),   
    ("saved_models/1trucks_3stops/maskppo_seed1_20260212_223935/best_model.zip", "sb3-maskppo", "base"),
    ("saved_models/1trucks_3stops/ppo_seed0_20260212_223719/best_model.zip", "sb3-ppo", "base"),   
         
    # #5T3S
    ("saved_models/ppov_seq_5T3S_spu256_ep5_ent0.1_g32_m256_vk5_ck3_s1_8197/", "variable-ppo", "detour"), 
    ("saved_models/5trucks_3stops/maskppo_seed1_20260212_223937/best_model.zip", "sb3-maskppo", "base"),
    ("saved_models/5trucks_3stops/ppo_seed1_20260212_223937/best_model.zip", "sb3-ppo", "base"),
    
    # # # #10T3S
    ("saved_models/ppov_seq_10T3S_spu256_ep5_ent0.1_g32_m256_vk5_ck5_s0_8197/", "variable-ppo", "detour"),    
    #("saved_models/ppov_seq_10T3S_spu256_ep5_ent0.1_g32_m256_vk5_ck2_s0_8197/", "variable-ppo", "detour"),  
    ### ("saved_models/ppov_seq_10T3S_spu256_ep5_ent0.1_g32_m256_vk5_ck2_hl2_s0_3485/", "variable-ppo", "detour"),  
    ("saved_models/10trucks_3stops/maskppo_seed0_20260212_223718/best_model.zip", "sb3-maskppo", "base"),
    ("saved_models/10trucks_3stops/ppo_seed1_20260212_223935/best_model.zip", "sb3-ppo", "base"),
    
    # # #30T3S
    ## ("saved_models/ppov_seq_30T3S_spu256_ep5_ent0.1_g32_m256_vk5_ck3_s0_2236/", "variable-ppo", "detour"),  
    ("saved_models/ppov_seq_30T3S_spu256_ep5_ent0.1_g32_m256_vk5_ck2_hl2_s0_3485/", "variable-ppo", "detour"),  
    ("saved_models/30trucks_3stops/maskppo_seed1_20260212_223936/best_model.zip", "sb3-maskppo", "base"),
    ("saved_models/30trucks_3stops/ppo_seed0_20260212_223719/best_model.zip", "sb3-ppo", "base"),
    
    # #50T3S
    ####("saved_models/ppov_seq_50T3S_spu256_ep5_ent0.1_g32_m256_vk5_ck3_s0_2236/", "variable-ppo", "detour"), 
    ###("saved_models/ppov_seq_50T3S_spu256_ep5_ent0.1_g32_m256_vk5_ck2_hl2_s0_3485/", "variable-ppo", "detour"), 
    ("saved_models/ppov_seq_50T3S_spu256_ep5_ent0.1_g32_m256_vk5_ck2_hl3_s0_3485/", "variable-ppo", "detour"), 
    ("saved_models/50trucks_3stops/maskppo_seed0_20260212_223719/best_model.zip", "sb3-maskppo", "base"), #more training needed !!!!
    ("saved_models/50trucks_3stops/ppo_seed1_20260212_223935/best_model.zip", "sb3-ppo", "base"),
    
    #100T3S
    ("saved_models/ppov_seq_100T3S_spu256_ep5_ent0.1_g32_m256_vk5_ck2_hl2_s0_3485/", "variable-ppo", "detour"), 
    ("saved_models/100trucks_3stops/maskppo_seed1_20260212_223936/best_model.zip", "sb3-maskppo", "base"),  #more training needed !!!!
    ("saved_models/100trucks_3stops/ppo_seed1_20260212_223936/best_model.zip", "sb3-ppo", "base"),
             
    
    ("optimal", "optimal", "base"),  # Gurobi-based optimal MILP solver, usually better opt-based solution
    ("heuristic", "heuristic", "base"),    
    ("optimal-simple", "optimal-simple", "base"),  # MP Robust - Gurobi solver with 20% energy safety margin
    
    
        
    # eVRP Single TRUCK
    # ("saved_models/Top5del_NewStateppov_1T10S_spu256_ep5_ent0.1_seed0_505/", "variable-ppo", "vrp"),        
    # ("saved_models/ppov_vrp_1T10S_lr0.0003_spu512_ep10_mb256_ent0.01_clip0.2_gm0.99_gl0.95_vc0.01_g32_m256_vk5_ck2_s0_6901/", "variable-ppo", "vrp"),        
    # ("saved_models/ppov_vrp_1T10S_lr0.0003_spu512_ep10_mb256_ent0.05_clip0.2_gm0.99_gl0.95_vc0.01_g32_m256_vk5_ck2_s0_6901/", "variable-ppo", "vrp"),        
    # ("saved_models/1trucks_10stops/maskppo_seed0_20260209_164605/best_model.zip", "sb3-maskppo", "base"),
        
    # ("savings", "savings", "base"),
    # ("nn-2opt", "nn-2opt", "base"),
    # ("optimal-vrp", "optimal-vrp", "vrp"),

]
CONFIG_FILE = "EVRoutingEnv/config_files/config.yaml"
# CONFIG_FILE = "EVRoutingEnv/config_files/config_vrp.yaml"
NUM_TRUCKS = 1  # Must match the configuration used during training
NUM_STOPS = 3
NUM_EVAL_SCENARIOS = 200
SEED = 1000
AUTO_DETECT_SB3_CONFIG = False  # Set False to force NUM_TRUCKS/NUM_STOPS

NUM_WORKERS = 24 
GPU_DEVICES = (0, 1, 2)
# Use GPU only for these policy types
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


def parse_policy_size(policy_path):
    """Return (num_trucks, num_stops) if encoded in path like '1T3S' or '1trucks_3stops'."""
    if not isinstance(policy_path, str):
        return None
    for part in policy_path.rstrip("/").split("/"):
        match = re.search(r"(\d+)T(\d+)S", part)
        if match:
            return int(match.group(1)), int(match.group(2))
        match = re.search(r"(\d+)trucks_(\d+)stops", part)
        if match:
            return int(match.group(1)), int(match.group(2))
    return None


def evaluate_policy(env, policy, gnn_state_space, policy_type, num_episodes, seed, config):
    """Evaluate a policy over multiple episodes (sequential reference)."""
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
    total_truck_times = []
    failures = []
    avg_completion_soc = []
    exec_times = []
    max_time_terminations = []
    max_steps_terminations = []
    vrp_feasible_flags = []

    env._default_gnn_state_space = gnn_state_space
    env.use_detour_mask = getattr(gnn_state_space, "use_detour", False)

    is_sb3_policy = policy_type.startswith("sb3-")

    for episode in tqdm(range(num_episodes), desc="Evaluating", leave=False):
        start_time = time.perf_counter()
        obs, info = env.reset(seed=seed + episode)
        episode_reward, episode_steps = 0.0, 0
        done = truncated = False
        if policy_type == "optimal":
            episode_policy = OptimalGurobiPolicy(verbose=False)
        elif policy_type == "optimal-simple":
            episode_policy = OptimalGurobiSimplePolicy(verbose=False)
        elif policy_type in ("optimal-vrp", "optimal_vrp"):
            episode_policy = OptimalVRPSingleTruckPolicy(verbose=False)
        else:
            episode_policy = policy

        if policy_type in ("optimal-vrp", "optimal_vrp"):
            episode_policy.episode_infeasible = False

        while not (done or truncated):
            if is_sb3_policy:
                if policy_type == "sb3-maskppo":
                    action_masks = get_action_mask(env)
                    action, _states = policy.predict(obs, action_masks=action_masks, deterministic=True)
                else:
                    action, _states = policy.predict(obs, deterministic=True)
            else:
                if policy_type in ("optimal", "optimal-simple", "optimal-vrp", "optimal_vrp"):
                    action = episode_policy.get_action(env)
                elif policy_type in ("heuristic", "savings", "nn-2opt"):
                    action = policy.get_action(env)
                elif policy_type in ("ppo-variable", "variable-ppo"):
                    gnn_state = gnn_state_space.get_state_GNN(env)
                    mask = torch.tensor(get_action_mask(env), dtype=torch.bool)
                    raw_action = policy.select_action(
                        gnn_state, deterministic=True, action_mask=mask
                    )
                    action = policy.to_env_action(gnn_state, int(raw_action))
                else:
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

        exec_times.append(time.perf_counter() - start_time)

        rewards.append(episode_reward)
        successes.append(1.0 if info["all_complete"] else 0.0)
        steps.append(episode_steps)
        completion_times.append(env.global_clock)

        max_time_reached = 1.0 if truncated and env.global_clock >= env.max_time else 0.0
        max_steps_reached = 1.0 if truncated and episode_steps >= env.max_episode_steps else 0.0
        max_time_terminations.append(max_time_reached)
        max_steps_terminations.append(max_steps_reached)

        trucks_info = info["trucks"]
        total_dist = 0.0
        total_charge = 0.0
        total_sessions = 0
        total_waiting = 0.0
        total_routing = 0.0
        total_unloading = 0.0
        total_truck_time = 0.0
        num_failed = 0
        num_deliveries = 0

        for t in trucks_info:
            total_dist += t["total_distance"]
            total_charge += t["total_charging_time"]
            total_sessions += t["num_charging_sessions"]
            total_waiting += t["waiting_time"]
            total_unloading += t["total_unloading_time"]
            total_truck_time += t["total_time"]

            if t["failed"]:
                num_failed += 1

            total_stops = len(t["delivery_sequence"])
            remaining = t["deliveries_remaining"]
            if total_stops > 0:
                num_deliveries += max(0, total_stops - 1 - remaining)

            truck_total_time = t["total_time"]
            truck_charging_time = t["total_charging_time"]
            truck_unloading_time = t["total_unloading_time"]
            truck_waiting_time = t["waiting_time"]
            truck_routing_time = truck_total_time - truck_charging_time - truck_unloading_time - truck_waiting_time
            total_routing += max(0.0, truck_routing_time)

        if trucks_info:
            avg_soc = float(np.mean([t.get("battery_percentage", 0.0) for t in trucks_info]))
        else:
            avg_soc = 0.0
        avg_completion_soc.append(avg_soc)

        distances.append(total_dist)
        charging_times.append(total_charge)
        total_deliveries.append(num_deliveries)
        num_charging_sessions.append(total_sessions)
        waiting_times.append(total_waiting)
        routing_times.append(total_routing)
        unloading_times.append(total_unloading)
        total_truck_times.append(total_truck_time)
        failures.append(num_failed)
        if policy_type in ("optimal-vrp", "optimal_vrp"):
            vrp_feasible_flags.append(not episode_policy.episode_infeasible)

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
        "mean_total_truck_time": np.mean(total_truck_times),
        "std_total_truck_time": np.std(total_truck_times),
        "mean_failures": np.mean(failures),
        "std_failures": np.std(failures),
        "mean_completion_soc": np.mean(avg_completion_soc),
        "std_completion_soc": np.std(avg_completion_soc),
        "mean_exec_time": np.mean(exec_times),
        "std_exec_time": np.std(exec_times),
        "max_time_terminations": np.sum(max_time_terminations),
        "max_steps_terminations": np.sum(max_steps_terminations),
        "vrp_feasible_flags": vrp_feasible_flags,
    }


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


def _get_gnn_state_space(space: str, policy_path: str, env_init: EventDrivenTruckEnv):
    mode = "vrp" if space == "vrp" else "nonflex"
    use_detour = space == "detour"
    gnn_cfg = _load_saved_gnn_state_config(policy_path)
    vrp_top_k = int(gnn_cfg.get("vrp_top_k_deliveries", 5))
    detour_top_k = int(gnn_cfg.get("detour_top_k_chargers", 2))
    detour_hop_limit = int(gnn_cfg.get("detour_hop_limit", 2))
    cache_key = (
        space,
        vrp_top_k,
        detour_top_k,
        detour_hop_limit,
        env_init.num_trucks,
        env_init.num_stops,
    )
    if cache_key not in _WORKER_CACHE["gnn_state_space"]:
        _WORKER_CACHE["gnn_state_space"][cache_key] = create_default_gnn_space(
            env_init,
            mode=mode,
            use_detour=use_detour,
            device="cpu",
            vrp_top_k_deliveries=vrp_top_k,
            detour_num_chargers_to_keep=detour_top_k,
            detour_hop_limit=detour_hop_limit,
        )
    return _WORKER_CACHE["gnn_state_space"][cache_key]


def _should_use_gpu(policy_type: str) -> bool:
    if policy_type.startswith("sb3-"):
        return True
    return policy_type in GPU_POLICY_TYPES


def _load_policy_cached(policy_path, policy_type, gnn_state_space, config, device):
    if policy_type in ("optimal", "optimal-simple", "optimal-vrp", "optimal_vrp"):
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
        elif resolved_type == "optimal-simple":
            episode_policy = OptimalGurobiSimplePolicy(verbose=False)
        elif resolved_type in ("optimal-vrp", "optimal_vrp"):
            episode_policy = OptimalVRPSingleTruckPolicy(verbose=False)
        else:
            episode_policy = policy

        if resolved_type in ("optimal-vrp", "optimal_vrp"):
            episode_policy.episode_infeasible = False

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
                if resolved_type in ("optimal", "optimal-simple", "optimal-vrp", "optimal_vrp"):
                    action = episode_policy.get_action(env)
                elif resolved_type in ("heuristic", "savings", "nn-2opt"):
                    action = episode_policy.get_action(env)
                elif resolved_type in ("ppo-variable", "variable-ppo"):
                    gnn_state = gnn_state_space.get_state_GNN(env)
                    mask = torch.tensor(get_action_mask(env), dtype=torch.bool)
                    raw_action = episode_policy.select_action(
                        gnn_state, deterministic=True, action_mask=mask
                    )
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
        total_truck_time = 0.0
        num_failed = 0
        num_deliveries = 0

        for t in trucks_info:
            total_dist += t["total_distance"]
            total_charge += t["total_charging_time"]
            total_sessions += t["num_charging_sessions"]
            total_waiting += t["waiting_time"]
            total_unloading += t["total_unloading_time"]
            total_truck_time += t["total_time"]

            if t["failed"]:
                num_failed += 1

            total_stops = len(t["delivery_sequence"])
            remaining = t["deliveries_remaining"]
            if total_stops > 0:
                num_deliveries += max(0, total_stops - 1 - remaining)

            truck_total_time = t["total_time"]
            truck_charging_time = t["total_charging_time"]
            truck_unloading_time = t["total_unloading_time"]
            truck_waiting_time = t["waiting_time"]
            truck_routing_time = truck_total_time - truck_charging_time - truck_unloading_time - truck_waiting_time
            total_routing += max(0.0, truck_routing_time)

        if trucks_info:
            avg_soc = float(np.mean([t.get("battery_percentage", 0.0) for t in trucks_info]))
        else:
            avg_soc = 0.0

        vrp_feasible = True
        if resolved_type in ("optimal-vrp", "optimal_vrp"):
            vrp_feasible = not episode_policy.episode_infeasible

        return {
            "policy_name": policy_name,
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
            "total_truck_time": total_truck_time,
            "failures": num_failed,
            "avg_completion_soc": avg_soc,
            "exec_time": exec_time,
            "max_time_termination": max_time_reached,
            "max_steps_termination": max_steps_reached,
            "vrp_feasible": vrp_feasible,
        }
    finally:
        env.close()


def _aggregate_episode_results(episode_results):
    if not episode_results:
        return {
            "mean_reward": float("nan"),
            "std_reward": float("nan"),
            "episode_rewards": [],
            "success_rate": float("nan"),
            "mean_total_distance": float("nan"),
            "std_total_distance": float("nan"),
            "mean_charging_time": float("nan"),
            "std_charging_time": float("nan"),
            "mean_steps": float("nan"),
            "std_steps": float("nan"),
            "mean_completion_time": float("nan"),
            "std_completion_time": float("nan"),
            "mean_deliveries": float("nan"),
            "std_deliveries": float("nan"),
            "mean_charging_sessions": float("nan"),
            "std_charging_sessions": float("nan"),
            "mean_waiting_time": float("nan"),
            "std_waiting_time": float("nan"),
            "mean_routing_time": float("nan"),
            "std_routing_time": float("nan"),
            "mean_unloading_time": float("nan"),
            "std_unloading_time": float("nan"),
            "mean_total_truck_time": float("nan"),
            "std_total_truck_time": float("nan"),
            "mean_failures": float("nan"),
            "std_failures": float("nan"),
            "mean_completion_soc": float("nan"),
            "std_completion_soc": float("nan"),
            "mean_exec_time": float("nan"),
            "std_exec_time": float("nan"),
            "max_time_terminations": 0,
            "max_steps_terminations": 0,
        }

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
    total_truck_times = []
    failures = []
    avg_completion_soc = []
    exec_times = []
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
        total_truck_times.append(result["total_truck_time"])
        failures.append(result["failures"])
        avg_completion_soc.append(result["avg_completion_soc"])
        exec_times.append(result["exec_time"])
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
        "mean_total_truck_time": np.mean(total_truck_times),
        "std_total_truck_time": np.std(total_truck_times),
        "mean_failures": np.mean(failures),
        "std_failures": np.std(failures),
        "mean_completion_soc": np.mean(avg_completion_soc),
        "std_completion_soc": np.std(avg_completion_soc),
        "mean_exec_time": np.mean(exec_times),
        "std_exec_time": np.std(exec_times),
        "max_time_terminations": np.sum(max_time_terminations),
        "max_steps_terminations": np.sum(max_steps_terminations),
    }


def _filter_episode_results(episode_results, excluded_indices):
    if not excluded_indices:
        return episode_results
    return [
        result
        for result in episode_results
        if result.get("episode_idx") not in excluded_indices
    ]


def build_policies(policy_entries):
    policy_counter = {}
    policies = {}
    policy_full_names = {}
    for policy_entry in policy_entries:
        policy_path, policy_type = policy_entry[0], policy_entry[1]
        gnn_space_type = policy_entry[2] if len(policy_entry) > 2 else "base"
        print(f"Loading: {policy_path} ({policy_type})...")

        if policy_path == "heuristic":
            name = "Heuristic"
            full_name = "Heuristic"
        elif policy_path == "optimal":
            name = "Optimal (Gurobi)"
            full_name = "Optimal (Gurobi)"
        elif policy_path == "optimal-simple":
            name = "MP Robust"
            full_name = "MP Robust"
        elif policy_path in ("optimal-vrp", "optimal_vrp"):
            name = "Optimal VRP"
            full_name = "Optimal VRP"
        elif policy_type.startswith("sb3-"):
            dir_path = os.path.dirname(policy_path) if policy_path.endswith(".zip") else policy_path
            full_base_name = os.path.basename(dir_path.rstrip("/"))
            truncated_base_name = full_base_name[:40]
            name = f"SB3-{truncated_base_name}"
            full_name = f"SB3-{full_base_name}"
        else:
            full_base_name = os.path.basename(policy_path.rstrip("/"))
            base_name = full_base_name[:30]
            if base_name in policy_counter:
                policy_counter[base_name] += 1
                name = f"{base_name}_v{policy_counter[base_name]}"
            else:
                policy_counter[base_name] = 1
                name = base_name
            full_name = full_base_name

        policies[name] = {
            "path": policy_path,
            "type": policy_type,
            "gnn_space": gnn_space_type,
        }
        policy_full_names[name] = full_name

    return policies, policy_full_names


def run_parallel_eval(
    policy_entries,
    config,
    num_trucks,
    num_stops,
    num_eval_scenarios,
    seed,
    num_workers,
    gpu_devices,
    auto_detect_sb3_config=False,
    print_summary=True,
):
    atexit.register(_cleanup_children)

    local_config = copy.deepcopy(config)

    if any((len(entry) > 2 and entry[2] == "detour") for entry in policy_entries):
        if local_config["delivery"].get("enable_flexible_delivery_order", False):
            print(
                "Detected detour policies; forcing sequential delivery order (enable_flexible_delivery_order=False)."
            )
            local_config["delivery"]["enable_flexible_delivery_order"] = False

    sb3_present = any(len(entry) >= 2 and entry[1].startswith("sb3-") for entry in policy_entries)
    sb3_configs = []
    for entry in policy_entries:
        if len(entry) < 2:
            continue
        if entry[1].startswith("sb3-"):
            detected = _extract_sb3_config(entry[0])
            if detected:
                sb3_configs.append(detected)
    unique_sb3_configs = set(sb3_configs)

    eval_num_trucks = num_trucks
    eval_num_stops = num_stops
    if auto_detect_sb3_config and unique_sb3_configs:
        if len(unique_sb3_configs) > 1:
            raise ValueError(
                f"SB3 policies must share the same training config, found: {sorted(unique_sb3_configs)}"
            )
        eval_num_trucks, eval_num_stops = next(iter(unique_sb3_configs))
        print(
            f"Detected SB3 training config from path: {eval_num_trucks} trucks, {eval_num_stops} stops"
        )
    elif sb3_present and auto_detect_sb3_config:
        print(
            "SB3 policy detected but training config not encoded in path; using default constants (may mismatch)."
        )
    else:
        print(f"Using default config constants: {eval_num_trucks} trucks, {eval_num_stops} stops")

    local_config["environment"]["num_trucks"] = eval_num_trucks
    local_config["environment"]["num_stops"] = eval_num_stops

    policies, policy_full_names = build_policies(policy_entries)

    if policy_full_names:
        print("\nLegend (full policy names):")
        for short_name in sorted(policy_full_names.keys()):
            print(f"  {short_name} -> {policy_full_names[short_name]}")

    print(f"\n{'='*90}")
    print(f"Evaluating {len(policies)} policies over {num_eval_scenarios} scenarios")
    print(f"Environment: {eval_num_trucks} trucks, {eval_num_stops} stops\n")

    tasks = []
    gpu_counter = 0
    for policy_name, policy_info in policies.items():
        for episode_idx in range(num_eval_scenarios):
            if _should_use_gpu(policy_info["type"]):
                gpu_id = gpu_devices[gpu_counter % len(gpu_devices)]
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
                    episode_idx,
                    seed,
                    local_config,
                    device,
                )
            )

    episode_results_by_policy = {
        name: [None for _ in range(num_eval_scenarios)] for name in policies.keys()
    }

    mp_context = mp.get_context("spawn")
    executor = ProcessPoolExecutor(max_workers=num_workers, mp_context=mp_context)
    try:
        futures = [executor.submit(_run_episode_task, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating", leave=False):
            result = future.result()
            policy_name = result["policy_name"]
            episode_idx = result["episode_idx"]
            episode_results_by_policy[policy_name][episode_idx] = result
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        _cleanup_children()

    excluded_indices = set()
    vrp_policy_names = [
        name
        for name, info in policies.items()
        if info["type"] in ("optimal-vrp", "optimal_vrp")
    ]
    if vrp_policy_names:
        vrp_name = vrp_policy_names[0]
        for result in episode_results_by_policy.get(vrp_name, []):
            if result is None:
                continue
            if not result.get("vrp_feasible", True):
                excluded_indices.add(result.get("episode_idx"))
        if excluded_indices:
            sorted_indices = sorted(excluded_indices)
            print(f"Excluded seed indices: {sorted_indices}")
            excluded_path = os.path.join(project_root, "results", "vrp_excluded_seeds.txt")
            os.makedirs(os.path.dirname(excluded_path), exist_ok=True)
            with open(excluded_path, "w", encoding="utf-8") as handle:
                handle.write("# Excluded VRP infeasible seeds\n")
                handle.write(f"seed_base={seed}\n")
                handle.write("episode_idx,seed\n")
                for idx in sorted_indices:
                    handle.write(f"{idx},{seed + idx}\n")
            print(
                f"\nExcluded {len(excluded_indices)} seeds where VRP Optimal was infeasible from comparisons."
            )

    results = {}
    for policy_name, episode_results in episode_results_by_policy.items():
        if any(result is None for result in episode_results):
            missing = [i for i, r in enumerate(episode_results) if r is None]
            raise RuntimeError(f"Missing episode results for {policy_name}: {missing}")
        filtered_results = _filter_episode_results(episode_results, excluded_indices)
        results[policy_name] = _aggregate_episode_results(filtered_results)

    if print_summary:
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

            for name in sorted(results.keys()):
                if name == baseline_name:
                    continue
                policy_rewards = np.array(results[name]["episode_rewards"])
                policy_mean = results[name]["mean_reward"]

                wins = np.sum(policy_rewards > baseline_rewards)
                ties = np.sum(policy_rewards == baseline_rewards)
                losses = np.sum(policy_rewards < baseline_rewards)
                win_rate = (wins / len(policy_rewards)) * 100

                mean_diff = policy_mean - baseline_mean
                diff_str = f"+{mean_diff:.0f}" if mean_diff >= 0 else f"{mean_diff:.0f}"

                print(
                    f"  {name:40s}: Win Rate: {win_rate:5.1f}% ({wins}W/{ties}T/{losses}L)  Δ Reward: {diff_str}"
                )
            print(f"{'='*90}\n")

            for name in results.keys():
                if name == baseline_name:
                    results[name]["win_rate_vs_baseline"] = 50.0
                else:
                    policy_rewards = np.array(results[name]["episode_rewards"])
                    wins = np.sum(policy_rewards > baseline_rewards)
                    results[name]["win_rate_vs_baseline"] = (wins / len(policy_rewards)) * 100

        if len(results) > 1:
            print(f"\n{'='*90}")
            print("Pairwise Episode Win Counts (A > B)")
            print(f"{'='*90}")

            sorted_names = sorted(results.keys())
            rewards_by_name = {
                name: np.array(results[name]["episode_rewards"]) for name in sorted_names
            }
            episode_counts = [len(rewards_by_name[name]) for name in sorted_names]
            min_episodes = min(episode_counts)
            if len(set(episode_counts)) > 1:
                print(
                    f"Warning: mismatched episode counts {episode_counts}; using first {min_episodes} episodes."
                )

            col_width = 16
            print("A \\ B".ljust(col_width), end="")
            for name in sorted_names:
                print(f" {name[:col_width-1]:<{col_width-1}}", end="")
            print()
            print("-" * (col_width + col_width * len(sorted_names)))

            for name_a in sorted_names:
                print(f"{name_a[:col_width-1]:<{col_width}}", end="")
                rewards_a = rewards_by_name[name_a][:min_episodes]
                for name_b in sorted_names:
                    rewards_b = rewards_by_name[name_b][:min_episodes]
                    wins = int(np.sum(rewards_a > rewards_b))
                    ties = int(np.sum(rewards_a == rewards_b))
                    losses = int(np.sum(rewards_a < rewards_b))
                    cell = f"{wins}W/{ties}T/{losses}L"
                    print(f" {cell:<{col_width-1}}", end="")
                print()
            print(f"{'='*90}\n")

            print(f"\n{'='*90}")
            print("Pairwise Avg Reward Diff (when A wins / A loses)")
            print(f"{'='*90}")

            def _avg_or_none(values):
                return float(np.mean(values)) if len(values) > 0 else None

            print("A \\ B".ljust(col_width), end="")
            for name in sorted_names:
                print(f" {name[:col_width-1]:<{col_width-1}}", end="")
            print()
            print("-" * (col_width + col_width * len(sorted_names)))

            for name_a in sorted_names:
                print(f"{name_a[:col_width-1]:<{col_width}}", end="")
                rewards_a = rewards_by_name[name_a][:min_episodes]
                for name_b in sorted_names:
                    rewards_b = rewards_by_name[name_b][:min_episodes]
                    diffs = rewards_a - rewards_b
                    win_diffs = diffs[diffs > 0]
                    lose_diffs = diffs[diffs < 0]
                    avg_win = _avg_or_none(win_diffs)
                    avg_lose = _avg_or_none(lose_diffs)
                    win_str = f"+{avg_win:.0f}" if avg_win is not None else "--"
                    lose_str = f"{avg_lose:.0f}" if avg_lose is not None else "--"
                    cell = f"{win_str}/{lose_str}"
                    print(f" {cell:<{col_width-1}}", end="")
                print()
            print(f"{'='*90}\n")

        def wrap_name(name, width=20):
            """Wrap long policy names into multiple lines."""
            if len(name) <= width:
                return [name.ljust(width)]
            words = name.replace("_", " ").replace("-", " ").split()
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
        metric_col_width = 25 + 1
        separator_width = metric_col_width + 1 + (col_width + 1) * len(sorted_names) + 1

        print(f"\n{'='*separator_width}")
        effective_scenarios = num_eval_scenarios - len(excluded_indices)
        print(f"RESULTS (averaged over {effective_scenarios} scenarios)")
        print(f"Environment: {eval_num_trucks} trucks, {eval_num_stops} stops")
        print(f"{'='*separator_width}\n")

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
                    print(" |", end="")
            if i < max_name_lines - 1:
                print()
                print(" " * metric_col_width + "|", end="")
        print(" |")
        print("-" * separator_width)

        metrics = [
            ("Reward", "mean_reward", "std_reward", ".0f"),
        ]

        if baseline_name:
            metrics.append(("Win Rate vs Baseline (%)", "win_rate_vs_baseline", None, ".1f"))

        metrics.extend(
            [
                ("Success Rate (%)", "success_rate", None, ".1f", 100),
                ("Avg SoC at End (%)", "mean_completion_soc", "std_completion_soc", ".1f"),
                ("Exec Time (s)", "mean_exec_time", "std_exec_time", ".2f"),
                ("Deliveries", "mean_deliveries", "std_deliveries", ".1f"),
                ("Steps", "mean_steps", "std_steps", ".1f"),
                ("Total Time (h)", "mean_completion_time", "std_completion_time", ".1f"),
                ("Distance (km)", "mean_total_distance", "std_total_distance", ".0f"),
                ("Charging Time (h)", "mean_charging_time", "std_charging_time", ".1f"),
                ("Charging Sessions", "mean_charging_sessions", "std_charging_sessions", ".1f"),
                ("Waiting Time (h)", "mean_waiting_time", "std_waiting_time", ".1f"),
                ("Routing Time (h)", "mean_routing_time", "std_routing_time", ".1f"),
                ("Unloading Time (h)", "mean_unloading_time", "std_unloading_time", ".1f"),
                ("Total Truck Time (h)", "mean_total_truck_time", "std_total_truck_time", ".1f"),
                ("Failures", "mean_failures", "std_failures", ".1f"),
                ("Max Time Reached", "max_time_terminations", None, ".0f"),
                ("Max Steps Reached", "max_steps_terminations", None, ".0f"),
            ]
        )

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

    return {
        "episode_results_by_policy": episode_results_by_policy,
        "policy_full_names": policy_full_names,
        "policies": policies,
        "excluded_indices": excluded_indices,
        "results": results,
        "num_trucks": eval_num_trucks,
        "num_stops": eval_num_stops,
    }


def main():
    """Evaluate policies with hardcoded parameters in parallel."""
    config = load_config(CONFIG_FILE)
    run_parallel_eval(
        POLICIES,
        config,
        NUM_TRUCKS,
        NUM_STOPS,
        NUM_EVAL_SCENARIOS,
        SEED,
        NUM_WORKERS,
        GPU_DEVICES,
        auto_detect_sb3_config=AUTO_DETECT_SB3_CONFIG,
        print_summary=True,
    )


if __name__ == "__main__":
    main()
