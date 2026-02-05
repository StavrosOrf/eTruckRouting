"""
Minimal experiment runner that can launch PPO-GNN (parallel only) and SB3 jobs in tmux.
Toggle what to run via run_algorithms. Uses common exp_name and group_name helpers.
"""

import itertools
import os
import time
import yaml

python_path = "/home/sorfanouda/EVPR/.venv/bin/python"
config_path = "EVRoutingEnv/config_files/config_vrp.yaml"
# config_path = "EVRoutingEnv/config_files/config.yaml"

# Select which stacks to run: choose any of ["ppo-variable", "sb3"]

run_algorithms = ["ppo-variable"]

# PPO-V (parallel) settings
ppo_params = {
    "learning_rate": 3e-4,
    "num_gcn_layers": 3,
    "minibatch_size": 256,
    "max_episodes": 100_000_000,
    "max_timesteps": 10_000_000,
    "eval_freq": 5000,
    "num_parallel_envs": 1,
    "num_eval_envs": 12,
    "project": "evpr-newtests",
}

ppo_grid = {
    "steps_per_update": [256],
    "epochs": [5],
    "entropy_coef": [0.1],
    "gnn_hidden_dim": [32],
    "mlp_hidden_dim": [256],
    "seed": [0],
}

# SB3 settings
sb3_grid = {
    "algorithm": ["maskppo","dqn", "qrdqn"],
    "seed": [0],
}
sb3_params = {
    "total_steps": 10_000_000,
    "eval_freq": 1000,
    "n_eval_episodes": 50,
    "project": "evpr-newtests",
    "save_dir": "./saved_models",
    "device": "cuda",
}


def load_env_meta(cfg_path: str):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    delivery_cfg = cfg["delivery"]
    traffic_cfg = cfg["traffic"]
    charging_cfg = cfg["charging"]
    env_cfg = cfg["environment"]

    flexible = delivery_cfg["enable_flexible_delivery_order"]

    gnn_state = "vrp" if flexible else "detour"
    flags = []
    if traffic_cfg["enable_traffic"]:
        flags.append("T")
        if traffic_cfg["enable_energy_uncertainty"]:
            flags.append("E")
    if delivery_cfg["enable_stochastic_unloading"]:
        flags.append("U")
    if charging_cfg["use_realistic_curve"]:
        flags.append("C")
    if flexible:
        flags.append("F")

    suffix = "".join(flags) if flags else "Det"
    max_time = env_cfg["max_time"]

    return {
        "gnn_state": gnn_state,
        "uncertainty_suffix": suffix,
        "num_trucks": env_cfg["num_trucks"],
        "num_stops": env_cfg["num_stops"],
        "max_time": max_time,
        "problem_type": "vrp" if flexible else gnn_state,
    }


def build_names(base_algo: str, meta: dict, extra: str) -> tuple[str, str]:
    group = f"{meta['problem_type']}_L_{meta['num_trucks']}T{meta['num_stops']}S_{meta['uncertainty_suffix']}"
    exp = f"{base_algo}_{meta['num_trucks']}T{meta['num_stops']}S_{extra}_{int(time.time()) % 10000}"
    return exp, group


def tmux_send(session: str, command: str):
    full = f'tmux new-session -d -s "{session}" \\; send-keys "{command}" Enter'
    os.system(full)
    print(full)


meta = load_env_meta(config_path)
counter = 0

if "ppo-variable" in run_algorithms:
    ppo_training_script = "scripts/training/train_PPO_Variable_parallel.py"
    ppo_jobs = list(itertools.product(*ppo_grid.values()))
    print(f"PPO-V jobs: {len(ppo_jobs)}")
    for job in ppo_jobs:
        steps_per_update, epochs, entropy_coef, gnn_hidden_dim, mlp_hidden_dim, seed = job
        exp_name, group_name = build_names("ppov", meta, f"spu{steps_per_update}_ep{epochs}_ent{entropy_coef}_seed{seed}")
        cmd = (
            f"{python_path} {ppo_training_script}"
            f" --config {config_path}"
            f" --seed {seed}"
            f" --lr {ppo_params['learning_rate']}"
            f" --gnn-state-space {meta['gnn_state']}"
            f" --gnn-hidden-dim {gnn_hidden_dim}"
            f" --mlp-hidden-dim {mlp_hidden_dim}"
            f" --actor-gcn-layers {ppo_params['num_gcn_layers']}"
            f" --critic-gcn-layers {ppo_params['num_gcn_layers']}"
            f" --max-episodes {ppo_params['max_episodes']}"
            f" --max-timesteps {ppo_params['max_timesteps']}"
            f" --eval-freq {ppo_params['eval_freq']}"
            f" --num-trucks {meta['num_trucks']}"
            f" --num-stops {meta['num_stops']}"
            f" --max-time {meta['max_time']}"
            f" --wandb-project {ppo_params['project']}"
            f" --wandb-entity stavrosorf"
            f" --group-name {group_name}"
            f" --ppo-steps-per-update {steps_per_update}"
            f" --ppo-epochs {epochs}"
            f" --ppo-minibatch-size {ppo_params['minibatch_size']}"
            f" --ppo-clip 0.2"
            f" --ppo-entropy-coef {entropy_coef}"
            f" --exp-name {exp_name}"
            f" --num-parallel-envs {ppo_params['num_parallel_envs']}"
            f" --num-eval-envs {ppo_params['num_eval_envs']}"
        )
        tmux_send(exp_name, cmd)
        counter += 1

if "sb3" in run_algorithms:
    sb3_jobs = list(itertools.product(*sb3_grid.values()))
    print(f"SB3 jobs: {len(sb3_jobs)}")
    for job in sb3_jobs:
        algorithm, seed = job
        exp_name, _ = build_names(f"sb3-{algorithm}", meta, f"seed{seed}")
        cmd = (
            f"{python_path} scripts/training/train_sb3_event_driven.py"
            f" --algo {algorithm}"
            f" --seed {seed}"
            f" --config {config_path}"
            f" --steps {sb3_params['total_steps']}"
            f" --eval-freq {sb3_params['eval_freq']}"
            f" --n-eval-episodes {sb3_params['n_eval_episodes']}"
            f" --project {sb3_params['project']}"
            f" --save-dir {sb3_params['save_dir']}"
            f" --device {sb3_params['device']}"
        )
        tmux_send(exp_name, cmd)
        counter += 1

print(f"\nLaunched {counter} tmux sessions")
