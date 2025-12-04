"""
Experiment runner for SB3 algorithms - runs various algorithm and seed configurations in separate tmux panes.
"""

import os
import time
import itertools

# Configuration
config = "truck_env/config_files/config.yaml"
python_path = "/home/sorfanouda/EVPR/.venv/bin/python"

# Training parameters
counter = 0
total_steps = 10_000_000
eval_freq = 1_000
n_eval_episodes = 30

# Define hyperparameter grids
hyperparam_grids = {
    "algorithm": ["maskppo", "qrdqn"],
    "seed": [0],
}

# Generate all combinations
all_combinations = list(
    itertools.product(
        hyperparam_grids["algorithm"],
        hyperparam_grids["seed"],
    )
)

# Get environment info from config for group naming
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from truck_env.utils.utils import load_config

env_config = load_config(config)
num_trucks = env_config["environment"]["num_trucks"]
num_stops = env_config["environment"]["num_stops"]

# Sweep name for grouping in wandb
sweep_name = f"SB3_Sweep_{num_trucks}trucks_{num_stops}stops"

print(f"Total configurations to run: {len(all_combinations)}")
print(f"Sweep name: {sweep_name}\n")

for config_idx, (algorithm, seed) in enumerate(all_combinations, 1):
    # Print configuration being launched
    print(
        f"[{config_idx}/{len(all_combinations)}] Launching: {algorithm.upper()} | seed={seed}"
    )

    # Build experiment name
    exp_name = f"sb3_{algorithm}_seed{seed}_trucks{num_trucks}_stops{num_stops}"

    # Build command
    command = (
        f'tmux new-session -d -s "{exp_name}" \\; send-keys "'
        f"{python_path} train_sb3_event_driven.py"
        f" --algo {algorithm}"
        f" --seed {seed}"
        f" --config {config}"
        f" --steps {total_steps}"
        f" --eval-freq {eval_freq}"
        f" --n-eval-episodes {n_eval_episodes}"
        f" --project evpr-sb3"
        f" --save-dir ./saved_models"
        f'" Enter'
    )

    # Execute command
    os.system(command)
    print(f"  Session: {exp_name}")
    counter += 1

    # Wait before starting next experiment to avoid race conditions
    time.sleep(2)

print(f"\n✓ Launched {counter} experiments in separate tmux sessions")
print(f"  Total: {len(all_combinations)} configurations")
print(f"\nHyperparameter grid:")
for param, values in hyperparam_grids.items():
    print(f"  {param}: {values}")
print(f"\nEnvironment settings:")
print(f"  Trucks: {num_trucks}")
print(f"  Stops: {num_stops}")
print(f"  Total steps: {total_steps:,}")
print(f"  Eval frequency: {eval_freq:,}")
print("\nUseful commands:")
print("  tmux ls                        - List all sessions")
print("  tmux attach -t <session-id>    - Attach to a session")
print("  tmux kill-session -t <session> - Kill a specific session")
print("  pkill -f train_sb3             - Kill all SB3 training sessions")
