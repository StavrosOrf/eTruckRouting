"""
Experiment runner for EVPR - runs various hyperparameter configurations in separate tmux panes.
"""

import os
import time
import itertools

# Configuration
config = "truck_env/config_files/config.yaml"
python_path = "/home/sorfanouda/EVPR/.venv/bin/python"

# Training parameters
counter = 0
max_episodes = 100_000_000
max_timesteps = 10_000_000
eval_freq = 500
batch_size = 64

# Environment settings
num_trucks = 10
num_stops = 3
max_time = 200.0

# Fixed hyperparameters for this sweep
learning_rate = 3e-4
hidden_dim = 64
num_gcn_layers = 3

# Define hyperparameter grids
hyperparam_grids = {
    # 'algorithm': ['ppo', 'ppo-variable'],
    "algorithm": ["ppo-variable"],
    "steps_per_update": [64, 512],
    "epochs": [5],
    "entropy_coef": [0.1],
    "gnn_hidden_dim": [32],
    "mlp_hidden_dim": [256],
    "seed": [0],
}

# Generate all combinations
all_combinations = list(
    itertools.product(
        hyperparam_grids["algorithm"],
        hyperparam_grids["steps_per_update"],
        hyperparam_grids["epochs"],
        hyperparam_grids["entropy_coef"],
        hyperparam_grids["gnn_hidden_dim"],
        hyperparam_grids["mlp_hidden_dim"],
        hyperparam_grids["seed"],
    )
)

# Sweep name for grouping in wandb


print(f"Total configurations to run: {len(all_combinations)}")
# print(f"Sweep name: {sweep_name}\n")

for config_idx, (
    algorithm,
    steps_per_update,
    epochs,
    entropy_coef,
    gnn_hidden_dim,
    mlp_hidden_dim,
    seed,
) in enumerate(all_combinations, 1):
    # Print configuration being launched
    print(
        f"[{config_idx}/{len(all_combinations)}] Launching: {algorithm} | steps={steps_per_update} | epochs={epochs} | entropy={entropy_coef} | seed={seed}"
    )

    # Build experiment name
    exp_name = f"NewFeasibleSpace_FixedGraph_{algorithm}_steps={steps_per_update}_epochs={epochs}_ent={entropy_coef}_seed={seed}"
    exp_name += f"_gnnhd={gnn_hidden_dim}_mlphd={mlp_hidden_dim}"
    sweep_name = f"Sweep_Trucks_{num_trucks}_stops_{num_stops}"

    # Build command
    command = (
        f'tmux new-session -d \\; send-keys " {python_path} train.py'
        f" --algo {algorithm}"
        f" --config {config}"
        f" --seed {seed}"
        f" --lr {learning_rate}"
        f" --gnn-hidden-dim {gnn_hidden_dim}"
        f" --mlp-hidden-dim {mlp_hidden_dim}"
        f" --actor-gcn-layers {num_gcn_layers}"
        f" --critic-gcn-layers {num_gcn_layers}"
        f" --batch-size {batch_size}"
        f" --max-episodes {max_episodes}"
        f" --max-timesteps {max_timesteps}"
        f" --eval-freq {eval_freq}"
        f" --num-trucks {num_trucks}"
        f" --num-stops {num_stops}"
        f" --max-time {max_time}"
        f" --wandb-project evpr-experiments"
        f" --wandb-entity stavrosorf"
        f" --group-name {sweep_name}"
        f" --ppo-steps-per-update {steps_per_update}"
        f" --ppo-epochs {epochs}"
        f" --ppo-minibatch-size 256"
        f" --ppo-clip 0.2"
        f" --ppo-entropy-coef {entropy_coef}"
        f" --exp-name {exp_name}"
        f'" Enter'
    )

    # Execute command
    os.system(command)
    print(command)
    counter += 1

    # Wait before starting next experiment to avoid race conditions
    time.sleep(3)

print(f"\n✓ Launched {counter} experiments in separate tmux sessions")
print(f"  Total: {len(all_combinations)} configurations")
print(f"\nHyperparameter grid:")
for param, values in hyperparam_grids.items():
    print(f"  {param}: {values}")
print("\nUseful commands:")
print("  tmux ls                        - List all sessions")
print("  tmux attach -t <session-id>    - Attach to a session")
print("  pkill -f train.py              - Kill all training sessions")
