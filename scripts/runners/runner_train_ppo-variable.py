"""
Experiment runner for EVPR - runs various hyperparameter configurations in separate tmux panes.
"""

import os
import time
import itertools
import yaml

# Configuration
config = "EVRoutingEnv/config_files/config.yaml"
python_path = "/home/sorfanouda/EVPR/.venv/bin/python"

# Training parameterss
counter = 0
max_episodes = 100_000_000
max_timesteps = 10_000_000
eval_freq = 5000
batch_size = 64

# Environment settings
num_trucks = 15
num_stops = 5
max_time = 200.0

# Fixed hyperparameters for this sweep
learning_rate = 3e-4
hidden_dim = 64
num_gcn_layers = 3

# Define hyperparameter grids
hyperparam_grids = {
    "steps_per_update": [1024],
    # "steps_per_update": [128, 512, 1024],
    "epochs": [5],
    "entropy_coef": [0.1],
    "gnn_hidden_dim": [32],
    "mlp_hidden_dim": [256],
    "seed": [0],
}

# Generate all combinations
all_combinations = list(
    itertools.product(
        hyperparam_grids["steps_per_update"],
        hyperparam_grids["epochs"],
        hyperparam_grids["entropy_coef"],
        hyperparam_grids["gnn_hidden_dim"],
        hyperparam_grids["mlp_hidden_dim"],
        hyperparam_grids["seed"],
    )
)

# Load config file to extract uncertainty settings
with open(config, 'r') as f:
    config_data = yaml.safe_load(f)

# Build uncertainty suffix for group name (short format)
uncertainty_flags = []
if config_data.get('traffic', {}).get('enable_traffic', False):
    uncertainty_flags.append('T')  # Traffic
    if config_data.get('traffic', {}).get('enable_energy_uncertainty', False):
        uncertainty_flags.append('E')  # Energy uncertainty
if config_data.get('delivery', {}).get('enable_stochastic_unloading', False):
    uncertainty_flags.append('U')  # Unloading
if config_data.get('charging', {}).get('use_realistic_curve', False):
    uncertainty_flags.append('C')  # Charging curve (CCCV)
if config_data.get('delivery', {}).get('enable_flexible_delivery_order', False):
    uncertainty_flags.append('F')  # Flexible delivery

uncertainty_suffix = ''.join(uncertainty_flags) if uncertainty_flags else 'Det'  # Deterministic if none

print(f"Total configurations to run: {len(all_combinations)}")
print(f"Uncertainty flags: {uncertainty_suffix}\n")

for config_idx, (
    steps_per_update,
    epochs,
    entropy_coef,
    gnn_hidden_dim,
    mlp_hidden_dim,
    seed,
) in enumerate(all_combinations, 1):
    # Print configuration being launched
    print(
        f"[{config_idx}/{len(all_combinations)}] Launching: ppo-variable | steps={steps_per_update} | epochs={epochs} | entropy={entropy_coef} | seed={seed}"
    )

    # Build experiment name
    exp_name = f"PenalizeSoC_Base_steps={steps_per_update}_epochs={epochs}_ent={entropy_coef}_seed={seed}"
    exp_name += f"_gnnhd={gnn_hidden_dim}_mlphd={mlp_hidden_dim}"
    
    # Group name includes environment size and uncertainty types
    group_name = f"{num_trucks}T{num_stops}S_{uncertainty_suffix}"

    # Build command
    command = (
        f'tmux new-session -d \\; send-keys " {python_path} scripts/training/train_PPO_Variable.py'
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
        f" --group-name {group_name}"
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
