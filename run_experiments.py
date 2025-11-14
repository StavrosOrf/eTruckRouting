"""
Experiment runner for EVPR - runs various hyperparameter configurations in separate tmux panes.
"""

import os
import time

# Configuration
config = "truck_env/config_files/config.yaml"
python_path = "/home/sorfanouda/EVPR/.venv/bin/python"

# Training parameters
counter = 0
max_episodes = 10_000_000
max_timesteps = 1_000_000
eval_freq = 500
batch_size = 64

# Environment settings
num_trucks = 2
num_stops = 3
max_time = 200.0

# PPO Hyperparameter Sweep (8 runs)
ppo_configs = [
    # Config 1: Baseline
    {'steps_per_update': 128, 'epochs': 10, 'entropy_coef': 0.01, 'seed': 0},
    # Config 2: More frequent updates
    {'steps_per_update': 64, 'epochs': 10, 'entropy_coef': 0.01, 'seed': 0},
    # Config 3: More training per update
    {'steps_per_update': 128, 'epochs': 15, 'entropy_coef': 0.01, 'seed': 0},
    # Config 4: Higher exploration
    {'steps_per_update': 32, 'epochs': 10, 'entropy_coef': 0.05, 'seed': 0},
    # Config 5: Much higher exploration
    {'steps_per_update': 32, 'epochs': 10, 'entropy_coef': 0.1, 'seed': 0},
    # Config 6: Combined: more updates + higher exploration
    {'steps_per_update': 32, 'epochs': 15, 'entropy_coef': 0.05, 'seed': 0},
    # Config 7: Alternative seed with best config
    {'steps_per_update': 128, 'epochs': 10, 'entropy_coef': 0.05, 'seed': 42},
    # Config 8: Aggressive exploration
    {'steps_per_update': 64, 'epochs': 15, 'entropy_coef': 0.1, 'seed': 42},
]

# Sweep name for grouping in wandb
sweep_name = "PPO_Hyperparameter_Sweep_num_trucks_" + str(num_trucks) + "_stops_" + str(num_stops)

# Fixed hyperparameters for this sweep
learning_rate = 3e-4
hidden_dim = 64
num_gcn_layers = 3

for config_idx, ppo_config in enumerate(ppo_configs, 1):
    seed = ppo_config['seed']
    steps_per_update = ppo_config['steps_per_update']
    epochs = ppo_config['epochs']
    entropy_coef = ppo_config['entropy_coef']
    
    # Base command
    command = 'tmux new-session -d \\; send-keys " ' + python_path + ' train.py' + \
        ' --algo ppo' + \
        ' --config ' + config + \
        ' --seed ' + str(seed) + \
        ' --lr ' + str(learning_rate) + \
        ' --gnn-hidden-dim ' + str(hidden_dim) + \
        ' --actor-gcn-layers ' + str(num_gcn_layers) + \
        ' --critic-gcn-layers ' + str(num_gcn_layers) + \
        ' --batch-size ' + str(batch_size) + \
        ' --max-episodes ' + str(max_episodes) + \
        ' --max-timesteps ' + str(max_timesteps) + \
        ' --eval-freq ' + str(eval_freq) + \
        ' --num-trucks ' + str(num_trucks) + \
        ' --num-stops ' + str(num_stops) + \
        ' --max-time ' + str(max_time) + \
        ' --wandb-project evpr-experiments' + \
        ' --wandb-entity stavrosorf' + \
        ' --group-name ' + sweep_name + \
        ' --ppo-steps-per-update ' + str(steps_per_update) + \
        ' --ppo-epochs ' + str(epochs) + \
        ' --ppo-minibatch-size 256' + \
        ' --ppo-clip 0.2' + \
        ' --ppo-entropy-coef ' + str(entropy_coef) + \
        ' --exp-name ' + \
        f'PPO_' + \
        f'steps={steps_per_update}_' + \
        f'epochs={epochs}_' + \
        f'ent={entropy_coef}_' + \
        f'seed={seed}'
    
    command += '" Enter'
    
    # Execute command
    os.system(command=command)
    print(f"[{config_idx}/8] Started: PPO | steps={steps_per_update} | epochs={epochs} | entropy={entropy_coef} | seed={seed}")
    
    # Wait before starting next experiment to avoid race conditions
    time.sleep(3)
    counter += 1

print(f"\n✓ Launched {counter} experiments in separate tmux sessions")
print("Use 'tmux ls' to see all sessions")
print("Use 'tmux attach -t <session-id>' to attach to a session")
print("Use 'pkill -f train.py' to kill all training sessions (if needed)")
