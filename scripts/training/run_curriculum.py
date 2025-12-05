#!/usr/bin/env python3
"""Simple tmux runner for curriculum learning with seed and strategy grids."""

import os
import subprocess
from pathlib import Path

# ============================================================================
# CONFIGURATION - Edit these grids
# ============================================================================
SEED_GRID = [0]
CURRICULUM_GRID = ['uniform', 'staged', 'mixed']
NUM_GPUS = 1
SESSION_NAME = "curriculum"

# ============================================================================
# Script logic
# ============================================================================

def get_project_root():
    return str(Path(__file__).parent.parent.parent)

def create_command(seed, curriculum, gpu_id, project_root):
    exp_name = f"curriculum_{curriculum}_seed{seed}"
    
    cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} python scripts/training/train_curriculum.py"
    
    if curriculum in ['staged', 'mixed']:
        config_path = f"EVRoutingEnv/config_files/curriculum_config_{curriculum}.json"
        cmd += f" --curriculum-config {config_path}"
    else:
        cmd += f" --curriculum-strategy {curriculum} --truck-range 3 8 --stop-range 3 8"
    
    cmd += f" --exp-name {exp_name} --seed {seed}"
    return exp_name, cmd

def main():
    project_root = get_project_root()
    
    # Generate experiments
    experiments = []
    exp_id = 0
    for seed in SEED_GRID:
        for curriculum in CURRICULUM_GRID:
            gpu_id = exp_id % NUM_GPUS
            exp_name, cmd = create_command(seed, curriculum, gpu_id, project_root)
            experiments.append((exp_id, gpu_id, seed, curriculum, exp_name, cmd))
            exp_id += 1
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"CURRICULUM TMUX RUNNER: {len(experiments)} experiments")
    print(f"{'='*70}")
    for exp_id, gpu_id, seed, curriculum, exp_name, _ in experiments:
        print(f"[{exp_id}] GPU{gpu_id}: {curriculum:8s} seed={seed}")
    print(f"{'='*70}\n")
    
    # Kill existing session if exists
    subprocess.run(["tmux", "kill-session", "-t", SESSION_NAME], 
                   capture_output=True)
    
    # Create session with first window
    subprocess.run(["tmux", "new-session", "-s", SESSION_NAME, 
                   "-n", "exp0", "-d"], check=True)
    
    venv = f"{project_root}/.venv"
    activate = f"source {venv}/bin/activate" if os.path.exists(venv) else ""
    
    # Setup windows - each experiment in its own window
    for i, (exp_id, gpu_id, seed, curriculum, exp_name, cmd) in enumerate(experiments):
        if i > 0:
            # Create new window for each experiment after the first
            subprocess.run(["tmux", "new-window", "-t", SESSION_NAME, 
                          "-n", f"exp{i}"], check=True)
        
        subprocess.run(["tmux", "send-keys", "-t", f"{SESSION_NAME}:exp{i}",
                       f"cd {project_root}", "C-m"])
        if activate:
            subprocess.run(["tmux", "send-keys", "-t", f"{SESSION_NAME}:exp{i}",
                           activate, "C-m"])
        subprocess.run(["tmux", "send-keys", "-t", f"{SESSION_NAME}:exp{i}",
                       cmd, "C-m"])
    
    # Monitor window
    subprocess.run(["tmux", "new-window", "-t", SESSION_NAME, "-n", "gpu"])
    subprocess.run(["tmux", "send-keys", "-t", f"{SESSION_NAME}:gpu",
                   "watch -n 3 nvidia-smi", "C-m"])
    
    subprocess.run(["tmux", "select-window", "-t", f"{SESSION_NAME}:exp0"])
    
    print(f"Session '{SESSION_NAME}' created!")
    print(f"Attach: tmux attach -t {SESSION_NAME}")
    print(f"Switch windows: Ctrl-b [0-{len(experiments)}]")
    print(f"Detach: Ctrl-b d")
    print(f"Kill: tmux kill-session -t {SESSION_NAME}\n")

if __name__ == '__main__':
    main()
