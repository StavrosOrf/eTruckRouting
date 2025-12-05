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
SESSION_PREFIX = "curr"

# ============================================================================
# Script logic
# ============================================================================

def get_project_root():
    return str(Path(__file__).parent.parent.parent)

def create_command(seed, curriculum, gpu_id, project_root):
    exp_name = f"curriculum_{curriculum}_seed{seed}"
    config_path = f"EVRoutingEnv/config_files/curriculum_config_{curriculum}.json"
    
    cmd = (f"CUDA_VISIBLE_DEVICES={gpu_id} python scripts/training/train_curriculum.py "
           f"--curriculum-config {config_path} "
           f"--exp-name {exp_name} "
           f"--seed {seed}")
    
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
    
    venv = f"{project_root}/.venv"
    activate = f"source {venv}/bin/activate" if os.path.exists(venv) else ""
    
    # Create a separate session for each experiment
    session_names = []
    for exp_id, gpu_id, seed, curriculum, exp_name, cmd in experiments:
        session_name = f"{SESSION_PREFIX}_{curriculum}_s{seed}"
        session_names.append(session_name)
        
        # Kill existing session if exists
        subprocess.run(["tmux", "kill-session", "-t", session_name], 
                       capture_output=True)
        
        # Create new session
        subprocess.run(["tmux", "new-session", "-s", session_name, "-d"])
        
        # Setup environment
        subprocess.run(["tmux", "send-keys", "-t", session_name,
                       f"cd {project_root}", "C-m"])
        if activate:
            subprocess.run(["tmux", "send-keys", "-t", session_name,
                           activate, "C-m"])
        
        # Run training command
        subprocess.run(["tmux", "send-keys", "-t", session_name,
                       cmd, "C-m"])
    
    print(f"Created {len(experiments)} tmux sessions:")
    for session_name in session_names:
        print(f"  - {session_name}")
    print(f"\nAttach to any session: tmux attach -t <session_name>")
    print(f"List sessions: tmux ls")
    print(f"Kill all: tmux kill-server\n")

if __name__ == '__main__':
    main()
