#!/usr/bin/env python3
"""
Script to initialize and launch a hyperparameter sweep on SLURM.
This script will:
1. Initialize the sweep with wandb
2. Submit SLURM jobs to run sweep agents
3. Provide monitoring commands
"""

import os
import re
import subprocess
import sys
import argparse
from datetime import datetime
from pathlib import Path


class Colors:
    """ANSI color codes for terminal output."""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color


def print_colored(message, color=Colors.NC):
    """Print colored message to terminal."""
    print(f"{color}{message}{Colors.NC}")


def run_command(cmd, capture_output=True, check=True):
    """Run shell command and return output."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=capture_output,
            text=True,
            check=check
        )
        return result.stdout if capture_output else None
    except subprocess.CalledProcessError as e:
        print_colored(f"ERROR: Command failed: {cmd}", Colors.RED)
        print_colored(f"Error: {e.stderr}", Colors.RED)
        sys.exit(1)


def check_wandb_login():
    """Check if user is logged in to wandb."""
    try:
        result = subprocess.run(
            ["wandb", "status"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if "Logged in" not in result.stdout:
            print_colored("Please log in to Weights & Biases:", Colors.YELLOW)
            subprocess.run(["wandb", "login"], check=True)
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
        print_colored("Please log in to Weights & Biases:", Colors.YELLOW)
        subprocess.run(["wandb", "login"], check=True)


def initialize_sweep(config_file):
    """Initialize wandb sweep and return sweep ID."""
    print_colored(f"\nInitializing sweep with config: {config_file}", Colors.GREEN)
    
    output = run_command(f"wandb sweep {config_file}")
    print(output)
    
    # Extract sweep ID from output
    match = re.search(r'wandb agent .*/([a-z0-9]+)', output)
    if not match:
        print_colored("ERROR: Failed to extract sweep ID from wandb output", Colors.RED)
        print(f"Output was:\n{output}")
        sys.exit(1)
    
    sweep_id = match.group(1)
    return sweep_id


def submit_slurm_jobs(sweep_id, num_agents, use_gpu, entity, project):
    """Submit SLURM array job for sweep agents."""
    script_name = "run_sweep.sh" if use_gpu else "run_sweep_cpu.sh"
    
    print_colored(f"\nSubmitting {'GPU' if use_gpu else 'CPU'} jobs...", Colors.GREEN)
    
    # Create sbatch command
    cmd = f"sbatch --array=0-{num_agents-1} {script_name} {entity}/{project}/{sweep_id}"
    output = run_command(cmd)
    
    # Extract job ID
    match = re.search(r'Submitted batch job (\d+)', output)
    if not match:
        print_colored("ERROR: Failed to extract job ID from sbatch output", Colors.RED)
        print(f"Output was:\n{output}")
        sys.exit(1)
    
    job_id = match.group(1)
    return job_id


def save_sweep_info(sweep_id, job_id, num_agents, config_file, use_gpu, entity, project):
    """Save sweep information to a file."""
    info_file = f"logs/sweep_{sweep_id}_info.txt"
    
    user = os.environ.get('USER', 'unknown')
    script_name = "run_sweep.sh" if use_gpu else "run_sweep_cpu.sh"
    
    content = f"""Sweep Information
=================
Sweep ID: {sweep_id}
Job ID: {job_id}
Number of Agents: {num_agents}
Configuration: {config_file}
Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Using GPU: {use_gpu}
Entity: {entity}
Project: {project}

Commands:
---------
# Monitor sweep progress on W&B:
https://wandb.ai/{entity}/{project}/sweeps/{sweep_id}

# Check job status:
squeue -u {user} -j {job_id}

# Check job details:
scontrol show job {job_id}

# Cancel sweep jobs:
scancel {job_id}

# View logs:
tail -f logs/sweep_{job_id}_*.out

# Add more agents:
sbatch --array=0-4 {script_name} {entity}/{project}/{sweep_id}
"""
    
    with open(info_file, 'w') as f:
        f.write(content)
    
    return info_file


def main():
    parser = argparse.ArgumentParser(
        description='Launch hyperparameter sweep on SLURM cluster'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='sweep_config.yaml',
        help='Path to sweep configuration file'
    )
    parser.add_argument(
        '--num-agents',
        type=int,
        default=10,
        help='Number of sweep agents to run in parallel'
    )
    parser.add_argument(
        '--cpu',
        action='store_true',
        help='Use CPU-only jobs (default: GPU)'
    )
    parser.add_argument(
        '--entity',
        type=str,
        default='stavrosorf',
        help='W&B entity (username or team)'
    )
    parser.add_argument(
        '--project',
        type=str,
        default='evpr-td3-gnn-sweep',
        help='W&B project name'
    )
    
    args = parser.parse_args()
    
    print_colored("="*50, Colors.GREEN)
    print_colored("EVPR Hyperparameter Sweep Launcher", Colors.GREEN)
    print_colored("="*50, Colors.GREEN)
    
    # Check if config file exists
    if not os.path.exists(args.config):
        print_colored(f"ERROR: Sweep configuration file not found: {args.config}", Colors.RED)
        sys.exit(1)
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    # Check wandb login
    check_wandb_login()
    
    # Initialize sweep
    sweep_id = initialize_sweep(args.config)
    print_colored(f"\nSweep initialized with ID: {sweep_id}", Colors.GREEN)
    
    # Submit SLURM jobs
    job_id = submit_slurm_jobs(
        sweep_id,
        args.num_agents,
        not args.cpu,
        args.entity,
        args.project
    )
    
    print_colored(f"Submitted SLURM array job: {job_id}", Colors.GREEN)
    print_colored(f"Number of agents: {args.num_agents}", Colors.GREEN)
    
    # Save sweep info
    info_file = save_sweep_info(
        sweep_id,
        job_id,
        args.num_agents,
        args.config,
        not args.cpu,
        args.entity,
        args.project
    )
    
    # Print summary
    print_colored("\n" + "="*50, Colors.GREEN)
    print_colored("Sweep launched successfully!", Colors.GREEN)
    print_colored("="*50, Colors.GREEN)
    
    print_colored("\nSweep Dashboard:", Colors.YELLOW)
    print(f"https://wandb.ai/{args.entity}/{args.project}/sweeps/{sweep_id}")
    
    user = os.environ.get('USER', 'unknown')
    script_name = "run_sweep.sh" if not args.cpu else "run_sweep_cpu.sh"
    
    print_colored("\nUseful Commands:", Colors.YELLOW)
    print(f"# Monitor job status:")
    print(f"  squeue -u {user} -j {job_id}")
    print(f"\n# View logs (replace X with task ID):")
    print(f"  tail -f logs/sweep_{job_id}_X.out")
    print(f"\n# Cancel all sweep jobs:")
    print(f"  scancel {job_id}")
    print(f"\n# Add more agents (5 more):")
    print(f"  sbatch --array=0-4 {script_name} {args.entity}/{args.project}/{sweep_id}")
    print_colored(f"\nSweep info saved to: {info_file}", Colors.GREEN)


if __name__ == "__main__":
    main()
