"""
Benchmark environment execution time with different numbers of trucks.

This script measures how long the environment runs with varying numbers of trucks
and creates visualizations showing the relationship between problem size and runtime.
"""

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.utils.utils import load_config


# ============================================================================
# Configuration
# ============================================================================

CONFIG_FILE = "EVRoutingEnv/config_files/config.yaml"

# Range of trucks to test
TRUCK_COUNTS = [3, 5, 8, 10, 15, 20, 25, 30]

# Number of episodes per configuration (for averaging)
NUM_EPISODES = 5

# Number of stops per truck (can also vary this)
NUM_STOPS = 5

# Maximum steps per episode (safety limit)
MAX_STEPS_PER_EPISODE = 500

# Output directory for plots
OUTPUT_DIR = "results/benchmark"

# Random seed for reproducibility
SEED = 42


# ============================================================================
# Benchmark Functions
# ============================================================================

def get_valid_action(env) -> int:
    """
    Get a valid action using action mask from the environment.
    
    Args:
        env: Environment instance
    
    Returns:
        A valid action index
    """
    from EVRoutingEnv.state.action_mask import get_action_mask
    
    action_mask = get_action_mask(env)
    valid_actions = np.where(action_mask)[0]
    
    if len(valid_actions) > 0:
        return np.random.choice(valid_actions)
    else:
        # Fallback: if no valid actions (shouldn't happen), return 0
        return 0


def benchmark_single_episode(
    num_trucks: int,
    num_stops: int,
    seed: int,
    max_steps: int = MAX_STEPS_PER_EPISODE
) -> Tuple[float, int, bool]:
    """
    Run a single episode and measure execution time.
    
    Args:
        num_trucks: Number of trucks in the environment
        num_stops: Number of delivery stops per truck
        seed: Random seed for reproducibility
        max_steps: Maximum number of steps to run
    
    Returns:
        Tuple of (execution_time, num_steps, episode_done)
    """
    # Load and modify config
    config = load_config(CONFIG_FILE)
    config['environment']['num_trucks'] = num_trucks
    config['environment']['num_stops'] = num_stops
    config['environment']['max_episode_steps'] = max_steps
    
    # Create environment (disable plotting and verbose for speed)
    env = EventDrivenTruckEnv(
        config=config,
        verbose=False,
        enable_plotting=False
    )
    
    # Set numpy random seed for action sampling
    np.random.seed(seed)
    
    # Reset environment
    start_time = time.time()
    obs, info = env.reset(seed=seed)
    
    # Run episode
    done = False
    truncated = False
    step_count = 0
    
    while not (done or truncated) and step_count < max_steps:
        # Get a valid action using the action mask
        action = get_valid_action(env)
        obs, reward, done, truncated, info = env.step(action)
        step_count += 1
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Clean up
    env.close()
    
    return execution_time, step_count, (done or truncated)


def benchmark_configuration(
    num_trucks: int,
    num_stops: int,
    num_episodes: int = NUM_EPISODES,
    base_seed: int = SEED
) -> Dict[str, float]:
    """
    Benchmark a configuration over multiple episodes.
    
    Args:
        num_trucks: Number of trucks
        num_stops: Number of delivery stops
        num_episodes: Number of episodes to run for averaging
        base_seed: Base random seed
    
    Returns:
        Dictionary with timing statistics
    """
    times = []
    steps = []
    completions = []
    
    for i in range(num_episodes):
        execution_time, step_count, completed = benchmark_single_episode(
            num_trucks=num_trucks,
            num_stops=num_stops,
            seed=base_seed + i
        )
        
        times.append(execution_time)
        steps.append(step_count)
        completions.append(completed)
    
    # Calculate per-step times
    time_per_step = [times[i] / steps[i] if steps[i] > 0 else 0 for i in range(len(times))]
    
    return {
        'num_trucks': num_trucks,
        'num_stops': num_stops,
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'mean_steps': np.mean(steps),
        'std_steps': np.std(steps),
        'mean_time_per_step': np.mean(time_per_step),
        'std_time_per_step': np.std(time_per_step),
        'completion_rate': np.mean(completions),
    }


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_results(results: List[Dict], output_dir: str):
    """
    Create visualizations of benchmark results.
    
    Args:
        results: List of benchmark result dictionaries
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data
    truck_counts = [r['num_trucks'] for r in results]
    mean_times = [r['mean_time'] for r in results]
    std_times = [r['std_time'] for r in results]
    mean_steps = [r['mean_steps'] for r in results]
    mean_time_per_step = [r['mean_time_per_step'] for r in results]
    std_time_per_step = [r['std_time_per_step'] for r in results]
    completion_rates = [r['completion_rate'] for r in results]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Environment Runtime Benchmark', fontsize=16, fontweight='bold')
    
    # Plot 1: Execution time vs number of trucks
    ax1 = axes[0, 0]
    ax1.errorbar(truck_counts, mean_times, yerr=std_times, 
                 marker='o', linewidth=2, capsize=5, capthick=2)
    ax1.set_xlabel('Number of Trucks', fontsize=12)
    ax1.set_ylabel('Execution Time (seconds)', fontsize=12)
    ax1.set_title('Average Episode Runtime vs Problem Size', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Steps per episode vs number of trucks
    ax2 = axes[0, 1]
    ax2.plot(truck_counts, mean_steps, marker='s', linewidth=2, color='green')
    ax2.set_xlabel('Number of Trucks', fontsize=12)
    ax2.set_ylabel('Steps per Episode', fontsize=12)
    ax2.set_title('Average Steps vs Problem Size', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Time per step
    ax3 = axes[1, 0]
    ax3.errorbar(truck_counts, mean_time_per_step, yerr=std_time_per_step,
                 marker='^', linewidth=2, color='orange', capsize=5, capthick=2)
    ax3.set_xlabel('Number of Trucks', fontsize=12)
    ax3.set_ylabel('Time per Step (seconds)', fontsize=12)
    ax3.set_title('Computational Cost per Decision Step', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Completion rate
    ax4 = axes[1, 1]
    ax4.bar(truck_counts, completion_rates, color='steelblue', alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Number of Trucks', fontsize=12)
    ax4.set_ylabel('Episode Completion Rate', fontsize=12)
    ax4.set_title('Episode Completion Rate vs Problem Size', fontsize=13, fontweight='bold')
    ax4.set_ylim([0, 1.1])
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save figure
    plot_path = os.path.join(output_dir, 'benchmark_results.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_path}")
    
    plt.close()
    
    # Create a second plot: log-scale for better visibility of scaling
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(truck_counts, mean_times, yerr=std_times, 
                marker='o', linewidth=2, capsize=5, capthick=2, label='Execution Time')
    ax.set_xlabel('Number of Trucks', fontsize=12)
    ax.set_ylabel('Execution Time (seconds)', fontsize=12)
    ax.set_title('Environment Runtime Scaling (Log Scale)', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend()
    
    plt.tight_layout()
    log_plot_path = os.path.join(output_dir, 'benchmark_results_log.png')
    plt.savefig(log_plot_path, dpi=300, bbox_inches='tight')
    print(f"Log-scale plot saved to: {log_plot_path}")
    
    plt.close()


def save_results_table(results: List[Dict], output_dir: str):
    """
    Save benchmark results as a formatted text table.
    
    Args:
        results: List of benchmark result dictionaries
        output_dir: Directory to save table
    """
    table_path = os.path.join(output_dir, 'benchmark_results.txt')
    
    with open(table_path, 'w') as f:
        f.write("=" * 120 + "\n")
        f.write("ENVIRONMENT RUNTIME BENCHMARK RESULTS\n")
        f.write("=" * 120 + "\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  - Episodes per configuration: {NUM_EPISODES}\n")
        f.write(f"  - Stops per truck: {NUM_STOPS}\n")
        f.write(f"  - Max steps per episode: {MAX_STEPS_PER_EPISODE}\n\n")
        f.write("-" * 120 + "\n")
        f.write(f"{'Trucks':<10} {'Mean Time (s)':<15} {'Mean Steps':<12} {'Time/Step (ms)':<16} {'Std Time/Step (ms)':<20} {'Completion Rate':<18}\n")
        f.write("-" * 120 + "\n")
        
        for r in results:
            f.write(f"{r['num_trucks']:<10} "
                   f"{r['mean_time']:<15.3f} "
                   f"{r['mean_steps']:<12.1f} "
                   f"{r['mean_time_per_step']*1000:<16.2f} "
                   f"{r['std_time_per_step']*1000:<20.2f} "
                   f"{r['completion_rate']:<18.2%}\n")
        
        f.write("-" * 100 + "\n")
    
    print(f"Results table saved to: {table_path}")


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution function."""
    print("=" * 80)
    print("ENVIRONMENT RUNTIME BENCHMARK")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  - Truck counts to test: {TRUCK_COUNTS}")
    print(f"  - Episodes per configuration: {NUM_EPISODES}")
    print(f"  - Stops per truck: {NUM_STOPS}")
    print(f"  - Max steps per episode: {MAX_STEPS_PER_EPISODE}")
    print(f"  - Output directory: {OUTPUT_DIR}")
    print("\nStarting benchmark...\n")
    
    results = []
    
    # Run benchmarks
    for num_trucks in tqdm(TRUCK_COUNTS, desc="Benchmarking configurations"):
        print(f"\n  Testing {num_trucks} trucks...")
        
        result = benchmark_configuration(
            num_trucks=num_trucks,
            num_stops=NUM_STOPS,
            num_episodes=NUM_EPISODES
        )
        
        results.append(result)
        
        print(f"    Mean time: {result['mean_time']:.3f} ± {result['std_time']:.3f} seconds")
        print(f"    Mean steps: {result['mean_steps']:.1f}")
        print(f"    Completion rate: {result['completion_rate']:.1%}")
    
    # Create visualizations
    print("\n" + "=" * 80)
    print("Creating visualizations...")
    plot_results(results, OUTPUT_DIR)
    
    # Save results table
    print("\nSaving results table...")
    save_results_table(results, OUTPUT_DIR)
    
    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
