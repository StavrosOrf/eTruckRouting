"""
Extended benchmark: Test environment runtime with different trucks AND stops configurations.

This script runs a more comprehensive benchmark that varies both the number of trucks
and the number of stops to understand how each dimension affects runtime.
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
from EVRoutingEnv.state.action_mask import get_action_mask


# ============================================================================
# Configuration
# ============================================================================

CONFIG_FILE = "EVRoutingEnv/config_files/config.yaml"

# Range of configurations to test
TRUCK_COUNTS = [3, 5, 8, 10, 15, 20]
STOP_COUNTS = [3, 5, 7, 10]

# Number of episodes per configuration
NUM_EPISODES = 3

# Maximum steps per episode (safety limit)
MAX_STEPS_PER_EPISODE = 500

# Output directory for plots
OUTPUT_DIR = "results/benchmark_extended"

# Random seed for reproducibility
SEED = 42


# ============================================================================
# Helper Functions
# ============================================================================

def get_valid_action(env) -> int:
    """Get a valid action using action mask."""
    action_mask = get_action_mask(env)
    valid_actions = np.where(action_mask)[0]
    return np.random.choice(valid_actions) if len(valid_actions) > 0 else 0


def run_episode(env, seed: int, max_steps: int) -> Tuple[float, int, bool]:
    """Run a single episode and measure performance."""
    np.random.seed(seed)
    
    start_time = time.time()
    obs, info = env.reset(seed=seed)
    
    done = False
    truncated = False
    step_count = 0
    
    while not (done or truncated) and step_count < max_steps:
        action = get_valid_action(env)
        obs, reward, done, truncated, info = env.step(action)
        step_count += 1
    
    execution_time = time.time() - start_time
    return execution_time, step_count, (done or truncated)


def benchmark_configuration(num_trucks: int, num_stops: int, num_episodes: int = NUM_EPISODES) -> Dict:
    """Benchmark a specific configuration."""
    config = load_config(CONFIG_FILE)
    config['environment']['num_trucks'] = num_trucks
    config['environment']['num_stops'] = num_stops
    config['environment']['max_episode_steps'] = MAX_STEPS_PER_EPISODE
    
    env = EventDrivenTruckEnv(config=config, verbose=False, enable_plotting=False)
    
    times = []
    steps = []
    completions = []
    
    for i in range(num_episodes):
        exec_time, step_count, completed = run_episode(env, SEED + i, MAX_STEPS_PER_EPISODE)
        times.append(exec_time)
        steps.append(step_count)
        completions.append(completed)
    
    env.close()
    
    # Calculate per-step times
    time_per_step = [times[i] / steps[i] if steps[i] > 0 else 0 for i in range(len(times))]
    
    return {
        'num_trucks': num_trucks,
        'num_stops': num_stops,
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'mean_steps': np.mean(steps),
        'mean_time_per_step': np.mean(time_per_step),
        'std_time_per_step': np.std(time_per_step),
        'completion_rate': np.mean(completions),
    }


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_heatmap(results: List[Dict], output_dir: str):
    """Create heatmap showing runtime for different truck/stop combinations."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Organize data into a matrix
    trucks = sorted(set(r['num_trucks'] for r in results))
    stops = sorted(set(r['num_stops'] for r in results))
    
    time_matrix = np.zeros((len(stops), len(trucks)))
    time_per_step_matrix = np.zeros((len(stops), len(trucks)))
    completion_matrix = np.zeros((len(stops), len(trucks)))
    
    for r in results:
        i = stops.index(r['num_stops'])
        j = trucks.index(r['num_trucks'])
        time_matrix[i, j] = r['mean_time']
        time_per_step_matrix[i, j] = r['mean_time_per_step'] * 1000  # Convert to ms
        completion_matrix[i, j] = r['completion_rate']
    
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))
    
    # Heatmap 1: Runtime
    im1 = ax1.imshow(time_matrix, cmap='YlOrRd', aspect='auto')
    ax1.set_xticks(range(len(trucks)))
    ax1.set_yticks(range(len(stops)))
    ax1.set_xticklabels(trucks)
    ax1.set_yticklabels(stops)
    ax1.set_xlabel('Number of Trucks', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Stops', fontsize=12, fontweight='bold')
    ax1.set_title('Average Episode Runtime (seconds)', fontsize=14, fontweight='bold')
    
    # Add text annotations
    for i in range(len(stops)):
        for j in range(len(trucks)):
            text = ax1.text(j, i, f'{time_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=9)
    
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Time (seconds)', fontsize=11)
    
    # Heatmap 2: Completion Rate
    im2 = ax2.imshow(completion_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax2.set_xticks(range(len(trucks)))
    ax2.set_yticks(range(len(stops)))
    ax2.set_xticklabels(trucks)
    ax2.set_yticklabels(stops)
    ax2.set_xlabel('Number of Trucks', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Stops', fontsize=12, fontweight='bold')
    ax2.set_title('Episode Completion Rate', fontsize=14, fontweight='bold')
    
    # Add text annotations
    for i in range(len(stops)):
        for j in range(len(trucks)):
            text = ax2.text(j, i, f'{completion_matrix[i, j]:.0%}',
                          ha="center", va="center", color="black", fontsize=9)
    
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Completion Rate', fontsize=11)
    
    # Heatmap 3: Time per Step
    im3 = ax3.imshow(time_per_step_matrix, cmap='viridis', aspect='auto')
    ax3.set_xticks(range(len(trucks)))
    ax3.set_yticks(range(len(stops)))
    ax3.set_xticklabels(trucks)
    ax3.set_yticklabels(stops)
    ax3.set_xlabel('Number of Trucks', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Number of Stops', fontsize=12, fontweight='bold')
    ax3.set_title('Time per Step (milliseconds)', fontsize=14, fontweight='bold')
    
    # Add text annotations
    for i in range(len(stops)):
        for j in range(len(trucks)):
            text = ax3.text(j, i, f'{time_per_step_matrix[i, j]:.1f}',
                          ha="center", va="center", color="white", fontsize=9)
    
    cbar3 = plt.colorbar(im3, ax=ax3)
    cbar3.set_label('Time per Step (ms)', fontsize=11)
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'heatmap_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to: {plot_path}")
    plt.close()


def plot_scaling_analysis(results: List[Dict], output_dir: str):
    """Create plots showing how runtime scales with trucks and stops."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Get unique stops values
    stops_values = sorted(set(r['num_stops'] for r in results))
    colors = plt.cm.viridis(np.linspace(0, 1, len(stops_values)))
    
    # Plot 1: Runtime vs Trucks (grouped by stops)
    ax1 = axes[0]
    for stop_count, color in zip(stops_values, colors):
        subset = [r for r in results if r['num_stops'] == stop_count]
        subset.sort(key=lambda x: x['num_trucks'])
        
        trucks = [r['num_trucks'] for r in subset]
        times = [r['mean_time'] for r in subset]
        
        ax1.plot(trucks, times, marker='o', linewidth=2, label=f'{stop_count} stops', color=color)
    
    ax1.set_xlabel('Number of Trucks', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
    ax1.set_title('Runtime Scaling with Trucks', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Runtime vs Stops (grouped by trucks)
    ax2 = axes[1]
    truck_values = sorted(set(r['num_trucks'] for r in results))
    colors2 = plt.cm.plasma(np.linspace(0, 1, len(truck_values)))
    
    for truck_count, color in zip(truck_values, colors2):
        subset = [r for r in results if r['num_trucks'] == truck_count]
        subset.sort(key=lambda x: x['num_stops'])
        
        stops = [r['num_stops'] for r in subset]
        times = [r['mean_time'] for r in subset]
        
        ax2.plot(stops, times, marker='s', linewidth=2, label=f'{truck_count} trucks', color=color)
    
    ax2.set_xlabel('Number of Stops', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
    ax2.set_title('Runtime Scaling with Stops', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'scaling_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Scaling analysis saved to: {plot_path}")
    plt.close()


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution function."""
    print("=" * 80)
    print("EXTENDED ENVIRONMENT BENCHMARK")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  - Truck counts: {TRUCK_COUNTS}")
    print(f"  - Stop counts: {STOP_COUNTS}")
    print(f"  - Episodes per config: {NUM_EPISODES}")
    print(f"  - Max steps: {MAX_STEPS_PER_EPISODE}")
    print(f"  - Output: {OUTPUT_DIR}")
    print(f"\nTotal configurations: {len(TRUCK_COUNTS) * len(STOP_COUNTS)}")
    print("\nStarting benchmark...\n")
    
    results = []
    total_configs = len(TRUCK_COUNTS) * len(STOP_COUNTS)
    
    with tqdm(total=total_configs, desc="Overall Progress") as pbar:
        for num_trucks in TRUCK_COUNTS:
            for num_stops in STOP_COUNTS:
                result = benchmark_configuration(num_trucks, num_stops, NUM_EPISODES)
                results.append(result)
                
                pbar.set_postfix({
                    'trucks': num_trucks, 
                    'stops': num_stops,
                    'time': f"{result['mean_time']:.2f}s"
                })
                pbar.update(1)
    
    print("\n" + "=" * 80)
    print("Creating visualizations...")
    plot_heatmap(results, OUTPUT_DIR)
    plot_scaling_analysis(results, OUTPUT_DIR)
    
    # Save results to file
    results_file = os.path.join(OUTPUT_DIR, 'benchmark_results_extended.txt')
    with open(results_file, 'w') as f:
        f.write("=" * 110 + "\n")
        f.write("EXTENDED BENCHMARK RESULTS\n")
        f.write("=" * 110 + "\n\n")
        f.write(f"{'Trucks':<10} {'Stops':<10} {'Time (s)':<12} {'Steps':<10} {'Time/Step (ms)':<18} {'Std Time/Step (ms)':<22} {'Complete':<12}\n")
        f.write("-" * 110 + "\n")
        for r in sorted(results, key=lambda x: (x['num_trucks'], x['num_stops'])):
            f.write(f"{r['num_trucks']:<10} {r['num_stops']:<10} "
                   f"{r['mean_time']:<12.3f} {r['mean_steps']:<10.1f} "
                   f"{r['mean_time_per_step']*1000:<18.2f} "
                   f"{r['std_time_per_step']*1000:<22.2f} "
                   f"{r['completion_rate']:<12.1%}\n")
        f.write("-" * 110 + "\n")
    
    print(f"Results table saved to: {results_file}")
    
    print("\n" + "=" * 80)
    print("EXTENDED BENCHMARK COMPLETE")
    print("=" * 80)
    print(f"\nAll results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
