"""
Validation Experiment for Realistic DC Fast Charging Curves.

This script validates the implementation of the CCCV (Constant Current - Constant Voltage)
charging model by comparing it against the linear charging model across different
SOC ranges and charging durations.

The script:
1. Creates controlled charging scenarios
2. Runs episodes with both linear and realistic charging modes
3. Generates detailed logs showing power curves and SOC progression
4. Produces summary statistics comparing charging efficiency and duration
"""

import sys
import os
import numpy as np
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from EVRoutingEnv.models.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.utils.utils import load_config


def create_test_config(use_realistic_curve: bool, output_dir: str) -> dict:
    """
    Create a test configuration for validation.
    
    Args:
        use_realistic_curve: Whether to use realistic CCCV model
        output_dir: Directory for output files
        
    Returns:
        Configuration dictionary
    """
    # Load base config using utility function
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "EVRoutingEnv/config_files/config.yaml"
    )
    
    config = load_config(config_path)
    
    # Override charging configuration
    config['charging']['use_realistic_curve'] = use_realistic_curve
    config['charging']['dcfast']['charge_rate'] = 150.0  # 150 kW peak
    config['charging']['dcfast']['taper_start_soc'] = 0.8  # Taper at 80%
    config['charging']['dcfast']['taper_power_min'] = 30.0  # Min 30 kW
    
    # Small test scenario
    config['environment']['num_trucks'] = 3
    config['environment']['num_stops'] = 2
    config['environment']['max_time'] = 100
    config['environment']['verbose'] = True
    
    # Set truck initial battery to partial (to trigger charging)
    config['truck']['initial_battery'] = 30.0  # Start at 30% to force charging
    
    return config


def run_validation_episode(config: dict, run_id: str, verbose: bool = True) -> dict:
    """
    Run a single validation episode.
    
    Args:
        config: Environment configuration
        run_id: Unique run identifier
        verbose: Enable verbose output
        
    Returns:
        Dictionary of episode results
    """
    # Create environment
    env = EventDrivenTruckEnv(
        config=config,
        verbose=verbose,
        enable_plotting=True,
        run_id=run_id
    )
    
    # Run episode with random actions (simple heuristic)
    obs, info = env.reset(seed=42)
    done = False
    truncated = False
    step_count = 0
    total_reward = 0.0
    
    while not (done or truncated):
        # Get feasible actions
        action_mask = env.mask_fn()
        feasible_actions = np.where(action_mask)[0]
        
        if len(feasible_actions) == 0:
            if verbose:
                print(f"[WARNING] No feasible actions at step {step_count}")
            break
        
        # Simple heuristic: prefer charging at chargers, else go to next delivery
        action = None
        active_truck = env.trucks[env.active_truck_id]
        
        if active_truck.current_node in env.charging_nodes:
            # At charger - charge for a reasonable duration based on SOC
            soc = active_truck.get_battery_percentage()
            if soc < 50:
                # Low battery - charge longer (try 3 hours)
                charge_actions = list(range(env.num_navigation_actions, env.action_space.n))
                feasible_charge = [a for a in charge_actions if action_mask[a]]
                if feasible_charge and len(feasible_charge) >= 3:
                    action = feasible_charge[2]  # 3 hours
                elif feasible_charge:
                    action = feasible_charge[0]
            elif soc < 80:
                # Medium battery - charge moderate (try 2 hours)
                charge_actions = list(range(env.num_navigation_actions, env.action_space.n))
                feasible_charge = [a for a in charge_actions if action_mask[a]]
                if feasible_charge and len(feasible_charge) >= 2:
                    action = feasible_charge[1]  # 2 hours
                elif feasible_charge:
                    action = feasible_charge[0]
        
        # If no charge action selected, go to next delivery
        if action is None:
            nav_actions = list(range(env.num_navigation_actions))
            feasible_nav = [a for a in nav_actions if action_mask[a]]
            if feasible_nav:
                # Prefer next delivery
                if action_mask[env.num_charging_nodes]:  # Next delivery action
                    action = env.num_charging_nodes
                else:
                    action = np.random.choice(feasible_nav)
        
        # Fallback to random feasible action
        if action is None:
            action = np.random.choice(feasible_actions)
        
        # Take step
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        
        if step_count >= 1000:  # Safety limit
            if verbose:
                print(f"[WARNING] Reached step limit")
            break
    
    # Collect results
    results = {
        "run_id": run_id,
        "use_realistic_curve": config['charging']['use_realistic_curve'],
        "total_reward": total_reward,
        "final_time": env.global_clock,
        "steps": step_count,
        "all_complete": info['all_complete'],
        "any_failed": info['any_failed'],
        "trucks": []
    }
    
    # Collect truck statistics
    for truck in env.trucks:
        truck_stats = {
            "truck_id": truck.truck_id,
            "is_complete": truck.is_complete,
            "failed": truck.failed,
            "total_distance": truck.total_distance_traveled,
            "total_charging_time": truck.total_charging_time,
            "num_charging_sessions": truck.num_charging_sessions,
            "final_battery": truck.current_battery,
            "final_soc_pct": truck.get_battery_percentage()
        }
        results["trucks"].append(truck_stats)
    
    # Close environment (saves logs)
    env.close()
    
    return results


def compare_charging_modes():
    """
    Main validation function - compares linear vs realistic charging.
    """
    print("="*80)
    print("DC Fast Charging Curve Validation Experiment")
    print("="*80)
    print()
    
    # Create output directory
    output_base = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "results/charging_validation"
    )
    os.makedirs(output_base, exist_ok=True)
    
    results_all = []
    
    # Run with linear charging
    print("\n" + "="*80)
    print("EXPERIMENT 1: Linear (Constant-Rate) Charging")
    print("="*80)
    
    output_linear = os.path.join(output_base, "linear_charging")
    config_linear = create_test_config(use_realistic_curve=False, output_dir=output_linear)
    results_linear = run_validation_episode(
        config=config_linear,
        run_id="linear_charging",
        verbose=True
    )
    results_all.append(results_linear)
    
    print(f"\n[LINEAR] Episode completed:")
    print(f"  Total time: {results_linear['final_time']:.2f}h")
    print(f"  Total reward: {results_linear['total_reward']:.2f}")
    print(f"  All complete: {results_linear['all_complete']}")
    print(f"  Any failed: {results_linear['any_failed']}")
    
    # Run with realistic charging
    print("\n" + "="*80)
    print("EXPERIMENT 2: Realistic (CCCV) Charging with SOC Tapering")
    print("="*80)
    
    output_realistic = os.path.join(output_base, "realistic_charging")
    config_realistic = create_test_config(use_realistic_curve=True, output_dir=output_realistic)
    results_realistic = run_validation_episode(
        config=config_realistic,
        run_id="realistic_charging",
        verbose=True
    )
    results_all.append(results_realistic)
    
    print(f"\n[REALISTIC] Episode completed:")
    print(f"  Total time: {results_realistic['final_time']:.2f}h")
    print(f"  Total reward: {results_realistic['total_reward']:.2f}")
    print(f"  All complete: {results_realistic['all_complete']}")
    print(f"  Any failed: {results_realistic['any_failed']}")
    
    # Compare results
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    
    print("\nCharging Time Comparison:")
    for i, results in enumerate(results_all):
        mode = "Linear" if i == 0 else "Realistic"
        total_charging_time = sum(t['total_charging_time'] for t in results['trucks'])
        avg_charging_time = total_charging_time / len(results['trucks'])
        print(f"  {mode:12s}: {total_charging_time:.2f}h total, {avg_charging_time:.2f}h avg per truck")
    
    print("\nCharging Sessions Comparison:")
    for i, results in enumerate(results_all):
        mode = "Linear" if i == 0 else "Realistic"
        total_sessions = sum(t['num_charging_sessions'] for t in results['trucks'])
        avg_sessions = total_sessions / len(results['trucks'])
        print(f"  {mode:12s}: {total_sessions} total sessions, {avg_sessions:.1f} avg per truck")
    
    print("\nEpisode Duration:")
    time_diff = results_realistic['final_time'] - results_linear['final_time']
    time_diff_pct = (time_diff / results_linear['final_time']) * 100
    print(f"  Linear:     {results_linear['final_time']:.2f}h")
    print(f"  Realistic:  {results_realistic['final_time']:.2f}h")
    print(f"  Difference: {time_diff:+.2f}h ({time_diff_pct:+.1f}%)")
    
    # Save comparison results
    comparison_file = os.path.join(output_base, "comparison_results.json")
    with open(comparison_file, 'w') as f:
        json.dump({
            "linear": results_linear,
            "realistic": results_realistic,
            "summary": {
                "time_difference_hours": time_diff,
                "time_difference_percent": time_diff_pct
            }
        }, f, indent=2)
    
    print(f"\n[SAVED] Comparison results to: {comparison_file}")
    print(f"[SAVED] Linear logs to: {output_linear}/charging_logs/")
    print(f"[SAVED] Realistic logs to: {output_realistic}/charging_logs/")
    
    print("\n" + "="*80)
    print("Validation Complete!")
    print("="*80)
    print("\nCheck the charging_logs/ subdirectories for detailed session logs:")
    print("  - charging_sessions_*.json: Individual charging session details")
    print("  - charging_summary_*.json: Aggregated statistics by model type")
    print("\nKey metrics to examine:")
    print("  - taper_factor: Ratio of avg to peak power (1.0 = no taper, <1.0 = tapered)")
    print("  - power_curve: Time-series (time, power, soc) for each session")
    print("  - model_used: 'linear' or 'cccv'")


if __name__ == "__main__":
    compare_charging_modes()
