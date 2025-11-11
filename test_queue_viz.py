"""
Test script to visualize charging queue dynamics.
Uses heuristic policy to ensure chargers are used.
"""

from truck_env.models.event_driven_env import EventDrivenTruckEnv
from truck_env.baselines.heuristic_policy import HeuristicPolicy

def test_queue_visualization(seed: int = 42):
    """Test queue visualization with a heuristic policy."""
    config_file = "truck_env/config_files/config.yaml"

    # Create environment with plotting enabled
    env = EventDrivenTruckEnv(
        config=config_file, 
        run_id="queue_viz_test",
        verbose=False,  # Reduce verbosity for cleaner output
        enable_plotting=True
    )

    obs, info = env.reset(seed=seed)
    env.action_space.seed(seed)

    print("\n" + "="*80)
    print("TESTING CHARGING QUEUE VISUALIZATION")
    print("="*80)
    print(f"Environment: {env.num_trucks} trucks, {env.num_charging_nodes} chargers")
    print(f"Max simulation time: {env.max_time} hours")
    print("="*80 + "\n")

    total_reward = 0.0
    total_steps = 0
    max_steps = 10000  # Safety limit to prevent infinite loops
    
    # Use heuristic policy for more realistic behavior
    policy = HeuristicPolicy(verbose=False)

    while total_steps < max_steps:
        # Use heuristic policy to get action (pass environment, not observation)
        action = policy.get_action(env)
        
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        total_steps += 1

        # Print progress every 10 steps
        if total_steps % 10 == 0:
            active_trucks = info.get('num_active_trucks', 0)
            print(f"Step {total_steps}: Time={info['global_clock']:.1f}h, "
                  f"Active Trucks={active_trucks}, Reward={reward:.2f}")

        if done or truncated:
            break
    
    if total_steps >= max_steps:
        print(f"\n⚠ WARNING: Reached maximum steps ({max_steps}). Possible infinite loop.")

    print("\n" + "="*80)
    print("SIMULATION COMPLETE")
    print("="*80)
    print(f"Total Steps: {total_steps}")
    print(f"Total Time: {info['global_clock']:.2f} hours")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"All trucks complete: {info.get('all_complete', False)}")
    print(f"Any trucks failed: {info.get('any_failed', False)}")
    print("="*80)
    
    # Print charging statistics
    charger_util = info.get('charger_utilization', {})
    if charger_util:
        print("\nCHARGING STATISTICS:")
        print("-" * 80)
        overall = charger_util.get('overall', {})
        print(f"Overall avg utilization: {overall.get('avg_utilization', 0)*100:.1f}%")
        print(f"Total charge sessions: {overall.get('total_sessions', 0)}")
        print(f"Total charge time: {overall.get('total_charge_time', 0):.1f} hours")
        
        print("\nBy Charger Type:")
        level2 = charger_util.get('level2', {})
        dcfast = charger_util.get('dcfast', {})
        print(f"  Level2: {level2.get('num_chargers', 0)} chargers, "
              f"{level2.get('avg_utilization', 0)*100:.1f}% avg utilization")
        print(f"  DCFast: {dcfast.get('num_chargers', 0)} chargers, "
              f"{dcfast.get('avg_utilization', 0)*100:.1f}% avg utilization")
        print("-" * 80)
    
    # Close environment to generate visualizations
    print("\nGenerating visualizations...")
    env.close()
    print(f"\n✓ All visualizations saved to: results/queue_viz_test/")
    print("  - initial_state.png")
    print("  - final_routes.png")
    print("  - charger_queue_dynamics.png")
    print("  - charger_utilization_heatmap.png")
    print("  - charger_statistics_summary.png")
    print("  - statistics.txt")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    test_queue_visualization(seed=42)
