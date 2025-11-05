"""
Example: Using SimpleTruckEnv with Multiple Trucks and Config File

This script demonstrates how to:
1. Load configuration from YAML
2. Create multi-truck environment
3. Use MultiDiscrete action space
4. Monitor multiple trucks
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simple_truck_env import SimpleTruckEnv, load_config, print_config_summary
import numpy as np


def example_1_basic_usage():
    """Example 1: Basic usage with default config."""
    print("="*80)
    print("EXAMPLE 1: Basic Multi-Truck Environment")
    print("="*80)
    
    # Create environment from default config
    env = SimpleTruckEnv()
    
    print(f"\nEnvironment created:")
    print(f"  Number of trucks: {env.num_trucks}")
    print(f"  Action space: {env.action_space}")
    print(f"  Observation space: {env.observation_space.shape}")
    
    # Reset environment
    obs, info = env.reset(seed=42)
    
    print(f"\nEpisode started with {info['num_trucks']} trucks")
    
    # Run a few steps
    for step in range(10):
        # Sample random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"\nStep {step + 1}: Reward = {reward:.2f}")
        
        if terminated or truncated:
            break
    
    env.close()
    print("\n" + "="*80 + "\n")


def example_2_custom_config():
    """Example 2: Using custom configuration."""
    print("="*80)
    print("EXAMPLE 2: Custom Configuration")
    print("="*80)
    
    # Load config
    config = load_config()
    
    # Modify configuration
    config['advanced']['num_trucks'] = 5
    config['environment']['num_stops'] = 3
    config['environment']['max_steps'] = 200
    
    # Create environment with custom config
    env = SimpleTruckEnv(config=config)
    
    print(f"\nCustom environment:")
    print(f"  Trucks: {env.num_trucks}")
    print(f"  Stops per truck: {env.num_stops}")
    print(f"  Max steps: {env.max_steps}")
    
    obs, info = env.reset()
    
    # Show truck details
    for truck_state in info['trucks']:
        print(f"\n  Truck {truck_state['truck_id']}:")
        print(f"    Type: {truck_state['truck_type']}")
        print(f"    Battery: {truck_state['battery_capacity']:.1f} kWh")
        print(f"    Deliveries: {truck_state['deliveries_remaining']}")
    
    env.close()
    print("\n" + "="*80 + "\n")


def example_3_manual_actions():
    """Example 3: Manually controlling trucks."""
    print("="*80)
    print("EXAMPLE 3: Manual Truck Control")
    print("="*80)
    
    # Create simple environment
    config = load_config()
    config['advanced']['num_trucks'] = 2
    config['environment']['num_stops'] = 2
    config['environment']['verbose'] = True
    
    env = SimpleTruckEnv(config=config)
    obs, info = env.reset(seed=100)
    
    print("\nManually controlling 2 trucks:")
    print("  Truck 0: Go to next delivery, no charge")
    print("  Truck 1: Go to charging station 0, charge for 1 hour")
    
    # Build action manually
    # Format: [nav_0, charge_0, nav_1, charge_1]
    action = np.array([
        env.num_charging_nodes,  # Truck 0: go to next delivery
        0,                        # Truck 0: no charging
        0,                        # Truck 1: go to charging station 0
        1,                        # Truck 1: charge for 1 hour
    ])
    
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"\nResult:")
    print(f"  Reward: {reward:.2f}")
    print(f"  Episode reward: {info['episode_reward']:.2f}")
    
    env.close()
    print("\n" + "="*80 + "\n")


def example_4_coordinated_strategy():
    """Example 4: Simple coordinated strategy."""
    print("="*80)
    print("EXAMPLE 4: Coordinated Truck Strategy")
    print("="*80)
    
    config = load_config()
    config['advanced']['num_trucks'] = 3
    config['environment']['num_stops'] = 3
    config['environment']['max_steps'] = 150
    config['environment']['verbose'] = False
    
    env = SimpleTruckEnv(config=config)
    obs, info = env.reset(seed=200)
    
    print(f"\nRunning coordinated strategy with {env.num_trucks} trucks...")
    print("Strategy: Trucks go to deliveries, charge only when battery < 30%")
    
    for step in range(150):
        # Build action based on simple strategy
        action = []
        
        for truck_state in info['trucks']:
            truck_id = truck_state['truck_id']
            battery_pct = truck_state['battery_percentage']
            
            # Navigation action
            if battery_pct < 30.0:
                # Low battery - find nearest charger
                nav_action = 0  # Go to first charging station (simplified)
            else:
                # Go to next delivery
                nav_action = env.num_charging_nodes
            
            # Charging action
            if battery_pct < 50.0 and truck_state['current_node'] in env.charging_nodes:
                charge_action = 2  # Charge for 2 hours
            else:
                charge_action = 0  # No charging
            
            action.extend([nav_action, charge_action])
        
        action = np.array(action)
        obs, reward, terminated, truncated, info = env.step(action)
        
        if step % 20 == 0:
            active = sum(1 for t in info['trucks'] if not (t['is_complete'] or t['failed']))
            print(f"  Step {step}: Active={active}, Reward={reward:.2f}, Total={info['episode_reward']:.2f}")
        
        if terminated or truncated:
            print(f"\nEpisode ended at step {step}")
            print(f"  All complete: {info['all_complete']}")
            print(f"  Any failed: {info['any_failed']}")
            print(f"  Total reward: {info['episode_reward']:.2f}")
            
            # Show final truck states
            for truck_state in info['trucks']:
                status = "✅ Complete" if truck_state['is_complete'] else ("❌ Failed" if truck_state['failed'] else "🚛 Active")
                print(f"    Truck {truck_state['truck_id']}: {status}, Time={truck_state['total_time']:.1f}h")
            break
    
    env.close()
    print("\n" + "="*80 + "\n")


def example_5_action_space_explained():
    """Example 5: Understanding the MultiDiscrete action space."""
    print("="*80)
    print("EXAMPLE 5: Understanding MultiDiscrete Actions")
    print("="*80)
    
    config = load_config()
    config['advanced']['num_trucks'] = 2
    
    env = SimpleTruckEnv(config=config)
    
    print(f"\nAction space structure:")
    print(f"  Type: MultiDiscrete")
    print(f"  Shape: {env.action_space.nvec}")
    print(f"  Number of trucks: {env.num_trucks}")
    
    print(f"\nEach truck has 2 action components:")
    print(f"  1. Navigation action (0 to {env.num_navigation_actions - 1}):")
    print(f"     - 0 to {env.num_charging_nodes - 1}: Go to specific charging station")
    print(f"     - {env.num_charging_nodes}: Go to next delivery")
    print(f"  2. Charging action (0 to {env.num_charge_actions - 1}):")
    print(f"     - 0: No charging")
    print(f"     - 1 to {env.num_charge_actions - 1}: Charge for 1-4 hours")
    
    print(f"\nAction array format: [nav_0, charge_0, nav_1, charge_1, ...]")
    print(f"For {env.num_trucks} trucks: length = {len(env.action_space.nvec)}")
    
    # Show some example actions
    print(f"\nExample actions:")
    
    # All trucks go to deliveries
    action1 = np.array([env.num_charging_nodes, 0] * env.num_trucks)
    print(f"  All to deliveries: {action1}")
    
    # First truck charges
    action2 = np.array([0, 2, env.num_charging_nodes, 0])
    print(f"  Truck 0 charges at station 0 for 2h, Truck 1 to delivery: {action2}")
    
    # Random action
    action3 = env.action_space.sample()
    print(f"  Random sample: {action3}")
    
    env.close()
    print("\n" + "="*80 + "\n")


def example_6_monitoring_trucks():
    """Example 6: Monitoring individual trucks during execution."""
    print("="*80)
    print("EXAMPLE 6: Monitoring Individual Trucks")
    print("="*80)
    
    config = load_config()
    config['advanced']['num_trucks'] = 3
    config['environment']['num_stops'] = 2
    config['environment']['verbose'] = False
    
    env = SimpleTruckEnv(config=config)
    obs, info = env.reset(seed=300)
    
    print(f"\nMonitoring {env.num_trucks} trucks over 10 steps...")
    
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"\nStep {step + 1}:")
        for truck_state in info['trucks']:
            battery_pct = truck_state['battery_percentage']
            deliveries_left = truck_state['deliveries_remaining']
            
            # Battery indicator
            if battery_pct > 70:
                battery_icon = "🟢"
            elif battery_pct > 30:
                battery_icon = "🟡"
            else:
                battery_icon = "🔴"
            
            # Status
            if truck_state['is_complete']:
                status = "✅ DONE"
            elif truck_state['failed']:
                status = "❌ FAIL"
            else:
                status = f"📦 {deliveries_left} left"
            
            print(f"  Truck {truck_state['truck_id']}: {battery_icon} {battery_pct:5.1f}% | {status} | Time: {truck_state['total_time']:5.1f}h")
        
        if terminated or truncated:
            print(f"\nEpisode complete!")
            break
    
    env.close()
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    print("\n" + "🚛 " * 20)
    print("SimpleTruckEnv - Multi-Truck Examples")
    print("🚛 " * 20 + "\n")
    
    try:
        example_1_basic_usage()
        example_2_custom_config()
        example_3_manual_actions()
        example_4_coordinated_strategy()
        example_5_action_space_explained()
        example_6_monitoring_trucks()
        
        print("="*80)
        print("All examples completed successfully!")
        print("="*80)
        print("\nKey takeaways:")
        print("  • Use config.yaml to configure the environment")
        print("  • MultiDiscrete action space: [nav, charge] per truck")
        print("  • Monitor trucks via info['trucks']")
        print("  • Coordinate multiple trucks with custom strategies")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ Example failed with error:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
