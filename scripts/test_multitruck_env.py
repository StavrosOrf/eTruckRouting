"""
Test script for SimpleTruckEnv with multiple trucks and MultiDiscrete actions
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simple_truck_env import SimpleTruckEnv, create_env_from_config, load_config
import numpy as np


def test_config_loading():
    """Test loading environment from config file."""
    print("="*80)
    print("TEST 1: Loading Multi-Truck Environment from Config")
    print("="*80)
    
    # Load config
    config = load_config()
    print(f"\nConfig loaded:")
    print(f"  Number of trucks: {config['advanced']['num_trucks']}")
    print(f"  Delivery stops: {config['environment']['num_stops']}")
    
    # Create environment from config
    env = create_env_from_config()
    
    print(f"\n✅ Environment created:")
    print(f"   Action space: {env.action_space}")
    print(f"   - Type: MultiDiscrete")
    print(f"   - Shape: {env.action_space.nvec.shape}")
    print(f"   - Values: {env.action_space.nvec}")
    print(f"   Observation space: {env.observation_space.shape}")
    print(f"   Number of trucks: {env.num_trucks}")
    
    env.close()
    print("\n✅ TEST PASSED\n")


def test_multitruck_reset():
    """Test reset with multiple trucks."""
    print("="*80)
    print("TEST 2: Multi-Truck Reset")
    print("="*80)
    
    env = create_env_from_config()
    obs, info = env.reset(seed=42)
    
    print(f"\n✅ Environment reset:")
    print(f"   Observation shape: {obs.shape}")
    print(f"   Number of trucks: {info['num_trucks']}")
    
    for i, truck_state in enumerate(info['trucks']):
        print(f"\n   Truck {i}:")
        print(f"     Type: {truck_state['truck_type']}")
        print(f"     Start: node {truck_state['delivery_sequence'][0]}")
        print(f"     Deliveries: {len(truck_state['delivery_sequence']) - 1}")
        print(f"     Battery: {truck_state['current_battery']:.1f}/{truck_state['battery_capacity']:.1f} kWh")
    
    env.close()
    print("\n✅ TEST PASSED\n")


def test_multidiscrete_actions():
    """Test MultiDiscrete action execution."""
    print("="*80)
    print("TEST 3: MultiDiscrete Action Execution")
    print("="*80)
    
    env = create_env_from_config()
    obs, info = env.reset(seed=123)
    
    print(f"\nAction space nvec: {env.action_space.nvec}")
    print(f"Expected format: [action_0, action_1, action_2]")
    print(f"Each action: 0-{env.num_charging_nodes-1}=chargers, {env.num_charging_nodes}=delivery, {env.num_charging_nodes+1}+=charge")
    
    # Create a manual action for all trucks
    # Truck 0: go to next delivery
    # Truck 1: charge for 2 hours
    # Truck 2: go to next delivery
    
    num_trucks = env.num_trucks
    action = []
    for i in range(num_trucks):
        if i == 1:
            # Truck 1: charge for 2 hours (charging action index 1 = 2 hours)
            truck_action = env.num_navigation_actions + 1  # charge 2h
        else:
            # Other trucks: go to delivery
            truck_action = env.num_charging_nodes  # Next delivery
        
        action.append(truck_action)
    
    action = np.array(action)
    print(f"\nExecuting action: {action}")
    
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"\n✅ Step executed:")
    print(f"   Reward: {reward:.2f}")
    print(f"   Terminated: {terminated}")
    print(f"   All complete: {info['all_complete']}")
    print(f"   Any failed: {info['any_failed']}")
    
    env.close()
    print("\n✅ TEST PASSED\n")


def test_random_episode():
    """Test complete episode with random actions."""
    print("="*80)
    print("TEST 4: Random Episode with Multiple Trucks")
    print("="*80)
    
    config = load_config()
    # Make it easier to complete
    config['environment']['num_stops'] = 2
    config['environment']['max_steps'] = 100
    config['advanced']['num_trucks'] = 2
    
    env = SimpleTruckEnv(config=config, verbose=False)
    obs, info = env.reset(seed=456)
    
    print(f"\nRunning episode with {info['num_trucks']} trucks...")
    
    for step in range(100):
        # Random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if step % 10 == 0:
            active_trucks = sum(1 for t in info['trucks'] if not (t['is_complete'] or t['failed']))
            print(f"  Step {step}: Active trucks: {active_trucks}, Reward: {reward:.2f}")
        
        if terminated or truncated:
            print(f"\nEpisode ended at step {step}")
            print(f"  All complete: {info['all_complete']}")
            print(f"  Any failed: {info['any_failed']}")
            print(f"  Total reward: {info['episode_reward']:.2f}")
            break
    
    env.close()
    print("\n✅ TEST PASSED\n")


def test_charging_queue():
    """Test charging queue simulation."""
    print("="*80)
    print("TEST 5: Charging Queue Simulation")
    print("="*80)
    
    config = load_config()
    config['advanced']['num_trucks'] = 3
    config['environment']['verbose'] = True
    
    env = SimpleTruckEnv(config=config)
    obs, info = env.reset(seed=789)
    
    print(f"\nSending all trucks to charge for 2 hours...")
    
    # Send all trucks to charge (action = nav_actions + 1 = charge 2h)
    action = np.array([env.num_navigation_actions + 1] * env.num_trucks)
    
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"\n✅ Charging queue simulation completed")
    print(f"   Total reward: {reward:.2f}")
    
    env.close()
    print("\n✅ TEST PASSED\n")


def test_action_space_verification():
    """Verify action space is correctly configured."""
    print("="*80)
    print("TEST 6: Action Space Verification")
    print("="*80)
    
    for num_trucks in [1, 2, 5]:
        config = load_config()
        config['advanced']['num_trucks'] = num_trucks
        
        env = SimpleTruckEnv(config=config)
        
        # Check action space
        expected_length = num_trucks  # 1 action per truck
        actual_length = len(env.action_space.nvec)
        
        print(f"\nTrucks: {num_trucks}")
        print(f"  Expected action length: {expected_length}")
        print(f"  Actual action length: {actual_length}")
        print(f"  Match: {expected_length == actual_length}")
        
        # Sample action
        action = env.action_space.sample()
        print(f"  Sample action shape: {action.shape}")
        print(f"  Sample action: {action}")
        print(f"  Action range per truck: 0 to {env.num_navigation_actions + env.num_charge_actions - 1}")
        
        assert expected_length == actual_length, f"Action space mismatch for {num_trucks} trucks"
        
        env.close()
    
    print("\n✅ TEST PASSED\n")


def test_observation_space_verification():
    """Verify observation space is correctly sized."""
    print("="*80)
    print("TEST 7: Observation Space Verification")
    print("="*80)
    
    obs_dim_per_truck = 10
    
    for num_trucks in [1, 2, 5]:
        config = load_config()
        config['advanced']['num_trucks'] = num_trucks
        
        env = SimpleTruckEnv(config=config)
        obs, info = env.reset()
        
        expected_dim = num_trucks * obs_dim_per_truck
        actual_dim = obs.shape[0]
        
        print(f"\nTrucks: {num_trucks}")
        print(f"  Expected obs dimension: {expected_dim}")
        print(f"  Actual obs dimension: {actual_dim}")
        print(f"  Match: {expected_dim == actual_dim}")
        
        assert expected_dim == actual_dim, f"Observation space mismatch for {num_trucks} trucks"
        
        env.close()
    
    print("\n✅ TEST PASSED\n")


if __name__ == "__main__":
    print("\n" + "🚛 " * 20)
    print("Multi-Truck SimpleTruckEnv Test Suite")
    print("🚛 " * 20 + "\n")
    
    try:
        test_config_loading()
        test_multitruck_reset()
        test_multidiscrete_actions()
        test_random_episode()
        test_charging_queue()
        test_action_space_verification()
        test_observation_space_verification()
        
        print("\n" + "="*80)
        print("🎉 ALL TESTS PASSED!")
        print("="*80)
        print("\nMulti-truck environment with MultiDiscrete actions is working correctly!")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED with error:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
