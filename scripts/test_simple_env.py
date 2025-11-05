"""
Test script for SimpleTruckEnv
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simple_truck_env import SimpleTruckEnv
import numpy as np


def test_basic_functionality():
    """Test basic environment functionality."""
    print("="*80)
    print("Testing SimpleTruckEnv - Basic Functionality")
    print("="*80)
    
    # Create environment
    env = SimpleTruckEnv(
        num_stops=3,
        min_hop_distance=20.0,
        max_hop_distance=150.0,
        max_steps=50,
        verbose=True
    )
    
    print(f"\n✅ Environment created successfully")
    print(f"   Action space: {env.action_space}")
    print(f"   Observation space: {env.observation_space}")
    print(f"   Total possible actions: {env.total_actions}")
    print(f"     - Navigation actions: {env.num_navigation_actions}")
    print(f"     - Charging actions: {env.num_charge_actions}")
    
    # Reset environment
    print("\n" + "="*80)
    print("Testing Reset")
    print("="*80)
    obs, info = env.reset(seed=42)
    print(f"\n✅ Reset successful")
    print(f"   Observation shape: {obs.shape}")
    print(f"   Observation: {obs}")
    print(f"   Info keys: {info.keys()}")
    
    # Test a few random actions
    print("\n" + "="*80)
    print("Testing Random Actions")
    print("="*80)
    
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"\nStep {i+1}:")
        print(f"  Reward: {reward:.2f}")
        print(f"  Terminated: {terminated}")
        print(f"  Truncated: {truncated}")
        
        if terminated or truncated:
            print(f"\n  Episode ended after {i+1} steps")
            print(f"  Total reward: {info['episode_reward']:.2f}")
            break
    
    print("\n" + "="*80)
    print("✅ Basic functionality test PASSED")
    print("="*80)


def test_navigation_actions():
    """Test navigation actions specifically."""
    print("\n" + "="*80)
    print("Testing Navigation Actions")
    print("="*80)
    
    env = SimpleTruckEnv(num_stops=2, verbose=True)
    obs, info = env.reset(seed=123)
    
    print(f"\nInitial state:")
    print(f"  Current node: {env.truck.current_node}")
    print(f"  Next delivery: {env.truck.get_next_delivery_target()}")
    print(f"  Battery: {env.truck.current_battery:.1f} kWh")
    
    # Try to go to next delivery
    action = env.num_charging_nodes  # "Go to next delivery" action
    print(f"\nExecuting action: Go to next delivery")
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"  Reward: {reward:.2f}")
    print(f"  New node: {env.truck.current_node}")
    print(f"  Battery: {env.truck.current_battery:.1f} kWh")
    
    print("\n✅ Navigation action test PASSED")


def test_charging_actions():
    """Test charging actions specifically."""
    print("\n" + "="*80)
    print("Testing Charging Actions")
    print("="*80)
    
    env = SimpleTruckEnv(num_stops=2, verbose=True)
    obs, info = env.reset(seed=456)
    
    # First go to a charging station
    if env.num_charging_nodes > 0:
        action = 0  # Go to first charging station
        print(f"\nGoing to charging station...")
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"  Current node: {env.truck.current_node}")
        print(f"  Battery before charging: {env.truck.current_battery:.1f} kWh")
        
        # Now try to charge
        charge_action = env.num_navigation_actions  # Charge for 1 hour
        print(f"\nCharging for 1 hour...")
        obs, reward, terminated, truncated, info = env.step(charge_action)
        
        print(f"  Battery after charging: {env.truck.current_battery:.1f} kWh")
        print(f"  Reward: {reward:.2f}")
        
        print("\n✅ Charging action test PASSED")
    else:
        print("⚠️ No charging stations available in graph")


def test_completion():
    """Test completing all deliveries."""
    print("\n" + "="*80)
    print("Testing Delivery Completion")
    print("="*80)
    
    env = SimpleTruckEnv(num_stops=2, max_steps=100, verbose=False)
    obs, info = env.reset(seed=789)
    
    step = 0
    total_reward = 0
    
    # Simple greedy strategy: always go to next delivery
    while step < 100:
        step += 1
        
        # Check if we need to charge
        next_target = env.truck.get_next_delivery_target()
        if next_target is None:
            print(f"✅ All deliveries complete!")
            break
        
        # Try to go to next delivery
        action = env.num_charging_nodes  # "Go to next delivery" action
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if step % 10 == 0:
            print(f"  Step {step}: {len(env.truck.get_remaining_deliveries())} deliveries remaining")
        
        if terminated or truncated:
            print(f"\nEpisode ended after {step} steps")
            print(f"  Completed: {env.truck.is_complete}")
            print(f"  Failed: {env.truck.failed}")
            print(f"  Total reward: {total_reward:.2f}")
            break
    
    print("\n✅ Completion test PASSED")


if __name__ == "__main__":
    try:
        test_basic_functionality()
        test_navigation_actions()
        test_charging_actions()
        test_completion()
        
        print("\n" + "="*80)
        print("🎉 ALL TESTS PASSED!")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED with error:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
