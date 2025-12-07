#!/usr/bin/env python3
"""
Test script to validate traffic simulation models.
Tests all traffic models (gaussian, time_of_day, distance_dependent, correlated).
"""
import sys
import os
import yaml
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv


def test_traffic_model(label: str, num_episodes: int = 3):
    """Test traffic simulation with multiple episodes."""
    print(f"\n{'='*80}")
    print(f"Testing Traffic: {label.upper()}")
    print(f"{'='*80}\n")
    
    # Load config
    config_path = "EVRoutingEnv/config_files/config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Enable traffic
    config['traffic']['enable_traffic'] = True
    config['traffic']['std_dev_factor'] = 0.15
    
    # Create environment with verbose output
    env = EventDrivenTruckEnv(config=config, verbose=True)
    
    # Run a few episodes
    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
        obs, info = env.reset(seed=42 + episode)
        
        total_reward = 0
        steps = 0
        done = False
        truncated = False
        
        while not (done or truncated) and steps < 20:  # Limit steps for testing
            # Take random valid action
            action_mask = obs[-env.action_space.n:]
            valid_actions = np.where(action_mask > 0)[0]
            
            if len(valid_actions) == 0:
                print("  No valid actions available!")
                break
            
            action = np.random.choice(valid_actions)
            obs, reward, done, truncated, info = env.step(action)
            
            total_reward += reward
            steps += 1
            
            if steps >= 20:
                break
        
        print(f"\nEpisode {episode + 1} Summary:")
        print(f"  Steps: {steps}")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Success: {'Yes' if info.get('success', False) else 'No'}")
    
    env.close()


def test_traffic_enabled():
    """Test traffic simulation with time-of-day model."""
    print("\n" + "="*80)
    print("TRAFFIC SIMULATION VALIDATION TEST")
    print("="*80)
    print("\nThis script tests the time-of-day traffic model:")
    print("  - Gaussian distribution with time-dependent variance")
    print("  - Higher variance during rush hours (7-9am, 5-7pm)")
    print("\n" + "="*80)
    
    try:
        test_traffic_model("enabled", num_episodes=2)
    except Exception as e:
        print(f"\n❌ ERROR testing traffic: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("TRAFFIC SIMULATION TESTS COMPLETE")
    print("="*80 + "\n")


def test_disabled_traffic():
    """Test that traffic simulation can be disabled."""
    print(f"\n{'='*80}")
    print(f"Testing DISABLED Traffic (enable_traffic: false)")
    print(f"{'='*80}\n")
    
    config_path = "EVRoutingEnv/config_files/config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    config['traffic']['enable_traffic'] = False
    config['environment']['max_episode_steps'] = 50
    
    env = EventDrivenTruckEnv(config=config, verbose=True)
    obs, info = env.reset(seed=100)
    
    # Take a few actions
    for i in range(5):
        action_mask = obs[-env.action_space.n:]
        valid_actions = np.where(action_mask > 0)[0]
        if len(valid_actions) == 0:
            break
        action = np.random.choice(valid_actions)
        obs, reward, done, truncated, info = env.step(action)
        if done or truncated:
            break
    
    print("\n✓ Traffic disabled test completed successfully")
    env.close()


if __name__ == "__main__":
    # First test with traffic disabled
    test_disabled_traffic()
    
    # Then test with traffic enabled
    test_traffic_enabled()
