"""
Simple test script for HierarchicalTruckRoutingEnv with verbose output.
Tests the environment with random actions and shows detailed debug information.
"""

import sys
import os

# Add parent directory to path to enable imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from truck_env.truck_env import HierarchicalTruckRoutingEnv


def test_verbose_mode():
    """Test environment with verbose=True."""
    print("\n" + "="*80)
    print("TEST 1: VERBOSE MODE (verbose=True, debug=False)")
    print("="*80 + "\n")
    
    env = HierarchicalTruckRoutingEnv(config={"verbose": True, "debug": False})
    
    # Run a few steps
    obs, info = env.reset()
    
    for step in range(5):
        # Get random actions
        actions = {agent_id: env.get_action_space(agent_id).sample() 
                  for agent_id in env.agents}
        
        obs, rewards, terminateds, truncateds, infos = env.step(actions)
        
        if terminateds.get("__all__", False):
            print(f"\n✅ Episode terminated at step {step + 1}")
            break
    
    print("\n" + "="*80 + "\n")


def test_debug_mode():
    """Test environment with debug=True."""
    print("\n" + "="*80)
    print("TEST 2: DEBUG MODE (verbose=True, debug=True)")
    print("="*80 + "\n")
    
    env = HierarchicalTruckRoutingEnv(config={"verbose": True, "debug": True})
    
    # Run a few steps
    obs, info = env.reset()
    
    for step in range(3):
        # Get random actions
        actions = {agent_id: env.get_action_space(agent_id).sample() 
                  for agent_id in env.agents}
        
        obs, rewards, terminateds, truncateds, infos = env.step(actions)
        
        if terminateds.get("__all__", False):
            print(f"\n✅ Episode terminated at step {step + 1}")
            break
    
    print("\n" + "="*80 + "\n")


def test_quiet_mode():
    """Test environment with no verbose output."""
    print("\n" + "="*80)
    print("TEST 3: QUIET MODE (verbose=False, debug=False)")
    print("="*80 + "\n")
    
    env = HierarchicalTruckRoutingEnv(config={"verbose": False, "debug": False})
    
    # Run a few steps
    obs, info = env.reset()
    print(f"✅ Environment reset successfully. Agents: {len(obs)}")
    
    total_rewards = {agent: 0.0 for agent in env.agents}
    
    for step in range(10):
        # Get random actions
        actions = {agent_id: env.get_action_space(agent_id).sample() 
                  for agent_id in env.agents}
        
        obs, rewards, terminateds, truncateds, infos = env.step(actions)
        
        for agent_id, reward in rewards.items():
            total_rewards[agent_id] += reward
        
        if terminateds.get("__all__", False):
            print(f"✅ Episode terminated at step {step + 1}")
            break
    
    print(f"\n📊 Summary after {step + 1} steps:")
    print(f"  Total cumulative rewards: {sum(total_rewards.values()):.2f}")
    print(f"  Episode done: {terminateds.get('__all__', False)}")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("HIERARCHICAL TRUCK ROUTING ENVIRONMENT - VERBOSE MODE TESTS")
    print("="*80)
    
    # Test 1: Verbose mode only
    test_verbose_mode()
    
    # Test 2: Debug mode (includes verbose)
    test_debug_mode()
    
    # Test 3: Quiet mode
    test_quiet_mode()
    
    print("\n" + "="*80)
    print("✨ ALL TESTS COMPLETED SUCCESSFULLY!")
    print("="*80 + "\n")
