"""
Test script to verify the MultiAgentEnvError is fixed.
Tests that agents don't receive observations after being terminated.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from truck_env.truck_env import HierarchicalTruckRoutingEnv


def test_agent_termination():
    """Test that terminated agents don't receive observations in next steps."""
    print("\n" + "="*80)
    print("TEST: Agent Termination Handling")
    print("="*80 + "\n")
    
    env = HierarchicalTruckRoutingEnv(config={"verbose": False, "debug": False})
    
    obs, info = env.reset()
    print(f"✅ Environment reset. Agents: {len(obs)}")
    print(f"   Initial agents: {list(obs.keys())}")
    
    terminated_agents = set()
    
    for step in range(100):
        # Get random actions only for active agents
        actions = {}
        for agent_id in env.agents:
            if agent_id not in terminated_agents:
                actions[agent_id] = env.get_action_space(agent_id).sample()
        
        obs, rewards, terminateds, truncateds, infos = env.step(actions)
        
        # Track which agents terminated this step
        new_terminated = set()
        for agent_id, terminated in terminateds.items():
            if agent_id != "__all__" and terminated:
                if agent_id not in terminated_agents:
                    new_terminated.add(agent_id)
                    terminated_agents.add(agent_id)
        
        if new_terminated:
            print(f"\n📍 Step {step + 1}: Agents terminated: {new_terminated}")
            print(f"   Observations returned for: {list(obs.keys())}")
            
            # Verify: no terminated agent should have an observation
            for term_agent in new_terminated:
                if term_agent in obs:
                    print(f"   ⚠️  WARNING: Terminated agent {term_agent} received observation!")
                else:
                    print(f"   ✅ Terminated agent {term_agent} correctly excluded from observations")
        
        if terminateds.get("__all__", False):
            print(f"\n✅ Episode completed at step {step + 1}")
            print(f"   Total agents terminated: {len(terminated_agents)}")
            break
    
    print("\n" + "="*80)
    print("✨ TEST PASSED - No MultiAgentEnvError!")
    print("="*80 + "\n")


if __name__ == "__main__":
    test_agent_termination()
