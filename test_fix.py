"""
Quick test to verify the MultiAgentEnvError fix
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from truck_env.truck_env import HierarchicalTruckRoutingEnv

def test_terminated_agents():
    """Test that terminated agents don't cause MultiAgentEnvError"""
    print("\n" + "="*80)
    print("Testing terminated agents handling...")
    print("="*80 + "\n")
    
    env = HierarchicalTruckRoutingEnv(config={"verbose": False, "debug": False})
    obs, info = env.reset()
    
    print(f"✅ Environment initialized with {len(env.agents)} agents")
    print(f"   Agents: {env.agents}\n")
    
    for step in range(500):
        # Get random actions
        actions = {agent_id: env.get_action_space(agent_id).sample() 
                  for agent_id in env.agents}
        
        obs, rewards, terminateds, truncateds, infos = env.step(actions)
        
        # Check that no terminated agents have observations
        for agent_id in env.agents:
            if terminateds.get(agent_id, False):
                if agent_id in obs:
                    print(f"❌ ERROR: Terminated agent {agent_id} has observation!")
                    return False
            else:
                if agent_id not in obs:
                    print(f"⚠️  WARNING: Active agent {agent_id} missing observation")
        
        # Report progress
        # if step % 10 == 0:
        # print SOC of each truck
        for i, truck in enumerate(env.trucks):
            print(f"Truck {truck['id']} SOC: {truck['current_battery']/truck['battery_capacity']*100:.2f}%")
                  
        done_agents = sum(1 for a in env.agents if terminateds.get(a, False))
        print(f"Step {step:3d}: {done_agents} agents done, {len(obs)} agents active")
        
        if terminateds.get("__all__", False):
            print(f"\n✅ Episode terminated at step {step + 1}")
            break
    
    print("\n" + "="*80)
    print("✅ TEST PASSED - No MultiAgentEnvError!")
    print("="*80 + "\n")
    return True

if __name__ == "__main__":
    success = test_terminated_agents()
    sys.exit(0 if success else 1)
