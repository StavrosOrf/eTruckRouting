"""
Quick test script to verify training setup works correctly.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from truck_env.models.event_driven_env import EventDrivenTruckEnv
from truck_env.state.gnn_state_space import GNNStateSpace
from algo.TD3_actionGNN import TD3_ActionGNN
from algo.replay_buffer import ReplayBuffer
from truck_env.utils.utils import load_config

print("="*80)
print("TESTING TRAINING SETUP")
print("="*80)

# Load config
print("\n1. Loading configuration...")
config = load_config('truck_env/config_files/config.yaml')
print("   ✓ Config loaded successfully")

# Create environment
print("\n2. Creating environment...")
env = EventDrivenTruckEnv(
    config=config,
    verbose=False,
    enable_plotting=False,
    run_id="test_training"
)
print(f"   ✓ Environment created")
print(f"   - Action space: {env.action_space.n} discrete actions")
print(f"   - Num chargers: {env.num_charging_nodes}")

# Initialize GNN state space
print("\n3. Initializing GNN state space...")
gnn_state_space = GNNStateSpace(
    num_trucks=config['environment']['num_trucks'],
    num_stops=config['environment']['num_stops'],
    max_time=config['environment']['max_time'],
    num_charging_nodes=env.num_charging_nodes
)
print("   ✓ GNN state space initialized")

# Define feature sizes
fx_node_sizes = {
    'ev': 13,  # Truck features
    'cs': 5,   # Charger features
    'tr': 2,   # Delivery features
    'env': 1   # Environment features
}
print(f"   - Feature sizes: {fx_node_sizes}")

# Initialize TD3 agent
print("\n4. Initializing TD3 agent...")
action_dim = env.action_space.n
max_action = 1.0

policy = TD3_ActionGNN(
    action_dim=action_dim,
    max_action=max_action,
    fx_node_sizes=fx_node_sizes,
    discount=0.99,
    tau=0.005,
    policy_noise=0.2,
    noise_clip=0.5,
    policy_freq=2,
    fx_dim=8,
    fx_GNN_hidden_dim=32,
    mlp_hidden_dim=256,
    lr=3e-4,
    discrete_actions=action_dim,
    actor_num_gcn_layers=3,
    critic_num_gcn_layers=3
)
print("   ✓ TD3 agent initialized")
print(f"   - Device: {policy.device}")
print(f"   - Action dim: {action_dim}")

# Initialize replay buffer
print("\n5. Initializing replay buffer...")
replay_buffer = ReplayBuffer(max_size=10000)
print("   ✓ Replay buffer initialized")

# Test episode
print("\n6. Running test episode...")
obs, info = env.reset(seed=42)
gnn_state = gnn_state_space.get_state_GNN(env)
print(f"   ✓ Environment reset")
print(f"   - Initial GNN state type: {type(gnn_state)}")
print(f"   - Active truck: {env.active_truck_id}")

# Test action selection
print("\n7. Testing action selection...")
raw_action = policy.select_action(gnn_state, expl_noise=0.1)
# Clip action to valid range
action = raw_action % env.action_space.n  # Use modulo to wrap to valid range
print(f"   ✓ Action selected: {action} (raw: {raw_action})")
print(f"   - Action type: {type(action)}")

# Take a step
print("\n8. Testing environment step...")
next_obs, reward, done, truncated, info = env.step(action)
next_gnn_state = gnn_state_space.get_state_GNN(env)
print(f"   ✓ Step executed")
print(f"   - Reward: {reward:.2f}")
print(f"   - Done: {done}, Truncated: {truncated}")

# Add to replay buffer
print("\n9. Testing replay buffer...")
replay_buffer.add(gnn_state, action, next_gnn_state, reward, done)
print(f"   ✓ Transition added to buffer")
print(f"   - Buffer size: {len(replay_buffer)}")

# Collect more samples
print("\n10. Collecting more samples...")
for i in range(50):
    raw_action = policy.select_action(next_gnn_state, expl_noise=0.1)
    action = raw_action % env.action_space.n  # Clip to valid range
    obs, reward, done, truncated, info = env.step(action)
    new_state = gnn_state_space.get_state_GNN(env)
    replay_buffer.add(next_gnn_state, action, new_state, reward, done)
    next_gnn_state = new_state
    
    if done or truncated:
        obs, info = env.reset()
        next_gnn_state = gnn_state_space.get_state_GNN(env)

print(f"   ✓ Collected 51 transitions")
print(f"   - Buffer size: {len(replay_buffer)}")

# Test training
print("\n11. Testing training step...")
if len(replay_buffer) >= 32:
    critic_loss, actor_loss = policy.train(replay_buffer, batch_size=32)
    print(f"   ✓ Training step completed")
    print(f"   - Critic loss: {critic_loss:.4f}")
    print(f"   - Actor loss: {actor_loss if actor_loss else 'N/A (delayed update)'}")
else:
    print(f"   ⚠ Not enough samples for training (need 32, have {len(replay_buffer)})")

# Test save/load
print("\n12. Testing save/load...")
import tempfile
with tempfile.NamedTemporaryFile(delete=False) as tmp:
    tmp_path = tmp.name

policy.save(tmp_path)
print(f"   ✓ Model saved to {tmp_path}")

# Create new policy and load
policy2 = TD3_ActionGNN(
    action_dim=action_dim,
    max_action=max_action,
    fx_node_sizes=fx_node_sizes,
    discrete_actions=action_dim
)
policy2.load(tmp_path)
print(f"   ✓ Model loaded successfully")

# Cleanup
os.remove(tmp_path + "_actor")
os.remove(tmp_path + "_actor_optimizer")
os.remove(tmp_path + "_critic")
os.remove(tmp_path + "_critic_optimizer")

print("\n" + "="*80)
print("✅ ALL TESTS PASSED!")
print("="*80)
print("\nThe training setup is working correctly. You can now run:")
print("  python train.py --help")
print("  python train.py --no-wandb --max-timesteps 10000")
print("="*80)
