"""
Quick integration test to verify training loop works with batching and processing.
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


def test_training_integration():
    """Test that the full training loop works end-to-end."""
    print("\n" + "="*80)
    print("TRAINING INTEGRATION TEST")
    print("="*80)
    
    # Setup
    config = load_config('truck_env/config_files/config.yaml')
    config['environment']['num_trucks'] = 1
    config['environment']['num_stops'] = 3
    
    env = EventDrivenTruckEnv(config=config, verbose=False, enable_plotting=False)
    gnn_state_space = GNNStateSpace(
        num_trucks=1,
        num_stops=3,
        max_time=200.0,
        num_charging_nodes=env.num_charging_nodes
    )
    
    # Use CPU for faster testing
    device = torch.device('cpu')
    
    policy = TD3_ActionGNN(
        action_dim=env.action_space.n,
        max_action=1.0,
        fx_node_sizes={'ev': 13, 'cs': 5, 'tr': 2},
        fx_GNN_hidden_dim=32,
        mlp_hidden_dim=128,
        lr=3e-4,
        discrete_actions=env.action_space.n,
        actor_num_gcn_layers=2,
        critic_num_gcn_layers=2,
        min_charging_duration=0.5,
        max_charging_duration=10.0,
        device=device
    )
    
    replay_buffer = ReplayBuffer(max_size=10000)
    
    # Collect initial experience
    print("\nPhase 1: Collecting initial experience...")
    episodes_collected = 0
    total_timesteps = 0
    
    while len(replay_buffer) < 100:
        state_dict, info = env.reset()
        state = gnn_state_space.get_state_GNN(env)
        episode_reward = 0
        episode_steps = 0
        done = False
        truncated = False
        
        while not (done or truncated) and episode_steps < 50:
            # Select action using actor
            node_id, charging_dur, is_charging = policy.select_action(state, expl_noise=0.5)
            
            # Get action_idx for replay buffer storage
            action_idx = None
            for idx, (nid, is_chrg) in enumerate(state.action_to_node_map):
                if nid == node_id and is_chrg == is_charging:
                    action_idx = idx
                    break
            
            # Take step
            next_state_dict, reward, done, truncated, info = env.step(
                (node_id, charging_dur, is_charging)
            )
            
            next_state = gnn_state_space.get_state_GNN(env)
            
            # Store transition
            replay_buffer.add(
                state, 
                (action_idx, charging_dur), 
                next_state, 
                reward, 
                float(done)
            )
            
            state = next_state
            episode_reward += reward
            episode_steps += 1
            total_timesteps += 1
        
        episodes_collected += 1
        
        if episodes_collected % 5 == 0:
            print(f"  Episodes: {episodes_collected}, Buffer size: {len(replay_buffer)}, "
                  f"Last reward: {episode_reward:.2f}")
    
    print(f"✓ Collected {len(replay_buffer)} transitions from {episodes_collected} episodes")
    
    # Training phase
    print("\nPhase 2: Training with batched updates...")
    batch_size = 16
    num_updates = 20
    
    actor_losses = []
    critic_losses = []
    
    for i in range(num_updates):
        critic_loss, actor_loss = policy.train(replay_buffer, batch_size=batch_size)
        
        critic_losses.append(critic_loss)
        if actor_loss is not None:
            actor_losses.append(actor_loss)
        
        if (i + 1) % 5 == 0:
            avg_critic = np.mean(critic_losses[-5:])
            avg_actor = np.mean(actor_losses[-5:]) if len(actor_losses) >= 5 else None
            
            if avg_actor is not None:
                print(f"  Update {i+1}/{num_updates}: "
                      f"Critic Loss={avg_critic:.2f}, Actor Loss={avg_actor:.4f}")
            else:
                print(f"  Update {i+1}/{num_updates}: "
                      f"Critic Loss={avg_critic:.2f}, Actor Loss=Delayed")
    
    print(f"✓ Completed {num_updates} training updates")
    print(f"  Total actor updates: {len(actor_losses)}")
    print(f"  Avg critic loss: {np.mean(critic_losses):.2f}")
    if len(actor_losses) > 0:
        print(f"  Avg actor loss: {np.mean(actor_losses):.4f}")
    
    # Evaluation phase
    print("\nPhase 3: Testing trained policy...")
    eval_episodes = 3
    eval_rewards = []
    
    for ep in range(eval_episodes):
        state_dict, info = env.reset(seed=1000 + ep)
        state = gnn_state_space.get_state_GNN(env)
        episode_reward = 0
        episode_steps = 0
        done = False
        truncated = False
        
        while not (done or truncated) and episode_steps < 50:
            # Select action without exploration (greedy)
            node_id, charging_dur, is_charging = policy.select_action(state, expl_noise=0)
            
            next_state_dict, reward, done, truncated, info = env.step(
                (node_id, charging_dur, is_charging)
            )
            
            next_state = gnn_state_space.get_state_GNN(env)
            state = next_state
            episode_reward += reward
            episode_steps += 1
        
        eval_rewards.append(episode_reward)
        print(f"  Episode {ep+1}: Reward={episode_reward:.2f}, Steps={episode_steps}")
    
    print(f"✓ Evaluation complete: Avg reward={np.mean(eval_rewards):.2f}")
    
    # Verify everything works
    print("\n" + "="*80)
    print("INTEGRATION TEST RESULTS")
    print("="*80)
    
    checks = [
        ("Replay buffer populated", len(replay_buffer) >= 100),
        ("Training updates completed", len(critic_losses) == num_updates),
        ("Actor updated multiple times", len(actor_losses) > 0),
        ("Critic losses finite", all(np.isfinite(l) for l in critic_losses)),
        ("Actor losses finite", all(np.isfinite(l) for l in actor_losses)),
        ("Evaluation episodes completed", len(eval_rewards) == eval_episodes),
        ("Evaluation rewards finite", all(np.isfinite(r) for r in eval_rewards)),
    ]
    
    all_passed = True
    for check_name, passed in checks:
        status = "✓" if passed else "✗"
        print(f"{status} {check_name}")
        all_passed = all_passed and passed
    
    env.close()
    
    if all_passed:
        print("\n✓ ALL INTEGRATION CHECKS PASSED")
        print("  Batching, processing, and training loop all working correctly!")
        return True
    else:
        print("\n✗ SOME CHECKS FAILED")
        return False


if __name__ == "__main__":
    success = test_training_integration()
    sys.exit(0 if success else 1)
