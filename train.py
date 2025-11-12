"""
Training loop for the TD3 Action-GNN agent using wandb to log results.
"""

import argparse
import os
import sys
import numpy as np
import torch
import wandb
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from truck_env.models.event_driven_env import EventDrivenTruckEnv
from truck_env.state.gnn_state_space import GNNStateSpace
from algo.TD3_actionGNN import TD3_ActionGNN
from algo.replay_buffer import ReplayBuffer
from truck_env.utils.utils import load_config


def parse_args():
    """Parse command line arguments for training hyperparameters."""
    parser = argparse.ArgumentParser(description='Train TD3 Action-GNN agent for Electric Truck Routing')
    
    # Environment parameters
    env_group = parser.add_argument_group('Environment')
    env_group.add_argument('--config', type=str, default='truck_env/config_files/config.yaml',
                          help='Path to environment config file')
    env_group.add_argument('--num-trucks', type=int, default=1,
                          help='Number of trucks (overrides config)')
    env_group.add_argument('--num-stops', type=int, default=None,
                          help='Number of delivery stops per truck (overrides config)')
    env_group.add_argument('--max-time', type=float, default=None,
                          help='Maximum simulation time in hours (overrides config)')
    env_group.add_argument('--enable-traffic', action='store_true',
                          help='Enable traffic simulation')
    
    # Training parameters
    train_group = parser.add_argument_group('Training')
    train_group.add_argument('--seed', type=int, default=0,
                            help='Random seed for reproducibility')
    train_group.add_argument('--max-episodes', type=int, default=1000,
                            help='Maximum number of training episodes')
    train_group.add_argument('--max-timesteps', type=int, default=1000000,
                            help='Maximum number of timesteps')
    train_group.add_argument('--eval-freq', type=int, default=100,
                            help='Evaluation frequency (in timesteps)')
    train_group.add_argument('--eval-episodes', type=int, default=10,
                            help='Number of episodes for evaluation')
    train_group.add_argument('--batch-size', type=int, default=64,
                            help='Batch size for training')
    train_group.add_argument('--start-timesteps', type=int, default=300,
                            help='Timesteps before training starts (random policy)')
    train_group.add_argument('--buffer-size', type=int, default=1000000,
                            help='Replay buffer size')
    
    # TD3 hyperparameters
    td3_group = parser.add_argument_group('TD3 Algorithm')
    td3_group.add_argument('--discount', type=float, default=0.99,
                          help='Discount factor (gamma)')
    td3_group.add_argument('--tau', type=float, default=0.005,
                          help='Target network update rate')
    td3_group.add_argument('--policy-noise', type=float, default=0.2,
                          help='Policy noise for target smoothing')
    td3_group.add_argument('--noise-clip', type=float, default=0.5,
                          help='Range to clip target policy noise')
    td3_group.add_argument('--policy-freq', type=int, default=2,
                          help='Frequency of delayed policy updates')
    td3_group.add_argument('--expl-noise', type=float, default=0.1,
                          help='Exploration noise (std of Gaussian)')
    
    # Network architecture
    net_group = parser.add_argument_group('Network Architecture')
    net_group.add_argument('--feature-dim', type=int, default=8,
                          help='Feature dimension for node embeddings')
    net_group.add_argument('--gnn-hidden-dim', type=int, default=32,
                          help='Hidden dimension for GNN layers')
    net_group.add_argument('--mlp-hidden-dim', type=int, default=256,
                          help='Hidden dimension for MLP layers in critic')
    net_group.add_argument('--actor-gcn-layers', type=int, default=3, choices=[3, 4, 5, 6],
                          help='Number of GCN layers in actor')
    net_group.add_argument('--critic-gcn-layers', type=int, default=3, choices=[3, 4, 5],
                          help='Number of GCN layers in critic')
    net_group.add_argument('--lr', type=float, default=3e-4,
                          help='Learning rate for both actor and critic')
    
    # Logging and output
    log_group = parser.add_argument_group('Logging')
    log_group.add_argument('--wandb-project', type=str, default='evpr-td3-gnn',
                          help='Wandb project name')
    log_group.add_argument('--wandb-entity', type=str, default= 'stavrosorf',
                          help='Wandb entity (username or team)')
    log_group.add_argument('--exp-name', type=str, default=None,
                          help='Experiment name (auto-generated if not provided)')
    log_group.add_argument('--no-wandb', action='store_true',
                          help='Disable wandb logging')
    log_group.add_argument('--verbose', action='store_true',
                          help='Enable verbose output')
    
    return parser.parse_args()


def evaluate_policy(env, policy, gnn_state_space, eval_episodes=10, seed=0):
    """Evaluate the current policy."""
    eval_rewards = []
    eval_success_rate = []
    
    for episode in range(eval_episodes):
        obs, info = env.reset(seed=seed + episode)
        episode_reward = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            # Get GNN state
            gnn_state = gnn_state_space.get_state_GNN(env)
            
            # Select action without exploration noise
            raw_action = policy.select_action(gnn_state, expl_noise=0)
            # Map node index to valid action (clip to action space)
            action = int(raw_action) % env.action_space.n
            
            # Take action
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
        
        eval_rewards.append(episode_reward)
        eval_success_rate.append(1.0 if info.get('all_complete', False) else 0.0)
    
    return {
        'mean_reward': np.mean(eval_rewards),
        'std_reward': np.std(eval_rewards),
        'success_rate': np.mean(eval_success_rate)
    }


def train(args):
    """Main training loop."""
    
    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load config and override with command line args
    config = load_config(args.config)
    if args.num_trucks is not None:
        config['environment']['num_trucks'] = args.num_trucks
    if args.num_stops is not None:
        config['environment']['num_stops'] = args.num_stops
    if args.max_time is not None:
        config['environment']['max_time'] = args.max_time
    if args.enable_traffic:
        config['traffic']['enable_traffic'] = True
    
    # Generate experiment name if not provided
    if args.exp_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.exp_name = f"TD3_GNN_{timestamp}"
        
    group_name = f"{config['environment']['num_trucks']}trucks_{config['environment']['num_stops']}stops"
    
    # Initialize wandb
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.exp_name,
            group=group_name,
            config=vars(args)
        )
    
    # Create environment
    env = EventDrivenTruckEnv(
        config=config,
        verbose=args.verbose,
        enable_plotting=False,
        run_id=args.exp_name
    )
    
    # Initialize GNN state space
    gnn_state_space = GNNStateSpace(
        num_trucks=config['environment']['num_trucks'],
        num_stops=config['environment']['num_stops'],
        max_time=config['environment']['max_time'],
        num_charging_nodes=env.num_charging_nodes
    )
    
    # Get action dimension and max action
    action_dim = env.action_space.n
    max_action = 1.0  # For discrete actions with softmax
    
    # Define feature sizes for different node types
    # Adjust based on your actual GNN state implementation
    fx_node_sizes = {
        'ev': 13,  # Truck features
        'cs': 5,   # Charger features
        'tr': 2,   # Delivery features
        'env': 1   # Environment features (if any)
    }
    
    # Initialize TD3 agent
    policy = TD3_ActionGNN(
        action_dim=action_dim,
        max_action=max_action,
        fx_node_sizes=fx_node_sizes,
        discount=args.discount,
        tau=args.tau,
        policy_noise=args.policy_noise,
        noise_clip=args.noise_clip,
        policy_freq=args.policy_freq,
        fx_dim=args.feature_dim,
        fx_GNN_hidden_dim=args.gnn_hidden_dim,
        mlp_hidden_dim=args.mlp_hidden_dim,
        lr=args.lr,
        discrete_actions=action_dim,
        actor_num_gcn_layers=args.actor_gcn_layers,
        critic_num_gcn_layers=args.critic_gcn_layers
    )
    
    # Initialize replay buffer
    replay_buffer = ReplayBuffer(max_size=args.buffer_size)
    
    # Create save directory with proper structure: {project}/saved_models/{run_id}/
    save_dir = os.path.join("saved_models", args.exp_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "model")
    
    # Track best model
    best_eval_reward = -float('inf')
    
    # Training loop
    total_timesteps = 0
    episode_num = 0
    episode_reward = 0
    episode_timesteps = 0
    
    obs, info = env.reset(seed=args.seed)
    gnn_state = gnn_state_space.get_state_GNN(env)
    
    print(f"\n{'='*80}")
    print(f"Starting Training: {args.exp_name}")
    print(f"{'='*80}")
    print(f"Environment: {config['environment']['num_trucks']} trucks, {config['environment']['num_stops']} stops")
    print(f"Max timesteps: {args.max_timesteps}")
    print(f"Replay buffer size: {args.buffer_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"{'='*80}\n")
    
    for t in range(args.max_timesteps):
        episode_timesteps += 1
        
        # Select action
        if t < args.start_timesteps:
            # Random action for initial exploration
            action = env.action_space.sample()
        else:
            # Select action with exploration noise
            raw_action = policy.select_action(gnn_state, expl_noise=args.expl_noise)
            # Map node index to valid action (clip to action space)
            action = int(raw_action) % env.action_space.n
        
        # Perform action
        next_obs, reward, done, truncated, info = env.step(action)
        next_gnn_state = gnn_state_space.get_state_GNN(env)
        
        # Store transition in replay buffer
        replay_buffer.add(gnn_state, action, next_gnn_state, reward, float(done))
        
        gnn_state = next_gnn_state
        episode_reward += reward
        total_timesteps += 1
        
        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            critic_loss, actor_loss = policy.train(replay_buffer, args.batch_size)
            
            # Log training metrics
            if not args.no_wandb and t % 100 == 0:  # Log every 100 steps
                log_dict = {
                    'train/critic_loss': critic_loss,
                    'train/timestep': total_timesteps
                }
                if actor_loss is not None:
                    log_dict['train/actor_loss'] = actor_loss
                wandb.log(log_dict)
        
        # Episode ended
        if done or truncated:
            # Log episode statistics
            if not args.no_wandb:
                wandb.log({
                    'train/episode_reward': episode_reward,
                    'train/episode_length': episode_timesteps,
                    'train/episode': episode_num,
                    'train/success': 1.0 if info.get('all_complete', False) else 0.0,
                    'train/timestep': total_timesteps
                })
            
            if args.verbose or episode_num % 10 == 0:
                
                if t >= args.start_timesteps:
                    print(f"Episode {episode_num}: Reward={episode_reward:.2f}, Steps={episode_timesteps}, "
                      f"Success={info.get('all_complete', False)}, Timestep={total_timesteps}")
                else:
                    print(f"[Collecting] Episode {episode_num}: Reward={episode_reward:.2f}, Steps={episode_timesteps}, "
                      f"Success={info.get('all_complete', False)}")
            
            # Reset environment
            obs, info = env.reset(seed=args.seed + episode_num + 1)
            gnn_state = gnn_state_space.get_state_GNN(env)
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
        
        # Evaluate policy
        if (t + 1) % args.eval_freq == 0 and t >= args.start_timesteps:
            eval_results = evaluate_policy(env, policy, gnn_state_space, 
                                         args.eval_episodes, args.seed + 1000)
            
            print(f"\n{'='*80}")
            print(f"Evaluation at timestep {total_timesteps}")
            print(f"Mean Reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
            print(f"Success Rate: {eval_results['success_rate']*100:.1f}%")
            
            # Save best model
            if eval_results['mean_reward'] > best_eval_reward:
                best_eval_reward = eval_results['mean_reward']
                policy.save(f"{save_path}_best")
                print(f"🌟 New best model saved! Reward: {best_eval_reward:.2f}")
            
            print(f"{'='*80}\n")
            
            if not args.no_wandb:
                wandb.log({
                    'eval/mean_reward': eval_results['mean_reward'],
                    'eval/std_reward': eval_results['std_reward'],
                    'eval/success_rate': eval_results['success_rate'],
                    'eval/best_reward': best_eval_reward,
                    'eval/timestep': total_timesteps
                })
    
    # Final save of the last model
    policy.save(f"{save_path}_final")
    print(f"\nTraining completed.")
    print(f"Final model saved to: {save_path}_final")
    print(f"Best model saved to: {save_path}_best (Reward: {best_eval_reward:.2f})")
    print(f"Save directory: {save_dir}")
    
    # Close environment and wandb
    env.close()
    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    train(args)