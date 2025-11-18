"""
Training loop for the TD3 Action-GNN agent using wandb to log results.
"""

import argparse
import copy
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
from algo.PPO_actionGNN import PPOActionGNN
from algo.replay_buffer import ReplayBuffer
from truck_env.utils.utils import load_config


def parse_args():
    """Parse command line arguments for training hyperparameters."""
    parser = argparse.ArgumentParser(description='Train TD3 Action-GNN agent for Electric Truck Routing')
    parser.add_argument('--algo', type=str, choices=['td3', 'ppo'], default='ppo',
                        help='Choose which RL algorithm to run')
    
    # Environment parameters
    env_group = parser.add_argument_group('Environment')
    env_group.add_argument('--config', type=str, default='truck_env/config_files/config.yaml',
                          help='Path to environment config file')
    env_group.add_argument('--num-trucks', type=int, default=1,
                          help='Number of trucks (overrides config)')
    env_group.add_argument('--num-stops', type=int, default=3,
                          help='Number of delivery stops per truck (overrides config)')
    env_group.add_argument('--max-time', type=float, default=200.0,
                          help='Maximum simulation time in hours (overrides config)')
    env_group.add_argument('--enable-traffic', action='store_true',
                          help='Enable traffic simulation')
    
    # Training parameters
    train_group = parser.add_argument_group('Training')
    train_group.add_argument('--seed', type=int, default=0,
                            help='Random seed for reproducibility')
    train_group.add_argument('--max-episodes', type=int, default=100_000,
                            help='Maximum number of training episodes')
    train_group.add_argument('--max-timesteps', type=int, default=1000000,
                            help='Maximum number of timesteps')
    train_group.add_argument('--max-episode-steps', type=int, default=200,
                            help='Maximum steps per episode (prevents infinite episodes)')
    train_group.add_argument('--eval-freq', type=int, default=500,
                            help='Evaluation frequency (in timesteps)')
    train_group.add_argument('--eval-episodes', type=int, default=10,
                            help='Number of episodes for evaluation')
    train_group.add_argument('--batch-size', type=int, default=256,
                            help='Batch size for training')
    train_group.add_argument('--start-timesteps', type=int, default=5000,
                            help='Timesteps before training starts (random policy)')
    train_group.add_argument('--buffer-size', type=int, default=500000,
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
    td3_group.add_argument('--target-action-temp', type=float, default=1.5,
                          help='Temperature for sampling target actions (>0)')

    # PPO hyperparameters
    ppo_group = parser.add_argument_group('PPO Algorithm')
    ppo_group.add_argument('--ppo-steps-per-update', type=int, default=128,
                           help='Number of timesteps collected before each PPO update')
    ppo_group.add_argument('--ppo-epochs', type=int, default=5,
                           help='Number of gradient epochs per PPO update')
    ppo_group.add_argument('--ppo-minibatch-size', type=int, default=256,
                           help='Minibatch size for PPO updates')
    ppo_group.add_argument('--gae-lambda', type=float, default=0.95,
                           help='GAE lambda parameter')
    ppo_group.add_argument('--ppo-clip', type=float, default=0.2,
                           help='Clipping coefficient for PPO')
    ppo_group.add_argument('--ppo-entropy-coef', type=float, default=0.01,
                           help='Entropy bonus coefficient')
    ppo_group.add_argument('--ppo-value-coef', type=float, default=0.5,
                           help='Value loss coefficient')
    ppo_group.add_argument('--ppo-max-grad-norm', type=float, default=0.5,
                           help='Gradient clipping value for PPO')
    
    # Network architecture
    net_group = parser.add_argument_group('Network Architecture')
    net_group.add_argument('--feature-dim', type=int, default=32,
                          help='Feature dimension for node embeddings')
    net_group.add_argument('--gnn-hidden-dim', type=int, default=64,
                          help='Hidden dimension for GNN layers')
    net_group.add_argument('--mlp-hidden-dim', type=int, default=256,
                          help='Hidden dimension for MLP layers in critic')
    net_group.add_argument('--actor-gcn-layers', type=int, default=3, choices=[3, 4, 5, 6],
                          help='Number of GCN layers in actor')
    net_group.add_argument('--critic-gcn-layers', type=int, default=3, choices=[3, 4, 5],
                          help='Number of GCN layers in critic')
    net_group.add_argument('--lr', type=float, default=3e-4,
                          help='Learning rate for both actor and critic')
    net_group.add_argument('--min-charging-duration', type=float, default=1,
                          help='Minimum charging duration in hours')
    net_group.add_argument('--max-charging-duration', type=float, default=10.0,
                          help='Maximum charging duration in hours')
    
    # Logging and output
    log_group = parser.add_argument_group('Logging')
    log_group.add_argument('--wandb-project', type=str, default='evpr-td3-gnn',
                          help='Wandb project name')
    log_group.add_argument('--wandb-entity', type=str, default= 'stavrosorf',
                          help='Wandb entity (username or team)')
    log_group.add_argument('--exp-name', type=str, default=None,
                          help='Experiment name (auto-generated if not provided)')
    log_group.add_argument('--group-name', type=str, default=None,
                          help='Wandb group name for organizing related experiments')
    log_group.add_argument('--no-wandb', action='store_true',
                          help='Disable wandb logging')
    log_group.add_argument('--diag-log-freq', type=int, default=1000,
                          help='How often (timesteps) to log action/Q diagnostics')
    log_group.add_argument('--verbose', action='store_true',
                          help='Enable verbose output')
    
    return parser.parse_args()


def compute_action_mask(env):
    """Return boolean mask (True=feasible) for the discrete env action space."""
    num_actions = env.action_space.n
    mask = np.zeros(num_actions, dtype=bool)

    if env.active_truck_id is None:
        return mask

    truck = env.trucks[env.active_truck_id]
    if truck.failed or truck.is_complete:
        return mask

    current_node = int(truck.current_node)
    battery = float(truck.current_battery)

    # Charger navigation actions (0 .. num_charging_nodes-1)
    for idx, charger_node in enumerate(env.charging_nodes):
        if idx >= env.num_charging_nodes:
            break
        charger_node = int(charger_node)
        energy = env.transport_graph.get_path_energy(current_node, charger_node)
        feasible = (not np.isinf(energy)) and (energy <= battery)
        mask[idx] = feasible

    # Next delivery action
    next_delivery = truck.get_next_delivery_target()
    delivery_idx = env.num_charging_nodes
    if next_delivery is not None:
        energy = env.transport_graph.get_path_energy(current_node, int(next_delivery))
        mask[delivery_idx] = (not np.isinf(energy)) and (energy <= battery)
    else:
        mask[delivery_idx] = False

    # Charging actions
    can_charge_here = (current_node in env.charging_nodes) and (truck.get_battery_percentage() < 95.0)
    charge_start = env.num_navigation_actions
    for i in range(env.num_charge_actions):
        mask[charge_start + i] = can_charge_here

    if not mask.any():
        mask[delivery_idx] = True
    return mask


def evaluate_policy(env, policy, gnn_state_space, eval_episodes=10, seed=0, max_steps=200):
    """Evaluate the current policy and collect detailed metrics."""
    eval_rewards = []
    eval_success_rate = []
    eval_episode_lengths = []
    eval_total_charging_time = []
    eval_num_charging_sessions = []
    eval_waiting_time = []
    eval_total_distance = []
    eval_episode_time = []
    
    for episode in range(eval_episodes):
        obs, info = env.reset(seed=seed + episode)
        episode_reward = 0
        episode_steps = 0
        done = False
        truncated = False
        
        while not (done or truncated) and episode_steps < max_steps:
            # Get GNN state from the CORRECT environment (eval_env, not train env)
            gnn_state = gnn_state_space.get_state_GNN(env)
            action_mask = compute_action_mask(env)
            
            # Select action without exploration noise (greedy)
            raw_action = policy.select_action(gnn_state, expl_noise=0, action_mask=action_mask)
            if isinstance(raw_action, tuple):
                action = raw_action
            else:
                # Map node index to valid action (clip to action space)
                action = int(raw_action) % env.action_space.n
            
            # Take action
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            episode_steps += 1
        
        # Collect episode metrics from environment info
        eval_rewards.append(episode_reward)
        eval_success_rate.append(1.0 if info.get('all_complete', False) else 0.0)
        eval_episode_lengths.append(episode_steps)
        
        # Extract metrics from truck states in info (these are episode-specific)
        total_charging_time = 0.0
        num_charging_sessions = 0
        total_waiting_time = 0.0
        total_distance = 0.0
        
        for truck_info in info.get('trucks', []):
            total_charging_time += truck_info.get('total_charging_time', 0.0)
            num_charging_sessions += truck_info.get('num_charging_sessions', 0)
            total_waiting_time += truck_info.get('waiting_time', 0.0)
            total_distance += truck_info.get('total_distance', 0.0)
        
        eval_total_charging_time.append(total_charging_time)
        eval_num_charging_sessions.append(num_charging_sessions)
        eval_waiting_time.append(total_waiting_time)
        eval_total_distance.append(total_distance)
        
        # Get episode time from global clock
        eval_episode_time.append(info.get('global_clock', 0.0))
    
    return {
        'mean_reward': np.mean(eval_rewards),
        'std_reward': np.std(eval_rewards),
        'success_rate': np.mean(eval_success_rate),
        'mean_episode_length': np.mean(eval_episode_lengths),
        'mean_charging_time': np.mean(eval_total_charging_time),
        'mean_num_charging_sessions': np.mean(eval_num_charging_sessions),
        'mean_waiting_time': np.mean(eval_waiting_time),
        'mean_total_distance': np.mean(eval_total_distance),
        'mean_episode_time': np.mean(eval_episode_time) if eval_episode_time else 0.0
    }


def train_td3(args):
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
        
    # Use user-provided group name or default to environment configuration
    if args.group_name is not None:
        group_name = args.group_name
    else:
        group_name = f"{config['environment']['num_trucks']}trucks_{config['environment']['num_stops']}stops"
    
    # Initialize wandb
    if not args.no_wandb:
        _run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.exp_name,
            group=group_name,
            config=vars(args),
            save_code=True
        )
        #log code files
        _run.log_code(".")
    
    # Create training and evaluation environments (evaluation env stays isolated)
    env = EventDrivenTruckEnv(
        config=config,
        verbose=False,
        enable_plotting=False,
        run_id=args.exp_name
    )
    eval_env = EventDrivenTruckEnv(
        config=copy.deepcopy(config),
        verbose=False,
        enable_plotting=False,
        run_id=f"{args.exp_name}_eval"
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
        'ev': 10,  # Truck features (reduced from 13)
        'cs': 5,   # Charger features (increased from 4, now includes time_for_queue_to_empty)
        'tr': 3,   # Delivery features (node_type, node_id, delivery_sequence_index)
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
        critic_num_gcn_layers=args.critic_gcn_layers,
        min_charging_duration=args.min_charging_duration,
        max_charging_duration=args.max_charging_duration,
        target_action_temperature=args.target_action_temp
    )
    
    # Initialize replay buffer
    replay_buffer = ReplayBuffer(max_size=args.buffer_size)
    
    # Create save directory with proper structure: {project}/saved_models/{run_id}/
    save_dir = os.path.join("saved_models", args.exp_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "model")
    
    # Track best model
    best_eval_reward = None
    best_model_path = None
    
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
    print(f"Max steps per episode: {args.max_episode_steps}")
    print(f"Replay buffer size: {args.buffer_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"{'='*80}\n")
    
    for t in range(args.max_timesteps):
        episode_timesteps += 1
        
                
        action = policy.select_action(gnn_state, expl_noise=args.expl_noise)
        # action is now a tuple: (node_id, charging_duration, is_charging)
        
        # Perform action (env.step handles both integer and tuple formats)
        # print(f"Taking action: {action}, step: {t}")
        next_obs, reward, done, truncated, info = env.step(action)
        next_gnn_state = gnn_state_space.get_state_GNN(env)
        
        # Store transition in replay buffer
        # For tuple actions, replay buffer expects (action_idx, charging_duration)
        if isinstance(action, tuple):
            node_id, charging_duration, is_charging = action
            # We need to convert back to action_idx for storage
            # The replay buffer and training use action indices internally
            # Get action_idx from the state's action_to_node_map
            if hasattr(gnn_state, 'action_to_node_map'):
                # Find which action index corresponds to this (node_id, is_charging) pair
                action_idx = None
                for idx, (mapped_node, mapped_is_charging) in enumerate(gnn_state.action_to_node_map):
                    if mapped_node == node_id and mapped_is_charging == is_charging:
                        action_idx = idx
                        break
                
                if action_idx is None:
                    # Fallback: shouldn't happen but handle gracefully
                    action_idx = 0
                    if args.verbose:
                        print(f"Warning: Could not find action_idx for (node={node_id}, is_charging={is_charging})")
                
                replay_buffer.add(gnn_state, (action_idx, charging_duration), next_gnn_state, reward, float(done))
            else:
                # Fallback for states without action_to_node_map
                replay_buffer.add(gnn_state, (0, charging_duration), next_gnn_state, reward, float(done))
        else:
            # Legacy integer action
            replay_buffer.add(gnn_state, action, next_gnn_state, reward, float(done))
        
        gnn_state = next_gnn_state
        episode_reward += reward
        total_timesteps += 1
        
        # Check if episode reached maximum steps
        if episode_timesteps >= args.max_episode_steps:
            truncated = True
            if args.verbose:
                print(f"Episode {episode_num} truncated at {episode_timesteps} steps (max={args.max_episode_steps})")
        
        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            critic_loss, actor_loss = policy.train(replay_buffer, args.batch_size)
            
            # Log training metrics
            if not args.no_wandb:  # Log every 100 steps
                log_dict = {
                    'train/critic_loss': critic_loss,
                    'train/timestep': total_timesteps
                }
                if actor_loss is not None:
                    log_dict['train/actor_loss'] = actor_loss
                    
                wandb.log(log_dict)

            if ((not args.no_wandb) or args.verbose) and (total_timesteps % args.diag_log_freq == 0):
                diag_metrics = policy.get_action_diagnostics(gnn_state)
                diag_metrics['train/timestep'] = total_timesteps
                if not args.no_wandb:
                    wandb.log(diag_metrics)
                if args.verbose:
                    print(f"Diag@{total_timesteps}: top1={diag_metrics['diag/top1_logit']:.3f}, "
                          f"gap={diag_metrics['diag/top_gap']:.3f}, entropy={diag_metrics['diag/action_entropy']:.3f}, "
                          f"Qbest={diag_metrics['diag/q_best']:.3f}")
        
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
            
            # if args.verbose or episode_num % 10 == 0:
                
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
            eval_results = evaluate_policy(eval_env, policy, gnn_state_space, 
                                         args.eval_episodes, args.seed + 1000, args.max_episode_steps)
            
            print(f"\n{'='*80}")
            print(f"Evaluation at timestep {total_timesteps}")
            print(f"Mean Reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
            print(f"Success Rate: {eval_results['success_rate']*100:.1f}%")
            print(f"Episode Length: {eval_results['mean_episode_length']:.1f} steps")
            print(f"Charging Time: {eval_results['mean_charging_time']:.2f} hours")
            print(f"Charging Sessions: {eval_results['mean_num_charging_sessions']:.1f}")
            print(f"Waiting Time: {eval_results['mean_waiting_time']:.2f} hours")
            print(f"Distance Traveled: {eval_results['mean_total_distance']:.1f} km")
            if eval_results['mean_episode_time'] > 0:
                print(f"Episode Time: {eval_results['mean_episode_time']:.2f} hours")
            
            # Save best model
            if best_eval_reward is None or eval_results['mean_reward'] > best_eval_reward:
                best_eval_reward = eval_results['mean_reward']
                best_model_path = f"{save_path}_best"
                policy.save(best_model_path)
                print(f"🌟 New best model saved! Reward: {best_eval_reward:.2f}")
            
            print(f"{'='*80}\n")
            
            if not args.no_wandb:
                wandb.log({
                    'eval/mean_reward': eval_results['mean_reward'],
                    'eval/std_reward': eval_results['std_reward'],
                    'eval/success_rate': eval_results['success_rate'],
                    'eval/best_reward': best_eval_reward,
                    'eval/episode_length': eval_results['mean_episode_length'],
                    'eval/charging_time': eval_results['mean_charging_time'],
                    'eval/num_charging_sessions': eval_results['mean_num_charging_sessions'],
                    'eval/waiting_time': eval_results['mean_waiting_time'],
                    'eval/total_distance': eval_results['mean_total_distance'],
                    'eval/episode_time': eval_results['mean_episode_time'],
                    'eval/timestep': total_timesteps
                })
        if episode_num >= args.max_episodes:
            if args.verbose:
                print(f"Reached max episodes ({args.max_episodes}). Ending training loop.")
            break
    
    # Final save of the last model
    final_model_path = f"{save_path}_final"
    policy.save(final_model_path)
    print(f"\nTraining completed.")
    print(f"Final model saved to: {final_model_path}")
    if best_model_path is not None and best_eval_reward is not None:
        print(f"Best model saved to: {best_model_path} (Reward: {best_eval_reward:.2f})")
    else:
        print("Best model not saved (evaluation not run or no improvement).")
    print(f"Save directory: {save_dir}")
    
    # Close environment and wandb
    env.close()
    eval_env.close()
    if not args.no_wandb:
        wandb.finish()


def train_ppo(args):
    """PPO training loop using the GNN state representation."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = load_config(args.config)
    if args.num_trucks is not None:
        config['environment']['num_trucks'] = args.num_trucks
    if args.num_stops is not None:
        config['environment']['num_stops'] = args.num_stops
    if args.max_time is not None:
        config['environment']['max_time'] = args.max_time
    if args.enable_traffic:
        config['traffic']['enable_traffic'] = True

    if args.exp_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.exp_name = f"PPO_GNN_{timestamp}"

    # Use user-provided group name or default to environment configuration
    if args.group_name is not None:
        group_name = args.group_name
    else:
        group_name = f"{config['environment']['num_trucks']}trucks_{config['environment']['num_stops']}stops"

    if not args.no_wandb:
        _run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.exp_name,
            group=group_name,
            config=vars(args),
            save_code=True
        )
        _run.log_code(".")

    env = EventDrivenTruckEnv(
        config=config,
        verbose=False,
        enable_plotting=False,
        run_id=args.exp_name
    )
    eval_env = EventDrivenTruckEnv(
        config=copy.deepcopy(config),
        verbose=False,
        enable_plotting=False,
        run_id=f"{args.exp_name}_eval"
    )

    gnn_state_space = GNNStateSpace(
        num_trucks=config['environment']['num_trucks'],
        num_stops=config['environment']['num_stops'],
        max_time=config['environment']['max_time'],
        num_charging_nodes=env.num_charging_nodes
    )

    action_dim = env.action_space.n
    node_feature_dims = {
        'truck': 10,  # Reduced from 13
        'delivery': 3,  # node_type, node_id, delivery_sequence_index
        'charger': 5  # Increased from 4, now includes time_for_queue_to_empty
    }

    policy = PPOActionGNN(
        action_dim=action_dim,
        node_feature_dims=node_feature_dims,
        hidden_dim=args.gnn_hidden_dim,
        num_layers=args.actor_gcn_layers,
        mlp_dim=args.mlp_hidden_dim,
        lr=args.lr,
        gamma=args.discount,
        gae_lambda=args.gae_lambda,
        clip_coef=args.ppo_clip,
        value_coef=args.ppo_value_coef,
        entropy_coef=args.ppo_entropy_coef,
        max_grad_norm=args.ppo_max_grad_norm,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.ppo_minibatch_size
    )

    save_dir = os.path.join("saved_models", args.exp_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "ppo_model")
    best_eval_reward = None
    best_model_path = None

    total_timesteps = 0
    episode_num = 0
    episode_reward = 0.0
    episode_timesteps = 0

    obs, info = env.reset(seed=args.seed)
    gnn_state = gnn_state_space.get_state_GNN(env)

    print(f"\n{'='*80}")
    print(f"Starting PPO Training: {args.exp_name}")
    print(f"{'='*80}")
    print(f"Environment: {config['environment']['num_trucks']} trucks, {config['environment']['num_stops']} stops")
    print(f"Max timesteps: {args.max_timesteps}")
    print(f"Steps per PPO update: {args.ppo_steps_per_update}")
    print(f"Minibatch size: {args.ppo_minibatch_size}")
    print(f"{'='*80}\n")

    for t in range(args.max_timesteps):
        episode_timesteps += 1

        mask_np = compute_action_mask(env)
        mask_tensor = torch.tensor(mask_np, dtype=torch.bool)
        action, logprob, value = policy.act(gnn_state, action_mask=mask_tensor)
        next_obs, reward, done, truncated, info = env.step(action)
        next_gnn_state = gnn_state_space.get_state_GNN(env)

        done_flag = done or truncated
        if episode_timesteps >= args.max_episode_steps:
            done_flag = True
            truncated = True

        policy.store_transition(gnn_state, action, logprob, reward, done_flag, value, mask_tensor)

        gnn_state = next_gnn_state
        episode_reward += reward
        total_timesteps += 1

        if (t + 1) % args.ppo_steps_per_update == 0:
            last_value = policy.value(gnn_state)
            update_stats = policy.update(last_value)
            if update_stats and not args.no_wandb:
                wandb.log({
                    'train/policy_loss': update_stats.get('policy_loss', 0.0),
                    'train/value_loss': update_stats.get('value_loss', 0.0),
                    'train/entropy': update_stats.get('entropy', 0.0),
                    'train/timestep': total_timesteps
                })

        if done_flag:
            if not args.no_wandb:
                wandb.log({
                    'train/episode_reward': episode_reward,
                    'train/episode_length': episode_timesteps,
                    'train/episode': episode_num,
                    'train/success': 1.0 if info.get('all_complete', False) else 0.0,
                    'train/timestep': total_timesteps
                })

            print(f"PPO Episode {episode_num}: Reward={episode_reward:.2f}, "
                  f"Steps={episode_timesteps}, Success={info.get('all_complete', False)}")

            obs, info = env.reset(seed=args.seed + episode_num + 1)
            gnn_state = gnn_state_space.get_state_GNN(env)
            episode_reward = 0.0
            episode_timesteps = 0
            episode_num += 1

            if episode_num >= args.max_episodes:
                if args.verbose:
                    print(f"Reached max episodes ({args.max_episodes}). Ending PPO training loop.")
                break

        if (t + 1) % args.eval_freq == 0:
            eval_results = evaluate_policy(eval_env, policy, gnn_state_space,
                                           args.eval_episodes, args.seed + 1000, args.max_episode_steps)

            print(f"\n{'='*80}")
            print(f"PPO Evaluation at timestep {total_timesteps}")
            print(f"Mean Reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
            print(f"Success Rate: {eval_results['success_rate']*100:.1f}%")
            print(f"Episode Length: {eval_results['mean_episode_length']:.1f} steps")
            print(f"Charging Time: {eval_results['mean_charging_time']:.2f} hours")
            print(f"Charging Sessions: {eval_results['mean_num_charging_sessions']:.1f}")
            print(f"Waiting Time: {eval_results['mean_waiting_time']:.2f} hours")
            print(f"Distance Traveled: {eval_results['mean_total_distance']:.1f} km")
            if eval_results['mean_episode_time'] > 0:
                print(f"Episode Time: {eval_results['mean_episode_time']:.2f} hours")

            if best_eval_reward is None or eval_results['mean_reward'] > best_eval_reward:
                best_eval_reward = eval_results['mean_reward']
                best_model_path = f"{save_path}_best"
                policy.save(best_model_path)
                print(f"🌟 New best PPO model saved! Reward: {best_eval_reward:.2f}")

            print(f"{'='*80}\n")

            if not args.no_wandb:
                wandb.log({
                    'eval/mean_reward': eval_results['mean_reward'],
                    'eval/std_reward': eval_results['std_reward'],
                    'eval/success_rate': eval_results['success_rate'],
                    'eval/best_reward': best_eval_reward,
                    'eval/episode_length': eval_results['mean_episode_length'],
                    'eval/charging_time': eval_results['mean_charging_time'],
                    'eval/num_charging_sessions': eval_results['mean_num_charging_sessions'],
                    'eval/waiting_time': eval_results['mean_waiting_time'],
                    'eval/total_distance': eval_results['mean_total_distance'],
                    'eval/episode_time': eval_results['mean_episode_time'],
                    'eval/timestep': total_timesteps
                })

    if len(policy.buffer.rewards) > 0:
        last_value = policy.value(gnn_state)
        policy.update(last_value)

    final_model_path = f"{save_path}_final"
    policy.save(final_model_path)
    print(f"\nPPO training completed.")
    print(f"Final PPO model saved to: {final_model_path}")
    if best_model_path is not None and best_eval_reward is not None:
        print(f"Best PPO model saved to: {best_model_path} (Reward: {best_eval_reward:.2f})")
    else:
        print("Best PPO model not saved (evaluation not run or no improvement).")
    print(f"Save directory: {save_dir}")

    env.close()
    eval_env.close()
    if not args.no_wandb:
        wandb.finish()


def train(args):
    if args.algo == 'td3':
        train_td3(args)
    else:
        train_ppo(args)


if __name__ == "__main__":
    args = parse_args()
    train(args)
