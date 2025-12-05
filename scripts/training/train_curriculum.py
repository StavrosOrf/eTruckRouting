"""
Training loop for PPO-Variable with Curriculum Learning.

Trains a single agent on episodes with varying numbers of trucks and stops,
using curriculum strategies for robust generalization.
"""

import argparse
import copy
import json
import os
import sys
import numpy as np
import torch
import wandb
from datetime import datetime
from collections import defaultdict

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from EVRoutingEnv.models.curriculum_env import (
    CurriculumEnvWrapper,
    UniformRandomStrategy,
    StagedCurriculumStrategy,
    MixedCurriculumStrategy
)
from EVRoutingEnv.models.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.state.gnn_state_space import GNNStateSpace
from algo.PPO_VariableActionGNN import PPOVariableActionGNN
from EVRoutingEnv.utils.utils import load_config


def save_network_config(save_dir, config_dict):
    """Save neural network configuration to JSON file."""
    config_file = os.path.join(save_dir, "ppo_network_config.json")
    
    serializable_config = {}
    for key, value in config_dict.items():
        if isinstance(value, (str, int, float, bool, list, dict, type(None))):
            serializable_config[key] = value
        elif isinstance(value, np.ndarray):
            serializable_config[key] = value.tolist()
        else:
            serializable_config[key] = str(value)
    
    with open(config_file, 'w') as f:
        json.dump(serializable_config, f, indent=4)
    
    print(f"Network configuration saved to: {config_file}")


def parse_args():
    """Parse command line arguments for curriculum training."""
    parser = argparse.ArgumentParser(
        description='Train PPO-Variable with Curriculum Learning on Variable Problem Sizes'
    )
    
    # Environment parameters
    env_group = parser.add_argument_group('Environment')
    env_group.add_argument('--config', type=str, default='EVRoutingEnv/config_files/config.yaml',
                          help='Path to base environment config file')
    env_group.add_argument('--max-time', type=float, default=200.0,
                          help='Maximum simulation time in hours')
    env_group.add_argument('--enable-traffic', action='store_true',
                          help='Enable traffic simulation')
    
    # Curriculum parameters
    curr_group = parser.add_argument_group('Curriculum Learning')
    curr_group.add_argument('--curriculum-strategy', type=str, 
                           choices=['uniform', 'staged', 'mixed'], default='uniform',
                           help='Curriculum sampling strategy')
    curr_group.add_argument('--truck-range', type=int, nargs=2, default=[3, 8],
                           metavar=('MIN', 'MAX'),
                           help='Range of truck numbers (min max)')
    curr_group.add_argument('--stop-range', type=int, nargs=2, default=[3, 8],
                           metavar=('MIN', 'MAX'),
                           help='Range of stop numbers (min max)')
    curr_group.add_argument('--curriculum-config', type=str, default=None,
                           help='Path to detailed curriculum config JSON (overrides other curriculum args)')
    
    # Training parameters
    train_group = parser.add_argument_group('Training')
    train_group.add_argument('--seed', type=int, default=0,
                            help='Random seed for reproducibility')
    train_group.add_argument('--max-episodes', type=int, default=100_000,
                            help='Maximum number of training episodes')
    train_group.add_argument('--max-timesteps', type=int, default=2_000_000,
                            help='Maximum number of timesteps')
    train_group.add_argument('--eval-freq', type=int, default=1000,
                            help='Evaluation frequency (in timesteps)')
    train_group.add_argument('--eval-episodes', type=int, default=50,
                            help='Number of episodes per evaluation configuration')
    train_group.add_argument('--eval-configs', type=str, default='1,3,5,7,10',
                            help='Comma-separated list of truck counts for evaluation')
    
    # PPO hyperparameters
    ppo_group = parser.add_argument_group('PPO Algorithm')
    ppo_group.add_argument('--ppo-steps-per-update', type=int, default=512,
                           help='Number of timesteps collected before each PPO update')
    ppo_group.add_argument('--ppo-epochs', type=int, default=10,
                           help='Number of gradient epochs per PPO update')
    ppo_group.add_argument('--ppo-minibatch-size', type=int, default=256,
                           help='Minibatch size for PPO updates')
    ppo_group.add_argument('--gamma', type=float, default=0.99,
                           help='Discount factor')
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
    net_group.add_argument('--gnn-hidden-dim', type=int, default=64,
                          help='Hidden dimension for GNN layers')
    net_group.add_argument('--mlp-hidden-dim', type=int, default=256,
                          help='Hidden dimension for MLP layers')
    net_group.add_argument('--actor-gcn-layers', type=int, default=3,
                          help='Number of GCN layers in actor')
    net_group.add_argument('--lr', type=float, default=3e-4,
                          help='Learning rate')
    
    # Logging and output
    log_group = parser.add_argument_group('Logging')
    log_group.add_argument('--wandb-project', type=str, default='evpr-curriculum',
                          help='Wandb project name')
    log_group.add_argument('--wandb-entity', type=str, default='stavrosorf',
                          help='Wandb entity (username or team)')
    log_group.add_argument('--exp-name', type=str, default=None,
                          help='Experiment name (auto-generated if not provided)')
    log_group.add_argument('--group-name', type=str, default=None,
                          help='Wandb group name')
    log_group.add_argument('--no-wandb', action='store_true',
                          help='Disable wandb logging')
    log_group.add_argument('--verbose', action='store_true',
                          help='Enable verbose output')
    
    return parser.parse_args()


def create_curriculum_strategy(args):
    """Create curriculum strategy based on arguments."""
    if args.curriculum_config:
        # Load from JSON config file
        with open(args.curriculum_config, 'r') as f:
            config = json.load(f)
        
        strategy_type = config['strategy']
        if strategy_type == 'uniform':
            return UniformRandomStrategy(
                truck_range=tuple(config['truck_range']),
                stop_range=tuple(config['stop_range']),
                seed=args.seed
            )
        elif strategy_type == 'staged':
            return StagedCurriculumStrategy(
                stages=config['stages'],
                seed=args.seed
            )
        elif strategy_type == 'mixed':
            return MixedCurriculumStrategy(
                difficulty_levels=config['difficulty_levels'],
                seed=args.seed
            )
    else:
        # Use command-line arguments
        if args.curriculum_strategy == 'uniform':
            return UniformRandomStrategy(
                truck_range=tuple(args.truck_range),
                stop_range=tuple(args.stop_range),
                seed=args.seed
            )
        elif args.curriculum_strategy == 'staged':
            # Default staged curriculum
            stages = [
                {
                    'episodes': 500,
                    'truck_range': (args.truck_range[0], (args.truck_range[0] + args.truck_range[1]) // 2),
                    'stop_range': (args.stop_range[0], (args.stop_range[0] + args.stop_range[1]) // 2)
                },
                {
                    'episodes': 1000,
                    'truck_range': ((args.truck_range[0] + args.truck_range[1]) // 2, args.truck_range[1]),
                    'stop_range': ((args.stop_range[0] + args.stop_range[1]) // 2, args.stop_range[1])
                },
                {
                    'episodes': -1,  # Forever
                    'truck_range': tuple(args.truck_range),
                    'stop_range': tuple(args.stop_range)
                }
            ]
            return StagedCurriculumStrategy(stages=stages, seed=args.seed)
        elif args.curriculum_strategy == 'mixed':
            # Default mixed curriculum with 3 difficulty levels
            truck_span = args.truck_range[1] - args.truck_range[0]
            stop_span = args.stop_range[1] - args.stop_range[0]
            
            difficulty_levels = [
                {
                    'truck_range': (args.truck_range[0], args.truck_range[0] + truck_span // 3),
                    'stop_range': (args.stop_range[0], args.stop_range[0] + stop_span // 3),
                    'weight': 0.4
                },
                {
                    'truck_range': (args.truck_range[0] + truck_span // 3, args.truck_range[1] - truck_span // 3),
                    'stop_range': (args.stop_range[0] + stop_span // 3, args.stop_range[1] - stop_span // 3),
                    'weight': 0.4
                },
                {
                    'truck_range': (args.truck_range[1] - truck_span // 3, args.truck_range[1]),
                    'stop_range': (args.stop_range[1] - stop_span // 3, args.stop_range[1]),
                    'weight': 0.2
                }
            ]
            return MixedCurriculumStrategy(difficulty_levels=difficulty_levels, seed=args.seed)
    
    raise ValueError(f"Unknown curriculum strategy: {args.curriculum_strategy}")


def evaluate_policy_on_sizes(policy, base_config, gnn_state_spaces, eval_configs, 
                             num_episodes=5, seed=1000):
    """
    Evaluate policy on multiple fixed problem sizes.
    
    Args:
        policy: PPO policy to evaluate
        base_config: Base environment configuration
        gnn_state_spaces: Dict mapping (trucks, stops) -> GNNStateSpace
        eval_configs: List of (num_trucks, num_stops) tuples to evaluate on
        num_episodes: Number of episodes per configuration
        seed: Base random seed
    
    Returns:
        Dict mapping configuration to performance metrics
    """
    results = {}
    
    for num_trucks, num_stops in eval_configs:
        config = copy.deepcopy(base_config)
        config['environment']['num_trucks'] = num_trucks
        config['environment']['num_stops'] = num_stops
        config['environment']['max_episode_steps'] = int(num_trucks * num_stops * 7.5)
        
        # Create evaluation environment
        eval_env = EventDrivenTruckEnv(
            config=config,
            verbose=False,
            enable_plotting=False,
            run_id=f"eval_{num_trucks}t_{num_stops}s"
        )
        
        # Get or create GNN state space for this size
        size_key = (num_trucks, num_stops)
        if size_key not in gnn_state_spaces:
            gnn_state_spaces[size_key] = GNNStateSpace(
                num_trucks=num_trucks,
                num_stops=num_stops,
                max_time=config['environment']['max_time'],
                num_charging_nodes=eval_env.num_charging_nodes,
                device="cpu",
                verbose=False
            )
        gnn_state_space = gnn_state_spaces[size_key]
        
        # Run evaluation episodes
        episode_rewards = []
        episode_success = []
        episode_lengths = []
        episode_times = []
        
        for ep in range(num_episodes):
            obs, info = eval_env.reset(seed=seed + ep)
            episode_reward = 0
            episode_steps = 0
            done = False
            truncated = False
            
            while not (done or truncated):
                gnn_state = gnn_state_space.get_state_GNN(eval_env)
                raw_action = policy.select_action(gnn_state, expl_noise=0)
                action = policy.to_env_action(gnn_state, int(raw_action))
                
                obs, reward, done, truncated, info = eval_env.step(action)
                episode_reward += reward
                episode_steps += 1
            
            episode_rewards.append(episode_reward)
            episode_success.append(1.0 if info.get('all_complete', False) else 0.0)
            episode_lengths.append(episode_steps)
            episode_times.append(info.get('global_clock', 0.0))
        
        results[size_key] = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'success_rate': np.mean(episode_success),
            'mean_length': np.mean(episode_lengths),
            'mean_time': np.mean(episode_times),
        }
        
        eval_env.close()
    
    return results


def train(args):
    """Main curriculum training loop."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load base config
    base_config = load_config(args.config)
    if args.max_time is not None:
        base_config['environment']['max_time'] = args.max_time
    if args.enable_traffic:
        base_config['traffic']['enable_traffic'] = True
    
    # Generate experiment name
    if args.exp_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.exp_name = f"curriculum_{args.curriculum_strategy}_{timestamp}"
    
    if args.group_name is None:
        args.group_name = f"curriculum_t{args.truck_range[0]}-{args.truck_range[1]}_s{args.stop_range[0]}-{args.stop_range[1]}"
    
    # Initialize wandb
    if not args.no_wandb:
        _run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.exp_name,
            group=args.group_name,
            config=vars(args),
            save_code=True
        )
        _run.log_code(".")
    
    # Create curriculum strategy
    curriculum_strategy = create_curriculum_strategy(args)
    
    # Create curriculum environment
    train_env = CurriculumEnvWrapper(
        base_config=base_config,
        curriculum_strategy=curriculum_strategy,
        verbose=args.verbose,
        enable_plotting=False
    )
    
    # Create GNN state spaces for different sizes (will be populated as needed)
    gnn_state_spaces = {}
    
    # Get initial state to determine network architecture
    obs, info = train_env.reset(seed=args.seed)
    initial_trucks = info['curriculum']['num_trucks']
    initial_stops = info['curriculum']['num_stops']
    
    initial_state_space = GNNStateSpace(
        num_trucks=initial_trucks,
        num_stops=initial_stops,
        max_time=base_config['environment']['max_time'],
        num_charging_nodes=train_env.num_charging_nodes,
        device="cpu",
        verbose=False
    )
    gnn_state_spaces[(initial_trucks, initial_stops)] = initial_state_space
    
    gnn_state = initial_state_space.get_state_GNN(train_env.env)
    
    # Determine node feature dimensions from initial state
    node_feature_dims = {}
    for node_type in gnn_state.node_types:
        features = gnn_state[node_type].x
        if features.dim() == 1:
            feature_dim = int(features.numel())
        else:
            feature_dim = int(features.shape[-1])
        node_feature_dims[node_type] = feature_dim
    
    # Get action dimension (use max from truck range for compatibility)
    max_trucks = args.truck_range[1]
    max_stops = args.stop_range[1]
    dummy_config = copy.deepcopy(base_config)
    dummy_config['environment']['num_trucks'] = max_trucks
    dummy_config['environment']['num_stops'] = max_stops
    dummy_env = EventDrivenTruckEnv(config=dummy_config, verbose=False, enable_plotting=False)
    action_dim = dummy_env.action_space.n
    dummy_env.close()
    
    # Create PPO policy
    policy_kwargs = dict(
        action_dim=action_dim,
        node_feature_dims=node_feature_dims,
        hidden_dim=args.gnn_hidden_dim,
        num_layers=args.actor_gcn_layers,
        mlp_dim=args.mlp_hidden_dim,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.ppo_clip,
        value_coef=args.ppo_value_coef,
        entropy_coef=args.ppo_entropy_coef,
        max_grad_norm=args.ppo_max_grad_norm,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.ppo_minibatch_size,
        charge_durations=base_config["charging"]["charge_durations"],
    )
    
    policy = PPOVariableActionGNN(**policy_kwargs)
    
    # Save directories
    save_dir = os.path.join("saved_models", args.exp_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "ppo_curriculum")
    
    # Save configuration
    ppo_config = {
        "algo": "ppo-variable-curriculum",
        "curriculum_strategy": args.curriculum_strategy,
        "truck_range": args.truck_range,
        "stop_range": args.stop_range,
        **policy_kwargs,
        "seed": args.seed
    }
    save_network_config(save_dir, ppo_config)
    
    # Training setup
    total_timesteps = 0
    episode_num = 0
    episode_reward = 0.0
    episode_timesteps = 0
    best_eval_reward = None
    best_model_path = None
    
    # Parse evaluation configurations
    eval_truck_counts = [int(x.strip()) for x in args.eval_configs.split(',')]
    # Use middle of stop range for evaluation
    eval_stop_count = (args.stop_range[0] + args.stop_range[1]) // 2
    eval_configs = [(t, eval_stop_count) for t in eval_truck_counts]
    
    print(f"\n{'='*80}")
    print(f"Starting Curriculum Learning Training: {args.exp_name}")
    print(f"{'='*80}")
    print(f"Curriculum Strategy: {args.curriculum_strategy}")
    print(f"Truck Range: {args.truck_range[0]} - {args.truck_range[1]}")
    print(f"Stop Range: {args.stop_range[0]} - {args.stop_range[1]}")
    print(f"Max timesteps: {args.max_timesteps}")
    print(f"Steps per PPO update: {args.ppo_steps_per_update}")
    print(f"Evaluation configs: {eval_configs}")
    print(f"{'='*80}\n")
    
    # Training loop
    for t in range(args.max_timesteps):
        episode_timesteps += 1
        
        # Get current problem size and appropriate state space
        curr_trucks = info['curriculum']['num_trucks']
        curr_stops = info['curriculum']['num_stops']
        size_key = (curr_trucks, curr_stops)
        
        if size_key not in gnn_state_spaces:
            gnn_state_spaces[size_key] = GNNStateSpace(
                num_trucks=curr_trucks,
                num_stops=curr_stops,
                max_time=base_config['environment']['max_time'],
                num_charging_nodes=train_env.num_charging_nodes,
                device="cpu",
                verbose=False
            )
        
        gnn_state_space = gnn_state_spaces[size_key]
        gnn_state = gnn_state_space.get_state_GNN(train_env.env)
        
        # Select action
        action, logprob, value = policy.act(gnn_state)
        env_action = policy.to_env_action(gnn_state, action)
        
        # Take step
        next_obs, reward, done, truncated, info = train_env.step(env_action)
        done_flag = done or truncated
        
        # Get next state if episode continues
        if done_flag:
            next_gnn_state = None
        else:
            next_gnn_state = gnn_state_space.get_state_GNN(train_env.env)
        
        # Store transition
        policy.store_transition(gnn_state, action, logprob, reward, done_flag, value)
        
        # Update state
        if not done_flag:
            gnn_state = next_gnn_state
        
        episode_reward += reward
        total_timesteps += 1
        
        # PPO update
        if (t + 1) % args.ppo_steps_per_update == 0:
            last_value = policy.value(gnn_state) if not done_flag else 0.0
            update_stats = policy.update(last_value)
            
            if update_stats and not args.no_wandb:
                wandb.log({
                    'train/policy_loss': update_stats.get('policy_loss', 0.0),
                    'train/value_loss': update_stats.get('value_loss', 0.0),
                    'train/entropy': update_stats.get('entropy', 0.0),
                    'train/timestep': total_timesteps
                })
        
        # Episode end
        if done_flag:
            curr_trucks = info['curriculum']['num_trucks']
            curr_stops = info['curriculum']['num_stops']
            
            if not args.no_wandb:
                wandb.log({
                    'train/episode_reward': episode_reward,
                    'train/episode_length': episode_timesteps,
                    'train/episode': episode_num,
                    'train/success': 1.0 if info.get('all_complete', False) else 0.0,
                    'train/num_trucks': curr_trucks,
                    'train/num_stops': curr_stops,
                    'train/timestep': total_timesteps
                })
            
            if args.verbose or episode_num % 10 == 0:
                print(f"Episode {episode_num} ({curr_trucks}t, {curr_stops}s): "
                      f"Reward={episode_reward:.2f}, Steps={episode_timesteps}, "
                      f"Success={info.get('all_complete', False)}")
            
            # Reset for next episode
            obs, info = train_env.reset(seed=args.seed + episode_num + 1)
            episode_reward = 0.0
            episode_timesteps = 0
            episode_num += 1
            
            if episode_num >= args.max_episodes:
                print(f"Reached max episodes ({args.max_episodes}). Ending training.")
                break
        
        # Evaluation
        if (t + 1) % args.eval_freq == 0:
            print(f"\n{'='*80}")
            print(f"Evaluation at timestep {total_timesteps}")
            print(f"{'='*80}")
            
            eval_results = evaluate_policy_on_sizes(
                policy, base_config, gnn_state_spaces, eval_configs,
                num_episodes=args.eval_episodes, seed=args.seed + 10000
            )
            
            # Calculate aggregate metrics
            avg_reward = np.mean([r['mean_reward'] for r in eval_results.values()])
            avg_success = np.mean([r['success_rate'] for r in eval_results.values()])
            
            # Log results
            for size_key, metrics in eval_results.items():
                num_trucks, num_stops = size_key
                print(f"\n{num_trucks} trucks, {num_stops} stops:")
                print(f"  Reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
                print(f"  Success: {metrics['success_rate']*100:.1f}%")
                print(f"  Length: {metrics['mean_length']:.1f} steps")
                
                if not args.no_wandb:
                    wandb.log({
                        f'eval/{num_trucks}t_{num_stops}s/mean_reward': metrics['mean_reward'],
                        f'eval/{num_trucks}t_{num_stops}s/success_rate': metrics['success_rate'],
                        f'eval/{num_trucks}t_{num_stops}s/mean_length': metrics['mean_length'],
                        'eval/timestep': total_timesteps
                    })
            
            print(f"\nAggregate: Reward={avg_reward:.2f}, Success={avg_success*100:.1f}%")
            
            if not args.no_wandb:
                wandb.log({
                    'eval/aggregate_reward': avg_reward,
                    'eval/aggregate_success': avg_success,
                    'eval/timestep': total_timesteps
                })
            
            # Save best model based on aggregate success rate
            if best_eval_reward is None or avg_success > best_eval_reward:
                best_eval_reward = avg_success
                best_model_path = f"{save_path}_best"
                policy.save(best_model_path)
                print(f"🌟 New best model saved! Success rate: {best_eval_reward*100:.1f}%")
            
            print(f"{'='*80}\n")
            
            # Log curriculum statistics
            curr_stats = train_env.get_curriculum_stats()
            if not args.no_wandb:
                wandb.log({
                    'curriculum/total_episodes': curr_stats['total_episodes'],
                    'curriculum/unique_configs': len(curr_stats['performance_by_size']),
                    'eval/timestep': total_timesteps
                })
    
    # Final update if buffer has data
    if len(policy.buffer.rewards) > 0:
        last_value = 0.0
        policy.update(last_value)
    
    # Save final model
    final_model_path = f"{save_path}_final"
    policy.save(final_model_path)
    
    print(f"\n{'='*80}")
    print("Training completed!")
    print(f"Final model saved to: {final_model_path}")
    if best_model_path:
        print(f"Best model saved to: {best_model_path} (Success: {best_eval_reward*100:.1f}%)")
    print(f"Save directory: {save_dir}")
    
    # Print curriculum statistics
    curr_stats = train_env.get_curriculum_stats()
    print(f"\nCurriculum Statistics:")
    print(f"  Total episodes: {curr_stats['total_episodes']}")
    print(f"  Unique configurations: {len(curr_stats['performance_by_size'])}")
    print(f"{'='*80}\n")
    
    train_env.close()
    
    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    train(args)
