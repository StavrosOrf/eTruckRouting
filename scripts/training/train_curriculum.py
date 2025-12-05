#!/usr/bin/env python3
"""
Simplified Curriculum Learning Training for PPO-Variable.
Trains on variable-sized problems using a curriculum schedule defined in JSON.
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
from pathlib import Path

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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Curriculum Learning Training')
    
    # Required
    parser.add_argument('--curriculum-config', type=str, required=True,
                       help='Path to curriculum config JSON file')
    parser.add_argument('--exp-name', type=str, required=True,
                       help='Experiment name')
    
    # Training
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed')
    parser.add_argument('--max-timesteps', type=int, default=20_000_000,
                       help='Maximum training timesteps')
    parser.add_argument('--eval-freq', type=int, default=10_000,
                       help='Evaluation frequency (timesteps)')
    parser.add_argument('--eval-episodes', type=int, default=20,
                       help='Episodes per evaluation config')
    
    # PPO
    parser.add_argument('--ppo-steps', type=int, default=1024,
                       help='Steps before PPO update')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate')
    
    # Environment
    parser.add_argument('--config', type=str, default='EVRoutingEnv/config_files/config.yaml',
                       help='Base environment config')
    
    # Logging
    parser.add_argument('--wandb-project', type=str, default='evpr-curriculum',
                       help='Wandb project name')
    parser.add_argument('--no-wandb', action='store_true',
                       help='Disable wandb')
    
    return parser.parse_args()


def load_curriculum_strategy(config_path, seed):
    """Load curriculum strategy from JSON config."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    strategy_type = config['strategy']
    
    if strategy_type == 'uniform':
        return UniformRandomStrategy(
            truck_range=tuple(config['truck_range']),
            stop_range=tuple(config['stop_range']),
            seed=seed
        ), config
    elif strategy_type == 'staged':
        return StagedCurriculumStrategy(
            stages=config['stages'],
            seed=seed
        ), config
    elif strategy_type == 'mixed':
        return MixedCurriculumStrategy(
            difficulty_levels=config['difficulty_levels'],
            seed=seed
        ), config
    else:
        raise ValueError(f"Unknown strategy: {strategy_type}")


def generate_eval_configs(curriculum_config, num_samples=30):
    """
    Generate robust evaluation configurations by sampling from different difficulty levels.
    Returns list of (num_trucks, num_stops) tuples.
    """
    configs = []
    strategy = curriculum_config['strategy']
    np.random.seed(42)  # Fixed seed for reproducible eval
    
    if strategy == 'uniform':
        truck_range = curriculum_config['truck_range']
        stop_range = curriculum_config['stop_range']
        
        # Sample uniformly across the range
        for _ in range(num_samples):
            trucks = np.random.randint(truck_range[0], truck_range[1] + 1)
            stops = np.random.randint(stop_range[0], stop_range[1] + 1)
            configs.append((trucks, stops))
    
    elif strategy == 'staged':
        # Sample from each stage proportionally
        stages = curriculum_config['stages']
        valid_stages = [s for s in stages if s['episodes'] != -1]
        
        samples_per_stage = num_samples // len(valid_stages)
        for stage in valid_stages:
            for _ in range(samples_per_stage):
                trucks = np.random.randint(stage['truck_range'][0], stage['truck_range'][1] + 1)
                stops = np.random.randint(stage['stop_range'][0], stage['stop_range'][1] + 1)
                configs.append((trucks, stops))
        
        # Add samples from final stage
        final_stage = stages[-1]
        for _ in range(num_samples - len(configs)):
            trucks = np.random.randint(final_stage['truck_range'][0], final_stage['truck_range'][1] + 1)
            stops = np.random.randint(final_stage['stop_range'][0], final_stage['stop_range'][1] + 1)
            configs.append((trucks, stops))
    
    elif strategy == 'mixed':
        # Sample from each difficulty level according to weights
        levels = curriculum_config['difficulty_levels']
        weights = np.array([level['weight'] for level in levels])
        weights = weights / weights.sum()
        
        level_samples = np.random.multinomial(num_samples, weights)
        
        for level, n_samples in zip(levels, level_samples):
            for _ in range(n_samples):
                trucks = np.random.randint(level['truck_range'][0], level['truck_range'][1] + 1)
                stops = np.random.randint(level['stop_range'][0], level['stop_range'][1] + 1)
                configs.append((trucks, stops))
    
    # Remove duplicates while preserving order
    seen = set()
    unique_configs = []
    for config in configs:
        if config not in seen:
            seen.add(config)
            unique_configs.append(config)
    
    return unique_configs


def evaluate_policy(policy, base_config, eval_configs, num_episodes_per_config, seed):
    """
    Evaluate policy on multiple configurations.
    Returns dict with results per config and aggregate metrics.
    """
    results = {}
    all_rewards = []
    all_success = []
    
    for num_trucks, num_stops in eval_configs:
        config = copy.deepcopy(base_config)
        config['environment']['num_trucks'] = num_trucks
        config['environment']['num_stops'] = num_stops
        config['environment']['max_episode_steps'] = int(num_trucks * num_stops * 7.5)
        
        eval_env = EventDrivenTruckEnv(config=config, verbose=False, enable_plotting=False)
        
        state_space = GNNStateSpace(
            num_trucks=num_trucks,
            num_stops=num_stops,
            max_time=config['environment']['max_time'],
            num_charging_nodes=eval_env.num_charging_nodes,
            device="cpu",
            verbose=False
        )
        
        episode_rewards = []
        episode_success = []
        
        for ep in range(num_episodes_per_config):
            obs, info = eval_env.reset(seed=seed + ep)
            episode_reward = 0
            done = False
            truncated = False
            
            while not (done or truncated):
                gnn_state = state_space.get_state_GNN(eval_env)
                raw_action = policy.select_action(gnn_state, expl_noise=0)
                action = policy.to_env_action(gnn_state, int(raw_action))
                
                obs, reward, done, truncated, info = eval_env.step(action)
                episode_reward += reward
            
            episode_rewards.append(episode_reward)
            episode_success.append(1.0 if info.get('all_complete', False) else 0.0)
        
        results[(num_trucks, num_stops)] = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'success_rate': np.mean(episode_success),
        }
        
        all_rewards.extend(episode_rewards)
        all_success.extend(episode_success)
        
        eval_env.close()
    
    # Aggregate metrics across all configs
    results['aggregate'] = {
        'mean_reward': np.mean(all_rewards),
        'std_reward': np.std(all_rewards),
        'success_rate': np.mean(all_success),
    }
    
    return results


def train(args):
    """Main training loop."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load configurations
    base_config = load_config(args.config)
    curriculum_strategy, curriculum_config = load_curriculum_strategy(args.curriculum_config, args.seed)
    
    # Initialize wandb
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            entity='stavrosorf',
            name=args.exp_name,
            config={**vars(args), **curriculum_config},
            save_code=True
        )
    
    # Create curriculum environment
    train_env = CurriculumEnvWrapper(
        base_config=base_config,
        curriculum_strategy=curriculum_strategy,
        verbose=False,
        enable_plotting=False
    )
    
    # Initialize state spaces cache
    state_spaces = {}
    
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
    state_spaces[(initial_trucks, initial_stops)] = initial_state_space
    
    gnn_state = initial_state_space.get_state_GNN(train_env.env)
    
    # Determine node feature dimensions
    node_feature_dims = {}
    for node_type in gnn_state.node_types:
        features = gnn_state[node_type].x
        feature_dim = int(features.shape[-1]) if features.dim() > 1 else int(features.numel())
        node_feature_dims[node_type] = feature_dim
    
    # Determine max action dimension
    max_trucks = max([s['truck_range'][1] for s in curriculum_config.get('stages', [curriculum_config])]) \
                 if 'stages' in curriculum_config else \
                 max([level['truck_range'][1] for level in curriculum_config.get('difficulty_levels', [curriculum_config])])
    max_stops = max([s['stop_range'][1] for s in curriculum_config.get('stages', [curriculum_config])]) \
                if 'stages' in curriculum_config else \
                max([level['stop_range'][1] for level in curriculum_config.get('difficulty_levels', [curriculum_config])])
    
    dummy_config = copy.deepcopy(base_config)
    dummy_config['environment']['num_trucks'] = max_trucks
    dummy_config['environment']['num_stops'] = max_stops
    dummy_env = EventDrivenTruckEnv(config=dummy_config, verbose=False, enable_plotting=False)
    action_dim = dummy_env.action_space.n
    dummy_env.close()
    
    # Create PPO policy
    policy = PPOVariableActionGNN(
        action_dim=action_dim,
        node_feature_dims=node_feature_dims,
        hidden_dim=64,
        num_layers=3,
        mlp_dim=256,
        lr=args.lr,
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        ppo_epochs=10,
        minibatch_size=256,
        charge_durations=base_config["charging"]["charge_durations"],
    )
    
    # Save directory
    save_dir = Path("saved_models") / args.exp_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate evaluation configurations
    eval_configs = generate_eval_configs(curriculum_config, num_samples=30)
    
    print(f"\n{'='*80}")
    print(f"Training: {args.exp_name}")
    print(f"Strategy: {curriculum_config['strategy']}")
    print(f"Max timesteps: {args.max_timesteps:,}")
    print(f"Evaluation: {len(eval_configs)} diverse configs")
    print(f"{'='*80}\n")
    
    # Training loop
    total_timesteps = 0
    episode_num = 0
    episode_reward = 0.0
    episode_timesteps = 0
    best_success_rate = 0.0
    
    for t in range(args.max_timesteps):
        episode_timesteps += 1
        
        # Get current state space
        curr_trucks = info['curriculum']['num_trucks']
        curr_stops = info['curriculum']['num_stops']
        size_key = (curr_trucks, curr_stops)
        
        if size_key not in state_spaces:
            state_spaces[size_key] = GNNStateSpace(
                num_trucks=curr_trucks,
                num_stops=curr_stops,
                max_time=base_config['environment']['max_time'],
                num_charging_nodes=train_env.num_charging_nodes,
                device="cpu",
                verbose=False
            )
        
        state_space = state_spaces[size_key]
        gnn_state = state_space.get_state_GNN(train_env.env)
        
        # Act
        action, logprob, value = policy.act(gnn_state)
        env_action = policy.to_env_action(gnn_state, action)
        
        # Step
        next_obs, reward, done, truncated, info = train_env.step(env_action)
        done_flag = done or truncated
        
        # Get next state
        next_gnn_state = None if done_flag else state_space.get_state_GNN(train_env.env)
        
        # Store transition
        policy.store_transition(gnn_state, action, logprob, reward, done_flag, value)
        
        episode_reward += reward
        total_timesteps += 1
        
        # PPO update
        if (t + 1) % args.ppo_steps == 0:
            last_value = policy.value(next_gnn_state) if not done_flag else 0.0
            update_stats = policy.update(last_value)
            
            if update_stats and not args.no_wandb:
                wandb.log({
                    'train/policy_loss': update_stats.get('policy_loss', 0.0),
                    'train/value_loss': update_stats.get('value_loss', 0.0),
                    'train/entropy': update_stats.get('entropy', 0.0),
                    'timestep': total_timesteps
                })
        
        # Episode end
        if done_flag:
            if not args.no_wandb:
                wandb.log({
                    'train/reward': episode_reward,
                    'train/length': episode_timesteps,
                    'train/success': 1.0 if info.get('all_complete', False) else 0.0,
                    'train/trucks': curr_trucks,
                    'train/stops': curr_stops,
                    'timestep': total_timesteps
                })
            
            if episode_num % 100 == 0:
                print(f"Ep {episode_num} | T={curr_trucks} S={curr_stops} | "
                      f"R={episode_reward:.1f} | Steps={episode_timesteps}")
            
            # Reset
            obs, info = train_env.reset(seed=args.seed + episode_num + 1)
            episode_reward = 0.0
            episode_timesteps = 0
            episode_num += 1
        
        # Evaluation
        if (t + 1) % args.eval_freq == 0:
            print(f"\n{'='*60}")
            print(f"Evaluation @ {total_timesteps:,} timesteps")
            print(f"{'='*60}")
            
            eval_results = evaluate_policy(
                policy, base_config, eval_configs, 
                args.eval_episodes, seed=args.seed + 100_000
            )
            
            # Log aggregate
            agg = eval_results['aggregate']
            print(f"Aggregate: Reward={agg['mean_reward']:.2f}±{agg['std_reward']:.2f}, "
                  f"Success={agg['success_rate']*100:.1f}%")
            
            if not args.no_wandb:
                wandb.log({
                    'eval/reward': agg['mean_reward'],
                    'eval/success_rate': agg['success_rate'],
                    'timestep': total_timesteps
                })
                
                # Log per-config metrics (sample subset to avoid spam)
                for i, ((trucks, stops), metrics) in enumerate(list(eval_results.items())[:5]):
                    if (trucks, stops) == 'aggregate':
                        continue
                    wandb.log({
                        f'eval_detailed/t{trucks}_s{stops}_reward': metrics['mean_reward'],
                        f'eval_detailed/t{trucks}_s{stops}_success': metrics['success_rate'],
                        'timestep': total_timesteps
                    })
            
            # Save best model
            if agg['success_rate'] > best_success_rate:
                best_success_rate = agg['success_rate']
                policy.save(str(save_dir / "ppo_best"))
                print(f"✓ New best model saved! Success={best_success_rate*100:.1f}%")
            
            print(f"{'='*60}\n")
    
    # Final model
    policy.save(str(save_dir / "ppo_final"))
    
    print(f"\n{'='*80}")
    print(f"Training completed!")
    print(f"Best success rate: {best_success_rate*100:.1f}%")
    print(f"Models saved to: {save_dir}")
    print(f"{'='*80}\n")
    
    train_env.close()
    
    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    train(args)
