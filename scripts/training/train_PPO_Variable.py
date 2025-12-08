"""
Training loop for PPO-Variable Action-GNN agent using wandb to log results.
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

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.state.gnn_state_space import GNNStateSpace
from algo.PPO_VariableActionGNN import PPOVariableActionGNN
from EVRoutingEnv.utils.utils import load_config


def save_network_config(save_dir, config_dict):
    """Save neural network configuration to JSON file in the save directory.
    
    Args:
        save_dir: Directory where the config should be saved
        config_dict: Dictionary containing the network configuration
    """
    config_file = os.path.join(save_dir, "ppo_network_config.json")
    
    # Convert non-serializable types to serializable ones
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
    """Parse command line arguments for training hyperparameters."""
    parser = argparse.ArgumentParser(description='Train PPO-Variable Action-GNN agent for Electric Truck Routing')
    
    # Environment parameters
    env_group = parser.add_argument_group('Environment')
    env_group.add_argument('--config', type=str, default='EVRoutingEnv/config_files/config.yaml',
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
    train_group.add_argument('--eval-freq', type=int, default=500,
                            help='Evaluation frequency (in timesteps)')
    train_group.add_argument('--eval-episodes', type=int, default=30,
                            help='Number of episodes for evaluation')
    train_group.add_argument('--batch-size', type=int, default=256,
                            help='Batch size for training')
    
    # PPO hyperparameters
    ppo_group = parser.add_argument_group('PPO Algorithm')
    ppo_group.add_argument('--ppo-steps-per-update', type=int, default=128,
                           help='Number of timesteps collected before each PPO update')
    ppo_group.add_argument('--ppo-epochs', type=int, default=10,
                           help='Number of gradient epochs per PPO update')
    ppo_group.add_argument('--ppo-minibatch-size', type=int, default=256,
                           help='Minibatch size for PPO updates')
    ppo_group.add_argument('--gamma', type=float, default=0.99,
                           help='Discount factor for rewards')
    ppo_group.add_argument('--gae-lambda', type=float, default=0.95,
                           help='GAE lambda parameter')
    ppo_group.add_argument('--ppo-clip', type=float, default=0.2,
                           help='Clipping coefficient for PPO')
    ppo_group.add_argument('--ppo-entropy-coef', type=float, default=0.01,
                           help='Entropy bonus coefficient')
    ppo_group.add_argument('--ppo-value-coef', type=float, default=0.01,
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
    log_group.add_argument('--wandb-project', type=str, default='evpr-ppo-variable',
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


def evaluate_policy(env, policy, gnn_state_space, eval_episodes=10, seed=0):
    """Evaluate the current policy and collect detailed metrics."""
    eval_rewards = []
    eval_success_rate = []
    eval_episode_lengths = []
    eval_total_charging_time = []
    eval_num_charging_sessions = []
    eval_waiting_time = []
    eval_total_distance = []
    eval_episode_time = []
    eval_total_routing_time = []
    eval_total_unloading_time = []
    eval_total_failures = []
    eval_total_deliveries = []
    
    for episode in range(eval_episodes):
        obs, info = env.reset(seed=seed + episode)
        episode_reward = 0
        episode_steps = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            # Get GNN state from the CORRECT environment (eval_env, not train env)
            gnn_state = gnn_state_space.get_state_GNN(env)
            
            # Select action without exploration noise (greedy)
            # For PPO-variable, don't pass action_mask since GNN state has its own action space
            if isinstance(policy, PPOVariableActionGNN):
                raw_action = policy.select_action(gnn_state, expl_noise=0)
                action = policy.to_env_action(gnn_state, int(raw_action))
            else:
                action_mask = compute_action_mask(env)
                raw_action = policy.select_action(gnn_state, expl_noise=0, action_mask=action_mask)
                if isinstance(raw_action, tuple):
                    action = raw_action
                else:
                    action = int(raw_action) % env.action_space.n
            
            # Take action
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            episode_steps += 1
        
        # Collect episode metrics from environment info
        eval_rewards.append(episode_reward)
        eval_success_rate.append(1.0 if info['all_complete'] else 0.0)
        eval_episode_lengths.append(episode_steps)
        
        # Extract metrics from truck states in info (these are episode-specific)
        total_charging_time = 0.0
        num_charging_sessions = 0
        total_waiting_time = 0.0
        total_distance = 0.0
        total_routing_time = 0.0
        total_unloading_time = 0.0
        num_failures = 0
        num_deliveries = 0
        
        for truck_info in info['trucks']:
            total_charging_time += truck_info['total_charging_time']
            num_charging_sessions += truck_info['num_charging_sessions']
            total_waiting_time += truck_info['waiting_time']
            total_distance += truck_info['total_distance']
            total_unloading_time += truck_info['total_unloading_time']
            
            # Count failures
            if truck_info['failed']:
                num_failures += 1
            
            # Count completed deliveries (total deliveries - remaining deliveries)
            total_deliveries = len(truck_info['delivery_sequence']) - 1  # Exclude depot
            deliveries_remaining = truck_info['deliveries_remaining']
            deliveries_completed = total_deliveries - deliveries_remaining
            num_deliveries += deliveries_completed
            
            # Calculate routing time: total_time - charging - unloading - waiting
            truck_total_time = truck_info['total_time']
            truck_charging_time = truck_info['total_charging_time']
            truck_unloading_time = truck_info['total_unloading_time']
            truck_waiting_time = truck_info['waiting_time']
            truck_routing_time = truck_total_time - truck_charging_time - truck_unloading_time - truck_waiting_time
            total_routing_time += max(0.0, truck_routing_time)  # Ensure non-negative
        
        eval_total_charging_time.append(total_charging_time)
        eval_num_charging_sessions.append(num_charging_sessions)
        eval_waiting_time.append(total_waiting_time)
        eval_total_distance.append(total_distance)
        eval_total_routing_time.append(total_routing_time)
        eval_total_unloading_time.append(total_unloading_time)
        eval_total_failures.append(num_failures)
        eval_total_deliveries.append(num_deliveries)
        
        # Get episode time from global clock
        eval_episode_time.append(info['global_clock'])
    
    return {
        'mean_reward': np.mean(eval_rewards),
        'std_reward': np.std(eval_rewards),
        'success_rate': np.mean(eval_success_rate),
        'mean_episode_length': np.mean(eval_episode_lengths),
        'mean_charging_time': np.mean(eval_total_charging_time),
        'mean_num_charging_sessions': np.mean(eval_num_charging_sessions),
        'mean_waiting_time': np.mean(eval_waiting_time),
        'mean_total_distance': np.mean(eval_total_distance),
        'mean_episode_time': np.mean(eval_episode_time) if eval_episode_time else 0.0,
        'mean_routing_time': np.mean(eval_total_routing_time),
        'mean_unloading_time': np.mean(eval_total_unloading_time),
        'mean_failures': np.mean(eval_total_failures),
        'mean_deliveries': np.mean(eval_total_deliveries)
    }

def train(args):
    """PPO-Variable training loop using the GNN state representation."""
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
    verbose = args.verbose
    env = EventDrivenTruckEnv(
        config=config,
        verbose=verbose,
        enable_plotting=False,
        run_id=args.exp_name
    )
    eval_env = EventDrivenTruckEnv(
        config=copy.deepcopy(config),
        verbose=verbose,
        enable_plotting=False,
        run_id=f"{args.exp_name}_eval"
    )

    gnn_state_space = GNNStateSpace(
        num_trucks=config['environment']['num_trucks'],
        num_stops=config['environment']['num_stops'],
        max_time=config['environment']['max_time'],
        num_charging_nodes=env.num_charging_nodes,
        device="cpu",  # Always create states on CPU for buffer storage
        verbose=verbose  # Disable verbose output during training
    )

    action_dim = env.action_space.n

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
    if not hasattr(gnn_state, 'feasible_action_mask'):
        raise ValueError("GNN state missing feasible_action_mask attribute required for PPO training.")

    node_feature_dims = {}
    for node_type in gnn_state.node_types:
        features = gnn_state[node_type].x
        if features.dim() == 1:
            feature_dim = int(features.numel())
        else:
            feature_dim = int(features.shape[-1])
        node_feature_dims[node_type] = feature_dim

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
        charge_durations=config["charging"]["charge_durations"],
    )

    policy = PPOVariableActionGNN(**policy_kwargs)

    # Save network configuration once
    ppo_config = {
        "algo": "ppo-variable",  # Always use ppo-variable
        "action_dim": action_dim,
        "node_feature_dims": node_feature_dims,
        "hidden_dim": args.gnn_hidden_dim,
        "num_layers": args.actor_gcn_layers,
        "mlp_dim": args.mlp_hidden_dim,
        "lr": args.lr,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip_coef": args.ppo_clip,
        "value_coef": args.ppo_value_coef,
        "entropy_coef": args.ppo_entropy_coef,
        "max_grad_norm": args.ppo_max_grad_norm,
        "ppo_epochs": args.ppo_epochs,
        "minibatch_size": args.ppo_minibatch_size,
        "charge_durations": config["charging"]["charge_durations"],
        "seed": args.seed
    }
    
    save_network_config(save_dir, ppo_config)

    print(f"\n{'='*80}")
    print(f"Starting PPO-Variable Training: {args.exp_name}")
    print(f"{'='*80}")
    print(f"Environment: {config['environment']['num_trucks']} trucks, {config['environment']['num_stops']} stops")
    print(f"Max timesteps: {args.max_timesteps}")
    print(f"Steps per PPO update: {args.ppo_steps_per_update}")
    print(f"Minibatch size: {args.ppo_minibatch_size}")
    print(f"{'='*80}\n")

    for t in range(args.max_timesteps):
        episode_timesteps += 1

        # PPO-variable has its own action space with feasible_action_mask in GNN state
        action, logprob, value = policy.act(gnn_state)
        env_action = policy.to_env_action(gnn_state, action)
        
        next_obs, reward, done, truncated, info = env.step(env_action)
        
        done_flag = done or truncated

        # Only get next state if episode is not done
        if done_flag:
            next_gnn_state = None  # Episode is over, no next state
        else:
            next_gnn_state = gnn_state_space.get_state_GNN(env)

        # Store transition
        policy.store_transition(gnn_state, action, logprob, reward, done_flag, value)

        # Only update current state if episode continues
        if not done_flag:
            gnn_state = next_gnn_state
        episode_reward += reward
        total_timesteps += 1

        if (t + 1) % args.ppo_steps_per_update == 0:
            last_value = policy.value(gnn_state)
            update_stats = policy.update(last_value)
            if update_stats and not args.no_wandb:
                wandb.log({
                    'train/policy_loss': update_stats['policy_loss'],
                    'train/value_loss': update_stats['value_loss'],
                    'train/entropy': update_stats['entropy'],
                    'train/timestep': total_timesteps
                })

        if done_flag:
            if not args.no_wandb:
                wandb.log({
                    'train/episode_reward': episode_reward,
                    'train/episode_length': episode_timesteps,
                    'train/episode': episode_num,
                    'train/success': 1.0 if info['all_complete'] else 0.0,
                    'train/timestep': total_timesteps
                })

            print(f"PPO Episode {episode_num}: Reward={episode_reward:.2f}, "
                  f"Steps={episode_timesteps}, Success={info['all_complete']}")

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
                                           args.eval_episodes, args.seed + 1000)

            print(f"\n{'='*80}")
            print(f"PPO Evaluation at timestep {total_timesteps}")
            print(f"Mean Reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
            print(f"Success Rate: {eval_results['success_rate']*100:.1f}%")
            print(f"Episode Length: {eval_results['mean_episode_length']:.1f} steps")
            print(f"Charging Time: {eval_results['mean_charging_time']:.2f} hours")
            print(f"Charging Sessions: {eval_results['mean_num_charging_sessions']:.1f}")
            print(f"Routing Time: {eval_results['mean_routing_time']:.2f} hours")
            print(f"Unloading Time: {eval_results['mean_unloading_time']:.2f} hours")
            print(f"Waiting Time: {eval_results['mean_waiting_time']:.2f} hours")
            print(f"Distance Traveled: {eval_results['mean_total_distance']:.1f} km")
            print(f"Truck Failures: {eval_results['mean_failures']:.1f}")
            print(f"Deliveries Completed: {eval_results['mean_deliveries']:.1f}")
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
                    'eval/routing_time': eval_results['mean_routing_time'],
                    'eval/unloading_time': eval_results['mean_unloading_time'],
                    'eval/waiting_time': eval_results['mean_waiting_time'],
                    'eval/total_distance': eval_results['mean_total_distance'],
                    'eval/episode_time': eval_results['mean_episode_time'],
                    'eval/failures': eval_results['mean_failures'],
                    'eval/deliveries': eval_results['mean_deliveries'],
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


if __name__ == "__main__":
    args = parse_args()
    train(args)
