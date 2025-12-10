"""
Training loop for PPO-Variable Action-GNN agent with TRUE parallel environments.
Uses multiprocessing to run multiple environments simultaneously for faster training.
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
from tqdm import trange
from multiprocessing import Process, Pipe
from typing import List, Callable, Tuple, Optional
import traceback

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.state.gnn_state_space import GNNStateSpace
from algo.PPO_VariableActionGNN import PPOVariableActionGNN
from EVRoutingEnv.utils.utils import load_config
import yaml


def save_environment_config(save_dir, config_dict):
    """Save environment configuration to YAML file in the save directory."""
    config_file = os.path.join(save_dir, "config.yaml")
    
    with open(config_file, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    
    print(f"Environment configuration saved to: {config_file}")


def save_network_config(save_dir, config_dict):
    """Save neural network configuration to JSON file in the save directory."""
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


def worker_process(remote, parent_remote, env_config, worker_id, verbose=False):
    """Worker process that runs a single environment.
    
    Args:
        remote: Communication pipe endpoint for this worker
        parent_remote: Parent's pipe endpoint (closed in worker)
        env_config: Configuration dict for environment
        worker_id: Unique ID for this worker
        verbose: Whether to enable verbose output
    """
    parent_remote.close()
    
    try:
        # Create environment in this process
        env = EventDrivenTruckEnv(
            config=copy.deepcopy(env_config),
            verbose=verbose,
            enable_plotting=False,
            run_id=f"parallel_worker_{worker_id}"
        )
        
        # Create GNN state space in this process
        gnn_state_space = GNNStateSpace(
            num_trucks=env_config['environment']['num_trucks'],
            num_stops=env_config['environment']['num_stops'],
            max_time=env_config['environment']['max_time'],
            num_charging_nodes=env.num_charging_nodes,
            device="cpu",
            verbose=False
        )
        
        current_obs = None
        current_info = None
        
        while True:
            try:
                cmd, data = remote.recv()
                
                if cmd == 'reset':
                    seed = data
                    obs, info = env.reset(seed=seed)
                    gnn_state = gnn_state_space.get_state_GNN(env)
                    current_obs = obs
                    current_info = info
                    remote.send(('success', (gnn_state, info)))
                
                elif cmd == 'step':
                    action = data
                    obs, reward, done, truncated, info = env.step(action)
                    
                    # Get next state if episode continues
                    if done or truncated:
                        gnn_state = None
                        episode_info = {
                            'done': True,
                            'truncated': truncated,
                            'all_complete': info.get('all_complete', False),
                            'global_clock': info.get('global_clock', 0.0),
                            'trucks': info.get('trucks', [])
                        }
                    else:
                        gnn_state = gnn_state_space.get_state_GNN(env)
                        episode_info = {
                            'done': False,
                            'truncated': False
                        }
                    
                    current_obs = obs
                    current_info = info
                    remote.send(('success', (gnn_state, reward, done, truncated, episode_info)))
                
                elif cmd == 'close':
                    env.close()
                    remote.close()
                    break
                
                else:
                    remote.send(('error', f'Unknown command: {cmd}'))
            
            except Exception as e:
                error_msg = f"Worker {worker_id} error in command '{cmd}': {str(e)}\n{traceback.format_exc()}"
                remote.send(('error', error_msg))
    
    except Exception as e:
        error_msg = f"Worker {worker_id} initialization error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)


class ParallelEnvs:
    """Manages multiple environments running in parallel processes."""
    
    def __init__(self, env_config: dict, num_envs: int, verbose: bool = False):
        """Initialize parallel environments.
        
        Args:
            env_config: Configuration dict for environments
            num_envs: Number of parallel environments
            verbose: Whether to enable verbose output
        """
        self.num_envs = num_envs
        self.env_config = env_config
        self.verbose = verbose
        
        # Create pipes for communication
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(num_envs)])
        
        # Start worker processes
        self.processes = []
        for i, (work_remote, remote) in enumerate(zip(self.work_remotes, self.remotes)):
            p = Process(
                target=worker_process,
                args=(work_remote, remote, env_config, i, verbose),
                daemon=True
            )
            p.start()
            self.processes.append(p)
            work_remote.close()
        
        print(f"Started {num_envs} parallel environment workers")
    
    def reset(self, seeds: List[int]) -> Tuple[List, List]:
        """Reset all environments with given seeds.
        
        Args:
            seeds: List of seeds for each environment
            
        Returns:
            Tuple of (gnn_states, infos)
        """
        assert len(seeds) == self.num_envs, f"Expected {self.num_envs} seeds, got {len(seeds)}"
        
        # Send reset commands to all workers
        for remote, seed in zip(self.remotes, seeds):
            remote.send(('reset', seed))
        
        # Collect results
        gnn_states = []
        infos = []
        for i, remote in enumerate(self.remotes):
            status, data = remote.recv()
            if status == 'error':
                raise RuntimeError(f"Worker {i} reset error: {data}")
            gnn_state, info = data
            gnn_states.append(gnn_state)
            infos.append(info)
        
        return gnn_states, infos
    
    def reset_single(self, env_idx: int, seed: int) -> Tuple:
        """Reset a single environment with a given seed.
        
        Args:
            env_idx: Index of the environment to reset
            seed: Seed for the environment
            
        Returns:
            Tuple of (gnn_state, info)
        """
        assert 0 <= env_idx < self.num_envs, f"env_idx {env_idx} out of range [0, {self.num_envs})"
        
        # Send reset command to specific worker
        self.remotes[env_idx].send(('reset', seed))
        
        # Collect result
        status, data = self.remotes[env_idx].recv()
        if status == 'error':
            raise RuntimeError(f"Worker {env_idx} reset error: {data}")
        
        return data
    
    def step(self, actions: List) -> Tuple[List, List[float], List[bool], List[bool], List[dict]]:
        """Step all environments with given actions.
        
        Args:
            actions: List of actions for each environment
            
        Returns:
            Tuple of (gnn_states, rewards, dones, truncateds, infos)
        """
        assert len(actions) == self.num_envs, f"Expected {self.num_envs} actions, got {len(actions)}"
        
        # Send step commands to all workers
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        
        # Collect results
        gnn_states = []
        rewards = []
        dones = []
        truncateds = []
        infos = []
        
        for i, remote in enumerate(self.remotes):
            status, data = remote.recv()
            if status == 'error':
                raise RuntimeError(f"Worker {i} step error: {data}")
            gnn_state, reward, done, truncated, info = data
            gnn_states.append(gnn_state)
            rewards.append(reward)
            dones.append(done)
            truncateds.append(truncated)
            infos.append(info)
        
        return gnn_states, rewards, dones, truncateds, infos
    
    def close(self):
        """Close all worker processes."""
        for remote in self.remotes:
            remote.send(('close', None))
        
        for p in self.processes:
            p.join(timeout=5.0)
            if p.is_alive():
                p.terminate()
        
        print(f"Closed {self.num_envs} parallel environment workers")


def evaluate_policy_parallel(
    config: dict,
    policy: PPOVariableActionGNN,
    eval_episodes: int = 50,
    seed: int = 0,
    num_parallel_envs: int = 10,
    verbose: bool = False
) -> dict:
    """Evaluate policy using parallel environments.
    
    Args:
        config: Environment configuration
        policy: Policy to evaluate
        eval_episodes: Total number of evaluation episodes
        seed: Base random seed
        num_parallel_envs: Number of parallel environments
        verbose: Whether to enable verbose output
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Create parallel environments for evaluation
    parallel_envs = ParallelEnvs(config, num_parallel_envs, verbose=False)
    
    # Storage for metrics
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
    
    # Episode tracking for each environment
    episode_rewards = [0.0] * num_parallel_envs
    episode_steps = [0] * num_parallel_envs
    episodes_completed = 0
    episode_counters = list(range(num_parallel_envs))
    
    # Reset all environments
    seeds = [seed + i for i in range(num_parallel_envs)]
    gnn_states, infos = parallel_envs.reset(seeds)
    
    with trange(eval_episodes, desc="Eval episodes") as pbar:
        while episodes_completed < eval_episodes:
            # Select actions for all environments (batched for efficiency)
            active_indices = [i for i, state in enumerate(gnn_states) if state is not None]
            active_states = [gnn_states[i] for i in active_indices]
            
            actions = [None] * num_parallel_envs
            env_actions = [None] * num_parallel_envs
            
            if active_states:
                # Batch process all active states at once
                batch_actions, _, _ = policy.act_batch(active_states, deterministic=True)
                
                for idx, active_idx in enumerate(active_indices):
                    action = batch_actions[idx]
                    env_action = policy.to_env_action(active_states[idx], action)
                    actions[active_idx] = action
                    env_actions[active_idx] = env_action
            
            # Step all environments in parallel
            next_states, rewards, dones, truncateds, infos = parallel_envs.step(env_actions)
            
            # Process results for each environment
            for i in range(num_parallel_envs):
                if actions[i] is None:  # This environment was already done
                    continue
                
                episode_rewards[i] += rewards[i]
                episode_steps[i] += 1
                
                if dones[i] or truncateds[i]:
                    # Episode completed - collect metrics
                    info = infos[i]
                    eval_rewards.append(episode_rewards[i])
                    eval_success_rate.append(1.0 if info.get('all_complete', False) else 0.0)
                    eval_episode_lengths.append(episode_steps[i])
                    
                    # Extract detailed metrics
                    if 'trucks' in info:
                        total_charging_time = 0.0
                        num_charging_sessions = 0
                        total_waiting_time = 0.0
                        total_distance = 0.0
                        total_routing_time = 0.0
                        total_unloading_time = 0.0
                        num_failures = 0
                        num_deliveries = 0
                        
                        for truck_info in info['trucks']:
                            total_charging_time += truck_info.get('total_charging_time', 0.0)
                            num_charging_sessions += truck_info.get('num_charging_sessions', 0)
                            total_waiting_time += truck_info.get('waiting_time', 0.0)
                            total_distance += truck_info.get('total_distance', 0.0)
                            total_unloading_time += truck_info.get('total_unloading_time', 0.0)
                            
                            if truck_info.get('failed', False):
                                num_failures += 1
                            
                            total_deliveries = len(truck_info.get('delivery_sequence', [])) - 1
                            deliveries_remaining = truck_info.get('deliveries_remaining', 0)
                            deliveries_completed = max(0, total_deliveries - deliveries_remaining)
                            num_deliveries += deliveries_completed
                            
                            truck_total_time = truck_info.get('total_time', 0.0)
                            truck_charging_time = truck_info.get('total_charging_time', 0.0)
                            truck_unloading_time = truck_info.get('total_unloading_time', 0.0)
                            truck_waiting_time = truck_info.get('waiting_time', 0.0)
                            truck_routing_time = (truck_total_time - truck_charging_time - 
                                                truck_unloading_time - truck_waiting_time)
                            total_routing_time += max(0.0, truck_routing_time)
                        
                        eval_total_charging_time.append(total_charging_time)
                        eval_num_charging_sessions.append(num_charging_sessions)
                        eval_waiting_time.append(total_waiting_time)
                        eval_total_distance.append(total_distance)
                        eval_total_routing_time.append(total_routing_time)
                        eval_total_unloading_time.append(total_unloading_time)
                        eval_total_failures.append(num_failures)
                        eval_total_deliveries.append(num_deliveries)
                        eval_episode_time.append(info.get('global_clock', 0.0))
                    
                    episodes_completed += 1
                    pbar.update(1)
                    
                    # Reset this environment if we need more episodes
                    if episodes_completed < eval_episodes:
                        episode_rewards[i] = 0.0
                        episode_steps[i] = 0
                        next_episode_num = episodes_completed + num_parallel_envs - 1
                        new_seed = seed + next_episode_num
                        # Reset just this one environment
                        gnn_state, info = parallel_envs.reset_single(i, new_seed)
                        gnn_states[i] = gnn_state
                        episode_counters[i] = next_episode_num
                    else:
                        gnn_states[i] = None  # Mark as done
                else:
                    # Episode continues
                    gnn_states[i] = next_states[i]
    
    # Close parallel environments
    parallel_envs.close()
    
    return {
        'mean_reward': np.mean(eval_rewards) if eval_rewards else 0.0,
        'std_reward': np.std(eval_rewards) if eval_rewards else 0.0,
        'success_rate': np.mean(eval_success_rate) if eval_success_rate else 0.0,
        'mean_episode_length': np.mean(eval_episode_lengths) if eval_episode_lengths else 0.0,
        'mean_charging_time': np.mean(eval_total_charging_time) if eval_total_charging_time else 0.0,
        'mean_num_charging_sessions': np.mean(eval_num_charging_sessions) if eval_num_charging_sessions else 0.0,
        'mean_waiting_time': np.mean(eval_waiting_time) if eval_waiting_time else 0.0,
        'mean_total_distance': np.mean(eval_total_distance) if eval_total_distance else 0.0,
        'mean_episode_time': np.mean(eval_episode_time) if eval_episode_time else 0.0,
        'mean_routing_time': np.mean(eval_total_routing_time) if eval_total_routing_time else 0.0,
        'mean_unloading_time': np.mean(eval_total_unloading_time) if eval_total_unloading_time else 0.0,
        'mean_failures': np.mean(eval_total_failures) if eval_total_failures else 0.0,
        'mean_deliveries': np.mean(eval_total_deliveries) if eval_total_deliveries else 0.0
    }


def parse_args():
    """Parse command line arguments for training hyperparameters."""
    parser = argparse.ArgumentParser(description='Train PPO-Variable with TRUE Parallel Environments')
    
    # Environment parameters
    env_group = parser.add_argument_group('Environment')
    env_group.add_argument('--config', type=str, default='EVRoutingEnv/config_files/config.yaml',
                          help='Path to environment config file')
    env_group.add_argument('--num-trucks', type=int, default=None,
                          help='Number of trucks (overrides config)')
    env_group.add_argument('--num-stops', type=int, default=None,
                          help='Number of delivery stops per truck (overrides config)')
    env_group.add_argument('--max-time', type=float, default=None,
                          help='Maximum simulation time in hours (overrides config)')
    env_group.add_argument('--enable-traffic', action='store_true',
                          help='Enable traffic simulation')
    
    # Parallel training parameters
    parallel_group = parser.add_argument_group('Parallel Training')
    parallel_group.add_argument('--num-parallel-envs', type=int, default=8,
                               help='Number of parallel environments for training')
    parallel_group.add_argument('--num-eval-envs', type=int, default=10,
                               help='Number of parallel environments for evaluation')
    
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
    train_group.add_argument('--eval-episodes', type=int, default=50,
                            help='Number of episodes for evaluation')
    
    # PPO hyperparameters
    ppo_group = parser.add_argument_group('PPO Algorithm')
    ppo_group.add_argument('--ppo-steps-per-update', type=int, default=512,
                           help='Total timesteps collected before each PPO update (across all parallel envs)')
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
    
    # Logging and output
    log_group = parser.add_argument_group('Logging')
    log_group.add_argument('--wandb-project', type=str, default='evpr-ppo-variable-parallel',
                          help='Wandb project name')
    log_group.add_argument('--wandb-entity', type=str, default='stavrosorf',
                          help='Wandb entity (username or team)')
    log_group.add_argument('--exp-name', type=str, default=None,
                          help='Experiment name (auto-generated if not provided)')
    log_group.add_argument('--group-name', type=str, default=None,
                          help='Wandb group name for organizing related experiments')
    log_group.add_argument('--no-wandb', action='store_true',
                          help='Disable wandb logging')
    log_group.add_argument('--verbose', action='store_true',
                          help='Enable verbose output')
    
    return parser.parse_args()


def train(args):
    """PPO-Variable training loop with TRUE parallel environments."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load and modify config
    config = load_config(args.config)
    if args.num_trucks is not None:
        config['environment']['num_trucks'] = args.num_trucks
    if args.num_stops is not None:
        config['environment']['num_stops'] = args.num_stops
    if args.max_time is not None:
        config['environment']['max_time'] = args.max_time
    if args.enable_traffic:
        config['traffic']['enable_traffic'] = True

    # Generate experiment name
    if args.exp_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.exp_name = f"PPO_GNN_Parallel_{timestamp}"

    # Setup wandb
    if args.group_name is not None:
        group_name = args.group_name
    else:
        group_name = f"{config['environment']['num_trucks']}trucks_{config['environment']['num_stops']}stops_parallel"

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

    # Create save directory
    save_dir = os.path.join("saved_models", args.exp_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "ppo_model")
    
    save_environment_config(save_dir, config)

    # Create a temporary environment to get dimensions
    temp_env = EventDrivenTruckEnv(
        config=copy.deepcopy(config),
        verbose=False,
        enable_plotting=False,
        run_id="temp_env"
    )
    
    temp_gnn_state_space = GNNStateSpace(
        num_trucks=config['environment']['num_trucks'],
        num_stops=config['environment']['num_stops'],
        max_time=config['environment']['max_time'],
        num_charging_nodes=temp_env.num_charging_nodes,
        device="cpu",
        verbose=False
    )
    
    # Get state dimensions from temporary environment
    temp_env.reset(seed=args.seed)
    temp_state = temp_gnn_state_space.get_state_GNN(temp_env)
    
    node_feature_dims = {}
    for node_type in temp_state.node_types:
        features = temp_state[node_type].x
        if features.dim() == 1:
            feature_dim = int(features.numel())
        else:
            feature_dim = int(features.shape[-1])
        node_feature_dims[node_type] = feature_dim
    
    action_dim = temp_env.action_space.n
    temp_env.close()

    # Initialize policy
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

    # Save network configuration
    ppo_config = {
        "algo": "ppo-variable-parallel",
        "num_parallel_envs": args.num_parallel_envs,
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
    print(f"Starting PPO-Variable PARALLEL Training: {args.exp_name}")
    print(f"{'='*80}")
    print(f"Environment: {config['environment']['num_trucks']} trucks, {config['environment']['num_stops']} stops")
    print(f"Parallel Training Envs: {args.num_parallel_envs}")
    print(f"Parallel Eval Envs: {args.num_eval_envs}")
    print(f"Max timesteps: {args.max_timesteps}")
    print(f"Steps per PPO update: {args.ppo_steps_per_update}")
    print(f"Minibatch size: {args.ppo_minibatch_size}")
    print(f"{'='*80}\n")

    # Create parallel training environments
    parallel_envs = ParallelEnvs(config, args.num_parallel_envs, verbose=args.verbose)

    # Initialize training state
    best_eval_reward = None
    best_model_path = None
    total_timesteps = 0
    total_episodes = 0
    
    # Episode tracking for each parallel environment
    episode_rewards = [0.0] * args.num_parallel_envs
    episode_lengths = [0] * args.num_parallel_envs
    
    # Reset all parallel environments
    seeds = [args.seed + i for i in range(args.num_parallel_envs)]
    gnn_states, infos = parallel_envs.reset(seeds)
    
    # Training loop
    steps_since_update = 0
    
    try:
        with trange(args.max_timesteps, desc="Training") as pbar:
            while total_timesteps < args.max_timesteps:
                # Select actions for all environments (batched for efficiency)
                batch_actions, batch_logprobs, batch_values = policy.act_batch(gnn_states, deterministic=False)
                
                # Convert policy actions to environment actions
                env_actions = []
                for i, (action, gnn_state) in enumerate(zip(batch_actions, gnn_states)):
                    env_action = policy.to_env_action(gnn_state, action)
                    env_actions.append(env_action)
                
                actions = batch_actions
                logprobs = batch_logprobs
                values = batch_values
                
                # Step all environments in parallel
                next_states, rewards, dones, truncateds, infos = parallel_envs.step(env_actions)
                
                # Store transitions for all environments
                for i in range(args.num_parallel_envs):
                    done_flag = dones[i] or truncateds[i]
                    policy.store_transition(
                        gnn_states[i], 
                        actions[i], 
                        logprobs[i], 
                        rewards[i], 
                        done_flag, 
                        values[i]
                    )
                    
                    episode_rewards[i] += rewards[i]
                    episode_lengths[i] += 1
                    
                    # Handle episode completion
                    if done_flag:
                        info = infos[i]
                        
                        if not args.no_wandb:
                            wandb.log({
                                'train/episode_reward': episode_rewards[i],
                                'train/episode_length': episode_lengths[i],
                                'train/episode': total_episodes,
                                'train/success': 1.0 if info.get('all_complete', False) else 0.0,
                                'train/timestep': total_timesteps + i
                            })
                        
                        if args.verbose:
                            print(f"Episode {total_episodes} (env {i}): "
                                  f"Reward={episode_rewards[i]:.2f}, "
                                  f"Steps={episode_lengths[i]}, "
                                  f"Success={info.get('all_complete', False)}")
                        
                        total_episodes += 1
                        episode_rewards[i] = 0.0
                        episode_lengths[i] = 0
                        
                        # Reset this environment
                        new_seed = args.seed + total_timesteps + i
                        new_state, new_info = parallel_envs.reset_single(i, new_seed)
                        next_states[i] = new_state
                
                # Update current states
                gnn_states = next_states
                
                # Update counters
                total_timesteps += args.num_parallel_envs
                steps_since_update += args.num_parallel_envs
                pbar.update(args.num_parallel_envs)
                
                # PPO update
                if steps_since_update >= args.ppo_steps_per_update:
                    # Compute last values for all environments (batched)
                    last_values = policy.value_batch(gnn_states)
                    
                    # Use mean of last values for GAE computation
                    mean_last_value = torch.tensor(last_values).mean()
                    update_stats = policy.update(mean_last_value)
                    
                    if update_stats and not args.no_wandb:
                        wandb.log({
                            'train/policy_loss': update_stats['policy_loss'],
                            'train/value_loss': update_stats['value_loss'],
                            'train/entropy': update_stats['entropy'],
                            'train/timestep': total_timesteps
                        })
                    
                    steps_since_update = 0
                
                # Evaluation
                if total_timesteps % args.eval_freq < args.num_parallel_envs:
                    print(f"\n{'='*80}")
                    print(f"Evaluating at timestep {total_timesteps}...")
                    print(f"{'='*80}")
                    
                    eval_results = evaluate_policy_parallel(
                        config=config,
                        policy=policy,
                        eval_episodes=args.eval_episodes,
                        seed=args.seed + 100000,
                        num_parallel_envs=args.num_eval_envs,
                        verbose=False
                    )
                    
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
                            'eval/routing_time': eval_results['mean_routing_time'],
                            'eval/unloading_time': eval_results['mean_unloading_time'],
                            'eval/waiting_time': eval_results['mean_waiting_time'],
                            'eval/total_distance': eval_results['mean_total_distance'],
                            'eval/episode_time': eval_results['mean_episode_time'],
                            'eval/failures': eval_results['mean_failures'],
                            'eval/deliveries': eval_results['mean_deliveries'],
                            'eval/timestep': total_timesteps
                        })
                
                # Check if max episodes reached
                if total_episodes >= args.max_episodes:
                    print(f"Reached max episodes ({args.max_episodes}). Ending training.")
                    break
    
    finally:
        # Clean up parallel environments
        parallel_envs.close()
    
    # Final update if needed
    if len(policy.buffer.rewards) > 0:
        last_values = policy.value_batch(gnn_states)
        mean_last_value = torch.tensor(last_values).mean()
        policy.update(mean_last_value)
    
    # Save final model
    final_model_path = f"{save_path}_final"
    policy.save(final_model_path)
    
    print(f"\n{'='*80}")
    print(f"Training completed!")
    print(f"Final model saved to: {final_model_path}")
    if best_model_path is not None and best_eval_reward is not None:
        print(f"Best model saved to: {best_model_path} (Reward: {best_eval_reward:.2f})")
    print(f"Save directory: {save_dir}")
    print(f"{'='*80}\n")
    
    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    train(args)
