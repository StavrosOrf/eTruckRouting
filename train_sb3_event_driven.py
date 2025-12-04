"""
Train the EventDrivenTruckEnv using Stable-Baselines3 discrete action space algorithms.
Supports: PPO, MaskablePPO, DQN, and QR-DQN.
"""

import os
import argparse
import torch
import wandb
import numpy as np
from datetime import datetime

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import MaskablePPO, QRDQN
from sb3_contrib.common.wrappers import ActionMasker
from wandb.integration.sb3 import WandbCallback

from truck_env.models.event_driven_env import EventDrivenTruckEnv
from truck_env.state.action_mask import get_action_mask


def mask_fn(env) -> np.ndarray:
    """Return action mask for the current environment state.
    
    The env passed here may be wrapped (e.g., Monitor), so we need to
    unwrap it to get the base EventDrivenTruckEnv.
    """
    # Unwrap to get the base environment
    base_env = env
    while hasattr(base_env, 'env'):
        base_env = base_env.env
    return get_action_mask(base_env)


def make_env(config_path: str, seed: int, enable_plotting: bool = False):
    """Create and return environment wrapped with Monitor."""
    def _init():
        env = EventDrivenTruckEnv(
            config=config_path,
            verbose=False,
            enable_plotting=enable_plotting
        )
        env = Monitor(env)
        return env
    return _init


def make_masked_env(config_path: str, seed: int, enable_plotting: bool = False):
    """Create and return environment wrapped for MaskablePPO with Monitor and ActionMasker."""
    def _init():
        env = EventDrivenTruckEnv(
            config=config_path,
            verbose=False,
            enable_plotting=enable_plotting
        )
        env = Monitor(env)
        env = ActionMasker(env, mask_fn)
        return env
    return _init


def train_sb3_agent(
    algo: str,
    seed: int,
    config_path: str,
    device: str,
    total_steps: int,
    eval_freq: int = 1000,
    n_eval_episodes: int = 10,
    save_dir: str = "./saved_models",
    use_wandb: bool = True,
    project_name: str = "evrp-sb3",
    **kwargs
):
    """
    Train an SB3 agent on EventDrivenTruckEnv.
    
    Args:
        algo: Algorithm name ('ppo', 'maskppo', 'dqn', 'qrdqn')
        seed: Random seed
        config_path: Path to environment config file
        device: Device to use ('cpu' or 'cuda')
        total_steps: Total timesteps to train
        eval_freq: Evaluation frequency
        n_eval_episodes: Number of evaluation episodes
        save_dir: Directory to save models
        use_wandb: Whether to use wandb logging
        project_name: Wandb project name
    """
    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    
    algo = algo.lower()
    
    # Load config to extract environment parameters for group naming
    from truck_env.utils.utils import load_config
    config = load_config(config_path)
    
    print(f"\n{'='*60}")
    print(f"Training Configuration")
    print(f"{'='*60}")
    print(f"Algorithm: {algo.upper()}")
    print(f"Seed: {seed}")
    print(f"Device: {device}")
    print(f"Total steps: {total_steps:,}")
    print(f"Config: {config_path}")
    print(f"{'='*60}\n")
    
    # Initialize wandb with group name based on environment config
    run_name = f"{algo}_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    num_trucks = config['environment']['num_trucks']
    num_stops = config['environment']['num_stops']
    group_name = f"{num_trucks}trucks_{num_stops}stops"
    
    if use_wandb:
        wandb.init(
            project=project_name,
            entity="stavrosorf",
            name=run_name,
            group=group_name,
            config={
                "algorithm": algo,
                "seed": seed,
                "total_steps": total_steps,
                "device": device,
                "config_path": config_path,
                "num_trucks": num_trucks,
                "num_stops": num_stops,
            },
            sync_tensorboard=True,
            save_code=True,
        )
    
    # Create environments
    use_masked = (algo == "maskppo")
    env_fn = make_masked_env if use_masked else make_env
    
    train_env = DummyVecEnv([env_fn(config_path, seed)])
    eval_env = DummyVecEnv([env_fn(config_path, seed + 100)])
    
    # Setup evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=1,
    )
    
    # Setup callbacks
    callbacks = [eval_callback]
    if use_wandb:
        callbacks.append(WandbCallback(
            gradient_save_freq=1000,
            model_save_path=f"{save_dir}/{run_name}",
            verbose=2,
        ))
    
    # Create model based on algorithm
    model = None
    
    # Try to use tensorboard if available
    try:
        import tensorboard
        tb_log = f"./logs/{run_name}"
    except ImportError:
        print("Warning: tensorboard not installed, logging disabled")
        tb_log = None
    
    if algo == "ppo":
        print("Initializing PPO with default parameters...")
        model = PPO(
            "MlpPolicy",
            train_env,
            verbose=1,
            device=device,
            tensorboard_log=tb_log,
            seed=seed,
        )
    
    elif algo == "maskppo":
        print("Initializing MaskablePPO with default parameters...")
        model = MaskablePPO(
            "MlpPolicy",
            train_env,
            verbose=1,
            device=device,
            tensorboard_log=tb_log,
            seed=seed,
        )
    
    elif algo == "dqn":
        print("Initializing DQN with default parameters...")
        model = DQN(
            "MlpPolicy",
            train_env,
            verbose=1,
            device=device,
            tensorboard_log=tb_log,
            seed=seed,
        )
    
    elif algo == "qrdqn":
        print("Initializing QR-DQN with default parameters...")
        model = QRDQN(
            "MlpPolicy",
            train_env,
            verbose=1,
            device=device,
            tensorboard_log=tb_log,
            seed=seed,
        )
    
    else:
        raise ValueError(f"Unknown algorithm: {algo}. Supported: ppo, maskppo, dqn, qrdqn")
    
    # Train the model
    print(f"\nStarting training for {total_steps:,} steps...")
    print(f"{'='*60}\n")
    
    model.learn(
        total_timesteps=total_steps,
        progress_bar=True,
        callback=callbacks,
    )
    
    # Save the final model
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"{run_name}_final")
    model.save(model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Close wandb
    if use_wandb:
        wandb.finish()
    
    return model, model_path


def main():
    parser = argparse.ArgumentParser(
        description="Train EventDrivenTruckEnv with SB3 algorithms"
    )
    parser.add_argument(
        "--algo",
        type=str,
        choices=["ppo", "maskppo", "dqn", "qrdqn"],
        required=True,
        help="Algorithm to use",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="truck_env/config_files/config.yaml",
        help="Path to environment config file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cpu or cuda)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100000,
        help="Total training steps",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=1000,
        help="Evaluation frequency",
    )
    parser.add_argument(
        "--n-eval-episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./saved_models",
        help="Directory to save models",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable wandb logging",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="evpr-sb3",
        help="Wandb project name",
    )
    
    args = parser.parse_args()
    
    # Train the agent
    model, model_path = train_sb3_agent(
        algo=args.algo,
        seed=args.seed,
        config_path=args.config,
        device=args.device,
        total_steps=args.steps,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        save_dir=args.save_dir,
        use_wandb=not args.no_wandb,
        project_name=args.project,
    )
    
    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"Final model saved to: {model_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
