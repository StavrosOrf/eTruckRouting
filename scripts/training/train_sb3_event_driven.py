"""
Train the EventDrivenTruckEnv using Stable-Baselines3 discrete action space algorithms.
Supports: PPO, MaskablePPO, DQN, and QR-DQN.

LEGACY, for the inherited preassigned-route problem. The revision's learned
arms -- including the flat-state MaskPPO equivalent, DeepSets-PPO, the
state-GNN, the attention model, and the unmasked ablation -- all train through
scripts/training/train_canonical_ppo.py, so that the observation, mask,
curriculum, reward, seed stream, and budget are identical by construction and
only the architecture or the ablated component differs.
"""

import os
import sys
import argparse
import torch
import wandb
import numpy as np
from datetime import datetime

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import MaskablePPO, QRDQN
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.evaluation import (
    evaluate_policy as evaluate_maskable_policy,
)
from wandb.integration.sb3 import WandbCallback

from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.state.action_mask import get_action_mask
from EVRoutingEnv.state.gnn_utils import create_default_gnn_space


class MaskableEvalCallback(EvalCallback):
    """Custom EvalCallback that properly handles MaskablePPO with action masks.

    Note: Even though the eval_env is wrapped with ActionMasker, the standard
    evaluate_policy function doesn't automatically pass action masks to
    MaskablePPO.predict(). This callback explicitly retrieves masks from the
    ActionMasker wrapper and passes them to ensure only valid actions are selected.
    """

    def _evaluate_maskable_policy(self):
        """Evaluate MaskablePPO with explicit action mask passing."""
        episode_rewards = []
        episode_lengths = []

        for _ in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            done = False
            episode_reward = 0.0
            episode_length = 0

            while not done:
                # Get action masks from the environment
                # The eval_env is a VecEnv, so we need to get the first (and only) env
                env = self.eval_env.envs[0]

                # Get action masks from ActionMasker wrapper
                action_masks = env.action_masks()

                # Predict with action masks
                action, _ = self.model.predict(
                    obs, action_masks=action_masks, deterministic=self.deterministic
                )

                obs, reward, done, info = self.eval_env.step(action)
                episode_reward += reward[0]
                episode_length += 1

                if self.render:
                    self.eval_env.render()

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

        return episode_rewards, episode_lengths

    def _on_step(self) -> bool:
        """Override to use maskable evaluation for MaskablePPO."""
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Use custom maskable evaluation if the model is MaskablePPO
            if isinstance(self.model, MaskablePPO):
                episode_rewards, episode_lengths = self._evaluate_maskable_policy()
            else:
                # Standard evaluation for other algorithms
                from stable_baselines3.common.evaluation import (
                    evaluate_policy as sb3_evaluate_policy,
                )

                episode_rewards, episode_lengths = sb3_evaluate_policy(
                    self.model,
                    self.eval_env,
                    n_eval_episodes=self.n_eval_episodes,
                    render=self.render,
                    deterministic=self.deterministic,
                    return_episode_rewards=True,
                    warn=self.warn,
                    callback=self._log_success_callback,
                )

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(
                episode_lengths
            )
            self.last_mean_reward = mean_reward

            if self.verbose >= 1:
                print(
                    f"Eval num_timesteps={self.num_timesteps}, "
                    f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}"
                )
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")

            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record(
                "time/total_timesteps", self.num_timesteps, exclude="tensorboard"
            )
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(
                        os.path.join(self.best_model_save_path, "best_model")
                    )
                self.best_mean_reward = mean_reward
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training


def mask_fn(env) -> np.ndarray:
    """Return action mask for the current environment state.

    The env passed here may be wrapped (e.g., Monitor), so we need to
    unwrap it to get the base EventDrivenTruckEnv.
    """
    # Unwrap to get the base environment
    base_env = env
    while hasattr(base_env, "env"):
        base_env = base_env.env
    return get_action_mask(base_env)


def make_env(config, seed: int, enable_plotting: bool = False, 
             use_detour: bool = False, vrp_top_k: int = 5,
             detour_top_k: int = 2, detour_hop_limit: int = 2):
    """Create and return environment wrapped with Monitor."""

    def _init():
        env = EventDrivenTruckEnv(
            config=config, verbose=False, enable_plotting=enable_plotting
        )
        # Set up GNN state space for consistent action masking
        mode = "vrp" if getattr(env, "enable_flexible_delivery_order", False) else "nonflex"
        gnn_space = create_default_gnn_space(
            env, 
            mode=mode, 
            use_detour=use_detour,
            device="cpu",
            vrp_top_k_deliveries=vrp_top_k,
            detour_num_chargers_to_keep=detour_top_k,
            detour_hop_limit=detour_hop_limit,
        )
        env._default_gnn_state_space = gnn_space
        env.use_detour_mask = use_detour
        env = Monitor(env)
        return env

    return _init


def make_masked_env(config, seed: int, enable_plotting: bool = False,
                    use_detour: bool = False, vrp_top_k: int = 5,
                    detour_top_k: int = 2, detour_hop_limit: int = 2):
    """Create and return environment wrapped for MaskablePPO with Monitor and ActionMasker."""

    def _init():
        env = EventDrivenTruckEnv(
            config=config, verbose=False, enable_plotting=enable_plotting
        )
        # Set up GNN state space for consistent action masking
        mode = "vrp" if getattr(env, "enable_flexible_delivery_order", False) else "nonflex"
        gnn_space = create_default_gnn_space(
            env, 
            mode=mode, 
            use_detour=use_detour,
            device="cpu",
            vrp_top_k_deliveries=vrp_top_k,
            detour_num_chargers_to_keep=detour_top_k,
            detour_hop_limit=detour_hop_limit,
        )
        env._default_gnn_state_space = gnn_space
        env.use_detour_mask = use_detour
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
    num_trucks: int | None = None,
    num_stops: int | None = None,
    use_detour: bool = False,
    vrp_top_k_deliveries: int = 5,
    detour_top_k_chargers: int = 2,
    detour_hop_limit: int = 2,
    **kwargs,
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
    from EVRoutingEnv.utils.utils import load_config

    config = load_config(config_path)
    if num_trucks is not None:
        config["environment"]["num_trucks"] = num_trucks
    if num_stops is not None:
        config["environment"]["num_stops"] = num_stops

    print(f"\n{'='*60}")
    print(f"Training Configuration")
    print(f"{'='*60}")
    print(f"Algorithm: {algo.upper()}")
    print(f"Seed: {seed}")
    print(f"Device: {device}")
    print(f"Total steps: {total_steps:,}")
    print(f"Config: {config_path}")
    print(f"Use Detour: {use_detour}")
    if use_detour:
        print(f"Detour Top-K Chargers: {detour_top_k_chargers}")
        print(f"Detour Hop Limit: {detour_hop_limit}")
    print(f"VRP Top-K Deliveries: {vrp_top_k_deliveries}")
    print(f"{'='*60}\n")

    # Initialize wandb with group name based on environment config
    run_name = f"{algo}_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    num_trucks = config["environment"]["num_trucks"]
    num_stops = config["environment"]["num_stops"]
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
                "use_detour": use_detour,
                "vrp_top_k_deliveries": vrp_top_k_deliveries,
                "detour_top_k_chargers": detour_top_k_chargers,
                "detour_hop_limit": detour_hop_limit,
            },
            sync_tensorboard=True,
            save_code=True,
        )

    # Create environments
    use_masked = algo == "maskppo"
    env_fn = make_masked_env if use_masked else make_env

    train_env = DummyVecEnv([env_fn(config, seed, False, use_detour, 
                                     vrp_top_k_deliveries, detour_top_k_chargers, 
                                     detour_hop_limit)])
    eval_env = DummyVecEnv([env_fn(config, seed + 100, False, use_detour,
                                    vrp_top_k_deliveries, detour_top_k_chargers,
                                    detour_hop_limit)])

    save_path = f"./saved_models/{group_name}/{run_name}/"
    os.makedirs(f"./saved_models/{group_name}", exist_ok=True)
    os.makedirs(save_path, exist_ok=True)

    # Setup evaluation callback - use custom callback for MaskablePPO
    if use_masked:
        eval_callback = MaskableEvalCallback(
            eval_env,
            best_model_save_path=save_path,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            render=False,
            verbose=1,
        )
    else:
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=save_path,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            render=False,
            verbose=1,
        )

    # Setup callbacks
    callbacks = [eval_callback]
    if use_wandb:
        callbacks.append(
            WandbCallback(
                verbose=2,
            )
        )

    # Create model based on algorithm
    model = None

    import tensorboard

    tb_log = f"./logs/{run_name}"

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
        raise ValueError(
            f"Unknown algorithm: {algo}. Supported: ppo, maskppo, dqn, qrdqn"
        )

    # Train the model
    print(f"\nStarting training for {total_steps:,} steps...")
    print(f"{'='*60}\n")

    model.learn(
        total_timesteps=total_steps,
        progress_bar=True,
        callback=callbacks,
    )
    
    model.save(f"{save_path}/last_model.zip")    
    model_path = f"{save_path}/last_model.zip"
    
    # Save GNN state space configuration for consistent evaluation
    import yaml
    gnn_config = {
        "gnn_state_space": {
            "vrp_top_k_deliveries": vrp_top_k_deliveries,
            "detour_top_k_chargers": detour_top_k_chargers,
            "detour_hop_limit": detour_hop_limit,
        }
    }
    config_save_path = f"{save_path}/config.yaml"
    with open(config_save_path, "w") as f:
        yaml.dump(gnn_config, f, default_flow_style=False)
    print(f"Saved GNN state space config to: {config_save_path}")

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
        default="EVRoutingEnv/config_files/config.yaml",
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
    parser.add_argument(
        "--num-trucks",
        type=int,
        default=None,
        help="Override num_trucks in the environment config",
    )
    parser.add_argument(
        "--num-stops",
        type=int,
        default=None,
        help="Override num_stops in the environment config",
    )
    parser.add_argument(
        "--use-detour",
        action="store_true",
        help="Enable detour-based action masking for sequential delivery",
    )
    parser.add_argument(
        "--vrp-top-k",
        type=int,
        default=5,
        help="Number of top-k deliveries to keep in VRP mode action space",
    )
    parser.add_argument(
        "--detour-top-k",
        type=int,
        default=2,
        help="Number of top-k chargers to keep in detour mode",
    )
    parser.add_argument(
        "--detour-hop-limit",
        type=int,
        default=2,
        help="Maximum charger hops after charging before forcing delivery in detour mode",
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
        num_trucks=args.num_trucks,
        num_stops=args.num_stops,
        use_detour=args.use_detour,
        vrp_top_k_deliveries=args.vrp_top_k,
        detour_top_k_chargers=args.detour_top_k,
        detour_hop_limit=args.detour_hop_limit,
    )

    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"Final model saved to: {model_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
