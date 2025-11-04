import torch
import gymnasium as gym
import numpy as np
import os
import sys
import signal
import pprint
import pickle
import traceback

# Set environment variable to avoid GPU warning
os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from gymnasium.spaces.utils import flatten_space, flatten
from ray.rllib.algorithms.callbacks import DefaultCallbacks

# Add current directory to Python path for Ray workers
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from truck_env.truck_env import HierarchicalTruckRoutingEnv
from truck_env.utils import (
    get_graph,
    get_observation_space,
    get_high_level_action_space,
    get_low_level_action_space,
    GRAPH_MAPPINGS,
)

# =============================================================================
# CONFIGURATION FUNCTIONS (Abstracted as requested)
# =============================================================================


class DebugCallback(DefaultCallbacks):
    def on_episode_end(self, *, episode, **kwargs):
        # EpisodeV2 doesn't have get_rewards(), use agent_rewards instead
        if hasattr(episode, 'agent_rewards'):
            rewards = episode.agent_rewards
        else:
            # Fallback for older API
            rewards = {}
        
        total_reward = sum(sum(r) if isinstance(r, list) else r for r in rewards.values()) if rewards else 0.0

        print("📊 Episode ended.")
        for aid, r in rewards.items():
            total = sum(r) if isinstance(r, list) else r
            print(f"  - {aid}: {total:.2f}")
        print(f"💰 Total reward: {total_reward:.2f}")

        # Store in custom_data
        if hasattr(episode, 'custom_data'):
            episode.custom_data["total_reward"] = total_reward


# =============================================================================
# POLICY CONFIGURATIONS
# =============================================================================


def get_high_level_policy_config():
    """Configuration for route planning policy."""
    return {
        "model": {
            "fcnet_hiddens": [128, 64, 32],
            "fcnet_activation": "relu"
        },
        "lr": 0.0005,
        "entropy_coeff": 0.02,
    }


def get_low_level_policy_config():
    """Configuration for charging management policy."""
    return {
        "model": {
            "fcnet_hiddens": [64, 32, 16],
            "fcnet_activation": "tanh"
        },
        "lr": 0.0003,
        "entropy_coeff": 0.01,
    }


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def policy_mapping_fn(agent_id, episode, **kwargs):
    """Map agents to their respective policies."""
    if agent_id.endswith("_route_planner"):
        return "high_level_policy"
    elif agent_id.endswith("_charge_manager"):
        return "low_level_policy"
    else:
        raise ValueError(f"Unknown agent_id: {agent_id}")


def create_hierarchical_truck_config():
    """Create RLlib configuration for hierarchical truck routing."""

    print("creating config")
    tune.register_env("hierarchical_truck_env",
                      lambda config: HierarchicalTruckRoutingEnv(config))
    raw_obs_space = get_observation_space(
        get_graph())  # , len(get_truck_configs()))
    flat_obs_space = flatten_space(raw_obs_space)

    # Use new API stack configuration
    config = (
        PPOConfig()
        .environment(
            "hierarchical_truck_env",
            env_config={}
        )
        .rl_module(
            model_config={
                "fcnet_hiddens": [128, 64, 32],
                "fcnet_activation": "relu",
            }
        )
        .multi_agent(
            policies={
                "high_level_policy": (
                    None,
                    flat_obs_space,
                    get_high_level_action_space(get_graph()),
                    get_high_level_policy_config()
                ),
                "low_level_policy": (
                    None,
                    flat_obs_space,
                    get_low_level_action_space(),
                    get_low_level_policy_config()
                ),
            },
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=["high_level_policy", "low_level_policy"],
        )
        .env_runners(
            # Reduce fragment length
            rollout_fragment_length=10,
            # Number of parallel rollout workers (processes)
            num_env_runners=2,
            # Number of envs created per rollout worker
            num_envs_per_env_runner=1,
        )
        .training(
            # RLlib API uses these parameter names in this version
            train_batch_size=2000,
            minibatch_size=256,
            num_epochs=10,  # Changed from num_sgd_iter
            lr=0.0003,
            entropy_coeff=0.01,
        )
        .callbacks(DebugCallback)
        .framework("torch")
    )
    print("end of creating config")
    return config


def eval(checkpoint_path):
    if not ray.is_initialized():
        ray.init(
            num_cpus=1, 
            _temp_dir="/tmp/ray1",
            address=None, 
            object_store_memory=10**9,
        )
    try:
        # Create config
        config = create_hierarchical_truck_config()

        # Build algo object
        algo = config.build()

        # Restore weights
        algo.restore(checkpoint_path)
        print("\nStarting Evaluation...")

        # Create a new environment instance for evaluation
        eval_env = HierarchicalTruckRoutingEnv()

        # Use the trained algorithm for evaluation
        for ep in range(3):
            # Initialize episode
            episode_rewards = {agent: 0 for agent in eval_env.possible_agents}
            obs, _ = eval_env.reset()
            terminated = {"__all__": False}

            print(f"\n=== Evaluation Episode {ep+1} ===")

            # Track routes
            for truck in eval_env.trucks:
                truck['route'] = [truck['current_node']]  # Start route

            # Run episode
            while not terminated["__all__"]:
                actions = {}
                for agent_id in eval_env.agents:
                    policy_id = policy_mapping_fn(agent_id, None)

                    # Get action from trained policy
                    module = algo.get_module(policy_id)
                    obs_tensor = torch.tensor(
                        [obs[agent_id]], dtype=torch.float32)

                    action_out = module.forward_inference({
                        "obs": obs_tensor
                    })
                    # print("action_out:", action_out , flush=True)
                    # action = action_out["actions"][0]
                    # action_out = module.forward_inference({
                    #    "obs": np.array([obs[agent_id]])
                    # })
                    logits = action_out["action_dist_inputs"]
                    action = torch.argmax(logits, dim=1)[0].item()
                    actions[agent_id] = action
                    # actions[agent_id] = action_out["actions"][0]
                    # actions[agent_id] = algo.compute_single_action(
                    #    obs[agent_id],
                    #    policy_id=policy_id,
                    #    explore=False  # Disable exploration for evaluation
                    # )

                # Execute actions
                obs, rewards, terminated, truncated, info = eval_env.step(
                    actions)

                # Accumulate rewards
                for agent_id, reward in rewards.items():
                    episode_rewards[agent_id] += reward

                # Update routes
                for i, truck in enumerate(eval_env.trucks):
                    if truck['current_node'] != truck['route'][-1]:
                        truck['route'].append(truck['current_node'])

            # Print results
            print(f"\nEpisode {ep+1} Results:")
            for i, truck in enumerate(eval_env.trucks):
                # Convert to OSM IDs if available
                if GRAPH_MAPPINGS["index_to_node"]:
                    route_osn = [GRAPH_MAPPINGS["index_to_node"][idx]
                                 for idx in truck['route']]
                else:
                    route_osn = truck['route']

                print(f"  Truck {i}:")
                print(f"    Route: {route_osn}")
                print(f"    Charging sessions: {truck['charging_sessions']}")
                print(f"    Total distance: {truck['total_distance']:.2f} km")
                print(
                    f"    Final battery: {truck['current_battery']:.2f}/{truck['battery_capacity']} kWh")
                print(f"    Time taken: {truck['time_elapsed']:.2f} hours")

            # Print rewards
            print("\nAgent Rewards:")
            for agent_id, reward in episode_rewards.items():
                print(f"  {agent_id}: {reward:.2f}")

        algo.stop()
    except Exception as e:
        print(f"Error occurred: {e}")
        traceback.print_exc()
        ray.shutdown()
        # Force termination if needed
        os.kill(os.getpid(), signal.SIGTERM)
    finally:
        if ray.is_initialized():
            ray.shutdown()


def main():
    """Main training loop."""
    if not ray.is_initialized():
        print("Initializing Ray...")
        ray.init(
            num_cpus=2, 
            _temp_dir="/tmp/ray1",
            address=None, 
            object_store_memory=10**9,
        )
        
    config = create_hierarchical_truck_config()
    
    # Use build_learner_group instead of build to avoid deprecation warning
    # Actually, just use build() - the warning is about internal implementation
    algo = config.build()

    print("\nStarting hierarchical truck routing training...")
    print("=" * 60)

    checkpoint_dir = './saved_models/'

    for iteration in range(30):
        result = algo.train()
        # pprint.pprint(result)
        print(f"Iteration {iteration + 1}:")
        print(
            f"  Episode Reward Mean: {result.get('env_runners', 'N/A').get('episode_return_mean', 'N/A')}")
        print(
            f"  High Level Policy Reward: {result.get('env_runners', 'N/A').get('module_episode_returns_mean', {}).get('high_level_policy', 'N/A')}")
        print(
            f"  Low Level Policy Reward: {result.get('env_runners', 'N/A').get('module_episode_returns_mean', {}).get('low_level_policy', 'N/A')}")
        print(f"  Time passed Total: {result.get('time_total_s', 'N/A')}")

        print("-" * 40)
    print("Training completed!")
    # Save checkpoint

    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = algo.save(checkpoint_dir)
    print(f"Saved checkpoint to: {checkpoint_path}")
    algo.stop()
    # except Exception as e:
    #     print(f"Error occurred: {e}")
    #     traceback.print_exc()
    #     ray.shutdown()
    #     # Force termination if needed
    #     os.kill(os.getpid(), signal.SIGTERM)
    # finally:
    #     if ray.is_initialized():
    #         ray.shutdown()


if __name__ == "__main__":
    main()
