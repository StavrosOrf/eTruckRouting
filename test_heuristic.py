"""
Test script for the heuristic policy.

Demonstrates how the heuristic algorithm makes decisions to ensure trucks
complete all deliveries without getting stranded.
"""

import sys

sys.path.insert(0, "/home/sorfanouda/EVPR")

from truck_env.models.event_driven_env import EventDrivenTruckEnv
from truck_env.baselines.heuristic_policy import HeuristicPolicy
import numpy as np


def test_heuristic_policy(
    config_path: str, num_episodes: int = 3, verbose: bool = True
):
    """
    Test the heuristic policy across multiple episodes.

    Args:
        config_path: Path to config.yaml
        num_episodes: Number of episodes to run
        verbose: Print detailed information
    """
    # Create environment
    env = EventDrivenTruckEnv(
        config_path, verbose=False, enable_plotting=True, run_id="heuristic_test"
    )

    # Create heuristic policy
    policy = HeuristicPolicy(verbose=verbose)

    print(f"{'='*80}")
    print(f"Testing Heuristic Policy - {num_episodes} Episodes")
    print(f"{'='*80}\n")

    episode_results = []

    for episode in range(num_episodes):
        print(f"\n{'='*60}")
        print(f"Episode {episode + 1}/{num_episodes}")
        print(f"{'='*60}")

        obs, info = env.reset(seed=episode)
        done = False
        truncated = False
        step_count = 0
        total_reward = 0.0
        truck_failures = 0

        while not done and not truncated:
            # Get heuristic action with explanation and log it
            action, explanation = policy.get_action_with_explanations(env)

            if action is None:
                # No more deliveries
                break

            # Take step with heuristic action
            obs, reward, done, truncated, info = env.step(action)
            step_count += 1
            total_reward += reward

            # Track failures
            if any(truck.failed for truck in env.trucks):
                truck_failures = sum(1 for truck in env.trucks if truck.failed)

            # Print action information
            truck_id = info.get("active_truck_id")
            if verbose and step_count <= 10:  # Print first 10 steps
                action_str = env._action_to_string(action)
                print(f"Step {step_count}: Truck {truck_id} → {action_str} (reward: {reward:+.1f})")
                # Also print one-line explanation summary
                print(explanation.split("\n")[0])

            if done or truncated:
                break

        # Episode summary
        all_complete = all(truck.is_complete for truck in env.trucks)
        any_failed = any(truck.failed for truck in env.trucks)

        result = {
            "episode": episode + 1,
            "steps": step_count,
            "reward": total_reward,
            "all_complete": all_complete,
            "any_failed": any_failed,
            "failures": truck_failures,
            "time": env.global_clock,
        }
        episode_results.append(result)

        print(f"\nEpisode {episode + 1} Summary:")
        print(f"  - Steps: {step_count}")
        print(f"  - Total Reward: {total_reward:.2f}")
        print(f"  - Time: {env.global_clock:.2f} hours")
        print(f"  - All Deliveries Complete: {all_complete}")
        print(f"  - Any Truck Failed: {any_failed}")
        if any_failed:
            print(f"  - Number of Failures: {truck_failures}")

    # Print overall statistics
    print(f"\n{'='*60}")
    print(f"Overall Statistics ({num_episodes} episodes)")
    print(f"{'='*60}")

    avg_steps = np.mean([r["steps"] for r in episode_results])
    avg_reward = np.mean([r["reward"] for r in episode_results])
    avg_time = np.mean([r["time"] for r in episode_results])
    success_rate = sum(1 for r in episode_results if r["all_complete"]) / len(
        episode_results
    )
    failure_rate = sum(1 for r in episode_results if r["any_failed"]) / len(
        episode_results
    )

    print(f"  - Avg Steps: {avg_steps:.1f}")
    print(f"  - Avg Reward: {avg_reward:.2f}")
    print(f"  - Avg Time: {avg_time:.2f} hours")
    print(f"  - Success Rate: {success_rate:.1%}")
    print(f"  - Failure Rate: {failure_rate:.1%}")

    print(f"\n{'='*60}")
    print(f"Decision Statistics")
    print(f"{'='*60}")
    policy.print_statistics()

    return episode_results


def compare_heuristic_vs_random(
    config_path: str, num_episodes: int = 3, verbose: bool = False
):
    """
    Compare heuristic policy vs random policy.

    Args:
        config_path: Path to config.yaml
        num_episodes: Number of episodes to run
        verbose: Print detailed information
    """
    print(f"\n{'='*80}")
    print(f"Comparing Heuristic vs Random Policy")
    print(f"{'='*80}\n")

    # Test heuristic
    print("Testing HEURISTIC Policy...")
    heuristic_results = test_heuristic_policy(config_path, num_episodes, verbose=False)

    # Test random
    print("\n\nTesting RANDOM Policy...")
    env = EventDrivenTruckEnv(config_path, verbose=False, enable_plotting=False)

    random_results = []
    for episode in range(num_episodes):
        obs, info = env.reset(seed=episode)
        done = False
        truncated = False
        step_count = 0
        total_reward = 0.0

        while not done and not truncated:
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            step_count += 1
            total_reward += reward

            if done or truncated:
                break

        all_complete = all(truck.is_complete for truck in env.trucks)
        any_failed = any(truck.failed for truck in env.trucks)

        result = {
            "episode": episode + 1,
            "steps": step_count,
            "reward": total_reward,
            "all_complete": all_complete,
            "any_failed": any_failed,
            "time": env.global_clock,
        }
        random_results.append(result)

        print(
            f"Episode {episode + 1}: Steps={step_count}, Reward={total_reward:.2f}, Complete={all_complete}"
        )

    # Compare
    print(f"\n{'='*80}")
    print(f"COMPARISON")
    print(f"{'='*80}")

    h_avg_steps = np.mean([r["steps"] for r in heuristic_results])
    r_avg_steps = np.mean([r["steps"] for r in random_results])

    h_avg_reward = np.mean([r["reward"] for r in heuristic_results])
    r_avg_reward = np.mean([r["reward"] for r in random_results])

    h_success = sum(1 for r in heuristic_results if r["all_complete"]) / len(
        heuristic_results
    )
    r_success = sum(1 for r in random_results if r["all_complete"]) / len(
        random_results
    )

    print(f"\n{'Policy':<20} {'Avg Steps':<15} {'Avg Reward':<15} {'Success Rate':<15}")
    print(f"{'-'*65}")
    print(
        f"{'Heuristic':<20} {h_avg_steps:<15.1f} {h_avg_reward:<15.2f} {h_success:<15.1%}"
    )
    print(
        f"{'Random':<20} {r_avg_steps:<15.1f} {r_avg_reward:<15.2f} {r_success:<15.1%}"
    )

    print(f"\nImprovement (Heuristic vs Random):")
    print(f"  - Steps: {((r_avg_steps - h_avg_steps) / r_avg_steps):.1%} better")
    print(
        f"  - Reward: {((h_avg_reward - r_avg_reward) / abs(r_avg_reward)):.1%} better"
    )
    print(f"  - Success Rate: {(h_success - r_success):.1%} improvement")


if __name__ == "__main__":
    config_path = "/home/sorfanouda/EVPR/truck_env/config_files/config.yaml"

    # Test heuristic policy
    print("\n" + "=" * 80)
    print("HEURISTIC POLICY TEST")
    print("=" * 80)
    test_heuristic_policy(config_path, num_episodes=3, verbose=True)

    # Compare with random policy
    print("\n" + "=" * 80)
    print("COMPARISON TEST")
    print("=" * 80)
    compare_heuristic_vs_random(config_path, num_episodes=3, verbose=False)

    print("\n✓ Heuristic policy tests complete!")
