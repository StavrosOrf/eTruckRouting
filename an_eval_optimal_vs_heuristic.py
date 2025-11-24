"""Compare the Gurobi optimal plan against the built-in heuristic policy."""

from __future__ import annotations

import argparse
import copy
from typing import Dict, List

import numpy as np

from truck_env.baselines.heuristic_policy import HeuristicPolicy
from truck_env.models.event_driven_env import EventDrivenTruckEnv
from truck_env.optimization import GurobiTruckRoutingSolver
from truck_env.utils.utils import load_config


def _aggregate_truck_times(info: Dict) -> Dict[str, float]:
    """Extract total and maximum time spent across trucks from env info."""
    trucks = info.get("trucks", [])
    total_time = sum(t.get("total_time", 0.0) for t in trucks)
    makespan = max((t.get("total_time", 0.0) for t in trucks), default=0.0)
    total_distance = sum(t.get("total_distance", 0.0) for t in trucks)
    return {
        "total_time": total_time,
        "makespan": makespan,
        "total_distance": total_distance,
    }


def run_heuristic_episode(
    env: EventDrivenTruckEnv,
    policy: HeuristicPolicy,
    seed: int,
    max_steps: int,
) -> Dict[str, float]:
    """Roll out the heuristic policy on the environment."""
    env.reset(seed=seed)
    done = False
    truncated = False
    episode_reward = 0.0
    steps = 0

    while not (done or truncated) and steps < max_steps:
        action = policy.get_action(env)
        _, reward, done, truncated, info = env.step(action)
        episode_reward += reward
        steps += 1

    metrics = _aggregate_truck_times(info)
    metrics.update(
        {
            "reward": episode_reward,
            "success": float(info.get("all_complete", False)),
            "global_clock": info.get("global_clock", env.global_clock),
            "steps": steps,
        }
    )
    return metrics


def run_optimal_episode(
    env: EventDrivenTruckEnv,
    solver: GurobiTruckRoutingSolver,
    seed: int,
    max_steps: int,
) -> Dict[str, float]:
    """Roll out the optimal plan by querying the solver for actions."""
    env.reset(seed=seed)
    solver.reset_policy()
    done = False
    truncated = False
    episode_reward = 0.0
    steps = 0

    while not (done or truncated) and steps < max_steps:
        action = solver.get_action(env)
        _, reward, done, truncated, info = env.step(action)
        episode_reward += reward
        steps += 1

    metrics = _aggregate_truck_times(info)
    metrics.update(
        {
            "reward": episode_reward,
            "success": float(info.get("all_complete", False)),
            "global_clock": info.get("global_clock", env.global_clock),
            "steps": steps,
        }
    )
    return metrics


def summarize(values: List[float]) -> str:
    """Format mean ± std for printing."""
    if not values:
        return "n/a"
    arr = np.asarray(values, dtype=float)
    return f"{arr.mean():.2f} ± {arr.std(ddof=0):.2f}"


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate heuristic policy vs Gurobi optimal plan."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="truck_env/config_files/config.yaml",
        help="Path to the environment configuration file.",
    )
    parser.add_argument(
        "--num-trucks",
        type=int,
        default=10,
        help="Override number of trucks (optional).",
    )
    parser.add_argument(
        "--num-stops",
        type=int,
        default=3,
        help="Override number of delivery stops (optional).",
    )
    parser.add_argument(
        "--scenarios",
        type=int,
        default=1,
        help="Number of random scenarios to evaluate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Base random seed.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Safety cap on environment steps for heuristic rollout.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-scenario details.",
    )

    args = parser.parse_args()

    base_config = load_config(args.config)
    if args.num_trucks is not None:
        base_config["environment"]["num_trucks"] = args.num_trucks
    if args.num_stops is not None:
        base_config["environment"]["num_stops"] = args.num_stops

    heuristic_env = EventDrivenTruckEnv(
        config=copy.deepcopy(base_config), verbose=False, enable_plotting=False
    )
    optimal_env = EventDrivenTruckEnv(
        config=copy.deepcopy(base_config), verbose=False, enable_plotting=False
    )
    heuristic_policy = HeuristicPolicy(verbose=False)
    optimal_solver = GurobiTruckRoutingSolver(
        config_path=args.config, env=optimal_env, seed=args.seed, auto_reset=False
    )

    scenario_results: List[Dict[str, Dict]] = []

    for idx in range(args.scenarios):
        scenario_seed = args.seed + idx
        heuristic_stats = run_heuristic_episode(
            heuristic_env, heuristic_policy, scenario_seed, args.max_steps
        )
        optimal_stats = run_optimal_episode(
            optimal_env, optimal_solver, scenario_seed, args.max_steps
        )
        scenario_results.append(
            {
                "seed": scenario_seed,
                "heuristic": heuristic_stats,
                "optimal": optimal_stats,
            }
        )

        if args.verbose:
            gap = heuristic_stats["total_time"] - optimal_stats["total_time"]
            print(
                f"[Scenario {idx+1}/{args.scenarios} | seed={scenario_seed}] "
                f"Heuristic total={heuristic_stats['total_time']:.2f}, "
                f"Optimal total={optimal_stats['total_time']:.2f}, "
                f"Gap={gap:.2f}"
            )

    heuristic_env.close()
    optimal_env.close()

    heur_times = [r["heuristic"]["total_time"] for r in scenario_results]
    heur_makespans = [r["heuristic"]["makespan"] for r in scenario_results]
    heur_success = [r["heuristic"]["success"] for r in scenario_results]
    heur_rewards = [r["heuristic"]["reward"] for r in scenario_results]

    opt_times = [r["optimal"]["total_time"] for r in scenario_results]
    opt_makespans = [r["optimal"]["makespan"] for r in scenario_results]
    opt_rewards = [r["optimal"]["reward"] for r in scenario_results]
    opt_success = [r["optimal"]["success"] for r in scenario_results]
    absolute_gaps = [h - o for h, o in zip(heur_times, opt_times)]
    relative_gaps = [
        (gap / o) * 100 if o > 1e-6 else 0.0 for gap, o in zip(absolute_gaps, opt_times)
    ]

    print("\n=== Optimal vs Heuristic Comparison ===")
    print(f"Scenarios evaluated: {args.scenarios}")
    print(f"Heuristic success rate: {np.mean(heur_success) * 100:.1f}%")
    print(f"Heuristic reward: {summarize(heur_rewards)}")
    print(f"Optimal success rate: {np.mean(opt_success) * 100:.1f}%")
    print(f"Optimal reward: {summarize(opt_rewards)}")
    print(f"Total time  - Heuristic: {summarize(heur_times)}")
    print(f"Total time  - Optimal  : {summarize(opt_times)}")
    print(f"Makespan    - Heuristic: {summarize(heur_makespans)}")
    print(f"Makespan    - Optimal  : {summarize(opt_makespans)}")
    print(f"Absolute gap (heuristic - optimal): {summarize(absolute_gaps)}")
    print(f"Relative gap %: {summarize(relative_gaps)}")

    worst = max(
        scenario_results,
        key=lambda r: r["heuristic"]["total_time"] - r["optimal"]["total_time"],
        default=None,
    )
    if worst:
        gap = worst["heuristic"]["total_time"] - worst["optimal"]["total_time"]
        print(
            f"\nWorst gap scenario seed={worst['seed']} "
            f"(gap={gap:.2f} hours, heuristic total={worst['heuristic']['total_time']:.2f}, "
            f"optimal total={worst['optimal']['total_time']:.2f})"
        )


if __name__ == "__main__":
    main()
