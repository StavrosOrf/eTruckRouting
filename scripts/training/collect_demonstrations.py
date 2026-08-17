"""Collect and cache demonstrator trajectories from the train split.

Rolling a deep-horizon controller is expensive, so demonstrations are gathered
once, in parallel, and cached to a compressed archive that every training run
reuses.  Only ``train`` seeds are ever rolled.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from algo.behavior_cloning import collect_demonstrations
from EVRoutingEnv.baselines.canonical_baselines import (
    GreedyHeuristicPolicy,
    HeuristicParameters,
    MPCParameters,
    RollingHorizonMPCPolicy,
)
from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.utils.utils import load_config
from scripts.evaluation.canonical_harness import split_seeds


def build_demonstrator(name: str):
    """Build one controller, or the oracle ensemble of all of them.

    ``ensemble`` solves each scenario with every controller and keeps the fastest
    successful trace, so the cloned teacher is at least as strong as its best
    member on every scenario.  It is far too slow to serve as a live baseline,
    which is exactly why it belongs offline as a teacher rather than in the
    comparison table.
    """
    if name == "heuristic":
        return GreedyHeuristicPolicy(
            HeuristicParameters(
                energy_safety_factor=1.15, target_soc=1.0, demand_weight=2.0
            )
        )
    if name == "mpc":
        return RollingHorizonMPCPolicy(
            MPCParameters(
                horizon=6, branching=2, energy_safety_factor=1.15, target_soc=0.8
            )
        )
    if name == "ensemble":
        from EVRoutingEnv.baselines.exact_optimization import (
            ExactPlannerParameters,
            MathematicalProgrammingPolicy,
        )

        return [
            build_demonstrator("mpc"),
            RollingHorizonMPCPolicy(
                MPCParameters(
                    horizon=8, branching=3, energy_safety_factor=1.05, target_soc=1.0
                )
            ),
            build_demonstrator("heuristic"),
            MathematicalProgrammingPolicy(
                ExactPlannerParameters(time_limit_seconds=8.0, workers=1)
            ),
        ]
    raise ValueError(f"unknown demonstrator {name!r}")


def _worker(payload: tuple[str, str, list[int], bool]) -> dict:
    config_path, demonstrator_name, seeds, successful_only = payload
    config = load_config(config_path)
    env = EventDrivenTruckEnv(config, verbose=False, enable_plotting=False)
    try:
        dataset = collect_demonstrations(
            env,
            build_demonstrator(demonstrator_name),
            seeds,
            successful_only=successful_only,
        )
    finally:
        env.close()
    return {
        "observations": (
            np.stack(dataset.observations)
            if dataset.observations
            else np.zeros((0, 0), dtype=np.float32)
        ),
        "actions": np.asarray(dataset.actions, dtype=np.int64),
        "masks": (
            np.stack(dataset.masks) if dataset.masks else np.zeros((0, 0), dtype=bool)
        ),
        "episodes": dataset.episodes,
        "successful_episodes": dataset.successful_episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="EVRoutingEnv/config_files/config_joint.yaml"
    )
    parser.add_argument(
        "--demonstrator", default="mpc", choices=["mpc", "heuristic", "ensemble"]
    )
    parser.add_argument("--scenarios", type=int, default=3000)
    parser.add_argument("--seed-offset", type=int, default=500_000)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--keep-failures", action="store_true")
    parser.add_argument("--output", default="results/canonical/demonstrations")
    arguments = parser.parse_args()

    # Demonstration seeds sit far from index 0 so they never collide with the
    # PPO rollout stream, which walks the train namespace from the start.
    seeds = split_seeds("train", arguments.seed_offset + arguments.scenarios)[
        arguments.seed_offset :
    ]
    chunk = max(1, len(seeds) // arguments.workers)
    batches = [seeds[index : index + chunk] for index in range(0, len(seeds), chunk)]
    payloads = [
        (arguments.config, arguments.demonstrator, batch, not arguments.keep_failures)
        for batch in batches
    ]

    started = time.perf_counter()
    print(
        f"collecting {len(seeds)} {arguments.demonstrator} demonstrations "
        f"across {len(batches)} workers...",
        flush=True,
    )
    results = []
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        for index, result in enumerate(pool.map(_worker, payloads), start=1):
            results.append(result)
            print(
                f"  batch {index}/{len(batches)}: "
                f"{result['successful_episodes']}/{result['episodes']} successful, "
                f"{len(result['actions'])} transitions",
                flush=True,
            )

    populated = [result for result in results if len(result["actions"]) > 0]
    if not populated:
        raise SystemExit("no demonstrations were collected")
    observations = np.concatenate([result["observations"] for result in populated])
    actions = np.concatenate([result["actions"] for result in populated])
    masks = np.concatenate([result["masks"] for result in populated])
    episodes = sum(result["episodes"] for result in results)
    successes = sum(result["successful_episodes"] for result in results)

    destination = Path(arguments.output)
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / f"{arguments.demonstrator}.npz"
    np.savez_compressed(
        archive, observations=observations, actions=actions, masks=masks
    )
    metadata = {
        "demonstrator": arguments.demonstrator,
        "config": arguments.config,
        "split": "train",
        "scenarios": len(seeds),
        "episodes": episodes,
        "successful_episodes": successes,
        "transitions": int(actions.shape[0]),
        "observation_width": int(observations.shape[1]),
        "action_width": int(masks.shape[1]),
        "keep_failures": bool(arguments.keep_failures),
        "seconds": time.perf_counter() - started,
    }
    (destination / f"{arguments.demonstrator}_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True)
    )
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)
    print(f"wrote {archive}", flush=True)


if __name__ == "__main__":
    main()
