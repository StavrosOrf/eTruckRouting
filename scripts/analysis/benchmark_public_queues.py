"""Wall time of greedy episodes with and without outside traffic (plan 5.4).

Run from the repository root:
    PYTHONPATH=. python scripts/analysis/benchmark_public_queues.py --seeds 5

Target: with 10 trucks on Texas long-haul at u = 0.90 and all public stations,
the median episode time is at most 1.5x the same episodes without outside
traffic. Prints one JSON row per variant.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import statistics
import time


os.environ.setdefault("MPLCONFIGDIR", "/tmp/evrp_matplotlib")

from EVRoutingEnv.baselines.heuristics.canonical_baselines import GreedyHeuristicPolicy
from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.utils.utils import load_config


VARIANTS = {
    "no_outside_traffic": None,
    "kullman_on_arrival": {"information": "on_arrival"},
    "kullman_realtime": {"information": "realtime_queue"},
    "kullman_on_arrival_outages": {"information": "on_arrival", "outages": {"enabled": True}},
}


def run(config_path: str, variant: dict | None, seeds: list[int], trucks: int) -> dict:
    config = copy.deepcopy(load_config(config_path))
    config["environment"]["num_trucks"] = trucks
    if variant is not None:
        config["public_charging"] = {
            "enabled": True,
            "exogenous_traffic": {"utilization": 0.9},
            **copy.deepcopy(variant),
        }
    env = EventDrivenTruckEnv(config, verbose=False, enable_plotting=False)
    policy = GreedyHeuristicPolicy()
    times, steps, outcomes = [], [], []
    for seed in seeds:
        started = time.perf_counter()
        observation, info = env.reset(seed=seed)
        done, count = False, 0
        while not done:
            observation, _, terminated, truncated, info = env.step(policy(env, observation, info))
            done = terminated or truncated
            count += 1
        times.append(time.perf_counter() - started)
        steps.append(count)
        outcomes.append(info["termination_reason"])
    return {"median_seconds": statistics.median(times), "seconds": times, "steps": steps,
            "outcomes": outcomes}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="EVRoutingEnv/config_files/config_joint_texas_longhaul.yaml")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--trucks", type=int, default=10)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    args = parser.parse_args()
    seeds = list(range(args.seeds))
    baseline = None
    for name in args.variants:
        row = run(args.config, VARIANTS[name], seeds, args.trucks)
        if name == "no_outside_traffic":
            baseline = row["median_seconds"]
        if baseline:
            row["ratio_to_no_outside_traffic"] = row["median_seconds"] / baseline
        print(json.dumps({"variant": name, **row}), flush=True)


if __name__ == "__main__":
    main()
