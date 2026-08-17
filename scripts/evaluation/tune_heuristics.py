"""Grid-search the non-learning baselines on the validation split only.

A weak, untuned baseline would flatter the learned policy, so the heuristic, the
CP-SAT planner, and the MPC controller each get an explicit hyperparameter
search under the same selection rule used for the learned architectures.
Results are written to disk so the chosen settings can be frozen before any test
scenario is touched.

The search is run under the campaign objective (``--objective``, fleet travel
hours by default). That matters beyond the ranking: the CP-SAT grid includes the
nominal objective itself, so the planner is allowed to minimize total route time
rather than makespan when that is what the campaign reports. Comparing a
travel-time policy against a makespan-tuned planner would measure the objective
mismatch instead of the method.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

from scripts.evaluation.canonical_harness import (
    OBJECTIVE_KEYS,
    EpisodeOutcome,
    selection_score,
    split_seeds,
    summarize,
)


HEURISTIC_GRID = {
    "energy_safety_factor": [1.05, 1.15, 1.25, 1.4],
    "target_soc": [0.6, 0.8, 1.0],
    "demand_weight": [0.0, 0.5, 2.0],
}
CPSAT_GRID = {
    "energy_safety_factor": [1.05, 1.15, 1.25],
    "average_charging_power_kw": [150.0, 300.0, 600.0],
    "objective": ["makespan", "total_time"],
}
MPC_GRID = {
    "horizon": [3, 4, 6, 8],
    "branching": [2, 3, 4],
    "energy_safety_factor": [1.05, 1.15, 1.25],
    "target_soc": [0.8, 1.0],
}
# The search budget is deliberately part of the grid: a metaheuristic that is
# only allowed 500 iterations is a weak baseline by construction, and R1.6 asks
# for a strong one.  The nominal search costs ~0.1 s per 2000 iterations here,
# so the largest budget is still far cheaper than the CP-SAT time limit.
ALNS_GRID = {
    "iterations": [2000, 10000, 30000],
    "max_destroy_fraction": [0.3, 0.4],
    "energy_safety_factor": [1.05, 1.15, 1.25],
    "target_soc": [0.8, 1.0],
    "objective": ["total_time"],
}


def _combinations(grid: dict) -> list[dict]:
    keys = sorted(grid)
    return [
        dict(zip(keys, values, strict=True))
        for values in itertools.product(*(grid[key] for key in keys))
    ]


def _build(method: str, overrides: dict):
    """Instantiate one baseline at one grid point, inside the worker process."""
    from EVRoutingEnv.baselines.canonical_baselines import (
        GreedyHeuristicPolicy,
        HeuristicParameters,
        MPCParameters,
        RollingHorizonMPCPolicy,
    )

    if method == "heuristic":
        parameters = replace(HeuristicParameters(), **overrides)
        return GreedyHeuristicPolicy(parameters), parameters
    if method == "mpc":
        parameters = replace(MPCParameters(), **overrides)
        return RollingHorizonMPCPolicy(parameters), parameters
    if method == "cpsat":
        from EVRoutingEnv.baselines.exact_optimization import (
            ExactPlannerParameters,
            MathematicalProgrammingPolicy,
        )

        parameters = replace(
            ExactPlannerParameters(time_limit_seconds=15.0, workers=2), **overrides
        )
        return MathematicalProgrammingPolicy(parameters), parameters
    if method == "alns":
        from EVRoutingEnv.baselines.alns import ALNSParameters, ALNSPolicy

        parameters = replace(
            ALNSParameters(time_limit_seconds=15.0), **overrides
        )
        return ALNSPolicy(parameters), parameters
    raise ValueError(f"unknown method {method!r}")


def _evaluate_point(job: tuple[str, dict, str, list[int]]) -> dict:
    """Score one grid point on one shard of validation seeds."""
    from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
    from EVRoutingEnv.utils.utils import load_config
    from scripts.evaluation.canonical_harness import evaluate_policy

    method, overrides, config_path, seeds = job
    policy, parameters = _build(method, overrides)
    environment = EventDrivenTruckEnv(
        load_config(config_path), verbose=False, enable_plotting=False
    )
    started = time.perf_counter()
    try:
        outcomes = evaluate_policy(environment, policy, seeds)
    finally:
        environment.close()
    return {
        "key": json.dumps(overrides, sort_keys=True),
        "parameters": parameters.as_dict(),
        "rows": [outcome.as_dict() for outcome in outcomes],
        "seconds": time.perf_counter() - started,
    }


def tune(
    method: str,
    config_path: str,
    seeds: list[int],
    objective: str,
    workers: int,
    shards: int,
) -> list[dict]:
    """Rank one baseline's grid, sharding scenarios across worker processes."""
    grid = {
        "heuristic": HEURISTIC_GRID,
        "cpsat": CPSAT_GRID,
        "mpc": MPC_GRID,
        "alns": ALNS_GRID,
    }[method]
    points = _combinations(grid)
    shards = max(1, min(shards, len(seeds)))
    jobs = [
        (method, overrides, config_path, seeds[index::shards])
        for overrides in points
        for index in range(shards)
    ]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            completed = list(pool.map(_evaluate_point, jobs))
    else:
        completed = [_evaluate_point(job) for job in jobs]

    merged: dict[str, dict] = {}
    for part in completed:
        entry = merged.setdefault(
            part["key"],
            {"parameters": part["parameters"], "outcomes": [], "seconds": 0.0},
        )
        entry["outcomes"].extend(EpisodeOutcome(**row) for row in part["rows"])
        entry["seconds"] += part["seconds"]

    results = []
    for entry in merged.values():
        summary = summarize(entry["outcomes"])
        results.append(
            {
                "parameters": entry["parameters"],
                "summary": summary,
                "seconds": entry["seconds"],
            }
        )
    results.sort(key=lambda entry: selection_score(entry["summary"], objective))
    for entry in results:
        summary = entry["summary"]
        travel = summary["mean_travel_time_successful"]
        print(
            f"  {method} {entry['parameters']} -> "
            f"success={summary['success_rate']:.3f} "
            f"frac={summary['mean_completed_fraction']:.3f} "
            f"travel={'n/a' if travel is None else f'{travel:.1f}'}",
            flush=True,
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="EVRoutingEnv/config_files/config_joint.yaml"
    )
    parser.add_argument("--scenarios", type=int, default=30)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--output", default="results/canonical/tuning")
    parser.add_argument(
        "--methods", nargs="+", default=["heuristic", "mpc"], choices=[
            "heuristic", "mpc", "cpsat", "alns"
        ]
    )
    parser.add_argument(
        "--objective", default="travel_time", choices=sorted(OBJECTIVE_KEYS)
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--shards",
        type=int,
        default=4,
        help="Scenario shards per grid point; more shards, finer parallelism.",
    )
    arguments = parser.parse_args()

    seeds = split_seeds("validation", arguments.scenarios, arguments.base_seed)
    destination = Path(arguments.output)
    destination.mkdir(parents=True, exist_ok=True)

    for method in arguments.methods:
        print(f"Tuning {method} on the validation split...", flush=True)
        results = tune(
            method,
            arguments.config,
            seeds,
            arguments.objective,
            arguments.workers,
            arguments.shards,
        )
        _write(destination / f"{method}_tuning.json", results, seeds, arguments)
        print(f"Best {method}:", json.dumps(results[0], indent=2))


def _write(path: Path, results: list[dict], seeds: list[int], arguments) -> None:
    path.write_text(
        json.dumps(
            {
                "split": "validation",
                "config": arguments.config,
                "objective": arguments.objective,
                "scenario_seeds": seeds,
                "ranked_results": results,
                "best": results[0] if results else None,
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
