"""Validate the branch-and-price solver against brute force and the environment.

For every (config, trucks, customers, seed) instance:
* the Julia solver must report OPTIMAL;
* its objective must equal an independent brute-force optimum (small sizes);
* its plan, replayed on the environment with noise off, must succeed with the
  same total travel time;
* its bound may never exceed the travel time of the env's own certified
  witness plan.

    PYTHONPATH=. python scripts/analysis/validate_branch_and_price.py \\
        --out results/branch_and_price/validation.json --seeds 10
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

from EVRoutingEnv.baselines.optimization.branch_and_price.brute_force import exact_optimum
from EVRoutingEnv.baselines.optimization.branch_and_price.export import write_instance
from EVRoutingEnv.baselines.optimization.branch_and_price.julia_bridge import (
    RUNNER,
    BPParameters,
    solution_from_certificate,
)
from EVRoutingEnv.baselines.optimization.branch_and_price.validate import nominal_replay
from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.utils.utils import load_config

JOINT = "EVRoutingEnv/config_files/config_joint.yaml"
WINDOWS = "EVRoutingEnv/config_files/config_joint_time_windows.yaml"


def _tight_windows(config: dict) -> None:
    config["problem"]["time_windows"].update(
        {"earliest_min": 0.0, "earliest_max": 30.0, "window_width_min": 4.0, "window_width_max": 12.0}
    )


def _slow_charging(config: dict) -> None:
    # Charging dominates the clock, so the charging-time relaxation can fail
    # and the exact time-aware phase has to decide.
    config["charging"]["station_power_classes_kw"] = [40.0, 60.0, 150.0]
    config["environment"]["max_time"] = 60.0


# name -> (base config, in-place override)
CONFIGS = {
    "default": (JOINT, None),
    "time_windows": (WINDOWS, None),
    "tight_windows": (WINDOWS, _tight_windows),
    "slow_charging": (JOINT, _slow_charging),
}


def make_env(config_name: str, trucks: int, customers: int) -> EventDrivenTruckEnv:
    path, override = CONFIGS[config_name]
    config = deepcopy(load_config(path))
    config["environment"].update(
        {"num_stops": customers, "num_trucks": trucks, "allow_variable_num_stops": False}
    )
    if override is not None:
        override(config)
    return EventDrivenTruckEnv(config, verbose=False, enable_plotting=False)


def witness_travel(instance: dict) -> float:
    index = {node["id"]: k for k, node in enumerate(instance["nodes"])}
    depot = instance["nodes"][0]["id"]
    total = 0.0
    for route in instance["warm_start"]["witness"]:
        here = depot
        for node, _, charge in route:
            if not charge and node != here:
                total += instance["time_h"][index[here]][index[node]]
                here = node
    return total


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    parser.add_argument("--sizes", nargs="+", default=["2x4", "2x5", "3x5", "2x6", "3x6", "3x7"])
    parser.add_argument("--brute-force-max", type=int, default=7, help="largest customer count to brute force")
    parser.add_argument("--time-limit", type=float, default=600.0)
    parser.add_argument("--solver-arg", nargs=2, action="append", default=[], metavar=("KEY", "VALUE"),
                        help="extra BPParameters field for the Julia solver, e.g. --solver-arg cuts_per_round 0")
    args = parser.parse_args(argv)

    work = args.out.parent / (args.out.stem + "_instances")
    work.mkdir(parents=True, exist_ok=True)
    cases, pairs = [], []
    for config_name in args.configs:
        for size in args.sizes:
            trucks, customers = (int(x) for x in size.split("x"))
            for seed in range(args.seeds):
                tag = f"{config_name}_{trucks}x{customers}_s{seed}"
                env = make_env(config_name, trucks, customers)
                try:
                    env.reset(seed=seed)
                except Exception as error:  # the generator found no certified instance
                    print(f"{tag:32s} skipped: {type(error).__name__}: {error}", flush=True)
                    continue
                instance = write_instance(env, work / f"{tag}.json")
                cases.append((tag, config_name, trucks, customers, seed, instance))
                pairs.append(f"{work / (tag + '.json')} {work / (tag + '.result.json')}")
    (work / "batch.txt").write_text("\n".join(pairs) + "\n")

    params = BPParameters(time_limit_seconds=args.time_limit, extra=dict(args.solver_arg))
    started = time.perf_counter()
    subprocess.run([str(RUNNER), "--batch", str(work / "batch.txt"), *params.cli()], check=True,
                   stdout=open(work / "solver.log", "w"), stderr=subprocess.STDOUT)
    solver_seconds = time.perf_counter() - started

    rows, failures = [], 0
    for tag, config_name, trucks, customers, seed, instance in cases:
        certificate = json.loads((work / f"{tag}.result.json").read_text())
        solution = solution_from_certificate(certificate, certificate["wall_seconds"])
        env = make_env(config_name, trucks, customers)
        env.reset(seed=seed)
        replay = nominal_replay(env, solution.routes)
        brute = exact_optimum(instance) if customers <= args.brute_force_max else None
        witness = witness_travel(instance)
        checks = {
            "proven_optimal": solution.proven_optimal,
            "replay_feasible": replay["feasible"],
            "replay_travel_matches": abs(replay["total_travel_hours"] - solution.objective_hours) <= 1e-6,
            "bound_below_witness": solution.best_bound_hours <= witness + 1e-6,
            "matches_brute_force": None if brute is None else abs(brute - solution.objective_hours) <= 1e-6,
        }
        ok = all(v for v in checks.values() if v is not None)
        failures += not ok
        rows.append({
            "instance": tag, "config": config_name, "trucks": trucks, "customers": customers, "seed": seed,
            "status": solution.status, "objective_hours": solution.objective_hours,
            "best_bound_hours": solution.best_bound_hours, "brute_force_hours": brute,
            "witness_hours": witness, "replay_travel_hours": replay["total_travel_hours"],
            "replay_queue_hours": replay["total_queue_hours"], "replay_reason": replay["reason"],
            "solver_seconds": certificate["wall_seconds"], "nodes": certificate["nodes_explored"],
            "routes": len(solution.routes), "checks": checks, "ok": ok,
        })
        print(f"{tag:32s} {solution.status:8s} bp={solution.objective_hours:10.4f} "
              f"brute={'-' if brute is None else f'{brute:10.4f}'} replay={replay['total_travel_hours']:10.4f} "
              f"{'OK' if ok else 'FAIL ' + str(checks)}", flush=True)

    summary = {
        "instances": len(rows), "failures": failures, "solver_wall_seconds": solver_seconds,
        "brute_force_compared": sum(r["brute_force_hours"] is not None for r in rows), "rows": rows,
    }
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"{len(rows) - failures}/{len(rows)} instances passed every check")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
