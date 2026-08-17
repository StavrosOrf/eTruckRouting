"""Measure the achievable ceiling on a scenario split.

Answers the question that decides whether a success-rate target is meaningful:
does every generated instance admit *any* complete feasible plan?

For each scenario the nominal CP-SAT reference is solved.  Because that model
relaxes the simulator (nominal draws, linear charging, no queues, no step
limit), its verdicts bound reality:

* ``INFEASIBLE`` proves no policy can ever succeed on that instance -- but only
  when ``--max-chargers`` is *not* used, since restricting the station set makes
  the model a restriction rather than a relaxation;
* a feasible solution proves only that the *relaxed* problem is solvable, which
  is necessary but not sufficient for the stochastic problem;
* ``UNKNOWN`` proves nothing at all.

Measured limitation: on the primary configuration this formulation returns
UNKNOWN on most instances within 150 s and never proves optimality, leaving an
~85% gap between incumbent and bound. It is therefore **not usable as an
optimality denominator**; see ``build_best_known.py`` for the reference actually
used. What it does establish is that at least some generated instances are
provably infeasible, so 100% success is impossible by construction.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from EVRoutingEnv.baselines.optimality_reference import solve_reference
from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.utils.utils import load_config
from scripts.evaluation.canonical_harness import split_seeds


def _worker(payload: tuple) -> dict:
    config_path, seed, time_limit, positions, max_chargers = payload
    config = load_config(config_path)
    env = EventDrivenTruckEnv(config, verbose=False, enable_plotting=False)
    try:
        env.reset(seed=int(seed))
        solution = solve_reference(
            env,
            positions=positions,
            time_limit_seconds=time_limit,
            workers=1,
            max_chargers=max_chargers,
        )
    finally:
        env.close()
    record = solution.as_dict()
    record["scenario_seed"] = int(seed)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="EVRoutingEnv/config_files/config_joint.yaml")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--scenarios", type=int, default=30)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--time-limit", type=float, default=120.0)
    parser.add_argument("--positions", type=int, default=None)
    parser.add_argument("--max-chargers", type=int, default=None)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--output", default="results/canonical/ceiling")
    arguments = parser.parse_args()

    seeds = split_seeds(arguments.split, arguments.offset + arguments.scenarios)[
        arguments.offset :
    ]
    payloads = [
        (arguments.config, seed, arguments.time_limit, arguments.positions, arguments.max_chargers)
        for seed in seeds
    ]

    started = time.perf_counter()
    print(
        f"solving the nominal reference for {len(seeds)} {arguments.split} scenarios "
        f"across {arguments.workers} workers...",
        flush=True,
    )
    records = []
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        for index, record in enumerate(pool.map(_worker, payloads), start=1):
            records.append(record)
            print(
                f"  [{index}/{len(seeds)}] seed={record['scenario_seed']} "
                f"status={record['status']} "
                f"makespan={record['optimal_makespan']} "
                f"bound={record['best_bound']} "
                f"{record['wall_seconds']:.0f}s",
                flush=True,
            )

    feasible = sum(1 for r in records if r["feasible"])
    infeasible = sum(1 for r in records if r["proven_infeasible"])
    unknown = len(records) - feasible - infeasible
    proven_optimal = sum(1 for r in records if r["proven_optimal"])

    summary = {
        "split": arguments.split,
        "scenarios": len(records),
        "feasible": feasible,
        "proven_infeasible": infeasible,
        "unknown": unknown,
        "proven_optimal": proven_optimal,
        "feasible_fraction_upper_bound": feasible / max(len(records), 1),
        "note": (
            "The reference relaxes the simulator, so 'feasible' is necessary but "
            "not sufficient for stochastic success; the feasible fraction upper "
            "bounds any policy's achievable success rate. 'unknown' scenarios "
            "hit the solver time limit and are neither proven either way."
        ),
        "records": records,
        "seconds": time.perf_counter() - started,
    }
    destination = Path(arguments.output)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"{arguments.split}_ceiling.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True))

    print()
    print(f"feasible          : {feasible}/{len(records)}")
    print(f"proven infeasible : {infeasible}/{len(records)}")
    print(f"unknown (timeout) : {unknown}/{len(records)}")
    print(f"proven optimal    : {proven_optimal}/{len(records)}")
    print(f"upper bound on achievable success: {summary['feasible_fraction_upper_bound']:.3f}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
