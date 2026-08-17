"""Quality against compute for the two search baselines.

E3 asks for any scalability claim to be supported by quality-versus-runtime
curves, and the execution checklist asks for an optimizer-budget sensitivity.
Both baselines expose a budget knob -- the CP-SAT time limit and the ALNS
iteration count -- so this sweeps them on held-out scenarios and reports what
each additional second buys.

The learned policy has no comparable knob: its cost per decision is fixed by the
network, which is the point of the comparison rather than an omission. Its
measured per-decision time is reported alongside for reference.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from scripts.evaluation.canonical_harness import split_seeds, summarize


def _run(job: tuple) -> dict:
    from EVRoutingEnv.baselines.alns import ALNSParameters, ALNSPolicy
    from EVRoutingEnv.baselines.exact_optimization import (
        ExactPlannerParameters,
        MathematicalProgrammingPolicy,
    )
    from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
    from EVRoutingEnv.utils.utils import load_config
    from scripts.evaluation.canonical_harness import evaluate_policy

    method, budget, config_path, seeds, frozen = job
    if method == "cpsat":
        parameters = ExactPlannerParameters(
            time_limit_seconds=float(budget),
            workers=1,
            objective=frozen["objective"],
            average_charging_power_kw=frozen["average_charging_power_kw"],
            energy_safety_factor=frozen["energy_safety_factor"],
        )
        policy = MathematicalProgrammingPolicy(parameters)
    else:
        parameters = ALNSParameters(
            iterations=int(budget),
            time_limit_seconds=600.0,
            objective=frozen["objective"],
            energy_safety_factor=frozen["energy_safety_factor"],
            target_soc=frozen["target_soc"],
        )
        policy = ALNSPolicy(parameters)

    environment = EventDrivenTruckEnv(
        load_config(config_path), verbose=False, enable_plotting=False
    )
    started = time.perf_counter()
    try:
        outcomes = evaluate_policy(environment, policy, seeds)
    finally:
        environment.close()
    elapsed = time.perf_counter() - started
    summary = summarize(outcomes)
    return {
        "method": method,
        "budget": budget,
        "success_rate": summary["success_rate"],
        "travel_hours_successful": summary["mean_travel_time_successful"],
        "seconds_per_episode": elapsed / len(seeds),
        "episodes": len(seeds),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="EVRoutingEnv/config_files/config_joint.yaml"
    )
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    parser.add_argument("--scenarios", type=int, default=60)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--frozen", default="results/canonical/frozen_baselines_revision.json"
    )
    parser.add_argument(
        "--output", default="results/canonical/optimizer_budget/sweep.json"
    )
    arguments = parser.parse_args()

    frozen = json.loads(Path(arguments.frozen).read_text())
    cpsat = frozen["cpsat"]["parameters"]
    alns = frozen["alns"]["parameters"]
    seeds = split_seeds(arguments.split, arguments.scenarios)

    jobs = [
        (
            "cpsat",
            budget,
            arguments.config,
            seeds,
            {
                "objective": cpsat["objective"],
                "average_charging_power_kw": cpsat["average_charging_power_kw"],
                "energy_safety_factor": cpsat["energy_safety_factor"],
            },
        )
        for budget in (0.5, 1.0, 2.0, 5.0, 15.0, 45.0)
    ] + [
        (
            "alns",
            budget,
            arguments.config,
            seeds,
            {
                "objective": alns["objective"],
                "energy_safety_factor": alns["energy_safety_factor"],
                "target_soc": alns["target_soc"],
            },
        )
        for budget in (100, 500, 2000, 10000, 50000)
    ]

    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        rows = list(pool.map(_run, jobs))

    for row in sorted(rows, key=lambda item: (item["method"], item["budget"])):
        travel = row["travel_hours_successful"]
        print(
            f"  {row['method']:<6} budget={row['budget']:>8} "
            f"success={row['success_rate']:.3f} "
            f"travel={'n/a' if travel is None else f'{travel:.1f}'} "
            f"{row['seconds_per_episode']:.2f}s/episode",
            flush=True,
        )

    destination = Path(arguments.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {"split": arguments.split, "scenario_seeds": seeds, "rows": rows},
            indent=2,
            sort_keys=True,
        )
    )
    print(f"wrote {destination}", flush=True)


if __name__ == "__main__":
    main()
