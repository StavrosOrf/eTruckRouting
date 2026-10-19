"""Measure completion-mask restrictions along new independent reference routes.

The references execute their ordinary actions; queried masks never repair their
choices. Only the already opened 128-case development bank is used. These are
new solver executions, not replays of the cached BKS trajectories.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import json
import multiprocessing
from pathlib import Path

import numpy as np
import yaml

from EVRoutingEnv.evaluation.metrics import extract_operational_metrics
from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.state.action_mask import policy_action_mask
from EVRoutingEnv.state.features import ACTION_FEATURES
from scripts.evaluation.run_canonical_campaign import build_policy


_ENV = None


def run_chunk(task):
    global _ENV
    directory, name, seeds = task
    root = Path(directory)
    if _ENV is None:
        config = yaml.safe_load((root / "config.yaml").read_text())
        _ENV = EventDrivenTruckEnv(config, verbose=False, enable_plotting=False)
    env = _ENV
    settings = json.loads((root / "baselines.json").read_text())[name]
    anchors = json.loads((root / "physical_scenarios.json").read_text())
    results = []
    for seed in seeds:
        policy = build_policy(name, settings)
        reset = getattr(policy, "reset", None)
        if callable(reset):
            reset()
        observation, info = env.reset(seed=seed)
        initial_scenario = deepcopy(info.get("scenario"))
        physical = {key: value for key, value in initial_scenario.items() if key != "config_sha256"}
        if physical != anchors[str(seed)]:
            raise ValueError(f"physical scenario changed: {name}, {seed}")
        terminated = truncated = False
        trace = []
        while not (terminated or truncated) and len(trace) < 10000:
            action = int(policy(env, observation, info))
            original = deepcopy(env.config["environment"])
            allowed = {}
            try:
                for stage in ("completion", "relaxed_capacity", "exact_capacity"):
                    env.config["environment"].update(
                        completion_shield=True,
                        completion_capacity_lookahead=stage != "completion",
                        completion_capacity_exact_lookahead=stage == "exact_capacity",
                        completion_work_escape_lookahead=False,
                    )
                    allowed[stage] = bool(policy_action_mask(env)[action])
            finally:
                # Preserve the reference's configuration and execution rules.
                env.config["environment"].clear()
                env.config["environment"].update(original)
            features = env.get_canonical_features().action_features[action]
            record = {
                "step": len(trace), "truck": env.active_truck_id,
                "action": action, "allowed": allowed,
                "kind_code": int(features[ACTION_FEATURES.index("kind_code")]),
                "target": int(features[ACTION_FEATURES.index("target_node")]),
                "execution_legal": bool(features[ACTION_FEATURES.index("feasible")]),
                "customers_served": sum(task.is_served for task in env.task_registry.tasks()),
            }
            trace.append(record)
            observation, _, terminated, truncated, info = env.step(action)
        metrics = extract_operational_metrics(env).as_dict()
        result = {
            "method": name, "scenario_seed": seed,
            "success": metrics["success"],
            "termination_reason": metrics["termination_reason"],
            "operating_hours": metrics["total_operating_time"],
            "decisions": len(trace),
            "physical_scenario_matches_cached_reference": True,
            "rejected_steps": {stage: [row["step"] for row in trace if not row["allowed"][stage]] for stage in ("completion", "relaxed_capacity", "exact_capacity")},
            "execution_illegal_steps": [row["step"] for row in trace if not row["execution_legal"]],
            "policy_diagnostics": policy.diagnostics() if hasattr(policy, "diagnostics") else {},
        }
        (root / "traces" / f"{name}_{seed}.json").write_text(
            json.dumps({**result, "trace": trace, "initial_scenario": initial_scenario}, indent=2) + "\n"
        )
        results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    root = args.directory.resolve()
    (root / "traces").mkdir(exist_ok=True)
    methods = ("alns_planned", "cpsat_planned")
    seeds = list(range(1000002000, 1000002128))
    tasks = [(str(root), name, seeds[start:start + 8]) for name in methods for start in range(0, len(seeds), 8)]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("fork")) as pool:
        for rows in pool.map(run_chunk, tasks):
            results.extend(rows)
            print(f"completed {len(results)}/{len(methods) * len(seeds)} reference episodes", flush=True)
    if {(row["method"], row["scenario_seed"]) for row in results} != {(name, seed) for name in methods for seed in seeds}:
        raise ValueError("incomplete reference-route coverage")
    summaries = {}
    for name in methods:
        rows = [row for row in results if row["method"] == name]
        successful = [row for row in rows if row["success"]]
        summaries[name] = {
            "episodes": len(rows), "successes": len(successful),
            "mean_successful_operating_hours": float(np.mean([row["operating_hours"] for row in successful])) if successful else None,
            "successful_routes_rejected": {stage: sum(bool(row["rejected_steps"][stage]) for row in successful) for stage in ("completion", "relaxed_capacity", "exact_capacity")},
            "successful_decisions_rejected": {stage: sum(len(row["rejected_steps"][stage]) for row in successful) for stage in ("completion", "relaxed_capacity", "exact_capacity")},
        }
    availability = {}
    for stage in ("completion", "relaxed_capacity", "exact_capacity"):
        paired = []
        for seed in seeds:
            successful = [row for row in results if row["scenario_seed"] == seed and row["success"]]
            admissible = [row for row in successful if not row["rejected_steps"][stage]]
            if admissible:
                unrestricted = min(row["operating_hours"] for row in successful)
                restricted = min(row["operating_hours"] for row in admissible)
                paired.append(100 * (restricted / unrestricted - 1))
        availability[stage] = {
            "scenarios_with_observed_admissible_successful_reference": len(paired),
            "mean_cost_increase_within_new_reference_pool_percent": float(np.mean(paired)) if paired else None,
            "maximum_cost_increase_within_new_reference_pool_percent": max(paired) if paired else None,
            "interpretation": "comparison of observed reference routes; not a restricted optimum or lower bound",
        }
    report = {
        "status": "completed", "role": "mask availability on new independent reference executions; reused development validation only",
        "scenarios": len(seeds), "reference_methods": methods, "summaries": summaries,
        "observed_reference_availability": availability,
        "final_validation_opened": False, "test_opened": False,
        "cached_BKS_trajectories_reproduced": False, "candidate_policy_weights_modified": False,
        "results": sorted(results, key=lambda row: (row["method"], row["scenario_seed"])),
    }
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
