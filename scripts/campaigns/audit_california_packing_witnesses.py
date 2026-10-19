"""Measure exact-packing mask behavior along feasible development witnesses."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import yaml

from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.state.action_mask import (
    bounded_payload_packing,
    policy_action_mask,
    remaining_payload_can_fit,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    root = args.directory.resolve()
    config = yaml.safe_load((root / "config.yaml").read_text())
    work_escape = json.loads((root / "manifest.json").read_text()).get("work_escape", False)
    config["environment"].update(completion_shield=True, completion_capacity_lookahead=True)
    results = []
    mask_seconds = 0.0
    decisions = 0
    for seed in range(1000002000, 1000002128):
        env = EventDrivenTruckEnv(config, verbose=False, enable_plotting=False)
        try:
            env.reset(seed=seed)
            extra = []
            baseline_rejections = 0
            success = False
            for index, (truck, action) in enumerate(list(env.feasibility_witness)):
                if env.active_truck_id != truck:
                    raise ValueError("witness event order differs")
                env.config["environment"]["completion_capacity_exact_lookahead"] = work_escape
                env.config["environment"]["completion_work_escape_lookahead"] = False
                relaxed = policy_action_mask(env)
                env.config["environment"]["completion_capacity_exact_lookahead"] = True
                env.config["environment"]["completion_work_escape_lookahead"] = work_escape
                started = time.perf_counter()
                exact = policy_action_mask(env)
                mask_seconds += time.perf_counter() - started
                decisions += 1
                baseline_rejections += int(not relaxed[action])
                if relaxed[action] and not exact[action]:
                    extra.append(index)
                _, _, _, _, info = env.step(action)
                success = bool(info.get("successful", False))
            results.append({"seed": seed, "witness_success": success, "newly_rejected_steps": extra, "relaxed_mask_rejected_steps": baseline_rejections})
        finally:
            env.close()
    artifact = {"role": "offline witness-preservation audit; witnesses are never policy inputs", "scenarios": len(results), "decisions": decisions, "mean_exact_mask_seconds": mask_seconds / max(1, decisions), "results": results}
    (root / "witness_audit.json").write_text(json.dumps(artifact, indent=2, allow_nan=False) + "\n")
    if any(not result["witness_success"] or result["newly_rejected_steps"] for result in results):
        raise ValueError("the exact restriction rejected a witnessed feasible route")
    snapshot = json.loads((root / "counterexample.json").read_text())[0]["last_customer_state"]
    tasks = [SimpleNamespace(**task) for task in snapshot["tasks"] if task["status"] == "unassigned"]
    payloads = []
    for truck in snapshot["trucks"]:
        if truck["complete"] or truck["failed"] or truck["returning"]:
            continue
        committed = sum(task["demand"] for task in snapshot["tasks"] if task["status"] != "unassigned" and task["claimed_by"] == truck["id"])
        payloads.append((truck["id"], max(0.0, truck["payload"] - committed)))
    counterexample = {"payloads": payloads, "demands": [task.demand for task in tasks], "relaxed_accepts": remaining_payload_can_fit(tasks, payloads), "bounded_exact_result": bounded_payload_packing(tasks, payloads)}
    (root / "packing_counterexample_audit.json").write_text(json.dumps(counterexample, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"preserved_witnesses": len(results), "mean_exact_mask_seconds": artifact["mean_exact_mask_seconds"], "counterexample": counterexample}), flush=True)


if __name__ == "__main__":
    main()
