"""Replay opened development-validation failures from frozen policy weights."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml

from algo.canonical_policy import CanonicalActorCritic
from EVRoutingEnv.evaluation.metrics import extract_operational_metrics
from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.state.action_mask import policy_action_mask
from EVRoutingEnv.state.features import ACTION_FEATURES
from EVRoutingEnv.state.feasibility import ActionKind


def trace(root, name, seed, output):
    methods = json.loads((root / "methods.json").read_text())
    settings = methods[name]
    config = yaml.safe_load((root / "config.yaml").read_text())
    for section, values in settings.get("environment_overrides", {}).items():
        config[section].update(values)
    policy = CanonicalActorCritic.load(settings["checkpoint"], prefix=settings.get("prefix", "best")).eval()
    env = EventDrivenTruckEnv(config, verbose=False, enable_plotting=False)
    observation, info = env.reset(seed=seed)
    records = []
    terminated = truncated = False
    kinds = list(ActionKind)
    kind_column = ACTION_FEATURES.index("kind_code")
    target_column = ACTION_FEATURES.index("target_node")
    charge_column = ACTION_FEATURES.index("charge_value")
    try:
        while not (terminated or truncated) and len(records) < 10000:
            mask = policy_action_mask(env)
            truck = env.trucks[env.active_truck_id]
            with torch.no_grad():
                result = policy(torch.as_tensor(observation).unsqueeze(0), torch.as_tensor(mask).unsqueeze(0))
                probabilities = result.distribution().probs[0].cpu().numpy()
            action = int(result.logits[0].argmax())
            features = env.get_canonical_features().action_features
            selected = features[action]
            records.append({
                "step": len(records), "truck": truck.truck_id, "time": env.global_clock,
                "node": truck.current_node, "battery": truck.current_battery,
                "payload": truck.remaining_payload, "action": action,
                "kind": kinds[int(selected[kind_column])].value,
                "target": int(selected[target_column]), "charge_value": float(selected[charge_column]),
                "customers_served": sum(task.is_served for task in env.task_registry.tasks()),
                "top_actions": [{"slot": int(index), "probability": float(probabilities[index]), "kind": kinds[int(features[index, kind_column])].value, "target": int(features[index, target_column])} for index in np.argsort(probabilities)[-5:][::-1] if probabilities[index] > 0],
            })
            observation, _, terminated, truncated, info = env.step(action)
        metrics = extract_operational_metrics(env).as_dict()
        summary = {
            "method": name, "scenario_seed": seed, "success": metrics["success"],
            "termination_reason": metrics["termination_reason"],
            "operating_hours": metrics["total_operating_time"], "decisions": len(records),
            "action_counts": dict(Counter(record["kind"] for record in records)),
            "remaining_tasks": [{"node": task.node_id, "demand": task.demand, "status": str(task.status), "claimed_by": task.claimed_by} for task in env.task_registry.tasks() if not task.is_served],
            "trucks": [{"id": truck.truck_id, "payload": truck.remaining_payload, "battery": truck.current_battery, "node": truck.current_node, "destination": truck.route_destination, "retired": truck.is_complete, "failed": truck.failed} for truck in env.trucks],
        }
        output.mkdir(parents=True, exist_ok=True)
        (output / f"{name}_{seed}.json").write_text(json.dumps({**summary, "trace": records}, indent=2) + "\n")
        print(json.dumps(summary), flush=True)
        return summary
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--failures-per-method", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(1)
    root = args.directory.resolve()
    summaries = []
    for name in args.methods:
        rows = []
        for path in sorted((root / "shards").glob(f"*/validation/{name}/episode_rows.jsonl")):
            rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
        failures = sorted((row for row in rows if not row["success"]), key=lambda row: row["scenario_seed"])
        for row in failures[:args.failures_per_method]:
            result = trace(root, name, row["scenario_seed"], root / "failure_traces")
            if result["success"] != row["success"] or not np.isclose(result["operating_hours"], row["total_operating_time"]):
                raise ValueError(f"frozen replay differs from evaluation: {name}, {row['scenario_seed']}")
            summaries.append(result)
    (root / "failure_traces/summary.json").write_text(json.dumps(summaries, indent=2) + "\n")


if __name__ == "__main__":
    main()
