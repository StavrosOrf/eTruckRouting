"""Trace known validation failures without looking at locked test scenarios."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from algo.canonical_policy import CanonicalActorCritic
from EVRoutingEnv.evaluation.metrics import extract_operational_metrics
from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.state.action_mask import policy_action_mask
from EVRoutingEnv.state.feasibility import ActionKind
from EVRoutingEnv.state.features import ACTION_FEATURES


def diagnose(root, label):
    torch.set_num_threads(1)
    config = yaml.safe_load((root / "configs/california_10t30d.yaml").read_text())
    config["environment"]["completion_shield"] = True
    kind_column = ACTION_FEATURES.index("kind_code")
    target_column = ACTION_FEATURES.index("target_node")
    output = root / "diagnostics" / label
    output.mkdir(parents=True, exist_ok=True)
    summary = []
    for name in ("shield_s0", "links_shield_s0"):
        policy = CanonicalActorCritic.load(
            root / "evaluation/frozen_65k" / name, prefix="best"
        ).eval()
        path = (
            root / "evaluation/comparison_65k/validation" / name / "episode_rows.jsonl"
        )
        failures = [
            row
            for row in map(json.loads, path.read_text().splitlines())
            if not row["success"]
        ][:3]
        for failure in failures:
            env = EventDrivenTruckEnv(config, verbose=False, enable_plotting=False)
            observation, _ = env.reset(seed=failure["scenario_seed"])
            trace = []
            visited = {}
            progress = {}
            revisits = 0
            terminated = truncated = False
            try:
                while not (terminated or truncated):
                    mask = policy_action_mask(env)
                    truck = env.trucks[env.active_truck_id]
                    if progress.get(truck.truck_id) != truck.completed_service_count:
                        progress[truck.truck_id] = truck.completed_service_count
                        visited[truck.truck_id] = set()
                    with torch.no_grad():
                        result = policy(
                            torch.as_tensor(observation).unsqueeze(0),
                            torch.as_tensor(mask).unsqueeze(0),
                        )
                        probabilities = result.distribution().probs[0].numpy()
                    action = int(np.argmax(result.logits[0].numpy()))
                    features = env.get_canonical_features().action_features
                    selected = features[action]
                    kind = list(ActionKind)[int(selected[kind_column])]
                    target = int(selected[target_column])
                    if kind is ActionKind.CHARGER:
                        revisits += int(target in visited[truck.truck_id])
                        visited[truck.truck_id].add(target)
                    trace.append(
                        {
                            "truck": truck.truck_id,
                            "time": env.global_clock,
                            "kind": str(kind),
                            "target": target,
                            "battery": truck.current_battery,
                            "payload": truck.remaining_payload,
                            "customers_served": sum(
                                task.is_served for task in env.task_registry.tasks()
                            ),
                            "probability_by_kind": {
                                str(value): float(
                                    probabilities[
                                        features[:, kind_column] == index
                                    ].sum()
                                )
                                for index, value in enumerate(ActionKind)
                            },
                        }
                    )
                    observation, _, terminated, truncated, _ = env.step(action)
                metrics = extract_operational_metrics(env).as_dict()
                item = {
                    "method": name,
                    "seed": failure["scenario_seed"],
                    "original_failure": failure["termination_reason"],
                    "success": metrics["success"],
                    "termination_reason": metrics["termination_reason"],
                    "operating_hours": metrics["total_operating_time"],
                    "charger_revisits_without_service": revisits,
                    "decisions": len(trace),
                    "remaining_tasks": [
                        {
                            "node": task.node_id,
                            "demand": task.demand,
                            "status": str(task.status),
                            "claimed_by": task.claimed_by,
                        }
                        for task in env.task_registry.tasks()
                        if not task.is_served
                    ],
                    "trucks": [
                        {
                            "id": truck.truck_id,
                            "payload": truck.remaining_payload,
                            "battery": truck.current_battery,
                            "node": truck.current_node,
                            "destination": truck.route_destination,
                            "retired": truck.is_complete,
                            "failed": truck.failed,
                        }
                        for truck in env.trucks
                    ],
                }
                summary.append(item)
                (output / f"{name}_{failure['scenario_seed']}.json").write_text(
                    json.dumps({**item, "trace_last100": trace[-100:]}, indent=2) + "\n"
                )
                print(json.dumps(item), flush=True)
            finally:
                env.close()
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--source-label", required=True)
    args = parser.parse_args()
    diagnose(args.directory.resolve(), args.source_label)


if __name__ == "__main__":
    main()
