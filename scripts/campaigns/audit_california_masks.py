"""Inspect actual policy-mask changes along already opened failure traces."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import yaml

from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.state.action_mask import completion_action_mask, get_action_mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path)
    args = parser.parse_args()
    root = args.directory.resolve()
    results = []
    for filename in ("bc_progress_s0_1000002075.json", "bc_progress_s1_1000002060.json", "bc_progress_s2_1000002009.json"):
        recorded = json.loads((root / "failure_traces" / filename).read_text())
        config = yaml.safe_load((root / "config.yaml").read_text())
        config["environment"]["completion_shield"] = True
        env = EventDrivenTruckEnv(config, verbose=False, enable_plotting=False)
        try:
            env.reset(seed=recorded["scenario_seed"])
            changes = []
            selected_pruned = []
            last_customer_state = None
            for step in recorded["trace"]:
                if not np.isclose(env.global_clock, step["time"]) or env.active_truck_id != step["truck"]:
                    raise ValueError("replay state differs")
                legal = get_action_mask(env)
                env.config["environment"]["completion_capacity_lookahead"] = False
                control = completion_action_mask(env, legal)
                env.config["environment"]["completion_capacity_lookahead"] = True
                treatment = completion_action_mask(env, legal)
                if not np.array_equal(control, treatment):
                    changes.append({"step": step["step"], "control_count": int(control.sum()), "treatment_count": int(treatment.sum()), "changed_actions": np.flatnonzero(control != treatment).tolist()})
                if control[step["action"]] and not treatment[step["action"]]:
                    selected_pruned.append(step["step"])
                if step["kind"] == "customer":
                    tasks = [{key: None if isinstance(value, float) and not math.isfinite(value) else value for key, value in task.as_dict().items()} for task in env.task_registry.tasks() if not task.is_served]
                    last_customer_state = {"step": step, "trucks": [{"id": truck.truck_id, "payload": truck.remaining_payload, "complete": truck.is_complete, "failed": truck.failed, "returning": getattr(truck, "return_to_depot_pending", False), "destination": truck.route_destination, "node": truck.current_node} for truck in env.trucks], "tasks": tasks, "control_mask": np.flatnonzero(control).tolist(), "capacity_mask": np.flatnonzero(treatment).tolist()}
                env.step(step["action"])
            results.append({"trace": filename, "changes": changes, "selected_actions_pruned": selected_pruned, "last_customer_state": last_customer_state})
        finally:
            env.close()
    (root / "mask_trajectory_audit.json").write_text(json.dumps(results, indent=2) + "\n")
    for result in results:
        print(result["trace"], "states_changed", len(result["changes"]), "selected_actions_pruned", result["selected_actions_pruned"])


if __name__ == "__main__":
    main()
