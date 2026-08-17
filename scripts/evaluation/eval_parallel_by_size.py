"""Evaluate size-matched policies and write per-episode CSVs.

LEGACY, for the inherited preassigned-route problem only. Size transfer in the
revision is measured by the size_transfer regimes of
scripts/evaluation/run_generalization_campaign.py, which hold the action
envelope fixed so a trained policy stays applicable and publish the full
artifact contract.
"""

import csv
import os
import sys
from datetime import datetime

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

import eval_parallel_policies as epp

SIZE_GRID = [
    (1, 3),
    (5, 3),
    (10, 3),
    (30, 3),
    (50, 3),
    (100, 3),
]


def _size_label(num_trucks, num_stops):
    return f"{num_trucks}T{num_stops}S"


def _append_policy_if_missing(policies, entry):
    if entry not in policies:
        policies.append(entry)


def _policies_for_size(num_trucks, num_stops):
    matched = []
    for entry in epp.POLICIES:
        size = epp.parse_policy_size(entry[0])
        if size == (num_trucks, num_stops):
            matched.append(entry)

    _append_policy_if_missing(matched, ("heuristic", "heuristic", "base"))
    if num_trucks < 30:
        _append_policy_if_missing(matched, ("optimal", "optimal", "base"))
    else:
        _append_policy_if_missing(matched, ("optimal-simple", "optimal-simple", "base"))

    return matched


def _write_csv(rows, csv_path, fieldnames):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    config = epp.load_config(epp.CONFIG_FILE)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(project_root, "results", f"evaluation_{timestamp}")

    fieldnames = [
        "num_trucks",
        "num_stops",
        "policy_name",
        "policy_full_name",
        "policy_path",
        "policy_type",
        "gnn_space",
        "episode_idx",
        "seed",
        "reward",
        "success",
        "distance",
        "charging_time",
        "steps",
        "completion_time",
        "deliveries",
        "charging_sessions",
        "waiting_time",
        "routing_time",
        "unloading_time",
        "total_truck_time",
        "failures",
        "avg_completion_soc",
        "exec_time",
        "max_time_termination",
        "max_steps_termination",
        "vrp_feasible",
    ]

    for num_trucks, num_stops in SIZE_GRID:
        size_label = _size_label(num_trucks, num_stops)
        policy_entries = _policies_for_size(num_trucks, num_stops)
        if not policy_entries:
            print(f"No policies found for {size_label}; skipping.")
            continue

        print(f"\nRunning evaluation for {size_label}...")
        eval_output = epp.run_parallel_eval(
            policy_entries,
            config,
            num_trucks,
            num_stops,
            epp.NUM_EVAL_SCENARIOS,
            epp.SEED,
            epp.NUM_WORKERS,
            epp.GPU_DEVICES,
            auto_detect_sb3_config=False,
            print_summary=False,
        )

        excluded_indices = eval_output["excluded_indices"]
        policies = eval_output["policies"]
        policy_full_names = eval_output["policy_full_names"]
        episode_results_by_policy = eval_output["episode_results_by_policy"]

        rows = []
        for policy_name in sorted(episode_results_by_policy.keys()):
            policy_info = policies[policy_name]
            full_name = policy_full_names.get(policy_name, policy_name)
            for result in episode_results_by_policy[policy_name]:
                if result is None:
                    continue
                episode_idx = result["episode_idx"]
                if episode_idx in excluded_indices:
                    continue
                rows.append(
                    {
                        "num_trucks": num_trucks,
                        "num_stops": num_stops,
                        "policy_name": policy_name,
                        "policy_full_name": full_name,
                        "policy_path": policy_info["path"],
                        "policy_type": policy_info["type"],
                        "gnn_space": policy_info["gnn_space"],
                        "episode_idx": episode_idx,
                        "seed": epp.SEED + episode_idx,
                        "reward": result["reward"],
                        "success": result["success"],
                        "distance": result["distance"],
                        "charging_time": result["charging_time"],
                        "steps": result["steps"],
                        "completion_time": result["completion_time"],
                        "deliveries": result["deliveries"],
                        "charging_sessions": result["charging_sessions"],
                        "waiting_time": result["waiting_time"],
                        "routing_time": result["routing_time"],
                        "unloading_time": result["unloading_time"],
                        "total_truck_time": result["total_truck_time"],
                        "failures": result["failures"],
                        "avg_completion_soc": result["avg_completion_soc"],
                        "exec_time": result["exec_time"],
                        "max_time_termination": result["max_time_termination"],
                        "max_steps_termination": result["max_steps_termination"],
                        "vrp_feasible": result["vrp_feasible"],
                    }
                )

        csv_path = os.path.join(output_dir, f"eval_{size_label}.csv")
        _write_csv(rows, csv_path, fieldnames)
        print(f"Wrote {len(rows)} rows to {csv_path}")

    print(f"All results saved under {output_dir}")


if __name__ == "__main__":
    main()
