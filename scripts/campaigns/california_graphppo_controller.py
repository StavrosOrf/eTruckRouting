"""Promote completed GraphPPO waves on validation under a fixed compute envelope.

SLURM dependencies run this only after a training array finishes. It never waits
on a login node, consults test data, or declares near-optimality from training.
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
from pathlib import Path


def dump(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def rank_runs(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    groups = {}
    for run in manifest["runs"]:
        path = directory / "training" / run["name"] / "validation_history.json"
        if not path.exists():
            raise RuntimeError(f"missing validation history: {run['name']}")
        history = json.loads(path.read_text())
        if not history:
            raise RuntimeError(f"empty validation history: {run['name']}")
        # Match the trainer's feasibility-first checkpoint selection.
        best = min(
            history,
            key=lambda row: (
                -row["success_rate"],
                row.get("mean_operating_time_successful") or math.inf,
            ),
        )
        arm = run["name"].rsplit("_s", 1)[0]
        groups.setdefault(arm, []).append((run, best))
    ranked = []
    for arm, entries in groups.items():
        costs = [row["mean_operating_time_successful"] for _, row in entries]
        ranked.append(
            {
                "arm": arm,
                "runs": [run for run, _ in entries],
                "mean_success": sum(row["success_rate"] for _, row in entries)
                / len(entries),
                "mean_cost": sum(costs) / len(costs)
                if all(cost is not None for cost in costs)
                else None,
                "per_seed": [row for _, row in entries],
            }
        )
    return sorted(
        ranked, key=lambda row: (-row["mean_success"], row["mean_cost"] or math.inf)
    )


def schedule(root, dependency, current, next_budget):
    """One short coordinator allocation after each successful training wave."""
    manifest = json.loads((root / "manifest.json").read_text())
    controller = root / "controller.py"
    command = [
        manifest["python"],
        str(controller),
        "advance",
        "--directory",
        str(root),
        "--current",
        str(current),
        "--next-budget",
        str(next_budget),
    ]
    job = subprocess.check_output(
        [
            "sbatch",
            "--parsable",
            "--account=tdsei17344",
            "--partition=cbuild",
            "--nodes=1",
            "--ntasks=1",
            "--cpus-per-task=32",
            "--time=00:10:00",
            "--job-name=ca10t3d_promote",
            f"--dependency=afterok:{dependency}",
            "--kill-on-invalid-dep=yes",
            f"--output={root}/logs/promote-%j.out",
            "--wrap",
            shlex.join(command),
        ],
        text=True,
    ).strip()
    return job


def advance(root, current, budget):
    ranked = rank_runs(current)
    dump(root / f"ranking_before_{budget}.json", ranked)
    if budget == 0:
        dump(
            root / "training_complete.json",
            {
                "status": "awaiting_independent_evaluation",
                "ranking": ranked,
                "goal_achieved": False,
                "next": "Rescore finalists on disjoint validation; freeze methods; evaluate 1000 locked test scenarios against independent references.",
            },
        )
        return
    threshold = 0.85 if budget <= 4_000_000 else 0.95
    eligible = [row for row in ranked if row["mean_success"] >= threshold]
    if len(eligible) < 2:
        dump(
            root / "promotion_stopped.json",
            {
                "reason": "fewer_than_two_reliable_arms",
                "threshold": threshold,
                "ranking": ranked,
                "goal_achieved": False,
            },
        )
        raise RuntimeError("validation reliability gate failed; diagnosis is required")
    finalists = eligible[:2]
    base = json.loads((root / "manifest.json").read_text())
    runs = []
    for row in finalists:
        runs.extend(row["runs"])
        if budget >= 12_000_000:
            prototype = row["runs"][0]
            existing = {run["seed"] for run in row["runs"]}
            runs.extend(
                {**prototype, "name": f"{row['arm']}_s{seed}", "seed": seed}
                for seed in range(5)
                if seed not in existing
            )
    runs.sort(key=lambda run: (run["seed"], run["name"]))
    if len(runs) % 2:
        raise RuntimeError("paired bundles require an even number of promoted runs")
    wave = root / "waves" / f"{root.name}_{budget}"
    wave.mkdir(parents=True, exist_ok=False)
    (wave / "training").symlink_to(root / "training", target_is_directory=True)
    (wave / "logs").symlink_to(root / "logs", target_is_directory=True)
    dump(wave / "manifest.json", {**base, "runs": runs, "promoted_from": str(current)})
    workspace = Path(base["python"]).parents[2]
    result = subprocess.check_output(
        [
            base["python"],
            str(workspace / "scripts/campaigns/california_graphppo.py"),
            "submit",
            "--directory",
            str(wave),
            "--budget",
            str(budget),
            "--bundles",
            str(len(runs) // 2),
        ],
        text=True,
    ).strip()
    job = result.splitlines()[-1]
    next_budget = 12_000_000 if budget < 12_000_000 else 0
    coordinator = schedule(root, job, wave, next_budget)
    dump(
        root / f"promotion_{budget}.json",
        {
            "training_job": job,
            "coordinator_job": coordinator,
            "wave": str(wave),
            "finalists": [row["arm"] for row in finalists],
            "transitions_per_run": budget,
        },
    )
    print(json.dumps({"job": job, "next_coordinator": coordinator}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("schedule", "advance"))
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--current", type=Path)
    parser.add_argument("--dependency")
    parser.add_argument("--next-budget", type=int, default=4_000_000)
    args = parser.parse_args()
    root = args.directory.resolve()
    current = (args.current or root).resolve()
    if args.action == "schedule":
        print(schedule(root, args.dependency, current, args.next_budget), flush=True)
    else:
        advance(root, current, args.next_budget)


if __name__ == "__main__":
    main()
