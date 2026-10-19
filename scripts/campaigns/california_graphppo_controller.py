"""Promote completed GraphPPO waves on validation under a fixed compute envelope.

SLURM dependencies run this only after a training array finishes. It never waits
on a login node, consults test data, or declares near-optimality from training.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
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
    current_manifest = json.loads((current / "manifest.json").read_text())
    expected = current_manifest.get("transition_budget", 1_000_000)
    incomplete = []
    for run in current_manifest["runs"]:
        path = current / "training" / run["name"] / "training_history.json"
        steps = (
            json.loads(path.read_text()).get("timesteps", []) if path.exists() else []
        )
        if not steps or steps[-1] < expected:
            incomplete.append(run["name"])
    if incomplete:
        # A clean signal exit can finish an allocation before the interaction
        # budget. Continue complete checkpoints rather than promoting it.
        launch_wave(
            root,
            current,
            current_manifest["runs"],
            expected,
            budget,
            {"resuming_incomplete": incomplete},
        )
        return
    ranked = rank_runs(current)
    dump(root / f"ranking_before_{budget}.json", ranked)
    if budget == 0:
        hold = root / "evaluation_hold.json"
        if hold.exists():
            dump(
                root / "training_complete.json",
                {
                    "status": "awaiting_joint_validation_selection",
                    "ranking": ranked,
                    "coordination": json.loads(hold.read_text()),
                    "goal_achieved": False,
                },
            )
            maybe_start_joint_evaluation(root, json.loads(hold.read_text()))
            return
        pipeline = root / "evaluation_pipeline.py"
        evaluation_job = None
        if pipeline.exists():
            base = json.loads((root / "manifest.json").read_text())
            result = subprocess.check_output(
                [
                    base["python"],
                    str(pipeline),
                    "prepare",
                    "--directory",
                    str(root),
                    "--current",
                    str(current),
                ],
                text=True,
            ).strip()
            evaluation_job = result.splitlines()[-1]
        dump(
            root / "training_complete.json",
            {
                "status": "independent_evaluation_queued"
                if evaluation_job
                else "awaiting_independent_evaluation",
                "evaluation_job": evaluation_job,
                "ranking": ranked,
                "goal_achieved": False,
                "next": "Rescore finalists on disjoint validation; freeze methods; evaluate 1000 locked test scenarios against independent references.",
            },
        )
        return
    threshold = 0.85 if budget <= 4_000_000 else 0.95
    eligible = [row for row in ranked if row["mean_success"] >= threshold]
    if not eligible:
        dump(
            root / "promotion_stopped.json",
            {
                "reason": "no_reliable_arms",
                "threshold": threshold,
                "ranking": ranked,
                "goal_achieved": False,
            },
        )
        hold = root / "evaluation_hold.json"
        if hold.exists():
            maybe_start_joint_evaluation(root, json.loads(hold.read_text()))
        raise RuntimeError("validation reliability gate failed; diagnosis is required")
    finalists = eligible[:2]
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
    next_budget = 12_000_000 if budget < 12_000_000 else 0
    launch_wave(
        root,
        current,
        runs,
        budget,
        next_budget,
        {"finalists": [row["arm"] for row in finalists]},
    )


def launch_wave(root, current, runs, budget, next_budget, details):
    base = json.loads((root / "manifest.json").read_text())
    override = root / "promotion_source.json"
    if override.exists():
        source = json.loads(override.read_text())
        base["source"] = source["source"]
        base["promotion_source_revision"] = source["revision"]
    wave = root / "waves" / f"{root.name}_{budget}"
    retry = 0
    while wave.exists():
        retry += 1
        wave = root / "waves" / f"{root.name}_{budget}_resume{retry}"
    wave.mkdir(parents=True, exist_ok=False)
    (wave / "training").symlink_to(root / "training", target_is_directory=True)
    (wave / "logs").symlink_to(root / "logs", target_is_directory=True)
    (wave / "source").symlink_to(base["source"], target_is_directory=True)
    dump(
        wave / "manifest.json",
        {
            **base,
            "runs": runs,
            "promoted_from": str(current),
            "transition_budget": budget,
        },
    )
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
            str((len(runs) + 1) // 2),
        ],
        text=True,
    ).strip()
    job = result.splitlines()[-1]
    coordinator = schedule(root, job, wave, next_budget)
    dump(
        root / f"promotion_{budget}.json",
        {
            "training_job": job,
            "coordinator_job": coordinator,
            "wave": str(wave),
            **details,
            "transitions_per_run": budget,
        },
    )
    print(json.dumps({"job": job, "next_coordinator": coordinator}), flush=True)


def maybe_start_joint_evaluation(root, coordination):
    """The last finished cohort starts one shared, validation-only selection."""
    campaigns = [Path(value) for value in coordination["campaigns"]]
    if root not in campaigns:
        raise ValueError("cohort is missing from the joint evaluation coordination")
    master = campaigns[0]
    with (master / "joint_selection.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (master / "final_evaluation").exists():
            return
        queue = subprocess.check_output(
            ["squeue", "-h", "-u", os.environ["USER"], "-o", "%j"], text=True
        ).splitlines()
        if any(
            name.startswith(campaign.name) for name in queue for campaign in campaigns
        ):
            return
        finalists = []
        completed = []
        for campaign in campaigns:
            status = campaign / "training_complete.json"
            if status.exists():
                value = json.loads(status.read_text())
                for arm in value["ranking"]:
                    runs = arm["runs"]
                    if len(runs) != 5 or {run["seed"] for run in runs} != set(range(5)):
                        raise ValueError(
                            "joint finalists must include all five training seeds"
                        )
                    for run in runs:
                        directory = campaign / "training" / run["name"]
                        history = json.loads(
                            (directory / "training_history.json").read_text()
                        )
                        if (
                            not history["timesteps"]
                            or history["timesteps"][-1] < 12_000_000
                        ):
                            raise ValueError(
                                "joint final evaluation requires completed convergence runs"
                            )
                        finalists.append({**run, "training_directory": str(directory)})
                completed.append({"campaign": str(campaign), "status": value["status"]})
            elif (campaign / "promotion_stopped.json").exists():
                completed.append(
                    {
                        "campaign": str(campaign),
                        "status": "screening_reliability_gate_failed",
                    }
                )
            else:
                return
        if not finalists:
            dump(
                master / "joint_selection_stopped.json",
                {
                    "reason": "no_five_seed_finalists",
                    "cohorts": completed,
                    "goal_achieved": False,
                },
            )
            return
        if len({run["name"] for run in finalists}) != len(finalists):
            raise ValueError("joint finalist run names must be unique across cohorts")
        current = master / "joint_finalists"
        current.mkdir(exist_ok=True)
        base = json.loads((master / "manifest.json").read_text())
        dump(
            current / "manifest.json", {**base, "runs": finalists, "cohorts": completed}
        )
        job = (
            subprocess.check_output(
                [
                    base["python"],
                    str(master / "evaluation_pipeline.py"),
                    "prepare",
                    "--directory",
                    str(master),
                    "--current",
                    str(current),
                ],
                text=True,
            )
            .strip()
            .splitlines()[-1]
        )
        dump(
            master / "joint_final_evaluation.json",
            {"validation_job": job, "cohorts": completed, "goal_achieved": False},
        )


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
