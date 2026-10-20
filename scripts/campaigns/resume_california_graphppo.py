"""Inspect or resume a restored campaign using NEW local SLURM job IDs.

Planning is read only. Submission requires a takeover acknowledgement: stop
the original campaign's writers first, or designate this as a distinct replica.
"""

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys


WORK = Path(__file__).resolve().parents[2]
BASE = WORK / "results/california_graphppo"
FAMILIES = ("ca10t3d_large_expert_v14", "ca10t3d_recovery_v15", "ca10t3d_action_edges_v16",
            "ca10t3d_objective_v8", "ca10t3d_message_identity_v17")


def plan(family):
    root = BASE / family
    if (root / "promotion_stopped.json").exists() or (root / "training_complete.json").exists():
        return {"family": family, "action": "held_for_shared_selection", "parallel_bound": 0}
    promotions = [path for path in root.glob("promotion_*.json") if path.stem.split("_", 1)[1].isdigit()]
    current = root
    if promotions:
        latest = max(promotions, key=lambda path: int(path.stem.split("_", 1)[1]))
        current = Path(json.loads(latest.read_text())["wave"])
    manifest = json.loads((current / "manifest.json").read_text())
    budget = manifest.get("transition_budget", 1_000_000)
    width = manifest.get("runs_per_allocation", 2)
    bundles = math.ceil(len(manifest["runs"]) / width)
    records = []
    for run in manifest["runs"]:
        directory = root / "training" / run["name"]
        history = directory / "training_history.json"
        steps = json.loads(history.read_text()).get("timesteps", []) if history.exists() else []
        records.append({"name": run["name"], "history_steps": steps[-1] if steps else 0,
                        "full_state_available": (directory / "training.pt").exists(),
                        "comparison_only": run.get("comparison_only", False)})
    return {"family": family, "action": "resume_or_advance", "root": str(root), "current": str(current),
            "budget": budget, "next_budget": 4_000_000 if budget <= 1_000_000 else 12_000_000 if budget <= 4_000_000 else 0,
            "width": width, "bundles": bundles,
            "parallel_bound": min(bundles, manifest.get("max_concurrent_allocations", min(30, bundles))),
            "resources": manifest.get("slurm_resources", {}), "runs": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "submit"))
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    parser.add_argument("--takeover", action="store_true", help="acknowledge original writers are stopped / this is an intentionally separate replica")
    args = parser.parse_args()
    if not (BASE / "handoff_restore.json").exists():
        parser.error("restore the portable bundle into a separate checkout first")
    records = [plan(family) for family in args.families]
    bound = sum(row["parallel_bound"] for row in records)
    if bound > 30:
        raise RuntimeError(f"parallel training bound {bound} exceeds 30")
    if args.action == "plan":
        print(json.dumps({"maximum_parallel_training_allocations": bound, "families": records}, indent=2))
        return
    if not args.takeover:
        parser.error("submit requires --takeover to avoid duplicate campaign writers")
    spec = importlib.util.spec_from_file_location("campaign_launcher", WORK / "scripts/campaigns/california_graphppo.py")
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    jobs = []
    for row in records:
        if row["action"] != "resume_or_advance":
            continue
        current, root = Path(row["current"]), Path(row["root"])
        manifest = json.loads((current / "manifest.json").read_text())
        resources = row["resources"]
        if resources.get("partition", "gpu_a100") not in {"gpu_mig", "gpu_a100"}:
            raise ValueError("only gpu_mig and gpu_a100 are permitted")
        indices = launcher.incomplete_training_bundles(current, manifest["runs"], row["width"], row["bundles"], row["budget"])
        training = None
        if indices:
            launcher.submit(current, row["bundles"], row["budget"])
            training = json.loads((current / "submissions.jsonl").read_text().splitlines()[-1])["job"].split(";")[0]
        command = [manifest["python"], str(root / "controller.py"), "advance", "--directory", str(root),
                   "--current", str(current), "--next-budget", str(row["next_budget"])]
        (root / "logs").mkdir(exist_ok=True)
        batch = ["sbatch", "--parsable", "--account=" + resources["account"], "--partition=gpu_mig",
                 "--gpus=1", "--nodes=1", "--ntasks=1", "--cpus-per-task=9", "--mem=60G", "--time=00:30:00",
                 "--job-name=ca10t3d_promote", "--output=" + str(root / "logs/takeover-%j.out")]
        if training:
            batch += ["--dependency=afterok:" + training, "--kill-on-invalid-dep=yes"]
        batch += ["--wrap", shlex.join(command)]
        coordinator = subprocess.check_output(batch, text=True).strip()
        jobs.append({"family": row["family"], "training_job": training, "coordinator_job": coordinator,
                     "current": row["current"], "next_budget": row["next_budget"]})
        # Persist after every family so a partial scheduler error is recoverable.
        (BASE / "takeover_jobs.json").write_text(json.dumps(jobs, indent=2) + "\n")
    print(json.dumps(jobs, indent=2))


if __name__ == "__main__":
    main()
