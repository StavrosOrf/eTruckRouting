"""Freeze live GraphPPO checkpoints and compare them with cached references.

This is development validation on the previously opened 128-case bank. It does
not open the final validation or test banks, or add candidates to the BKS pool.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path


COHORTS = (
    "ca10t3d_v1",
    "ca10t3d_progress_v3",
    "ca10t3d_imitation_v5",
    "ca10t3d_capacity_v7",
    "ca10t3d_objective_v8",
)
REFERENCES = ("heuristic", "mpc", "alns", "cpsat", "alns_planned", "cpsat_planned")
COUNT = 128
OFFSET = 2000
SHARDS = 4


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pipeline(root):
    spec = importlib.util.spec_from_file_location("intermediate_pipeline", root / "evaluation_pipeline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify(root):
    for name, expected in read(root / "input_hashes.json").items():
        if digest(Path(name)) != expected:
            raise ValueError(f"frozen intermediate input changed: {name}")


def freeze_run(directory, destination, *, prefix="best"):
    if prefix not in ("best", "pretrained"):
        raise ValueError("checkpoint prefix must be best or pretrained")
    history_name = "validation_history.json" if prefix == "best" else "pretrain_history.json"
    files = (f"{prefix}.pt", f"{prefix}_config.json", "run_config.json", "resolved_config.json", history_name)
    destination.mkdir(parents=True)
    for _ in range(5):
        before = {name: digest(directory / name) for name in files}
        for name in files:
            shutil.copy2(directory / name, destination / name)
        after = {name: digest(directory / name) for name in files}
        if before == after and all(digest(destination / name) == value for name, value in before.items()):
            return
    raise RuntimeError(f"checkpoint kept changing during freeze: {directory}")


def prepare(root, workspace, cohorts=COHORTS, *, include_all_arms=False, source_override=None, helper_directory=None, checkpoint_prefix="best", method_prefix=""):
    if root.exists():
        raise FileExistsError(f"inspect existing intermediate evaluation before continuing: {root}")
    root.mkdir(parents=True)
    base = workspace / "results/california_graphppo"
    scripts = helper_directory or workspace / "scripts/campaigns"
    for source, name in ((Path(__file__), "intermediate.py"), (scripts / "evaluate_california_graphppo.py", "evaluation_pipeline.py"), (scripts / "report_california_graphppo.py", "paired_report.py")):
        shutil.copy2(source, root / name)
    source = source_override or base / "ca10t3d_capacity_v7/source"
    config = root / "config.yaml"
    shutil.copy2(base / "ca10t3d_v1/configs/california_10t30d.yaml", config)
    methods = {}
    for cohort in cohorts:
        campaign = base / cohort
        promotions = [(int(path.stem.split("_", 1)[1]), path) for path in campaign.glob("promotion_*.json") if path.stem.split("_", 1)[1].isdigit()]
        if promotions and not include_all_arms:
            stage = read(max(promotions)[1])
            manifest = read(Path(stage["wave"]) / "manifest.json")
        else:
            manifest = read(campaign / "manifest.json")
        for run in manifest["runs"]:
            if run["seed"] not in range(3):
                continue
            name = method_prefix + run["name"]
            checkpoint = root / "checkpoints" / name
            freeze_run(campaign / "training" / run["name"], checkpoint, prefix=checkpoint_prefix)
            policy = read(checkpoint / f"{checkpoint_prefix}_config.json")
            expected = {"max_trucks": 10, "max_customers": 30, "state_encoder": "hetero_graph", "action_head": "complete_gcn"}
            if any(policy.get(key) != value for key, value in expected.items()):
                raise ValueError(f"unexpected policy/task configuration: {name}")
            if read(checkpoint / "run_config.json")["arguments"]["seed"] != run["seed"]:
                raise ValueError(f"training seed mismatch: {name}")
            environment = read(checkpoint / "resolved_config.json")["environment"]
            if environment.get("completion_capacity_exact_lookahead", False) and "completion_capacity_exact_lookahead" not in (source / "EVRoutingEnv/state/action_mask.py").read_text():
                raise ValueError("evaluation source does not support the trained exact-packing restriction")
            if environment.get("completion_work_escape_lookahead", False) and "completion_work_escape_lookahead" not in (source / "EVRoutingEnv/state/action_mask.py").read_text():
                raise ValueError("evaluation source does not support the trained work-escape restriction")
            best = None
            pretraining = None
            if checkpoint_prefix == "best":
                history = read(checkpoint / "validation_history.json")
                best = min(history, key=lambda row: (-row["success_rate"], row.get("mean_operating_time_successful") or float("inf")))
            else:
                history = read(checkpoint / "pretrain_history.json")
                expected_epochs = read(checkpoint / "run_config.json")["arguments"]["pretrain_epochs"]
                if not history or history[-1]["epoch"] != expected_epochs:
                    raise ValueError(f"pretraining is incomplete: {name}")
                pretraining = history[-1]
            if name in methods:
                raise ValueError(f"duplicate method: {name}")
            methods[name] = {
                "checkpoint": str(checkpoint), "prefix": checkpoint_prefix, "deterministic": True,
                "training_seed": run["seed"], "arm": name.rsplit("_s", 1)[0], "cohort": cohort,
                "selection_validation": best, "checkpoint_sha256": digest(checkpoint / f"{checkpoint_prefix}.pt"),
                "environment_overrides": {"environment": {key: bool(environment.get(key, False)) for key in ("completion_shield", "completion_capacity_lookahead", "completion_capacity_exact_lookahead", "completion_work_escape_lookahead")}},
            }
            if pretraining is not None:
                methods[name]["pretraining_holdout"] = pretraining
    groups = {}
    for method in methods.values():
        groups.setdefault(method["arm"], set()).add(method["training_seed"])
    if not groups or any(seeds != {0, 1, 2} for seeds in groups.values()):
        raise ValueError("each intermediate variant requires three independent training seeds")
    write(root / "methods.json", methods)
    reference_inputs = []
    evaluation = base / "ca10t3d_v1/evaluation"
    for name in REFERENCES:
        if name in ("alns_planned", "cpsat_planned"):
            directories = [evaluation / "strong_references_128/shards" / str(index) / "validation" / name for index in range(SHARDS)]
        else:
            directories = [evaluation / "checkpoint_1m/results/validation" / name]
        for index, directory in enumerate(directories):
            target = root / "references" / name / str(index)
            target.mkdir(parents=True)
            for filename in ("episode_rows.jsonl", "manifest.json"):
                shutil.copy2(directory / filename, target / filename)
            reference_inputs.append({"method": name, "directory": str(target), "original_directory": str(directory)})
    write(root / "reference_inputs.json", reference_inputs)
    write(root / "manifest.json", {
        "source": str(source), "config": str(config), "python": str(workspace / ".venv/bin/python"),
        "created_at": datetime.now(UTC).isoformat(), "evaluation_role": "reused intermediate development validation; not final validation or test",
        "scenario_count": COUNT, "validation_offset": OFFSET, "training_seeds": [0, 1, 2],
        "reference_methods": REFERENCES, "candidate_methods": list(methods), "goal_achieved": False,
        "cohorts": list(cohorts), "include_all_arms": include_all_arms,
        "checkpoint_prefix": checkpoint_prefix, "method_prefix": method_prefix,
    })
    files = [path for path in root.rglob("*") if path.is_file()]
    files += [path for path in source.rglob("*.py") if "__pycache__" not in path.parts]
    files += list((source / "EVRoutingEnv/data/case_studies/california").glob("*.json"))
    write(root / "input_hashes.json", {str(path): digest(path) for path in files})
    verify(root)
    write(root / "state.json", {"status": "prepared", "methods": len(methods), "goal_achieved": False})
    print(json.dumps({"directory": str(root), "methods": len(methods)}), flush=True)


def submit(root):
    verify(root)
    if (root / "jobs.json").exists():
        raise FileExistsError("evaluation already submitted; inspect recorded jobs")
    base = read(root / "manifest.json")
    command = [base["python"], str(root / "intermediate.py")]
    common = ["sbatch", "--parsable", "--account=tdsei17344", "--partition=cbuild", "--nodes=1", "--ntasks=1"]
    job = subprocess.check_output(common + ["--cpus-per-task=12", "--mem=64G", "--time=02:00:00", "--job-name=ca10t3d_intermediate", f"--array=0-{SHARDS - 1}%{SHARDS}", f"--output={root}/score-%A_%a.out", "--wrap", shlex.join(command + ["score", "--directory", str(root)])], text=True).strip()
    write(root / "jobs.json", {"scoring_job": job, "status": "scoring_submitted"})
    merge = subprocess.check_output(common + ["--cpus-per-task=2", "--mem=8G", "--time=00:20:00", "--job-name=ca10t3d_intermediate_report", f"--dependency=afterok:{job}", "--kill-on-invalid-dep=yes", f"--output={root}/merge-%j.out", "--wrap", shlex.join(command + ["merge", "--directory", str(root)])], text=True).strip()
    write(root / "jobs.json", {"scoring_job": job, "merge_job": merge, "status": "submitted"})
    print(json.dumps(read(root / "jobs.json")), flush=True)


def score(root, index):
    if index not in range(SHARDS):
        raise ValueError("invalid scenario shard")
    verify(root)
    module = pipeline(root)
    module.run_methods(root, root / "methods.json", "validation", COUNT // SHARDS, OFFSET + index * (COUNT // SHARDS), root / "shards" / str(index))
    verify(root)
    write(root / f"shard_{index}_state.json", {"status": "completed", "goal_achieved": False})


def merge(root):
    verify(root)
    module = pipeline(root)
    methods = read(root / "methods.json")
    rows, manifests = {}, {}
    inputs = [(item["method"], Path(item["directory"])) for item in read(root / "reference_inputs.json")]
    inputs += [(name, root / "shards" / str(index) / "validation" / name) for name in methods for index in range(SHARDS)]
    for name, directory in inputs:
        values = module.load_rows(directory / "episode_rows.jsonl")
        module.validate_task_rows(values, name)
        target = rows.setdefault(name, {})
        for row in values:
            if row["scenario_seed"] in target:
                raise ValueError(f"duplicate scenario: {name}")
            target[row["scenario_seed"]] = row
        manifests[str(directory)] = read(directory / "manifest.json")
    expected = set(range(1_000_000_000 + OFFSET, 1_000_000_000 + OFFSET + COUNT))
    if set(rows) != set(methods) | set(REFERENCES) or any(set(values) != expected for values in rows.values()):
        raise ValueError("incomplete method/scenario coverage")
    module.check_manifests(manifests)
    reporter = module.reporter(root)
    report = reporter.summarize(rows, REFERENCES)
    report["evaluation_role"] = read(root / "manifest.json")["evaluation_role"]
    report["training_seed_uncertainty"] = {arm: reporter.hierarchical_intervals(rows, REFERENCES, [name for name, settings in methods.items() if settings["arm"] == arm]) for arm in sorted({settings["arm"] for settings in methods.values()})}
    write(root / "independent_reference_report.json", report)
    components = {}
    for arm in sorted({settings["arm"] for settings in methods.values()}):
        records = []
        for name, settings in methods.items():
            if settings["arm"] != arm:
                continue
            for seed, row in rows[name].items():
                references = [rows[ref][seed] for ref in REFERENCES if rows[ref][seed]["success"]]
                if not row["success"] or not references:
                    continue
                reference = min(references, key=lambda item: item["total_operating_time"])
                denominator = reference["total_operating_time"]
                record = {key: 100 * (row[key] - reference[key]) / denominator for key in ("total_travel_time", "total_charging_time", "total_queue_time", "total_service_time", "total_time_window_waiting")}
                record.update(paired_excess_percent=100 * (row["total_operating_time"] / denominator - 1), candidate_vehicles=row["vehicles_used"], reference_vehicles=reference["vehicles_used"])
                records.append(record)
        components[arm] = {"successful_pairs": len(records), "mean": {key: sum(record[key] for record in records) / len(records) for key in records[0]} if records else {}}
    write(root / "cost_components.json", components)
    write(root / "state.json", {"status": "completed", "methods": len(methods), "scenarios": COUNT, "goal_achieved": False})
    for name in methods:
        result = report["methods"][name]
        print(name, result["success_rate"], result["mean_operating_hours_successful"], result["mean_paired_excess_percent"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "submit", "score", "merge"))
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--shard", type=int)
    parser.add_argument("--cohorts", nargs="+", default=list(COHORTS))
    parser.add_argument("--include-all-arms", action="store_true")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--helper-directory", type=Path)
    parser.add_argument("--checkpoint-prefix", choices=("best", "pretrained"), default="best")
    parser.add_argument("--method-prefix", default="")
    args = parser.parse_args()
    root = args.directory.resolve()
    if args.action == "prepare":
        prepare(root, args.workspace.resolve(), args.cohorts, include_all_arms=args.include_all_arms, source_override=args.source.resolve() if args.source else None, helper_directory=args.helper_directory.resolve() if args.helper_directory else None, checkpoint_prefix=args.checkpoint_prefix, method_prefix=args.method_prefix)
    elif args.action == "submit":
        submit(root)
    elif args.action == "score":
        score(root, args.shard if args.shard is not None else int(os.environ["SLURM_ARRAY_TASK_ID"]))
    else:
        merge(root)


if __name__ == "__main__":
    main()
