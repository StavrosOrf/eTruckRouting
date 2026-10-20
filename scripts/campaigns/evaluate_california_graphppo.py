"""Freeze finalists, select on disjoint validation, then run locked test shards.

This file and the paired reporter are copied into the campaign before use.
Model inference always runs from the campaign's pinned simulator/policy source.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import shlex
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path


VALIDATION_COUNT = 512
VALIDATION_OFFSET = 10000
TEST_COUNT = 1000
TEST_SHARDS = 8
TEST_PER_SHARD = TEST_COUNT // TEST_SHARDS


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    seeds = [row["scenario_seed"] for row in rows]
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"duplicate scenario seeds: {path}")
    for row in rows:
        cost = float(row["total_operating_time"])
        if not math.isfinite(cost) or cost < 0 or (row.get("success") and cost == 0):
            raise ValueError(f"invalid operating cost: {path}")
    return rows


def validate_task_rows(values, name):
    for row in values:
        if row["customers_total"] != 30 or row["scenario"]["num_trucks"] != 10:
            raise ValueError(f"evaluation task is not 10 trucks / 30 customers: {name}")
        if row["success"] and (row["customers_served"] != 30 or row["truncated"]):
            raise ValueError(f"success flag contradicts completion metrics: {name}")


def normalized_config(config):
    config = copy.deepcopy(config)
    # Only these declared policy restrictions may differ. In particular, do not
    # discard deadlines, action budgets, charging physics or noise parameters.
    config.get("environment", {}).pop("completion_shield", None)
    config.get("environment", {}).pop("completion_capacity_lookahead", None)
    config.get("environment", {}).pop("completion_capacity_exact_lookahead", None)
    config.get("environment", {}).pop("completion_work_escape_lookahead", None)
    return config


def check_manifests(manifests):
    first = next(iter(manifests.values()))
    anchor = normalized_config(first["resolved_config"])
    for name, manifest in manifests.items():
        if normalized_config(manifest["resolved_config"]) != anchor:
            raise ValueError(
                f"evaluation physical/execution configuration differs: {name}"
            )
        if manifest.get("dependencies") != first.get("dependencies"):
            raise ValueError(f"evaluation dependency versions differ: {name}")


def reporter(root):
    spec = importlib.util.spec_from_file_location(
        "campaign_paired_report", root / "paired_report.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def base_manifest(root):
    base = read(root / "manifest.json")
    lock_path = root / "final_evaluation/source_lock.json"
    if lock_path.exists():
        lock = read(lock_path)
        actual = {
            str(path)
            for path in Path(lock["source"]).rglob("*.py")
            if "__pycache__" not in path.parts
        }
        if actual != set(lock["source_python_files"]):
            raise ValueError("frozen evaluation source file set changed")
        for name, expected in lock["sha256"].items():
            if digest(Path(name)) != expected:
                raise ValueError(f"frozen evaluation input changed: {name}")
        base["source"], base["config"] = lock["source"], lock["config"]
        return base
    override = root / "promotion_source.json"
    if override.exists():
        base["source"] = read(override)["source"]
    evaluation_override = root / "evaluation_source.json"
    if evaluation_override.exists():
        base["source"] = read(evaluation_override)["source"]
    return base


def freeze_source(root, evaluation):
    base = base_manifest(root)
    config = evaluation / "config.yaml"
    shutil.copy2(base["config"], config)
    source = Path(base["source"])
    source_files = [
        path for path in source.rglob("*.py") if "__pycache__" not in path.parts
    ]
    files = source_files + [
        config,
        root / "evaluation_pipeline.py",
        root / "paired_report.py",
    ]
    reference_settings = root / "baselines_final.json"
    if not reference_settings.exists():
        reference_settings = root / "baselines_pilot.json"
    if reference_settings.exists():
        files.append(reference_settings)
    # Map data remains shared/read-only; pin its content as well as its path.
    data = source / "EVRoutingEnv/data/case_studies/california"
    files.extend(data.glob("*.json"))
    write(
        evaluation / "source_lock.json",
        {
            "source": str(source),
            "config": str(config),
            "source_python_files": [str(path) for path in source_files],
            "sha256": {str(path): digest(path) for path in files},
        },
    )


def schedule(root, action, *, dependency=None, array=None, hours=4):
    base = base_manifest(root)
    command = [
        base["python"],
        str(root / "evaluation_pipeline.py"),
        action,
        "--directory",
        str(root),
    ]
    args = [
        "sbatch",
        "--parsable",
        "--account=tdsei17344",
        "--partition=gpu_mig",
            "--gpus=1",
        "--nodes=1",
        "--ntasks=1",
        "--cpus-per-task=9",
        f"--time={hours:02d}:00:00",
        f"--job-name=ca10t3d_{action}",
        f"--output={root}/logs/{action}-%A_%a.out",
    ]
    if dependency:
        args.extend([f"--dependency=afterok:{dependency}", "--kill-on-invalid-dep=yes"])
    if array is not None:
        args.append(f"--array=0-{array - 1}%{array}")
    args.extend(["--wrap", shlex.join(command)])
    return subprocess.check_output(args, text=True).strip()


def prepare(root, current):
    evaluation = root / "final_evaluation"
    if evaluation.exists():
        raise FileExistsError(
            "final evaluation already prepared; inspect its state before restarting"
        )
    evaluation.mkdir()
    freeze_source(root, evaluation)
    manifest = read(current / "manifest.json")
    methods = {}
    for run in manifest["runs"]:
        directory = Path(run.get("training_directory", root / "training" / run["name"]))
        config = read(directory / "best_config.json")
        expected = {
            "max_trucks": 10,
            "max_customers": 30,
            "state_encoder": "hetero_graph",
            "action_head": "complete_gcn",
        }
        if any(config.get(key) != value for key, value in expected.items()):
            raise ValueError(
                f"checkpoint does not implement the requested GraphPPO task: {run['name']}"
            )
        frozen = evaluation / "frozen_finalists" / run["name"]
        frozen.mkdir(parents=True)
        for name in (
            "best.pt",
            "best_config.json",
            "run_config.json",
            "resolved_config.json",
            "validation_history.json",
        ):
            shutil.copy2(directory / name, frozen / name)
        arm = run["name"].rsplit("_s", 1)[0]
        config_env = read(frozen / "resolved_config.json")["environment"]
        run_config = read(frozen / "run_config.json")
        if run_config["arguments"]["seed"] != run["seed"]:
            raise ValueError("training seed provenance differs from run manifest")
        methods[run["name"]] = {
            "checkpoint": str(frozen),
            "prefix": "best",
            "deterministic": True,
            "training_seed": run["seed"],
            "arm": arm,
            "checkpoint_sha256": digest(frozen / "best.pt"),
            "policy_config_sha256": digest(frozen / "best_config.json"),
            "environment_overrides": {
                "environment": {
                    "completion_shield": bool(
                        config_env.get("completion_shield", False)
                    ),
                    "completion_capacity_lookahead": bool(
                        config_env.get("completion_capacity_lookahead", False)
                    ),
                    "completion_capacity_exact_lookahead": bool(
                        config_env.get("completion_capacity_exact_lookahead", False)
                    ),
                    "completion_work_escape_lookahead": bool(
                        config_env.get("completion_work_escape_lookahead", False)
                    ),
                }
            },
        }
    groups = {}
    for method in methods.values():
        groups.setdefault(method["arm"], set()).add(method["training_seed"])
    if not groups or any(seeds != set(range(5)) for seeds in groups.values()):
        raise ValueError(
            "each finalist architecture requires five independent training seeds"
        )
    write(evaluation / "validation_methods.json", methods)
    job = schedule(root, "validation")
    write(
        evaluation / "pipeline_state.json",
        {
            "status": "disjoint_validation_queued",
            "validation_job": job,
            "validation_seed_namespace": "validation",
            "validation_offset": VALIDATION_OFFSET,
            "validation_scenarios": VALIDATION_COUNT,
            "goal_achieved": False,
        },
    )
    print(job, flush=True)


def archive_incomplete(output, split, methods, expected):
    for name in methods:
        directory = output / split / name
        if not directory.exists():
            continue
        rows = directory / "episode_rows.jsonl"
        complete = rows.exists() and (directory / "summary.json").exists()
        if complete:
            complete = {row["scenario_seed"] for row in load_rows(rows)} == expected
        if not complete:
            archive = (
                output
                / "failed_attempts"
                / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
            )
            archive.mkdir(parents=True)
            directory.rename(archive / name)


def run_methods(root, methods_path, split, count, offset, output):
    base = base_manifest(root)
    methods = read(methods_path)
    namespace = 1_000_000_000 if split == "validation" else 2_000_000_000
    expected = set(range(namespace + offset, namespace + offset + count))
    archive_incomplete(output, split, methods, expected)
    # Child evaluation processes also use tempfile for transport caches.
    # Select shared scratch explicitly even when called outside a coordinator.
    runtime_root = Path(base["python"]).parents[2] / "results/california_graphppo/runtime_cache"
    scratch = runtime_root / f"evaluation_{os.environ.get('SLURM_JOB_ID', 'local')}_{os.getpid()}"
    scratch.mkdir(parents=True, exist_ok=True)
    environment = dict(
        os.environ,
        PYTHONPATH=base["source"],
        TMPDIR=str(scratch),
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MPLCONFIGDIR=str(scratch / "matplotlib"),
        EVRP_SCENARIO_CACHE=str(scratch / "scenarios"),
    )
    subprocess.run(
        [
            base["python"],
            "scripts/evaluation/run_canonical_campaign.py",
            "--config",
            base["config"],
            "--methods",
            str(methods_path),
            "--split",
            split,
            "--scenarios",
            str(count),
            "--base-seed",
            str(offset),
            "--max-policy-steps",
            "10000",
            "--workers",
            str(min(12, len(methods), int(os.environ.get("EVRP_EVAL_WORKERS", len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 1)))),
            "--output",
            str(output),
        ],
        cwd=base["source"],
        env=environment,
        check=True,
    )
    rows = {}
    manifests = {}
    for name in methods:
        directory = output / split / name
        values = load_rows(directory / "episode_rows.jsonl")
        if {row["scenario_seed"] for row in values} != expected:
            raise ValueError(f"incomplete evaluation: {name}")
        validate_task_rows(values, name)
        rows[name] = {row["scenario_seed"]: row for row in values}
        manifests[name] = read(directory / "manifest.json")
    check_manifests(manifests)
    # Exact physical parity is checked in addition to execution/config parity.
    report = reporter(root)
    anchor = next(iter(rows.values()))
    for name, values in rows.items():
        for seed, row in values.items():
            if report.physical_scenario(row) != report.physical_scenario(anchor[seed]):
                raise ValueError(f"different physical scenario: {name}, {seed}")
    return rows


def choose_architecture(methods, rows, paired_report=None):
    groups = {}
    for name, settings in methods.items():
        values = list(rows[name].values())
        successful = [row for row in values if row["success"]]
        entry = {
            "method": name,
            "seed": settings["training_seed"],
            "success_rate": len(successful) / len(values),
            "operating_hours_successful": sum(
                row["total_operating_time"] for row in successful
            )
            / len(successful)
            if successful
            else None,
        }
        if paired_report is not None:
            paired = paired_report["methods"][name]
            entry.update(
                paired_excess_percent=paired["mean_paired_excess_percent"],
                paired_reference_complete=(
                    paired_report["reference_scenario_coverage"] == len(values)
                    and paired["paired_reference_coverage"] == len(successful)
                ),
            )
        groups.setdefault(settings["arm"], []).append(entry)
    ranking = []
    for arm, entries in groups.items():
        if len(entries) != 5 or {row["seed"] for row in entries} != set(range(5)):
            raise ValueError("architecture selection requires five training seeds")
        costs = [row["operating_hours_successful"] for row in entries]
        recipe = {
            "arm": arm,
            "per_seed": entries,
            "mean_success": sum(row["success_rate"] for row in entries) / 5,
            "mean_operating_hours": sum(costs) / 5
            if all(cost is not None for cost in costs)
            else None,
        }
        if paired_report is not None:
            gaps = [row["paired_excess_percent"] for row in entries]
            recipe.update(
                mean_paired_excess_percent=(
                    sum(gaps) / 5 if all(gap is not None for gap in gaps) else None
                ),
                paired_reference_complete=all(
                    row["paired_reference_complete"] for row in entries
                ),
            )
        ranking.append(recipe)
    reliable = [
        row
        for row in ranking
        if row["mean_success"] >= 0.99
        and (
            paired_report is None
            or (
                row["paired_reference_complete"]
                and row["mean_paired_excess_percent"] is not None
                and row["mean_paired_excess_percent"] <= 5
            )
        )
    ]
    # The actual campaign supplies paired references and gates both targets.
    reliable.sort(
        key=lambda row: row["mean_paired_excess_percent"]
        if paired_report is not None
        else row["mean_operating_hours"]
    )
    return (reliable[0] if reliable else None), ranking


def validation(root):
    evaluation = root / "final_evaluation"
    if (evaluation / "test_lock.json").exists():
        raise RuntimeError(
            "test already unlocked; inspect its jobs before retrying validation"
        )
    methods_path = evaluation / "validation_methods.json"
    methods = read(methods_path)
    baseline_path = root / "baselines_final.json"
    if not baseline_path.exists():
        baseline_path = root / "baselines_pilot.json"
    references = read(baseline_path)
    if set(methods) & set(references):
        raise ValueError("candidate and independent reference names overlap")
    combined_path = evaluation / "validation_all_methods.json"
    write(combined_path, {**references, **methods})
    rows = run_methods(
        root,
        combined_path,
        "validation",
        VALIDATION_COUNT,
        VALIDATION_OFFSET,
        evaluation / "validation_rescore",
    )
    paired = reporter(root).summarize(rows, list(references))
    write(evaluation / "validation_reference_report.json", paired)
    selected, ranking = choose_architecture(methods, rows, paired)
    write(
        evaluation / "validation_selection.json",
        {"selected": selected, "ranking": ranking, "selection_split": "validation"},
    )
    if selected is None:
        write(
            evaluation / "pipeline_state.json",
            {
                "status": "needs_validation_performance_improvement",
                "goal_achieved": False,
                "test_unlocked": False,
                "ranking": ranking,
                "required_success_rate": 0.99,
                "required_mean_paired_excess_percent": 5,
            },
        )
        return
    test_methods = references.copy()
    test_methods.update(
        {
            name: settings
            for name, settings in methods.items()
            if settings["arm"] == selected["arm"]
        }
    )
    write(evaluation / "test_methods.json", test_methods)
    # Lock the entire test seed set and checkpoint hashes before any test run.
    ledger_path = root / "test_seed_ledger.json"
    ledger = read(ledger_path) if ledger_path.exists() else []
    offset = max((entry["offset"] + entry["count"] for entry in ledger), default=0)
    write(
        evaluation / "test_lock.json",
        {
            "selected_arm": selected["arm"],
            "methods_sha256": digest(evaluation / "test_methods.json"),
            "scenario_seeds": list(
                range(2_000_000_000 + offset, 2_000_000_000 + offset + TEST_COUNT)
            ),
            "test_offset": offset,
            "references": list(read(baseline_path)),
            "test_scenarios": TEST_COUNT,
            "training_seeds": list(range(5)),
            "selection_split": "validation",
            "checkpoint_sha256": {
                name: settings["checkpoint_sha256"]
                for name, settings in test_methods.items()
                if "checkpoint" in settings
            },
        },
    )
    ledger.append(
        {
            "offset": offset,
            "count": TEST_COUNT,
            "test_lock_sha256": digest(evaluation / "test_lock.json"),
        }
    )
    write(ledger_path, ledger)
    job = schedule(root, "test-shard", array=TEST_SHARDS)
    aggregation = schedule(root, "aggregate", dependency=job, hours=1)
    write(
        evaluation / "pipeline_state.json",
        {
            "status": "locked_test_queued",
            "test_job": job,
            "aggregate_job": aggregation,
            "selected_arm": selected["arm"],
            "goal_achieved": False,
        },
    )


def check_lock(root):
    evaluation = root / "final_evaluation"
    lock = read(evaluation / "test_lock.json")
    base_manifest(root)  # Verify the simulator, map and physical configuration.
    if digest(evaluation / "test_methods.json") != lock["methods_sha256"]:
        raise ValueError("test method settings changed after locking")
    methods = read(evaluation / "test_methods.json")
    for name, settings in methods.items():
        if "checkpoint" in settings:
            directory = Path(settings["checkpoint"])
            if digest(directory / "best.pt") != lock["checkpoint_sha256"][name]:
                raise ValueError(f"checkpoint changed after locking: {name}")
            if (
                digest(directory / "best_config.json")
                != settings["policy_config_sha256"]
            ):
                raise ValueError(f"policy configuration changed after locking: {name}")
    return lock, methods


def test_shard(root, index):
    lock, _ = check_lock(root)
    if not 0 <= index < TEST_SHARDS:
        raise ValueError("test shard index outside the locked range")
    run_methods(
        root,
        root / "final_evaluation/test_methods.json",
        "test",
        TEST_PER_SHARD,
        lock["test_offset"] + index * TEST_PER_SHARD,
        root / "final_evaluation/test_shards" / str(index),
    )


def aggregate(root):
    lock, methods = check_lock(root)
    evaluation = root / "final_evaluation"
    rows = {name: {} for name in methods}
    manifests = {}
    for index in range(TEST_SHARDS):
        shard = evaluation / "test_shards" / str(index) / "test"
        for name in methods:
            values = load_rows(shard / name / "episode_rows.jsonl")
            validate_task_rows(values, name)
            for row in values:
                seed = row["scenario_seed"]
                if seed in rows[name]:
                    raise ValueError(f"overlapping test shards: {name}, {seed}")
                rows[name][seed] = row
            manifest = read(shard / name / "manifest.json")
            if name in manifests and normalized_config(
                manifest["resolved_config"]
            ) != normalized_config(manifests[name]["resolved_config"]):
                raise ValueError(f"configuration differs between test shards: {name}")
            if name in manifests and manifest.get("dependencies") != manifests[
                name
            ].get("dependencies"):
                raise ValueError(
                    f"dependency versions differ between test shards: {name}"
                )
            manifests[name] = manifest
    for name in methods:
        if set(rows[name]) != set(lock["scenario_seeds"]):
            raise ValueError(f"locked test scenario coverage incomplete: {name}")
    check_manifests(manifests)
    reporting = reporter(root)
    report = reporting.summarize(rows, lock["references"])
    write(evaluation / "test_report.json", report)
    seed_summaries = {
        name: report["methods"][name]
        for name, settings in methods.items()
        if "checkpoint" in settings
    }
    mean_success = sum(row["success_rate"] for row in seed_summaries.values()) / 5
    gaps = [row["mean_paired_excess_percent"] for row in seed_summaries.values()]
    mean_gap = sum(gaps) / 5 if all(gap is not None for gap in gaps) else None
    complete = (
        len(seed_summaries) == 5
        and report["reference_scenario_coverage"] == TEST_COUNT
        and all(
            row["paired_reference_coverage"] == row["successes"]
            for row in seed_summaries.values()
        )
    )
    target_met = bool(
        complete and mean_success >= 0.99 and mean_gap is not None and mean_gap <= 5
    )
    assessment = {
        "status": "targets_met_pending_goal_audit"
        if target_met
        else "needs_performance_improvement",
        "point_targets_met": target_met,
        "reference_type": "independent_best_known",
        "certified_optimality": False,
        "evaluation_complete": complete,
        "mean_success_across_five_seeds": mean_success,
        "mean_paired_excess_percent_across_five_seeds": mean_gap,
        "per_seed": seed_summaries,
        "test_scenarios": TEST_COUNT,
        "training_seed_count": len(seed_summaries),
        "note": "Observed test targets; report confidence intervals and audit provenance before declaring the goal achieved.",
        **reporting.hierarchical_intervals(
            rows, lock["references"], list(seed_summaries)
        ),
    }
    write(evaluation / "goal_assessment.json", assessment)
    write(evaluation / "pipeline_state.json", assessment)
    print(json.dumps(assessment, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("prepare", "validation", "test-shard", "aggregate")
    )
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--current", type=Path)
    parser.add_argument("--shard", type=int)
    args = parser.parse_args()
    if args.action == "prepare" and args.current is None:
        parser.error("prepare requires --current with the completed training wave")
    root = args.directory.resolve()
    if args.action == "prepare":
        prepare(root, args.current.resolve())
    elif args.action == "validation":
        validation(root)
    elif args.action == "test-shard":
        test_shard(
            root,
            args.shard
            if args.shard is not None
            else int(os.environ["SLURM_ARRAY_TASK_ID"]),
        )
    else:
        aggregate(root)


if __name__ == "__main__":
    main()
