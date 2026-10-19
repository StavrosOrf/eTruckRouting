"""Isolate capacity-lookahead effects on frozen, trained imitation policies."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def load_pipeline(root):
    spec = importlib.util.spec_from_file_location("trained_mask_pipeline", root / "evaluation_pipeline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify(root):
    for name, expected in read(root / "input_hashes.json").items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"frozen trained-mask input changed: {name}")


def prepare(root, comparison, source=None, exact=False, work_escape=False):
    if root.exists():
        raise FileExistsError("inspect existing mask experiment before resubmission")
    root.mkdir(parents=True)
    for name in ("config.yaml",):
        shutil.copy2(comparison / name, root / name)
    for source_name, name in (("evaluate_california_graphppo.py", "evaluation_pipeline.py"), ("report_california_graphppo.py", "paired_report.py")):
        shutil.copy2(Path(__file__).with_name(source_name), root / name)
    shutil.copy2(__file__, root / "run.py")
    base = read(comparison / "manifest.json")
    if source is not None:
        base["source"] = str(source)
    arm = "bc_progress_work_escape" if work_escape else "bc_progress_exact" if exact else "bc_progress_capacity"
    base.update(config=str(root / "config.yaml"), comparison=str(comparison), evaluation_role="controlled post-training mask ablation on reused 128-case development validation", goal_achieved=False)
    base.update(candidate_arm=arm, exact_packing=exact, work_escape=work_escape)
    write(root / "manifest.json", base)
    originals = read(comparison / "methods.json")
    methods = {}
    for seed in range(3):
        original = f"bc_progress_s{seed}"
        settings = copy.deepcopy(originals[original])
        settings["environment_overrides"]["environment"]["completion_capacity_lookahead"] = True
        if exact:
            settings["environment_overrides"]["environment"]["completion_capacity_exact_lookahead"] = True
        if work_escape:
            settings["environment_overrides"]["environment"]["completion_work_escape_lookahead"] = True
        settings["arm"] = arm
        settings["control"] = original
        methods[f"{arm}_s{seed}"] = settings
    write(root / "methods.json", methods)
    hashes = read(comparison / "input_hashes.json")
    if source is not None:
        hashes.update({str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source.rglob("*.py") if "__pycache__" not in path.parts})
    if exact:
        shutil.copy2(Path(__file__).with_name("audit_california_packing_witnesses.py"), root / "witness_audit.py")
        shutil.copy2(comparison / "mask_trajectory_audit.json", root / "counterexample.json")
    for path in root.iterdir():
        if path.is_file():
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    write(root / "input_hashes.json", hashes)
    verify(root)
    command = [base["python"], str(root / "run.py"), "run", "--directory", str(root)]
    job = subprocess.check_output(["sbatch", "--parsable", "--account=tdsei17344", "--partition=cbuild", "--nodes=1", "--ntasks=1", "--cpus-per-task=4", "--mem=16G", "--time=00:30:00", "--job-name=ca10t3d_trained_mask", f"--output={root}/score-%j.out", "--wrap", shlex.join(command)], text=True).strip()
    write(root / "job.json", {"job": job, "goal_achieved": False})
    print(job, flush=True)


def run(root):
    verify(root)
    module = load_pipeline(root)
    base = read(root / "manifest.json")
    if base.get("exact_packing"):
        environment = dict(os.environ, PYTHONPATH=base["source"], OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        subprocess.run([base["python"], str(root / "witness_audit.py"), "--directory", str(root)], cwd=base["source"], env=environment, check=True)
    rows = module.run_methods(root, root / "methods.json", "validation", 128, 2000, root / "results")
    comparison = Path(base["comparison"])
    methods = read(root / "methods.json")
    manifests = {name: read(root / "results/validation" / name / "manifest.json") for name in methods}
    inputs = [(entry["method"], Path(entry["directory"])) for entry in read(comparison / "reference_inputs.json")]
    inputs += [(f"bc_progress_s{seed}", comparison / "shards" / str(shard) / "validation" / f"bc_progress_s{seed}") for seed in range(3) for shard in range(4)]
    for name, directory in inputs:
        target = rows.setdefault(name, {})
        for row in module.load_rows(directory / "episode_rows.jsonl"):
            if row["scenario_seed"] in target:
                raise ValueError(f"duplicate scenario: {name}")
            target[row["scenario_seed"]] = row
        manifests[str(directory)] = read(directory / "manifest.json")
    expected = set(range(1000002000, 1000002128))
    for name, values in rows.items():
        if set(values) != expected:
            raise ValueError(f"incomplete coverage: {name}")
        module.validate_task_rows(values.values(), name)
    module.check_manifests(manifests)
    reporter = module.reporter(root)
    references = base["reference_methods"]
    report = reporter.summarize(rows, references)
    report["evaluation_role"] = base["evaluation_role"]
    report["training_seed_uncertainty"] = {arm: reporter.hierarchical_intervals(rows, references, [f"{arm}_s{seed}" for seed in range(3)]) for arm in ("bc_progress", base.get("candidate_arm", "bc_progress_capacity"))}
    write(root / "independent_reference_report.json", report)
    verify(root)
    write(root / "state.json", {"status": "completed", "goal_achieved": False})
    for name in sorted(set(rows) - set(references)):
        print(name, json.dumps(report["methods"][name]), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--comparison", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--exact", action="store_true")
    parser.add_argument("--work-escape", action="store_true")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.directory.resolve(), args.comparison.resolve(), args.source.resolve() if args.source else None, args.exact, args.work_escape)
    else:
        run(args.directory.resolve())


if __name__ == "__main__":
    main()
