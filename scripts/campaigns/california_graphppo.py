"""Immutable 10T/30-customer GraphPPO experiments and bounded SLURM bundles.

Run `prepare`, then `submit`. Each bundle owns a single allocated GPU and
independent CPU sets. Selection uses validation; test remains locked.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "results/california_graphppo/ca10t3d_v1"


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def prepare(destination: Path):
    if (destination / "manifest.json").exists():
        raise FileExistsError(f"immutable campaign already exists: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("logs", "training", "evaluation", "configs"):
        (destination / name).mkdir(exist_ok=True)
    config = yaml.safe_load(
        (ROOT / "EVRoutingEnv/config_files/config_joint.yaml").read_text()
    )
    config["environment"].update(
        num_trucks=10, num_stops=30, allow_variable_num_stops=False
    )
    config["network"]["data_path"] = str(
        ROOT / "EVRoutingEnv/data/case_studies/california"
    )
    # Keep the existing benchmark's physical model fixed. Nonlinear charging
    # and time-window variants are distinct experiments, never silent changes.
    config_path = destination / "configs/california_10t30d.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    arms = [
        ("normalized", []),
        ("shield", ["--completion-shield"]),
        ("links", ["--action-entity-links"]),
        ("links_shield", ["--action-entity-links", "--completion-shield"]),
    ]
    runs = [
        {"name": f"{name}_s{seed}", "seed": seed, "flags": flags}
        for seed in range(3)
        for name, flags in arms
    ]
    baselines = {
        "heuristic": {"parameters": {"energy_safety_factor": 1.2, "target_soc": 0.9}},
        "mpc": {
            "parameters": {"horizon": 6, "branching": 2, "energy_safety_factor": 1.2}
        },
        "alns": {
            "parameters": {
                "objective": "total_time",
                "iterations": 2000,
                "time_limit_seconds": 10,
                "average_charging_power_kw": 135,
                "energy_safety_factor": 1.2,
                "seed": 0,
            }
        },
        "cpsat": {
            "parameters": {
                "objective": "total_time",
                "time_limit_seconds": 20,
                "workers": 2,
                "average_charging_power_kw": 135,
                "energy_safety_factor": 1.2,
            }
        },
    }
    write_json(destination / "baselines.json", baselines)
    source = destination / "source"
    source.mkdir()
    for directory in ("algo", "EVRoutingEnv", "scripts"):
        shutil.copytree(
            ROOT / directory,
            source / directory,
            ignore=shutil.ignore_patterns("__pycache__", "data", "baselines", "*.out"),
        )
    shutil.copytree(
        ROOT / "EVRoutingEnv/baselines/heuristics",
        source / "EVRoutingEnv/baselines/heuristics",
    )
    shutil.copytree(
        ROOT / "EVRoutingEnv/baselines/optimization",
        source / "EVRoutingEnv/baselines/optimization",
        ignore=shutil.ignore_patterns("__pycache__", "julia", "branch_and_price"),
    )
    shutil.copy2(
        ROOT / "EVRoutingEnv/baselines/__init__.py",
        source / "EVRoutingEnv/baselines/__init__.py",
    )
    (source / "EVRoutingEnv/data").symlink_to(
        ROOT / "EVRoutingEnv/data", target_is_directory=True
    )
    (destination / "source.diff").write_text(
        subprocess.check_output(["git", "diff"], cwd=ROOT, text=True)
    )
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    write_json(
        destination / "manifest.json",
        {
            "version": 1,
            "commit": commit,
            "runs": runs,
            "config": str(config_path),
            "source": str(source),
            "python": str(ROOT / ".venv/bin/python"),
            "objective": "operating_time",
            "max_concurrent_training_jobs": 30,
            "problem": "joint_fleet",
            "trucks": 10,
            "total_customers": 30,
            "training": {"schedule_timesteps": 12_000_000, "imitation": False},
        },
    )
    print(destination, flush=True)


def train_bundle(destination, bundle, budget, width):
    manifest = json.loads((destination / "manifest.json").read_text())
    runs = manifest["runs"][bundle * width : (bundle + 1) * width]
    if not runs:
        raise ValueError("empty bundle")
    cpus = sorted(os.sched_getaffinity(0))
    children = []
    stop = False

    def terminate(signum, frame):
        nonlocal stop
        stop = True
        for child, _handle in children:
            if child.poll() is None:
                child.send_signal(signal.SIGUSR1)

    signal.signal(signal.SIGUSR1, terminate)
    signal.signal(signal.SIGTERM, terminate)
    for slot, run in enumerate(runs):
        assigned = cpus[
            slot * len(cpus) // len(runs) : (slot + 1) * len(cpus) // len(runs)
        ]
        if len(assigned) < 2:
            raise ValueError("at least two allocated CPUs required per training run")
        output = destination / "training" / run["name"]
        resume = output / "training.pt"
        command = [
            "taskset",
            "-c",
            ",".join(map(str, assigned)),
            manifest["python"],
            "scripts/training/train_canonical_ppo.py",
            "--config",
            manifest["config"],
            "--state-encoder",
            "hetero_graph",
            "--action-head",
            "complete_gcn",
            "--num-envs",
            "16",
            "--rollout-workers",
            str(max(1, len(assigned) - 2)),
            "--torch-threads",
            "1",
            "--rollout-steps",
            "128",
            "--epochs",
            "4",
            "--minibatch-size",
            "256",
            "--learning-rate",
            "0.0003",
            "--entropy-coefficient",
            "0.02",
            "--gamma",
            "0.999",
            "--time-multiplier",
            "10",
            "--success-bonus",
            "6000",
            "--incompletion-penalty",
            "8000",
            "--stranding-penalty",
            "4000",
            "--total-timesteps",
            "12000000",
            "--schedule-timesteps",
            "12000000",
            "--stop-after-timesteps",
            str(budget),
            "--train-scenarios",
            "20000",
            "--validation-scenarios",
            "64",
            "--validate-every",
            "50",
            "--checkpoint-every",
            "25",
            "--selection-objective",
            "operating_time",
            "--policy-action-mask",
            "hard",
            "--device",
            "cuda:0",
            "--seed",
            str(run["seed"]),
            "--output",
            str(destination / "training"),
            "--run-name",
            run["name"],
            *run["flags"],
        ]
        if resume.exists():
            command.extend(["--resume", str(resume)])
        environment = dict(
            os.environ,
            PYTHONPATH=manifest["source"],
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
        )
        scratch = Path(os.environ.get("TMPDIR", "/tmp")) / f"evrp_{os.getpid()}_{slot}"
        scratch.mkdir(parents=True, exist_ok=True)
        environment.update(
            TMPDIR=str(scratch),
            MPLCONFIGDIR=str(scratch / "matplotlib"),
            EVRP_SCENARIO_CACHE=str(scratch / "scenarios"),
        )
        handle = (destination / "logs" / f"{run['name']}_{budget}.log").open("a")
        handle.write("COMMAND " + repr(command) + "\n")
        handle.flush()
        child = subprocess.Popen(
            command,
            cwd=manifest["source"],
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        children.append((child, handle))
    codes = []
    for child, handle in children:
        codes.append(child.wait())
        handle.close()
    write_json(
        destination / "logs" / f"bundle_{bundle}_{budget}.json",
        {
            "exit_codes": codes,
            "stop_signal": stop,
            "runs": [run["name"] for run in runs],
        },
    )
    if any(codes):
        raise SystemExit(1)


def submit(destination, bundles, budget):
    # One named campaign wave at a time enforces a global ceiling and avoids
    # simultaneous writers to a run. Arrays are not independently set to %30.
    name = destination.name
    manifest = json.loads((destination / "manifest.json").read_text())
    if not 1 <= bundles <= (len(manifest["runs"]) + 1) // 2:
        raise ValueError("bundle count exceeds available experiments")
    if budget < 1:
        raise ValueError("transition budget must be positive")
    family = (destination / "training").resolve().parent.name
    active = subprocess.check_output(
        ["squeue", "-h", "-u", os.environ["USER"], "-o", "%j|%i"], text=True
    ).splitlines()
    matching = [row for row in active if row.split("|", 1)[0].startswith(family)]
    if matching:
        raise RuntimeError(f"campaign already has active training jobs: {matching}")
    wrapper = ROOT / "scripts/campaigns/california_graphppo.sbatch"
    command = [
        "sbatch",
        "--parsable",
        "--job-name",
        name,
        f"--array=0-{bundles - 1}%{min(30, bundles)}",
        f"--output={destination}/logs/slurm-%A_%a.out",
        str(wrapper),
        str(destination),
        str(budget),
    ]
    job = subprocess.check_output(command, text=True).strip()
    with (destination / "submissions.jsonl").open("a") as handle:
        handle.write(
            json.dumps({"job": job, "budget": budget, "bundles": bundles}) + "\n"
        )
    print(job, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("prepare", "submit", "train", "status", "references")
    )
    parser.add_argument("--directory", type=Path, default=DEFAULT)
    parser.add_argument("--bundle", type=int, default=0)
    parser.add_argument("--bundles", type=int, default=2)
    parser.add_argument("--width", type=int, default=2)
    parser.add_argument("--budget", type=int, default=65536)
    args = parser.parse_args()
    directory = args.directory.resolve()
    if args.action == "prepare":
        prepare(directory)
    elif args.action == "submit":
        submit(directory, args.bundles, args.budget)
    elif args.action == "train":
        train_bundle(directory, args.bundle, args.budget, args.width)
    elif args.action == "references":
        manifest = json.loads((directory / "manifest.json").read_text())
        subprocess.run(
            [
                manifest["python"],
                "scripts/evaluation/run_canonical_campaign.py",
                "--config",
                manifest["config"],
                "--methods",
                str(directory / "baselines.json"),
                "--split",
                "validation",
                "--scenarios",
                "64",
                "--workers",
                "4",
                "--output",
                str(directory / "evaluation/references"),
            ],
            cwd=manifest["source"],
            env=dict(os.environ, PYTHONPATH=manifest["source"]),
            check=True,
        )
    else:
        for path in sorted((directory / "training").glob("*/validation_history.json")):
            history = json.loads(path.read_text())
            if history:
                best = min(
                    history,
                    key=lambda row: (
                        -row["success_rate"],
                        row.get("mean_operating_time_successful") or float("inf"),
                    ),
                )
                print(path.parent.name, json.dumps(best), flush=True)


if __name__ == "__main__":
    main()
