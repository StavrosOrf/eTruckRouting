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
import time
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "results/california_graphppo/ca10t3d_v1"


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def prepare(destination: Path, arm_set="base"):
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
    if arm_set == "progress":
        arms = [
            ("v2_shield", ["--completion-shield"]),
            ("progress_shield", ["--completion-shield", "--progress-action-features"]),
            ("v2_links_shield", ["--completion-shield", "--action-entity-links"]),
            (
                "progress_links_shield",
                [
                    "--completion-shield",
                    "--action-entity-links",
                    "--progress-action-features",
                ],
            ),
        ]
    if arm_set in ("imitation", "capacity", "objective", "retention"):
        common = [
            "--completion-shield",
            "--progress-action-features",
            "--demonstrations",
            str(destination / "demonstrations/alns_planned.npz"),
            "--pretrain-epochs",
            "20",
            "--pretrain-learning-rate",
            "0.0003",
            "--learning-rate",
            "0.0001",
            "--entropy-coefficient",
            "0.005",
        ]
        if arm_set == "retention":
            common = [*common, "--gamma", "1.0", "--expert-decay-timesteps", "12000000"]
            arms = [
                ("bc_gamma1_control", common),
                ("bc_retain01", [*common, "--expert-coefficient", "0.1"]),
                ("bc_retain1", [*common, "--expert-coefficient", "1.0"]),
            ]
        elif arm_set in ("capacity", "objective"):
            common = [
                *common,
                "--completion-capacity-lookahead",
                "--action-entity-links",
            ]
            if arm_set == "objective":
                common = [*common, "--nonlinear-edge-messages", "--gamma", "1.0"]
                arms = [
                    ("bc_nonlinear_gamma1", common),
                    (
                        "bc_nonlinear_cost40",
                        [
                            *common,
                            "--time-multiplier", "40",
                            "--success-bonus", "24000",
                            "--incompletion-penalty", "32000",
                            "--stranding-penalty", "16000",
                        ],
                    ),
                ]
            else:
                arms = [
                    ("bc_capacity_links", common),
                    (
                        "bc_capacity_nonlinear_links",
                        [*common, "--nonlinear-edge-messages"],
                    ),
                ]
        else:
            arms = [
                ("bc_progress", common),
                ("bc_links_progress", [*common, "--action-entity-links"]),
            ]
    runs = [
        {"name": f"{name}_s{seed}", "seed": seed, "flags": flags}
        for seed in range(3)
        for name, flags in arms
    ]
    if arm_set == "cost_finetune":
        runs = []
        parents = ROOT / "results/california_graphppo/intermediate_20261018_2200/checkpoints"
        for seed in range(3):
            common = [
                "--completion-shield", "--progress-action-features",
                "--learning-rate", "0.0001", "--entropy-coefficient", "0.005",
                "--gamma", "1.0", "--reset-value-head",
                "--init-from", str(parents / f"bc_progress_s{seed}"),
            ]
            for name, flags in (
                ("warm_control", [*common, "--delivery-bonus", "500", "--gae-lambda", "0.95"]),
                ("warm_cost_mc", [*common, "--delivery-bonus", "0", "--gae-lambda", "1.0"]),
            ):
                runs.append({"name": f"{name}_s{seed}", "seed": seed, "flags": flags})
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
            "arm_set": arm_set,
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
            "training": {
                "schedule_timesteps": 12_000_000,
                "imitation": arm_set in ("imitation", "capacity", "objective", "retention"),
            },
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
    # The pinned trainer floors total_timesteps / rollout_batch. Round its
    # limit up so the final 12M milestone can actually be reached on resume.
    rollout_batch = 16 * 128
    training_limit = ((12_000_000 + rollout_batch - 1) // rollout_batch) * rollout_batch

    def terminate(signum, frame):
        nonlocal stop
        stop = True
        for child, _handle in children:
            if child.poll() is None:
                child.send_signal(signal.SIGUSR1)

    signal.signal(signal.SIGUSR1, terminate)
    signal.signal(signal.SIGTERM, terminate)
    for slot, run in enumerate(runs):
        # Keep worker counts unchanged when the final bundle has one run;
        # complete checkpoints include the original worker/seed partition.
        assigned = cpus[slot * len(cpus) // width : (slot + 1) * len(cpus) // width]
        if len(assigned) < 2:
            raise ValueError("at least two allocated CPUs required per training run")
        output = destination / "training" / run["name"]
        resume = output / "training.pt"
        flags = list(run["flags"])
        if resume.exists() and "--init-from" in flags:
            index = flags.index("--init-from")
            del flags[index : index + 2]
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
            str(training_limit),
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
            *flags,
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


def unresolved_training_dependencies(dependencies):
    """Keep live dependencies; confirm terminal jobs before omitting old IDs."""
    terminal = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "BOOT_FAIL", "DEADLINE", "PREEMPTED", "REVOKED"}
    pending = []
    for job in dependencies:
        established = False
        for attempt in range(3):
            queue = subprocess.run(["squeue", "-h", "-j", job, "-o", "%i|%T"], capture_output=True, text=True)
            if queue.returncode == 0 and queue.stdout.strip():
                pending.append(job)
                established = True
                break
            if queue.returncode == 0 or "Invalid job id" in queue.stderr:
                accounting = subprocess.run(["sacct", "-n", "-X", "-j", job, "--format=State%30", "-P"], capture_output=True, text=True)
                states = [line.strip().split()[0].rstrip("+") for line in accounting.stdout.splitlines() if line.strip()]
                if accounting.returncode == 0 and states and all(state in terminal for state in states):
                    established = True
                    break
            if attempt < 2:
                time.sleep(attempt + 1)
        if not established:
            raise RuntimeError(f"cannot establish whether training dependency {job} is terminal")
    return pending


def submit(destination, bundles, budget):
    # One wave per family avoids simultaneous writers. Declared dependencies
    # additionally bound overlap between cohorts with different stage sizes.
    name = destination.name
    manifest = json.loads((destination / "manifest.json").read_text())
    if not 1 <= bundles <= (len(manifest["runs"]) + 1) // 2:
        raise ValueError("bundle count exceeds available experiments")
    if budget < 1:
        raise ValueError("transition budget must be positive")
    campaign_root = (destination / "training").resolve().parent
    family = campaign_root.name
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
    dependency_file = campaign_root / "training_dependencies.json"
    dependencies = []
    declared_dependencies = []
    if dependency_file.exists():
        declared_dependencies = json.loads(dependency_file.read_text()).get("afterany", [])
        if any(not isinstance(job, str) or not job.isdigit() for job in declared_dependencies):
            raise ValueError("training dependencies must be SLURM array/job IDs")
        dependencies = unresolved_training_dependencies(declared_dependencies)
        if dependencies:
            command[2:2] = ["--dependency=afterany:" + ":".join(dependencies), "--kill-on-invalid-dep=yes"]
    job = subprocess.check_output(command, text=True).strip()
    with (destination / "submissions.jsonl").open("a") as handle:
        handle.write(
            json.dumps({"job": job, "budget": budget, "bundles": bundles, "afterany": dependencies, "declared_afterany": declared_dependencies}) + "\n"
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
    parser.add_argument(
        "--arm-set",
        choices=("base", "progress", "imitation", "capacity", "objective", "retention", "cost_finetune"),
        default="base",
    )
    parser.add_argument("--budget", type=int, default=65536)
    args = parser.parse_args()
    directory = args.directory.resolve()
    if args.action == "prepare":
        prepare(directory, args.arm_set)
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
