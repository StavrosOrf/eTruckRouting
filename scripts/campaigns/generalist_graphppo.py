"""One generalist GraphPPO policy for the five suite problems: campaign driver.

Plan: ``docs/campaigns/generalist_graphppo_suite_campaign.md``. Stages, in order:

    PYTHONPATH=. python scripts/campaigns/generalist_graphppo.py prepare --directory D --bks-from B [--arm-set screen]
    ... submit --directory D --stage references --pack 3   # 3 GPU shares work through the reference shards
    ... merge-references --directory D
    ... submit --directory D --stage train --budget 2000000 [--pack K] [--runs-per-gpu 1] [--arms A B]
    ... submit --directory D --stage evaluate --split validation --tag 2M   # promote on validation only
    ... report --directory D --split validation --tag 2M --promote 3
    ... generalist_near_optimal_gate.py --campaign D --split validation --tag 2M --bks-dir B

``train``, ``train-pack`` and ``references`` also run inside an allocation (the
sbatch wrappers call them). With ``--pack K`` one job trains K runs at once,
each pinned to its own GPU (or sharing one, ``--runs-per-gpu``) and its own
disjoint CPU slice, with logs kept per run. ``prepare`` freezes the problems, arms, seed counts and a source
snapshot; ``prepare --references-from`` reuses another campaign's merged
references, and ``prepare --bks-from`` freezes a copy of the best-known
solutions (BKS) the imitation arms clone and pair their reward against. The
zero-shot size probes are written and evaluated only with ``prepare --probes``.
``prepare --problems-dir DIR`` trains on every ``*.yaml`` of DIR instead of
the suite (static where the configuration has no ``requests`` block, dynamic
otherwise; every later stage reads the frozen ``problems.json``), and
``prepare --train-seeds N`` sets the train seeds per problem (references,
plan caches and ``--train-scenarios`` follow the manifest).
``submit --test-only`` asks the scheduler to check every composed command
without submitting it. Evaluation rows are kept per tag (``evaluation/<split>/<tag>/``), so
a later checkpoint is never confused with an earlier one; baselines are shared in
``evaluation/<split>/baselines/``. The test bank is opened once, with
``evaluate --split test``.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "results/generalist_graphppo/suite_v1"
SUITE = ROOT / "EVRoutingEnv/config_files/generalist_suite"
PROBLEMS = ("ca_sameday", "ca_drayage_combined", "tx_longhaul", "tx_longhaul_congested", "pnw_longhaul_hos")
DYNAMIC = ("ca_sameday", "ca_drayage_combined")
# Reference policy per problem; "auto" is greedy dispatch for requests and the
# greedy navigator otherwise.
EXTRA_BASELINES = {
    "ca_sameday": ["wait_at_depot"],
    "ca_drayage_combined": ["wait_at_depot", "kullman_myopic"],
    "tx_longhaul_congested": ["kullman_myopic"],
}
# Zero-shot size probes: evaluation only, never trained on; written and
# evaluated only with ``prepare --probes``.
PROBES = {
    "tx_longhaul_t5c15": ("tx_longhaul", 5, 15),
    "tx_longhaul_t20c60": ("tx_longhaul", 20, 60),
    "ca_sameday_t5": ("ca_sameday", 5, None),
    "ca_sameday_t20": ("ca_sameday", 20, None),
}
SEEDS = {"train": 2048, "validation": 128, "test": 500}
SHARD = 256
SCHEDULE = 8_000_000
COMMON = [
    "--num-envs", "40", "--rollout-steps", "128", "--epochs", "4", "--minibatch-size", "256",
    "--learning-rate", "0.0003", "--entropy-coefficient", "0.01", "--gamma", "1.0",
    # 64 chargers: at 32 the BKS demonstrations lost 0.77 % of their decisions
    # (1.9 % on the long-haul problems) to charger choices outside the subgraph;
    # at 64 Texas keeps all but 0.32 % (pilot, 2026-10-23).
    "--gae-lambda", "0.95", "--target-kl", "0.05", "--charger-budget", "64",
    "--hidden-dim", "128", "--encoder-layers", "3", "--action-head-layers", "2",
    "--bc-episodes", "64", "--pretrain-epochs", "20", "--pretrain-learning-rate", "0.0003",
    "--train-scenarios", str(SEEDS["train"]), "--validation-scenarios", "16",
    "--validate-every", "25", "--checkpoint-every", "10",
]
# Imitation arms: demonstrations and paired rewards from the best-known
# solutions frozen into <campaign>/bks by ``prepare --bks-from`` (``{bks}`` is
# substituted there). The DAgger experts are the BKS planners at a reduced
# budget, one per problem (algo/generalist_experts.py).
BKS_DIR = "{bks}"


def expert_flags(names, dynamic) -> list[str]:
    """``--dagger-expert`` entries: the rolling planner on request problems, the long-haul plan elsewhere."""
    return [
        f"{name}=algo.generalist_experts:{'rolling_request' if name in dynamic else 'longhaul_plan'}"
        for name in names
    ]


EXPERTS = expert_flags(PROBLEMS, DYNAMIC)
FULL = ["--fleet-assignment", "--reward-reference", "scenario"]
IMITATION = [
    "--demonstrations", BKS_DIR, "--bc-episodes", "256", "--pretrain-epochs", "20",
    "--reward-references", BKS_DIR,
]
RETAINED = ["--bc-coefficient", "0.1", "--bc-coefficient-final", "0.0", "--bc-anneal-timesteps", "2000000"]
DAGGER = [
    "--dagger-expert", *EXPERTS, "--dagger-every", "10", "--dagger-episodes", "8",
    "--dagger-beta", "0.5", "--dagger-beta-final", "0.0", "--dagger-beta-anneal-timesteps", "2000000",
    "--dagger-pretrain-rounds", "2",
]
# suite_v2 (user decision 2026-10-25, design 4.4 / 9.5): the planner's route
# plan in the observation, withheld in 30 % of train-seed episodes; static plans
# come from the time-0 plan cache frozen into <campaign>/bks.
PLAN = ["--plan-conditioned", "--plan-cache", BKS_DIR, "--plan-dropout", "0.3"]
# suite_v3 (user decision 2026-10-27, option B): the imitation term stays at 0.1
# and the DAgger expert's share falls only to 0.2 over the 4M run, instead of
# both fading to zero; a 4-layer encoder; checkpoints chosen on 64 validation
# seeds per problem instead of 16.
HELD = ["--bc-coefficient", "0.1", "--bc-coefficient-final", "0.1"]
DAGGER_HELD = [
    "--dagger-expert", *EXPERTS, "--dagger-every", "10", "--dagger-episodes", "8",
    "--dagger-beta", "0.5", "--dagger-beta-final", "0.2", "--dagger-beta-anneal-timesteps", "4000000",
    "--dagger-pretrain-rounds", "2",
]
DEEP = ["--encoder-layers", "4", "--validation-scenarios", "64"]
# suite_v4 (user decision 2026-10-28): the planner's charging moves to the
# active truck's next stop join the plan input (static problems); otherwise
# full_dagger_plan unchanged, so it compares with suite_v2's run at <= 4M.
CHARGE_PLAN = ["--charge-plan"]
# suite_v5 (user decision 2026-10-31): PPO practises the problem it is deployed
# on. suite_v4 trained without the filters, with the plan withheld in 30 % of
# episodes, at entropies whose samples ran far below the greedy policy, and with
# advantages scaled across the mixture, and it could read one hidden bit through
# a waiting truck's wake-up (--public-wake closes that). Warm-started from suite_v4's best.pt of
# the same seed, at a lower learning rate; checkpoints kept at every validation.
SUITE_V4_RUN = str(ROOT / "results/generalist_graphppo/suite_v4/training/full_dagger_plan_charge_s{seed}")
DEPLOYED = [
    "--safety-mask", "--safety-margin", "0", "--retire-guard", "--public-wake", "--plan-dropout", "0", "--advantage-by-problem",
    "--learning-rate", "0.0001", "--entropy-coefficient", "0.001", "--validation-scenarios", "64",
    "--save-validation-checkpoints", "--init-from", SUITE_V4_RUN, "--pretrain-epochs", "0", "--dagger-pretrain-rounds", "0",
]
# 512-step rollouts (20480 observations) do not fit a 40 GB GPU collated, so the
# rollout stays on the CPU and minibatches move over; a 1024-observation
# minibatch's activations do not fit either, so the 1024-decision gradient
# step is 4 accumulated minibatches of 256.
BIGBATCH = [
    "--rollout-steps", "512", "--minibatch-size", "256", "--accumulate-minibatches", "4", "--rollout-on-cpu",
    "--validate-every", "6", "--checkpoint-every", "3", "--dagger-every", "3",
]
# Imitation sized by cost for the 40-problem broad set (user decision 1, 2026-11-08): see the broad recipes below.
BROAD_IMITATION_BUDGET = [
    "--bc-max-decisions-per-problem", "2500", "--dagger-episodes", "1", "--dagger-every", "6", "--dagger-max-states", "40000",
]
ANCHOR = [
    "--bc-coefficient", "0.05", "--bc-coefficient-final", "0.05", "--dagger-expert", *EXPERTS, "--dagger-every", "10",
    "--dagger-episodes", "8", "--dagger-beta", "0.3", "--dagger-beta-final", "0.3",
]
RECIPES = {
    "base": [],
    "fleet": ["--fleet-assignment"],
    "paired": ["--reward-reference", "scenario"],
    "full": FULL,
    "full_bks": [*FULL, *IMITATION],
    "full_bks_retained": [*FULL, *IMITATION, *RETAINED],
    # DAgger labels reach the PPO loss only through the retained term, so this
    # arm keeps it: full_bks -> +retained -> +DAgger, one change per rung.
    "full_dagger": [*FULL, *IMITATION, *RETAINED, *DAGGER],
    "full_bks_plan": [*FULL, *IMITATION, *PLAN],
    "full_bks_retained_plan": [*FULL, *IMITATION, *RETAINED, *PLAN],
    "full_dagger_plan": [*FULL, *IMITATION, *RETAINED, *DAGGER, *PLAN],
    "full_dagger_plan_deep": [*FULL, *IMITATION, *HELD, *DAGGER_HELD, *PLAN, *DEEP],
    "full_dagger_plan_charge": [*FULL, *IMITATION, *RETAINED, *DAGGER, *PLAN, *CHARGE_PLAN],
    # Pure PPO from the warm start: any change is PPO's.
    "ppo_deployed": [*FULL, "--reward-references", BKS_DIR, "--bc-episodes", "0", *PLAN, *CHARGE_PLAN, *DEPLOYED],
    "ppo_deployed_anchor": [*FULL, *IMITATION, *ANCHOR, *PLAN, *CHARGE_PLAN, *DEPLOYED],
    # One critic per mechanism family (2026-11-01): the gradient probe found no
    # actor conflict between problems but shared value structure only within the
    # long-haul family, so the critics split and the actor stays shared.
    "ppo_deployed_anchor_critics": [*FULL, *IMITATION, *ANCHOR, *PLAN, *CHARGE_PLAN, *DEPLOYED, "--critic-heads", "family"],
    # Variance reduction (2026-11-01): the same probe found a problem's PPO
    # gradient on two independent rollouts has cosine ~ 0 at the training batch,
    # so these arms raise the decisions per update, pair the rollouts by seed
    # (10 groups of 4 per rollout; a later flag overrides COMMON's), or turn the
    # surrogate off for the imitation-only control.
    # 4x the decisions behind every gradient step; cadences in updates are
    # divided by 4 so validations, checkpoints and DAgger rounds keep their
    # step spacing. The group arm needs the long rollouts too: suite episodes
    # run ~160 decisions, so at 128 steps no episode would complete inside a
    # rollout and every step would fall back to GAE.
    # suite_v6: the passing arm's recipe trained from scratch on the broad problem
    # set with the deployed filters on, per-problem advantages, 16 selection seeds
    # per problem and a checkpoint at every validation (pre-registration suite_v6).
    # suite_v5b/v5c verdicts (preregistration_suite_v5b.md amendments 7-8): the family critics (2026-11-06) and the
    # big batch (2026-11-07) each showed a noticeable improvement attributable to PPO on 3 of 3 seeds, so
    # "--critic-heads family" and BIGBATCH (512-step rollouts, 4 x 256 accumulated minibatches, rollout on the CPU,
    # validation / checkpoint / DAgger cadences divided by 4) enter the suite_v6 recipe; the group baseline (1 of 3) does not.
    # User decision 1 of 2026-11-08 (pre-registration suite_v6 amendments 15-16): the imitation data is sized by cost on the
    # 40-problem set, where a replayed observation weighs 0.2-1.2 MB and the rolling-request expert takes up to an hour per
    # episode on 50-100 trucks: behaviour cloning takes whole demonstration episodes up to 2 500 decisions per problem
    # (about 100k observations in all, half of suite_v5c's), DAgger runs one episode per problem per round every 6 updates
    # (40 per round, the suite's absolute budget) and keeps 40k states, and the budget is 4M steps.
    "broad_dagger_plan_charge": [
        *FULL, *IMITATION, *RETAINED, *DAGGER, *PLAN, *CHARGE_PLAN,
        "--safety-mask", "--safety-margin", "0", "--retire-guard", "--public-wake", "--advantage-by-problem",
        "--validation-scenarios", "16", "--save-validation-checkpoints", "--critic-heads", "family", *BIGBATCH,
        *BROAD_IMITATION_BUDGET,
    ],
    # suite_v6's imitation-only control (pre-registration suite_v6, "PPO's
    # contribution"): the broad recipe with the policy gradient switched off.
    "broad_imitation_only": [
        *FULL, *IMITATION, *RETAINED, *DAGGER, *PLAN, *CHARGE_PLAN,
        "--safety-mask", "--safety-margin", "0", "--retire-guard", "--public-wake", "--advantage-by-problem",
        "--validation-scenarios", "16", "--save-validation-checkpoints", "--critic-heads", "family", *BIGBATCH,
        *BROAD_IMITATION_BUDGET, "--policy-gradient-weight", "0",
    ],
    "ppo_deployed_anchor_bigbatch": [*FULL, *IMITATION, *ANCHOR, *PLAN, *CHARGE_PLAN, *DEPLOYED, *BIGBATCH],
    "ppo_deployed_anchor_group": [
        *FULL, *IMITATION, *ANCHOR, *PLAN, *CHARGE_PLAN, *DEPLOYED, *BIGBATCH, "--group-size", "4", "--advantage-mode", "group",
    ],
    "imitation_only_deployed": [*FULL, *IMITATION, *ANCHOR, *PLAN, *CHARGE_PLAN, *DEPLOYED, "--policy-gradient-weight", "0"],
}
SCREEN = ("base", "fleet", "paired", "full", "full_bks", "full_bks_retained", "full_dagger")
STATIC = tuple(name for name in PROBLEMS if name not in DYNAMIC)
SLURM = {
    # Snellius gpu_a100: four A100s, 72 cores and 480 GB per node, so one GPU
    # share is 18 cores and about 120 GB. Every job on a GPU partition must hold
    # a GPU, so the CPU stage (references) takes one as well; cbuild is the
    # fallback (--partition=cbuild --gpus-per-task=0 --cpus-per-task=16 by hand).
    "account": "tdsei17344",
    "gpu_partition": "gpu_a100",
    "cpu_partition": "gpu_a100",
    "cpus_per_gpu": 18,  # CPUs per training run, also with --runs-per-gpu > 1
    "gpu_mem_per_cpu": "6500M",  # 117 GB per 18-core share
    "cpu_cores": 18,
    "cpu_node_cores": 18,  # a packed references job is one GPU share
    "cpu_mem_per_cpu": "6500M",
    "cpu_gpus": 1,  # GPUs held by a CPU-stage job (0 on a CPU partition)
    "overflow": {"gpu_h100": 16, "gpu_mig": 9},  # partition -> cores per GPU or MIG slice
    "max_concurrent": 20,  # this campaign's share of the 30-allocation cap
}
BKS_FILES = ("validation.json", "train_demonstrations.json")
BKS_OPTIONAL = ("summary.json", "train_demonstrations_summary.json", "plans_train.json", "plans_validation.json")
PLAN_FILES = ("plans_train.json", "plans_validation.json")
SLOT_COSTS = "slot_costs.json"  # optional, in the campaign directory (see ``train``)


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def read_manifest(destination: Path) -> dict:
    return json.loads((destination / "manifest.json").read_text())


def read_problems(manifest: dict) -> list[dict]:
    """The campaign's frozen problem rows (``problems.json``; the probes are kept apart)."""
    return json.loads(Path(manifest["problems"]).read_text())


def problem_names(directory: Path) -> list[str]:
    """The problems of a configuration directory: the suite in its fixed order, any other sorted by name."""
    if directory.resolve() == SUITE.resolve():
        return list(PROBLEMS)
    names = sorted(path.stem for path in directory.glob("*.yaml"))
    if not names:
        raise SystemExit(f"no *.yaml problem configurations in {directory}")
    return names


def is_static(config: str | Path) -> bool:
    """A problem whose configuration has no ``requests`` block is static (long haul); the rest are dynamic."""
    return "requests" not in (yaml.safe_load(Path(config).read_text()) or {})


def static_names(rows: list[dict]) -> list[str]:
    return [row["name"] for row in rows if is_static(row["config"])]


def with_experts(flags: list[str], experts: list[str]) -> list[str]:
    """The recipe's per-problem DAgger experts (the suite's) replaced by ``experts``."""
    if not any(flag in EXPERTS for flag in flags):
        return flags
    replaced = []
    for flag in flags:
        if flag == EXPERTS[0]:
            replaced.extend(experts)
        elif flag not in EXPERTS:
            replaced.append(flag)
    return replaced


def arms_for(
    arm_set: str, recipe: str, seeds: int, bks: str | None = None, arms: list[str] | None = None,
    problems=None, dynamic=DYNAMIC,
) -> list[dict]:
    """The runs of an arm set; ``{bks}`` in a recipe becomes the frozen BKS copy.

    ``problems`` (default: the suite) are the campaign's problems, ``dynamic``
    those with requests: the leave-one-out and specialist sets and the DAgger
    experts of every recipe follow them.
    """
    problems = list(PROBLEMS if problems is None else problems)
    experts = expert_flags(problems, dynamic)
    if arm_set == "screen":
        chosen = [(name, RECIPES[name]) for name in SCREEN]
    elif arm_set == "named":
        if not arms:
            raise SystemExit("--arm-set named needs --arms")
        unknown = sorted(set(arms) - set(RECIPES))
        if unknown:
            raise SystemExit(f"unknown recipes {unknown}; known: {sorted(RECIPES)}")
        chosen = [(name, RECIPES[name]) for name in arms]
    elif arm_set == "lopo":
        # Leave one problem out: trained on four, the fifth is zero-shot.
        chosen = [
            (f"{recipe}_minus_{held}", [*RECIPES[recipe], "--train-problems", *[p for p in problems if p != held]])
            for held in problems
        ]
    elif arm_set == "specialists":
        chosen = [(f"{recipe}_only_{name}", [*RECIPES[recipe], "--train-problems", name]) for name in problems]
    else:
        raise ValueError(f"unknown arm set {arm_set!r}")
    runs = []
    for seed in range(seeds):
        for name, flags in chosen:
            flags = with_experts(flags, experts)
            flags = [flag.replace(BKS_DIR, bks) if bks else flag for flag in flags]
            flags = [flag.replace("{seed}", str(seed)) for flag in flags]
            if any(BKS_DIR in flag for flag in flags):
                raise SystemExit(f"arm {name!r} needs best-known solutions: prepare --bks-from DIR")
            runs.append({"name": f"{name}_s{seed}", "arm": name, "seed": seed, "flags": flags})
    return runs


def snapshot_source(source: Path) -> str:
    """Copy the code a job runs into ``source`` (with the uncommitted diff beside it); returns the commit."""
    source.mkdir()
    for directory in ("algo", "EVRoutingEnv", "scripts"):
        shutil.copytree(
            ROOT / directory,
            source / directory,
            ignore=shutil.ignore_patterns("__pycache__", "data", "baselines", "results", "*.out"),
        )
    shutil.copytree(ROOT / "EVRoutingEnv/baselines/heuristics", source / "EVRoutingEnv/baselines/heuristics")
    shutil.copytree(
        ROOT / "EVRoutingEnv/baselines/optimization",
        source / "EVRoutingEnv/baselines/optimization",
        ignore=shutil.ignore_patterns("__pycache__", "julia", "branch_and_price"),
    )
    shutil.copy2(ROOT / "EVRoutingEnv/baselines/__init__.py", source / "EVRoutingEnv/baselines/__init__.py")
    (source / "EVRoutingEnv/data").symlink_to(ROOT / "EVRoutingEnv/data", target_is_directory=True)
    source.with_name(f"{source.name}.diff").write_text(subprocess.check_output(["git", "diff"], cwd=ROOT, text=True))
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "algo", "scripts", "EVRoutingEnv"], cwd=ROOT, text=True,
    )
    source.with_name(f"{source.name}.untracked").write_text(untracked)
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def prepare(
    destination: Path,
    arm_set: str,
    recipe: str,
    seeds: int,
    references_from: Path | None = None,
    bks_from: Path | None = None,
    probes: bool = False,
    arms: list[str] | None = None,
    schedule: int = SCHEDULE,
    problems_dir: Path = SUITE,
    train_seeds: int = SEEDS["train"],
    exclude: list[str] | None = None,
    num_envs: int | None = None,
) -> None:
    if (destination / "manifest.json").exists():
        raise FileExistsError(f"immutable campaign already exists: {destination}")
    if train_seeds < 1:
        raise SystemExit("--train-seeds must be positive")
    if num_envs is not None and num_envs < 1:
        raise SystemExit("--num-envs must be positive")
    # The campaign's problems: the suite's five in their order, or every
    # configuration of --problems-dir minus --exclude; static where no requests
    # block is set.
    names = problem_names(problems_dir)
    excluded = sorted(set(exclude or ()))
    unknown = [name for name in excluded if name not in names]
    if unknown:
        raise SystemExit(f"--exclude names problems not in {problems_dir}: {unknown}")
    names = [name for name in names if name not in excluded]
    if not names:
        raise SystemExit("--exclude leaves no problem")
    dynamic = [name for name in names if not is_static(problems_dir / f"{name}.yaml")]
    static = [name for name in names if name not in dynamic]
    if references_from is not None:
        # Merged references only: teacher runs are deterministic per seed.
        wanted = ["summary.json", *(f"{name}.json" for name in names)]
        missing = [name for name in wanted if not (references_from / name).is_file()]
        if missing:
            raise SystemExit(f"{references_from} lacks merged references {missing}")
    if bks_from is not None:
        lacking = [f"{name}/{file}" for name in names for file in BKS_FILES if not (bks_from / name / file).is_file()]
        if lacking:
            raise SystemExit(f"{bks_from} lacks best-known solutions {lacking}")
    bks = str(destination / "bks") if bks_from is not None else None
    # Refused arms leave nothing behind.
    runs = arms_for(arm_set, recipe, seeds, bks=bks, arms=arms, problems=names, dynamic=dynamic)
    if any("--plan-conditioned" in run["flags"] for run in runs):
        lacking = [f"{name}/{file}" for name in static for file in PLAN_FILES if not (bks_from / name / file).is_file()]
        if lacking:
            raise SystemExit(f"plan-conditioned arms need the time-0 plan cache: {bks_from} lacks {lacking}")
    common = list(COMMON)
    common[common.index("--train-scenarios") + 1] = str(train_seeds)
    if num_envs is not None:
        # One sub-environment per problem per rollout slot; frozen like every other flag.
        common[common.index("--num-envs") + 1] = str(num_envs)
    num_envs = int(common[common.index("--num-envs") + 1])
    if num_envs % len(names):
        # balanced_slots refuses at training time; pass --num-envs (a multiple of the problem count).
        print(f"warning: --num-envs {num_envs} is not a multiple of the {len(names)} problems", flush=True)
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("logs", "training", "evaluation", "configs", "references"):
        (destination / name).mkdir(exist_ok=True)
    problems = []
    for name in names:
        target = destination / "configs" / f"{name}.yaml"
        shutil.copy2(problems_dir / f"{name}.yaml", target)
        problems.append({"name": name, "config": str(target), "teacher": "auto"})
    probe_rows = []
    for name, (source, trucks, stops) in (PROBES.items() if probes else ()):
        config = yaml.safe_load((SUITE / f"{source}.yaml").read_text())
        config["environment"]["num_trucks"] = trucks
        if stops is not None:
            config["environment"]["num_stops"] = stops
        target = destination / "configs" / f"{name}.yaml"
        target.write_text(yaml.safe_dump(config, sort_keys=False))
        probe_rows.append({"name": name, "config": str(target), "teacher": "auto"})
    write_json(destination / "problems.json", problems)
    write_json(destination / "probes.json", probe_rows)
    if references_from is not None:
        for name in wanted:
            shutil.copy2(references_from / name, destination / "references" / name)
    if bks_from is not None:
        # The imitation arms clone and pair against this frozen copy; the global
        # BKS directory may grow later (test seeds at P5) without touching it.
        for name in names:
            (destination / "bks" / name).mkdir(parents=True)
            for file in (*BKS_FILES, *BKS_OPTIONAL):
                if (bks_from / name / file).is_file():
                    shutil.copy2(bks_from / name / file, destination / "bks" / name / file)

    source = destination / "source"
    commit = snapshot_source(source)
    write_json(destination / "manifest.json", {
        "version": 2,
        "arm_set": arm_set,
        "recipe": recipe,
        "arms": list(arms) if arms else None,
        "commit": commit,
        "source": str(source),
        "python": str(ROOT / ".venv/bin/python"),
        "problems": str(destination / "problems.json"),
        "problems_dir": str(problems_dir),
        "excluded": excluded,
        "probes": str(destination / "probes.json"),
        "seeds": {**SEEDS, "train": int(train_seeds)},
        "reference_shard": SHARD,
        "schedule_timesteps": schedule,
        "common_flags": common,
        "runs": runs,
        "extra_baselines": EXTRA_BASELINES,
        "slurm": SLURM,
        "references_from": str(references_from) if references_from is not None else None,
        "bks_from": str(bks_from) if bks_from is not None else None,
        "bks": bks,
        "probes_enabled": bool(probes),
    })
    print(destination, flush=True)


# ---------------------------------------------------------------------------
# References: the reference (teacher) policy on train and validation seeds
# ---------------------------------------------------------------------------


def reference_tasks(manifest: dict) -> list[tuple[str, str, int]]:
    """(problem, split, shard) per array task; test is excluded until evaluation."""
    names = [row["name"] for row in json.loads(Path(manifest["problems"]).read_text())]
    tasks = []
    for split in ("train", "validation"):
        shards = -(-manifest["seeds"][split] // manifest["reference_shard"])
        tasks.extend((name, split, shard) for name in names for shard in range(shards))
    return tasks


def run_references(destination: Path, task: int, workers: int) -> None:
    from algo.generalist_ppo import SuiteProblem, run_reference_episodes
    from scripts.evaluation.canonical_harness import split_seeds

    manifest = read_manifest(destination)
    name, split, shard = reference_tasks(manifest)[task]
    problems = {row["name"]: SuiteProblem(**row) for row in json.loads(Path(manifest["problems"]).read_text())}
    seeds = split_seeds(split, manifest["seeds"][split])[shard * manifest["reference_shard"]:(shard + 1) * manifest["reference_shard"]]
    target = destination / "references" / "shards" / f"{name}_{split}_{shard:03d}.json"
    if target.exists():
        print(f"{target} exists", flush=True)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    rows = run_reference_episodes(problems[name], seeds, "teacher", workers=workers)
    partial = target.with_suffix(".partial")
    write_json(partial, rows)
    partial.replace(target)
    print(f"{name} {split} shard {shard}: {len(rows)} episodes in {time.perf_counter() - started:.0f}s "
          f"on {workers} workers -> {target}", flush=True)


def pending_reference_tasks(destination: Path, manifest: dict) -> list[int]:
    shards = destination / "references" / "shards"
    return [
        index for index, (name, split, shard) in enumerate(reference_tasks(manifest))
        if not (shards / f"{name}_{split}_{shard:03d}.json").exists()
    ]


def merge_references(destination: Path) -> None:
    manifest = read_manifest(destination)
    expected = reference_tasks(manifest)
    shards = destination / "references" / "shards"
    missing = [task for task in expected if not (shards / f"{task[0]}_{task[1]}_{task[2]:03d}.json").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} reference shards missing, e.g. {missing[:3]}")
    summary = {}
    for name in sorted({task[0] for task in expected}):
        rows = {}
        for _, split, shard in (task for task in expected if task[0] == name):
            for row in json.loads((shards / f"{name}_{split}_{shard:03d}.json").read_text()):
                rows[str(row["seed"])] = {**row, "split": split}
        write_json(destination / "references" / f"{name}.json", rows)
        values = list(rows.values())
        summary[name] = {
            "episodes": len(values),
            "failed": sum(row["failed"] for row in values),
            "success": sum(row["success"] for row in values),
            "median_hours": sorted(row["hours"] for row in values)[len(values) // 2],
            "mean_served_fraction": sum(row["served"] / max(1, row["tasks"]) for row in values) / len(values),
        }
    write_json(destination / "references" / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


# ---------------------------------------------------------------------------
# Training inside an allocation
# ---------------------------------------------------------------------------


def completed_run(destination: Path, run: dict, budget: int) -> bool:
    folder = destination / "training" / run["name"]
    required = ("training.pt", "training_history.json", "validation_history.json", "best.pt", "final.pt", "run_config.json")
    if any(not (folder / name).is_file() or (folder / name).stat().st_size == 0 for name in required):
        return False
    try:
        history = json.loads((folder / "training_history.json").read_text())
        return bool(history and history[-1]["timesteps"] >= budget)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def train(destination: Path, run_index: int, budget: int, device: str | None, build_store_only: bool = False) -> None:
    """Train one run; with ``build_store_only`` only replay its demonstrations into the campaign's shared store."""
    manifest = read_manifest(destination)
    run = manifest["runs"][run_index]
    if not build_store_only and completed_run(destination, run, budget):
        print(f"{run['name']} already reached {budget}", flush=True)
        return
    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus) < 3:
        raise SystemExit("at least three CPUs are required (trainer + two rollout workers)")
    output = destination / "training"
    resume = output / run["name"] / "training.pt"
    rollout = 40 * 128
    limit = -(-manifest["schedule_timesteps"] // rollout) * rollout
    if device is None:
        device = "cuda:0" if os.environ.get("CUDA_VISIBLE_DEVICES", "") not in ("", "-1") else "cpu"
    command = [
        manifest["python"], "scripts/training/train_generalist_ppo.py",
        "--problems", manifest["problems"],
        "--references", str(destination / "references"),
        *manifest["common_flags"],
        "--total-timesteps", str(limit),
        "--schedule-timesteps", str(manifest["schedule_timesteps"]),
        "--stop-after-timesteps", str(budget),
        "--rollout-workers", str(len(cpus) - 2),
        "--seed", str(run["seed"]),
        "--device", device,
        "--output", str(output),
        "--run-name", run["name"],
        *run["flags"],
    ]
    if "--demonstrations" in run["flags"]:
        # Replayed demonstrations are shared by every run and resume of the campaign (one store per
        # distinct replay, keyed by fingerprint): the replay ran 1 h 45 min per run on the broad set.
        command.extend(["--demonstration-store", str(destination / "demonstrations")])
    if (destination / SLOT_COSTS).exists():
        # Measured seconds per decision and per episode of each problem: slots are dealt to the
        # rollout workers by cost. Timing only; the file may be added or replaced between resumes.
        command.extend(["--slot-costs", str(destination / SLOT_COSTS)])
    if build_store_only:
        command.append("--build-demonstration-store-only")
    elif resume.exists():
        command.extend(["--resume", str(resume)])
    scratch = destination / "logs" / "runtime" / f"{os.environ.get('SLURM_JOB_ID', 'local')}_{run_index}"
    scratch.mkdir(parents=True, exist_ok=True)
    environment = dict(
        os.environ,
        PYTHONPATH=manifest["source"],
        OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
        TMPDIR=str(scratch), MPLCONFIGDIR=str(scratch / "matplotlib"),
        EVRP_SCENARIO_CACHE=str(scratch / "scenarios"),
        # Chunks of very different sizes follow each other on the device: growable segments keep
        # the allocator from fragmenting between them.
        PYTORCH_CUDA_ALLOC_CONF=os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
    )
    log_name = "demonstrations.log" if build_store_only else f"{run['name']}_{budget}.log"
    with (destination / "logs" / log_name).open("a") as handle:
        handle.write("COMMAND " + repr(command) + "\n")
        handle.flush()
        child = subprocess.Popen(command, cwd=manifest["source"], env=environment, stdout=handle, stderr=subprocess.STDOUT)

        def forward(signum, frame):
            if child.poll() is None:
                child.send_signal(signal.SIGUSR1)

        signal.signal(signal.SIGUSR1, forward)
        signal.signal(signal.SIGTERM, forward)
        code = child.wait()
    if code:
        raise SystemExit(code)


# ---------------------------------------------------------------------------
# Packing several training runs into one allocation
# ---------------------------------------------------------------------------


def cpu_slices(cpus, count: int, minimum: int = 3) -> list[list[int]]:
    """Split the allocation's CPUs into ``count`` disjoint contiguous slices.

    Contiguous ids keep a run's rollout workers on one socket where possible;
    the remainder goes to the first slices.
    """
    cpus = sorted(int(cpu) for cpu in cpus)
    if count < 1:
        raise ValueError("at least one slice is required")
    if len(cpus) < minimum * count:
        raise SystemExit(f"{len(cpus)} CPUs cannot give {count} runs {minimum} CPUs each")
    width, extra = divmod(len(cpus), count)
    slices, start = [], 0
    for position in range(count):
        end = start + width + (position < extra)
        slices.append(cpus[start:end])
        start = end
    return slices


def visible_gpus() -> list[str]:
    value = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    return [] if value in ("", "-1", "NoDevFiles") else [item for item in value.split(",") if item]


def gpu_assignment(gpus: list[str], count: int, runs_per_gpu: int) -> list[str | None]:
    """The CUDA_VISIBLE_DEVICES entry of each run: runs ``i*R .. i*R+R-1`` share GPU ``i``."""
    if not gpus:
        return [None] * count
    if runs_per_gpu < 1:
        raise ValueError("runs_per_gpu must be positive")
    needed = -(-count // runs_per_gpu)
    if len(gpus) < needed:
        raise SystemExit(f"{count} runs at {runs_per_gpu} per GPU need {needed} GPUs; {len(gpus)} are visible")
    return [gpus[position // runs_per_gpu] for position in range(count)]


def plan_pack(
    destination: Path,
    run_indices: list[int],
    budget: int,
    runs_per_gpu: int,
    cpus,
    gpus: list[str],
    device: str | None = None,
) -> list[dict]:
    """One child per unfinished run: its command, GPU and disjoint CPU slice."""
    manifest = read_manifest(destination)
    pending = [index for index in run_indices if not completed_run(destination, manifest["runs"][index], budget)]
    if not pending:
        return []
    slices = cpu_slices(cpus, len(pending))
    assigned = gpu_assignment(gpus, len(pending), runs_per_gpu)
    plan = []
    for index, cpu_set, gpu in zip(pending, slices, assigned, strict=True):
        command = [
            sys.executable, str(Path(__file__).resolve()), "train",
            "--directory", str(destination), "--run", str(index), "--budget", str(budget),
        ]
        if device is not None:
            command.extend(["--device", device])
        plan.append({"run": index, "name": manifest["runs"][index]["name"], "cpus": cpu_set, "gpu": gpu, "command": command})
    return plan


def supervise(plan: list[dict], cwd: str | Path | None = None) -> list[int]:
    """Start every child pinned to its slice; forward USR1/TERM to all as USR1.

    Every trainer then checkpoints at its next update and exits, so the next
    submission resumes all of them from their ``training.pt``.
    """
    children = []
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGUSR1, signal.SIGTERM)}

    def forward(signum, frame):
        for _, child in children:
            if child.poll() is None:
                child.send_signal(signal.SIGUSR1)

    for sig in previous:
        signal.signal(sig, forward)
    try:
        for item in plan:
            environment = dict(os.environ)
            if item["gpu"] is not None:
                environment["CUDA_VISIBLE_DEVICES"] = str(item["gpu"])
            child = subprocess.Popen(
                item["command"], cwd=cwd, env=environment,
                preexec_fn=functools.partial(os.sched_setaffinity, 0, item["cpus"]),
            )
            children.append((item, child))
            print(f"[pack] {item['name']}: pid {child.pid} gpu {item['gpu']} cpus {compact(item['cpus'])}", flush=True)
        codes = [child.wait() for _, child in children]
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    for (item, _), code in zip(children, codes, strict=True):
        print(f"[pack] {item['name']}: exit {code}", flush=True)
    return codes


def compact(cpus) -> str:
    """CPU ids as ranges, e.g. ``0-15,32``."""
    parts, cpus = [], sorted(cpus)
    start = previous = None
    for cpu in [*cpus, None]:
        if cpu is not None and previous is not None and cpu == previous + 1:
            previous = cpu
            continue
        if start is not None:
            parts.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = cpu
    return ",".join(parts)


def train_pack(destination: Path, run_indices: list[int], budget: int, runs_per_gpu: int, device: str | None) -> None:
    plan = plan_pack(destination, run_indices, budget, runs_per_gpu, os.sched_getaffinity(0), visible_gpus(), device)
    if not plan:
        print(f"runs {run_indices} already reached {budget}", flush=True)
        return
    codes = supervise(plan, cwd=read_manifest(destination)["source"])
    if any(codes):
        raise SystemExit(1)


def claim_shard(path: Path, stale_seconds: float = 1800.0) -> bool:
    """Atomically claim a shard; a claim older than ``stale_seconds`` is taken over.

    A shard takes a few minutes on one node, so an old claim belongs to a job
    that died (for example at its time limit).
    """
    owner = f"{os.environ.get('SLURM_JOB_ID', 'local')}:{os.getpid()} {time.time():.0f}\n"
    try:
        handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            age = time.time() - path.stat().st_mtime
        except FileNotFoundError:
            return claim_shard(path, stale_seconds)
        if age < stale_seconds:
            return False
        fresh = path.with_name(f"{path.name}.{os.getpid()}")
        fresh.write_text(owner)
        fresh.replace(path)
        return True
    with os.fdopen(handle, "w") as stream:
        stream.write(owner)
    return True


def reference_pack(destination: Path, tasks: list[int], workers: int, steal: bool = True) -> None:
    """Shards one after another, each on every core of the allocation.

    The job's own share comes first; with ``steal`` it then takes every shard
    that is still unclaimed, so the first job to start finishes the stage and
    jobs that start later find little left. Claims keep jobs off each other's shards.
    """
    manifest = read_manifest(destination)
    table = reference_tasks(manifest)
    shards = destination / "references" / "shards"
    shards.mkdir(parents=True, exist_ok=True)
    order = list(tasks)
    if steal:
        order += [task for task in pending_reference_tasks(destination, manifest) if task not in order]
    skipped, failed = [], []
    for task in order:
        name, split, shard = table[task]
        stem = shards / f"{name}_{split}_{shard:03d}"
        if stem.with_suffix(".json").exists():
            continue
        claim = stem.with_suffix(".claim")
        if not claim_shard(claim):
            skipped.append(task)
            continue
        try:
            run_references(destination, task, workers)
        except Exception as error:  # noqa: BLE001 - one shard's failure must not end the pack
            # The shard stays pending (no result file); the rest of the pack goes on.
            failed.append((task, f"{name} {split} shard {shard}", f"{type(error).__name__}: {error}"))
            print(f"FAILED {name} {split} shard {shard} (task {task}): {type(error).__name__}: {error}", flush=True)
        finally:
            claim.unlink(missing_ok=True)
    if skipped:
        print(f"left to the jobs that claimed them: tasks {skipped}", flush=True)
    if failed:
        raise SystemExit("reference shards failed (still pending): " + "; ".join(f"{label} (task {task})" for task, label, _ in failed))


def pack_groups(items: list[int], size: int) -> list[list[int]]:
    """Consecutive groups of ``size`` (the last may be shorter)."""
    if size < 1:
        raise SystemExit("--pack must be positive")
    return [items[start:start + size] for start in range(0, len(items), size)]


def strided_groups(items: list[int], count: int) -> list[list[int]]:
    """``count`` groups taking every count-th item, so each gets a mix of problems."""
    if count < 1:
        raise SystemExit("--pack must be positive")
    return [group for group in (items[start::count] for start in range(count)) if group]


# ---------------------------------------------------------------------------
# Submission
# ---------------------------------------------------------------------------


def train_hours(budget: int) -> int:
    # About 250 decisions/s per A100 share at 16 rollout workers, plus behaviour
    # cloning on up to 256 episodes per problem, DAgger rounds and validation.
    # A run that overruns checkpoints on USR1 and resumes at the next submission.
    return 3 if budget <= 250_000 else 12 if budget <= 2_000_000 else 24 if budget <= 6_000_000 else 36


def submit(
    destination: Path,
    stage: str,
    budget: int | None,
    dry_run: bool,
    arms: list[str] | None = None,
    split: str = "validation",
    tag: str | None = None,
    runs: list[str] | None = None,
    pack: int | None = None,
    runs_per_gpu: int = 1,
    cpus: int | None = None,
    partition: str | None = None,
    cpus_per_gpu: int | None = None,
    test_only: bool = False,
    time_limit: int | None = None,
    decision: dict | None = None,
    dependency: str | None = None,
    mem_per_cpu: str | None = None,
) -> None:
    """Compose (and unless ``dry_run``, submit) the sbatch call(s) of a stage.

    ``--pack`` puts several units of work in one allocation: K training runs on
    one node (ceil(K/R) GPUs, K CPU slices), or J reference jobs that share the
    pending shards. Each array element receives its group as an argument,
    ``0,1`` for runs 0 and 1. ``partition`` moves a GPU stage to an overflow
    partition (with its own cores per GPU); ``test_only`` runs every command
    through ``sbatch --test-only``, which checks it without submitting;
    ``dependency`` is handed to sbatch as is (``afterany:JOBID`` holds the
    stage until that job ends, which keeps the allocation cap); ``mem_per_cpu``
    replaces the manifest's memory per core (a job is charged by the largest
    of its core, GPU and memory shares of the node, so twice the memory of a
    GPU share costs two shares).
    """
    manifest = read_manifest(destination)
    slurm = {**SLURM, **manifest["slurm"]}
    gpu_partition, per_gpu = slurm["gpu_partition"], slurm["cpus_per_gpu"]
    if partition is not None and partition != gpu_partition:
        overflow = slurm.get("overflow", {})
        if partition not in overflow:
            raise SystemExit(f"partition {partition!r} is neither {gpu_partition!r} nor one of {sorted(overflow)}")
        gpu_partition, per_gpu = partition, overflow[partition]
    if cpus_per_gpu is not None:
        per_gpu = cpus_per_gpu
    if mem_per_cpu is not None:
        slurm = {**slurm, "gpu_mem_per_cpu": mem_per_cpu, "cpu_mem_per_cpu": mem_per_cpu}
    throttle = slurm["max_concurrent"]
    common = [
        "sbatch", "--parsable", *(["--test-only"] if test_only else []),
        f"--account={slurm['account']}", f"--output={destination}/logs/slurm-%A_%a.out",
        *([f"--dependency={dependency}"] if dependency else []),
    ]
    cpu_gpus = [f"--gpus-per-task={slurm['cpu_gpus']}"] if slurm.get("cpu_gpus", 0) > 0 else []
    commands = []
    if stage == "references":
        pending = pending_reference_tasks(destination, manifest)
        if not pending:
            raise SystemExit("every reference shard exists; run merge-references")
        if pack:
            # Whole nodes, each working through its share of shards on all cores.
            groups = strided_groups(pending, pack)
            cores, hours = cpus or slurm["cpu_node_cores"], time_limit or 2
        else:
            groups = [[task] for task in pending]
            cores, hours = cpus or slurm["cpu_cores"], time_limit or 4
        commands.append((groups, [
            *common, f"--job-name={destination.name}_ref", f"--partition={slurm['cpu_partition']}",
            "--nodes=1", "--ntasks=1", f"--cpus-per-task={cores}", f"--mem-per-cpu={slurm['cpu_mem_per_cpu']}",
            *cpu_gpus, f"--time={hours:02d}:00:00",
            f"--array=0-{len(groups) - 1}%{throttle}",
            str(ROOT / "scripts/campaigns/generalist_graphppo_cpu.sbatch"), str(destination), "references",
            *(",".join(map(str, group)) for group in groups),
        ]))
    elif stage == "train":
        if budget is None or budget < 1:
            raise SystemExit("--budget is required for the train stage")
        if runs_per_gpu < 1:
            raise SystemExit("--runs-per-gpu must be positive")
        known = {run["arm"] for run in manifest["runs"]}
        if arms and not set(arms) <= known:
            raise SystemExit(f"unknown arms {sorted(set(arms) - known)}; the manifest has {sorted(known)}")
        pending = [
            index for index, run in enumerate(manifest["runs"])
            if (not arms or run["arm"] in arms) and not completed_run(destination, run, budget)
        ]
        if not pending:
            raise SystemExit(f"every selected run already reached {budget}")
        if not (destination / "references" / "summary.json").exists():
            raise SystemExit("merge the references first")
        # One array per group size, so a short last group asks only for what it uses.
        by_size: dict[int, list[list[int]]] = {}
        for group in pack_groups(pending, pack or 1):
            by_size.setdefault(len(group), []).append(group)
        for size, groups in sorted(by_size.items(), reverse=True):
            commands.append((groups, [
                *common, f"--job-name={destination.name}_train", f"--partition={gpu_partition}",
                "--nodes=1", "--ntasks=1", f"--cpus-per-task={size * per_gpu}",
                f"--mem-per-cpu={slurm['gpu_mem_per_cpu']}", f"--gpus-per-task={-(-size // runs_per_gpu)}",
                f"--time={time_limit or train_hours(budget):02d}:00:00", "--signal=B:USR1@300",
                f"--array=0-{len(groups) - 1}%{throttle}",
                str(ROOT / "scripts/campaigns/generalist_graphppo_gpu.sbatch"), str(destination), str(budget),
                str(runs_per_gpu), *(",".join(map(str, group)) for group in groups),
            ]))
    elif stage == "demonstrations":
        # One CPU-stage job replays the demonstrations of run 0 into the shared store before training.
        if not any("--demonstrations" in run["flags"] for run in manifest["runs"]):
            raise SystemExit("no run of this campaign clones demonstrations")
        commands.append(([[0]], [
            *common, f"--job-name={destination.name}_demos", f"--partition={slurm['cpu_partition']}",
            "--nodes=1", "--ntasks=1", f"--cpus-per-task={cpus or slurm['cpu_cores']}",
            f"--mem-per-cpu={slurm['cpu_mem_per_cpu']}", *cpu_gpus, f"--time={time_limit or 4:02d}:00:00",
            "--array=0-0",
            str(ROOT / "scripts/campaigns/generalist_graphppo_cpu.sbatch"), str(destination), "demonstrations", "0",
        ]))
    elif stage == "evaluate":
        if not tag:
            raise SystemExit("--tag is required for the evaluate stage")
        if runs:
            unknown = sorted(set(runs) - {run["name"] for run in manifest["runs"]})
            if unknown:
                raise SystemExit(f"unknown runs {unknown}")
        if decision and not (dry_run or test_only):
            write_decision(destination / "evaluation" / split / tag, decision, snapshot=True)
        # One allocation: the baselines, then every run's best.pt on the GPU.
        hours = time_limit or (12 if split == "validation" else 24)
        commands.append((None, [
            *common, f"--job-name={destination.name}_eval_{tag}", f"--partition={gpu_partition}",
            "--nodes=1", "--ntasks=1", f"--cpus-per-task={cpus or per_gpu}",
            f"--mem-per-cpu={slurm['gpu_mem_per_cpu']}", "--gpus-per-task=1", f"--time={hours:02d}:00:00",
            str(ROOT / "scripts/campaigns/generalist_graphppo_eval.sbatch"), str(destination), split, tag, *(runs or []),
        ]))
    else:
        raise SystemExit(f"unknown stage {stage!r}")
    for groups, command in commands:
        print(" ".join(command), flush=True)
        if dry_run:
            continue
        if test_only:
            # The scheduler validates the request and reports when it could
            # start; nothing is submitted or recorded.
            result = subprocess.run(command, text=True, capture_output=True)
            print((result.stdout + result.stderr).strip(), flush=True)
            if result.returncode:
                raise SystemExit(result.returncode)
            continue
        job = subprocess.check_output(command, text=True).strip()
        with (destination / "logs" / "submissions.jsonl").open("a") as handle:
            handle.write(json.dumps({
                "stage": stage, "budget": budget, "tag": tag, "job": job, "groups": groups,
                "runs_per_gpu": runs_per_gpu if stage == "train" else None, "command": command,
            }) + "\n")
        print(f"submitted {job}", flush=True)


# ---------------------------------------------------------------------------
# Evaluation and report
# ---------------------------------------------------------------------------


DECISION = "decision.json"


def write_decision(folder: Path, settings: dict, snapshot: bool = False) -> None:
    """Freeze a tag's decision-time settings (track A: feasibility mask, search); evaluate applies them.

    With ``snapshot`` the current code is copied into ``<tag>/source`` and the
    evaluation wrapper runs that copy instead of the campaign's training
    snapshot, which predates track A.
    """
    path = folder / DECISION
    if snapshot:
        settings = {**settings, "source": str(folder / "source")}
    if path.exists() and json.loads(path.read_text()) != settings:
        raise SystemExit(f"{path} already holds other settings {json.loads(path.read_text())}; use a new tag")
    folder.mkdir(parents=True, exist_ok=True)
    if snapshot and not (folder / "source").exists():
        snapshot_source(folder / "source")
    write_json(path, settings)


def apply_decision(folder: Path) -> dict:
    """Set the environment of a tag's decision-time settings for this process and its forked workers."""
    settings = json.loads((folder / DECISION).read_text()) if (folder / DECISION).exists() else {}
    os.environ["EVRP_SAFETY_MASK"] = "on" if settings.get("safety_mask") else "off"
    os.environ["EVRP_SAFETY_MARGIN"] = str(float(settings.get("safety_margin", 0.0)))
    os.environ["EVRP_RETIRE_GUARD"] = "on" if settings.get("retire_guard") else "off"
    os.environ["EVRP_PUBLIC_WAKE"] = "on" if settings.get("public_wake") else "off"
    return settings


CHECKPOINT_LABELS = ("best", "pretrained", "final")


def checkpoint_label(value: str) -> str:
    """evaluate's ``--checkpoint``: a named checkpoint or a ``ckpt_<update>`` the trainer saved at a validation."""
    if value in CHECKPOINT_LABELS or re.fullmatch(r"ckpt_\d+", value):
        return value
    raise argparse.ArgumentTypeError(f"checkpoint must be one of {CHECKPOINT_LABELS} or ckpt_<update>, not {value!r}")


def decision_from(
    safety_mask: bool, safety_margin: float | None, search: str | None = None, k: int = 4, n: int = 16,
    retire_guard: bool = False, checkpoint: str | None = None, batch: int | None = None, public_wake: bool = False,
) -> dict | None:
    if safety_margin is not None and not safety_mask:
        raise SystemExit("--safety-margin needs --safety-mask")
    if retire_guard and not safety_mask:
        raise SystemExit("--retire-guard needs --safety-mask")
    settings = {}
    if safety_mask:
        settings.update(safety_mask=True, safety_margin=float(safety_margin or 0.0))
    if retire_guard:
        settings.update(retire_guard=True)
    if checkpoint and checkpoint != "best":
        settings.update(checkpoint=checkpoint)
    if batch is not None:
        settings.update(envs_per_problem=int(batch))
    if public_wake:
        settings.update(public_wake=True)
    if search:
        settings.update(search=search, k=int(k) if search == "lookahead" else None, n=int(n) if search == "best_of_n" else None)
    return settings or None


def evaluate(
    destination: Path,
    split: str,
    tag: str,
    count: int | None,
    workers: int,
    device: str,
    runs: list[str] | None,
    envs_per_problem: int = 16,
    plan_mode: str | None = None,
    decision: dict | None = None,
) -> None:
    import torch

    from algo.generalist_policy import GeneralistActorCritic, GeneralistPolicyConfig
    from algo.generalist_ppo import (
        ReferenceTable,
        SuiteProblem,
        SuiteReward,
        evaluate_policy_suite,
        evaluation_envs,
        run_reference_episodes,
    )
    from scripts.evaluation.canonical_harness import split_seeds

    manifest = read_manifest(destination)
    # Plan-conditioned runs: plans on (no dropout) unless the plan-free mode is
    # asked for, explicitly or by a tag ending in "_noplan"; static plans from the
    # frozen cache (validation) or the global BKS directory (test, built at P5).
    plan_mode = plan_mode or ("off" if tag.endswith("_noplan") else "on")
    cache = manifest.get("bks_from") if split == "test" else manifest.get("bks")
    if cache:
        os.environ["EVRP_PLAN_CACHE"] = str(cache)
    os.environ["EVRP_PLAN_MODE"] = plan_mode
    os.environ["EVRP_PLAN_DROPOUT"] = "0"
    problems = [SuiteProblem(**row) for row in json.loads(Path(manifest["problems"]).read_text())]
    problems += [SuiteProblem(**row) for row in json.loads(Path(manifest["probes"]).read_text())]
    count = count or manifest["seeds"][split]
    seeds = split_seeds(split, count)
    folder = destination / "evaluation" / split
    (folder / tag).mkdir(parents=True, exist_ok=True)
    if decision:
        write_decision(folder / tag, decision)
    applied = apply_decision(folder / tag)
    if applied:
        print(f"decision-time settings ({tag}): {applied}", flush=True)
    envs_per_problem = applied.get("envs_per_problem", envs_per_problem)
    # Baselines on exactly these seeds; the teacher rows are also the
    # per-scenario references of the score.
    for problem in problems:
        for policy in ["teacher", *manifest["extra_baselines"].get(problem.name, [])]:
            target = folder / "baselines" / f"{problem.name}__{policy}.json"
            if target.exists() and {row["seed"] for row in json.loads(target.read_text())} == set(seeds):
                continue
            target.parent.mkdir(exist_ok=True)
            rows = run_reference_episodes(problem, seeds, policy, workers=workers)
            for row in rows:
                row.pop("actions", None)
            partial = target.with_suffix(".partial")  # a concurrent evaluation never reads a half-written file
            write_json(partial, rows)
            partial.replace(target)
            print(f"baseline {problem.name} {policy}: {len(rows)}", flush=True)
    references = ReferenceTable({
        problem.name: {
            row["seed"]: row for row in json.loads((folder / "baselines" / f"{problem.name}__teacher.json").read_text())
        }
        for problem in problems
    })
    reward = SuiteReward(reference="scenario")
    pending = [
        run for run in manifest["runs"]
        if (not runs or run["name"] in runs)
        and (destination / "training" / run["name"] / "best.pt").exists()
        and not (folder / tag / f"{run['name']}.json").exists()
    ]
    # Track A search plays the static problems episode by episode on the CPU;
    # the other problems (and every problem without a search) go batched to the GPU.
    search = applied.get("search")
    static = set(static_names(read_problems(manifest))) if search else set()
    searched = [problem for problem in problems if problem.name in static]
    batched = [problem for problem in problems if problem not in searched]
    # Workers are forked per observation spec before any policy touches CUDA.
    specs = {
        run["name"]: GeneralistPolicyConfig.load(destination / "training" / run["name"] / "best_config.json").observation_spec
        for run in pending
    }
    envs = {
        spec: evaluation_envs(
            batched, [seeds] * len(batched), spec, reward, references,
            workers=workers, envs_per_problem=envs_per_problem,
        )
        for spec in set(specs.values())
    } if batched else {}
    searched_rows = {}
    if searched and pending:
        from algo.generalist_search import evaluate_search

        jobs = [
            (run["name"], str(destination / "training" / run["name"]), problem.name, problem.config, seed,
             references.scale(problem.name, seed, reward.reference)[0], search, applied.get("k"), applied.get("n"), reward)
            for run in pending for problem in searched for seed in seeds
        ]
        # One episode file per job's runs, so a crashed or timed-out search resumes.
        partial = folder / tag / f"search_{'_'.join(sorted(run['name'] for run in pending))}.partial.jsonl"
        searched_rows = evaluate_search(jobs, workers, partial=partial)
    try:
        for run in pending:
            rows = []
            if batched:
                checkpoint = destination / "training" / run["name"]
                policy = GeneralistActorCritic.load(checkpoint, prefix=applied.get("checkpoint", "best"), map_location=device).to(device)
                with torch.no_grad():
                    rows = evaluate_policy_suite(
                        policy, batched, [seeds] * len(batched), references, reward,
                        device=device, envs=envs[specs[run["name"]]],
                    )
            rows = [*rows, *sorted(searched_rows.get(run["name"], []), key=lambda row: (row["problem"], row["seed"]))]
            write_json(folder / tag / f"{run['name']}.json", rows)
            print(f"{run['name']} ({tag}): {len(rows)} episodes", flush=True)
    finally:
        for value in envs.values():
            value.close()


PROMOTION_KEYS = ("score", "served_fraction", "failure_rate", "success_rate", "hours")


def problem_stats(rows: list[dict], name: str) -> dict | None:
    """Mean served fraction, failure, success, hours and score of ``name``'s rows (score may be absent)."""
    group = [row for row in rows if row.get("problem") == name]
    if not group:
        return None

    def mean(values) -> float:
        values = list(values)
        return float(sum(values) / len(values))

    scores = [float(row["score"]) for row in group if "score" in row]
    return {
        "score": mean(scores) if len(scores) == len(group) else float("nan"),
        "served_fraction": mean(row["served"] / max(1, row["tasks"]) for row in group),
        "failure_rate": mean(float(bool(row["failed"])) for row in group),
        "success_rate": mean(float(bool(row["success"])) for row in group),
        "hours": mean(float(row["hours"]) for row in group),
    }


def promote(destination: Path, split: str, tag: str, count: int) -> dict:
    """Arm means over seeds on the tag's rows, eligibility against the teacher, the top ``count``.

    Eligible: on every problem, the arm's served fraction is at least the
    teacher's and its failure rate at most the teacher's. Promoted: the
    ``count`` eligible arms with the highest mean selection score. Written to
    ``evaluation/<split>/<tag>/promotion.json``.
    """
    manifest = read_manifest(destination)
    folder = destination / "evaluation" / split
    names = [row["name"] for row in json.loads(Path(manifest["problems"]).read_text())]
    teacher = {}
    for name in names:
        path = folder / "baselines" / f"{name}__teacher.json"
        if not path.exists():
            raise SystemExit(f"no teacher baseline for {name} in {folder / 'baselines'}")
        summary = problem_stats(json.loads(path.read_text()), name)
        if summary is None:
            raise SystemExit(f"the teacher baseline for {name} has no rows")
        teacher[name] = {"served_fraction": summary["served_fraction"], "failure_rate": summary["failure_rate"]}
    by_arm: dict[str, dict[str, dict]] = {}
    for run in manifest["runs"]:
        path = folder / tag / f"{run['name']}.json"
        if not path.exists():
            continue
        rows = json.loads(path.read_text())
        per_problem = {}
        for name in names:
            summary = problem_stats(rows, name)
            if summary is not None:
                per_problem[name] = summary
        by_arm.setdefault(run["arm"], {})[run["name"]] = per_problem
    result = {"split": split, "tag": tag, "count": count, "teacher": teacher, "arms": {}, "promoted": []}
    for arm, runs in by_arm.items():
        problems = {}
        for name in names:
            values = [per[name] for per in runs.values() if name in per]
            if values:
                problems[name] = {key: sum(v[key] for v in values) / len(values) for key in PROMOTION_KEYS}
        reasons = []
        incomplete = sorted(run for run, per in runs.items() if set(per) != set(names))
        if incomplete:
            reasons.append(f"rows missing for some problem in {incomplete}")
        for name, value in problems.items():
            if value["served_fraction"] < teacher[name]["served_fraction"] - 1e-9:
                reasons.append(f"{name}: served {value['served_fraction']:.3f} < teacher {teacher[name]['served_fraction']:.3f}")
            if value["failure_rate"] > teacher[name]["failure_rate"] + 1e-9:
                reasons.append(f"{name}: failure {value['failure_rate']:.3f} > teacher {teacher[name]['failure_rate']:.3f}")
        mean_score = sum(p["score"] for p in problems.values()) / len(problems) if problems else float("-inf")
        result["arms"][arm] = {
            "runs": sorted(runs), "seeds": len(runs), "problems": problems, "mean_score": mean_score,
            "eligible": not reasons, "reasons": reasons,
        }
    eligible = sorted((arm for arm, entry in result["arms"].items() if entry["eligible"]),
                      key=lambda arm: -result["arms"][arm]["mean_score"])
    result["promoted"] = eligible[:count]
    write_json(folder / tag / "promotion.json", result)
    return result


def promotion_lines(result: dict) -> list[str]:
    lines = [
        "## Promotion", "",
        "Arm means over seeds. Eligible: on every problem, served fraction >= the teacher's and failure rate "
        f"<= the teacher's. Promoted (top {result['count']} eligible by mean score): "
        f"{', '.join(result['promoted']) or 'none'}.", "",
        "| Arm | Seeds | Mean score | Eligible | Reasons |", "| --- | --- | --- | --- | --- |",
    ]
    order = sorted(result["arms"], key=lambda arm: -result["arms"][arm]["mean_score"])
    for arm in order:
        entry = result["arms"][arm]
        lines.append(f"| {arm} | {entry['seeds']} | {entry['mean_score']:.3f} | {'yes' if entry['eligible'] else 'no'} | "
                     f"{'; '.join(entry['reasons'])} |")
    lines.append("")
    return lines


def report(destination: Path, split: str, tag: str, promote_count: int | None = None) -> None:
    from algo.generalist_ppo import ReferenceTable, SuiteReward, summarize_suite
    from EVRoutingEnv.evaluation.statistics import paired_bootstrap_difference

    manifest = read_manifest(destination)
    folder = destination / "evaluation" / split
    names = [row["name"] for row in json.loads(Path(manifest["problems"]).read_text())]
    names += [row["name"] for row in json.loads(Path(manifest["probes"]).read_text())]
    teacher = {
        name: {row["seed"]: row for row in json.loads((folder / "baselines" / f"{name}__teacher.json").read_text())}
        for name in names if (folder / "baselines" / f"{name}__teacher.json").exists()
    }
    references, reward = ReferenceTable(teacher), SuiteReward(reference="scenario")
    methods = {}
    for path in sorted((folder / "baselines").glob("*.json")):
        name, policy = path.stem.split("__")
        rows = json.loads(path.read_text())
        for row in rows:
            row["score"] = reward.score(row, references.scale(name, row["seed"], "scenario")[0])
        methods.setdefault(f"baseline:{policy}", []).extend(rows)
    for run in manifest["runs"]:
        path = folder / tag / f"{run['name']}.json"
        if path.exists():
            methods.setdefault(run["arm"], []).extend({**row, "run": run["name"]} for row in json.loads(path.read_text()))
    lines = [f"# Generalist GraphPPO, {split}, {tag}", "", f"Campaign `{destination.name}`, arm set `{manifest['arm_set']}`. "
             "Score: reference-normalized return against the greedy teacher on the same seed (higher is better). "
             "Hours: mean operating hours. Δhours: paired mean vs teacher on seeds where both completed (static) "
             "or on all seeds (dynamic), 95 % bootstrap interval.", ""]
    for name in names:
        lines += [f"## {name}", "", "| Method | Episodes | Score | Served | Success | Failures | Hours | Δhours vs teacher |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for method, rows in methods.items():
            summary = summarize_suite(rows, [name]).get(name)
            if summary is None:
                continue
            group = [row for row in rows if row["problem"] == name]
            pairs = [(teacher[name][row["seed"]], row) for row in group if row["seed"] in teacher.get(name, {})]
            pairs = [(a, b) for a, b in pairs if a["dynamic"] or (a["success"] and b["success"])]
            delta = "n/a"
            if len(pairs) >= 2 and not method.endswith(":teacher"):
                result = paired_bootstrap_difference([a["hours"] for a, _ in pairs], [b["hours"] for _, b in pairs], resamples=2000)
                delta = f"{result['mean_difference']:+.1f} [{result['ci_low']:+.1f}, {result['ci_high']:+.1f}]"
            lines.append(
                f"| {method} | {summary['episodes']} | {summary['score']:.3f} | {summary['served_fraction']:.3f} | "
                f"{summary['success_rate']:.3f} | {summary['failure_rate']:.3f} | {summary['hours']:.1f} | {delta} |"
            )
        lines.append("")
    if promote_count:
        lines += promotion_lines(promote(destination, split, tag, promote_count))
    (folder / tag).mkdir(parents=True, exist_ok=True)
    (folder / tag / "report.md").write_text("\n".join(lines))
    print("\n".join(lines), flush=True)


def _indices(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("prepare")
    command.add_argument("--directory", type=Path, default=DEFAULT)
    command.add_argument("--arm-set", default="screen", choices=["screen", "named", "lopo", "specialists"])
    command.add_argument("--arms", nargs="*", default=None, help="named: the recipes to run, in this order.")
    command.add_argument("--recipe", default="full", choices=sorted(RECIPES))
    command.add_argument("--seeds", type=int, default=3)
    command.add_argument("--references-from", type=Path, default=None, help="Merged references directory to copy.")
    command.add_argument("--bks-from", type=Path, default=None,
                         help="Best-known solutions (<problem>/{validation,train_demonstrations}.json) to freeze.")
    command.add_argument("--probes", action=argparse.BooleanOptionalAction, default=False,
                         help="Also write and evaluate the zero-shot size probes.")
    command.add_argument("--schedule", type=int, default=SCHEDULE,
                         help="Timesteps over which the learning rate decays to zero (frozen in the manifest).")
    command.add_argument("--problems-dir", type=Path, default=SUITE,
                         help="Train on every *.yaml of this directory (default: the suite's five problems); "
                              "a problem without a requests block is static.")
    command.add_argument("--train-seeds", type=int, default=SEEDS["train"],
                         help="Train seeds per problem (references, plan caches, --train-scenarios).")
    command.add_argument("--exclude", nargs="*", default=(), metavar="PROBLEM",
                         help="Leave these problems of --problems-dir out of the campaign (recorded in the manifest).")
    command.add_argument("--num-envs", type=int, default=None,
                         help="Sub-environments per rollout step (frozen in common_flags; a multiple of the problem count).")
    command = commands.add_parser("references")
    command.add_argument("--directory", type=Path, default=DEFAULT)
    group = command.add_mutually_exclusive_group(required=True)
    group.add_argument("--task", type=int, help="One shard index.")
    group.add_argument("--tasks", type=_indices, help="Comma-separated shard indices, run one after another.")
    command.add_argument("--workers", type=int, default=max(1, len(os.sched_getaffinity(0)) - 1))
    command = commands.add_parser("merge-references")
    command.add_argument("--directory", type=Path, default=DEFAULT)
    command = commands.add_parser("train")
    command.add_argument("--directory", type=Path, default=DEFAULT)
    command.add_argument("--run", type=int, required=True, help="Index into manifest runs.")
    command.add_argument("--budget", type=int, required=True)
    command.add_argument("--device", default=None)
    command = commands.add_parser("demonstrations", help="Replay the demonstrations of one run into the shared store.")
    command.add_argument("--directory", type=Path, default=DEFAULT)
    command.add_argument("--run", type=int, default=0, help="Index into manifest runs (its imitation flags).")
    command.add_argument("--device", default="cpu")
    command.add_argument("--tasks", default=None, help=argparse.SUPPRESS)  # the CPU wrapper passes its group
    command.add_argument("--workers", type=int, default=None, help=argparse.SUPPRESS)
    command = commands.add_parser("train-pack", help="Several runs in one allocation, each on its own GPU and CPU slice.")
    command.add_argument("--directory", type=Path, default=DEFAULT)
    command.add_argument("--runs", type=_indices, required=True, help="Comma-separated indices into manifest runs.")
    command.add_argument("--budget", type=int, required=True)
    command.add_argument("--runs-per-gpu", type=int, default=1)
    command.add_argument("--device", default=None)
    command = commands.add_parser("submit")
    command.add_argument("--directory", type=Path, default=DEFAULT)
    command.add_argument("--stage", required=True, choices=["references", "demonstrations", "train", "evaluate"])
    command.add_argument("--budget", type=int, default=None)
    command.add_argument("--arms", nargs="*", default=None, help="train: only the runs of these arms.")
    command.add_argument("--split", default="validation", choices=["validation", "test"])
    command.add_argument("--tag", default=None, help="evaluate: checkpoint label, e.g. 2M, 8M, final.")
    command.add_argument("--runs", nargs="*", default=None, help="evaluate: only these runs.")
    command.add_argument("--pack", type=int, default=None,
                         help="train: K runs per job (one node). references: J whole-node jobs sharing the shards.")
    command.add_argument("--runs-per-gpu", type=int, default=1, help="train with --pack: runs sharing one GPU.")
    command.add_argument("--cpus", type=int, default=None, help="references/evaluate: CPUs per job instead of the manifest's.")
    command.add_argument("--partition", default=None, help="train/evaluate: an overflow GPU partition from the manifest.")
    command.add_argument("--cpus-per-gpu", type=int, default=None, help="train/evaluate: CPUs per GPU instead of the partition's.")
    command.add_argument("--dry-run", action="store_true", help="Print the commands only.")
    command.add_argument("--test-only", action="store_true", help="Run each command through sbatch --test-only.")
    command.add_argument("--mem-per-cpu", default=None, metavar="SIZE",
                         help="Memory per core instead of the manifest's (e.g. 13000M: twice a GPU share, charged as two).")
    command.add_argument("--dependency", default=None,
                         help="sbatch --dependency value, e.g. afterany:JOBID (wait for a job to end).")
    command.add_argument("--hours", type=int, default=None,
                         help="train/evaluate: time limit in hours instead of the default for the budget or split.")
    command.add_argument("--safety-mask", action="store_true",
                         help="evaluate: decision-time feasibility mask (track A), frozen in <tag>/decision.json.")
    command.add_argument("--safety-margin", type=float, default=None,
                         help="evaluate with --safety-mask: hours to be home before the horizon.")
    command.add_argument("--retire-guard", action="store_true",
                         help="evaluate with --safety-mask: no retirement of a truck the plan still routes customers to.")
    command.add_argument("--checkpoint", default=None, type=checkpoint_label,
                         help="evaluate: the checkpoint of every run (default best; also ckpt_<update> saved at a "
                              "validation), frozen in <tag>/decision.json.")
    command.add_argument("--eval-batch", type=int, default=None,
                         help="evaluate: environments per problem, frozen in <tag>/decision.json (robustness to batching).")
    command.add_argument("--public-wake", action="store_true",
                         help="evaluate: a waiting truck's wake-up from public information only, frozen in <tag>/decision.json.")
    command.add_argument("--search", default=None, choices=["lookahead", "best_of_n"],
                         help="evaluate: decision-time search on the static problems (track A).")
    command.add_argument("--search-k", type=int, default=4, help="lookahead: candidates per decision.")
    command.add_argument("--search-n", type=int, default=16, help="best_of_n: sampled rollouts.")
    command = commands.add_parser("evaluate")
    command.add_argument("--directory", type=Path, default=DEFAULT)
    command.add_argument("--split", default="validation", choices=["validation", "test"])
    command.add_argument("--tag", required=True, help="Checkpoint label, e.g. 2M, 8M, final.")
    command.add_argument("--count", type=int, default=None, help="Seeds per problem (default: manifest).")
    command.add_argument("--runs", nargs="*", default=None)
    command.add_argument("--workers", type=int, default=max(1, len(os.sched_getaffinity(0)) - 1))
    command.add_argument("--envs-per-problem", type=int, default=16)
    command.add_argument("--device", default="cpu")
    command.add_argument("--plan-mode", default=None, choices=["on", "off"],
                         help="Plan-conditioned runs: evaluate with or without the plan (default: off for tags ending in _noplan).")
    command.add_argument("--safety-mask", action="store_true", help="Decision-time feasibility mask (track A).")
    command.add_argument("--safety-margin", type=float, default=None, help="With --safety-mask: hours to be home before the horizon.")
    command.add_argument("--retire-guard", action="store_true", help="With --safety-mask: no retirement of a truck the plan still routes customers to.")
    command.add_argument("--checkpoint", default=None, choices=["best", "pretrained", "final"], help="The checkpoint of every run (default best).")
    command.add_argument("--eval-batch", type=int, default=None, help="Environments per problem, frozen with the tag.")
    command.add_argument("--public-wake", action="store_true", help="A waiting truck's wake-up from public information only.")
    command.add_argument("--search", default=None, choices=["lookahead", "best_of_n"], help="Decision-time search (track A).")
    command.add_argument("--search-k", type=int, default=4)
    command.add_argument("--search-n", type=int, default=16)
    command = commands.add_parser("report")
    command.add_argument("--directory", type=Path, default=DEFAULT)
    command.add_argument("--split", default="validation", choices=["validation", "test"])
    command.add_argument("--tag", required=True)
    command.add_argument("--promote", type=int, default=None,
                         help="Append the promotion table: the N eligible arms with the best mean score.")
    arguments = parser.parse_args()
    destination = arguments.directory.resolve()
    if arguments.command == "prepare":
        references_from = arguments.references_from.resolve() if arguments.references_from else None
        bks_from = arguments.bks_from.resolve() if arguments.bks_from else None
        prepare(
            destination, arguments.arm_set, arguments.recipe, arguments.seeds, references_from,
            bks_from=bks_from, probes=arguments.probes, arms=arguments.arms, schedule=arguments.schedule,
            problems_dir=arguments.problems_dir.resolve(), train_seeds=arguments.train_seeds,
            exclude=list(arguments.exclude), num_envs=arguments.num_envs,
        )
    elif arguments.command == "references":
        if arguments.tasks is not None:
            reference_pack(destination, arguments.tasks, arguments.workers)
        else:
            run_references(destination, arguments.task, arguments.workers)
    elif arguments.command == "merge-references":
        merge_references(destination)
    elif arguments.command == "train":
        train(destination, arguments.run, arguments.budget, arguments.device)
    elif arguments.command == "train-pack":
        train_pack(destination, arguments.runs, arguments.budget, arguments.runs_per_gpu, arguments.device)
    elif arguments.command == "demonstrations":
        train(destination, arguments.run, 0, arguments.device, build_store_only=True)
    elif arguments.command == "submit":
        submit(
            destination, arguments.stage, arguments.budget, arguments.dry_run,
            arms=arguments.arms, split=arguments.split, tag=arguments.tag, runs=arguments.runs,
            pack=arguments.pack, runs_per_gpu=arguments.runs_per_gpu, cpus=arguments.cpus,
            partition=arguments.partition, cpus_per_gpu=arguments.cpus_per_gpu, test_only=arguments.test_only,
            time_limit=arguments.hours, dependency=arguments.dependency, mem_per_cpu=arguments.mem_per_cpu,
            decision=decision_from(
                arguments.safety_mask, arguments.safety_margin, arguments.search, arguments.search_k, arguments.search_n,
                arguments.retire_guard, arguments.checkpoint, arguments.eval_batch, arguments.public_wake,
            ),
        )
    elif arguments.command == "evaluate":
        evaluate(
            destination, arguments.split, arguments.tag, arguments.count, arguments.workers, arguments.device,
            arguments.runs, arguments.envs_per_problem, plan_mode=arguments.plan_mode,
            decision=decision_from(
                arguments.safety_mask, arguments.safety_margin, arguments.search, arguments.search_k, arguments.search_n,
                arguments.retire_guard, arguments.checkpoint, arguments.eval_batch, arguments.public_wake,
            ),
        )
    else:
        report(destination, arguments.split, arguments.tag, promote_count=arguments.promote)


if __name__ == "__main__":
    main()
