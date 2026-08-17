"""Score every method on one split through the immutable campaign runner.

All learned and non-learned policies pass through
:func:`run_evaluation_campaign`, so each method publishes a manifest, raw
failure-retaining episode rows, an aggregate summary, and inference timing under
the same contract.  Nothing here inspects results to choose a configuration:
policies arrive already frozen from validation.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from algo.canonical_policy import CanonicalActorCritic
from EVRoutingEnv.baselines.alns import ALNSParameters, ALNSPolicy
from EVRoutingEnv.baselines.canonical_baselines import (
    GreedyHeuristicPolicy,
    HeuristicParameters,
    MPCParameters,
    RandomFeasiblePolicy,
    RollingHorizonMPCPolicy,
)
from EVRoutingEnv.baselines.exact_optimization import (
    ExactPlannerParameters,
    MathematicalProgrammingPolicy,
)
from EVRoutingEnv.evaluation.artifacts import collect_run_manifest
from EVRoutingEnv.evaluation.runner import run_evaluation_campaign
from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.state.action_mask import policy_action_mask
from EVRoutingEnv.utils.utils import load_config
from scripts.evaluation.canonical_harness import split_seeds


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class CanonicalPolicyRunner:
    """Wrap a trained canonical policy in the runner's callback contract."""

    def __init__(
        self, checkpoint: Path, prefix: str = "best", deterministic: bool = True
    ):
        self.policy = CanonicalActorCritic.load(checkpoint, prefix=prefix)
        self.policy.eval()
        self.deterministic = deterministic

    def __call__(self, env, observation, info) -> int:
        batch = torch.as_tensor(np.expand_dims(observation, 0), dtype=torch.float32)
        # Whichever mask this environment was configured to hand a policy: an
        # arm trained without the feasibility mask must also be scored without
        # it, or the evaluation would quietly repair what the ablation removed.
        mask = torch.as_tensor(
            np.expand_dims(policy_action_mask(env), 0), dtype=torch.bool
        )
        actions, _, _ = self.policy.act(batch, mask, deterministic=self.deterministic)
        return int(actions[0].item())


def build_policy(name: str, settings: dict):
    """Instantiate one evaluated method from its frozen settings.

    A learned policy is recognised by carrying a ``checkpoint``, not by being
    called something in particular, so campaigns are free to name their runs
    after the method under test (``graphppo``, ``rl_ablation``, ...) without
    this function needing to know about them.
    """
    if "checkpoint" in settings:
        return CanonicalPolicyRunner(
            Path(settings["checkpoint"]),
            prefix=settings.get("prefix", "best"),
            deterministic=bool(settings.get("deterministic", True)),
        )
    if name == "random":
        return RandomFeasiblePolicy(seed=int(settings.get("seed", 0)))
    if name == "heuristic":
        return GreedyHeuristicPolicy(
            HeuristicParameters.from_settings(settings.get("parameters", {}))
        )
    if name == "mpc":
        return RollingHorizonMPCPolicy(MPCParameters(**settings.get("parameters", {})))
    if name == "cpsat":
        return MathematicalProgrammingPolicy(
            ExactPlannerParameters(**settings.get("parameters", {}))
        )
    if name == "alns":
        return ALNSPolicy(ALNSParameters(**settings.get("parameters", {})))
    raise ValueError(
        f"unknown method {name!r}; non-learned methods must be one of "
        "random/heuristic/mpc/cpsat/alns, and a learned method must supply a checkpoint"
    )


def score_method(job: tuple) -> tuple[str, dict | None]:
    """Score one method over every scenario, in its own process.

    Module level rather than a closure so it can be sent to a process pool.
    """
    (
        name,
        settings,
        config,
        seeds,
        split,
        destination,
        max_policy_steps,
        argv,
    ) = job
    # One thread per worker. Torch defaults to one thread per core, so N method
    # processes each spawning 64 threads drives the load average past 90 on a
    # 64-core host and starves everything else -- including any training sharing
    # the machine. Inference here is one small forward pass at a time, so extra
    # threads buy nothing anyway.
    torch.set_num_threads(1)

    target = Path(destination) / name
    if target.exists():
        print(f"skipping {name}: {target} already exists", flush=True)
        return name, None

    # A method may declare environment overrides -- the mask ablation is scored
    # without the feasibility mask, a generalization regime shifts the instance
    # distribution.  The override lands in the manifest as the resolved config,
    # so the artifact records the environment the method was actually scored in
    # rather than the campaign default.
    method_config = config
    overrides = settings.get("environment_overrides") or {}
    if overrides:
        method_config = deepcopy(config)
        for section, values in overrides.items():
            if section not in method_config or not isinstance(values, dict):
                raise ValueError(
                    f"{name}: environment_overrides.{section} must be a dict "
                    "naming an existing config section"
                )
            method_config[section].update(values)
        print(f"  {name}: config overrides {overrides}", flush=True)

    manifest = collect_run_manifest(
        run_id=f"{name}__{split}",
        algorithm=name,
        split=split,
        command=argv,
        resolved_config=method_config,
        scenario_seeds=seeds,
        repository_root=REPOSITORY_ROOT,
        checkpoint=settings.get("checkpoint"),
    )
    print(f"running {name} on {len(seeds)} {split} scenarios...", flush=True)

    def factory():
        return EventDrivenTruckEnv(method_config, verbose=False, enable_plotting=False)

    artifacts = run_evaluation_campaign(
        environment_factory=factory,
        policy=build_policy(name, settings),
        manifest=manifest,
        output_directory=target,
        max_policy_steps=max_policy_steps,
    )
    summary = json.loads(artifacts.summary_path.read_text())
    aggregate = summary["aggregate"]
    interval = aggregate["success_wilson_95"]
    makespan = aggregate["metrics"]["fleet_makespan"]["mean"]
    print(
        f"  {name}: success={aggregate['success_rate']:.3f} "
        f"[{interval['low']:.3f}, {interval['high']:.3f}] "
        f"makespan={makespan if makespan is None else round(makespan, 2)} "
        f"episodes={artifacts.episode_count}",
        flush=True,
    )
    return name, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="EVRoutingEnv/config_files/config_joint.yaml"
    )
    parser.add_argument(
        "--split", default="validation", choices=["train", "validation", "test"]
    )
    parser.add_argument("--scenarios", type=int, default=50)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--output", default="results/canonical/campaign")
    parser.add_argument(
        "--methods",
        default=None,
        help="Path to a JSON file mapping method name -> frozen settings.",
    )
    parser.add_argument("--max-policy-steps", type=int, default=1000)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Methods scored in parallel processes; each method stays sequential.",
    )
    arguments = parser.parse_args()

    methods = (
        json.loads(Path(arguments.methods).read_text())
        if arguments.methods
        else {"random": {}, "heuristic": {}, "mpc": {}, "cpsat": {}}
    )
    seeds = split_seeds(arguments.split, arguments.scenarios, arguments.base_seed)
    config = load_config(arguments.config)
    destination = Path(arguments.output) / arguments.split
    destination.mkdir(parents=True, exist_ok=True)

    jobs = [
        (
            name,
            settings,
            config,
            seeds,
            arguments.split,
            str(destination),
            arguments.max_policy_steps,
            tuple(sys.argv),
        )
        for name, settings in methods.items()
    ]

    published: dict[str, dict] = {}
    if arguments.workers > 1 and len(jobs) > 1:
        # Methods are independent runs over the same seeds, and the slowest one
        # (the CP-SAT planner) dominates a sequential campaign, so they are
        # scored in parallel processes. Each method is still internally
        # sequential, so per-episode inference timings stay comparable.
        with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
            for name, summary in pool.map(score_method, jobs):
                if summary is not None:
                    published[name] = summary
    else:
        for job in jobs:
            name, summary = score_method(job)
            if summary is not None:
                published[name] = summary

    # A resumed campaign must still publish an index covering every method,
    # including the ones this invocation skipped because they already existed.
    for name in methods:
        summary_path = destination / name / "summary.json"
        if name not in published and summary_path.exists():
            published[name] = json.loads(summary_path.read_text())

    if published:
        (destination / "campaign_index.json").write_text(
            json.dumps(
                {
                    "split": arguments.split,
                    "scenario_seeds": seeds,
                    "methods": published,
                },
                indent=2,
                sort_keys=True,
            )
        )
    print(f"campaign artifacts under {destination}", flush=True)


if __name__ == "__main__":
    main()
