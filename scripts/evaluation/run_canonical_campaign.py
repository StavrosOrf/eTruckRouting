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
from pathlib import Path

import numpy as np
import torch

from algo.canonical_policy import CanonicalActorCritic
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
        mask = torch.as_tensor(np.expand_dims(env.mask_fn(), 0), dtype=torch.bool)
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
    raise ValueError(
        f"unknown method {name!r}; non-learned methods must be one of "
        "random/heuristic/mpc/cpsat, and a learned method must supply a checkpoint"
    )


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

    def factory():
        return EventDrivenTruckEnv(config, verbose=False, enable_plotting=False)

    published = {}
    for name, settings in methods.items():
        run_id = f"{name}__{arguments.split}"
        target = destination / name
        if target.exists():
            print(f"skipping {name}: {target} already exists", flush=True)
            continue
        manifest = collect_run_manifest(
            run_id=run_id,
            algorithm=name,
            split=arguments.split,
            command=tuple(sys.argv),
            resolved_config=config,
            scenario_seeds=seeds,
            repository_root=REPOSITORY_ROOT,
            checkpoint=settings.get("checkpoint"),
        )
        print(
            f"running {name} on {len(seeds)} {arguments.split} scenarios...", flush=True
        )
        artifacts = run_evaluation_campaign(
            environment_factory=factory,
            policy=build_policy(name, settings),
            manifest=manifest,
            output_directory=target,
            max_policy_steps=arguments.max_policy_steps,
        )
        summary = json.loads(artifacts.summary_path.read_text())
        published[name] = summary
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
