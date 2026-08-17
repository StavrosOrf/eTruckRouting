"""Assemble the method set for the revision's final test campaign.

Everything here arrives already frozen: the classical baselines from their
validation grid searches, the learned runs from checkpoints selected on
validation scenarios only. This script does not choose anything -- it collects
what exists, records where each entry came from, and refuses to invent a method
whose checkpoint is missing.

Learned arms that changed the environment they trained in carry that environment
with them, so an unmasked or feature-ablated arm is scored under the conditions
it was trained under rather than under the campaign default.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.evaluation.select_architecture import environment_overrides_for


# name -> (checkpoint directory, human-readable role)
LEARNED_RUNS: dict[str, tuple[str, str]] = {
    "graphppo": (
        "results/canonical/graphppo_v2_stagec/c_tm15",
        "proposed model, frozen selection from document 10",
    ),
    "graphppo_seed1": (
        "results/canonical/graphppo_seeds/seed1_C",
        "seed replication of the headline ladder",
    ),
    "graphppo_seed2": (
        "results/canonical/graphppo_seeds/seed2_C",
        "seed replication of the headline ladder",
    ),
    "graphppo_matched": (
        "results/canonical/graphppo_v2/v2_tm10",
        "masked control at the mask ablation's budget (stage A only, 2M steps)",
    ),
    "mask_none": (
        "results/canonical/mask_ablation/mask_none_terminate",
        "no feasibility mask; infeasible actions strand the truck",
    ),
    "mask_none_seed1": (
        "results/canonical/mask_ablation/mask_seed1",
        "no feasibility mask, seed 1",
    ),
    "mask_none_seed2": (
        "results/canonical/mask_ablation/mask_seed2",
        "no feasibility mask, seed 2",
    ),
    "ppo_flat": (
        "results/canonical/learned_baselines/flat__independent",
        "flat state, independent action scoring (MaskPPO-equivalent)",
    ),
    "ppo_deepsets": (
        "results/canonical/learned_baselines/deep_sets__independent",
        "DeepSets-PPO",
    ),
    "ppo_stategnn": (
        "results/canonical/learned_baselines/hetero__independent",
        "state-GNN PPO, independent action scoring",
    ),
    "ppo_attention": (
        "results/canonical/learned_baselines/attention",
        "constructive attention baseline",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frozen", default="results/canonical/frozen_baselines_revision.json"
    )
    parser.add_argument(
        "--output", default="results/canonical/revision_methods.json"
    )
    parser.add_argument(
        "--include",
        nargs="+",
        default=None,
        help="Subset of learned runs to include; defaults to every one that exists.",
    )
    arguments = parser.parse_args()

    methods = json.loads(Path(arguments.frozen).read_text())
    wanted = arguments.include or list(LEARNED_RUNS)

    missing = []
    for name in wanted:
        if name not in LEARNED_RUNS:
            raise SystemExit(f"unknown learned run {name!r}")
        directory, role = LEARNED_RUNS[name]
        checkpoint = Path(directory)
        # run_config.json is written when training finishes, best.pt as soon as
        # the first validation round beats the initial bar. Requiring the former
        # keeps a still-training run out of a final campaign.
        if not (checkpoint / "run_config.json").exists():
            missing.append(name)
            continue
        if not (checkpoint / "best.pt").exists():
            raise SystemExit(
                f"{name}: {checkpoint} finished training without a best "
                "checkpoint; refusing to assemble a method set around it"
            )
        entry = {
            "checkpoint": str(checkpoint),
            "prefix": "best",
            "deterministic": True,
            "role": role,
        }
        overrides = environment_overrides_for(checkpoint)
        if overrides:
            entry["environment_overrides"] = overrides
        methods[name] = entry

    Path(arguments.output).write_text(json.dumps(methods, indent=2, sort_keys=True))
    print(f"wrote {arguments.output} with {len(methods)} methods:")
    for name in sorted(methods):
        role = methods[name].get("role", "classical baseline")
        overrides = methods[name].get("environment_overrides")
        suffix = f"   [env: {overrides['environment']}]" if overrides else ""
        print(f"  {name:<18}{role}{suffix}")
    if missing:
        print("\nnot yet trained, so excluded:")
        for name in missing:
            print(f"  {name:<18}{LEARNED_RUNS[name][0]}")


if __name__ == "__main__":
    main()
