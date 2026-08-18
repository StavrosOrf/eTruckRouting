"""Re-score every ablation, baseline, and seed replicate on held-out validation.

The component ablations and the architecture family are validation-split
evidence: nothing here is selected on, and the test split stays reserved for the
final campaign. Each run is re-scored on validation scenarios *beyond* the ones
used to choose its checkpoint, so the comparison is independent of the selection
that produced it.

Each run is scored in the environment it trained in -- an unmasked arm without
the feasibility mask, a feature-ablated arm with the same blocks blanked --
recovered from its own run_config.json rather than assumed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.evaluation.select_architecture import rescore_on_validation


# group -> (training root, runs, what the group answers)
GROUPS: dict[str, tuple[str, list[str], str]] = {
    "mask": (
        "results/canonical/mask_ablation",
        ["mask_none_terminate", "mask_none_penalize", "mask_seed1", "mask_seed2",
         "penalize_p1000"],
        "does the hard feasibility mask explain the result (R1.2, E3)",
    ),
    "architecture": (
        "results/canonical/learned_baselines",
        ["flat__independent", "deep_sets__independent", "hetero__independent",
         "attention"],
        "encoder and action-head family under equal information (R1.6)",
    ),
    "components": (
        "results/canonical/ablations",
        ["ablate_pooling", "ablate_edges", "ablate_queue", "ablate_active_truck"],
        "what each canonical component contributes (E3)",
    ),
    "seeds": (
        "results/canonical/graphppo_seeds",
        ["seed1_A", "seed2_A", "baseA_seed1", "baseA_seed2", "seed1_B", "seed2_B",
         "seed1_C", "seed2_C"],
        "training-seed variance of the ladder (R1.7)",
    ),
    "preassigned": (
        "results/canonical/preassigned",
        ["pre_masked_s0", "pre_masked_s1", "pre_unmasked_s0", "pre_unmasked_s1"],
        "do the mask and seed findings hold in the eTFRP-style setting (R1.1, R1.2)",
    ),
    "charging_actions": (
        "results/canonical/charging_actions",
        ["soc5", "duration"],
        "charging action-space granularity (R2.6)",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="EVRoutingEnv/config_files/config_joint.yaml"
    )
    parser.add_argument("--scenarios", type=int, default=150)
    parser.add_argument("--offset", type=int, default=40)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--groups", nargs="+", default=None, help="Defaults to every group."
    )
    parser.add_argument(
        "--output", default="results/canonical/ablation_summary.json"
    )
    arguments = parser.parse_args()

    groups = arguments.groups or list(GROUPS)
    report: dict[str, dict] = {}
    for name in groups:
        root, runs, question = GROUPS[name]
        available = [
            run for run in runs if (Path(root) / run / "run_config.json").exists()
        ]
        if not available:
            print(f"== {name}: nothing finished yet, skipping", flush=True)
            continue
        print(f"\n== {name}: {question}", flush=True)
        # The charging arms changed the action space, so they carry their own
        # config rather than the campaign default.
        config = arguments.config
        if name == "charging_actions":
            config = None
        for run in available:
            run_config = json.loads(
                (Path(root) / run / "run_config.json").read_text()
            )["arguments"]
            per_run_config = config or run_config["config"]
            scored = rescore_on_validation(
                [run],
                Path(root),
                per_run_config,
                scenarios=arguments.scenarios,
                offset=arguments.offset,
                workers=arguments.workers,
                objective="travel_time",
            )
            report.setdefault(name, {"question": question, "runs": {}})
            report[name]["runs"].update(scored)

    Path(arguments.output).write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"\nwrote {arguments.output}", flush=True)


if __name__ == "__main__":
    main()
