"""Compare campaign artifacts across methods with paired, failure-retaining statistics.

Reads the immutable ``episode_rows.jsonl`` published by
``run_canonical_campaign.py``, verifies that every method was scored on exactly
the same scenarios, and reports:

* success rate with a Wilson 95% interval, over all episodes including failures;
* completion and operating-time means over all episodes;
* makespan conditioned on successful episodes, with the conditioning stated;
* paired bootstrap differences against a chosen reference method.

Pairing is by scenario seed, so the comparison controls for instance difficulty.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from EVRoutingEnv.evaluation.statistics import (
    paired_bootstrap_difference,
    wilson_interval,
)


def load_rows(directory: Path) -> dict[int, dict]:
    """Load one method's episode rows keyed by scenario seed."""
    path = directory / "episode_rows.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing; the campaign for {directory.name} did not complete"
        )
    rows = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[int(row["scenario_seed"])] = row
    return rows


def _mean_over_successes(
    ordered: list[dict], successes: list[bool], key: str
) -> float | None:
    values = [
        float(row[key])
        for row, success in zip(ordered, successes, strict=True)
        if success and row.get(key) is not None
    ]
    return float(np.mean(values)) if values else None


def summarize_method(rows: dict[int, dict], seeds: list[int]) -> dict:
    ordered = [rows[seed] for seed in seeds]
    successes = [bool(row["success"]) for row in ordered]
    count = sum(successes)
    low, high = wilson_interval(count, len(ordered))
    makespans = [
        float(row["fleet_makespan"])
        for row, success in zip(ordered, successes, strict=True)
        if success and row.get("fleet_makespan") is not None
    ]
    return {
        "episodes": len(ordered),
        "success_count": count,
        "success_rate": count / len(ordered),
        "success_wilson_95": {"low": low, "high": high},
        "mean_completed_fraction": float(
            np.mean([float(row["completed_fraction"]) for row in ordered])
        ),
        "mean_operating_time_all_episodes": float(
            np.mean([float(row["total_operating_time"]) for row in ordered])
        ),
        "mean_makespan_successful_only": float(np.mean(makespans))
        if makespans
        else None,
        "makespan_conditioning": "successful_episodes",
        # The campaign objective. Conditioned on successes for the same reason
        # makespan is: travel time on an abandoned route is not a plan cost.
        "mean_travel_time_successful_only": _mean_over_successes(
            ordered, successes, "total_travel_time"
        ),
        "mean_distance_successful_only": _mean_over_successes(
            ordered, successes, "total_distance"
        ),
        "mean_charging_sessions_successful_only": _mean_over_successes(
            ordered, successes, "charging_sessions"
        ),
        "mean_energy_charged": float(
            np.mean([float(row["total_energy_charged"]) for row in ordered])
        ),
        "mean_seconds_per_decision": float(
            np.mean(
                [
                    float(row["inference_seconds"]) / max(1, int(row["policy_calls"]))
                    for row in ordered
                ]
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", default="results/canonical/campaign/test")
    parser.add_argument("--reference", default="mpc")
    parser.add_argument("--candidate", default="rl")
    parser.add_argument("--output", default=None)
    arguments = parser.parse_args()

    root = Path(arguments.campaign)
    methods = sorted(
        path.name for path in root.iterdir() if (path / "episode_rows.jsonl").exists()
    )
    if not methods:
        raise SystemExit(f"no completed campaign artifacts under {root}")

    loaded = {name: load_rows(root / name) for name in methods}
    shared = set.intersection(*(set(rows) for rows in loaded.values()))
    for name, rows in loaded.items():
        if set(rows) != shared:
            raise SystemExit(
                f"{name} was scored on a different scenario set; refusing to compare"
            )
    seeds = sorted(shared)

    summaries = {name: summarize_method(loaded[name], seeds) for name in methods}
    comparisons = {}
    if arguments.reference in loaded:
        reference_rows = loaded[arguments.reference]
        for name in methods:
            if name == arguments.reference:
                continue
            candidate_rows = loaded[name]
            # Makespan per method is conditioned on that method's own successes,
            # so the two sets differ whenever success rates differ. Restricting
            # to jointly successful scenarios is the only like-for-like speed
            # comparison.
            jointly_successful = [
                seed
                for seed in seeds
                if bool(reference_rows[seed]["success"])
                and bool(candidate_rows[seed]["success"])
                and reference_rows[seed].get("fleet_makespan") is not None
                and candidate_rows[seed].get("fleet_makespan") is not None
            ]
            def paired_on_solved(key: str, _jointly=jointly_successful):
                if len(_jointly) < 2:
                    return None
                return paired_bootstrap_difference(
                    [float(reference_rows[seed][key]) for seed in _jointly],
                    [float(candidate_rows[seed][key]) for seed in _jointly],
                    lower_is_better=True,
                )

            comparisons[name] = {
                "vs": arguments.reference,
                "jointly_successful_scenarios": len(jointly_successful),
                "makespan_on_jointly_successful": paired_on_solved("fleet_makespan"),
                # The headline objective: fleet travel hours, paired by scenario
                # so instance difficulty is controlled for and neither method is
                # credited for the scenarios it declined.
                "travel_time_on_jointly_successful": paired_on_solved(
                    "total_travel_time"
                ),
                "distance_on_jointly_successful": paired_on_solved("total_distance"),
                # Success is coded 0/1 and higher is better.
                "success_rate": paired_bootstrap_difference(
                    [float(reference_rows[seed]["success"]) for seed in seeds],
                    [float(candidate_rows[seed]["success"]) for seed in seeds],
                    lower_is_better=False,
                ),
                "completed_fraction": paired_bootstrap_difference(
                    [
                        float(reference_rows[seed]["completed_fraction"])
                        for seed in seeds
                    ],
                    [
                        float(candidate_rows[seed]["completed_fraction"])
                        for seed in seeds
                    ],
                    lower_is_better=False,
                ),
                "operating_time": paired_bootstrap_difference(
                    [
                        float(reference_rows[seed]["total_operating_time"])
                        for seed in seeds
                    ],
                    [
                        float(candidate_rows[seed]["total_operating_time"])
                        for seed in seeds
                    ],
                    lower_is_better=True,
                ),
            }

    report = {
        "campaign": str(root),
        "scenario_count": len(seeds),
        "methods": summaries,
        "paired_comparisons": comparisons,
    }
    print(_render_table(summaries, arguments.candidate))
    if comparisons:
        print(
            f"\nPaired differences versus {arguments.reference} "
            f"(positive success favours the candidate):"
        )
        for name, entry in sorted(comparisons.items()):
            success = entry["success_rate"]
            operating = entry["operating_time"]
            print(
                f"  {name:24s} success {success['mean_difference']:+.3f} "
                f"[{success['ci_low']:+.3f}, {success['ci_high']:+.3f}]"
                f"   operating hours {operating['mean_difference']:+.1f} "
                f"[{operating['ci_low']:+.1f}, {operating['ci_high']:+.1f}]"
            )
            solved = entry["jointly_successful_scenarios"]
            for label, key in (
                ("travel hours", "travel_time_on_jointly_successful"),
                ("makespan", "makespan_on_jointly_successful"),
            ):
                paired = entry[key]
                if paired is not None:
                    print(
                        f"  {'':24s} {label} on the {solved} jointly solved "
                        f"scenarios {paired['mean_difference']:+.1f} h "
                        f"[{paired['ci_low']:+.1f}, {paired['ci_high']:+.1f}]"
                    )
    destination = (
        Path(arguments.output) if arguments.output else root / "comparison.json"
    )
    destination.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"\nwrote {destination}")


def _render_table(summaries: dict, candidate: str) -> str:
    header = (
        f"{'method':<24}{'success':>9}{'95% CI':>18}{'completed':>11}"
        f"{'travel*':>10}{'makespan*':>11}{'op.time':>10}{'s/decision':>12}"
    )
    lines = [header, "-" * len(header)]
    for name, summary in sorted(
        summaries.items(), key=lambda item: -item[1]["success_rate"]
    ):
        interval = summary["success_wilson_95"]
        marker = " *" if name == candidate else ""
        bounds = f"[{interval['low']:.2f}, {interval['high']:.2f}]"
        lines.append(
            f"{name + marker:<24}"
            f"{summary['success_rate']:>9.3f}"
            f"{bounds:>18}"
            f"{summary['mean_completed_fraction']:>11.3f}"
            f"{_optional(summary['mean_travel_time_successful_only']):>10}"
            f"{_optional(summary['mean_makespan_successful_only']):>11}"
            f"{summary['mean_operating_time_all_episodes']:>10.1f}"
            f"{summary['mean_seconds_per_decision']:>12.5f}"
        )
    lines.append(
        "* travel hours and makespan are conditioned on each method's own "
        "successful episodes; use the paired block below for like-for-like."
    )
    return "\n".join(lines)


def _optional(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}"


if __name__ == "__main__":
    main()
