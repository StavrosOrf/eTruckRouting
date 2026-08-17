"""Best-known-solution reference, used as the optimality denominator.

Proving optimality for this problem is out of reach: the CP-SAT formulation in
``optimality_reference.py`` leaves an ~85% gap between incumbent and bound on
the primary configuration and never closes it (see ``measure_ceiling.py``).
Reporting a "gap to optimal" from that would be dishonest.

Standard practice when optimality cannot be proven is to report the gap to the
**best known solution**: per scenario, the best objective any method achieved.
This module builds that table from completed campaign artifacts and scores each
method against it.

Two properties worth stating whenever these numbers are quoted:

* the ratio is *relative*, so 100% means "matched the best anyone found here",
  not "optimal";
* a method that fails a scenario contributes no objective value at all, so
  ratios must always be read next to success rate, never instead of it.

The objective is selectable (``--objective``) and defaults to fleet travel
hours, the campaign objective. ``fleet_makespan`` remains available for the
earlier makespan-first tables.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_campaign(root: Path) -> dict[str, dict[int, dict]]:
    """Load every completed method under a campaign directory."""
    methods: dict[str, dict[int, dict]] = {}
    for directory in sorted(root.iterdir()):
        rows_path = directory / "episode_rows.jsonl"
        if not rows_path.exists():
            continue
        rows = {}
        for line in rows_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                rows[int(row["scenario_seed"])] = row
        methods[directory.name] = rows
    return methods


OBJECTIVE_FIELDS = {
    "travel_time": "total_travel_time",
    "makespan": "fleet_makespan",
    "distance": "total_distance",
    "operating_time": "total_operating_time",
}


def build_best_known(
    methods: dict[str, dict[int, dict]], objective: str = "travel_time"
) -> dict[int, dict]:
    """Per scenario, the lowest objective value achieved by any successful method."""
    field = OBJECTIVE_FIELDS[objective]
    seeds: set[int] = set()
    for rows in methods.values():
        seeds |= set(rows)

    best: dict[int, dict] = {}
    for seed in sorted(seeds):
        candidates = []
        for name, rows in methods.items():
            row = rows.get(seed)
            if row is None or not row.get("success"):
                continue
            value = row.get(field)
            if value is None:
                continue
            candidates.append((float(value), name))
        if candidates:
            value, owner = min(candidates)
            best[seed] = {
                "objective": objective,
                "best_value": value,
                "owner": owner,
                "solved_by": sorted(name for _, name in candidates),
            }
        else:
            best[seed] = {
                "objective": objective,
                "best_value": None,
                "owner": None,
                "solved_by": [],
            }
    return best


def score_methods(
    methods: dict[str, dict[int, dict]],
    best: dict[int, dict],
    objective: str = "travel_time",
) -> dict[str, dict]:
    field = OBJECTIVE_FIELDS[objective]
    scores: dict[str, dict] = {}
    for name, rows in methods.items():
        ratios = []
        matched = 0
        solved = 0
        for seed, reference in best.items():
            row = rows.get(seed)
            if row is None or not row.get("success"):
                continue
            value = row.get(field)
            if value is None or reference["best_value"] is None:
                continue
            solved += 1
            ratio = reference["best_value"] / float(value)
            ratios.append(ratio)
            if ratio >= 0.999:
                matched += 1
        scores[name] = {
            "episodes": len(rows),
            "successes": sum(1 for row in rows.values() if row.get("success")),
            "success_rate": (
                sum(1 for row in rows.values() if row.get("success")) / max(len(rows), 1)
            ),
            "scored_scenarios": solved,
            "mean_optimality_ratio": float(np.mean(ratios)) if ratios else None,
            "median_optimality_ratio": float(np.median(ratios)) if ratios else None,
            "within_0_1_percent_of_best": matched,
            "within_0_1_percent_fraction": matched / solved if solved else None,
        }
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", default="results/canonical/campaign/test")
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--objective", default="travel_time", choices=sorted(OBJECTIVE_FIELDS)
    )
    arguments = parser.parse_args()

    root = Path(arguments.campaign)
    methods = load_campaign(root)
    if not methods:
        raise SystemExit(f"no completed campaign artifacts under {root}")

    best = build_best_known(methods, arguments.objective)
    scores = score_methods(methods, best, arguments.objective)
    unsolved = sum(1 for entry in best.values() if entry["best_value"] is None)

    owners: dict[str, int] = {}
    for entry in best.values():
        if entry["owner"] is not None:
            owners[entry["owner"]] = owners.get(entry["owner"], 0) + 1

    header = (
        f"{'method':<14}{'success':>9}{'scored':>8}{'mean ratio':>12}"
        f"{'median':>9}{'>=99.9% of best':>17}{'owns best':>11}"
    )
    print(f"Best-known {arguments.objective} reference over {len(best)} scenarios "
          f"({unsolved} solved by no method)")
    print(header)
    print("-" * len(header))
    for name, entry in sorted(
        scores.items(), key=lambda item: -(item[1]["mean_optimality_ratio"] or 0.0)
    ):
        mean = entry["mean_optimality_ratio"]
        median = entry["median_optimality_ratio"]
        fraction = entry["within_0_1_percent_fraction"]
        print(
            f"{name:<14}{entry['success_rate']:>9.3f}{entry['scored_scenarios']:>8}"
            f"{('n/a' if mean is None else f'{mean:.4f}'):>12}"
            f"{('n/a' if median is None else f'{median:.4f}'):>9}"
            f"{('n/a' if fraction is None else f'{fraction:.3f}'):>17}"
            f"{owners.get(name, 0):>11}"
        )
    print()
    print("Ratio is best-known / achieved, so 1.0 means matching the best plan any")
    print("method found here. It is NOT a proven optimality gap, and it is computed")
    print("only over scenarios the method actually solved. The reference is partly")
    print("self-referential: the 'owns best' column is how many scenarios each")
    print("method defines the denominator for.")

    report = {
        "campaign": str(root),
        "objective": arguments.objective,
        "scenarios": len(best),
        "unsolved_by_any_method": unsolved,
        "best_known_owners": owners,
        "methods": scores,
        "best_known": {str(seed): entry for seed, entry in best.items()},
    }
    destination = (
        Path(arguments.output) if arguments.output else root / "best_known.json"
    )
    destination.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
