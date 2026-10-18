"""Failure-retaining paired quality against independent feasible references."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def physical_scenario(row):
    scenario = dict(row["scenario"])
    # Policy restrictions intentionally alter config hashes. The actual map,
    # customers, batteries, station properties, RNG and budgets must agree.
    scenario.pop("config_sha256", None)
    return scenario


def wilson(successes, count):
    z = 1.959963984540054
    p = successes / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    radius = z * math.sqrt((p * (1 - p) + z * z / (4 * count)) / count) / denominator
    return [max(0, center - radius), min(1, center + radius)]


def summarize(methods, references):
    missing = set(references) - set(methods)
    if missing:
        raise ValueError(f"missing reference methods: {missing}")
    expected = set(methods[references[0]])
    anchors = methods[references[0]]
    for name, rows in methods.items():
        if set(rows) != expected:
            raise ValueError(f"scenario coverage differs: {name}")
        for seed, row in rows.items():
            if physical_scenario(row) != physical_scenario(anchors[seed]):
                raise ValueError(f"physical scenario differs: {name}, seed {seed}")
    bks = {}
    for seed in expected:
        costs = [methods[name][seed]["total_operating_time"] for name in references
                 if methods[name][seed]["success"]]
        if costs:
            bks[seed] = min(costs)
    summaries = {}
    for name, rows in methods.items():
        successful = [row for row in rows.values() if row["success"]]
        paired = [100 * (row["total_operating_time"] / bks[seed] - 1)
                  for seed, row in sorted(rows.items()) if row["success"] and seed in bks and bks[seed] > 0]
        near = sum(row["success"] and seed in bks
                   and row["total_operating_time"] <= 1.05 * bks[seed]
                   for seed, row in rows.items())
        failures = {}
        for row in rows.values():
            if not row["success"]:
                reason = row.get("termination_reason") or "unreported"
                failures[reason] = failures.get(reason, 0) + 1
        interval = None
        if paired:
            values = np.asarray(paired)
            rng = np.random.default_rng(0)
            means = [float(rng.choice(values, len(values), replace=True).mean()) for _ in range(2000)]
            interval = np.quantile(means, [0.025, 0.975]).tolist()
        summaries[name] = {
            "episodes": len(rows), "successes": len(successful),
            "success_rate": len(successful) / len(rows),
            "success_wilson95": wilson(len(successful), len(rows)),
            "mean_operating_hours_successful": (
                sum(row["total_operating_time"] for row in successful) / len(successful)
                if successful else None
            ),
            "paired_reference_coverage": len(paired),
            "mean_paired_excess_percent": float(np.mean(paired)) if paired else None,
            "mean_paired_excess_bootstrap95": interval,
            "median_paired_excess_percent": float(np.median(paired)) if paired else None,
            "p95_paired_excess_percent": float(np.quantile(paired, 0.95)) if paired else None,
            "success_and_within5pct_fraction_all_scenarios": near / len(rows),
            "failure_reasons": failures,
        }
    return {"objective": "total_operating_time", "reference_type": "independent_best_known",
            "proven_optimality": False, "reference_methods": references,
            "reference_scenario_coverage": len(bks), "methods": summaries}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--references", nargs="+", default=["heuristic", "mpc", "alns", "cpsat"])
    args = parser.parse_args()
    methods = {}
    for path in sorted(args.campaign.glob("*/episode_rows.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if rows:
            methods[path.parent.name] = {row["scenario_seed"]: row for row in rows}
    report = summarize(methods, args.references)
    (args.campaign / "independent_reference_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
