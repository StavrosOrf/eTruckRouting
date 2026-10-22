"""Near-optimal gate of the generalist GraphPPO campaign (SWARM.md).

For every evaluated run (one set of weights) and every suite problem, compare
the run's validation rows with the problem's best-known solutions (BKS) on
the same seeds:

Static problems (success is the first criterion):
  1. failure rate <= max(1 %, BKS failure rate);
  2. success rate >= BKS success rate - 2 points;
  3. mean paired operating-hours excess vs BKS on jointly successful seeds
     <= 5 % (95 % paired bootstrap interval reported).

Dynamic problems (on-time services are the first criterion):
  1. failure rate <= max(1 %, BKS failure rate);
  2. mean on-time services >= 95 % of the BKS mean, paired per seed;
  3. operating hours reported (tie-break only), hindsight bound as headroom.

A problem without a BKS file is MISSING, one without rows for the run is
NO_ROWS; both fail the run overall. The goal holds when one run passes all
three criteria on every problem.

Usage::

    python scripts/campaigns/generalist_near_optimal_gate.py \
        --campaign results/generalist_graphppo/suite_v1 --split validation --tag T [--runs a b]

Rows are read from ``DIR/evaluation/<split>/<tag>/<run>.json`` (lists of
``episode_summary`` rows with ``problem`` and ``seed``); BKS files from
``--bks-dir/<problem>/<split>.json``. The table is printed and written to
``DIR/evaluation/<split>/<tag>/near_optimal_gate.{json,md}``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROBLEMS = ("ca_sameday", "ca_drayage_combined", "tx_longhaul", "tx_longhaul_congested", "pnw_longhaul_hos")
STATIC = frozenset({"tx_longhaul", "tx_longhaul_congested", "pnw_longhaul_hos"})
DYNAMIC = frozenset({"ca_sameday", "ca_drayage_combined"})

FAILURE_FLOOR = 0.01
SUCCESS_SLACK = 0.02
HOURS_EXCESS_LIMIT = 0.05
SERVICE_RATIO_LIMIT = 0.95


def load_bks(directory: Path, problem: str, split: str) -> dict[int, dict] | None:
    path = directory / problem / f"{split}.json"
    if not path.exists():
        return None
    return {int(seed): value for seed, value in json.loads(path.read_text()).items()}


def load_rows(path: Path) -> list[dict]:
    rows = json.loads(path.read_text())
    if isinstance(rows, dict):
        rows = rows.get("rows", [])
    return rows


def _bootstrap(baseline, candidate, *, resamples: int, seed: int = 0) -> dict:
    from EVRoutingEnv.evaluation.statistics import paired_bootstrap_difference

    return paired_bootstrap_difference(baseline, candidate, resamples=resamples, seed=seed)


def _rate(values) -> float:
    values = list(values)
    return float(sum(bool(v) for v in values) / len(values)) if values else float("nan")


def gate_problem(problem: str, rows: list[dict], bks: dict[int, dict] | None, *, resamples: int = 10_000) -> dict:
    """The three criteria of one run on one problem."""
    if bks is None:
        return {"problem": problem, "status": "MISSING", "reason": "no BKS file"}
    by_seed = {}
    for row in rows:
        if row.get("problem") == problem:
            by_seed[int(row["seed"])] = row
    seeds = sorted(seed for seed in by_seed if seed in bks)
    if not seeds:
        return {"problem": problem, "status": "NO_ROWS", "reason": "no rows on BKS seeds",
                "bks_seeds": len(bks), "run_seeds": len(by_seed)}
    run = [by_seed[seed] for seed in seeds]
    best = [bks[seed]["best"] for seed in seeds]
    dynamic = problem in DYNAMIC if problem in PROBLEMS else bool(run[0].get("dynamic", False))
    run_failure = _rate(row["failed"] for row in run)
    bks_failure = _rate(row["failed"] for row in best)
    failure_limit = max(FAILURE_FLOOR, bks_failure)
    criteria = {}
    criteria["failure"] = {
        "run": run_failure, "bks": bks_failure, "limit": failure_limit,
        "pass": run_failure <= failure_limit + 1e-12,
    }
    result = {
        "problem": problem, "dynamic": dynamic, "seeds": len(seeds),
        "bks_seeds": len(bks), "coverage": len(seeds) / max(1, len(bks)),
    }
    if not dynamic:
        run_success = _rate(row["success"] for row in run)
        bks_success = _rate(row["success"] for row in best)
        criteria["success"] = {
            "run": run_success, "bks": bks_success, "limit": bks_success - SUCCESS_SLACK,
            "pass": run_success >= bks_success - SUCCESS_SLACK - 1e-12,
        }
        joint = [(a, b) for a, b in zip(run, best, strict=True) if a["success"] and b["success"]]
        hours = {"pairs": len(joint), "limit": HOURS_EXCESS_LIMIT}
        if joint:
            excess = [a["hours"] / b["hours"] - 1.0 for a, b in joint]
            hours["mean_excess"] = float(sum(excess) / len(excess))
            hours["run_hours"] = float(sum(a["hours"] for a, _ in joint) / len(joint))
            hours["bks_hours"] = float(sum(b["hours"] for _, b in joint) / len(joint))
            if len(joint) >= 2:
                boot = _bootstrap([0.0] * len(excess), excess, resamples=resamples)
                hours["ci95"] = [boot["ci_low"], boot["ci_high"]]
            else:
                hours["ci95"] = [hours["mean_excess"], hours["mean_excess"]]
            hours["pass"] = hours["mean_excess"] <= HOURS_EXCESS_LIMIT + 1e-12
            hours["strict_pass"] = hours["ci95"][1] <= HOURS_EXCESS_LIMIT + 1e-12
        else:
            hours["pass"] = False
            hours["strict_pass"] = False
            hours["reason"] = "no jointly successful seeds"
        criteria["hours"] = hours
    else:
        served_run = [float(row["served"]) for row in run]
        served_bks = [float(row["served"]) for row in best]
        total_bks = sum(served_bks)
        ratio = sum(served_run) / total_bks if total_bks > 0 else (1.0 if sum(served_run) >= 0 else 0.0)
        service = {"run": sum(served_run) / len(run), "bks": total_bks / len(run), "ratio": ratio,
                   "limit": SERVICE_RATIO_LIMIT, "pass": ratio >= SERVICE_RATIO_LIMIT - 1e-12}
        if len(run) >= 2:
            boot = _bootstrap(served_bks, served_run, resamples=resamples)
            service["difference"] = boot["mean_difference"]
            service["difference_ci95"] = [boot["ci_low"], boot["ci_high"]]
        criteria["services"] = service
        hours = {
            "run": sum(float(row["hours"]) for row in run) / len(run),
            "bks": sum(float(row["hours"]) for row in best) / len(best),
            "pass": True, "tie_break_only": True,
        }
        bounds = [bks[seed].get("bound", {}) for seed in seeds]
        values = [b.get("value") for b in bounds if b and b.get("kind", "none") != "none" and b.get("value") is not None]
        if values:
            hours["hindsight_bound_served"] = sum(float(v) for v in values) / len(values)
        criteria["hours"] = hours
    result["criteria"] = criteria
    result["status"] = "PASS" if all(item["pass"] for item in criteria.values()) else "FAIL"
    return result


def gate_run(rows: list[dict], bks: dict[str, dict | None], problems=PROBLEMS, *, resamples: int = 10_000) -> dict:
    results = {problem: gate_problem(problem, rows, bks.get(problem), resamples=resamples) for problem in problems}
    overall = "PASS" if all(item["status"] == "PASS" for item in results.values()) else "FAIL"
    return {"problems": results, "overall": overall}


def _percent(value) -> str:
    return "n/a" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{100 * value:.1f}%"


def format_table(report: dict) -> str:
    lines = [
        "| Run | Problem | Status | Seeds | Failure run / limit | Success run / limit (static) or services ratio (dynamic) | Hours excess [95% CI] (static) or hours run / BKS (dynamic) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for run, gate in report["runs"].items():
        for problem, item in gate["problems"].items():
            status = item["status"]
            if status in ("MISSING", "NO_ROWS"):
                lines.append(f"| {run} | {problem} | {status} | - | - | - | - |")
                continue
            c = item["criteria"]
            failure = f"{_percent(c['failure']['run'])} / {_percent(c['failure']['limit'])} {'ok' if c['failure']['pass'] else 'X'}"
            if not item["dynamic"]:
                second = f"{_percent(c['success']['run'])} / {_percent(c['success']['limit'])} {'ok' if c['success']['pass'] else 'X'}"
                h = c["hours"]
                if "mean_excess" in h:
                    third = (f"{_percent(h['mean_excess'])} [{_percent(h['ci95'][0])}, {_percent(h['ci95'][1])}] "
                             f"n={h['pairs']} {'ok' if h['pass'] else 'X'}")
                else:
                    third = "no joint successes X"
            else:
                s = c["services"]
                second = f"{s['ratio']:.3f} (run {s['run']:.2f} / BKS {s['bks']:.2f}) {'ok' if s['pass'] else 'X'}"
                h = c["hours"]
                third = f"{h['run']:.1f} / {h['bks']:.1f}"
                if "hindsight_bound_served" in h:
                    third += f"; hindsight served {h['hindsight_bound_served']:.2f}"
            lines.append(f"| {run} | {problem} | {status} | {item['seeds']} | {failure} | {second} | {third} |")
        lines.append(f"| {run} | **overall** | **{gate['overall']}** | | | | |")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--runs", nargs="*", default=None)
    parser.add_argument("--bks-dir", type=Path, default=ROOT / "results/generalist_graphppo/bks")
    parser.add_argument("--problems", nargs="*", default=list(PROBLEMS))
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)
    if args.split == "test":
        raise SystemExit("the test bank is locked; the gate reads validation only unless the coordinator opens P5")
    folder = args.campaign / "evaluation" / args.split / args.tag
    if not folder.exists():
        raise SystemExit(f"no evaluation rows at {folder}")
    paths = sorted(path for path in folder.glob("*.json") if not path.name.startswith("near_optimal_gate"))
    if args.runs:
        wanted = set(args.runs)
        paths = [path for path in paths if path.stem in wanted]
        missing = wanted - {path.stem for path in paths}
        if missing:
            raise SystemExit(f"runs without rows in {folder}: {sorted(missing)}")
    bks = {problem: load_bks(args.bks_dir, problem, args.split) for problem in args.problems}
    report = {"campaign": str(args.campaign), "split": args.split, "tag": args.tag,
              "bks_dir": str(args.bks_dir), "runs": {}}
    for path in paths:
        rows = load_rows(path)
        if not isinstance(rows, list) or (rows and "seed" not in rows[0]):
            continue
        report["runs"][path.stem] = gate_run(rows, bks, args.problems, resamples=args.resamples)
    passing = [run for run, gate in report["runs"].items() if gate["overall"] == "PASS"]
    report["passing_runs"] = passing
    table = format_table(report)
    header = (f"# Near-optimal gate: {args.campaign.name}, {args.split}, {args.tag}\n\n"
              f"BKS: `{args.bks_dir}`. Static: failure <= max(1%, BKS), success >= BKS - 2 pts, "
              "mean paired hours excess on jointly successful seeds <= 5%. Dynamic: failure <= max(1%, BKS), "
              "on-time services >= 95% of BKS (paired). Passing runs: "
              f"{', '.join(passing) if passing else 'none'}.\n\n")
    print(header + table)
    if not args.no_write:
        (folder / "near_optimal_gate.json").write_text(json.dumps(report, indent=2))
        (folder / "near_optimal_gate.md").write_text(header + table + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
