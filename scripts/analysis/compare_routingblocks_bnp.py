"""Compare RoutingBlocks plans with branch-and-price certificates seed by seed.

    python scripts/analysis/compare_routingblocks_bnp.py results/rb_alns/ca10t3d_30s \\
        results/branch_and_price/ca10t3d_validation_2000 --out results/rb_alns/ca10t3d_30s/comparison.json

Per seed: RoutingBlocks objective, branch-and-price objective and bound, the
RoutingBlocks gap to the best proven bound, and whether the plan replayed in the
environment. Summary: how many seeds match a proven optimum, beat the exact
solver's incumbent, or fall short.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("rb_dir", type=Path)
    parser.add_argument("bnp_dir", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--tolerance", type=float, default=1e-6, help="relative tolerance for 'equal'")
    args = parser.parse_args(argv)

    rows = []
    for path in sorted(args.rb_dir.glob("seed_*.json")):
        reference = args.bnp_dir / path.name
        if not reference.exists():
            continue
        rb, bp = json.loads(path.read_text()), json.loads(reference.read_text())
        replay = rb.get("validation", {}).get("nominal_replay", {})
        rb_value, bp_value, bound = rb["objective_hours"], bp["objective_hours"], bp["best_bound_hours"]
        rows.append({
            "seed": path.stem.removeprefix("seed_"),
            "rb": rb_value,
            "bnp": bp_value,
            "bound": bound,
            "bnp_status": bp["status"],
            "rb_gap_to_bound": None if not bound else (rb_value - bound) / rb_value,
            "rb_vs_bnp": None if bp_value is None else (rb_value - bp_value) / bp_value,
            "replay_ok": bool(replay.get("feasible")),
            "rb_seconds": rb.get("wall_seconds"),
        })
    tol = args.tolerance
    proven = [r for r in rows if r["bnp_status"] == "OPTIMAL"]
    summary = {
        "seeds": len(rows),
        "replay_ok": sum(r["replay_ok"] for r in rows),
        "proven_optimal_seeds": len(proven),
        "rb_matches_proven_optimum": sum(abs(r["rb_vs_bnp"]) <= tol for r in proven),
        "rb_below_proven_optimum": sum(r["rb_vs_bnp"] < -tol for r in proven),  # must be 0 (would break the bound)
        "rb_worse_than_proven_optimum": sum(r["rb_vs_bnp"] > tol for r in proven),
        "rb_beats_unproven_incumbent": sum(r["rb_vs_bnp"] < -tol for r in rows if r["bnp_status"] != "OPTIMAL"),
        "worst_gap_to_proven_optimum": max((r["rb_vs_bnp"] for r in proven), default=None),
        "median_rb_gap_to_bound": statistics.median([r["rb_gap_to_bound"] for r in rows if r["rb_gap_to_bound"] is not None])
        if rows else None,
    }
    for key, value in summary.items():
        print(f"{key:34s} {value}")
    if args.out:
        args.out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
