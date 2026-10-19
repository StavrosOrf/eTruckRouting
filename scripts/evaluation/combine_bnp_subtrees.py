"""Combine the certificates of a split branch-and-price search into one.

A split run (``solve_branch_and_price --split-nodes K``) explores the top of
the tree and writes its K open nodes to ``OUT/split_<seed>.json``; one
``bnp_subtree.sbatch`` task searches each of them. The subtrees cover exactly
the unexplored part of the tree, so

- the best incumbent of any run is the upper bound;
- the least lower bound over the subtrees is a lower bound, where a subtree
  without a result yet counts with the bound it was split with.

When every subtree finished and the two meet, the incumbent is optimal. The
combined certificate replaces ``OUT/seed_<seed>.json`` (the split run's own is
kept as ``seed_<seed>.split.json``) after the usual nominal replay.

Example:
    python -m scripts.evaluation.combine_bnp_subtrees --out results/branch_and_price/ca_50x100_split \\
        --seed 1 --trucks 50 --customers 100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scripts.evaluation.solve_branch_and_price import attach_validation, build_env

GAP_TOL = 1e-6


def combine(out: Path, seed: int) -> dict:
    split = json.loads((out / f"split_{seed}.json").read_text())
    kept = out / f"seed_{seed}.split.json"
    if not kept.exists():
        (out / f"seed_{seed}.json").rename(kept)
    head = json.loads(kept.read_text())

    subtrees, best = [], head
    lower = float("inf")
    finished = 0
    for index, node in enumerate(split["nodes"]):
        path = out / f"subtrees_{seed}" / f"sub_{index}.result.json"
        row = {"index": index, "label": node["label"], "split_bound_hours": node["bound"], "status": "PENDING"}
        if path.exists():
            certificate = json.loads(path.read_text())
            finished += 1
            row.update({key: certificate[key] for key in ("status", "objective_hours", "best_bound_hours",
                                                          "nodes_explored", "nodes_open", "wall_seconds")})
            bound = certificate["best_bound_hours"]
            if certificate["status"] == "INFEASIBLE":
                bound = float("inf")
            lower = min(lower, node["bound"] if bound is None else bound)
            value = certificate["objective_hours"]
            if value is not None and (best["objective_hours"] is None or value < best["objective_hours"]):
                best = certificate
        else:
            lower = min(lower, node["bound"])
        subtrees.append(row)

    shared = {}
    for path in sorted((out / f"incumbents_{seed}").glob("incumbent_*.json")):
        shared[path.name] = json.loads(path.read_text()).get("objective_hours")

    upper = best["objective_hours"]
    combined = dict(best)
    optimal = (finished == len(subtrees) and upper is not None
               and upper - lower <= GAP_TOL * max(1.0, abs(upper)))
    if optimal:
        lower = upper
    combined.update({
        "status": "OPTIMAL" if optimal else "GAP",
        "proven_optimal": optimal,
        "objective_hours": upper,
        "best_bound_hours": lower if lower < float("inf") else None,
        "gap": max(0.0, (upper - lower) / upper) if upper and lower < float("inf") else None,
        "nodes_explored": head["nodes_explored"] + sum(r.get("nodes_explored", 0) for r in subtrees),
        "nodes_open": sum(r.get("nodes_open", 0) for r in subtrees) + len(subtrees) - finished,
        "wall_seconds": head["wall_seconds"] + max((r.get("wall_seconds", 0.0) for r in subtrees), default=0.0),
        "parallel": {
            "split_nodes": len(subtrees),
            "finished": finished,
            "split_run": {key: head[key] for key in ("status", "objective_hours", "best_bound_hours",
                                                     "root_lp_hours", "root_bound_hours", "nodes_explored",
                                                     "wall_seconds")},
            "job_seconds": head["wall_seconds"] + sum(r.get("wall_seconds", 0.0) for r in subtrees),
            "subtrees": subtrees,
            "shared_incumbents": shared,
        },
    })
    return combined


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config", default="EVRoutingEnv/config_files/config_joint.yaml")
    parser.add_argument("--trucks", type=int)
    parser.add_argument("--customers", type=int)
    args = parser.parse_args(argv)

    certificate = combine(args.out, args.seed)
    ok, reason = attach_validation(build_env(args), certificate, args.seed, args.config)
    (args.out / f"seed_{args.seed}.json").write_text(json.dumps(certificate, indent=1))
    info = certificate["parallel"]
    print(f"seed {args.seed}: {certificate['status']} objective {certificate['objective_hours']} "
          f"bound {certificate['best_bound_hours']} gap {certificate['gap']} "
          f"({info['finished']}/{info['split_nodes']} subtrees finished, {certificate['nodes_explored']} nodes) "
          f"replay {'OK' if ok else 'FAILED: ' + str(reason)}", flush=True)
    for row in info["subtrees"]:
        print(f"  subtree {row['index']:3d} {row['status']:12s} bound {row.get('best_bound_hours')} "
              f"(split {row['split_bound_hours']:.6f}) nodes {row.get('nodes_explored', '-')}  {row['label']}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
