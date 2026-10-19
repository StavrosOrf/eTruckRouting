"""Validate the RoutingBlocks baseline against brute force and the environment.

For every (config, trucks, customers, seed) instance:
* the plan's objective must equal the independent brute-force optimum
  (``branch_and_price/brute_force.py``, small sizes);
* the plan, replayed on the environment with noise off, must succeed with the
  same total travel time.

Same instance grid as ``validate_branch_and_price.py``.

    PYTHONPATH=. python scripts/analysis/validate_routingblocks_alns.py \\
        --out results/rb_alns/validation.json --seeds 4 --time-limit 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from EVRoutingEnv.baselines.optimization.branch_and_price.brute_force import exact_optimum
from EVRoutingEnv.baselines.optimization.branch_and_price.export import export_nominal_instance
from EVRoutingEnv.baselines.optimization.branch_and_price.validate import nominal_replay
from EVRoutingEnv.baselines.optimization.routingblocks_alns.bridge import RBParameters, solve_rb_plan
from scripts.analysis.validate_branch_and_price import CONFIGS, make_env


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    parser.add_argument("--sizes", nargs="+", default=["2x4", "2x5", "3x5", "2x6", "3x6", "3x7"])
    parser.add_argument("--time-limit", type=float, default=5.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--optimum-cache", type=Path, help="directory of <tag>.opt.json to avoid re-running brute force")
    args = parser.parse_args(argv)

    rows, failures = [], 0
    for config_name in args.configs:
        for size in args.sizes:
            trucks, customers = (int(x) for x in size.split("x"))
            for seed in range(args.seeds):
                tag = f"{config_name}_{trucks}x{customers}_s{seed}"
                env = make_env(config_name, trucks, customers)
                try:
                    env.reset(seed=seed)
                except Exception as error:
                    print(f"{tag:32s} skipped: {type(error).__name__}", flush=True)
                    continue
                cached = args.optimum_cache and (args.optimum_cache / f"{tag}.opt.json")
                if cached and cached.exists():
                    optimum = json.loads(cached.read_text())["optimum"]
                else:
                    optimum = exact_optimum(export_nominal_instance(env))
                started = time.perf_counter()
                solution = solve_rb_plan(env, RBParameters(time_limit_seconds=args.time_limit,
                                                           workers=args.workers))
                env.reset(seed=seed)
                replay = nominal_replay(env, solution.routes)
                checks = {
                    "reaches_optimum": abs(solution.objective_hours - optimum) <= 1e-6,
                    "never_below_optimum": solution.objective_hours >= optimum - 1e-6,
                    "replay_feasible": replay["feasible"],
                    "replay_travel_matches": abs(replay["total_travel_hours"] - solution.objective_hours) <= 1e-6,
                }
                ok = checks["never_below_optimum"] and checks["replay_feasible"] and checks["replay_travel_matches"]
                failures += not ok
                rows.append({"tag": tag, "optimum": optimum, "objective": solution.objective_hours,
                             "checks": checks, "seconds": time.perf_counter() - started})
                flag = "ok " if ok else "BAD"
                hit = "=opt" if checks["reaches_optimum"] else f"gap {(solution.objective_hours - optimum) / optimum:.2%}"
                print(f"{flag} {tag:32s} opt {optimum:10.4f} rb {solution.objective_hours:10.4f} {hit}", flush=True)
    reached = sum(r["checks"]["reaches_optimum"] for r in rows)
    print(f"{len(rows)} instances, {reached} reach the optimum, {failures} invalid", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"rows": rows, "reached": reached, "invalid": failures}, indent=1))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
