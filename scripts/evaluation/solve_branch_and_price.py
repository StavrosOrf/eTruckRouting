"""Solve scenarios to proven optimality with the branch-and-price baseline.

For every seed the nominal instance is exported, solved by the Julia solver
(one process for the whole batch), and the plan is replayed on the
environment with noise off. The certificate, with that replay attached, is
written to ``OUT/seed_<seed>.json``, where ``BranchAndPricePolicy``
(``certificate_dir``) and the campaign runner pick it up.

Examples:
    python -m scripts.evaluation.solve_branch_and_price --trucks 50 --customers 100 \\
        --split test --count 20 --time-limit 3600 --threads 8 --out results/branch_and_price/ca_50x100
    python -m scripts.evaluation.solve_branch_and_price --seeds 7 11 --out /tmp/bp
    # one Slurm array task per seed:
    python -m scripts.evaluation.solve_branch_and_price --split test --count 20 \\
        --index $SLURM_ARRAY_TASK_ID --out results/branch_and_price/ca_50x100
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

from EVRoutingEnv.baselines.optimization.branch_and_price.export import write_instance
from EVRoutingEnv.baselines.optimization.branch_and_price.julia_bridge import RUNNER, BPParameters
from EVRoutingEnv.baselines.optimization.branch_and_price.validate import nominal_replay, port_conflicts
from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.utils.utils import load_config
from scripts.evaluation.canonical_harness import split_seeds


def build_env(args) -> EventDrivenTruckEnv:
    config = deepcopy(load_config(args.config))
    if args.trucks is not None:
        config["environment"]["num_trucks"] = args.trucks
    if args.customers is not None:
        config["environment"].update({"num_stops": args.customers, "allow_variable_num_stops": False})
    return EventDrivenTruckEnv(config, verbose=False, enable_plotting=False)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default="EVRoutingEnv/config_files/config_joint.yaml")
    parser.add_argument("--trucks", type=int)
    parser.add_argument("--customers", type=int)
    parser.add_argument("--seeds", type=int, nargs="+", help="explicit scenario seeds")
    parser.add_argument("--split", choices=("train", "val", "test"), help="seed namespace (with --count)")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--index", type=int, help="solve only this position of the seed list (Slurm arrays)")
    parser.add_argument("--time-limit", type=float, default=3600.0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--ng-size", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    if args.seeds:
        seeds = list(args.seeds)
    elif args.split:
        seeds = split_seeds(args.split, args.count)
    else:
        parser.error("give --seeds or --split")
    if args.index is not None:
        seeds = [seeds[args.index]]
    args.out.mkdir(parents=True, exist_ok=True)
    instances = args.out / "instances"
    instances.mkdir(exist_ok=True)

    env = build_env(args)
    pending = []
    for seed in seeds:
        target = args.out / f"seed_{seed}.json"
        if target.exists() and not args.overwrite:
            print(f"seed {seed}: certificate exists, skipping", flush=True)
            continue
        env.reset(seed=seed)
        write_instance(env, instances / f"seed_{seed}.json")
        pending.append(seed)
    if not pending:
        return 0

    batch = instances / f"batch_{pending[0]}_{len(pending)}.txt"
    batch.write_text("".join(f"{instances / f'seed_{s}.json'} {instances / f'seed_{s}.result.json'}\n"
                             for s in pending))
    params = BPParameters(time_limit_seconds=args.time_limit, threads=args.threads,
                          ng_size=args.ng_size, verbose=True)
    log = instances / f"solver_{pending[0]}_{len(pending)}.log"
    started = time.perf_counter()
    with open(log, "w") as stream:
        subprocess.run([str(RUNNER), "--batch", str(batch), *params.cli()], check=True,
                       stdout=stream, stderr=subprocess.STDOUT,
                       env={**os.environ, "EVRP_BP_THREADS": str(args.threads)})
    print(f"solved {len(pending)} scenario(s) in {time.perf_counter() - started:.1f}s; log {log}", flush=True)

    failures = 0
    for seed in pending:
        certificate = json.loads((instances / f"seed_{seed}.result.json").read_text())
        env.reset(seed=seed)
        routes = [tuple((int(n), float(v), bool(c)) for n, v, c in r["witness"]) for r in certificate["routes"]]
        replay = nominal_replay(env, routes) if routes else {"feasible": False, "reason": "no incumbent"}
        ports = {int(n): int(env.charging_station.charger_capacity[n]) for n in env.charging_nodes}
        certificate["validation"] = {
            "nominal_replay": {k: v for k, v in replay.items() if k != "metrics"},
            "relaxed_schedule_port_conflicts": port_conflicts(certificate, ports),
            "scenario_seed": seed,
            "config": args.config,
            "trucks": env.num_trucks,
            "customers": len(env.joint_instance.tasks),
        }
        ok = replay["feasible"] and abs(replay["total_travel_hours"] - certificate["objective_hours"]) <= 1e-6
        failures += not ok
        (args.out / f"seed_{seed}.json").write_text(json.dumps(certificate, indent=1))
        print(f"seed {seed}: {certificate['status']} objective {certificate['objective_hours']} "
              f"bound {certificate['best_bound_hours']} gap {certificate['gap']} "
              f"nodes {certificate['nodes_explored']} {certificate['wall_seconds']:.0f}s "
              f"replay {'OK' if ok else 'FAILED: ' + str(replay.get('reason'))}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
