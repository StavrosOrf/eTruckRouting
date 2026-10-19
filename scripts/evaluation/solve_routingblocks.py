"""Solve scenarios with the RoutingBlocks baseline and write replay-validated certificates.

For every seed the nominal instance is exported, solved by ``run_solver`` in the
solver interpreter, and the plan is replayed on the environment with noise off.
The certificate, with that replay attached, is written to
``OUT/seed_<seed>.json``, where ``RoutingBlocksPolicy`` (``certificate_dir``)
and the campaign runner pick it up. When ``--bound-from`` points at a
branch-and-price certificate directory, the certified lower bound of the same
seed is attached and the gap to it is reported.

Examples:
    python -m scripts.evaluation.solve_routingblocks --trucks 50 --customers 100 \\
        --split test --count 20 --time-limit 600 --workers 8 --out results/rb_alns/ca_50x100
    # one Slurm array task per seed:
    python -m scripts.evaluation.solve_routingblocks --split test --count 20 \\
        --index $SLURM_ARRAY_TASK_ID --out results/rb_alns/ca_50x100
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

from EVRoutingEnv.baselines.optimization.branch_and_price.export import write_instance
from EVRoutingEnv.baselines.optimization.branch_and_price.validate import nominal_replay, port_conflicts
from EVRoutingEnv.baselines.optimization.routingblocks_alns.bridge import RBParameters, run_solver
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
    parser.add_argument("--split", choices=("train", "validation", "test"), help="seed namespace (with --count)")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--index", type=int, help="solve only this position of the seed list (Slurm arrays)")
    parser.add_argument("--time-limit", type=float, default=300.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--solver-seed", type=int, default=0)
    parser.add_argument("--objective", choices=("travel", "operating"), default="travel")
    parser.add_argument("--param", nargs=2, action="append", default=[], metavar=("KEY", "VALUE"),
                        help="any SearchParameters field, e.g. --param pool_interval_seconds 60")
    parser.add_argument("--warm-from", type=Path, action="append", default=[],
                        help="certificate whose routes seed the search; repeatable")
    parser.add_argument("--bound-from", type=Path,
                        help="directory of evrp-bp-result/v1 certificates (seed_<seed>.json) for the lower bound")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    if args.seeds:
        seeds = list(args.seeds)
    elif args.split:
        seeds = split_seeds(args.split, args.count, args.base_seed)
    else:
        parser.error("give --seeds or --split")
    if args.index is not None:
        seeds = [seeds[args.index]]
    args.out.mkdir(parents=True, exist_ok=True)
    instances = args.out / "instances"
    instances.mkdir(exist_ok=True)

    env = build_env(args)
    parameters = RBParameters(time_limit_seconds=args.time_limit, workers=args.workers, seed=args.solver_seed,
                              objective=args.objective, extra=dict(args.param))
    failures = 0
    for seed in seeds:
        target = args.out / f"seed_{seed}.json"
        if target.exists() and not args.overwrite:
            print(f"seed {seed}: certificate exists, skipping", flush=True)
            continue
        env.reset(seed=seed)
        sequences = []
        for path in args.warm_from:
            certificate = json.loads(Path(path).read_text())
            sequences += [[stop["node"] for stop in route["stops"] if stop["kind"] == "customer"]
                          for route in certificate.get("routes", [])]
        write_instance(env, instances / f"seed_{seed}.json", objective=args.objective, sequences=sequences)
        started = time.perf_counter()
        certificate = run_solver(instances / f"seed_{seed}.json", instances / f"seed_{seed}.result.json",
                                 parameters, log_path=instances / f"seed_{seed}.log",
                                 timeout=args.time_limit * 3 + 900)
        solve_seconds = time.perf_counter() - started

        env.reset(seed=seed)
        routes = [tuple((int(n), float(v), bool(c)) for n, v, c in r["witness"]) for r in certificate["routes"]]
        replay = nominal_replay(env, routes) if routes else {"feasible": False, "reason": "no incumbent"}
        ports = {int(n): int(env.charging_station.charger_capacity[n]) for n in env.charging_nodes}
        bound = None
        if args.bound_from is not None and (args.bound_from / f"seed_{seed}.json").exists():
            reference = json.loads((args.bound_from / f"seed_{seed}.json").read_text())
            bound = {"best_bound_hours": reference.get("best_bound_hours"),
                     "objective_hours": reference.get("objective_hours"), "status": reference.get("status")}
            if bound["best_bound_hours"] is not None:
                certificate["best_bound_hours"] = bound["best_bound_hours"]
                certificate["gap"] = max(0.0, (certificate["objective_hours"] - bound["best_bound_hours"])
                                         / certificate["objective_hours"])
        certificate["validation"] = {
            "nominal_replay": {k: v for k, v in replay.items() if k != "metrics"},
            "relaxed_schedule_port_conflicts": port_conflicts(certificate, ports),
            "scenario_seed": seed,
            "config": args.config,
            "trucks": env.num_trucks,
            "customers": len(env.joint_instance.tasks),
            "solve_seconds": solve_seconds,
            "branch_and_price_reference": bound,
        }
        ok = replay["feasible"] and abs(replay["total_travel_hours"] - certificate["objective_hours"]) <= 1e-6 \
            if args.objective == "travel" else replay["feasible"]
        failures += not ok
        target.write_text(json.dumps(certificate, indent=1))
        gap = certificate["gap"]
        print(f"seed {seed}: objective {certificate['objective_hours']:.4f} "
              f"bound {certificate['best_bound_hours']} gap {'-' if gap is None else format(gap, '.2%')} "
              f"{solve_seconds:.0f}s replay {'OK' if ok else 'FAILED: ' + str(replay.get('reason'))}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
