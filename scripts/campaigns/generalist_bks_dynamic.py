"""Best-known solutions (BKS) for the two dynamic request suite problems.

``ca_sameday`` (same-day freight, hard deadlines) and ``ca_drayage_combined``
(electric drayage with US hours of service and public stations). The portfolio,
each method executed closed loop on the same seed and never clairvoyant:

* ``greedy_dispatch``: the generalist's teacher for request problems;
* ``wait_at_depot``: trucks wait at the depot until a request is loadable;
* ``kullman_myopic``: Kullman's myopic station rules (drayage only: it needs
  public stations);
* ``rolling_request``: the rolling-horizon request planner, an ALNS
  re-optimization of every revealed request at every decision
  (``heuristics.rolling_request_planner``) with the parameters of
  ``algo.generalist_experts.ROLLING``, which is also the DAgger expert.

The bound per seed is the same planner made clairvoyant: it also sees the
request stream the simulator has not revealed yet, and plans with a raised
ALNS budget (``BOUND_ITERATIONS``; the planner caps iterations at four per
pending job). It is a heuristic hindsight estimate (``proven: false``): at
full scale it can fall below the causal BKS, so the recorded bound is the
larger of the two (``raised_to_bks`` marks those seeds) and the gate reads it
as headroom; ``check`` compares the clairvoyant planner with the exact search
``solve_request_hindsight`` on tiny instances (one truck, one to three
requests, 12 h, no uncertainty), the only ones that search exhausts. Outputs follow ``generalist_bks_common.py``. Commands::

    PYTHONPATH=. .venv/bin/python scripts/campaigns/generalist_bks_dynamic.py run \\
        --problem ca_sameday --methods greedy_dispatch wait_at_depot rolling_request --count 128 --workers 18
    ... bound --problem ca_sameday --count 128 --workers 18
    ... merge --problem ca_sameday
    ... run --problem ca_sameday --methods greedy_dispatch wait_at_depot rolling_request --split train --count 256 --workers 18
    ... demos --problem ca_sameday --count 256 --workers 18
    ... check --problem ca_sameday --seeds 20
    ... snapshot --name v1

The test split is refused unless ``--allow-test`` is given (P5 only).
"""

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.campaigns import generalist_bks_common as common  # noqa: E402
from scripts.campaigns.generalist_bks_common import OUT, make_env, read_jsonl, seeds_for  # noqa: E402

DYNAMIC_PROBLEMS = ("ca_sameday", "ca_drayage_combined")
METHODS = ("greedy_dispatch", "wait_at_depot", "kullman_myopic", "rolling_request")
PUBLIC_ONLY = frozenset({"kullman_myopic"})
BOUND_KIND = "clairvoyant_rolling_served"
# The bound's ALNS budget: enough that the planner's own cap (4 x pending jobs)
# binds on the full stream. Default 60 served 30/29 on two drayage seeds
# against the causal planner's 35/34; the cap gave 32/36 (2026-10-22).
BOUND_ITERATIONS = 300
BOUND_MIN_ITERATIONS = 60


def build_method(method: str, problem: str):
    """A fresh non-clairvoyant controller ``env -> action`` for one episode."""
    if method == "rolling_request":
        from algo.generalist_experts import ROLLING
        from EVRoutingEnv.baselines.heuristics.rolling_request_planner import RollingHorizonRequestPolicy

        return RollingHorizonRequestPolicy(ROLLING)
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}")
    from algo.generalist_ppo import build_teacher

    return build_teacher(method, None)


def check_methods(problem: str, methods) -> None:
    if problem == "ca_sameday" and PUBLIC_ONLY & set(methods):
        raise SystemExit(f"{sorted(PUBLIC_ONLY & set(methods))} need public stations; {problem} has none")


def bound_entry(row: dict, best: dict | None = None) -> dict | None:
    """The clairvoyant planner's services, floored at the BKS of the seed."""
    if row.get("value") is None:
        return None
    clairvoyant = float(row["value"])
    floor = float(best["served"]) if best is not None else clairvoyant
    return {
        "kind": BOUND_KIND, "value": max(clairvoyant, floor), "proven": False,
        "clairvoyant": clairvoyant, "raised_to_bks": clairvoyant < floor - 1e-9,
    }


# ---------------------------------------------------------------------------
# Hindsight bound: the clairvoyant rolling planner
# ---------------------------------------------------------------------------


def bound_planner(problem: str, env, iterations: int = BOUND_ITERATIONS, minimum: int = BOUND_MIN_ITERATIONS):
    """The clairvoyant rolling planner at the bound's raised budget."""
    from dataclasses import replace

    from algo.generalist_experts import ROLLING
    from EVRoutingEnv.baselines.heuristics.rolling_request_planner import RollingHorizonRequestPolicy

    if not getattr(env, "dynamic_requests", False):
        raise ValueError(f"{problem}: the bound needs a dynamic request mode")
    return RollingHorizonRequestPolicy(replace(ROLLING, clairvoyant=True, iterations=iterations, min_iterations=minimum))


def _bound_job(arguments):
    problem, seed, iterations = arguments
    env = common._WORKER_ENV.get(problem)
    if env is None:
        env = common._WORKER_ENV[problem] = make_env(problem)
    row = common.run_episode(env, bound_planner(problem, env, iterations), problem, seed, "clairvoyant_rolling")
    return {
        "seed": int(seed), "kind": BOUND_KIND, "value": int(row["served"]), "proven": False, "iterations": iterations,
        **{k: row[k] for k in ("served", "late", "lost", "hours", "failed", "success", "seconds",
                               "termination_reason") if k in row},
    }


def cmd_bound(args) -> None:
    problem = args.problem
    path = OUT / problem / "runs" / f"bound__{args.split}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {row["seed"] for row in read_jsonl(path)}
    seeds = [seed for seed in seeds_for(args.split, args.count, 0, args.allow_test) if seed not in done]
    print(f"{problem}: clairvoyant bound ({args.iterations} ALNS iterations) on {len(seeds)} seeds "
          f"with {args.workers} workers", flush=True)
    common._WORKER_ENV[problem] = make_env(problem)
    jobs = [(problem, seed, args.iterations) for seed in seeds]
    with open(path, "a") as handle:
        if args.workers <= 1:
            iterator = map(_bound_job, jobs)
            pool = None
        else:
            pool = mp.get_context("fork").Pool(args.workers)
            iterator = pool.imap_unordered(_bound_job, jobs, chunksize=1)
        for row in iterator:
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            print(f"  bound seed {row['seed']}: served {row['value']} in {row.get('seconds', 0):.0f}s", flush=True)
        if pool is not None:
            pool.close()
            pool.join()


# ---------------------------------------------------------------------------
# Sanity check of the clairvoyant planner against the exact hindsight search
# ---------------------------------------------------------------------------


def reduce(
    config: dict, *, trucks: int = 1, per_epoch: int = 1, max_time: float = 12.0,
    keep_public: bool = False, keep_hos: bool = True, stochastic: bool = False,
) -> dict:
    """``config`` shrunk so that ``solve_request_hindsight`` exhausts its tree.

    Measured on 2026-10-22: one truck, one request per epoch and a 12 h horizon
    give one to three requests on California, proven in seconds; at 16 h a
    three-request instance already exceeds 20,000 nodes (about 10 ms each).

    The stream is thinned through the per-epoch cap and the horizon;
    ``requests.max_requests`` is an envelope the stream must fit, never a cap,
    so it stays. Without ``stochastic`` the draws are switched off and the
    battery made unlimited, as in the hindsight oracle tests, so charging
    never binds and the search can drop charger actions exactly.
    """
    config = copy.deepcopy(config)
    config["environment"].update(num_trucks=int(trucks), max_time=float(max_time))
    block = config["requests"]
    block["max_requests_per_epoch"] = int(per_epoch)
    block["initial_requests"] = min(int(block.get("initial_requests", 0)), int(per_epoch))
    if not keep_public and "public_charging" in config:
        config["public_charging"]["enabled"] = False
    if not keep_hos and "hours_of_service" in config:
        # On a 12 h horizon the drayage model's idle-rest rule takes a 10 h daily
        # rest early and misses every request; without it the check compares routing.
        config["hours_of_service"]["enabled"] = False
    if not stochastic:
        config["truck"]["battery_capacity"] = 1e6
        config["traffic"].update(enable_traffic=False, enable_energy_uncertainty=False)
        config["delivery"]["enable_stochastic_unloading"] = False
    return config


def reduced_config(problem: str, **settings) -> dict:
    from EVRoutingEnv.utils.utils import load_config

    return reduce(load_config(common.config_path(problem)), **settings)


def run_check(problem: str, config: dict, seeds, *, max_nodes: int, stochastic: bool, out_dir: Path | None = None) -> dict:
    """Exact hindsight search vs the clairvoyant planner on every seed; writes ``hindsight_check.json``."""
    from algo.generalist_experts import rolling_request
    from EVRoutingEnv.baselines.optimization.perfect_information import solve_request_hindsight

    env_class = common.fast_env_class()
    rows = []
    for seed in seeds:
        exact_env = env_class(copy.deepcopy(config), verbose=False, enable_plotting=False)
        exact_env.reset(seed=int(seed))
        count = len(exact_env.request_engine.registry)
        started = time.perf_counter()
        solution = solve_request_hindsight(exact_env, max_nodes=max_nodes, routing_actions=stochastic)
        exact_seconds = time.perf_counter() - started
        planner_env = env_class(copy.deepcopy(config), verbose=False, enable_plotting=False)
        row = common.run_episode(
            planner_env, rolling_request(problem, planner_env, clairvoyant=True), problem, seed, "clairvoyant_rolling",
        )
        rows.append({
            "seed": int(seed), "requests": count, "exact_served": int(solution.served_on_time),
            "exact_proven": bool(solution.proven_optimal), "nodes_expanded": int(solution.nodes_expanded),
            "exact_seconds": exact_seconds, "planner_served": int(row["served"]),
            "planner_failed": bool(row["failed"]), "planner_seconds": float(row["seconds"]),
        })
        print(f"  seed {seed}: {count} requests; exact {solution.served_on_time} "
              f"({'proven' if solution.proven_optimal else 'capped'}, {solution.nodes_expanded} nodes, "
              f"{exact_seconds:.1f}s); clairvoyant planner {row['served']} ({row['seconds']:.1f}s)", flush=True)
    proven = [row for row in rows if row["exact_proven"]]
    gaps = [(row["exact_served"] - row["planner_served"]) / max(1, row["exact_served"]) for row in proven]
    summary = {
        "problem": problem, "trucks": config["environment"]["num_trucks"],
        "per_epoch": config["requests"]["max_requests_per_epoch"], "max_time": config["environment"]["max_time"],
        "stochastic": stochastic, "max_nodes": max_nodes,
        "seeds": len(rows), "proven_seeds": len(proven),
        "matches": sum(row["planner_served"] == row["exact_served"] for row in proven),
        "planner_exceeds": sum(row["planner_served"] > row["exact_served"] for row in proven),
        "mean_gap": (sum(gaps) / len(gaps)) if gaps else None,
        "mean_requests": (sum(r["requests"] for r in rows) / len(rows)) if rows else None,
        "mean_exact_seconds": (sum(r["exact_seconds"] for r in rows) / len(rows)) if rows else None,
        "mean_planner_seconds": (sum(r["planner_seconds"] for r in rows) / len(rows)) if rows else None,
    }
    summary["verdict"] = (
        "PASS" if proven and summary["planner_exceeds"] == 0 and summary["matches"] >= 0.9 * len(proven) else "FAIL"
    )
    out_dir = out_dir or (OUT / problem)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "hindsight_check.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"hindsight check {summary['verdict']}: the clairvoyant planner never exceeds the exact bound "
          f"({summary['planner_exceeds']} of {len(proven)}) and matches it on {summary['matches']} of {len(proven)}",
          flush=True)
    return summary


def cmd_check(args) -> int:
    config = reduced_config(
        args.problem, trucks=args.trucks, per_epoch=args.per_epoch, max_time=args.max_time,
        keep_public=args.keep_public, keep_hos=not args.no_hos, stochastic=args.stochastic,
    )
    seeds = seeds_for("validation", args.seeds, args.offset)
    summary = run_check(args.problem, config, seeds, max_nodes=args.max_nodes, stochastic=args.stochastic)
    return 0 if summary["verdict"] == "PASS" else 1


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_run(args) -> None:
    check_methods(args.problem, args.methods)
    common.run_portfolio(
        args.problem, args.methods, args.split, args.count, args.offset, args.workers, build_method,
        max_tasks_per_child=args.max_tasks_per_child, allow_test=args.allow_test,
    )


def cmd_merge(args) -> None:
    common.merge_split(
        args.problem, args.split, bound_entry, "served", methods=args.methods, count=args.count,
        allow_test=args.allow_test,
    )


def cmd_demos(args) -> None:
    common.write_demonstrations(args.problem, args.count, args.offset, args.workers, args.methods)


def cmd_snapshot(args) -> None:
    common.snapshot(args.name)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--problem", required=True, choices=DYNAMIC_PROBLEMS)
    run.add_argument("--methods", nargs="+", required=True, choices=METHODS)
    common.add_split_arguments(run, default_count=128)
    run.add_argument("--workers", type=int, default=1)
    run.add_argument("--max-tasks-per-child", type=int, default=None)
    merge = sub.add_parser("merge")
    merge.add_argument("--problem", required=True, choices=DYNAMIC_PROBLEMS)
    merge.add_argument("--methods", nargs="*", default=None)
    merge.add_argument("--split", default="validation", choices=("validation", "test"))
    merge.add_argument("--count", type=int, default=None, help="Seeds of the split (default: 128 validation, 500 test).")
    merge.add_argument("--allow-test", action="store_true", help="Open the test split (P5 only).")
    demos = sub.add_parser("demos")
    demos.add_argument("--problem", required=True, choices=DYNAMIC_PROBLEMS)
    demos.add_argument("--count", type=int, default=256)
    demos.add_argument("--offset", type=int, default=0)
    demos.add_argument("--workers", type=int, default=1)
    demos.add_argument("--methods", nargs="*", default=None)
    bound = sub.add_parser("bound")
    bound.add_argument("--problem", required=True, choices=DYNAMIC_PROBLEMS)
    bound.add_argument("--split", default="validation", choices=("validation", "test"))
    bound.add_argument("--count", type=int, default=128)
    bound.add_argument("--workers", type=int, default=1)
    bound.add_argument("--iterations", type=int, default=BOUND_ITERATIONS,
                       help="ALNS iterations per re-plan of the clairvoyant planner (capped at 4 per pending job).")
    bound.add_argument("--allow-test", action="store_true", help="Open the test split (P5 only).")
    check = sub.add_parser("check", help="Clairvoyant planner vs the exact hindsight search on reduced instances.")
    check.add_argument("--problem", required=True, choices=DYNAMIC_PROBLEMS)
    check.add_argument("--seeds", type=int, default=10)
    check.add_argument("--offset", type=int, default=0)
    check.add_argument("--trucks", type=int, default=1)
    check.add_argument("--per-epoch", type=int, default=1, help="Requests per epoch of the thinned stream.")
    check.add_argument("--max-time", type=float, default=12.0, help="12 h with one request per epoch gives 1-3 requests.")
    check.add_argument("--max-nodes", type=int, default=20_000,
                       help="About 10 ms per node: a 3-request instance exceeds 20k nodes, so keep the instances tiny.")
    check.add_argument("--keep-public", action="store_true", help="Keep public stations (drayage); slower search.")
    check.add_argument("--no-hos", action="store_true",
                       help="Disable hours of service in the reduced instances (the idle-rest rule confounds a 12 h horizon).")
    check.add_argument("--stochastic", action="store_true", help="Keep the draws and the battery; charger actions stay in.")
    snapshot = sub.add_parser("snapshot")
    snapshot.add_argument("--name", required=True)
    args = parser.parse_args(argv)
    result = {"run": cmd_run, "merge": cmd_merge, "demos": cmd_demos, "bound": cmd_bound,
              "check": cmd_check, "snapshot": cmd_snapshot}[args.command](args)
    return int(result or 0)


if __name__ == "__main__":
    raise SystemExit(main())
