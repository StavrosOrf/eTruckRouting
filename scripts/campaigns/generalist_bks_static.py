"""Best-known solutions (BKS) for the three static long-haul suite problems.

Builds ``results/generalist_graphppo/bks/<problem>/validation.json`` (and
``test.json`` once P5 opens) from a portfolio of independent, non-learning
methods, each executed closed loop in the simulator on the same seed:

* ``greedy_teacher``: the generalist's greedy teacher (``auto``);
* ``alns_respect``: the existing day-ahead ALNS (``heuristics.alns``) with the
  plan-respecting executor (``respect_planned_routes``), total-time objective,
  queue-aware on public stations;
* ``lh_plan``: the long-haul ALNS plan executed by re-solving the waypoint /
  charging / rest continuation at every decision
  (``baselines.optimization.longhaul``);
* ``lh_rolling``: ``lh_plan`` plus rolling re-planning of the remaining
  assignment whenever a truck finishes a customer late or early;
* ``lh_plan_safe``: ``lh_plan`` with wider duration and energy margins.

It also writes train-seed demonstrations of the best method per seed
(``train_demonstrations.json``, replay-verified in ``GeneralistEnv``), the
nominal lower bounds and a source snapshot for SLURM jobs. The shared
machinery, the output format and the test-split guard live in
``generalist_bks_common.py``. Commands::

    PYTHONPATH=. .venv/bin/python scripts/campaigns/generalist_bks_static.py run \\
        --problem tx_longhaul --methods greedy_teacher lh_plan --split validation --count 128 --workers 18
    ... bound --problem tx_longhaul --workers 18
    ... merge --problem tx_longhaul
    ... run --problem tx_longhaul --methods greedy_teacher lh_plan --split train --count 256 --workers 18
    ... demos --problem tx_longhaul --count 256 --workers 18
    ... snapshot --name v1

Rows are appended to ``<problem>/runs/<method>__<split>.jsonl`` as episodes
finish, so an interrupted run resumes where it stopped. The test split is
refused unless ``--allow-test`` is given (P5 only).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.campaigns import generalist_bks_common as common  # noqa: E402
from scripts.campaigns.generalist_bks_common import OUT, make_env, read_jsonl, seeds_for  # noqa: E402

STATIC_PROBLEMS = ("tx_longhaul", "tx_longhaul_congested", "pnw_longhaul_hos")
METHODS = ("greedy_teacher", "alns_respect", "lh_plan", "lh_rolling", "lh_plan_safe")
BOUND_KIND = "nominal_lower_bound_hours"


def build_method(method: str, problem: str):
    """A fresh controller ``env -> action`` for one episode."""
    if method == "greedy_teacher":
        from algo.generalist_ppo import build_teacher

        return build_teacher("auto", _DummyStatic())
    if method == "alns_respect":
        from EVRoutingEnv.baselines.heuristics.alns import ALNSParameters, ALNSPolicy

        return ALNSPolicy(ALNSParameters(
            iterations=20000, time_limit_seconds=10.0, objective="total_time",
            respect_planned_routes=True, queue_aware=problem == "tx_longhaul_congested",
        ))
    from EVRoutingEnv.baselines.optimization.longhaul.planner import PlannerSettings
    from EVRoutingEnv.baselines.optimization.longhaul.policy import LonghaulPlanPolicy, LonghaulSettings

    planner = PlannerSettings(iterations=8000, time_limit_seconds=40.0)
    if method == "lh_plan":
        return LonghaulPlanPolicy(LonghaulSettings(name=method, planner=planner, restarts=3))
    if method == "lh_rolling":
        return LonghaulPlanPolicy(LonghaulSettings(name=method, planner=planner, restarts=3, replan="events"))
    if method == "lh_plan_safe":
        from dataclasses import replace

        return LonghaulPlanPolicy(LonghaulSettings(
            name=method, planner=replace(planner, duration_margin=10.0, energy_expected=1.06),
            restarts=3, execution_margin=8.0,
        ))
    raise ValueError(f"unknown method {method!r}")


class _DummyStatic:
    dynamic_requests = False


def warm_network(env, seeds, methods) -> None:
    """Build the long-haul network and charge curve once in the parent, before the fork."""
    if not any(method.startswith("lh_") for method in methods):
        return
    from EVRoutingEnv.baselines.optimization.longhaul.network import charge_curve, network_for

    env.reset(seed=seeds[0])
    hos = getattr(env, "hos", None)
    network_for(env, max_hop_hours=float(hos.regime.driving_before_break) if hos is not None else None)
    charge_curve(env)


def bound_entry(row: dict, best: dict | None = None) -> dict | None:
    if row.get("value") is None:
        return None
    return {"kind": BOUND_KIND, "value": float(row["value"]), "proven": bool(row.get("proven", False))}


# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------


def _bound_job(arguments):
    problem, seed, time_limit = arguments
    from EVRoutingEnv.baselines.optimization.longhaul.bounds import nominal_lower_bound

    env = common._WORKER_ENV.get(problem)
    if env is None:
        env = common._WORKER_ENV[problem] = make_env(problem)
    env.reset(seed=int(seed))
    return {"seed": int(seed), **nominal_lower_bound(env, time_limit_seconds=time_limit)}


def cmd_bound(args) -> None:
    problem = args.problem
    path = OUT / problem / "runs" / f"bound__{args.split}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {row["seed"] for row in read_jsonl(path)}
    seeds = [seed for seed in seeds_for(args.split, args.count, 0, args.allow_test) if seed not in done]
    common._WORKER_ENV[problem] = make_env(problem)
    jobs = [(problem, seed, args.time_limit) for seed in seeds]
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
            print(f"  bound seed {row['seed']}: {row.get('value')} ({row.get('status')})", flush=True)
        if pool is not None:
            pool.close()
            pool.join()


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_run(args) -> None:
    common.run_portfolio(
        args.problem, args.methods, args.split, args.count, args.offset, args.workers, build_method,
        max_tasks_per_child=args.max_tasks_per_child, warm=warm_network, allow_test=args.allow_test,
    )


def cmd_merge(args) -> None:
    common.merge_split(
        args.problem, args.split, bound_entry, "hours", methods=args.methods, count=args.count,
        allow_test=args.allow_test,
    )


def cmd_demos(args) -> None:
    common.write_demonstrations(args.problem, args.count, args.offset, args.workers, args.methods)


def cmd_snapshot(args) -> None:
    common.snapshot(args.name)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--problem", required=True, choices=STATIC_PROBLEMS)
    run.add_argument("--methods", nargs="+", required=True, choices=METHODS)
    common.add_split_arguments(run, default_count=128)
    run.add_argument("--workers", type=int, default=1)
    run.add_argument("--max-tasks-per-child", type=int, default=None)
    merge = sub.add_parser("merge")
    merge.add_argument("--problem", required=True, choices=STATIC_PROBLEMS)
    merge.add_argument("--methods", nargs="*", default=None)
    merge.add_argument("--split", default="validation", choices=("validation", "test"))
    merge.add_argument("--count", type=int, default=None, help="Seeds of the split (default: 128 validation, 500 test).")
    merge.add_argument("--allow-test", action="store_true", help="Open the test split (P5 only).")
    demos = sub.add_parser("demos")
    demos.add_argument("--problem", required=True, choices=STATIC_PROBLEMS)
    demos.add_argument("--count", type=int, default=256)
    demos.add_argument("--offset", type=int, default=0)
    demos.add_argument("--workers", type=int, default=1)
    demos.add_argument("--methods", nargs="*", default=None)
    bound = sub.add_parser("bound")
    bound.add_argument("--problem", required=True, choices=STATIC_PROBLEMS)
    bound.add_argument("--split", default="validation", choices=("validation", "test"))
    bound.add_argument("--count", type=int, default=128)
    bound.add_argument("--workers", type=int, default=1)
    bound.add_argument("--time-limit", type=float, default=120.0)
    bound.add_argument("--allow-test", action="store_true", help="Open the test split (P5 only).")
    snapshot = sub.add_parser("snapshot")
    snapshot.add_argument("--name", required=True)
    args = parser.parse_args(argv)
    {"run": cmd_run, "merge": cmd_merge, "demos": cmd_demos, "bound": cmd_bound,
     "snapshot": cmd_snapshot}[args.command](args)


if __name__ == "__main__":
    main()
