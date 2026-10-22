"""Best-known solutions (BKS) for the three static long-haul suite problems.

Builds ``results/generalist_graphppo/bks/<problem>/validation.json`` in the
SWARM.md format from a portfolio of independent, non-learning methods, each
executed closed loop in the simulator on the same seed:

* ``greedy_teacher``: the generalist's greedy teacher (``auto``);
* ``alns_respect``: the existing day-ahead ALNS (``heuristics.alns``) with the
  plan-respecting executor (``respect_planned_routes``), total-time objective,
  queue-aware on public stations;
* ``lh_plan``: the long-haul ALNS plan executed by re-solving the waypoint /
  charging / rest continuation at every decision
  (``baselines.optimization.longhaul``);
* ``lh_rolling``: ``lh_plan`` plus rolling re-planning of the remaining
  assignment whenever a truck finishes a customer late or early.

It also writes train-seed demonstrations of the best method per seed
(``train_demonstrations.json``, replay-verified in ``GeneralistEnv``), the
nominal lower bounds and a README. Commands::

    PYTHONPATH=. .venv/bin/python scripts/campaigns/generalist_bks_static.py run \
        --problem tx_longhaul --methods greedy_teacher lh_plan --split validation --count 128 --workers 48
    ... merge --problem tx_longhaul
    ... bound --problem tx_longhaul --workers 48
    ... demos --problem tx_longhaul --count 256 --workers 48
    ... readme --problem tx_longhaul

Rows are appended to ``<problem>/runs/<method>__<split>.jsonl`` as episodes
finish, so an interrupted run resumes where it stopped. Never run on test.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STATIC_PROBLEMS = ("tx_longhaul", "tx_longhaul_congested", "pnw_longhaul_hos")
CONFIG_DIR = Path("EVRoutingEnv/config_files/generalist_suite")
OUT = Path(os.environ.get("BKS_OUT", ROOT / "results/generalist_graphppo/bks"))
METHODS = ("greedy_teacher", "alns_respect", "lh_plan", "lh_rolling", "lh_plan_safe")
STEP_CAP = 4000
SUMMARY_FIELDS = ("method", "success", "failed", "dynamic", "tasks", "served", "late", "lost", "unserved", "hours")


def config_path(problem: str) -> str:
    return str(CONFIG_DIR / f"{problem}.yaml")


def fast_env_class():
    from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv

    class FastEnv(EventDrivenTruckEnv):
        """The simulator without building an observation (identical dynamics)."""

        def _get_observation(self):
            return None

    return FastEnv


def make_env(problem: str):
    return fast_env_class()(config_path(problem), verbose=False, enable_plotting=False)


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


def run_episode(env, controller, problem: str, seed: int, method: str) -> dict:
    from algo.generalist_ppo import episode_summary

    env.reset(seed=int(seed))
    actions, started, done, capped, error = [], time.perf_counter(), False, False, None
    while not done:
        if len(actions) >= STEP_CAP:
            capped = True
            break
        try:
            action = int(controller(env))
        except (RuntimeError, ValueError) as exc:
            error = f"{type(exc).__name__}: {exc}"
            break
        actions.append(action)
        _, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    summary = episode_summary(env)
    if capped or error:
        summary["failed"] = True
        summary["success"] = False
        summary["termination_reason"] = "policy_step_cap" if capped else "policy_error"
    row = {
        "problem": problem, "seed": int(seed), "method": method,
        "seconds": time.perf_counter() - started, "actions": actions, **summary,
    }
    if error:
        row["error"] = error
    stats = getattr(controller, "stats", None)
    if stats:
        row["stats"] = {k: v for k, v in stats.items() if k != "plan"}
        plan = stats.get("plan")
        if plan:
            row["plan"] = {k: v for k, v in plan.items() if k != "routes"}
    diagnostics = getattr(controller, "diagnostics", None)
    if callable(diagnostics):
        row["stats"] = diagnostics()
    return row


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------

_WORKER_ENV: dict = {}


def _episode_job(arguments):
    problem, method, seed = arguments
    env = _WORKER_ENV.get(problem)
    if env is None:
        env = _WORKER_ENV[problem] = make_env(problem)
    try:
        return run_episode(env, build_method(method, problem), problem, seed, method)
    except Exception as exc:  # noqa: BLE001 - recorded, never dropped
        import traceback

        return {"problem": problem, "seed": int(seed), "method": method, "crashed": True,
                "error": "".join(traceback.format_exception(exc))[-2000:]}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def seeds_for(split: str, count: int, offset: int = 0) -> list[int]:
    from scripts.evaluation.canonical_harness import split_seeds

    if split == "test":
        raise SystemExit("the test bank is locked: BKS uses validation and train seeds only")
    return split_seeds(split, offset + count)[offset:]


def cmd_run(args) -> None:
    import multiprocessing as mp

    problem = args.problem
    folder = OUT / problem / "runs"
    folder.mkdir(parents=True, exist_ok=True)
    seeds = seeds_for(args.split, args.count, args.offset)
    tasks = []
    for method in args.methods:
        path = folder / f"{method}__{args.split}.jsonl"
        done = {row["seed"] for row in read_jsonl(path) if not row.get("crashed")}
        tasks.extend((problem, method, seed) for seed in seeds if seed not in done)
    print(f"{problem}: {len(tasks)} episodes to run on {args.workers} workers", flush=True)
    if not tasks:
        return
    # Build the network (and its transport cache) once before forking.
    _WORKER_ENV[problem] = make_env(problem)
    if any(method.startswith("lh_") for method in args.methods):
        from EVRoutingEnv.baselines.optimization.longhaul.network import charge_curve, network_for

        env = _WORKER_ENV[problem]
        env.reset(seed=seeds[0])
        hos = getattr(env, "hos", None)
        network_for(env, max_hop_hours=float(hos.regime.driving_before_break) if hos is not None else None)
        charge_curve(env)
    started = time.time()
    handles = {method: open(folder / f"{method}__{args.split}.jsonl", "a") for method in args.methods}
    try:
        if args.workers <= 1:
            iterator = map(_episode_job, tasks)
            pool = None
        else:
            pool = mp.get_context("fork").Pool(args.workers, maxtasksperchild=args.max_tasks_per_child)
            iterator = pool.imap_unordered(_episode_job, tasks, chunksize=1)
        for count, row in enumerate(iterator, start=1):
            handle = handles[row["method"]]
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            if count % 10 == 0 or count == len(tasks):
                print(f"  {count}/{len(tasks)} done, {time.time() - started:.0f}s; last {row['method']} "
                      f"seed {row['seed']}: success={row.get('success')} hours={row.get('hours')}", flush=True)
        if pool is not None:
            pool.close()
            pool.join()
    finally:
        for handle in handles.values():
            handle.close()


# ---------------------------------------------------------------------------
# Merge into the SWARM.md format
# ---------------------------------------------------------------------------


def outcome_key(row: dict) -> tuple:
    """Lexicographic: not failed > success > more served > fewer lost > fewer late > fewer hours."""
    return (
        bool(row["failed"]), not bool(row["success"]), -int(row["served"]),
        int(row["lost"]), int(row["late"]), float(row["hours"]),
    )


def compact(row: dict, method: str) -> dict:
    out = {field: row[field] for field in SUMMARY_FIELDS if field in row and field != "method"}
    out = {"method": method, **out}
    out["hours"] = float(out["hours"])
    return out


def load_method_rows(problem: str, split: str, methods=None) -> dict[str, dict[int, dict]]:
    folder = OUT / problem / "runs"
    out = {}
    for path in sorted(folder.glob(f"*__{split}.jsonl")):
        method = path.name.split("__")[0]
        if method == "bound" or (methods and method not in methods):
            continue
        rows = {}
        for row in read_jsonl(path):
            if row.get("crashed"):
                continue
            rows[int(row["seed"])] = row
        out[method] = rows
    return out


def cmd_merge(args) -> None:
    problem = args.problem
    seeds = seeds_for("validation", 128)
    methods = load_method_rows(problem, "validation", args.methods)
    bounds = {int(row["seed"]): row for row in read_jsonl(OUT / problem / "runs" / "bound__validation.jsonl")}
    result, missing = {}, []
    for seed in seeds:
        entries = {name: compact(rows[seed], name) for name, rows in methods.items() if seed in rows}
        if not entries:
            missing.append(seed)
            continue
        best_name = min(entries, key=lambda name: outcome_key(entries[name]))
        bound = bounds.get(seed)
        result[str(seed)] = {
            "best": dict(entries[best_name]),
            "methods": entries,
            "bound": (
                {"kind": "nominal_lower_bound_hours", "value": float(bound["value"]),
                 "proven": bool(bound.get("proven", False))}
                if bound is not None and bound.get("value") is not None
                else {"kind": "none", "value": 0.0, "proven": False}
            ),
        }
    if missing:
        print(f"WARNING: {len(missing)} validation seeds have no method rows: {missing[:5]}...", flush=True)
    target = OUT / problem / "validation.json"
    target.write_text(json.dumps(result, indent=1))
    summary = summarize(problem, result, methods)
    (OUT / problem / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def summarize(problem: str, result: dict, methods: dict) -> dict:
    import numpy as np

    seeds = [int(seed) for seed in result]
    out = {"problem": problem, "seeds": len(seeds), "methods": {}}

    def block(rows):
        rows = [row for row in rows if row is not None]
        if not rows:
            return None
        success = [row for row in rows if row["success"]]
        return {
            "episodes": len(rows),
            "success_rate": len(success) / len(rows),
            "failure_rate": float(np.mean([bool(row["failed"]) for row in rows])),
            "mean_served": float(np.mean([row["served"] for row in rows])),
            "mean_hours": float(np.mean([row["hours"] for row in rows])),
            "mean_hours_successful": float(np.mean([row["hours"] for row in success])) if success else None,
            "mean_seconds": float(np.mean([row.get("seconds", 0.0) for row in rows])) if "seconds" in rows[0] else None,
        }

    for name, rows in methods.items():
        out["methods"][name] = block([rows.get(seed) for seed in seeds])
    best = [result[str(seed)]["best"] for seed in seeds]
    out["bks"] = block(best)
    out["bks"]["best_method_counts"] = {
        name: sum(row["method"] == name for row in best) for name in sorted({row["method"] for row in best})
    }
    bounded = [
        (result[str(seed)]["bound"]["value"], result[str(seed)]["best"]["hours"])
        for seed in seeds
        if result[str(seed)]["bound"]["kind"] != "none" and result[str(seed)]["best"]["success"]
    ]
    if bounded:
        gaps = [hours / bound - 1.0 for bound, hours in bounded if bound > 0]
        out["bound"] = {
            "seeds": len(bounded),
            "mean_bound": float(np.mean([b for b, _ in bounded])),
            "mean_bks_hours": float(np.mean([h for _, h in bounded])),
            "mean_gap": float(np.mean(gaps)),
            "median_gap": float(np.median(gaps)),
            "max_gap": float(np.max(gaps)),
            "proven_fraction": float(np.mean([result[str(s)]["bound"]["proven"] for s in seeds
                                              if result[str(s)]["bound"]["kind"] != "none"])),
        }
    # Paired comparison of each method with the BKS on jointly successful seeds.
    for name, rows in methods.items():
        pairs = [(rows[s]["hours"], result[str(s)]["best"]["hours"]) for s in seeds
                 if s in rows and rows[s]["success"] and result[str(s)]["best"]["success"]]
        if pairs and out["methods"].get(name):
            out["methods"][name]["mean_excess_vs_bks"] = float(np.mean([a / b - 1.0 for a, b in pairs]))
    return out


# ---------------------------------------------------------------------------
# Train demonstrations
# ---------------------------------------------------------------------------


def _replay_job(arguments):
    problem, seed, actions = arguments
    from algo.generalist_observation import GeneralistEnv
    from algo.generalist_ppo import episode_summary

    env = _WORKER_ENV.get(("generalist", problem))
    if env is None:
        env = _WORKER_ENV[("generalist", problem)] = GeneralistEnv(config_path(problem))
    env.reset(seed=int(seed))
    for action in actions:
        _, _, terminated, truncated, _ = env.step(int(action))
        if terminated or truncated:
            break
    return int(seed), episode_summary(env)


def cmd_demos(args) -> None:
    import multiprocessing as mp

    problem = args.problem
    seeds = seeds_for("train", args.count, args.offset)
    methods = load_method_rows(problem, "train", args.methods)
    chosen = {}
    for seed in seeds:
        rows = [row for rows in methods.values() if (row := rows.get(seed)) is not None and "actions" in row]
        rows = [row for row in rows if not row["failed"]]
        if rows:
            chosen[seed] = min(rows, key=outcome_key)
    print(f"{problem}: {len(chosen)} non-failed train episodes of {len(seeds)} seeds", flush=True)
    jobs = [(problem, seed, row["actions"]) for seed, row in chosen.items()]
    if args.workers <= 1:
        replays = dict(map(_replay_job, jobs))
    else:
        with mp.get_context("fork").Pool(args.workers) as pool:
            replays = dict(pool.map(_replay_job, jobs, chunksize=1))
    demos, mismatches = [], []
    for seed, row in chosen.items():
        replay = replays[seed]
        same = (
            replay["success"] == row["success"] and replay["served"] == row["served"]
            and replay["failed"] == row["failed"] and abs(replay["hours"] - row["hours"]) < 1e-6
            and replay["steps"] == len(row["actions"])
        )
        if not same:
            mismatches.append(seed)
            continue
        demos.append({
            "problem": problem, "seed": seed, "policy": row["method"], "actions": row["actions"],
            **{k: row[k] for k in ("success", "failed", "termination_reason", "hours", "travel_hours",
                                   "charging_sessions", "completed_fraction", "steps", "dynamic",
                                   "tasks", "served", "late", "lost", "unserved") if k in row},
            "replay_verified": True,
        })
    target = OUT / problem / "train_demonstrations.json"
    target.write_text(json.dumps(demos))
    successes = sum(d["success"] for d in demos)
    print(f"{problem}: wrote {len(demos)} replay-verified demonstrations ({successes} successful), "
          f"{len(mismatches)} replay mismatches {mismatches[:5]}", flush=True)
    (OUT / problem / "train_demonstrations_summary.json").write_text(json.dumps({
        "problem": problem, "split": "train", "offset": args.offset, "seeds": len(seeds),
        "demonstrations": len(demos), "successful": successes, "replay_mismatches": mismatches,
        "methods": {m: sum(d["policy"] == m for d in demos) for m in sorted({d["policy"] for d in demos})},
        "mean_hours": (sum(d["hours"] for d in demos) / len(demos)) if demos else None,
    }, indent=2))


# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------


def _bound_job(arguments):
    problem, seed, time_limit = arguments
    from EVRoutingEnv.baselines.optimization.longhaul.bounds import nominal_lower_bound

    env = _WORKER_ENV.get(problem)
    if env is None:
        env = _WORKER_ENV[problem] = make_env(problem)
    env.reset(seed=int(seed))
    return {"seed": int(seed), **nominal_lower_bound(env, time_limit_seconds=time_limit)}


def cmd_bound(args) -> None:
    import multiprocessing as mp

    problem = args.problem
    path = OUT / problem / "runs" / "bound__validation.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {row["seed"] for row in read_jsonl(path)}
    seeds = [seed for seed in seeds_for("validation", args.count) if seed not in done]
    _WORKER_ENV[problem] = make_env(problem)
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
# Source snapshot for SLURM jobs
# ---------------------------------------------------------------------------


def cmd_snapshot(args) -> None:
    """Copy the code a job runs into ``bks/_source/<name>`` (data is linked)."""
    target = OUT / "_source" / args.name
    if target.exists():
        raise SystemExit(f"{target} exists")
    for package in ("EVRoutingEnv", "algo", "scripts"):
        for path in (ROOT / package).rglob("*"):
            if "__pycache__" in path.parts or path.is_dir():
                continue
            relative = path.relative_to(ROOT)
            if relative.parts[:2] == ("EVRoutingEnv", "data"):
                continue
            if path.suffix not in (".py", ".yaml", ".yml", ".json", ".sh", ".sbatch"):
                continue
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
    os.symlink(ROOT / "EVRoutingEnv" / "data", target / "EVRoutingEnv" / "data")
    print(target)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--problem", required=True, choices=STATIC_PROBLEMS)
    run.add_argument("--methods", nargs="+", required=True, choices=METHODS)
    run.add_argument("--split", default="validation", choices=("validation", "train"))
    run.add_argument("--count", type=int, default=128)
    run.add_argument("--offset", type=int, default=0)
    run.add_argument("--workers", type=int, default=1)
    run.add_argument("--max-tasks-per-child", type=int, default=None)
    merge = sub.add_parser("merge")
    merge.add_argument("--problem", required=True, choices=STATIC_PROBLEMS)
    merge.add_argument("--methods", nargs="*", default=None)
    demos = sub.add_parser("demos")
    demos.add_argument("--problem", required=True, choices=STATIC_PROBLEMS)
    demos.add_argument("--count", type=int, default=256)
    demos.add_argument("--offset", type=int, default=0)
    demos.add_argument("--workers", type=int, default=1)
    demos.add_argument("--methods", nargs="*", default=None)
    bound = sub.add_parser("bound")
    bound.add_argument("--problem", required=True, choices=STATIC_PROBLEMS)
    bound.add_argument("--count", type=int, default=128)
    bound.add_argument("--workers", type=int, default=1)
    bound.add_argument("--time-limit", type=float, default=120.0)
    snapshot = sub.add_parser("snapshot")
    snapshot.add_argument("--name", required=True)
    args = parser.parse_args(argv)
    {"run": cmd_run, "merge": cmd_merge, "demos": cmd_demos, "bound": cmd_bound,
     "snapshot": cmd_snapshot}[args.command](args)


if __name__ == "__main__":
    main()
