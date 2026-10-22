"""Shared pieces of the best-known-solution (BKS) builders of the generalist suite.

Both builders, ``generalist_bks_static.py`` for the three long-haul problems and
``generalist_bks_dynamic.py`` for the two request problems, execute a portfolio
of independent, non-learning controllers closed loop in the simulator on the
same seeds, keep the best outcome per seed, and write

    OUT/<problem>/runs/<method>__<split>.jsonl   one row per episode, resumable
    OUT/<problem>/runs/bound__<split>.jsonl      one bound row per seed
    OUT/<problem>/<split>.json                   {seed: {"best", "methods", "bound"}}
    OUT/<problem>/summary.json                   (validation; summary_<split>.json otherwise)
    OUT/<problem>/train_demonstrations.json      replay-verified best train episodes

``<split>.json`` is what ``generalist_near_optimal_gate.py`` reads (``best`` and
``bound``) and what :mod:`algo.generalist_imitation` turns into paired reward
references. ``OUT`` is ``results/generalist_graphppo/bks`` or ``$BKS_OUT``. The
test split is refused unless the caller passes ``allow_test`` (P5 only).
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import shutil
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONFIG_DIR = Path("EVRoutingEnv/config_files/generalist_suite")
OUT = Path(os.environ.get("BKS_OUT", ROOT / "results/generalist_graphppo/bks"))
STEP_CAP = 4000
SUMMARY_FIELDS = ("method", "success", "failed", "dynamic", "tasks", "served", "late", "lost", "unserved", "hours")
SPLIT_SEEDS = {"validation": 128, "test": 500}


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


def run_episode(env, controller, problem: str, seed: int, method: str) -> dict:
    """One closed-loop episode of ``controller`` (``env -> action``) on ``seed``."""
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
# Workers: one environment per problem per process, built before the fork
# ---------------------------------------------------------------------------

_WORKER_ENV: dict = {}
_BUILD_METHOD = None


def _episode_job(arguments):
    problem, method, seed = arguments
    env = _WORKER_ENV.get(problem)
    if env is None:
        env = _WORKER_ENV[problem] = make_env(problem)
    try:
        return run_episode(env, _BUILD_METHOD(method, problem), problem, seed, method)
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


def seeds_for(split: str, count: int, offset: int = 0, allow_test: bool = False) -> list[int]:
    from scripts.evaluation.canonical_harness import split_seeds

    if split == "test" and not allow_test:
        raise SystemExit("the test bank is locked: BKS uses validation and train seeds only (--allow-test opens P5)")
    return split_seeds(split, offset + count)[offset:]


def run_portfolio(
    problem: str,
    methods,
    split: str,
    count: int,
    offset: int,
    workers: int,
    build_method,
    *,
    max_tasks_per_child: int | None = None,
    warm=None,
    allow_test: bool = False,
) -> None:
    """Every ``method`` on the seeds of ``split``; rows append to ``runs/<method>__<split>.jsonl``.

    Finished seeds are skipped, so an interrupted run resumes where it stopped.
    ``warm(env, seeds, methods)`` runs once in the parent before the fork, for
    caches the workers should inherit.
    """
    global _BUILD_METHOD

    folder = OUT / problem / "runs"
    folder.mkdir(parents=True, exist_ok=True)
    seeds = seeds_for(split, count, offset, allow_test)
    tasks = []
    for method in methods:
        path = folder / f"{method}__{split}.jsonl"
        done = {row["seed"] for row in read_jsonl(path) if not row.get("crashed")}
        tasks.extend((problem, method, seed) for seed in seeds if seed not in done)
    print(f"{problem}: {len(tasks)} episodes to run on {workers} workers", flush=True)
    if not tasks:
        return
    _BUILD_METHOD = build_method
    _WORKER_ENV[problem] = make_env(problem)
    if warm is not None:
        warm(_WORKER_ENV[problem], seeds, methods)
    started = time.time()
    handles = {method: open(folder / f"{method}__{split}.jsonl", "a") for method in methods}
    try:
        if workers <= 1:
            iterator = map(_episode_job, tasks)
            pool = None
        else:
            pool = mp.get_context("fork").Pool(workers, maxtasksperchild=max_tasks_per_child)
            iterator = pool.imap_unordered(_episode_job, tasks, chunksize=1)
        for count_done, row in enumerate(iterator, start=1):
            handle = handles[row["method"]]
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            if count_done % 10 == 0 or count_done == len(tasks):
                print(f"  {count_done}/{len(tasks)} done, {time.time() - started:.0f}s; last {row['method']} "
                      f"seed {row['seed']}: success={row.get('success')} served={row.get('served')} "
                      f"hours={row.get('hours')}", flush=True)
        if pool is not None:
            pool.close()
            pool.join()
    finally:
        for handle in handles.values():
            handle.close()


# ---------------------------------------------------------------------------
# Merge into the gate's format
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


NO_BOUND = {"kind": "none", "value": 0.0, "proven": False}


def merge_split(
    problem: str,
    split: str,
    bound_entry,
    bound_metric: str,
    *,
    methods=None,
    count: int | None = None,
    allow_test: bool = False,
) -> dict:
    """Best outcome per seed over the methods' rows, with the bound; writes ``<split>.json``.

    ``bound_entry(row, best) -> {"kind", "value", "proven"} | None`` reads one
    row of ``runs/bound__<split>.jsonl`` next to the seed's best outcome (so a
    heuristic bound can be floored at the BKS). ``bound_metric`` is
    ``"hours"`` (a lower bound on operating hours) or ``"served"`` (an upper
    bound on services).
    """
    if bound_metric not in ("hours", "served"):
        raise ValueError("bound_metric must be 'hours' or 'served'")
    seeds = seeds_for(split, count or SPLIT_SEEDS.get(split, 128), 0, allow_test)
    methods_rows = load_method_rows(problem, split, methods)
    bounds = {int(row["seed"]): row for row in read_jsonl(OUT / problem / "runs" / f"bound__{split}.jsonl")}
    result, missing = {}, []
    for seed in seeds:
        entries = {name: compact(rows[seed], name) for name, rows in methods_rows.items() if seed in rows}
        if not entries:
            missing.append(seed)
            continue
        best_name = min(entries, key=lambda name: outcome_key(entries[name]))
        bound = bounds.get(seed)
        entry = bound_entry(bound, entries[best_name]) if bound is not None else None
        result[str(seed)] = {"best": dict(entries[best_name]), "methods": entries, "bound": entry or dict(NO_BOUND)}
    if missing:
        print(f"WARNING: {len(missing)} {split} seeds have no method rows: {missing[:5]}...", flush=True)
    (OUT / problem).mkdir(parents=True, exist_ok=True)
    (OUT / problem / f"{split}.json").write_text(json.dumps(result, indent=1))
    summary = summarize(problem, result, methods_rows, bound_metric)
    summary["split"] = split
    name = "summary.json" if split == "validation" else f"summary_{split}.json"
    (OUT / problem / name).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return result


def summarize(problem: str, result: dict, methods: dict, bound_metric: str = "hours") -> dict:
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
    with_bound = [seed for seed in seeds if result[str(seed)]["bound"]["kind"] != "none"]
    if bound_metric == "hours":
        bounded = [
            (result[str(seed)]["bound"]["value"], result[str(seed)]["best"]["hours"])
            for seed in with_bound if result[str(seed)]["best"]["success"]
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
                "proven_fraction": float(np.mean([result[str(s)]["bound"]["proven"] for s in with_bound])),
            }
    else:
        bounded = [(result[str(seed)]["bound"]["value"], result[str(seed)]["best"]["served"]) for seed in with_bound]
        if bounded:
            ratios = [served / bound for bound, served in bounded if bound > 0]
            out["bound"] = {
                "seeds": len(bounded),
                "mean_bound_served": float(np.mean([b for b, _ in bounded])),
                "mean_bks_served": float(np.mean([s for _, s in bounded])),
                "mean_ratio": float(np.mean(ratios)) if ratios else None,
                "min_ratio": float(np.min(ratios)) if ratios else None,
                # The bound is a heuristic (clairvoyant planner): where it fell
                # below the BKS it was raised to the BKS, and that is counted.
                "exceeded_seeds": int(sum(served > bound for bound, served in bounded)),
                "clairvoyant_below_bks_seeds": int(sum(
                    bool(result[str(s)]["bound"].get("raised_to_bks", False)) for s in with_bound
                )),
                "proven_fraction": float(np.mean([result[str(s)]["bound"]["proven"] for s in with_bound])),
            }
    # Paired comparison of each method with the BKS.
    for name, rows in methods.items():
        if not out["methods"].get(name):
            continue
        if bound_metric == "hours":
            pairs = [(rows[s]["hours"], result[str(s)]["best"]["hours"]) for s in seeds
                     if s in rows and rows[s]["success"] and result[str(s)]["best"]["success"]]
            if pairs:
                out["methods"][name]["mean_excess_vs_bks"] = float(np.mean([a / b - 1.0 for a, b in pairs]))
        else:
            pairs = [(rows[s]["served"], result[str(s)]["best"]["served"]) for s in seeds if s in rows]
            total = sum(b for _, b in pairs)
            if pairs and total > 0:
                out["methods"][name]["served_ratio_vs_bks"] = float(sum(a for a, _ in pairs) / total)
    return out


# ---------------------------------------------------------------------------
# Train demonstrations: the best non-failed episode per seed, replay-verified
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


def write_demonstrations(problem: str, count: int, offset: int, workers: int, methods=None) -> None:
    seeds = seeds_for("train", count, offset)
    rows_by_method = load_method_rows(problem, "train", methods)
    chosen = {}
    for seed in seeds:
        rows = [row for rows in rows_by_method.values() if (row := rows.get(seed)) is not None and "actions" in row]
        rows = [row for row in rows if not row["failed"]]
        if rows:
            chosen[seed] = min(rows, key=outcome_key)
    print(f"{problem}: {len(chosen)} non-failed train episodes of {len(seeds)} seeds", flush=True)
    jobs = [(problem, seed, row["actions"]) for seed, row in chosen.items()]
    if workers <= 1:
        replays = dict(map(_replay_job, jobs))
    else:
        with mp.get_context("fork").Pool(workers) as pool:
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
    (OUT / problem).mkdir(parents=True, exist_ok=True)
    (OUT / problem / "train_demonstrations.json").write_text(json.dumps(demos))
    successes = sum(d["success"] for d in demos)
    print(f"{problem}: wrote {len(demos)} replay-verified demonstrations ({successes} successful), "
          f"{len(mismatches)} replay mismatches {mismatches[:5]}", flush=True)
    (OUT / problem / "train_demonstrations_summary.json").write_text(json.dumps({
        "problem": problem, "split": "train", "offset": offset, "seeds": len(seeds),
        "demonstrations": len(demos), "successful": successes, "replay_mismatches": mismatches,
        "methods": {m: sum(d["policy"] == m for d in demos) for m in sorted({d["policy"] for d in demos})},
        "mean_hours": (sum(d["hours"] for d in demos) / len(demos)) if demos else None,
        "mean_served": (sum(d["served"] for d in demos) / len(demos)) if demos else None,
    }, indent=2))


# ---------------------------------------------------------------------------
# Source snapshot for SLURM jobs
# ---------------------------------------------------------------------------


def snapshot(name: str) -> Path:
    """Copy the code a job runs into ``OUT/_source/<name>`` (data is linked); make ``OUT/logs``."""
    target = OUT / "_source" / name
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
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    print(target)
    return target


def add_split_arguments(parser, *, default_count: int | None = None) -> None:
    """``--split``, ``--count``, ``--offset`` and ``--allow-test`` on a subcommand."""
    parser.add_argument("--split", default="validation", choices=("validation", "train", "test"))
    parser.add_argument("--count", type=int, default=default_count)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--allow-test", action="store_true", help="Open the test split (P5 only).")
