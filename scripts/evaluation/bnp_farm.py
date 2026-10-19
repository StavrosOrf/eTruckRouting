"""Work farm for a split branch-and-price search: recursive subtrees, any number of allocations.

A split run (``solve_branch_and_price --split-nodes K``) leaves its open nodes in
``split_<seed>.json``. The farm turns every open node into a queue item. A
worker searches one item as its subtree root for a time budget; if the subtree
is not closed by then it writes its own open nodes (with the column and cut
pool, so its children start warm) and each becomes a new item. Items live as
files in FARM/queue, FARM/running and FARM/done; a worker claims one by an
atomic rename, so ``work`` processes in several Slurm allocations can share one
farm. Every worker publishes and adopts incumbents through FARM/incumbents.

Bounds: the items in the queue and the running ones cover exactly the
unexplored part of the tree, so the least of their bounds (a running item's
own subtree bound once its log reports one) is a lower bound, and the best
incumbent any worker found is the upper bound. When the queue and the running
set are empty every subtree is closed and the incumbent is optimal.

    python -m scripts.evaluation.bnp_farm init --farm F --instance OUT/instances/seed_1.json \\
        --split OUT/split_1.json [--skip-closed OUT/subtrees_1] [--incumbents OUT/incumbents_1]
    python -m scripts.evaluation.bnp_farm work --farm F --workers 16 --worker-time 7200 -- --subset_row_limit 256
    python -m scripts.evaluation.bnp_farm status --farm F
    python -m scripts.evaluation.bnp_farm finalize --farm F --out OUT --seed 1 --trucks 50 --customers 100
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from EVRoutingEnv.baselines.optimization.branch_and_price.julia_bridge import RUNNER

GAP_TOL = 1e-6
TREE_BOUND = re.compile(r"tree: .* bound ([-0-9.eE+]+),")
# Children re-solve their root from the parent's pool: no root primal work
# (the farm shares incumbents), one enumeration attempt budget as usual.
WORKER_DEFAULTS = ["--lp_solver", "highs", "--threads", "1", "--verbose", "true", "--lns_time", "0",
                   "--dive_time", "0", "--root_mip_time", "20", "--mip_time", "20"]


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _write(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(data, indent=1))
    tmp.replace(path)


def _dirs(farm: Path) -> dict[str, Path]:
    return {name: farm / name for name in ("queue", "running", "done", "work", "incumbents")}


def _enqueue(farm: Path, split: Path, prefix: str, skip: set[int] = frozenset()) -> int:
    nodes = _read(split)["nodes"]
    count = 0
    for index, node in enumerate(nodes):
        if index in skip:
            continue
        item = {"id": f"{prefix}{index}", "split": str(split.resolve()), "index": index,
                "bound": node["bound"], "label": node["label"], "depth": prefix.count(".")}
        _write(farm / "queue" / f"{item['id']}.json", item)
        count += 1
    return count


def cmd_init(args) -> int:
    farm = args.farm
    for path in _dirs(farm).values():
        path.mkdir(parents=True, exist_ok=True)
    skip = set()
    if args.skip_closed:
        for result in Path(args.skip_closed).glob("sub_*.result.json"):
            certificate = _read(result)
            if certificate["status"] in ("OPTIMAL", "INFEASIBLE"):
                skip.add(int(result.name.split("_")[1].split(".")[0]))
    if args.incumbents:
        for path in Path(args.incumbents).glob("incumbent_*.json"):
            shutil.copy(path, farm / "incumbents" / path.name)
    count = _enqueue(farm, Path(args.split), "s", skip)
    _write(farm / "farm.json", {"instance": str(Path(args.instance).resolve()), "split": str(Path(args.split).resolve()),
                                "closed_before": sorted(skip), "created": time.time()})
    print(f"{count} items queued ({len(skip)} subtrees of the split already closed)")
    return 0


def upper_bound(farm: Path) -> tuple[float, str | None]:
    best, where = float("inf"), None
    for path in (farm / "incumbents").glob("incumbent_*.json"):
        try:
            value = _read(path).get("objective_hours")
        except (OSError, json.JSONDecodeError):
            continue
        if value is not None and value < best:
            best, where = value, path.name
    return best, where


def live_bound(item: dict, log: Path) -> float:
    bound = item["bound"]
    if log.exists():
        for match in TREE_BOUND.finditer(log.read_text(errors="replace")):
            bound = max(bound, float(match.group(1)))
    return bound


def summary(farm: Path) -> dict:
    d = _dirs(farm)
    upper, where = upper_bound(farm)
    queued = [_read(p) for p in d["queue"].glob("*.json")]
    running = [_read(p) for p in d["running"].glob("*.json")]
    done = [_read(p) for p in d["done"].glob("*.json")]
    bounds = [i["bound"] for i in queued] + [live_bound(i, d["work"] / i["id"] / "log") for i in running]
    lower = min(bounds, default=float("inf"))
    optimal = not queued and not running and done and all(i["outcome"] == "closed" for i in done)
    if optimal or lower >= upper - GAP_TOL * max(1.0, abs(upper)):
        lower = min(lower, upper)
    return {"upper": upper, "upper_source": where, "lower": lower,
            "gap": (upper - lower) / upper if upper < float("inf") and lower > -float("inf") else None,
            "queued": len(queued), "running": len(running), "done": len(done),
            "closed": sum(i["outcome"] == "closed" for i in done), "split": sum(i["outcome"] == "split" for i in done),
            "optimal": bool(optimal)}


def cmd_status(args) -> int:
    info = summary(args.farm)
    print(json.dumps(info, indent=1))
    return 0


def _bound(path: Path) -> float:
    try:
        return _read(path)["bound"]
    except (OSError, json.JSONDecodeError, KeyError):
        return float("inf")


def _claim(farm: Path) -> dict | None:
    """Move the queued item with the least bound to running (atomic rename), or None."""
    d = _dirs(farm)
    for path in sorted(d["queue"].glob("*.json"), key=_bound):
        target = d["running"] / path.name
        try:
            path.rename(target)
        except FileNotFoundError:
            continue                       # another farm process took it
        item = _read(target)
        item.update(host=socket.gethostname(), pid=os.getpid(), started=time.time())
        _write(target, item)
        return item
    return None


def _finish(farm: Path, item: dict, returncode: int) -> str:
    """Record a worker's outcome: closed, split (children queued) or retry (back to the queue)."""
    d = _dirs(farm)
    work = d["work"] / item["id"]
    result, split = work / "result.json", work / "split.json"
    outcome, detail = "retry", {}
    if result.exists():
        certificate = _read(result)
        detail = {key: certificate.get(key) for key in ("status", "objective_hours", "best_bound_hours",
                                                        "nodes_explored", "nodes_open", "wall_seconds")}
        upper, _ = upper_bound(farm)
        bound = certificate.get("best_bound_hours")
        if certificate["status"] in ("OPTIMAL", "INFEASIBLE"):
            outcome = "closed"
        elif split.exists():
            outcome = "split"
            detail["children"] = _enqueue(farm, split, f"{item['id']}.")
        elif bound is not None and bound >= upper - GAP_TOL * max(1.0, abs(upper)):
            outcome = "closed"             # nothing below the incumbent was left open
    item.update(outcome=outcome, returncode=returncode, finished=time.time(), **detail)
    if outcome == "retry":
        item.pop("host", None)
        _write(d["queue"] / f"{item['id']}.json", item)
    else:
        _write(d["done"] / f"{item['id']}.json", item)
    (d["running"] / f"{item['id']}.json").unlink(missing_ok=True)
    return outcome


def _job_end() -> float:
    """End of the current Slurm allocation (epoch seconds), or +inf outside Slurm."""
    job = os.environ.get("SLURM_JOB_ID")
    if not job:
        return float("inf")
    out = subprocess.run(["squeue", "-h", "-j", job, "-o", "%e"], capture_output=True, text=True).stdout.strip()
    try:
        return time.mktime(time.strptime(out, "%Y-%m-%dT%H:%M:%S"))
    except ValueError:
        return float("inf")


def cmd_work(args) -> int:
    farm = args.farm
    meta = _read(farm / "farm.json")
    end = min(_job_end(), time.time() + args.max_seconds)
    workers: dict[str, tuple[subprocess.Popen, dict]] = {}
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    extra = [a for a in args.solver_args if a != "--"]
    while not stopping:
        for key, (proc, item) in list(workers.items()):
            if proc.poll() is not None:
                outcome = _finish(farm, item, proc.returncode)
                print(f"[{time.strftime('%H:%M:%S')}] {key}: {outcome}", flush=True)
                del workers[key]
        left = end - time.time() - args.margin
        if summary(farm)["optimal"]:
            break
        while len(workers) < args.workers and left > args.min_worker_time:
            item = _claim(farm)
            if item is None:
                break
            work = farm / "work" / item["id"]
            work.mkdir(parents=True, exist_ok=True)
            for stale in ("result.json", "split.json"):
                (work / stale).unlink(missing_ok=True)
            budget = min(args.worker_time, left)
            command = [str(args.runner), meta["instance"], str(work / "result.json"), *WORKER_DEFAULTS,
                       "--time_limit", str(int(budget)), "--subtree_file", item["split"],
                       "--subtree_index", str(item["index"]), "--incumbent_dir", str(farm / "incumbents"),
                       "--incumbent_tag", item["id"], "--split_nodes", "1000000000",
                       "--split_file", str(work / "split.json"), *extra]
            with open(work / "log", "a") as log:
                proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                        env={**os.environ, "EVRP_BP_THREADS": "1"})
            workers[item["id"]] = (proc, item)
            print(f"[{time.strftime('%H:%M:%S')}] {item['id']} (bound {item['bound']:.6f}) for {budget:.0f}s",
                  flush=True)
        if not workers:
            if left <= args.min_worker_time:
                break
            # nothing to claim: done once nobody else is running (their splits would queue more)
            if not any((farm / "queue").glob("*.json")) and not any((farm / "running").glob("*.json")):
                break
        time.sleep(args.poll)
    # allocation ending or stopped: hand unfinished items back
    for key, (proc, item) in workers.items():
        if proc.poll() is None:
            proc.terminate()
    deadline = time.time() + 60
    for key, (proc, item) in workers.items():
        try:
            proc.wait(timeout=max(1.0, deadline - time.time()))
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        _finish(farm, item, proc.returncode)
    print(json.dumps(summary(farm)), flush=True)
    return 0


def cmd_finalize(args) -> int:
    """Certificate of the farm's search in OUT/seed_<seed>.json (replayed on the env)."""
    from scripts.evaluation.solve_branch_and_price import attach_validation, build_env

    farm = args.farm
    info = summary(farm)
    candidates = list((farm / "work").glob("*/result.json"))
    if args.extra_results:
        candidates += list(Path(args.extra_results).glob("*.result.json"))
    best = min((_read(p) for p in candidates), key=lambda c: c["objective_hours"] or float("inf"))
    certificate = dict(best)
    upper = certificate["objective_hours"]
    optimal = info["optimal"] and upper is not None and upper <= info["upper"] + GAP_TOL * max(1.0, abs(upper))
    lower = upper if optimal else min(info["lower"], upper)
    certificate.update({
        "status": "OPTIMAL" if optimal else "GAP",
        "proven_optimal": optimal,
        "best_bound_hours": lower,
        "gap": max(0.0, (upper - lower) / upper),
        "farm": {**info, "directory": str(farm)},
    })
    ok, reason = attach_validation(build_env(args), certificate, args.seed, args.config)
    out = args.out / f"seed_{args.seed}.json"
    if out.exists() and not args.overwrite:
        out = args.out / f"seed_{args.seed}.farm.json"
    out.write_text(json.dumps(certificate, indent=1))
    print(f"seed {args.seed}: {certificate['status']} objective {upper} bound {lower} gap {certificate['gap']} "
          f"replay {'OK' if ok else 'FAILED: ' + str(reason)} -> {out}")
    return 0 if ok else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("init")
    p.add_argument("--farm", type=Path, required=True)
    p.add_argument("--instance", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--skip-closed", help="subtrees_<seed> directory: skip split nodes whose subtree run closed")
    p.add_argument("--incumbents", help="incumbents_<seed> directory to seed the farm's incumbents from")
    p = sub.add_parser("work")
    p.add_argument("--farm", type=Path, required=True)
    p.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    p.add_argument("--worker-time", type=float, default=7200.0)
    p.add_argument("--min-worker-time", type=float, default=1800.0, help="launch nothing with less time left")
    p.add_argument("--margin", type=float, default=300.0, help="seconds kept free before the allocation ends")
    p.add_argument("--max-seconds", type=float, default=float("inf"))
    p.add_argument("--poll", type=float, default=15.0)
    p.add_argument("--runner", type=Path, default=Path(os.environ.get("EVRP_BP_RUNNER", RUNNER)))
    p.add_argument("solver_args", nargs=argparse.REMAINDER, help="after --: extra solver arguments")
    p = sub.add_parser("status")
    p.add_argument("--farm", type=Path, required=True)
    p = sub.add_parser("finalize")
    p.add_argument("--farm", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--config", default="EVRoutingEnv/config_files/config_joint.yaml")
    p.add_argument("--trucks", type=int)
    p.add_argument("--customers", type=int)
    p.add_argument("--extra-results", help="directory of earlier subtree results (e.g. OUT/subtrees_<seed>)")
    p.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    return {"init": cmd_init, "work": cmd_work, "status": cmd_status, "finalize": cmd_finalize}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
