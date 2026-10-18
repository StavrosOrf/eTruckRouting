"""Measure clairvoyant solver scaling in isolated, sequential processes.

Unprofiled wall-time runs measure throughput and certificate gaps. Use --profile
with --time-limit 0 and a fixed --max-nodes in a separate run to locate CPU costs
without treating profiler overhead as normal solver performance. Instance setup,
reset, initial witness, solve, and independent replay are timed separately.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import platform
import resource
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _rss_mib():
    # Current resident memory, rather than a high-water mark inherited from imports.
    pages = int(Path("/proc/self/statm").read_text().split()[1])
    return pages * os.sysconf("SC_PAGE_SIZE") / 2**20


def _worker(case, output):
    import cProfile
    import pstats
    from unittest.mock import patch

    import torch

    from EVRoutingEnv.baselines.optimization import perfect_information as planner
    from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
    from EVRoutingEnv.utils.utils import load_config

    torch.set_num_threads(1)
    config = load_config(case["config"])
    config["environment"].update(
        num_stops=case["customers"],
        num_trucks=case["trucks"],
        allow_variable_num_stops=False,
    )
    config["environment"].pop("scenario_cache_dir", None)
    os.environ.pop("EVRP_SCENARIO_CACHE", None)
    if case["charge_levels"] is not None:
        config["charging"]["target_soc_levels"] = case["charge_levels"]
    row = {**case, "status": "ERROR"}
    env = None
    try:
        started = time.perf_counter()
        env = EventDrivenTruckEnv(config, verbose=False, enable_plotting=False)
        row["initialization_s"] = time.perf_counter() - started
        started = time.perf_counter()
        env.reset(seed=case["seed"])
        row["reset_s"] = time.perf_counter() - started
        row["generation_attempts"] = env.joint_instance.generation_attempts_used
        row["road_nodes"] = env.transport_graph.num_nodes
        row["chargers"] = len(env.charging_nodes)
        row["charge_choices"] = len(env.charge_action_values)
        row["action_space"] = int(env.action_space.n)
        row["episode_step_limit"] = env.max_episode_steps
        row["witness_steps"] = len(env.feasibility_witness)
        row["scenario_sha256"] = planner.canonical_json_sha256(env.scenario_descriptor)
        row["instance_sha256"] = planner.canonical_json_sha256(
            {
                key: env.scenario_descriptor[key]
                for key in ("customers", "depot_node", "initial_battery_kwh")
            }
        )
        baseline = _rss_mib()
        peak = [baseline]
        stopped = threading.Event()

        def sample():
            while not stopped.wait(0.025):
                peak[0] = max(peak[0], _rss_mib())

        # Python 3.13 can attribute sampler-thread waits to cProfile callers.
        # Keep the CPU profile single-threaded; normal runs measure peak RSS.
        sampler = (
            None if case["profile"] else threading.Thread(target=sample, daemon=True)
        )
        if sampler is not None:
            sampler.start()
        original_replay = planner._replay
        witness_time = []
        search_started = time.perf_counter()

        def timed_witness(*args, **kwargs):
            result = original_replay(*args, **kwargs)
            witness_time.append(time.perf_counter() - search_started)
            return result

        profiler = cProfile.Profile() if case["profile"] else None
        before = resource.getrusage(resource.RUSAGE_SELF)
        parameters = planner.PerfectInformationParameters(
            objective=case["objective"],
            time_limit_seconds=case["time_limit"] or None,
            max_nodes=case["max_nodes"],
        )
        try:
            with patch.object(planner, "_replay", timed_witness):
                if profiler:
                    profiler.enable()
                try:
                    solution = planner.solve_perfect_information_plan(env, parameters)
                finally:
                    if profiler:
                        profiler.disable()
            solve_seconds = time.perf_counter() - search_started
        finally:
            stopped.set()
            if sampler is not None:
                sampler.join()
        after = resource.getrusage(resource.RUSAGE_SELF)
        peak[0] = max(peak[0], _rss_mib())
        row.update(solution.as_dict())
        row.pop("actions")
        row.update(
            solve_s=solve_seconds,
            cpu_s=(after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime),
            initial_feasible_s=witness_time[0] if witness_time else None,
            rss_measurement="endpoints" if case["profile"] else "25ms_samples",
            baseline_rss_mib=baseline,
            solve_peak_rss_mib=peak[0],
            solve_rss_growth_mib=max(0.0, peak[0] - baseline),
            process_peak_rss_mib=after.ru_maxrss / 1024,
            nodes_per_s=solution.nodes_expanded / solve_seconds,
            plan_steps=len(solution.actions),
            relative_gap=(
                max(0.0, solution.objective_hours - solution.best_bound_hours)
                / solution.objective_hours
                if solution.objective_hours
                else None
            ),
        )
        if profiler:
            profile_path = output / f"{case['case_id']}.pstats"
            profiler.dump_stats(profile_path)
            stream = io.StringIO()
            stats = pstats.Stats(profiler, stream=stream).strip_dirs()
            stats.sort_stats("cumulative").print_stats(40)
            stats.sort_stats("tottime").print_stats(30)
            (output / f"{case['case_id']}.profile.txt").write_text(stream.getvalue())
            row["profile_functions"] = [
                {
                    "file": str(Path(file).relative_to(ROOT))
                    if str(file).startswith(str(ROOT))
                    else file,
                    "line": line,
                    "function": function,
                    "primitive_calls": values[0],
                    "calls": values[1],
                    "self_s": values[2],
                    "cumulative_s": values[3],
                }
                for (file, line, function), values in sorted(
                    stats.stats.items(), key=lambda pair: pair[1][3], reverse=True
                )
            ][:100]
        if solution.objective_hours is not None:
            started = time.perf_counter()
            replay = planner.replay_perfect_information_plan(env, solution)
            row["replay_s"] = time.perf_counter() - started
            row["replay_verified"] = replay["operational_metrics"]["success"]
            row["replay_sha256"] = replay["replay_sha256"]
            row["charging_sessions"] = replay["operational_metrics"][
                "charging_sessions"
            ]
            row["queue_hours"] = replay["operational_metrics"]["total_queue_time"]
        return row
    except Exception as exc:  # noqa: BLE001 - Record failed benchmark cases.
        row.update(
            status="ERROR",
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
        return row
    finally:
        if env is not None:
            env.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="EVRoutingEnv/config_files/config_joint.yaml"
    )
    parser.add_argument(
        "--customers", nargs="+", type=int, default=[1, 2, 3, 4, 6, 8, 12, 16]
    )
    parser.add_argument("--trucks", nargs="+", type=int, default=[2])
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 31, 59])
    parser.add_argument(
        "--time-limit", type=float, default=10.0, help="0 disables wall limit"
    )
    parser.add_argument("--max-nodes", type=int, default=100_000)
    parser.add_argument(
        "--objective", choices=["makespan", "total_time"], default="makespan"
    )
    parser.add_argument("--charge-levels", nargs="+", type=float)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--case-timeout", type=float, default=120.0)
    parser.add_argument(
        "--cases", type=Path, help="JSON list of explicit customer/truck/seed overrides"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.worker:
        case = json.loads(args.worker)
        row = _worker(case, args.output)
        (args.output / f"{case['case_id']}.json").write_text(
            json.dumps(row, indent=2, allow_nan=False) + "\n"
        )
        return
    if (args.output / "results.jsonl").exists():
        parser.error("output already has results; choose a fresh directory")
    base = {
        "config": args.config,
        "time_limit": args.time_limit,
        "max_nodes": args.max_nodes,
        "objective": args.objective,
        "charge_levels": args.charge_levels,
        "profile": args.profile,
    }
    overrides = (
        json.loads(args.cases.read_text())
        if args.cases
        else [
            {"customers": customers, "trucks": trucks, "seed": seed}
            for customers in args.customers
            for trucks in args.trucks
            for seed in args.seeds
        ]
    )
    cases = [dict(base, **override) for override in overrides]
    for index, case in enumerate(cases):
        case["case_id"] = (
            f"{index:03d}-c{case['customers']}-t{case['trucks']}-s{case['seed']}"
        )
    metadata = {
        "command": sys.argv,
        "cases": cases,
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "cpu_affinity_count": len(os.sched_getaffinity(0)),
        "method": "sequential isolated processes; unprofiled solver RSS sampled every 25 ms; CPU profiles use RSS endpoints only; scenario disk cache disabled",
        "thread_limits": dict.fromkeys(
            ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"), "1"
        ),
    }
    (args.output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    rows = []
    worker_env = {
        **os.environ,
        **metadata["thread_limits"],
        "MPLCONFIGDIR": "/tmp/evrp_matplotlib",
        "PYTHONHASHSEED": "0",
    }
    for case in cases:
        started = time.perf_counter()
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "scripts.analysis.profile_perfect_information",
                    "--worker",
                    json.dumps(case),
                    "--output",
                    str(args.output),
                ],
                check=False,
                cwd=ROOT,
                env=worker_env,
                text=True,
                capture_output=True,
                timeout=args.case_timeout,
            )
            path = args.output / f"{case['case_id']}.json"
            if result.returncode != 0 or not path.exists():
                row = {
                    **case,
                    "status": "PROCESS_ERROR",
                    "error": result.stderr[-4000:],
                }
            else:
                row = json.loads(path.read_text())
            if result.stdout or result.stderr:
                (args.output / f"{case['case_id']}.log").write_text(
                    result.stdout + result.stderr
                )
        except subprocess.TimeoutExpired:
            row = {
                **case,
                "status": "PROCESS_TIMEOUT",
                "error": f"case exceeded {args.case_timeout}s including initialization/reset/solve/replay",
            }
        row["end_to_end_s"] = time.perf_counter() - started
        rows.append(row)
        with (args.output / "results.jsonl").open("a") as file:
            file.write(json.dumps(row, allow_nan=False) + "\n")
        flat = [
            {
                key: value
                for key, value in entry.items()
                if not isinstance(value, (dict, list))
            }
            for entry in rows
        ]
        with (args.output / "results.csv").open("w") as file:
            columns = list(dict.fromkeys(key for entry in flat for key in entry))
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            writer.writerows(flat)
        print(
            json.dumps(
                {
                    key: row.get(key)
                    for key in (
                        "case_id",
                        "status",
                        "reset_s",
                        "solve_s",
                        "initial_feasible_s",
                        "nodes_expanded",
                        "nodes_per_s",
                        "relative_gap",
                        "solve_rss_growth_mib",
                        "error",
                    )
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
