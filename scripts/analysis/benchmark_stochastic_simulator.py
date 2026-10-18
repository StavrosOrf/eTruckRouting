"""Measure fully stochastic joint-fleet simulation cost and outcome separately.

Run from the repository root with PYTHONPATH=. and one CPU math thread.
Each size/horizon is isolated in a child process with explicit wall/RSS budgets.
A budget stop is an experiment cutoff, never a simulator feasibility failure.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import multiprocessing as mp
import os
import platform
import queue
import resource
import subprocess
import time
import traceback
from copy import deepcopy
from pathlib import Path

import numpy as np
import psutil

from EVRoutingEnv.baselines.heuristics.canonical_baselines import (
    GreedyHeuristicPolicy,
    HeuristicParameters,
)
from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.utils.utils import load_config


def invariants(env):
    assert all(
        -1e-8 <= t.current_battery <= t.battery_capacity + 1e-8 for t in env.trucks
    )
    for truck in env.trucks:
        expected = (
            env.truck_initial_batteries[truck.truck_id]
            + truck.total_energy_charged
            - truck.total_energy_consumed
        )
        assert abs(expected - truck.current_battery) <= 1e-6
    owners = [task_id for truck in env.trucks for task_id in truck.served_task_ids]
    assert (
        len(owners)
        == len(set(owners))
        == sum(t.is_served for t in env.task_registry.tasks())
    )
    occupants = []
    for node in env.charging_nodes:
        busy = env.charging_station.charger_occupancy[node]
        occupants.extend(busy)
        assert len(busy) <= env.charging_station.charger_capacity[node]
        waiting = [
            entry["truck_id"] for entry in env.charging_station.charger_waitlist[node]
        ]
        assert len(waiting) == len(set(waiting))
    assert len(occupants) == len(set(occupants))


def worker(
    case,
    repeats,
    step_mode,
    policy_parameters,
    battery,
    episode_seconds,
    stop_on_failure,
    output,
):
    env = None
    try:
        trucks, customers, horizon = case
        config = deepcopy(load_config("EVRoutingEnv/config_files/config_joint.yaml"))
        config["environment"].update(
            num_trucks=trucks,
            num_stops=customers,
            allow_variable_num_stops=False,
            max_time=horizon,
            max_episode_steps=(
                200
                if step_mode == "default"
                else 100_000
                if step_mode == "horizon"
                else max(200, 16 * customers + 8 * trucks)
            ),
        )
        if step_mode == "configured":
            config["environment"].pop("max_episode_steps", None)
        config["truck"]["battery_capacity"] = battery
        assert (
            config["traffic"]["enable_traffic"]
            and config["traffic"]["enable_energy_uncertainty"]
        )
        assert config["delivery"]["enable_stochastic_unloading"]
        start = time.perf_counter()
        env = EventDrivenTruckEnv(config, verbose=False, enable_plotting=False)
        construct = time.perf_counter() - start
        if policy_parameters.get("policy") == "energy_path":
            from EVRoutingEnv.baselines.heuristics.energy_path import EnergyPathHeuristicPolicy

            policy = EnergyPathHeuristicPolicy()
        else:
            policy = GreedyHeuristicPolicy(HeuristicParameters(**policy_parameters))
        for repeat in range(repeats):
            row = {
                "trucks": trucks,
                "customers": customers,
                "battery_kwh": battery,
                "horizon_hours": horizon,
                "step_mode": step_mode,
                "max_episode_steps": env.max_episode_steps,
                "seed": 1_000_000_000 + repeat,
                "constructor_seconds": construct,
            }
            started = time.perf_counter()
            observation, info = env.reset(seed=row["seed"])
            row["reset_seconds"] = time.perf_counter() - started
            env.truck_initial_batteries = {
                t.truck_id: t.current_battery for t in env.trucks
            }
            output.put(
                row
                | {
                    "status": "initialized",
                    "observation_bytes": observation.nbytes,
                    "observation_floats": observation.size,
                }
            )
            step_times, policy_times, checks = [], [], 0.0
            last_progress = time.perf_counter()
            terminated = truncated = False
            while not (terminated or truncated):
                if time.perf_counter() - started > episode_seconds:
                    output.put(
                        row
                        | {
                            "status": "episode_wall_budget",
                            "episode_seconds": time.perf_counter() - started,
                            "steps": len(step_times),
                            "simulated_hours": env.global_clock,
                            "served": info["operational_metrics"]["customers_served"],
                        }
                    )
                    break
                clock = env.global_clock
                start = time.perf_counter()
                action = policy(env, observation, info)
                policy_times.append(time.perf_counter() - start)
                start = time.perf_counter()
                observation, _, terminated, truncated, info = env.step(action)
                step_times.append(time.perf_counter() - start)
                start = time.perf_counter()
                assert env.global_clock >= clock
                assert not (info["successful"] and truncated)
                invariants(env)
                checks += time.perf_counter() - start
                if time.perf_counter() - last_progress > 10:
                    output.put(
                        row
                        | {
                            "status": "progress",
                            "steps": len(step_times),
                            "episode_seconds": time.perf_counter() - started,
                            "simulated_hours": env.global_clock,
                            "simulator_seconds": sum(step_times),
                            "policy_seconds": sum(policy_times),
                            "step_median_ms": float(np.median(step_times)) * 1000,
                            "step_p95_ms": float(np.percentile(step_times, 95)) * 1000,
                            "completed_fraction": info["operational_metrics"][
                                "completed_fraction"
                            ],
                            "peak_rss_mib": resource.getrusage(
                                resource.RUSAGE_SELF
                            ).ru_maxrss
                            / 1024,
                        }
                    )
                    last_progress = time.perf_counter()
            if not (terminated or truncated):
                if stop_on_failure:
                    break
                continue
            metrics = info["operational_metrics"]
            row.update(
                status="completed",
                success=bool(info["successful"]),
                termination_reason=info["termination_reason"],
                terminated=terminated,
                truncated=truncated,
                simulated_hours=env.global_clock,
                completed_fraction=metrics["completed_fraction"],
                served=metrics["customers_served"],
                vehicles_used=metrics["vehicles_used"],
                queue_hours=metrics["total_queue_time"],
                travel_hours=metrics["total_travel_time"],
                steps=len(step_times),
                simulator_seconds=sum(step_times),
                policy_seconds=sum(policy_times),
                step_median_ms=float(np.median(step_times)) * 1000 if step_times else 0,
                step_p95_ms=float(np.percentile(step_times, 95)) * 1000
                if step_times
                else 0,
                policy_median_ms=float(np.median(policy_times)) * 1000
                if policy_times
                else 0,
                episode_seconds=time.perf_counter() - started,
                checks_seconds=checks,
                observation_floats=env.observation_space.shape[0],
                observation_bytes=observation.nbytes,
                peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                traffic_draws=len(env.traffic_simulator._uncertainty_cache),
                service_draws=len(env.delivery_simulator._uncertainty_cache),
            )
            output.put(row)
            if stop_on_failure and not row["success"]:
                break
    except Exception as error:  # noqa: BLE001 -- Persist any experiment failure.
        output.put(
            {
                "trucks": case[0],
                "customers": case[1],
                "horizon_hours": case[2],
                "step_mode": step_mode,
                "status": "error",
                "error": repr(error),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        if env is not None:
            env.close()


def network_limits():
    from EVRoutingEnv.models.core.transportation_graph import TransportationGraph
    from EVRoutingEnv.utils.utils import get_graph

    transport = TransportationGraph(
        get_graph(load_config("EVRoutingEnv/config_files/config_joint.yaml"))
    )
    nodes = sorted(transport.get_all_nodes())
    index = {node: i for i, node in enumerate(nodes)}
    stations = transport.get_charging_nodes()
    eligible = [
        node
        for node in nodes
        if node not in stations
        and transport.graph.in_degree(node) > 0
        and transport.graph.out_degree(node) > 0
    ]
    ordinary = [index[node] for node in eligible]
    station_rows = [index[node] for node in stations]
    energy = np.asarray(
        [[transport.get_path_energy(a, b) for b in nodes] for a in nodes],
        dtype=np.float64,
    )
    rows = []
    for budget in (100.0, 200.0, 300.0, 400.0 / 1.2, 400.0, 500.0, 1000.0, 10000.0):
        outbound = (energy[np.ix_(ordinary, ordinary)] <= budget) | (
            (energy[np.ix_(ordinary, station_rows)] <= budget)
            @ (energy[np.ix_(station_rows, ordinary)] <= budget)
        )
        round_trip = outbound & outbound.T
        np.fill_diagonal(round_trip, False)
        counts = round_trip.sum(axis=1)
        rows.append(
            {
                "nominal_leg_budget_kwh": budget,
                "max_customers": int(counts.max()),
                "min_customers": int(counts.min()),
                "median_customers": float(np.median(counts)),
                "best_depot": eligible[int(np.argmax(counts))],
            }
        )
    return {
        "nodes": len(nodes),
        "chargers": len(stations),
        "eligible": len(eligible),
        "unique_customer_ceiling": len(eligible) - 1,
        "reachability_assumption": "Nominal direct or one-charger reachability in each direction; not a complete-route feasibility certificate.",
        "step_cap_200_payload20_min_demand1_customer_bound": 190,
        "battery_reachability": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default="1x4,2x8,4x16,8x32,16x64,32x128,58x232")
    parser.add_argument("--horizons", default="200")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--battery-kwh", type=float, default=400.0)
    parser.add_argument(
        "--policy-parameters",
        default="{}",
        help="JSON overrides for the greedy heuristic; saved in metadata.",
    )
    parser.add_argument("--case-seconds", type=float, default=90)
    parser.add_argument("--episode-seconds", type=float, default=float("inf"))
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Skip remaining repeats after a failed/censored seed; it cannot pass the all-seeds criterion.",
    )
    parser.add_argument("--rss-mib", type=float, default=2048)
    parser.add_argument(
        "--step-mode",
        choices=["scaled", "default", "horizon", "configured"],
        default="scaled",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--network-limits", action="store_true")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.network_limits:
        args.output.write_text(json.dumps(network_limits(), indent=2) + "\n")
        return 0
    metadata = {
        "host": platform.node(),
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "python": platform.python_version(),
        "logical_cpus": os.cpu_count(),
        "math_threads": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_diff_sha256": hashlib.sha256(
            subprocess.check_output(["git", "diff", "--", "EVRoutingEnv"])
        ).hexdigest(),
        "source_files_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path("EVRoutingEnv").rglob("*.py"))
        },
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "gymnasium", "torch", "psutil")
        },
        "policy": (
            {"name": "EnergyPathHeuristicPolicy"}
            if json.loads(args.policy_parameters).get("policy") == "energy_path"
            else {
                "name": "GreedyHeuristicPolicy",
                "parameters": HeuristicParameters(
                    **json.loads(args.policy_parameters)
                ).as_dict(),
            }
        ),
        "stochastic": "travel + energy + unloading enabled",
        "arguments": vars(args) | {"output": str(args.output)},
    }
    args.output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    context = mp.get_context("fork")
    with args.output.open("w") as file:
        for size in args.cases.split(","):
            trucks, customers = map(int, size.lower().split("x"))
            for horizon in map(float, args.horizons.split(",")):
                case = (trucks, customers, horizon)
                messages = context.Queue()
                process = context.Process(
                    target=worker,
                    args=(
                        case,
                        args.repeats,
                        args.step_mode,
                        json.loads(args.policy_parameters),
                        args.battery_kwh,
                        args.episode_seconds,
                        args.stop_on_failure,
                        messages,
                    ),
                )
                process.start()
                begin = time.perf_counter()
                received = 0
                message_count = 0
                cutoff = None
                while process.is_alive() or not messages.empty():
                    if process.is_alive():
                        elapsed = time.perf_counter() - begin
                        try:
                            rss = (
                                psutil.Process(process.pid).memory_info().rss / 1024**2
                            )
                        except psutil.NoSuchProcess:
                            rss = 0
                        if elapsed > args.case_seconds:
                            cutoff = "wall_budget"
                        elif rss > args.rss_mib:
                            cutoff = "rss_budget"
                        if cutoff:
                            process.terminate()
                            process.join(timeout=5)
                            row = {
                                "trucks": trucks,
                                "customers": customers,
                                "horizon_hours": horizon,
                                "step_mode": args.step_mode,
                                "status": cutoff,
                                "case_elapsed_seconds": elapsed,
                                "rss_mib": rss,
                                "completed_repeats": received,
                            }
                            file.write(json.dumps(row) + "\n")
                            file.flush()
                            print(json.dumps(row), flush=True)
                            break
                    try:
                        row = messages.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    file.write(json.dumps(row) + "\n")
                    file.flush()
                    message_count += 1
                    received += int(row["status"] == "completed")
                    print(json.dumps(row), flush=True)
                process.join(timeout=5)
                if message_count == 0 and cutoff is None:
                    row = {
                        "trucks": trucks,
                        "customers": customers,
                        "horizon_hours": horizon,
                        "step_mode": args.step_mode,
                        "status": "worker_exit",
                        "exitcode": process.exitcode,
                    }
                    file.write(json.dumps(row) + "\n")
                    file.flush()
                    print(json.dumps(row), flush=True)
                messages.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
