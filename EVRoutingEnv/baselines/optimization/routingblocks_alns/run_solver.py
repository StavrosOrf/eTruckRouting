"""Command line entry point of the RoutingBlocks baseline (runs in the solver interpreter).

    python -m EVRoutingEnv.baselines.optimization.routingblocks_alns.run_solver \\
        --instance in.json --out out.json --time-limit 600 --workers 8

Reads an ``evrp-bp-instance/v1`` export and writes an ``evrp-heur-result/v1``
certificate whose per-route records mirror ``evrp-bp-result/v1`` (schedule,
witness steps), so the shared validators apply unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import fields
from pathlib import Path

import routingblocks

from .problem import Problem
from .parallel import solve_parallel
from .search import Search, SearchParameters

SCHEMA = "evrp-heur-result/v1"


def seed_sequences(problem: Problem) -> list[list[tuple]]:
    """Customer sequences (local ids) of the instance's warm-start routes."""
    index = {problem.env_node(v): v for v in problem.customers}
    warm = problem.instance.get("warm_start", {})
    sequences = []
    for route in list(warm.get("witness", [])) + list(warm.get("extra", [])):
        seq = tuple(index[int(node)] for node, _, charging in route if not charging and int(node) in index)
        if seq:
            sequences.append(seq)
    for seq in warm.get("sequences", []):
        ids = [int(x) for x in seq]
        if ids and all(x in index for x in ids):
            sequences.append(tuple(index[x] for x in ids))
    # The witness is one fleet plan: group it back into a single start.
    return [sequences] if sequences else []


def route_record(problem: Problem, sequence: tuple) -> dict:
    priced = problem.core.price_route(list(sequence))
    if priced is None:
        raise RuntimeError(f"route {sequence} is not exactly feasible")
    _, _, stops = priced
    simulated = problem.core.simulate(stops)
    if simulated is None:
        raise RuntimeError(f"route {sequence} failed re-simulation")
    cost, rows = simulated
    levels = problem.levels
    last = len(rows) - 1
    records, witness = [], []
    load = 0.0
    travel = 0.0
    previous = None
    for k, (node, level, arrive, arrive_soc, depart, depart_soc, charge_h) in enumerate(rows):
        env_id = problem.env_node(node)
        kind = "depot" if node == 0 else "customer" if node <= problem.n_customers else "charger"
        if kind == "customer":
            load += problem.demand[node]
        if previous is not None:
            travel += problem.hours[previous, node]
        records.append({
            "node": env_id,
            "kind": kind,
            "arrival_h": arrive,
            "arrival_kwh": arrive_soc,
            "level": levels[level] if level >= 0 else 0.0,
            "charge_h": charge_h,
            "departure_h": depart,
            "departure_kwh": depart_soc,
            "load_after": load,
        })
        if k > 0:
            witness.append([env_id, 0.0, False])
        if kind == "charger" and level >= 0:
            witness.append([env_id, levels[level], True])
        previous = node
    customers = [problem.nodes[v]["task_id"] for v in sequence]
    return {
        "feasible": True,
        "reason": "",
        "cost": cost,
        "travel_h": travel,
        "load": load,
        "customers": customers,
        "witness": witness,
        "stops": records,
    }


def solve(instance: dict, parameters: SearchParameters, workers: int = 1,
          objective: str | None = None) -> dict:
    started = time.time()
    problem = Problem(instance, objective=objective)
    seeds = seed_sequences(problem)
    if workers <= 1:
        search = Search(problem, parameters)
        result = search.run([list(s) for s in seeds])
        result["log"] = search.log[-200:]
        result["per_worker_iterations"] = [result["iterations"]]
        result["pool_columns"] = len(search.elite)
    else:
        result = solve_parallel(instance, parameters, workers, objective, seeds)
    records = [route_record(problem, seq) for seq in result["routes"]]
    objective_value = sum(r["travel_h"] for r in records) if problem.objective == "travel" else result["cost"]
    return {
        "schema": SCHEMA,
        "status": "HEURISTIC",
        "proven_optimal": False,
        "objective": problem.objective,
        "objective_hours": objective_value,
        "best_bound_hours": None,
        "gap": None,
        "wall_seconds": time.time() - started,
        "routes": records,
        "unused_trucks": problem.trucks - len(records),
        "search": {
            "workers": workers,
            "iterations": result["iterations"],
            "per_worker_iterations": result["per_worker_iterations"],
            "pricer_calls": result["pricer_calls"],
            "pricer_hits": result["pricer_hits"],
            "pool_columns": result["pool_columns"],
            "history": result["history"][-100:],
        },
        "assumptions": {
            "nominal": "mean leg times/energies, base service times, deterministic charging curve",
            "charger_ports": "relaxed (uncapacitated); verify and repair downstream",
            "pricing": "exact per route (discrete levels, margin, windows, horizon); detours that pass "
                       "two or more chargers before the first charge are not considered",
        },
        "parameters": parameters.as_dict(),
        "instance_meta": instance.get("meta", {}),
        "versions": {"python": sys.version.split()[0], "routingblocks": getattr(routingblocks, "__version__", "0.2.1")},
        "log": result["log"],
    }


def coerce(kind: type, text: str):
    if kind is bool:
        return text.strip().lower() in ("1", "true", "yes", "on")
    return kind(text)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--instance", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--objective", choices=("travel", "operating"))
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--param", nargs=2, action="append", default=[], metavar=("KEY", "VALUE"),
                        help="any other SearchParameters field")
    args = parser.parse_args(argv)
    values = {"time_limit_seconds": args.time_limit, "seed": args.seed, "verbose": args.verbose}
    types = {f.name: f.type for f in fields(SearchParameters)}
    for key, value in args.param:
        if key not in types:
            parser.error(f"unknown parameter {key}")
        values[key] = coerce(type(getattr(SearchParameters(), key)), value)
    certificate = solve(json.loads(args.instance.read_text()), SearchParameters(**values),
                        workers=args.workers, objective=args.objective)
    args.out.write_text(json.dumps(certificate, indent=1))
    print(f"objective {certificate['objective_hours']:.6f} h, {len(certificate['routes'])} routes, "
          f"{certificate['wall_seconds']:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
