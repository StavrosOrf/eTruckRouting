"""Nominal lower bound on fleet operating hours for the static long-haul problems.

The bound is the optimum (CP-SAT's certified best bound) of a relaxation of
the *nominal* problem (every leg at its table hours and kWh, unloading at its
base time, no queue waits):

* each pair of consecutive stops is joined by its waypoint closure: the
  cheapest chain of stations with every hop within ``capacity / margin`` kWh
  (and, with hours of service, within the 8 h driving-before-break limit), at
  cost ``hours + kWh / r_max`` where ``r_max`` is the fastest charging rate of
  the curve. Any executable route uses such hops, charges at least what it
  consumes beyond its initial ``capacity`` kWh (every route here consumes
  more: both depot legs exceed ``min_depot_energy_kwh``), and charges no
  faster than ``r_max``; so a route lasts at least
  ``L_R = sum closure cost + service - capacity / r_max``;
* with hours of service, a route's on-duty time W >= L_R and it needs a 10 h
  daily rest per started 14 h window after the first, so it lasts at least
  ``f(W) >= max(L_R, 24/14 L_R - 10)``; the bound is the larger of the two
  linear relaxations;
* payload is kept (demands floored, capacity rounded up), the horizon,
  discrete charge targets, station detours beyond the closure, waits and the
  1.2 energy margin at customers are dropped; costs are floored at 1e-3 h.

So the value is a proven lower bound for the nominal problem. It is not a
bound on realized hours, since realized legs can be up to 15 % faster.
"""

from __future__ import annotations

import math
import time

import numpy as np

from EVRoutingEnv.baselines.optimization.longhaul.network import (
    LonghaulNetwork,
    _closure,
    charge_curve,
    energy_margin,
)

_SCALE = 1000


def max_charge_rate(curve) -> float:
    """Fastest kWh per hour anywhere on the charging curve."""
    table = curve.tables[max(curve.levels)]
    grid = curve.grid
    best = 0.0
    for index in range(len(grid) - 1):
        dt = table[index] - table[index + 1]
        if dt > 1e-12:
            best = max(best, (grid[index + 1] - grid[index]) * curve.capacity / dt)
    return best


def _bound_network(env, rate: float) -> LonghaulNetwork:
    hos = getattr(env, "hos", None)
    position, energy, hours = env.transport_graph.cost_matrices()
    nodes = np.zeros(len(position), dtype=np.int64)
    for node, index in position.items():
        nodes[index] = int(node)
    network = LonghaulNetwork(
        position={int(k): int(v) for k, v in position.items()},
        nodes=nodes,
        hours=np.asarray(hours, dtype=float),
        energy=np.asarray(energy, dtype=float),
        chargers=np.asarray([position[int(n)] for n in env.charging_nodes], dtype=np.int64),
        capacity=float(env.config["truck"]["battery_capacity"]),
        margin=energy_margin(env),
        rate=rate,
        max_hop_hours=float(hos.regime.driving_before_break) if hos is not None else math.inf,
    )
    _closure(network)
    return network


_BOUND_NETWORKS: dict = {}


def nominal_lower_bound(env, *, time_limit_seconds: float = 120.0, workers: int = 1) -> dict:
    """CP-SAT bound of the nominal relaxation for ``env``'s current scenario."""
    from ortools.sat.python import cp_model

    started = time.perf_counter()
    curve = charge_curve(env)
    rate = max_charge_rate(curve)
    key = (str(env.config["network"].get("data_path")), round(rate, 9), getattr(env, "hos", None) is not None)
    network = _BOUND_NETWORKS.get(key)
    if network is None:
        network = _BOUND_NETWORKS[key] = _bound_network(env, rate)
    depot = network.position[int(env.joint_instance.depot_node)]
    tasks = list(env.joint_instance.tasks)
    points = [depot] + [network.position[int(task.node_id)] for task in tasks]
    n = len(points)
    cost = network.closure_cost[np.ix_(points, points)]
    if not np.isfinite(cost[~np.eye(n, dtype=bool)]).all():
        return {"value": None, "status": "UNREACHABLE", "proven": False}
    service = sum(float(task.base_service_time) for task in tasks)
    credit = float(network.capacity) / rate
    payload = float(env.trucks[0].payload_capacity)
    demands = [0] + [math.floor(float(task.demand) * _SCALE) for task in tasks]
    capacity = math.ceil(payload * _SCALE)
    trucks = len(env.trucks)
    hos = getattr(env, "hos", None) is not None
    variants = [("linear", 1.0, 0.0)]
    if hos:
        rest = float(env.hos.regime.daily_rest_hours)
        window = float(env.hos.regime.shift_window_hours)
        variants.append(("hos_windows", (window + rest) / window, rest))
    results = {}
    for name, factor, per_route in variants:
        model = cp_model.CpModel()
        arcs, literal = [], {}
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                x = model.NewBoolVar(f"x{i}_{j}")
                literal[(i, j)] = x
                arcs.append((i, j, x))
        model.AddMultipleCircuit(arcs)
        load = [model.NewIntVar(demands[i] if i else 0, capacity if i else 0, f"u{i}") for i in range(n)]
        for (i, j), x in literal.items():
            if i and j:
                model.Add(load[j] >= load[i] + demands[j]).OnlyEnforceIf(x)
        routes = sum(literal[(0, j)] for j in range(1, n))
        model.Add(routes <= trucks)
        model.Add(routes >= math.ceil(sum(demands) / capacity))
        # route cost = factor * (closure + service - credit) - per_route; all
        # constants are taken out so arc costs are floored integers.
        objective = []
        for (i, j), x in literal.items():
            value = factor * float(cost[i, j])
            if i == 0:
                value -= factor * credit + per_route
            objective.append(math.floor(value * _SCALE) * x)
        constant = factor * service
        model.Minimize(sum(objective))
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = float(time_limit_seconds)
        solver.parameters.num_workers = int(workers)
        status = solver.Solve(model)
        bound = solver.BestObjectiveBound() / _SCALE + constant
        incumbent = (
            solver.ObjectiveValue() / _SCALE + constant
            if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else None
        )
        results[name] = {
            "status": solver.StatusName(status),
            "bound": float(bound),
            "incumbent": None if incumbent is None else float(incumbent),
            "wall_seconds": solver.WallTime(),
        }
    best_name = max(results, key=lambda key: results[key]["bound"])
    best = results[best_name]
    return {
        "value": best["bound"],
        "status": best["status"],
        # CP-SAT's best bound is certified whatever the status.
        "proven": best["status"] in ("OPTIMAL", "FEASIBLE", "UNKNOWN"),
        "relaxation_solved": best["status"] == "OPTIMAL",
        "variant": best_name,
        "variants": results,
        "rate_kwh_per_hour": rate,
        "seconds": time.perf_counter() - started,
    }
