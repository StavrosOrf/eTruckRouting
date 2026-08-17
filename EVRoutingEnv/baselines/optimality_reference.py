"""Proven-optimal reference for the deterministic relaxation of the eTFRP.

"Optimality" needs a denominator.  This module supplies one by solving each
instance's *nominal* problem with CP-SAT over a position-indexed model that
carries battery state, charging decisions, payload, and the mandatory depot
return.

Two things it establishes:

1. **Feasibility.** If this model is infeasible, no policy can succeed on that
   instance, because the model relaxes what the simulator demands.
2. **A bound.** Its optimal makespan lower-bounds the makespan any policy can
   achieve, so ``optimal / achieved`` is a valid optimality ratio.

The relaxations that make it a bound, all stated explicitly:

* travel, energy, and service take nominal values, while the simulator draws
  them stochastically (energy up to 1.20x nominal);
* charging is linear at rated power, while the simulator applies a
  constant-voltage taper that only ever makes charging slower;
* charger queues are ignored, so waiting is free;
* the fleet is perfectly coordinated with no decision-step limit.

Each relaxation can only reduce the objective, so the value is a genuine lower
bound on achievable makespan and its feasibility an upper bound on achievable
success.

Units are centi-units throughout: 0.01 h of time and 0.01 kWh of energy, which
keeps every quantity integral for CP-SAT.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from EVRoutingEnv.models.core.customer import TaskStatus


_SCALE = 100
_UNREACHABLE = 10**7


@dataclass(frozen=True)
class ReferenceSolution:
    """Proven-optimal nominal plan, or evidence that none exists."""

    status: str
    feasible: bool
    optimal_makespan: float | None
    best_bound: float | None
    routes: dict[int, list[int]]
    wall_seconds: float
    positions: int

    @property
    def proven_optimal(self) -> bool:
        return self.status == "OPTIMAL"

    @property
    def proven_infeasible(self) -> bool:
        return self.status == "INFEASIBLE"

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "feasible": self.feasible,
            "optimal_makespan": self.optimal_makespan,
            "best_bound": self.best_bound,
            "routes": {str(key): value for key, value in self.routes.items()},
            "wall_seconds": self.wall_seconds,
            "positions": self.positions,
            "proven_optimal": self.proven_optimal,
            "proven_infeasible": self.proven_infeasible,
        }


def _build_tables(env, nodes: list[int], tasks: list, energy_multiplier: float):
    graph = env.transport_graph
    size = len(nodes)

    def arc(source: int, target: int, getter) -> int:
        if source == target:
            return 0
        try:
            value = float(getter(nodes[source], nodes[target]))
        except (KeyError, TypeError, ValueError):
            return _UNREACHABLE
        if not math.isfinite(value) or value < 0.0:
            return _UNREACHABLE
        return round(value * _SCALE)

    energy = [
        [
            min(
                _UNREACHABLE,
                round(arc(i, j, graph.get_path_energy) * energy_multiplier),
            )
            for j in range(size)
        ]
        for i in range(size)
    ]
    travel = [
        [arc(i, j, graph.get_time_distance) for j in range(size)] for i in range(size)
    ]
    service = [0] * size
    demand = [0] * size
    for offset, task in enumerate(tasks, start=1):
        service[offset] = round(float(task.base_service_time) * _SCALE)
        demand[offset] = round(float(task.demand) * _SCALE)
    return energy, travel, service, demand


def _nearest_chargers(env, depot: int, tasks: list, limit: int) -> list[int]:
    """Keep the stations most useful to this instance's geography.

    A route uses a handful of stations, but carrying all 25 makes the model an
    order of magnitude larger and it then times out without proving anything.
    Restricting the set is a *restriction*, not a relaxation: any plan found is
    still valid for the full problem, so feasible verdicts and objective values
    remain sound. Only infeasibility loses its meaning, which is why callers
    must not read INFEASIBLE as proof when a limit is applied.
    """
    graph = env.transport_graph
    anchors = [depot] + [int(task.node_id) for task in tasks]

    def cost(charger: int) -> float:
        total = 0.0
        for anchor in anchors:
            try:
                value = float(graph.get_path_energy(int(charger), int(anchor)))
            except (KeyError, TypeError, ValueError):
                value = math.inf
            total += min(value, 1e6) if math.isfinite(value) else 1e6
        return total

    ranked = sorted((int(node) for node in env.charging_nodes), key=cost)
    return sorted(ranked[:limit])


def solve_reference(
    env,
    *,
    positions: int | None = None,
    time_limit_seconds: float = 120.0,
    workers: int = 4,
    energy_multiplier: float = 1.0,
    max_chargers: int | None = None,
) -> ReferenceSolution:
    """Solve one instance's nominal problem toward proven optimality."""
    from ortools.sat.python import cp_model

    depot = int(env.joint_instance.depot_node)
    tasks = [
        task
        for task in env.task_registry.tasks()
        if task.status in (TaskStatus.UNASSIGNED, TaskStatus.CLAIMED)
    ]
    trucks = [t for t in env.trucks if not t.is_complete and not t.failed]
    if not tasks or not trucks:
        return ReferenceSolution("EMPTY", True, 0.0, 0.0, {}, 0.0, 0)

    chargers = (
        _nearest_chargers(env, depot, tasks, max_chargers)
        if max_chargers
        else sorted({int(node) for node in env.charging_nodes})
    )
    nodes = [depot] + [int(task.node_id) for task in tasks] + chargers
    index_of = {node: index for index, node in enumerate(nodes)}
    size = len(nodes)
    customer_indices = list(range(1, 1 + len(tasks)))
    charger_flag = [
        1 if index >= 1 + len(tasks) else 0 for index in range(size)
    ]
    power = [0] * size
    for index in range(1 + len(tasks), size):
        power[index] = round(
            float(env.charging_station.charger_power_kw.get(nodes[index], 0.0))
        )
    max_power = max(power) or 1

    energy, travel, service, demand = _build_tables(env, nodes, tasks, energy_multiplier)
    flat_energy = [energy[i][j] for i in range(size) for j in range(size)]
    flat_travel = [travel[i][j] for i in range(size) for j in range(size)]

    # Enough slots for every customer, a charging stop between each, and the
    # closing depot visit.
    slots = positions or (2 * len(tasks) + 3)
    capacity = round(float(trucks[0].battery_capacity) * _SCALE)
    horizon = round(float(env.max_time) * _SCALE)

    model = cp_model.CpModel()
    at: dict = {}
    battery: dict = {}
    clock: dict = {}
    charge: dict = {}
    charge_time: dict = {}
    served: dict[int, list] = {index: [] for index in customer_indices}
    makespan = model.NewIntVar(0, horizon, "makespan")

    for truck in trucks:
        tid = truck.truck_id
        payload = round(float(truck.payload_capacity) * _SCALE)

        for p in range(slots):
            at[tid, p] = model.NewIntVar(0, size - 1, f"at{tid}_{p}")
            battery[tid, p] = model.NewIntVar(0, capacity, f"bat{tid}_{p}")
            clock[tid, p] = model.NewIntVar(0, horizon, f"clk{tid}_{p}")
            charge[tid, p] = model.NewIntVar(0, capacity, f"chg{tid}_{p}")
            charge_time[tid, p] = model.NewIntVar(0, horizon, f"ct{tid}_{p}")

            # Charging is only possible at a station, and its duration is the
            # delivered energy divided by that station's rated power.
            here_is_charger = model.NewIntVar(0, 1, f"ischg{tid}_{p}")
            model.AddElement(at[tid, p], charger_flag, here_is_charger)
            not_a_charger = model.NewBoolVar(f"nochg{tid}_{p}")
            model.Add(here_is_charger == 0).OnlyEnforceIf(not_a_charger)
            model.Add(here_is_charger == 1).OnlyEnforceIf(not_a_charger.Not())
            model.Add(charge[tid, p] == 0).OnlyEnforceIf(not_a_charger)

            station_power = model.NewIntVar(0, max_power, f"pw{tid}_{p}")
            model.AddElement(at[tid, p], power, station_power)
            delivered = model.NewIntVar(0, capacity * max_power, f"del{tid}_{p}")
            model.AddMultiplicationEquality(
                delivered, [charge_time[tid, p], station_power]
            )
            model.Add(delivered == charge[tid, p])

        model.Add(at[tid, 0] == index_of.get(int(truck.current_node), 0))
        model.Add(battery[tid, 0] == round(float(truck.current_battery) * _SCALE))
        model.Add(clock[tid, 0] == 0)
        model.Add(charge[tid, 0] == 0)
        model.Add(at[tid, slots - 1] == 0)

        for p in range(slots - 1):
            pair = model.NewIntVar(0, size * size - 1, f"pr{tid}_{p}")
            model.Add(pair == at[tid, p] * size + at[tid, p + 1])
            leg_energy = model.NewIntVar(0, _UNREACHABLE, f"e{tid}_{p}")
            leg_travel = model.NewIntVar(0, _UNREACHABLE, f"t{tid}_{p}")
            model.AddElement(pair, flat_energy, leg_energy)
            model.AddElement(pair, flat_travel, leg_travel)
            # Forbid legs the road network cannot serve.
            model.Add(leg_energy < _UNREACHABLE)
            model.Add(leg_travel < _UNREACHABLE)

            leg_service = model.NewIntVar(0, max(service) if service else 0, f"s{tid}_{p}")
            model.AddElement(at[tid, p + 1], service, leg_service)

            model.Add(battery[tid, p] + charge[tid, p] <= capacity)
            model.Add(
                battery[tid, p + 1] == battery[tid, p] + charge[tid, p] - leg_energy
            )
            model.Add(
                clock[tid, p + 1]
                == clock[tid, p] + charge_time[tid, p] + leg_travel + leg_service
            )

        model.Add(makespan >= clock[tid, slots - 1])

        load = []
        for index in customer_indices:
            visits = []
            for p in range(slots):
                hit = model.NewBoolVar(f"v{tid}_{p}_{index}")
                model.Add(at[tid, p] == index).OnlyEnforceIf(hit)
                model.Add(at[tid, p] != index).OnlyEnforceIf(hit.Not())
                visits.append(hit)
                served[index].append(hit)
            model.AddAtMostOne(visits)
            load.extend(demand[index] * visit for visit in visits)
        # One tour, no reload: the whole route must fit in the payload bay.
        model.Add(sum(load) <= payload)

    for index in customer_indices:
        model.AddExactlyOne(served[index])

    model.Minimize(makespan)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_seconds)
    solver.parameters.num_workers = int(workers)
    status = solver.Solve(model)
    name = solver.StatusName(status)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return ReferenceSolution(
            status=name,
            feasible=False,
            optimal_makespan=None,
            best_bound=None,
            routes={},
            wall_seconds=solver.WallTime(),
            positions=slots,
        )

    routes: dict[int, list[int]] = {}
    for truck in trucks:
        sequence: list[int] = []
        for p in range(slots):
            node = int(nodes[solver.Value(at[truck.truck_id, p])])
            if not sequence or sequence[-1] != node:
                sequence.append(node)
        routes[truck.truck_id] = sequence

    return ReferenceSolution(
        status=name,
        feasible=True,
        optimal_makespan=solver.ObjectiveValue() / _SCALE,
        best_bound=solver.BestObjectiveBound() / _SCALE,
        routes=routes,
        wall_seconds=solver.WallTime(),
        positions=slots,
    )
