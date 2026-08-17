"""Mathematical-programming baseline for the joint fleet-routing problem.

The stochastic, event-driven problem is not directly solvable to optimality, so
this baseline follows the standard deterministic-equivalent construction:

1. build the nominal capacitated routing model (assignment, sequencing, payload,
   energy-proportional recharge time, and the campaign objective -- either
   fleet makespan or total route time, see ``ExactPlannerParameters``);
2. solve it with CP-SAT to proven optimality (or to the best bound within a time
   limit) on the nominal instance;
3. execute the resulting visit order open-loop in the stochastic simulator,
   repairing only energy feasibility through the same safe-navigation routine
   the tuned heuristic uses.

Step 3 matters for fairness: an optimal nominal plan that is executed naively
would fail on the first adverse energy draw, which would understate the
baseline.  Giving the planner the heuristic's repair layer makes it strictly
stronger, so any remaining gap reflects the value of closed-loop decisions
rather than a missing implementation.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from EVRoutingEnv.baselines.canonical_baselines import (
    ActionKind,
    CandidateAction,
    GreedyHeuristicPolicy,
    HeuristicParameters,
    _energy,
    _travel_hours,
    decode_feasible_actions,
)
from EVRoutingEnv.models.core.customer import TaskStatus


_TIME_SCALE = 100.0


@dataclass(frozen=True)
class ExactPlannerParameters:
    """Solver budget and nominal-model settings."""

    time_limit_seconds: float = 20.0
    workers: int = 4
    average_charging_power_kw: float = 300.0
    energy_safety_factor: float = 1.15
    replan_on_completion: bool = True
    # ``makespan`` minimizes the longest single route; ``total_time`` minimizes
    # the sum over trucks. They are different plans: makespan balances the two
    # routes even when that costs fleet hours. Whichever the campaign reports as
    # its objective is the one this baseline must be given, or the comparison
    # measures the objective mismatch rather than the method.
    objective: str = "makespan"

    def __post_init__(self) -> None:
        if self.objective not in ("makespan", "total_time"):
            raise ValueError(
                f"objective must be 'makespan' or 'total_time', got {self.objective!r}"
            )

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PlanSolution:
    """Nominal plan plus the solver evidence needed to report it honestly."""

    routes: dict[int, list[int]]
    status: str
    objective_hours: float | None
    best_bound_hours: float | None
    wall_seconds: float

    @property
    def proven_optimal(self) -> bool:
        return self.status == "OPTIMAL"


def solve_nominal_plan(
    env,
    parameters: ExactPlannerParameters | None = None,
) -> PlanSolution:
    """Solve the nominal assignment and sequencing model under the chosen objective."""
    from ortools.sat.python import cp_model

    parameters = parameters or ExactPlannerParameters()
    depot = int(env.joint_instance.depot_node)
    tasks = [
        task
        for task in env.task_registry.tasks()
        if task.status in (TaskStatus.UNASSIGNED, TaskStatus.CLAIMED)
    ]
    trucks = [
        truck for truck in env.trucks if not truck.is_complete and not truck.failed
    ]
    if not tasks or not trucks:
        return PlanSolution({}, "EMPTY", None, None, 0.0)

    nodes = [depot] + [int(task.node_id) for task in tasks]
    demands = [0.0] + [float(task.demand) for task in tasks]
    service = [0.0] + [float(task.base_service_time) for task in tasks]
    size = len(nodes)

    def arc_cost(source_index: int, target_index: int) -> int:
        """Nominal travel plus the recharge time that leg's energy implies."""
        travel = _travel_hours(env, nodes[source_index], nodes[target_index])
        energy = _energy(env, nodes[source_index], nodes[target_index])
        if not math.isfinite(travel) or not math.isfinite(energy):
            return -1
        recharge = energy / max(parameters.average_charging_power_kw, 1.0)
        return round((travel + recharge) * _TIME_SCALE)

    model = cp_model.CpModel()
    visit = {}
    arcs_by_truck: dict[int, dict[tuple[int, int], object]] = {}
    leg_costs = []

    for truck in trucks:
        truck_id = truck.truck_id
        arcs: list[tuple[int, int, object]] = []
        arc_literals: dict[tuple[int, int], object] = {}
        for node_index in range(size):
            literal = model.NewBoolVar(f"visit_t{truck_id}_n{node_index}")
            visit[(truck_id, node_index)] = literal
            if node_index > 0:
                # A self-loop marks a node this truck skips.
                arcs.append((node_index, node_index, literal.Not()))
        model.Add(visit[(truck_id, 0)] == 1)

        for source in range(size):
            for target in range(size):
                if source == target:
                    continue
                cost = arc_cost(source, target)
                if cost < 0:
                    continue
                literal = model.NewBoolVar(f"arc_t{truck_id}_{source}_{target}")
                arcs.append((source, target, literal))
                arc_literals[(source, target)] = literal
                model.AddImplication(literal, visit[(truck_id, source)])
                model.AddImplication(literal, visit[(truck_id, target)])
        model.AddCircuit(arcs)
        arcs_by_truck[truck_id] = arc_literals

        payload = round(float(truck.payload_capacity) * _TIME_SCALE)
        model.Add(
            sum(
                round(demands[index] * _TIME_SCALE) * visit[(truck_id, index)]
                for index in range(1, size)
            )
            <= payload
        )

        route_time = sum(
            arc_cost(source, target) * literal
            for (source, target), literal in arc_literals.items()
        ) + sum(
            round(service[index] * _TIME_SCALE) * visit[(truck_id, index)]
            for index in range(1, size)
        )
        leg_costs.append(route_time)

    for index in range(1, size):
        model.AddExactlyOne(visit[(truck.truck_id, index)] for truck in trucks)

    if parameters.objective == "total_time":
        model.Minimize(sum(leg_costs))
    else:
        horizon = round(float(env.max_time) * _TIME_SCALE) * max(1, len(trucks))
        makespan = model.NewIntVar(0, horizon, "makespan")
        for route_time in leg_costs:
            model.Add(makespan >= route_time)
        model.Minimize(makespan)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(parameters.time_limit_seconds)
    solver.parameters.num_workers = int(parameters.workers)
    status = solver.Solve(model)
    status_name = solver.StatusName(status)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return PlanSolution({}, status_name, None, None, solver.WallTime())

    routes: dict[int, list[int]] = {}
    for truck in trucks:
        truck_id = truck.truck_id
        successor = {
            source: target
            for (source, target), literal in arcs_by_truck[truck_id].items()
            if solver.BooleanValue(literal)
        }
        order: list[int] = []
        current = 0
        seen = set()
        while current in successor and current not in seen:
            seen.add(current)
            current = successor[current]
            if current == 0:
                break
            order.append(int(nodes[current]))
        routes[truck_id] = order

    return PlanSolution(
        routes=routes,
        status=status_name,
        objective_hours=solver.ObjectiveValue() / _TIME_SCALE,
        best_bound_hours=solver.BestObjectiveBound() / _TIME_SCALE,
        wall_seconds=solver.WallTime(),
    )


class MathematicalProgrammingPolicy:
    """Execute a CP-SAT nominal plan with energy-safe repair."""

    name = "cpsat_plan"

    def __init__(
        self,
        parameters: ExactPlannerParameters | None = None,
        heuristic_parameters: HeuristicParameters | None = None,
    ):
        self.parameters = parameters or ExactPlannerParameters()
        self.navigator = GreedyHeuristicPolicy(
            heuristic_parameters
            or HeuristicParameters(
                energy_safety_factor=self.parameters.energy_safety_factor,
                target_soc=1.0,
                demand_weight=2.0,
            )
        )
        self._plan: PlanSolution | None = None
        self._scenario_key: object = None
        self.last_plan: PlanSolution | None = None

    def reset(self) -> None:
        self._plan = None
        self._scenario_key = None

    def _ensure_plan(self, env) -> PlanSolution:
        key = (id(env), env.scenario_seed)
        if self._plan is None or self._scenario_key != key:
            self._plan = solve_nominal_plan(env, self.parameters)
            self._scenario_key = key
            self.last_plan = self._plan
        return self._plan

    def __call__(self, env, observation=None, info=None) -> int:
        plan = self._ensure_plan(env)
        candidates = decode_feasible_actions(env)
        truck = env.trucks[env.active_truck_id]
        by_kind: dict[ActionKind, list[CandidateAction]] = {}
        for candidate in candidates:
            by_kind.setdefault(candidate.kind, []).append(candidate)

        goal = self._next_planned_stop(env, plan, truck)
        if goal is None:
            goal = self.navigator._select_goal(
                env,
                truck,
                by_kind.get(ActionKind.CUSTOMER, []),
                by_kind.get(ActionKind.DEPOT, []),
            )
        return self.navigator.navigate_toward(env, truck, goal, candidates, by_kind)

    def _next_planned_stop(self, env, plan: PlanSolution, truck) -> int | None:
        """First stop on this truck's route that the fleet still owes."""
        route = plan.routes.get(truck.truck_id, [])
        if not route:
            return None
        payload = (
            float(truck.remaining_payload)
            if truck.remaining_payload is not None
            else math.inf
        )
        for node in route:
            try:
                task = env.task_registry.task_for_node(int(node))
            except KeyError:
                continue
            if task.status is TaskStatus.SERVED or task.status is TaskStatus.IN_SERVICE:
                continue
            if task.status is TaskStatus.CLAIMED and task.claimed_by != truck.truck_id:
                continue
            if float(task.demand) > payload + 1e-9:
                continue
            return int(node)
        return None
