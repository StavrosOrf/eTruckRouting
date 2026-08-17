"""Adaptive large neighbourhood search over the nominal fleet-routing problem.

R1.6 asks for a strong routing-and-charging metaheuristic rather than only a
constructive heuristic and an exact planner.  ALNS (Ropke and Pisinger) is the
standard answer for capacitated routing at this size, and it slots into the same
two-stage construction the CP-SAT baseline already uses:

1. search the nominal assignment/sequencing problem -- the same deterministic
   arc costs CP-SAT minimises, including the recharge time each leg's energy
   implies -- with destroy/repair operators under adaptive weights;
2. execute the resulting visit order in the stochastic simulator through the
   tuned heuristic's energy-safe navigation layer.

Sharing step 2 with CP-SAT and the greedy heuristic is what makes the comparison
about the search rather than about who repairs a broken plan more gracefully.
The search is deterministic given ``seed``, so a scenario replays exactly.
"""

from __future__ import annotations

import math
import random
import time
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
from EVRoutingEnv.baselines.exact_optimization import PlanSolution
from EVRoutingEnv.models.core.customer import TaskStatus


@dataclass(frozen=True)
class ALNSParameters:
    """Search budget, neighbourhood sizes, and acceptance schedule."""

    iterations: int = 2000
    time_limit_seconds: float = 10.0
    # Fraction of customers torn out per iteration, sampled in this range.
    min_destroy_fraction: float = 0.15
    max_destroy_fraction: float = 0.40
    # Simulated-annealing start temperature as a fraction of the initial cost.
    start_temperature_fraction: float = 0.05
    cooling_rate: float = 0.9985
    segment_length: int = 100
    reaction_factor: float = 0.35
    average_charging_power_kw: float = 300.0
    energy_safety_factor: float = 1.15
    target_soc: float = 1.0
    objective: str = "total_time"
    seed: int = 0
    replan_on_completion: bool = True

    def __post_init__(self) -> None:
        if self.objective not in ("makespan", "total_time"):
            raise ValueError(
                f"objective must be 'makespan' or 'total_time', got {self.objective!r}"
            )
        if not 0.0 < self.min_destroy_fraction <= self.max_destroy_fraction <= 1.0:
            raise ValueError("destroy fractions must satisfy 0 < min <= max <= 1")

    def as_dict(self) -> dict:
        return asdict(self)


class _NominalProblem:
    """Deterministic arc costs and demands read once per scenario."""

    def __init__(self, env, parameters: ALNSParameters):
        self.depot = int(env.joint_instance.depot_node)
        self.parameters = parameters
        tasks = [
            task
            for task in env.task_registry.tasks()
            if task.status in (TaskStatus.UNASSIGNED, TaskStatus.CLAIMED)
        ]
        self.customers = [int(task.node_id) for task in tasks]
        self.demand = {int(task.node_id): float(task.demand) for task in tasks}
        self.service = {
            int(task.node_id): float(task.base_service_time) for task in tasks
        }
        self.trucks = [
            truck for truck in env.trucks if not truck.is_complete and not truck.failed
        ]
        self.capacity = {
            truck.truck_id: (
                float(truck.remaining_payload)
                if truck.remaining_payload is not None
                else math.inf
            )
            for truck in self.trucks
        }
        self._arc_cache: dict[tuple[int, int], float] = {}
        self._env = env

    def arc(self, source: int, target: int) -> float:
        """Nominal travel hours plus the recharge time that leg's energy implies.

        This is exactly the quantity CP-SAT minimises, so the two planners are
        searching the same landscape and any difference is the search.
        """
        key = (source, target)
        cached = self._arc_cache.get(key)
        if cached is not None:
            return cached
        travel = _travel_hours(self._env, source, target)
        energy = _energy(self._env, source, target)
        if not math.isfinite(travel) or not math.isfinite(energy):
            cost = math.inf
        else:
            cost = travel + energy / max(
                self.parameters.average_charging_power_kw, 1.0
            )
        self._arc_cache[key] = cost
        return cost

    def route_cost(self, route: list[int]) -> float:
        if not route:
            return 0.0
        total = self.arc(self.depot, route[0])
        for index in range(len(route) - 1):
            total += self.arc(route[index], route[index + 1])
        total += self.arc(route[-1], self.depot)
        total += sum(self.service.get(node, 0.0) for node in route)
        return total

    def solution_cost(self, routes: dict[int, list[int]]) -> float:
        costs = [self.route_cost(route) for route in routes.values()]
        if not costs:
            return 0.0
        if any(not math.isfinite(cost) for cost in costs):
            return math.inf
        if self.parameters.objective == "makespan":
            return max(costs)
        return sum(costs)

    def load(self, route: list[int]) -> float:
        return sum(self.demand.get(node, 0.0) for node in route)

    def fits(self, truck_id: int, route: list[int], extra: int | None = None) -> bool:
        load = self.load(route)
        if extra is not None:
            load += self.demand.get(extra, 0.0)
        return load <= self.capacity[truck_id] + 1e-9


def _greedy_initial(problem: _NominalProblem) -> dict[int, list[int]]:
    """Cheapest-insertion construction, largest demand first."""
    routes: dict[int, list[int]] = {truck.truck_id: [] for truck in problem.trucks}
    pending = sorted(
        problem.customers, key=lambda node: -problem.demand.get(node, 0.0)
    )
    for node in pending:
        best = _best_insertion(problem, routes, node)
        if best is None:
            # No truck can take it within capacity: park it on the emptiest
            # route so the plan stays complete and the executor can still try.
            truck_id = min(routes, key=lambda key: problem.load(routes[key]))
            routes[truck_id].append(node)
            continue
        truck_id, position, _ = best
        routes[truck_id].insert(position, node)
    return routes


def _insertion_delta(
    problem: _NominalProblem, route: list[int], position: int, node: int
) -> float:
    previous = problem.depot if position == 0 else route[position - 1]
    following = problem.depot if position == len(route) else route[position]
    return (
        problem.arc(previous, node)
        + problem.arc(node, following)
        - problem.arc(previous, following)
        + problem.service.get(node, 0.0)
    )


def _best_insertion(
    problem: _NominalProblem,
    routes: dict[int, list[int]],
    node: int,
) -> tuple[int, int, float] | None:
    """Cheapest feasible (truck, position) for one customer."""
    best: tuple[int, int, float] | None = None
    for truck_id, route in routes.items():
        if not problem.fits(truck_id, route, extra=node):
            continue
        for position in range(len(route) + 1):
            delta = _insertion_delta(problem, route, position, node)
            if not math.isfinite(delta):
                continue
            if best is None or delta < best[2]:
                best = (truck_id, position, delta)
    return best


def _repair_greedy(
    problem: _NominalProblem,
    routes: dict[int, list[int]],
    removed: list[int],
    rng: random.Random,
) -> None:
    order = list(removed)
    rng.shuffle(order)
    for node in order:
        best = _best_insertion(problem, routes, node)
        if best is None:
            truck_id = min(routes, key=lambda key: problem.load(routes[key]))
            routes[truck_id].append(node)
            continue
        truck_id, position, _ = best
        routes[truck_id].insert(position, node)


def _repair_regret(
    problem: _NominalProblem,
    routes: dict[int, list[int]],
    removed: list[int],
    rng: random.Random,
) -> None:
    """Regret-2: insert whichever customer would suffer most from waiting."""
    pending = list(removed)
    while pending:
        best_node = None
        best_regret = -math.inf
        best_move: tuple[int, int] | None = None
        for node in pending:
            deltas: list[tuple[float, int, int]] = []
            for truck_id, route in routes.items():
                if not problem.fits(truck_id, route, extra=node):
                    continue
                for position in range(len(route) + 1):
                    delta = _insertion_delta(problem, route, position, node)
                    if math.isfinite(delta):
                        deltas.append((delta, truck_id, position))
            if not deltas:
                continue
            deltas.sort()
            regret = deltas[1][0] - deltas[0][0] if len(deltas) > 1 else math.inf
            if regret > best_regret:
                best_regret = regret
                best_node = node
                best_move = (deltas[0][1], deltas[0][2])
        if best_node is None:
            _repair_greedy(problem, routes, pending, rng)
            return
        truck_id, position = best_move
        routes[truck_id].insert(position, best_node)
        pending.remove(best_node)


def _destroy_random(
    problem: _NominalProblem,
    routes: dict[int, list[int]],
    count: int,
    rng: random.Random,
) -> list[int]:
    assigned = [(truck_id, node) for truck_id, r in routes.items() for node in r]
    rng.shuffle(assigned)
    removed = []
    for truck_id, node in assigned[:count]:
        routes[truck_id].remove(node)
        removed.append(node)
    return removed


def _destroy_worst(
    problem: _NominalProblem,
    routes: dict[int, list[int]],
    count: int,
    rng: random.Random,
) -> list[int]:
    """Remove the customers whose presence costs their route the most."""
    scored: list[tuple[float, int, int]] = []
    for truck_id, route in routes.items():
        for position, node in enumerate(route):
            previous = problem.depot if position == 0 else route[position - 1]
            following = (
                problem.depot if position == len(route) - 1 else route[position + 1]
            )
            saving = (
                problem.arc(previous, node)
                + problem.arc(node, following)
                - problem.arc(previous, following)
            )
            if math.isfinite(saving):
                scored.append((saving, truck_id, node))
    scored.sort(reverse=True)
    removed = []
    for _, truck_id, node in scored[:count]:
        routes[truck_id].remove(node)
        removed.append(node)
    return removed


def _destroy_related(
    problem: _NominalProblem,
    routes: dict[int, list[int]],
    count: int,
    rng: random.Random,
) -> list[int]:
    """Shaw removal: tear out a cluster of mutually close customers."""
    assigned = [(truck_id, node) for truck_id, r in routes.items() for node in r]
    if not assigned:
        return []
    seed_truck, seed_node = rng.choice(assigned)
    routes[seed_truck].remove(seed_node)
    removed = [seed_node]
    while len(removed) < count:
        remaining = [(t, n) for t, r in routes.items() for n in r]
        if not remaining:
            break
        reference = rng.choice(removed)
        remaining.sort(key=lambda pair: problem.arc(reference, pair[1]))
        truck_id, node = remaining[0]
        routes[truck_id].remove(node)
        removed.append(node)
    return removed


def _destroy_route(
    problem: _NominalProblem,
    routes: dict[int, list[int]],
    count: int,
    rng: random.Random,
) -> list[int]:
    """Empty one whole truck, forcing a different assignment split."""
    non_empty = [truck_id for truck_id, route in routes.items() if route]
    if not non_empty:
        return []
    truck_id = rng.choice(non_empty)
    removed = list(routes[truck_id])
    routes[truck_id] = []
    return removed


_DESTROY_OPERATORS = (
    ("random", _destroy_random),
    ("worst", _destroy_worst),
    ("related", _destroy_related),
    ("route", _destroy_route),
)
_REPAIR_OPERATORS = (
    ("greedy", _repair_greedy),
    ("regret", _repair_regret),
)


def _roulette(weights: list[float], rng: random.Random) -> int:
    total = sum(weights)
    if total <= 0.0:
        return rng.randrange(len(weights))
    threshold = rng.random() * total
    cumulative = 0.0
    for index, weight in enumerate(weights):
        cumulative += weight
        if cumulative >= threshold:
            return index
    return len(weights) - 1


def solve_alns_plan(env, parameters: ALNSParameters | None = None) -> PlanSolution:
    """Search the nominal plan with ALNS and report it like any other planner."""
    parameters = parameters or ALNSParameters()
    started = time.perf_counter()
    problem = _NominalProblem(env, parameters)
    if not problem.customers or not problem.trucks:
        return PlanSolution({}, "EMPTY", None, None, 0.0)

    # Seeded per scenario so a replay of the same scenario reproduces the plan.
    rng = random.Random((parameters.seed, int(env.scenario_seed)).__hash__())

    current = _greedy_initial(problem)
    current_cost = problem.solution_cost(current)
    best = {truck_id: list(route) for truck_id, route in current.items()}
    best_cost = current_cost
    initial_cost = current_cost

    temperature = max(
        1e-6, parameters.start_temperature_fraction * max(initial_cost, 1.0)
    )
    destroy_weights = [1.0] * len(_DESTROY_OPERATORS)
    repair_weights = [1.0] * len(_REPAIR_OPERATORS)
    destroy_scores = [0.0] * len(_DESTROY_OPERATORS)
    repair_scores = [0.0] * len(_REPAIR_OPERATORS)
    destroy_uses = [0] * len(_DESTROY_OPERATORS)
    repair_uses = [0] * len(_REPAIR_OPERATORS)

    total = len(problem.customers)
    iterations_run = 0
    for iteration in range(parameters.iterations):
        if time.perf_counter() - started > parameters.time_limit_seconds:
            break
        iterations_run = iteration + 1

        destroy_index = _roulette(destroy_weights, rng)
        repair_index = _roulette(repair_weights, rng)
        destroy_uses[destroy_index] += 1
        repair_uses[repair_index] += 1

        fraction = rng.uniform(
            parameters.min_destroy_fraction, parameters.max_destroy_fraction
        )
        count = max(1, min(total, round(fraction * total)))

        candidate = {truck_id: list(route) for truck_id, route in current.items()}
        removed = _DESTROY_OPERATORS[destroy_index][1](problem, candidate, count, rng)
        if not removed:
            continue
        _REPAIR_OPERATORS[repair_index][1](problem, candidate, removed, rng)
        candidate_cost = problem.solution_cost(candidate)

        score = 0.0
        if candidate_cost < best_cost - 1e-9:
            best = {truck_id: list(route) for truck_id, route in candidate.items()}
            best_cost = candidate_cost
            current, current_cost = candidate, candidate_cost
            score = 33.0
        elif candidate_cost < current_cost - 1e-9:
            current, current_cost = candidate, candidate_cost
            score = 13.0
        elif rng.random() < math.exp(
            -(candidate_cost - current_cost) / max(temperature, 1e-9)
        ):
            current, current_cost = candidate, candidate_cost
            score = 9.0
        destroy_scores[destroy_index] += score
        repair_scores[repair_index] += score

        temperature *= parameters.cooling_rate
        if (iteration + 1) % parameters.segment_length == 0:
            for index in range(len(destroy_weights)):
                if destroy_uses[index]:
                    destroy_weights[index] = (
                        1.0 - parameters.reaction_factor
                    ) * destroy_weights[index] + parameters.reaction_factor * (
                        destroy_scores[index] / destroy_uses[index]
                    )
            for index in range(len(repair_weights)):
                if repair_uses[index]:
                    repair_weights[index] = (
                        1.0 - parameters.reaction_factor
                    ) * repair_weights[index] + parameters.reaction_factor * (
                        repair_scores[index] / repair_uses[index]
                    )
            destroy_scores = [0.0] * len(_DESTROY_OPERATORS)
            repair_scores = [0.0] * len(_REPAIR_OPERATORS)
            destroy_uses = [0] * len(_DESTROY_OPERATORS)
            repair_uses = [0] * len(_REPAIR_OPERATORS)

    return PlanSolution(
        routes={truck_id: list(route) for truck_id, route in best.items()},
        # A metaheuristic proves nothing about optimality, and the status must
        # say so: this is the best solution found, never a bound.
        status=f"HEURISTIC_{iterations_run}_ITERATIONS",
        objective_hours=best_cost if math.isfinite(best_cost) else None,
        best_bound_hours=None,
        wall_seconds=time.perf_counter() - started,
    )


class ALNSPolicy:
    """Execute an ALNS nominal plan with the shared energy-safe repair layer."""

    name = "alns_plan"

    def __init__(
        self,
        parameters: ALNSParameters | None = None,
        heuristic_parameters: HeuristicParameters | None = None,
    ):
        self.parameters = parameters or ALNSParameters()
        self.navigator = GreedyHeuristicPolicy(
            heuristic_parameters
            or HeuristicParameters(
                energy_safety_factor=self.parameters.energy_safety_factor,
                target_soc=self.parameters.target_soc,
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
            self._plan = solve_alns_plan(env, self.parameters)
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
