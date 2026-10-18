"""Clairvoyant branch-and-bound over the seeded simulator's discrete actions.

Every branch executes actual traffic/energy/service realizations, nonlinear
charging, and shared FCFS queues. Exhaustion proves optimality for the configured
simulator: discrete charging choices, known realized leg costs, fixed road paths,
and episode time/step limits. This is an exact small-instance reference, not a
scalable MIP or a claim about continuous charging or unrestricted physical EVRP.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import asdict, dataclass

import numpy as np

from EVRoutingEnv.evaluation.artifacts import canonical_json_sha256
from EVRoutingEnv.evaluation.metrics import extract_operational_metrics


@dataclass(frozen=True)
class PerfectInformationParameters:
    objective: str = "makespan"
    information_mode: str = "realized"
    time_limit_seconds: float | None = 60.0
    max_nodes: int = 100_000

    def __post_init__(self):
        if self.information_mode not in {"realized", "simulator"}:
            raise ValueError("information_mode must be realized or simulator")
        if self.objective not in {"makespan", "total_time"}:
            raise ValueError("objective must be 'makespan' or 'total_time'")
        if self.time_limit_seconds is not None and (
            not math.isfinite(self.time_limit_seconds) or self.time_limit_seconds <= 0
        ):
            raise ValueError("time_limit_seconds must be finite and positive")
        if (
            isinstance(self.max_nodes, bool)
            or not isinstance(self.max_nodes, int)
            or self.max_nodes < 1
        ):
            raise ValueError("max_nodes must be a positive integer")


@dataclass(frozen=True)
class PerfectInformationSolution:
    status: str
    objective: str
    objective_hours: float | None
    best_bound_hours: float
    actions: tuple[tuple[int, int], ...]
    scenario_sha256: str
    nodes_expanded: int
    wall_seconds: float
    stop_reason: str
    information_mode: str = "realized"

    @property
    def proven_optimal(self) -> bool:
        return self.status == "OPTIMAL"

    def as_dict(self) -> dict:
        return {
            **asdict(self),
            "proven_optimal": self.proven_optimal,
            "optimality_scope": "configured_simulator_discrete_actions",
        }


def _clone(env):
    # Share read-only inputs, but copy all mutable episode state, preserving RNG
    # aliases within a branch. Graph caches are derived memoization only.
    shared = (
        env.transport_graph,
        env.config,
        env.joint_instance,
        env.scenario_descriptor,
    )
    memo = {id(value): value for value in shared}
    if getattr(env, "_scenario_graph_digest", None) is not None:
        memo[id(env._scenario_graph_digest)] = env._scenario_graph_digest
    shadow = copy.deepcopy(env, memo)
    shadow.verbose = shadow.enable_plotting = False
    shadow.plotter = shadow.stats_collector = shadow.charging_logger = None
    shadow.event_handler.verbose = False
    shadow.charging_station.verbose = False
    shadow.traffic_simulator.verbose = shadow.delivery_simulator.verbose = False
    shadow.charging_curve_model.verbose = False
    return shadow


def _require_initial_state(env):
    if env.scenario_descriptor is None or env.scenario_seed is None:
        raise ValueError("reset(seed=...) before solving or replaying")
    if env.episode_steps != 0 or env.global_clock != 0 or env._episode_ended:
        raise ValueError("a freshly reset scenario is required")


def _value(env, objective):
    if objective == "makespan":
        return float(env.global_clock)
    return extract_operational_metrics(env).total_operating_time


def _successful(env):
    return env._episode_ended and extract_operational_metrics(env).success


def _replay(env, actions):
    for truck_id, action in actions:
        if env._episode_ended or env.active_truck_id != truck_id:
            raise ValueError("action trace does not match the active truck")
        if not 0 <= action < env.action_space.n or not env.mask_fn()[action]:
            raise ValueError("action trace contains an infeasible discrete action")
        env.step(action)
    if not _successful(env):
        raise ValueError(
            "action trace does not complete every customer and depot return"
        )


def replay_perfect_information_plan(env, solution: PerfectInformationSolution) -> dict:
    """Replay via the public API on private state, returning realized schedules.

    Repeated replays must produce the same ``replay_sha256``. The live episode
    and its uncertainty streams remain untouched.
    """
    _require_initial_state(env)
    if canonical_json_sha256(env.scenario_descriptor) != solution.scenario_sha256:
        raise ValueError("plan belongs to a different scenario/configuration/network")
    if solution.objective_hours is None:
        raise ValueError("solution has no feasible incumbent to replay")
    shadow = _clone(env)
    shadow._perfect_information = solution.information_mode == "realized"
    _replay(shadow, solution.actions)
    actual = _value(shadow, solution.objective)
    if not math.isclose(actual, solution.objective_hours, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("replay objective differs from the saved solution")
    payload = {
        "scenario_sha256": solution.scenario_sha256,
        "information_mode": solution.information_mode,
        "actions": solution.actions,
        "operational_metrics": extract_operational_metrics(shadow).as_dict(),
        "routes": {
            str(truck.truck_id): [
                env.joint_instance.depot_node,
                *(
                    event["location"]
                    for event in truck.get_event_history()
                    if event["event_type"] == "ROUTING_END"
                ),
            ]
            for truck in shadow.trucks
        },
        "truck_event_history": {
            str(truck.truck_id): truck.get_event_history() for truck in shadow.trucks
        },
        "charger_queue_history": {
            str(node): history
            for node, history in shadow.charging_station.queue_history.items()
        },
    }
    return {**payload, "replay_sha256": canonical_json_sha256(payload)}


class _RemainingWorkBound:
    """Optimistic travel closure plus unserved minimum service durations.

    Every realized leg is at least 0.85 times its nominal duration (or nominal
    when traffic is off). Floyd-Warshall allows arbitrary intermediate stops,
    relaxing battery, payload, station, and order constraints. This closure is
    needed because energy-selected nominal paths need not obey a time triangle
    inequality. A bare direct-leg nominal cost is not a valid lower bound.
    """

    def __init__(self, env):
        nodes = sorted(
            {
                int(env.joint_instance.depot_node),
                *map(int, env.charging_nodes),
                *(int(task.node_id) for task in env.task_registry.tasks()),
            }
        )
        self.index = {node: i for i, node in enumerate(nodes)}
        factor = 0.85 if env.traffic_simulator.enable_traffic else 1.0
        self.hours = np.array(
            [
                [factor * env.transport_graph.get_time_distance(a, b) for b in nodes]
                for a in nodes
            ],
            dtype=float,
        )
        np.fill_diagonal(self.hours, 0.0)
        for k in range(len(nodes)):
            self.hours = np.minimum(
                self.hours, self.hours[:, k, None] + self.hours[None, k, :]
            )
        self.depot = int(env.joint_instance.depot_node)

    def travel(self, source, target):
        return float(self.hours[self.index[int(source)], self.index[int(target)]])

    def __call__(self, env, objective):
        clock = float(env.global_clock)
        positions = []
        remaining_travel = 0.0
        makespan = clock
        for truck in env.trucks:
            if truck.is_complete or truck.failed:
                continue
            # In-flight travel cannot be diverted. Other activity can be ignored
            # optimistically; its remaining service is counted below.
            if env.truck_states[truck.truck_id] == "routing":
                ready = float(truck.route_arrival_time)
                node = int(truck.route_destination)
            else:
                ready, node = clock, int(truck.current_node)
            home = self.travel(node, self.depot)
            positions.append((ready, node))
            remaining_travel += max(0.0, ready - clock) + home
            makespan = max(makespan, ready + home)
        remaining_service = 0.0
        for task in env.task_registry.tasks():
            if task.is_served:
                continue
            factor = (
                env.delivery_simulator.min_unloading_multiplier
                if env.delivery_simulator.enable_stochastic_unloading
                else 1.0
            )
            elapsed = (
                max(0.0, clock - task.service_started_at)
                if task.service_started_at is not None
                else 0.0
            )
            service = max(0.0, task.base_service_time * factor - elapsed)
            remaining_service += service
            earliest = min(
                (ready + self.travel(node, task.node_id) for ready, node in positions),
                default=math.inf,
            )
            makespan = max(
                makespan,
                max(earliest, task.earliest_service)
                + service
                + self.travel(task.node_id, self.depot),
            )
        accrued = _value(env, objective)
        bound = (
            makespan
            if objective == "makespan"
            else accrued + remaining_travel + remaining_service
        )
        # Keep roundoff in the shortest-path relaxation on the optimistic side.
        return max(accrued, bound - 1e-10) if math.isfinite(bound) else bound


def solve_perfect_information_plan(
    env,
    parameters: PerfectInformationParameters | None = None,
) -> PerfectInformationSolution:
    """Search all feasible discrete plans until exhaustion or a budget limit.

    Bounds use accrued time, guaranteed travel/service floors, and remaining
    in-flight travel. State dominance is not assumed: arrivals change time
    buckets, draws, and queue order. Invalid actions refuse without advancing
    time or fail a truck, so the hard mask loses no successful optimum.
    """
    _require_initial_state(env)
    parameters = parameters or PerfectInformationParameters()
    started = time.monotonic()
    root = _clone(env)
    root._perfect_information = parameters.information_mode == "realized"
    incumbent = math.inf
    best_actions = ()
    # The generation witness supplies an upper bound only. Assignments and
    # ordering remain free in the search, subject to configured model rules.
    witness = tuple((int(t), int(a)) for t, a in env.feasibility_witness)
    if witness:
        candidate = _clone(root)
        try:
            _replay(candidate, witness)
        except ValueError:
            pass
        else:
            incumbent, best_actions = _value(candidate, parameters.objective), witness

    lower_bound_for = _RemainingWorkBound(root)

    def frame(state, trace):
        actions = [int(a) for a, allowed in enumerate(state.mask_fn()) if allowed]
        # Explore deliveries before charge detours; ordering affects speed only.
        actions.sort(key=lambda a: (a < state.num_charging_nodes, a))
        return state, trace, iter(actions), lower_bound_for(state, parameters.objective)

    # Lazy DFS frames keep memory proportional to depth, not the search tree.
    pending = [frame(root, ())]
    expanded = 0
    stop_reason = "search_exhausted"
    while pending:
        state, trace, alternatives, bound = pending[-1]
        if bound >= incumbent:
            pending.pop()
            continue
        action = next(alternatives, None)
        if action is None:
            pending.pop()
            continue
        # Retain this parent's bound if a budget prevents exploring its child.
        if expanded >= parameters.max_nodes:
            stop_reason = "node_limit"
            break
        if (
            parameters.time_limit_seconds is not None
            and time.monotonic() - started >= parameters.time_limit_seconds
        ):
            stop_reason = "time_limit"
            break
        child = _clone(state)
        child_trace = (*trace, (int(child.active_truck_id), action))
        child.step(action)
        expanded += 1
        if any(truck.failed for truck in child.trucks):
            continue
        if _successful(child):
            value = _value(child, parameters.objective)
            if value < incumbent:
                incumbent, best_actions = value, child_trace
        elif (
            not child._episode_ended and _value(child, parameters.objective) < incumbent
        ):
            pending.append(frame(child, child_trace))

    feasible = math.isfinite(incumbent)
    if pending:
        status = "FEASIBLE" if feasible else "UNKNOWN"
    else:
        status = "OPTIMAL" if feasible else "INFEASIBLE"
    lower_bound = min((entry[3] for entry in pending), default=incumbent)
    if feasible:
        lower_bound = min(lower_bound, incumbent)
    elif not math.isfinite(lower_bound):
        lower_bound = 0.0
    return PerfectInformationSolution(
        status=status,
        objective=parameters.objective,
        objective_hours=incumbent if feasible else None,
        best_bound_hours=lower_bound,
        actions=best_actions,
        scenario_sha256=canonical_json_sha256(env.scenario_descriptor),
        nodes_expanded=expanded,
        wall_seconds=time.monotonic() - started,
        stop_reason=stop_reason,
        information_mode=parameters.information_mode,
    )
