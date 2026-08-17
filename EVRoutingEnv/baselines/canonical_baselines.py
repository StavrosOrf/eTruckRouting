"""Non-learning baselines for the primary joint fleet-routing problem.

Every policy here obeys the same hard feasibility mask as the learned policies
and returns a plain integer action, so all methods can be scored by the same
:func:`EVRoutingEnv.evaluation.runner.run_evaluation_campaign`.

The baselines read the simulator directly (transport graph, registry, fleet
state).  That gives them at least as much information as the canonical
observation exposes, which keeps the comparison honest in the baselines' favour.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields, replace

import numpy as np

from EVRoutingEnv.models.core.customer import TaskStatus
from EVRoutingEnv.state.feasibility import ActionKind
from EVRoutingEnv.state.features import ACTION_FEATURES


_KIND_CODES = {kind: index for index, kind in enumerate(ActionKind)}
_KIND_COLUMN = ACTION_FEATURES.index("kind_code")
_TARGET_COLUMN = ACTION_FEATURES.index("target_node")
_CHARGE_COLUMN = ACTION_FEATURES.index("charge_value")
_FEASIBLE_COLUMN = ACTION_FEATURES.index("feasible")
_ENERGY_COLUMN = ACTION_FEATURES.index("required_energy")


@dataclass(frozen=True)
class CandidateAction:
    """One feasible action decoded from the canonical action rows."""

    index: int
    kind: ActionKind
    target_node: int
    charge_value: float
    required_energy: float


def decode_feasible_actions(env) -> list[CandidateAction]:
    """Decode the hard-feasible action set from the canonical snapshot."""
    rows = env.get_canonical_features().action_features
    candidates: list[CandidateAction] = []
    for index, row in enumerate(rows):
        if row[_FEASIBLE_COLUMN] <= 0.5:
            continue
        kind = list(ActionKind)[int(row[_KIND_COLUMN])]
        candidates.append(
            CandidateAction(
                index=index,
                kind=kind,
                target_node=int(row[_TARGET_COLUMN]),
                charge_value=float(row[_CHARGE_COLUMN]),
                required_energy=float(row[_ENERGY_COLUMN]),
            )
        )
    if not candidates:
        raise RuntimeError(
            "no feasible action is available; the baseline refuses to relax the "
            "hard mask"
        )
    return candidates


@dataclass(frozen=True)
class HeuristicParameters:
    """Tunable knobs of the greedy heuristic.

    ``energy_safety_factor`` guards against realized-energy uncertainty;
    ``target_soc`` is the floor on how far a recharge tops the battery up;
    ``time_weight``/``demand_weight`` trade travel time against payload progress
    when choosing the next customer; and ``require_continuation`` refuses a
    customer the truck could not then leave.

    There is deliberately no state-of-charge trigger.  Recharging is driven by
    whether the committed leg is affordable, which subsumes a fixed threshold: a
    truck at 20% that only needs a short hop should not stop, and one at 70%
    facing a long leg should.
    """

    energy_safety_factor: float = 1.25
    target_soc: float = 0.90
    time_weight: float = 1.0
    demand_weight: float = 0.0
    require_continuation: bool = True

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_settings(cls, settings: dict) -> HeuristicParameters:
        """Build from a stored settings dict, ignoring retired knobs."""
        allowed = {field.name for field in fields(cls)}
        return cls(**{k: v for k, v in settings.items() if k in allowed})


class GreedyHeuristicPolicy:
    """Goal-directed nearest-neighbour routing with en-route recharging.

    Each decision first commits to a goal -- the cheapest pending customer that
    fits the remaining payload, or the depot once nothing is pending.  If the
    goal is directly reachable under a pessimistic energy multiplier the truck
    goes there; otherwise it hops to the charger that minimises
    ``time(here -> charger) + time(charger -> goal)``.

    The goal term is what makes the baseline strong.  Some customers sit further
    than one full battery away, so a charger choice that ignores the goal makes
    a truck shuttle between nearby stations forever instead of closing distance.
    """

    name = "greedy_heuristic"

    def __init__(self, parameters: HeuristicParameters | None = None):
        self.parameters = parameters or HeuristicParameters()
        if not 0.0 < self.parameters.target_soc <= 1.0:
            raise ValueError("target_soc must lie in (0, 1]")
        if self.parameters.energy_safety_factor < 1.0:
            raise ValueError("energy_safety_factor must be at least 1")

    def __call__(self, env, observation=None, info=None) -> int:
        candidates = decode_feasible_actions(env)
        truck = env.trucks[env.active_truck_id]
        by_kind: dict[ActionKind, list[CandidateAction]] = {}
        for candidate in candidates:
            by_kind.setdefault(candidate.kind, []).append(candidate)
        goal = self._select_goal(
            env,
            truck,
            by_kind.get(ActionKind.CUSTOMER, []),
            by_kind.get(ActionKind.DEPOT, []),
        )
        return self.navigate_toward(env, truck, goal, candidates, by_kind)

    def navigate_toward(
        self,
        env,
        truck,
        goal: int | None,
        candidates: list[CandidateAction],
        by_kind: dict[ActionKind, list[CandidateAction]] | None = None,
    ) -> int:
        """Safely advance one step toward ``goal``, recharging when required.

        Shared with the mathematical-programming baseline so that a planner and
        the heuristic execute plans through identical energy-safe machinery, and
        any difference between them comes from the plan itself.
        """
        if by_kind is None:
            by_kind = {}
            for candidate in candidates:
                by_kind.setdefault(candidate.kind, []).append(candidate)
        battery = float(truck.current_battery)
        capacity = float(truck.battery_capacity)
        origin = int(truck.current_node)
        safety = self.parameters.energy_safety_factor
        charge_actions = by_kind.get(ActionKind.CHARGE, [])
        customer_actions = by_kind.get(ActionKind.CUSTOMER, [])
        charger_actions = by_kind.get(ActionKind.CHARGER, [])
        depot_actions = by_kind.get(ActionKind.DEPOT, [])

        # A leg is only safe if the truck can reach the goal *and* still leave it.
        # Ignoring the onward leg is what strands trucks, because the model
        # requires a depot return after the last customer.
        leg_energy = self._leg_energy(env, origin, goal)
        reach_goal_directly = goal is not None and battery >= safety * leg_energy

        if charge_actions:
            required = (
                leg_energy
                if math.isfinite(leg_energy)
                else self._nearest_hop_energy(env, origin, charger_actions)
            )
            if battery < safety * required:
                desired = min(
                    1.0,
                    max(
                        self.parameters.target_soc,
                        safety * required / max(capacity, 1e-9),
                    ),
                )
                return self._pick_charge_action(charge_actions, desired)

        if reach_goal_directly:
            direct = [
                action
                for action in customer_actions + depot_actions
                if action.target_node == goal
            ]
            if direct:
                return direct[0].index
            # The goal is reachable but not yet actionable -- typically the depot
            # while the fleet still owes customers. Topping up in place burns
            # clock without burning energy, which beats driving in circles.
            if charge_actions:
                return self._pick_charge_action(charge_actions, 1.0)

        if charger_actions:
            # Some customers cannot be reached from the current station even on a
            # full battery. Aim at the station the goal *is* reachable from, so
            # each hop closes distance to that staging point rather than to a
            # goal no single leg can cover.
            staging = self._staging_charger(env, goal, capacity)
            aim = staging if staging is not None else goal
            return self._pick_charger(env, origin, aim, battery, charger_actions).index

        if customer_actions:
            reachable = [
                action
                for action in customer_actions
                if math.isfinite(_energy(env, origin, action.target_node))
            ]
            if reachable:
                return min(
                    reachable,
                    key=lambda action: _travel_hours(env, origin, action.target_node),
                ).index
        if depot_actions:
            return depot_actions[0].index
        if charge_actions:
            return self._pick_charge_action(charge_actions, 1.0)
        return candidates[0].index

    def _select_goal(
        self,
        env,
        truck,
        customer_actions: list[CandidateAction],
        depot_actions: list[CandidateAction],
    ) -> int | None:
        """Pick the next customer to aim for, or the depot when none remain."""
        origin = int(truck.current_node)
        payload = (
            float(truck.remaining_payload)
            if truck.remaining_payload is not None
            else math.inf
        )
        capacity = float(truck.battery_capacity)
        pending = _pending_customer_nodes(env)
        best_node: int | None = None
        best_score = math.inf
        for node in _claimable_customer_nodes(env, truck.truck_id):
            demand = _demand_for_node(env, node)
            if demand > payload + 1e-9:
                continue
            travel_time = _travel_hours(env, origin, node)
            if not math.isfinite(travel_time):
                continue
            if self.parameters.require_continuation and not self._servable(
                env, node, pending, capacity
            ):
                continue
            score = (
                self.parameters.time_weight * travel_time
                - self.parameters.demand_weight * demand
            )
            if score < best_score:
                best_score = score
                best_node = node
        if best_node is not None:
            return best_node
        if customer_actions:
            return int(customer_actions[0].target_node)
        # Nothing serviceable is left for this truck. The depot is still barred
        # while the fleet owes customers, so aim for it anyway: that keeps the
        # truck drifting toward its final destination instead of shuttling.
        return int(env.joint_instance.depot_node)

    def _leg_energy(self, env, origin: int, goal: int | None) -> float:
        """Energy to reach the goal *and* still reach a charger or the depot."""
        if goal is None:
            return 0.0
        outbound = _energy(env, origin, goal)
        if not math.isfinite(outbound):
            return math.inf
        if goal == int(env.joint_instance.depot_node):
            return outbound
        onward = _nearest_exit_energy(
            env, goal, _pending_customer_nodes(env), exclude=goal
        )
        return outbound + onward if math.isfinite(onward) else math.inf

    def _staging_charger(self, env, goal: int | None, capacity: float) -> int | None:
        """Cheapest station from which the goal is reachable on a full battery."""
        if goal is None:
            return None
        safety = self.parameters.energy_safety_factor
        onward = (
            0.0
            if goal == int(env.joint_instance.depot_node)
            else _nearest_exit_energy(
                env, goal, _pending_customer_nodes(env), exclude=goal
            )
        )
        if not math.isfinite(onward):
            return None
        best: int | None = None
        best_energy = math.inf
        for charger in env.charging_nodes:
            approach = _energy(env, int(charger), goal)
            if not math.isfinite(approach):
                continue
            if safety * (approach + onward) > capacity:
                continue
            if approach < best_energy:
                best_energy = approach
                best = int(charger)
        return best

    def _servable(self, env, node: int, pending: list[int], capacity: float) -> bool:
        """Reject customers that would strand a truck even on a full battery."""
        onward = _nearest_exit_energy(env, node, pending, exclude=node)
        if not math.isfinite(onward):
            return False
        approaches = [
            _energy(env, int(charger), node) for charger in env.charging_nodes
        ]
        approaches.append(_energy(env, int(env.joint_instance.depot_node), node))
        best_approach = min(
            (value for value in approaches if math.isfinite(value)), default=math.inf
        )
        if not math.isfinite(best_approach):
            return False
        safety = self.parameters.energy_safety_factor
        return capacity - safety * best_approach >= safety * onward

    @staticmethod
    def _nearest_hop_energy(
        env,
        origin: int,
        charger_actions: list[CandidateAction],
    ) -> float:
        hops = [
            energy
            for energy in (
                _energy(env, origin, action.target_node) for action in charger_actions
            )
            if math.isfinite(energy)
        ]
        return min(hops) if hops else 0.0

    @staticmethod
    def _pick_charge_action(
        charge_actions: list[CandidateAction],
        desired: float,
    ) -> int:
        at_or_above = [
            action for action in charge_actions if action.charge_value >= desired - 1e-9
        ]
        if at_or_above:
            return min(at_or_above, key=lambda action: action.charge_value).index
        return max(charge_actions, key=lambda action: action.charge_value).index

    def _pick_charger(
        self,
        env,
        origin: int,
        goal: int | None,
        battery: float,
        charger_actions: list[CandidateAction],
    ) -> CandidateAction:
        """Choose the station that best advances the truck toward its goal."""
        station = env.charging_station
        goal_distance_now = (
            _travel_hours(env, origin, goal) if goal is not None else 0.0
        )

        def congestion(node: int) -> float:
            queue = len(station.charger_waitlist.get(node, []))
            occupancy = len(station.charger_occupancy.get(node, []))
            capacity = max(1, int(station.charger_capacity.get(node, 1)))
            return (queue + max(0, occupancy - capacity + 1)) / capacity

        def score(action: CandidateAction) -> tuple[float, float, float]:
            node = action.target_node
            outbound = _travel_hours(env, origin, node)
            onward = _travel_hours(env, node, goal) if goal is not None else 0.0
            if not math.isfinite(onward):
                onward = 1e6
            power = float(station.charger_power_kw.get(node, 1.0))
            # Closing distance to the goal comes first: minimising the *sum*
            # instead makes the truck take many tiny hops between adjacent
            # stations, burning the step budget without covering ground.
            return (onward, outbound + congestion(node), -power)

        # Prefer stations that strictly close distance to the goal; fall back to
        # the full set only when none does, so the truck never shuttles.
        progressing = [
            action
            for action in charger_actions
            if goal is None
            or _travel_hours(env, action.target_node, goal) < goal_distance_now
        ]
        return min(progressing or charger_actions, key=score)


@dataclass(frozen=True)
class MPCParameters:
    """Settings for the rolling-horizon model predictive controller."""

    horizon: int = 4
    branching: int = 4
    energy_safety_factor: float = 1.25
    target_soc: float = 0.90
    unserved_penalty_hours: float = 50.0
    failure_penalty_hours: float = 500.0

    def as_dict(self) -> dict:
        return asdict(self)


class RollingHorizonMPCPolicy:
    """Receding-horizon controller over the deterministic nominal model.

    At every decision the controller enumerates the customers the active truck
    may still claim, scores each as the first stop of a bounded nominal rollout
    (travel + service + energy-implied recharge time, plus a penalty for stops
    the horizon cannot cover), and commits only to the best next goal -- the
    standard receding-horizon contract.

    Execution reuses the tuned heuristic's energy-safe navigation, so the
    controller and the greedy baseline differ only in *which* stop they choose,
    not in how safely they drive there.  Planning over raw actions instead made
    the controller strand trucks, which measured the execution layer rather
    than the lookahead.
    """

    name = "rolling_horizon_mpc"

    def __init__(
        self,
        parameters: MPCParameters | None = None,
        heuristic_parameters: HeuristicParameters | None = None,
    ):
        self.parameters = parameters or MPCParameters()
        if self.parameters.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.parameters.branching <= 0:
            raise ValueError("branching must be positive")
        self.navigator = GreedyHeuristicPolicy(
            heuristic_parameters
            or HeuristicParameters(
                energy_safety_factor=self.parameters.energy_safety_factor,
                target_soc=self.parameters.target_soc,
                demand_weight=2.0,
            )
        )

    def __call__(self, env, observation=None, info=None) -> int:
        candidates = decode_feasible_actions(env)
        truck = env.trucks[env.active_truck_id]
        by_kind: dict[ActionKind, list[CandidateAction]] = {}
        for candidate in candidates:
            by_kind.setdefault(candidate.kind, []).append(candidate)

        goal = self._plan_goal(env, truck)
        if goal is None:
            goal = self.navigator._select_goal(
                env,
                truck,
                by_kind.get(ActionKind.CUSTOMER, []),
                by_kind.get(ActionKind.DEPOT, []),
            )
        return self.navigator.navigate_toward(env, truck, goal, candidates, by_kind)

    def _plan_goal(self, env, truck) -> int | None:
        """Choose the next stop by bounded rollout over the nominal model."""
        capacity = float(truck.battery_capacity)
        payload = (
            float(truck.remaining_payload)
            if truck.remaining_payload is not None
            else math.inf
        )
        pending = _pending_customer_nodes(env)
        options = [
            node
            for node in _claimable_customer_nodes(env, truck.truck_id)
            if _demand_for_node(env, node) <= payload + 1e-9
            and self.navigator._servable(env, node, pending, capacity)
            and math.isfinite(_travel_hours(env, int(truck.current_node), node))
        ]
        if not options:
            return None

        state = _NominalState(
            node=int(truck.current_node),
            battery=float(truck.current_battery),
            capacity=capacity,
            payload=payload,
            clock=0.0,
            pending=frozenset(options),
        )
        best_node = options[0]
        best_cost = math.inf
        for node in options:
            successor = self._advance(env, state, node)
            cost = successor.clock + self._rollout(
                env, successor, self.parameters.horizon - 1
            )
            if cost < best_cost:
                best_cost = cost
                best_node = node
        return best_node

    def _rollout(self, env, state: _NominalState, depth: int) -> float:
        if not state.pending:
            travel = _travel_hours(env, state.node, int(env.joint_instance.depot_node))
            return (
                travel
                if math.isfinite(travel)
                else self.parameters.failure_penalty_hours
            )
        if depth <= 0:
            return self.parameters.unserved_penalty_hours * len(state.pending)

        reachable = [
            node
            for node in state.pending
            if _demand_for_node(env, node) <= state.payload + 1e-9
            and math.isfinite(_travel_hours(env, state.node, node))
        ]
        if not reachable:
            return self.parameters.unserved_penalty_hours * len(state.pending)
        reachable.sort(key=lambda node: _travel_hours(env, state.node, node))

        best = math.inf
        for node in reachable[: self.parameters.branching]:
            successor = self._advance(env, state, node)
            best = min(
                best,
                (successor.clock - state.clock)
                + self._rollout(env, successor, depth - 1),
            )
        return best

    def _advance(self, env, state: _NominalState, node: int) -> _NominalState:
        """Nominal transition priced with the simulator's own charging curve."""
        energy = _energy(env, state.node, node)
        travel = _travel_hours(env, state.node, node)
        if not math.isfinite(energy) or not math.isfinite(travel):
            return replace(
                state, clock=state.clock + self.parameters.failure_penalty_hours
            )
        safety = self.parameters.energy_safety_factor
        recharge_hours = 0.0
        battery = state.battery - energy
        if battery < safety * energy or battery < 0.0:
            # The nominal plan must pay for the charge this leg implies, using
            # the same nonlinear taper the simulator will actually apply.
            fastest = max(
                env.charging_nodes,
                key=lambda charger: float(
                    env.charging_station.charger_power_kw.get(int(charger), 0.0)
                ),
                default=None,
            )
            target = self.parameters.target_soc * state.capacity
            if fastest is not None:
                recharge_hours = nominal_charge_hours(
                    env,
                    int(fastest),
                    max(0.0, battery) / max(state.capacity, 1e-9),
                    self.parameters.target_soc,
                    state.capacity,
                )
            battery = max(target - energy, 0.0)
        return _NominalState(
            node=node,
            battery=battery,
            capacity=state.capacity,
            payload=state.payload - _demand_for_node(env, node),
            clock=state.clock + travel + recharge_hours + _service_hours(env, node),
            pending=state.pending - {node},
        )


@dataclass(frozen=True)
class _NominalState:
    node: int
    battery: float
    capacity: float
    payload: float
    clock: float
    pending: frozenset


class RandomFeasiblePolicy:
    """Uniform sampling over the hard-feasible set; the weakest reference."""

    name = "random_feasible"

    def __init__(self, seed: int = 0):
        self._rng = np.random.default_rng(seed)

    def __call__(self, env, observation=None, info=None) -> int:
        candidates = decode_feasible_actions(env)
        return int(self._rng.choice([action.index for action in candidates]))


_CHARGE_TIME_CACHE: dict[tuple, float] = {}


def nominal_charge_hours(
    env,
    charger_node: int,
    initial_soc: float,
    target_soc: float,
    capacity: float,
) -> float:
    """Charging time under the *same* nonlinear model the simulator executes.

    Planners previously priced a recharge as energy over rated power, which
    ignores the constant-voltage taper and understates the cost of topping up a
    nearly full battery.  Using the simulator's own curve removes that
    plan-versus-execution mismatch.  Results are memoized on a coarse state-of
    -charge grid because a receding-horizon rollout queries this constantly and
    the curve is smooth.
    """
    if target_soc <= initial_soc:
        return 0.0
    model = getattr(env, "charging_curve_model", None)
    station = getattr(env, "charging_station", None)
    if model is None or station is None:
        return 0.0

    charger_type = station.charger_type.get(int(charger_node), "DCFast")
    power = float(station.charger_power_kw.get(int(charger_node), 150.0))

    # Mirror the environment exactly: the model takes the flat per-type section
    # with the station's own power and the global realistic-curve flag injected.
    section_key = "dcfast" if charger_type == "DCFast" else "level2"
    charger_config = dict(env.charging_config.get(section_key, {}))
    charger_config["use_realistic_curve"] = env.charging_config.get(
        "use_realistic_curve", False
    )
    charger_config["charge_rate"] = power

    # The cache is process-wide, so the key carries every curve parameter that
    # could differ between configurations, not just the station identity.
    key = (
        charger_type,
        round(power, 1),
        round(initial_soc, 2),
        round(target_soc, 2),
        round(capacity, 1),
        tuple(sorted((str(name), str(value)) for name, value in charger_config.items())),
    )
    cached = _CHARGE_TIME_CACHE.get(key)
    if cached is not None:
        return cached
    # `calculate_charge_to_target` is the routine the simulator itself executes
    # and it integrates to an exact target. `estimate_charge_time` is a binary
    # search that fails to converge at a target of 1.0, where the constant
    # -voltage tail approaches the target asymptotically, so it is not used here.
    try:
        _, details = model.calculate_charge_to_target(
            initial_soc=round(initial_soc, 2),
            target_soc=round(target_soc, 2),
            battery_capacity=capacity,
            charger_config=charger_config,
            charger_type=charger_type,
        )
        hours = float(details["actual_charge_hours"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        hours = max(0.0, (target_soc - initial_soc) * capacity) / max(power, 1.0)
    if not math.isfinite(hours) or hours < 0.0:
        hours = max(0.0, (target_soc - initial_soc) * capacity) / max(power, 1.0)
    _CHARGE_TIME_CACHE[key] = hours
    return hours


def _pending_customer_nodes(env) -> list[int]:
    """Every customer the fleet still owes, regardless of which truck holds it."""
    return [
        int(task.node_id)
        for task in env.task_registry.tasks()
        if task.status in (TaskStatus.UNASSIGNED, TaskStatus.CLAIMED)
    ]


def _claimable_customer_nodes(env, truck_id: int) -> list[int]:
    """Customers *this* truck may still serve.

    A task claimed by another truck is permanently infeasible here, so treating
    it as a goal makes the truck chase a target it can never reach.
    """
    nodes: list[int] = []
    for task in env.task_registry.tasks():
        if (
            task.status is TaskStatus.UNASSIGNED
            or task.status is TaskStatus.CLAIMED
            and task.claimed_by == truck_id
        ):
            nodes.append(int(task.node_id))
    return nodes


def _nearest_exit_energy(
    env,
    origin: int,
    pending: list[int],
    exclude: int | None = None,
) -> float:
    """Energy to the cheapest safe continuation: a charger, or the depot."""
    best = math.inf
    for charger in env.charging_nodes:
        best = min(best, _energy(env, origin, int(charger)))
    remaining = [node for node in pending if node != exclude]
    if not remaining:
        best = min(best, _energy(env, origin, int(env.joint_instance.depot_node)))
    return best


def _task_for_node(env, node: int):
    try:
        return env.task_registry.task_for_node(int(node))
    except KeyError:
        return None


def _demand_for_node(env, node: int) -> float:
    task = _task_for_node(env, node)
    return float(task.demand) if task is not None else 0.0


def _service_hours(env, node: int) -> float:
    task = _task_for_node(env, node)
    return float(task.base_service_time) if task is not None else 0.0


def _energy(env, source: int, target: int) -> float:
    try:
        value = float(env.transport_graph.get_path_energy(int(source), int(target)))
    except (KeyError, TypeError, ValueError):
        return math.inf
    return value if math.isfinite(value) and value >= 0.0 else math.inf


def _travel_hours(env, source: int, target: int) -> float:
    try:
        value = float(env.transport_graph.get_time_distance(int(source), int(target)))
    except (KeyError, TypeError, ValueError):
        return math.inf
    return value if math.isfinite(value) and value >= 0.0 else math.inf
