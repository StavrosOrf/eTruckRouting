"""Size-agnostic graph observation for one policy across problem families.

The canonical flat observation pads every entity type to a per-configuration
envelope. Across the generalist suite that envelope ranges from 25 to 432
chargers, 30 to 200 request slots and 234 to 470 actions, so no fixed-width
network reads all five problems, and the union envelope would cost about
6.6 MB per decision. This view is computed from
:func:`extract_canonical_features`, so the hard mask and every value still come
from the one feasibility engine, and changes five things
(``docs/campaigns/generalist_graphppo_dayahead_plan.md`` sections 4.1-4.2):

* **Variable sets.** Rows exist only for live entities: every truck, the tasks
  or revealed requests not yet served or lost, a bounded set of chargers, and
  the selectable actions. Nothing is padded here; a batch is padded to its own
  largest graph by :func:`collate`.
* **Relational identity.** No node, task, request or truck id is a feature.
  Same node, heading to and ownership are edge flags, and an action row points
  at the row of its target entity instead of carrying a node id.
* **Bounded charger subgraph.** The ``charger_budget`` chargers with the lowest
  detour from the active truck towards the remaining work, feasible ones first,
  plus every charger a truck is at or heading to. Charger destinations outside
  the subgraph are not offered; the pooled context reports what was left out.
* **Physical units.** Durations are divided by the horizon, energy by battery
  capacity and loads by payload capacity; clock times become times relative to
  now.
* **Context node.** The problem's published parameters (battery, horizon,
  uncertainty band, which components are on), so one network can tell the
  problems apart without a problem id.

Fleet-assignment features (plan section 4.2) are optional columns that stay
zero unless :attr:`ObservationSpec.fleet_assignment` is set, so arms differ in
what the rows carry and not in their width.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from EVRoutingEnv.models.core.customer import TaskStatus
from EVRoutingEnv.models.core.requests import RequestKind, RequestStatus
from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.state.feasibility import ActionKind
from EVRoutingEnv.state.features import (
    ACTION_FEATURES,
    CHARGER_FEATURES,
    CUSTOMER_FEATURES,
    GLOBAL_FEATURES,
    TRUCK_FEATURES,
    TRUCK_HOS_FEATURES,
    TRUCK_STATES,
    extract_canonical_features,
)


GENERALIST_SCHEMA_VERSION = "generalist-v1"
NODE_TYPES = ("truck", "task", "charger")
RELATION_TYPES = tuple((source, target) for source in NODE_TYPES for target in NODE_TYPES)

# Reference scales that only bring raw parameters near one; the running
# normalizer does the rest. They are constants, never fitted per problem.
_REFERENCE_HORIZON_HOURS = 72.0
_REFERENCE_BATTERY_KWH = 400.0
_REFERENCE_PAYLOAD = 20.0
_REFERENCE_FLEET = 10.0
_REFERENCE_POWER_KW = 150.0
_REFERENCE_CHARGERS = 500.0
# Stands in for an unreachable or impossible cost, in horizons.
_UNREACHABLE = 2.0

_KINDS = tuple(ActionKind)
_TASK_STATUSES = tuple(TaskStatus)
_REQUEST_KINDS = tuple(RequestKind)
_REQUEST_STATUSES = tuple(RequestStatus)
_TERMINAL_REQUESTS = {
    _REQUEST_STATUSES.index(status)
    for status in (RequestStatus.SERVED, RequestStatus.SERVED_LATE, RequestStatus.LOST)
}

TRUCK_FEATURE_NAMES = (
    *(f"state_{state}" for state in TRUCK_STATES),
    "is_active", "is_charging", "must_leave_charger", "is_complete", "failed",
    "soc", "battery_ratio", "payload_fraction", "headroom_fraction",
    "depot_energy_fraction", "depot_hours", "depot_reachable",
    "routing_elapsed", "arrival_in", "charging_elapsed", "unloading_elapsed",
    "waiting_elapsed", "ready_elapsed", "next_event_in",
    "pending_discharge_fraction", "pending_charge_fraction", "pending_service_hours",
    "queue_position", "has_port_reservation", "return_to_depot_pending",
    "completed_services_fraction",
    *TRUCK_HOS_FEATURES,
    "on_board_count", "on_board_fraction", "in_station_queue", "committed_to_charge",
)
TASK_FEATURE_NAMES = (
    "demand_fraction", "service_hours", "opens_in", "closes_in", "has_time_window",
    *(f"task_{status}" for status in _TASK_STATUSES),
    "claimed", "claimed_by_active", "claimed_elapsed",
    "preassigned", "preassigned_to_active", "sequence_position", "service_elapsed",
    "depot_energy_fraction", "depot_hours", "depot_reachable",
    *(f"kind_{kind}" for kind in _REQUEST_KINDS),
    *(f"request_{status}" for status in _REQUEST_STATUSES),
    "has_pickup", "pickup_at_depot", "pickup_opens_in", "pickup_closes_in",
    "pickup_service_hours", "released_elapsed", "delay_hours",
    # Plan 4.2, zero unless enabled.
    "active_cost", "best_other_cost", "assignment_regret", "others_closer_fraction",
)
CHARGER_FEATURE_NAMES = (
    "dc_fast", "available", "power_ratio", "port_capacity", "occupancy_fraction",
    "waitlist_length", "known_workload_hours",
    "depot_energy_fraction", "depot_hours", "depot_reachable",
    "observed_busy_ports", "observed_queue_length", "observation_age_hours",
    "observed_available_ports", "expected_wait_hours", "known_utilization_prior",
    "occupancy_indicator",
    "detour_hours", "feasible_target", "at_active_truck",
)
ACTION_FEATURE_NAMES = (
    *(f"kind_{kind}" for kind in _KINDS),
    "charge_value", "demand_fraction", "required_energy_fraction",
    "leg_hours", "leg_reachable", "target_depot_hours", "target_pending_min_hours",
    "target_pending_mean_hours", "insertion_detour_hours", "target_is_depot",
    # Plan 4.2, zero unless enabled.
    "active_cost", "best_other_cost", "assignment_regret",
    "unclaimed_demand_fraction", "others_can_cover",
)
CONTEXT_FEATURE_NAMES = (
    # Published problem parameters, constant within an episode.
    "horizon_ratio", "battery_ratio", "payload_ratio", "fleet_ratio",
    "charger_count_log", "mean_power_ratio", "traffic_sigma", "rush_multiplier",
    "energy_min_multiplier", "energy_max_multiplier", "base_service_hours",
    "hos_enabled", "hos_rest_anywhere", "public_charging", "utilization",
    "outages", "port_availability", "depot_charging", "dynamic_requests",
    "mode_static", "mode_same_day", "mode_drayage", "hard_deadline",
    "nominal_eta", "target_soc_actions",
    # Progress.
    "clock", "remaining", "step_fraction", "trucks_working_fraction",
    "tasks_open_fraction", "tasks_claimed_fraction", "tasks_in_service_fraction",
    "tasks_served_fraction", "served_on_time_fraction", "served_late_fraction",
    "lost_fraction",
    # What the charger subgraph left out.
    "chargers_excluded_fraction", "feasible_chargers_excluded_fraction",
    "excluded_min_detour_hours",
)
EDGE_FEATURE_NAMES = (
    "energy_fraction", "hours", "reachable", "same_node", "heading_to", "owns",
)
# Hindsight summary of the hidden request stream (plan 9.4, asymmetric critic).
# Aggregates only, so the width never depends on a problem's size; the critic
# alone may read it and the actor path never does. Kept out of FEATURE_DIMS so
# the encoder's state dict is unchanged when the privileged critic is off.
_PRIVILEGED_BUCKETS = 4
PRIVILEGED_FEATURE_NAMES = (
    "hidden_fraction", "hidden_count", "hidden_demand_fraction",
    "next_reveal_in", "mean_release_lag", "mean_deadline_window", "hard_deadline_fraction",
    *(
        f"reveal_bucket{bucket}_{stat}"
        for bucket in range(_PRIVILEGED_BUCKETS)
        for stat in ("count", "demand_fraction", "urgency", "service_hours", "off_depot_pickup")
    ),
)
PRIVILEGED_DIM = len(PRIVILEGED_FEATURE_NAMES)
_REFERENCE_STREAM = 64.0

# Plan-conditioned input (plan 9.5): the rolling planner's live fleet plan as
# advisory per-row features on the dynamic problems. Kept out of FEATURE_DIMS
# so the encoder's state dict is unchanged when plan conditioning is off; the
# policy consumes these rows through gated, bias-free projections, so all-zero
# rows (static problems, dropped episodes, flag off) inject exactly nothing.
PLAN_TASK_FEATURE_NAMES = ("plan_next_active", "plan_next_any", "plan_rank_active")
PLAN_TRUCK_FEATURE_NAMES = ("plan_has_route",)
PLAN_ACTION_FEATURE_NAMES = ("plan_on_next",)
PLAN_TASK_DIM = len(PLAN_TASK_FEATURE_NAMES)
PLAN_TRUCK_DIM = len(PLAN_TRUCK_FEATURE_NAMES)
PLAN_ACTION_DIM = len(PLAN_ACTION_FEATURE_NAMES)

FEATURE_DIMS = {
    "truck": len(TRUCK_FEATURE_NAMES),
    "task": len(TASK_FEATURE_NAMES),
    "charger": len(CHARGER_FEATURE_NAMES),
    "action": len(ACTION_FEATURE_NAMES),
    "context": len(CONTEXT_FEATURE_NAMES),
    "edge": len(EDGE_FEATURE_NAMES),
}

_T = {name: index for index, name in enumerate(TRUCK_FEATURES)}
_C = {name: index for index, name in enumerate(CUSTOMER_FEATURES)}
_K = {name: index for index, name in enumerate(CHARGER_FEATURES)}
_A = {name: index for index, name in enumerate(ACTION_FEATURES)}
_G = {name: index for index, name in enumerate(GLOBAL_FEATURES)}
_HOS_HOURS = {
    "drive_left_before_break", "drive_left_in_shift", "window_left_hours", "nondriving_streak_hours",
}
_ROUTING_HOURS = (
    "leg_travel_hours", "target_depot_hours", "target_pending_min_hours",
    "target_pending_mean_hours", "insertion_detour_hours",
)


@dataclass(frozen=True)
class ObservationSpec:
    """The knobs of the view; everything else is fixed by the schema."""

    charger_budget: int = 32
    fleet_assignment: bool = False
    # Attach the hindsight stream summary for an asymmetric critic (plan 9.4).
    privileged: bool = False
    # Attach the rolling planner's live plan on dynamic problems (plan 9.5).
    plan_conditioning: bool = False
    # Fraction of episodes whose plan features are zeroed during training, so
    # the plan-free mode keeps working; evaluation uses the default 0.
    plan_dropout: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.charger_budget, bool) or not isinstance(self.charger_budget, int) or self.charger_budget < 1:
            raise ValueError("charger_budget must be a positive integer")
        if not 0.0 <= float(self.plan_dropout) < 1.0:
            raise ValueError("plan_dropout must be in [0, 1)")


@dataclass
class GraphObservation:
    """One decision epoch as variable-size typed rows (numpy, picklable).

    ``edges`` is one square block over ``trucks + tasks + chargers`` in that
    order, stored as float16 so that rollouts stay small; the same rounded
    values are seen when acting and when re-scoring, so PPO ratios start at one.
    """

    trucks: np.ndarray
    tasks: np.ndarray
    chargers: np.ndarray
    context: np.ndarray
    edges: np.ndarray
    actions: np.ndarray
    action_index: np.ndarray
    action_task: np.ndarray
    action_charger: np.ndarray
    active_truck: int
    # Critic-only hindsight summary (plan 9.4); ``None`` unless the spec asks.
    privileged: np.ndarray | None = None
    # Plan-conditioned rows (plan 9.5); ``None`` unless the spec asks. All-zero
    # rows mean "no plan" (static problem, dropped episode, or empty plan).
    plan_tasks: np.ndarray | None = None
    plan_trucks: np.ndarray | None = None
    plan_actions: np.ndarray | None = None

    @property
    def num_actions(self) -> int:
        return int(self.action_index.shape[0])

    def candidate(self, env_action: int) -> int | None:
        """Position of an environment action among the offered candidates."""
        hits = np.flatnonzero(self.action_index == int(env_action))
        return int(hits[0]) if hits.size else None


class GeneralistEnv(EventDrivenTruckEnv):
    """The simulator, observed through :func:`build_observation`.

    Only the observation changes: dynamics, masks, rewards and ``info`` are
    the parent's. Building the canonical flat vector would cost a second full
    feature extraction per step and is skipped.
    """

    def __init__(self, config, spec: ObservationSpec | None = None, **kwargs):
        self.observation_spec = spec or ObservationSpec()
        kwargs.setdefault("verbose", False)
        kwargs.setdefault("enable_plotting", False)
        super().__init__(config, **kwargs)

    def _get_observation(self) -> GraphObservation:
        return build_observation(self, self.observation_spec)


def build_observation(env, spec: ObservationSpec, features=None) -> GraphObservation:
    """Build the generalist view of the environment's current state."""
    features = extract_canonical_features(env) if features is None else features
    now = float(env.global_clock)
    horizon = max(float(env.max_time), 1e-6)
    index, energy, hours = _cost_matrices(env.transport_graph)
    depot = int(env.joint_instance.depot_node)

    truck_rows = features.truck_features
    truck_ids = sorted(int(truck.truck_id) for truck in env.trucks)
    trucks_by_id = {int(truck.truck_id): truck for truck in env.trucks}
    active_id = env.active_truck_id
    active_row = truck_ids.index(int(active_id)) if active_id is not None else -1
    battery = max(float(truck_rows[:, _T["battery_capacity_kwh"]].max()), 1e-6)
    payload = float(truck_rows[:, _T["payload_capacity"]].max())
    payload = payload if payload > 0 else 1.0
    origin = int(trucks_by_id[int(active_id)].current_node) if active_id is not None else depot

    # Tasks: static customers not yet served, or revealed requests not yet
    # served or lost.
    customer_rows = features.customer_features
    dynamic = bool(getattr(env, "dynamic_requests", False))
    if dynamic:
        keep = ~np.isin(
            customer_rows[:, _C["request_status_code"]].astype(int), list(_TERMINAL_REQUESTS)
        )
    else:
        keep = customer_rows[:, _C["status_code"]].astype(int) != _TASK_STATUSES.index(TaskStatus.SERVED)
    tasks = customer_rows[keep]
    task_nodes = np.asarray([_next_stop(row) for row in tasks], dtype=np.int64)
    task_owners = tasks[:, _C["claimed_by"]].astype(np.int64) if len(tasks) else np.zeros(0, np.int64)
    slot_of_row = _task_slots(env, tasks, dynamic)

    # Charger subgraph.
    action_rows = features.action_features
    charger_rows = features.charger_features
    charger_nodes = charger_rows[:, _K["node_id"]].astype(np.int64)
    kinds = action_rows[:, _A["kind_code"]].astype(int)
    feasible = action_rows[:, _A["feasible"]] > 0.5
    charger_kind = _KINDS.index(ActionKind.CHARGER)
    feasible_targets = set(
        action_rows[feasible & (kinds == charger_kind), _A["target_node"]].astype(int).tolist()
    )
    # Work the active truck could go to next: tasks nobody else has claimed,
    # its own tasks, and the depot.
    mine = (task_owners < 0) | (task_owners == (-2 if active_id is None else int(active_id)))
    goals = np.unique(np.concatenate([task_nodes[mine], [depot]]))
    order, detour = _charger_order(
        index, hours, origin, charger_nodes, goals,
        np.asarray([int(node) in feasible_targets for node in charger_nodes.tolist()], dtype=bool),
        spec.charger_budget,
    )
    truck_nodes = {int(truck.current_node) for truck in env.trucks}
    truck_nodes |= {int(truck.route_destination) for truck in env.trucks if truck.route_destination is not None}
    chosen = set(order)
    chosen |= {row for row, node in enumerate(charger_nodes.tolist()) if node in truck_nodes}
    selected = sorted(chosen, key=lambda row: (detour[row], int(charger_nodes[row])))
    excluded = [row for row in range(len(charger_nodes)) if row not in chosen]
    charger_row_of_node = {int(charger_nodes[row]): position for position, row in enumerate(selected)}

    # Fleet assignment (plan 4.2) from observable commitments only.
    assignment = _fleet_assignment(
        env, spec, truck_ids, trucks_by_id, truck_rows, active_id, tasks, task_nodes,
        index, hours, now, horizon, payload,
    )

    truck_features = np.stack([
        _truck_row(row, now, horizon, payload, len(customer_rows))
        for row in truck_rows
    ]).astype(np.float32)
    task_features = np.zeros((len(tasks), FEATURE_DIMS["task"]), dtype=np.float32)
    for position, row in enumerate(tasks):
        task_features[position] = _task_row(
            row, now, horizon, battery, payload, active_id, depot, assignment["task"][position],
        )
    charger_features = np.zeros((len(selected), FEATURE_DIMS["charger"]), dtype=np.float32)
    for position, row in enumerate(selected):
        charger_features[position] = _charger_row(
            charger_rows[row], horizon, battery, detour[row],
            int(charger_nodes[row]) in feasible_targets, int(charger_nodes[row]) == origin,
        )

    # Selectable actions: the hard mask, minus charger destinations outside the subgraph.
    positions, task_links, charger_links, action_features = [], [], [], []
    row_of_slot = {slot: position for position, slot in enumerate(slot_of_row)}
    for action, row in enumerate(action_rows):
        if not feasible[action]:
            continue
        kind = _KINDS[kinds[action]]
        target = int(row[_A["target_node"]])
        task_link, charger_link = -1, -1
        if kind is ActionKind.CHARGER:
            if target not in charger_row_of_node:
                continue
            charger_link = charger_row_of_node[target]
        elif kind is ActionKind.CUSTOMER:
            task_link = row_of_slot.get(action - env.num_charging_nodes, -1)
        elif kind in (ActionKind.CHARGE, ActionKind.QUEUE):
            charger_link = charger_row_of_node.get(target, -1)
        positions.append(action)
        task_links.append(task_link)
        charger_links.append(charger_link)
        action_features.append(_action_row(
            row, kind, horizon, battery, payload, target == depot,
            assignment["task"][task_link] if task_link >= 0 else None, assignment["depot"],
        ))

    # Plan-conditioned rows (plan 9.5): the rolling planner's cached fleet plan
    # on dynamic problems; all-zero on static problems and dropped episodes.
    plan_tasks = plan_trucks = plan_actions = None
    if spec.plan_conditioning:
        plan_tasks = np.zeros((len(tasks), PLAN_TASK_DIM), dtype=np.float32)
        plan_trucks = np.zeros((len(truck_ids), PLAN_TRUCK_DIM), dtype=np.float32)
        plan_actions = np.zeros((len(positions), PLAN_ACTION_DIM), dtype=np.float32)
        if dynamic and len(tasks):
            from algo.generalist_plan import tracker_for

            routes = tracker_for(env).routes_for(
                env, dropout=spec.plan_dropout,
                open_rids={int(slot) for slot in slot_of_row}, active_id=active_id,
            )
            next_active_row = -1
            for truck_position, truck_id in enumerate(truck_ids):
                route = [rid for rid in routes.get(truck_id, ()) if rid in row_of_slot]
                if not route:
                    continue
                plan_trucks[truck_position, 0] = 1.0
                plan_tasks[row_of_slot[route[0]], 1] = 1.0  # plan_next_any
                if active_id is not None and truck_id == int(active_id):
                    next_active_row = row_of_slot[route[0]]
                    plan_tasks[next_active_row, 0] = 1.0  # plan_next_active
                    for rank, rid in enumerate(route):
                        plan_tasks[row_of_slot[rid], 2] = 1.0 / (1.0 + rank)  # plan_rank_active
            if next_active_row >= 0:
                for candidate, row in enumerate(task_links):
                    if row == next_active_row:
                        plan_actions[candidate, 0] = 1.0

    context = _context_row(
        env, features.global_features, now, horizon, battery, payload,
        excluded, feasible_targets, charger_nodes, detour, dynamic,
    )
    edges = _edges(
        index, energy, hours, battery, horizon,
        [int(trucks_by_id[truck_id].current_node) for truck_id in truck_ids],
        [_optional_node(trucks_by_id[truck_id].route_destination) for truck_id in truck_ids],
        truck_ids, task_nodes, task_owners, charger_nodes[selected],
    )
    return GraphObservation(
        trucks=truck_features,
        tasks=task_features,
        chargers=charger_features,
        context=context,
        edges=edges,
        actions=np.asarray(action_features, dtype=np.float32).reshape(-1, FEATURE_DIMS["action"]),
        action_index=np.asarray(positions, dtype=np.int64),
        action_task=np.asarray(task_links, dtype=np.int64),
        action_charger=np.asarray(charger_links, dtype=np.int64),
        active_truck=active_row,
        privileged=privileged_summary(env, now, horizon, payload) if spec.privileged else None,
        plan_tasks=plan_tasks,
        plan_trucks=plan_trucks,
        plan_actions=plan_actions,
    )


def privileged_summary(env, now: float, horizon: float, payload: float) -> np.ndarray:
    """Fixed-width hindsight summary of the hidden request stream (plan 9.4).

    Aggregates only - counts, demand, reveal timing and windows over
    ``_PRIVILEGED_BUCKETS`` equal spans of the remaining horizon - never an
    identifier, so permutation and padding invariance are untouched. The vector
    is for the value head alone: feeding it to the actor would break causality.
    Static problems have no hidden stream and return zeros.
    """
    values = np.zeros(PRIVILEGED_DIM, dtype=np.float32)
    engine = getattr(env, "request_engine", None)
    if engine is None or not getattr(env, "dynamic_requests", False):
        return values
    requests = engine.registry.requests()
    hidden = [request for request in requests if request.status is RequestStatus.HIDDEN]
    if not hidden:
        return values
    capacity = max(payload * max(1, len(env.trucks)), 1e-6)
    depot = int(engine.depot)
    values[0] = len(hidden) / max(1, len(requests))
    values[1] = len(hidden) / _REFERENCE_STREAM
    values[2] = sum(request.demand for request in hidden) / capacity
    values[3] = max(0.0, min(request.reveal_time for request in hidden) - now) / horizon
    values[4] = float(np.mean([
        max(0.0, request.release_time - request.reveal_time) for request in hidden
    ])) / horizon
    bounded = [request for request in hidden if math.isfinite(request.delivery.latest)]
    values[5] = float(np.mean([
        min(request.delivery.latest - request.delivery.earliest, horizon) for request in bounded
    ])) / horizon if bounded else -1.0
    values[6] = len(bounded) / len(hidden)
    span = max(horizon - now, 1e-6) / _PRIVILEGED_BUCKETS
    buckets: list[list] = [[] for _ in range(_PRIVILEGED_BUCKETS)]
    for request in hidden:
        position = int(max(0.0, request.reveal_time - now) / span)
        buckets[min(position, _PRIVILEGED_BUCKETS - 1)].append(request)
    for bucket, members in enumerate(buckets):
        base = 7 + 5 * bucket
        if not members:
            values[base + 2] = values[base + 3] = -1.0
            continue
        values[base] = len(members) / (_REFERENCE_STREAM / _PRIVILEGED_BUCKETS)
        values[base + 1] = sum(request.demand for request in members) / capacity
        values[base + 2] = float(np.mean([
            min(request.delivery.latest - now, _UNREACHABLE * horizon) for request in members
        ])) / horizon
        values[base + 3] = float(np.mean([
            (request.pickup.service_hours if request.pickup is not None else 0.0)
            + request.delivery.service_hours
            for request in members
        ])) / horizon
        values[base + 4] = float(np.mean([
            request.pickup is not None and int(request.pickup.node) != depot for request in members
        ]))
    return values


def _cost_matrices(graph):
    """All-pairs nominal (index, energy, hours) with ``inf`` where unreachable."""
    matrices = getattr(graph, "cost_matrices", None)
    if callable(matrices):
        return matrices()
    position, dense = graph.dense_transport_matrix()
    reachable = dense[:, :, 2] > 0
    return (
        position,
        np.where(reachable, dense[:, :, 0], np.inf),
        np.where(reachable, dense[:, :, 1], np.inf),
    )


def _next_stop(row) -> int:
    """Where serving a task takes a truck next: its pickup until it is loaded."""
    status = int(row[_C["request_status_code"]])
    if row[_C["has_pickup"]] > 0.5 and status == _REQUEST_STATUSES.index(RequestStatus.OPEN):
        return int(row[_C["pickup_node"]])
    return int(row[_C["node_id"]])


def _task_slots(env, tasks, dynamic: bool) -> list[int]:
    """The customer action slot of each kept task row."""
    ids = tasks[:, _C["task_id"]].astype(int).tolist() if len(tasks) else []
    if dynamic:
        return ids  # slot r is request r
    offset = {int(task.task_id): slot for slot, task in enumerate(env.joint_instance.tasks)}
    return [offset[task_id] for task_id in ids]


def _lookup(index, matrix, sources, targets) -> np.ndarray:
    rows = np.asarray([index.get(int(node), -1) for node in sources], dtype=np.int64)
    columns = np.asarray([index.get(int(node), -1) for node in targets], dtype=np.int64)
    values = np.full((rows.size, columns.size), np.inf)
    valid = (rows[:, None] >= 0) & (columns[None, :] >= 0)
    if rows.size and columns.size:
        values[valid] = matrix[np.maximum(rows, 0)[:, None], np.maximum(columns, 0)[None, :]][valid]
    return values


def _charger_order(index, hours, origin, chargers, goals, feasible, budget) -> tuple[list[int], np.ndarray]:
    """The first ``budget`` chargers for the subgraph, and every charger's detour in hours.

    Every goal, nearest first, takes turns with two rankings: the cheapest stop
    on the way, ``hours(here, c) + hours(c, goal)``, and the longest hop, the
    charger closest to the goal. A feasible destination always precedes an
    infeasible one. A single ranking by detour to any goal spends the budget on
    chargers near work this truck is not heading to: on 677 greedy-teacher
    charger choices (Texas and Pacific Northwest long haul) it kept 84 % at a
    budget of 32, this order kept all of them. The detour can be negative:
    network hours come from minimum-energy paths, so passing a charger is
    sometimes faster.
    """
    if not len(chargers):
        return [], np.zeros(0)
    to_charger = _lookup(index, hours, [origin], chargers)[0]
    direct = _lookup(index, hours, [origin], goals)[0]
    onward = _lookup(index, hours, chargers, goals)
    via = to_charger[:, None] + onward
    detour = np.where(np.isfinite(via), via - direct[None, :], np.inf).min(axis=1)
    detour = np.where(np.isfinite(detour), detour, np.where(np.isfinite(to_charger), to_charger, 1e9))
    rankings = [
        np.lexsort((chargers, np.where(np.isfinite(cost[:, goal]), cost[:, goal], 1e9), ~feasible))
        for goal in np.argsort(np.where(np.isfinite(direct), direct, 1e9), kind="stable")
        for cost in (via, onward)
    ]
    order, seen = [], set()
    for depth in range(len(chargers)):
        for ranking in rankings:
            row = int(ranking[depth])
            if row not in seen:
                seen.add(row)
                order.append(row)
                if len(order) == budget:
                    return order, detour
    return order, detour


def _optional_node(node) -> int:
    return -1 if node is None else int(node)


def _since(value: float, now: float, horizon: float) -> float:
    return -1.0 if value < 0 else (now - value) / horizon


def _until(value: float, now: float, horizon: float) -> float:
    return -1.0 if value < 0 else (value - now) / horizon


def _hours(value: float, horizon: float) -> float:
    return value / horizon if value >= 0 else -1.0


def _one_hot(code: int, width: int) -> list[float]:
    values = [0.0] * width
    if 0 <= code < width:
        values[code] = 1.0
    return values


def _truck_row(row, now, horizon, payload, total_tasks) -> list[float]:
    capacity = max(float(row[_T["battery_capacity_kwh"]]), 1e-6)
    truck_payload = float(row[_T["payload_capacity"]])
    truck_payload = truck_payload if truck_payload > 0 else payload
    remaining = float(row[_T["remaining_payload"]])
    return [
        *_one_hot(int(row[_T["state_code"]]), len(TRUCK_STATES)),
        row[_T["is_active"]], row[_T["is_charging"]], row[_T["must_leave_charger"]],
        row[_T["is_complete"]], row[_T["failed"]],
        row[_T["battery_kwh"]] / capacity,
        capacity / _REFERENCE_BATTERY_KWH,
        remaining / truck_payload if remaining >= 0 else -1.0,
        row[_T["battery_minus_depot_energy"]] / capacity,
        row[_T["depot_energy_kwh"]] / capacity,
        row[_T["depot_travel_hours"]] / horizon,
        row[_T["depot_reachable"]],
        _since(row[_T["routing_started_at"]], now, horizon),
        _until(row[_T["route_arrival_at"]], now, horizon),
        _since(row[_T["charging_started_at"]], now, horizon),
        _since(row[_T["unloading_started_at"]], now, horizon),
        _since(row[_T["waiting_started_at"]], now, horizon),
        _since(row[_T["ready_since"]], now, horizon),
        _hours(row[_T["next_event_remaining_hours"]], horizon),
        row[_T["pending_discharge_kwh"]] / capacity,
        row[_T["pending_charge_kwh"]] / capacity,
        row[_T["pending_service_hours"]] / horizon,
        row[_T["queue_position"]], row[_T["has_port_reservation"]],
        row[_T["return_to_depot_pending"]],
        row[_T["completed_services"]] / max(1, total_tasks),
        *(_hours(row[_T[name]], horizon) if name in _HOS_HOURS else row[_T[name]]
          for name in TRUCK_HOS_FEATURES),
        row[_T["on_board_count"]], row[_T["on_board_demand"]] / truck_payload,
        row[_T["in_station_queue"]], row[_T["committed_to_charge"]],
    ]


def _task_row(row, now, horizon, battery, payload, active_id, depot, assignment) -> list[float]:
    owner = int(row[_C["claimed_by"]])
    preassigned = int(row[_C["preassigned_to"]])
    active = -2 if active_id is None else int(active_id)
    has_pickup = row[_C["has_pickup"]] > 0.5
    status = int(row[_C["request_status_code"]])
    return [
        row[_C["demand"]] / payload,
        row[_C["base_service_time"]] / horizon,
        (row[_C["earliest_service"]] - now) / horizon,
        (row[_C["latest_service"]] - now) / horizon,
        row[_C["has_time_window"]],
        *_one_hot(int(row[_C["status_code"]]), len(_TASK_STATUSES)),
        float(owner >= 0), float(owner == active), _since(row[_C["claimed_at"]], now, horizon),
        float(preassigned >= 0), float(preassigned == active), row[_C["sequence_position"]],
        _since(row[_C["service_started_at"]], now, horizon),
        row[_C["depot_energy_kwh"]] / battery,
        row[_C["depot_travel_hours"]] / horizon,
        row[_C["depot_reachable"]],
        *_one_hot(int(row[_C["request_kind_code"]]), len(_REQUEST_KINDS)),
        *_one_hot(status, len(_REQUEST_STATUSES)),
        float(has_pickup),
        float(has_pickup and int(row[_C["pickup_node"]]) == depot),
        (row[_C["pickup_earliest"]] - now) / horizon if has_pickup else -1.0,
        (row[_C["pickup_latest"]] - now) / horizon if has_pickup else -1.0,
        row[_C["pickup_service_hours"]] / horizon if has_pickup else -1.0,
        (now - row[_C["release_time"]]) / horizon,
        row[_C["delay_hours"]] / horizon,
        *assignment,
    ]


def _charger_row(row, horizon, battery, detour, feasible_target, at_active) -> list[float]:
    ports = max(float(row[_K["port_capacity"]]), 1.0)
    return [
        row[_K["type_code"]], row[_K["available"]],
        row[_K["power_kw"]] / _REFERENCE_POWER_KW,
        row[_K["port_capacity"]],
        row[_K["occupancy"]] / ports,
        row[_K["waitlist_length"]],
        row[_K["known_workload_hours"]] / horizon,
        row[_K["depot_energy_kwh"]] / battery,
        row[_K["depot_travel_hours"]] / horizon,
        row[_K["depot_reachable"]],
        row[_K["observed_busy_ports"]], row[_K["observed_queue_length"]],
        _hours(row[_K["observation_age_hours"]], horizon),
        row[_K["observed_available_ports"]],
        _hours(row[_K["expected_wait_given_memory"]], horizon),
        row[_K["known_utilization_prior"]], row[_K["occupancy_indicator"]],
        min(float(detour), _UNREACHABLE * horizon) / horizon,
        float(feasible_target), float(at_active),
    ]


def _action_row(row, kind, horizon, battery, payload, to_depot, task_assignment, depot_assignment) -> list[float]:
    required = float(row[_A["required_energy"]])
    if kind is ActionKind.CUSTOMER and task_assignment is not None:
        assignment = [*task_assignment[:3], 0.0, 0.0]
    elif kind in (ActionKind.DEPOT, ActionKind.END_SHIFT):
        assignment = [0.0, 0.0, 0.0, *depot_assignment]
    else:
        assignment = [0.0] * 5
    return [
        *_one_hot(_KINDS.index(kind), len(_KINDS)),
        row[_A["charge_value"]],
        row[_A["customer_demand"]] / payload,
        required / battery if required >= 0 else -1.0,
        row[_A["leg_travel_hours"]] / horizon,
        row[_A["leg_reachable"]],
        *(row[_A[name]] / horizon for name in _ROUTING_HOURS[1:]),
        float(to_depot and kind is ActionKind.DEPOT),
        *assignment,
    ]


def _fleet_assignment(
    env, spec, truck_ids, trucks_by_id, truck_rows, active_id, tasks, task_nodes,
    index, hours, now, horizon, payload,
) -> dict:
    """Marginal costs of serving each task by the active truck and by the others.

    A truck's availability is its committed next event (an ETA under
    ``nominal_eta``), and it continues from where it is heading. Payload
    feasibility uses each truck's remaining payload. Nothing here reads an
    unrealized draw.
    """
    count = len(tasks)
    zeros = {"task": np.zeros((count, 4)), "depot": [0.0, 0.0]}
    if not spec.fleet_assignment or active_id is None:
        return zeros
    demands = tasks[:, _C["demand"]] if count else np.zeros(0)
    origins, ready, room, others = [], [], [], []
    for row_index, truck_id in enumerate(truck_ids):
        truck = trucks_by_id[truck_id]
        if truck.is_complete or truck.failed:
            continue
        busy = env.truck_states[truck_id] == "routing" and truck.route_destination is not None
        origins.append(int(truck.route_destination) if busy else int(truck.current_node))
        event = float(truck_rows[row_index, _T["next_event_at"]])
        ready.append(max(0.0, event - now) if event >= 0 else 0.0)
        remaining = truck.remaining_payload
        room.append(float(payload if remaining is None else remaining))
        others.append(truck_id != int(active_id))
    if not count:
        return zeros
    legs = _lookup(index, hours, origins, task_nodes)
    costs = np.asarray(ready)[:, None] + legs
    fits = np.asarray(room)[:, None] + 1e-9 >= demands[None, :]
    is_other = np.asarray(others)
    active_position = int(np.flatnonzero(~is_other)[0]) if (~is_other).any() else None
    cap = _UNREACHABLE * horizon
    active_cost = (
        np.minimum(legs[active_position], cap) if active_position is not None else np.full(count, cap)
    )
    other_costs = np.where(is_other[:, None] & fits, costs, np.inf)
    best_other = np.minimum(other_costs.min(axis=0) if len(origins) else np.full(count, np.inf), cap)
    closer = (other_costs < active_cost[None, :]).sum(axis=0) / max(1, int(is_other.sum()))
    task_columns = np.stack(
        [active_cost / horizon, best_other / horizon, (best_other - active_cost) / horizon, closer], axis=1
    )
    open_codes = {_REQUEST_STATUSES.index(RequestStatus.OPEN)}
    unassigned = tasks[:, _C["claimed_by"]] < 0
    if getattr(env, "dynamic_requests", False):
        unassigned &= np.isin(tasks[:, _C["request_status_code"]].astype(int), list(open_codes))
    unclaimed = float(demands[unassigned].sum())
    other_room = float(np.asarray(room)[is_other].sum())
    return {
        "task": task_columns,
        "depot": [unclaimed / payload, float(other_room + 1e-9 >= unclaimed)],
    }


def _context_row(
    env, global_row, now, horizon, battery, payload, excluded, feasible_targets,
    charger_nodes, detour, dynamic,
) -> np.ndarray:
    config = env.config
    traffic = config.get("traffic", {})
    hos = config.get("hours_of_service") or {}
    public = config.get("public_charging") or {}
    requests = config.get("requests") or {}
    mode = requests.get("mode") if dynamic else "static"
    outages = public.get("outages") or {}
    powers = list(env.charging_station.charger_power_kw.values()) or [0.0]
    trucks = list(env.trucks)
    working = sum(not (truck.is_complete or truck.failed) for truck in trucks)
    total = max(1.0, float(global_row[_G["customers_total"]]))
    revealed = max(1.0, float(global_row[_G["requests_revealed"]]))
    excluded_feasible = sum(int(charger_nodes[row]) in feasible_targets for row in excluded)
    excluded_detour = min((detour[row] for row in excluded), default=0.0)
    return np.asarray([
        horizon / _REFERENCE_HORIZON_HOURS,
        battery / _REFERENCE_BATTERY_KWH,
        payload / _REFERENCE_PAYLOAD,
        len(trucks) / _REFERENCE_FLEET,
        math.log1p(len(charger_nodes)) / math.log1p(_REFERENCE_CHARGERS),
        float(np.mean(powers)) / _REFERENCE_POWER_KW,
        float(traffic.get("std_dev_factor", 0.0)) if traffic.get("enable_traffic") else 0.0,
        float(traffic.get("rush_hour_multiplier", 1.0)) if traffic.get("enable_traffic") else 1.0,
        float(traffic.get("min_energy_multiplier", 1.0)) if traffic.get("enable_energy_uncertainty") else 1.0,
        float(traffic.get("max_energy_multiplier", 1.0)) if traffic.get("enable_energy_uncertainty") else 1.0,
        float(config.get("problem", {}).get("base_service_time", 0.0)) / horizon,
        float(bool(hos.get("enabled"))),
        float(bool(hos.get("enabled")) and hos.get("rest_locations", "anywhere") == "anywhere"),
        float(bool(public.get("enabled"))),
        float((public.get("exogenous_traffic") or {}).get("utilization", 0.0)) if public.get("enabled") else 0.0,
        float(bool(outages.get("enabled"))),
        float(outages.get("port_availability", 1.0)) if outages.get("enabled") else 1.0,
        float(bool((config.get("depot_charging") or {}).get("enabled"))),
        float(dynamic),
        float(mode == "static"), float(mode == "same_day"), float(mode == "drayage"),
        float(dynamic and requests.get("deadline_type") == "hard"),
        float((config.get("observation") or {}).get("in_flight_information") == "nominal_eta"),
        float(env.charging_action_mode == "target_soc"),
        now / horizon,
        max(0.0, horizon - now) / horizon,
        env.episode_steps / max(1, env.max_episode_steps),
        working / max(1, len(trucks)),
        global_row[_G["customers_unassigned"]] / total,
        global_row[_G["customers_claimed"]] / total,
        global_row[_G["customers_in_service"]] / total,
        global_row[_G["customers_served"]] / total,
        global_row[_G["requests_served_on_time"]] / revealed,
        global_row[_G["requests_served_late"]] / revealed,
        global_row[_G["requests_lost"]] / revealed,
        len(excluded) / max(1, len(charger_nodes)),
        excluded_feasible / max(1, len(feasible_targets)),
        min(float(excluded_detour), _UNREACHABLE * horizon) / horizon,
    ], dtype=np.float32)


def _edges(
    index, energy, hours, battery, horizon, truck_nodes, truck_destinations, truck_ids,
    task_nodes, task_owners, charger_nodes,
) -> np.ndarray:
    nodes = np.concatenate([
        np.asarray(truck_nodes, dtype=np.int64), task_nodes, np.asarray(charger_nodes, dtype=np.int64),
    ])
    trucks, tasks = len(truck_nodes), len(task_nodes)
    energy_block = _lookup(index, energy, nodes, nodes)
    hours_block = _lookup(index, hours, nodes, nodes)
    reachable = np.isfinite(energy_block) & np.isfinite(hours_block)
    edges = np.zeros((nodes.size, nodes.size, FEATURE_DIMS["edge"]), dtype=np.float32)
    edges[..., 0] = np.where(reachable, energy_block / battery, 0.0)
    edges[..., 1] = np.where(reachable, hours_block / horizon, 0.0)
    edges[..., 2] = reachable
    edges[..., 3] = nodes[:, None] == nodes[None, :]
    destinations = np.asarray(truck_destinations, dtype=np.int64)
    heading = (destinations[:, None] == nodes[None, :]) & (destinations[:, None] >= 0)
    edges[:trucks, :, 4] = heading
    edges[:, :trucks, 4] = np.maximum(edges[:, :trucks, 4], heading.T)
    owns = np.asarray(truck_ids, dtype=np.int64)[:, None] == task_owners[None, :]
    edges[:trucks, trucks:trucks + tasks, 5] = owns
    edges[trucks:trucks + tasks, :trucks, 5] = owns.T
    return edges.astype(np.float16)


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GraphBatch:
    """Observations padded to the largest graph in the batch, with masks."""

    nodes: dict[str, torch.Tensor]
    node_masks: dict[str, torch.Tensor]
    pairwise: dict[tuple[str, str], torch.Tensor]
    pairwise_mask: dict[tuple[str, str], torch.Tensor]
    context: torch.Tensor
    actions: torch.Tensor
    action_mask: torch.Tensor
    action_task: torch.Tensor
    action_charger: torch.Tensor
    active_truck: torch.Tensor
    # Critic-only hindsight rows; zeros wherever an observation carried none.
    privileged: torch.Tensor | None = None
    # Plan-conditioned rows (plan 9.5); present only when the spec asks.
    plan_tasks: torch.Tensor | None = None
    plan_trucks: torch.Tensor | None = None
    plan_actions: torch.Tensor | None = None

    @property
    def batch_size(self) -> int:
        return int(self.context.shape[0])

    def select(self, rows: torch.Tensor) -> GraphBatch:
        """The sub-batch at ``rows``; padding widths are kept."""
        return GraphBatch(
            nodes={key: value[rows] for key, value in self.nodes.items()},
            node_masks={key: value[rows] for key, value in self.node_masks.items()},
            pairwise={key: value[rows] for key, value in self.pairwise.items()},
            pairwise_mask={key: value[rows] for key, value in self.pairwise_mask.items()},
            context=self.context[rows],
            actions=self.actions[rows],
            action_mask=self.action_mask[rows],
            action_task=self.action_task[rows],
            action_charger=self.action_charger[rows],
            active_truck=self.active_truck[rows],
            privileged=None if self.privileged is None else self.privileged[rows],
            plan_tasks=None if self.plan_tasks is None else self.plan_tasks[rows],
            plan_trucks=None if self.plan_trucks is None else self.plan_trucks[rows],
            plan_actions=None if self.plan_actions is None else self.plan_actions[rows],
        )


def collate(observations: list[GraphObservation], device: str | torch.device = "cpu") -> GraphBatch:
    """Pad a list of observations to their largest member, per entity type."""
    if not observations:
        raise ValueError("cannot collate an empty batch")
    batch = len(observations)
    counts = {
        "truck": [len(item.trucks) for item in observations],
        "task": [len(item.tasks) for item in observations],
        "charger": [len(item.chargers) for item in observations],
    }
    widths = {key: max(max(values), 1) for key, values in counts.items()}
    nodes, masks = {}, {}
    for key, attribute in (("truck", "trucks"), ("task", "tasks"), ("charger", "chargers")):
        values = np.zeros((batch, widths[key], FEATURE_DIMS[key]), dtype=np.float32)
        mask = np.zeros((batch, widths[key]), dtype=bool)
        for position, item in enumerate(observations):
            rows = getattr(item, attribute)
            values[position, : len(rows)] = rows
            mask[position, : len(rows)] = True
        nodes[key], masks[key] = values, mask

    pairwise = {
        relation: np.zeros((batch, widths[relation[0]], widths[relation[1]], FEATURE_DIMS["edge"]), dtype=np.float16)
        for relation in RELATION_TYPES
    }
    for position, item in enumerate(observations):
        bounds = np.cumsum([0, counts["truck"][position], counts["task"][position], counts["charger"][position]])
        span = {key: (bounds[offset], bounds[offset + 1]) for offset, key in enumerate(NODE_TYPES)}
        for source, target in RELATION_TYPES:
            (a, b), (c, d) = span[source], span[target]
            pairwise[(source, target)][position, : b - a, : d - c] = item.edges[a:b, c:d]

    privileged = np.zeros((batch, PRIVILEGED_DIM), dtype=np.float32)
    for position, item in enumerate(observations):
        if item.privileged is not None:
            privileged[position] = item.privileged

    # Plan rows (plan 9.5): built only when any observation carries them, so a
    # plan-free batch is byte-identical to today's.
    with_plan = any(item.plan_tasks is not None for item in observations)
    plan_tasks = plan_trucks = None
    if with_plan:
        plan_tasks = np.zeros((batch, widths["task"], PLAN_TASK_DIM), dtype=np.float32)
        plan_trucks = np.zeros((batch, widths["truck"], PLAN_TRUCK_DIM), dtype=np.float32)
        for position, item in enumerate(observations):
            if item.plan_tasks is not None and len(item.plan_tasks):
                plan_tasks[position, : len(item.plan_tasks)] = item.plan_tasks
            if item.plan_trucks is not None and len(item.plan_trucks):
                plan_trucks[position, : len(item.plan_trucks)] = item.plan_trucks

    action_width = max(max(item.num_actions for item in observations), 1)
    actions = np.zeros((batch, action_width, FEATURE_DIMS["action"]), dtype=np.float32)
    action_mask = np.zeros((batch, action_width), dtype=bool)
    action_task = np.full((batch, action_width), -1, dtype=np.int64)
    action_charger = np.full((batch, action_width), -1, dtype=np.int64)
    plan_actions_padded = (
        np.zeros((batch, action_width, PLAN_ACTION_DIM), dtype=np.float32) if with_plan else None
    )
    for position, item in enumerate(observations):
        width = item.num_actions
        actions[position, :width] = item.actions
        action_mask[position, :width] = True
        action_task[position, :width] = item.action_task
        action_charger[position, :width] = item.action_charger
        if plan_actions_padded is not None and item.plan_actions is not None and width:
            plan_actions_padded[position, :width] = item.plan_actions

    device = torch.device(device)
    node_masks = {key: torch.as_tensor(value, device=device) for key, value in masks.items()}
    return GraphBatch(
        nodes={key: torch.as_tensor(value, device=device) for key, value in nodes.items()},
        node_masks=node_masks,
        pairwise={key: torch.as_tensor(value, device=device) for key, value in pairwise.items()},
        pairwise_mask={
            (source, target): node_masks[source].unsqueeze(2) & node_masks[target].unsqueeze(1)
            for source, target in RELATION_TYPES
        },
        context=torch.as_tensor(np.stack([item.context for item in observations]), device=device),
        actions=torch.as_tensor(actions, device=device),
        action_mask=torch.as_tensor(action_mask, device=device),
        action_task=torch.as_tensor(action_task, device=device),
        action_charger=torch.as_tensor(action_charger, device=device),
        active_truck=torch.as_tensor([item.active_truck for item in observations], dtype=torch.long, device=device),
        privileged=torch.as_tensor(privileged, device=device),
        plan_tasks=None if plan_tasks is None else torch.as_tensor(plan_tasks, device=device),
        plan_trucks=None if plan_trucks is None else torch.as_tensor(plan_trucks, device=device),
        plan_actions=None if not with_plan else torch.as_tensor(plan_actions_padded, device=device),
    )
