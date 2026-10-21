"""Canonical typed features shared by flat, set, and graph encoders."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from EVRoutingEnv.models.core.customer import TaskStatus
from EVRoutingEnv.state.feasibility import (
    ActionKind,
    FeasibilityReason,
    joint_action_feasibility,
)


# v6 exposes action budgets, committed activity outcomes and FCFS queue order.
# v7 appends hours-of-service, cargo, public-station information and request
# columns; with every opt-in component disabled the v6 columns are unchanged.
SCHEMA_VERSION = "joint-fleet-v7"

NODE_TYPES = ("truck", "customer", "charger")
EDGE_FEATURES = ("nominal_energy_kwh", "nominal_travel_hours", "reachable")
RELATION_TYPES = tuple(
    (source, target) for source in NODE_TYPES for target in NODE_TYPES
)

# The depot is not a truck, a customer, or a charger, so it appears in none of
# the nine typed relations.  Yet a plan is only feasible if the fleet can still
# get home, and every episode ends with a mandatory depot return.  Without these
# columns a policy is asked to reserve energy for a destination it cannot see.
# Because the depot is a single distinguished node, its relation to each entity
# is a per-node feature rather than a tenth relation type.
DEPOT_FEATURES = (
    "depot_energy_kwh",
    "depot_travel_hours",
    "depot_reachable",
)

TRUCK_HISTORY_FEATURES = (
    "routing_started_at", "route_arrival_at", "charging_started_at",
    "unloading_started_at", "waiting_started_at", "ready_since",
    "next_event_at", "next_event_remaining_hours", "pending_discharge_kwh",
    "pending_charge_kwh", "pending_service_hours", "queue_position",
    "has_port_reservation", "return_to_depot_pending", "completed_services",
)
GLOBAL_HISTORY_FEATURES = (
    "episode_steps", "max_episode_steps", "remaining_episode_steps",
    "invalid_action_count", "invalid_action_budget", "remaining_invalid_actions",
    "invalid_actions_terminate", "max_time", "remaining_time",
)

# Appended in v7. Values are -1 when hours of service are disabled.
TRUCK_HOS_FEATURES = (
    "drive_left_before_break",
    "drive_left_in_shift",
    "window_left_hours",
    "nondriving_streak_hours",
    "eu_split_first_part_taken",
    "rest_required_now",
)
TRUCK_V7_FEATURES = (
    *TRUCK_HOS_FEATURES,
    "on_board_count",
    "on_board_demand",
    "in_station_queue",
    "committed_to_charge",
)
# Public-station information (Kullman et al. 2021 sec. 2.2; Dastpak et al.
# 2024). Without outside traffic the fleet-only state is fully observed.
CHARGER_V7_FEATURES = (
    "observed_busy_ports",
    "observed_queue_length",
    "observation_age_hours",
    "observed_available_ports",
    "expected_wait_given_memory",
    "known_utilization_prior",
    "occupancy_indicator",  # real-time "all up ports busy" (Dastpak et al. 2024), else -1
)
CUSTOMER_V7_FEATURES = (
    "request_kind_code",
    "request_status_code",
    "has_pickup",
    "pickup_node",
    "pickup_earliest",
    "pickup_latest",
    "pickup_service_hours",
    "release_time",
    "delay_hours",
)
GLOBAL_V7_FEATURES = (
    "requests_revealed",
    "requests_served_on_time",
    "requests_served_late",
    "requests_lost",
)

TRUCK_FEATURES = (
    "current_node",
    "battery_kwh",
    "battery_capacity_kwh",
    "remaining_payload",
    "payload_capacity",
    "state_code",
    "route_destination",
    "is_active",
    "is_charging",
    "must_leave_charger",
    "is_complete",
    "failed",
    # Energy headroom after paying for the return leg; negative means the truck
    # is already committed to a recharge before it can go home.
    "battery_minus_depot_energy",
    *DEPOT_FEATURES,
    *TRUCK_HISTORY_FEATURES,
    *TRUCK_V7_FEATURES,
)
CUSTOMER_FEATURES = (
    "task_id",
    "node_id",
    "demand",
    "base_service_time",
    "earliest_service",
    "latest_service",
    "has_time_window",
    "status_code",
    "claimed_by",
    "claimed_at",
    "preassigned_to",
    "sequence_position",
    "service_started_at",
    "served_at",
    *DEPOT_FEATURES,
    *CUSTOMER_V7_FEATURES,
)
CHARGER_FEATURES = (
    "node_id",
    "type_code",
    "available",
    "power_kw",
    "port_capacity",
    "occupancy",
    "waitlist_length",
    "known_workload_hours",
    *DEPOT_FEATURES,
    *CHARGER_V7_FEATURES,
)
# The campaign objective is fleet travel hours, and the action head scores each
# candidate from its own row plus a pooled state embedding. Without these
# columns the only cost signal on a row is ``required_energy``, so the policy is
# asked to minimize travel time while unable to see how long any leg takes, and
# a charger detour is indistinguishable from a stop that is already on the way.
#
# Every value is a *nominal* network quantity -- the same deterministic table the
# heuristic and CP-SAT baselines plan against -- so this exposes no realized
# traffic or energy draw that the simulator has not yet revealed.
ROUTING_ACTION_FEATURES = (
    # Active truck's current node -> this action's target.
    "leg_travel_hours",
    "leg_reachable",
    # Target -> depot: the return leg every episode must eventually pay.
    "target_depot_hours",
    # Target -> the unserved customers, i.e. how well placed this stop leaves
    # the truck for the work that remains.
    "target_pending_min_hours",
    "target_pending_mean_hours",
    # Hours this action adds over driving straight to the nearest unserved
    # customer. Near zero for a stop that is on the way, large for a detour.
    "insertion_detour_hours",
)

ACTION_FEATURES = (
    "kind_code",
    "target_node",
    "charge_value",
    "customer_demand",
    "required_energy",
    "feasible",
    "reason_code",
    *ROUTING_ACTION_FEATURES,
)
GLOBAL_FEATURES = (
    "global_clock",
    "active_truck_id",
    "customers_total",
    "customers_unassigned",
    "customers_claimed",
    "customers_in_service",
    "customers_served",
    "all_customers_served",
    *GLOBAL_HISTORY_FEATURES,
    *GLOBAL_V7_FEATURES,
)


@dataclass(frozen=True)
class CanonicalFleetFeatures:
    """One lossless, finite-valued feature snapshot for joint routing.

    ``pairwise_features`` is the single canonical owner of every typed
    source-target transport value.  Flat, padded-set, and graph adapters must
    all derive their pairwise inputs from this field so that no representation
    observes more of the transport network than any other.
    """

    schema_version: str
    truck_features: np.ndarray
    customer_features: np.ndarray
    charger_features: np.ndarray
    action_features: np.ndarray
    global_features: np.ndarray
    pairwise_features: dict[tuple[str, str], np.ndarray]
    truck_feature_names: tuple[str, ...] = TRUCK_FEATURES
    customer_feature_names: tuple[str, ...] = CUSTOMER_FEATURES
    charger_feature_names: tuple[str, ...] = CHARGER_FEATURES
    action_feature_names: tuple[str, ...] = ACTION_FEATURES
    global_feature_names: tuple[str, ...] = GLOBAL_FEATURES
    edge_feature_names: tuple[str, ...] = EDGE_FEATURES

    def node_counts(self) -> dict[str, int]:
        """Return the number of real rows held for each canonical node type."""
        return {
            "truck": int(self.truck_features.shape[0]),
            "customer": int(self.customer_features.shape[0]),
            "charger": int(self.charger_features.shape[0]),
        }

    def validate(self) -> None:
        expected = (
            (self.truck_features, len(self.truck_feature_names), "truck"),
            (self.customer_features, len(self.customer_feature_names), "customer"),
            (self.charger_features, len(self.charger_feature_names), "charger"),
            (self.action_features, len(self.action_feature_names), "action"),
        )
        for values, width, label in expected:
            if values.ndim != 2 or values.shape[1] != width:
                raise ValueError(
                    f"{label} feature shape {values.shape} does not match width {width}"
                )
            if not np.isfinite(values).all():
                raise ValueError(f"{label} features contain non-finite values")
        if self.global_features.shape != (len(self.global_feature_names),):
            raise ValueError("global feature vector has the wrong shape")
        if not np.isfinite(self.global_features).all():
            raise ValueError("global features contain non-finite values")
        self._validate_pairwise()

    def _validate_pairwise(self) -> None:
        if set(self.pairwise_features) != set(RELATION_TYPES):
            raise ValueError(
                "pairwise_features must contain exactly the nine typed relations"
            )
        counts = self.node_counts()
        width = len(self.edge_feature_names)
        for relation, values in self.pairwise_features.items():
            source, target = relation
            expected_shape = (counts[source], counts[target], width)
            if values.shape != expected_shape:
                raise ValueError(
                    f"relation {relation} shape {values.shape} does not match "
                    f"{expected_shape}"
                )
            if not np.isfinite(values).all():
                raise ValueError(f"relation {relation} contains non-finite values")


def extract_canonical_features(env) -> CanonicalFleetFeatures:
    """Extract canonical joint-routing features from current environment state."""
    if not env.joint_routing or env.task_registry is None:
        raise ValueError("canonical features require an initialized fleet task registry")

    state_codes = {
        state: index
        for index, state in enumerate(
            (
                "ready",
                "routing",
                "waiting_to_charge",
                "waiting_for_service",
                "charging",
                "unloading",
                "waiting_for_task",
                "complete",
                "failed",
                # Appended in v7.
                "resting",
                "waiting_for_requests",
                "loading",
            )
        )
    }
    depot_node = int(env.joint_instance.depot_node)
    truck_rows = []
    for truck in sorted(env.trucks, key=lambda item: item.truck_id):
        depot_columns = _depot_columns(env, int(truck.current_node), depot_node)
        headroom = float(truck.current_battery) - (
            depot_columns[0] if depot_columns[2] else float(truck.battery_capacity)
        )
        truck_rows.append(
            [
                int(truck.current_node),
                float(truck.current_battery),
                float(truck.battery_capacity),
                _optional_value(truck.remaining_payload),
                _optional_value(truck.payload_capacity),
                state_codes[env.truck_states[truck.truck_id]],
                _optional_value(truck.route_destination),
                int(truck.truck_id == env.active_truck_id),
                int(truck.is_charging),
                int(truck.must_leave_charger),
                int(truck.is_complete),
                int(truck.failed),
                headroom,
                *depot_columns,
                *_truck_history(env, truck),
                *_truck_v7(env, truck),
            ]
        )

    task_status_codes = {status: index for index, status in enumerate(TaskStatus)}
    customer_rows = []
    for task in ([] if getattr(env, "dynamic_requests", False) else env.task_registry.tasks()):
        has_time_window = math.isfinite(task.latest_service)
        customer_rows.append(
            [
                task.task_id,
                task.node_id,
                task.demand,
                task.base_service_time,
                task.earliest_service,
                task.latest_service if has_time_window else env.max_time,
                int(has_time_window),
                task_status_codes[task.status],
                _optional_value(task.claimed_by),
                _optional_value(task.claimed_at),
                _optional_value(task.preassigned_to),
                _optional_value(task.sequence_position),
                _optional_value(task.service_started_at),
                _optional_value(task.served_at),
                *_depot_columns(env, int(task.node_id), depot_node),
                *_static_request_columns(task),
            ]
        )
    if getattr(env, "dynamic_requests", False):
        customer_rows = _request_rows(env, depot_node)

    charger_type_codes = {"Level2": 0, "DCFast": 1}
    charger_rows = []
    present = _trucks_present(env) if getattr(env, "public_stations", None) is not None else {}
    for node in sorted(env.charging_nodes):
        end_times = [
            env.charging_station.truck_charge_end_time[truck_id]
            for truck_id in env.charging_station.charger_occupancy[node]
            if truck_id in env.charging_station.truck_charge_end_time
        ]
        known_workload = sum(
            max(0.0, end_time - env.global_clock) for end_time in end_times
        )
        charger_rows.append(
            [
                node,
                charger_type_codes[env.charging_station.charger_type[node]],
                int(env.charging_station.station_available[node]),
                env.charging_station.charger_power_kw[node],
                env.charging_station.charger_capacity[node],
                len(env.charging_station.charger_occupancy[node]),
                len(env.charging_station.charger_waitlist[node]),
                known_workload,
                *_depot_columns(env, int(node), depot_node),
                *_charger_v7(env, int(node), present),
            ]
        )

    decisions = joint_action_feasibility(env)
    action_targets, charge_values = _action_metadata(env)
    routing_columns = _routing_action_columns(env, action_targets, depot_node)
    kind_codes = {kind: index for index, kind in enumerate(ActionKind)}
    reason_codes = {reason: index for index, reason in enumerate(FeasibilityReason)}
    action_rows = []
    engine = getattr(env, "request_engine", None)
    for index, (decision, target, charge_value, routing) in enumerate(zip(
        decisions,
        action_targets,
        charge_values,
        routing_columns,
        strict=True,
    )):
        demand = 0.0
        if decision.action_kind is ActionKind.CUSTOMER and target >= 0:
            if engine is not None:
                # Several live requests may share a node: use the slot's request.
                demand = engine.registry.get(index - env.num_charging_nodes).demand
            else:
                demand = env.task_registry.task_for_node(target).demand
        action_rows.append(
            [
                kind_codes[decision.action_kind],
                target,
                charge_value,
                demand,
                _finite_or_default(decision.required_energy, -1.0),
                int(decision.feasible),
                reason_codes[decision.reason],
                *routing,
            ]
        )

    counts = env.task_registry.counts()
    global_features = np.asarray(
        [
            env.global_clock,
            _optional_value(env.active_truck_id),
            len(env.task_registry),
            counts[TaskStatus.UNASSIGNED.value],
            counts[TaskStatus.CLAIMED.value],
            counts[TaskStatus.IN_SERVICE.value],
            counts[TaskStatus.SERVED.value],
            int(env.task_registry.all_served()),
            env.episode_steps,
            env.max_episode_steps,
            max(0, env.max_episode_steps - env.episode_steps),
            env.invalid_action_count,
            env.invalid_action_budget,
            max(0, env.invalid_action_budget - env.invalid_action_count),
            int(env.invalid_action_mode == "terminate"),
            env.max_time,
            max(0.0, env.max_time - env.global_clock),
            *_global_v7(env),
        ],
        dtype=np.float32,
    )

    truck_features = _as_rows(truck_rows, len(TRUCK_FEATURES))
    customer_features = _as_rows(customer_rows, len(CUSTOMER_FEATURES))
    charger_features = _as_rows(charger_rows, len(CHARGER_FEATURES))
    node_ids = {
        "truck": truck_features[:, TRUCK_FEATURES.index("current_node")],
        "customer": customer_features[:, CUSTOMER_FEATURES.index("node_id")],
        "charger": charger_features[:, CHARGER_FEATURES.index("node_id")],
    }
    result = CanonicalFleetFeatures(
        schema_version=SCHEMA_VERSION,
        truck_features=truck_features,
        customer_features=customer_features,
        charger_features=charger_features,
        action_features=_as_rows(action_rows, len(ACTION_FEATURES)),
        global_features=global_features,
        pairwise_features=extract_pairwise_relations(
            env.transport_graph,
            {key: value.astype(np.int64) for key, value in node_ids.items()},
        ),
    )
    result = _apply_feature_ablations(env, result)
    result.validate()
    return result


def _truck_history(env, truck) -> list[float]:
    """Committed events are observable; unsampled future randomness is not."""
    tid = truck.truck_id
    event = min(
        (event for event in env.event_queue if event.truck_id == tid),
        default=None,
    ) if not (truck.failed or truck.is_complete) else None
    data = {} if event is None else event.data
    event_time = None if event is None else event.time
    arrival_time = truck.route_arrival_time
    if (
        _in_flight_mode(env) == "nominal_eta"
        and env.truck_states[tid] == "routing"
        and truck.route_destination is not None
    ):
        # A dispatcher sees a telematics ETA, not the realized leg: departure
        # plus nominal time, nominal energy, nominal service time.
        origin, destination = int(truck.current_node), int(truck.route_destination)
        nominal_hours = _path_value(env.transport_graph.get_time_distance, origin, destination)
        nominal_energy = _path_value(env.transport_graph.get_path_energy, origin, destination)
        start = (
            env.global_clock if truck.routing_start_time is None else truck.routing_start_time
        )
        arrival_time = start + (nominal_hours if math.isfinite(nominal_hours) else 0.0)
        event_time = max(env.global_clock, arrival_time)
        data = {
            "discharge": nominal_energy if math.isfinite(nominal_energy) else 0.0,
            "unloading_time": _nominal_service_hours(env, destination),
        }
    elif (
        _in_flight_mode(env) == "nominal_eta"
        and env.truck_states[tid] in {"unloading", "waiting_for_service"}
        and ("unloading_duration" in data or "unloading_time" in data)
    ):
        # The unloading draw is realized when service starts (static: at
        # departure); until it ends a dispatcher knows only the nominal time.
        nominal = _nominal_service_hours(env, int(truck.current_node))
        data = {**data, "unloading_duration": nominal, "unloading_time": nominal}
        if env.truck_states[tid] == "unloading" and truck.unloading_start_time is not None:
            event_time = max(env.global_clock, truck.unloading_start_time + nominal)
    queue_position = -1
    reserved = False
    stations = getattr(env, "public_stations", None)
    if stations is not None:
        # Pooled FCFS position (Kullman's z_k) and the port reservation. When a
        # port will free depends on unobserved outside services, so the queued
        # truck's scheduled wake-up is not part of the state.
        position = stations.queue_position(tid)
        queue_position = -1 if position is None else position
        reserved = stations.has_reservation(tid)
        if env.truck_states[tid] == "waiting_to_charge":
            event_time = None
    else:
        for node in env.charging_nodes:
            for position, entry in enumerate(env.charging_station.charger_waitlist[node]):
                if entry["truck_id"] == tid:
                    queue_position = position
            reserved |= tid in env.charging_station.pending_wake_trucks[node]
    return [
        _optional_value(truck.routing_start_time if env.truck_states[tid] == "routing" else None),
        _optional_value(arrival_time),
        _optional_value(truck.charge_start_time),
        _optional_value(truck.unloading_start_time),
        _optional_value(truck.waiting_start_time),
        _optional_value(env.truck_ready_times.get(tid)),
        -1.0 if event_time is None else event_time,
        -1.0 if event_time is None else max(0.0, event_time - env.global_clock),
        data.get("discharge", 0.0),
        data.get("charge_amount", 0.0),
        data.get("unloading_duration", data.get("unloading_time", 0.0)) or 0.0,
        queue_position,
        int(reserved),
        int(truck.return_to_depot_pending),
        truck.completed_service_count,
    ]


def _in_flight_mode(env) -> str:
    extensions = getattr(env, "extensions", None)
    return "realized" if extensions is None else extensions.in_flight_information


def _nominal_service_hours(env, node: int) -> float:
    """Base service time at a delivery node, without the stochastic draw."""
    engine = getattr(env, "request_engine", None)
    if engine is not None:
        return engine.nominal_service_hours(node)
    try:
        return float(env.task_registry.task_for_node(node).base_service_time)
    except KeyError:
        return 0.0


def _truck_v7(env, truck) -> list[float]:
    hos = getattr(env, "hos", None)
    hos_columns = [-1.0] * len(TRUCK_HOS_FEATURES) if hos is None else hos.feature_row(truck)
    engine = getattr(env, "request_engine", None)
    if engine is None:
        cargo = [0.0, 0.0]
    else:
        cargo = [
            float(len(engine.registry.on_board(truck.truck_id))),
            float(engine.registry.on_board_demand(truck.truck_id)),
        ]
    stations = getattr(env, "public_stations", None)
    queue = [0.0, 0.0] if stations is None else stations.truck_flags(truck.truck_id)
    return [*hos_columns, *cargo, *queue]


def _trucks_present(env) -> dict[int, set[int]]:
    """Fleet trucks physically at each node (not routing, still in service)."""
    present: dict[int, set[int]] = {}
    for truck in env.trucks:
        if env.truck_states[truck.truck_id] not in {"routing", "complete", "failed"}:
            present.setdefault(int(truck.current_node), set()).add(truck.truck_id)
    return present


def _charger_v7(env, node: int, present: dict[int, set[int]]) -> list[float]:
    stations = getattr(env, "public_stations", None)
    if stations is not None:
        return stations.feature_row(
            node, env.active_truck_id, env.global_clock, present.get(node, set())
        )
    station = env.charging_station
    return [
        float(len(station.charger_occupancy[node])),
        float(len(station.charger_waitlist[node])),
        0.0,
        float(station.charger_capacity[node]),
        0.0,
        0.0,
        -1.0,
    ]


# Legacy task status in the request lifecycle (static_delivery has no pickup).
_STATIC_STATUS = {
    TaskStatus.UNASSIGNED: "open",
    TaskStatus.CLAIMED: "claimed_delivery",
    TaskStatus.IN_SERVICE: "in_service",
    TaskStatus.SERVED: "served",
}


def _static_request_columns(task) -> list[float]:
    from EVRoutingEnv.models.core.requests import RequestKind, RequestStatus

    status = RequestStatus(_STATIC_STATUS[task.status])
    return [
        float(list(RequestKind).index(RequestKind.STATIC_DELIVERY)),
        float(list(RequestStatus).index(status)),
        0.0, -1.0, -1.0, -1.0, -1.0, 0.0, 0.0,
    ]


def _request_rows(env, depot_node: int) -> list[list[float]]:
    return env.request_engine.feature_rows(env, depot_node)


def _global_v7(env) -> list[float]:
    engine = getattr(env, "request_engine", None)
    if engine is not None:
        return engine.global_columns()
    tasks = env.task_registry.tasks()
    served = sum(task.is_served for task in tasks)
    return [float(len(tasks)), float(served), 0.0, 0.0]


# Named blocks a component ablation may blank.  Zeroing rather than removing
# keeps the observation width, the network shape, and the interaction budget
# identical, so an arm differs from its control in what the rows *carry* and
# nothing else -- the same discipline the routing-feature ablation already uses.
ABLATION_BLOCKS: dict[str, tuple[str, tuple[str, ...]]] = {
    "queue": (
        "charger",
        ("port_capacity", "occupancy", "waitlist_length", "known_workload_hours"),
    ),
    "active_truck": ("truck", ("is_active",)),
    "depot": ("all_nodes", DEPOT_FEATURES),
    "edges": ("pairwise", EDGE_FEATURES),
}


def _apply_feature_ablations(env, features: CanonicalFleetFeatures):
    """Blank the feature blocks this configuration ablates, if any."""
    requested = env.config.get("environment", {}).get("ablate_features") or ()
    if not requested:
        return features

    unknown = sorted(set(requested) - set(ABLATION_BLOCKS))
    if unknown:
        raise ValueError(
            f"unknown ablation blocks {unknown}; expected any of "
            f"{sorted(ABLATION_BLOCKS)}"
        )

    truck = features.truck_features.copy()
    customer = features.customer_features.copy()
    charger = features.charger_features.copy()
    pairwise = {key: value.copy() for key, value in features.pairwise_features.items()}
    blocks = {
        "truck": (truck, TRUCK_FEATURES),
        "customer": (customer, CUSTOMER_FEATURES),
        "charger": (charger, CHARGER_FEATURES),
    }

    for name in requested:
        scope, columns = ABLATION_BLOCKS[name]
        if scope == "pairwise":
            for array in pairwise.values():
                array[...] = 0.0
            continue
        targets = blocks.values() if scope == "all_nodes" else (blocks[scope],)
        for array, names in targets:
            for column in columns:
                if column in names:
                    array[:, names.index(column)] = 0.0

    return replace(
        features,
        truck_features=truck,
        customer_features=customer,
        charger_features=charger,
        pairwise_features=pairwise,
    )


def _depot_columns(env, node: int, depot_node: int) -> tuple[float, float, float]:
    """Nominal energy, travel time, and reachability from ``node`` to the depot.

    Returned as finite values with an explicit reachability flag, matching the
    convention used by the pairwise relation tensors: an unreachable pair is
    zeros with the flag cleared, never an infinity.
    """
    energy = _path_value(env.transport_graph.get_path_energy, node, depot_node)
    travel_hours = _path_value(env.transport_graph.get_time_distance, node, depot_node)
    if math.isfinite(energy) and math.isfinite(travel_hours):
        return (energy, travel_hours, 1.0)
    return (0.0, 0.0, 0.0)


def _routing_action_columns(
    env, action_targets: list[int], depot_node: int
) -> list[tuple[float, ...]]:
    """Per-action nominal leg cost and detour lookahead for the active truck.

    Unreachable pairs follow the convention used everywhere else in this module:
    zeros carrying an explicit reachability flag, never an infinity, so the flat
    observation stays finite.
    """
    width = len(ROUTING_ACTION_FEATURES)
    count = len(action_targets)
    # The ablation arm zeroes these columns rather than removing them, so the
    # observation width, the network shape, and the interaction budget stay
    # identical and the only difference is what the action rows carry.
    if not env.config.get("environment", {}).get("routing_action_features", True):
        return [(0.0,) * width] * count
    if env.active_truck_id is None:
        return [(0.0,) * width] * count

    origin = int(env.trucks[env.active_truck_id].current_node)
    pending = [
        int(task.node_id) for task in env.task_registry.tasks() if not task.is_served
    ]
    involved = {origin, depot_node, *pending}
    involved.update(int(target) for target in action_targets if target >= 0)
    position, dense = _dense_lookup(
        env.transport_graph, [np.asarray(sorted(involved), dtype=np.int64)]
    )
    # This runs on every environment step for every action, so it is written as
    # whole-array indexing rather than per-action lookups.
    hours = dense[:, :, EDGE_FEATURES.index("nominal_travel_hours")]
    reachable = dense[:, :, EDGE_FEATURES.index("reachable")] > 0.0

    targets = np.asarray(action_targets, dtype=np.int64)
    valid = targets >= 0
    rows = np.zeros(count, dtype=np.int64)
    rows[valid] = [position[int(node)] for node in targets[valid].tolist()]
    origin_row = position[origin]
    depot_column = position[depot_node]

    leg = np.where(valid & reachable[origin_row, rows], hours[origin_row, rows], np.nan)
    depot_hours = np.where(reachable[rows, depot_column], hours[rows, depot_column], 0.0)

    pending_columns = np.asarray(
        [position[node] for node in pending], dtype=np.int64
    )
    if pending_columns.size:
        # (actions, pending): hours from each candidate target to the work left.
        block = hours[rows[:, None], pending_columns[None, :]]
        usable = reachable[rows[:, None], pending_columns[None, :]] & (
            rows[:, None] != pending_columns[None, :]
        )
        counts = usable.sum(axis=1)
        masked = np.where(usable, block, np.inf)
        nearest = np.where(counts > 0, masked.min(axis=1, initial=np.inf), 0.0)
        totals = np.where(usable, block, 0.0).sum(axis=1)
        mean = np.divide(totals, counts, out=np.zeros(count), where=counts > 0)

        origin_usable = reachable[origin_row, pending_columns] & (
            origin_row != pending_columns
        )
        origin_values = hours[origin_row, pending_columns][origin_usable]
        origin_nearest = float(origin_values.min()) if origin_values.size else 0.0
        origin_has_work = bool(origin_values.size)
    else:
        nearest = np.zeros(count)
        mean = np.zeros(count)
        counts = np.zeros(count, dtype=np.int64)
        origin_nearest, origin_has_work = 0.0, False

    # Only meaningful while work remains on both sides of the comparison.
    detour = np.where(
        (counts > 0) & origin_has_work, leg + nearest - origin_nearest, leg
    )

    stacked = np.stack(
        [
            leg,
            np.ones(count),
            depot_hours,
            nearest,
            mean,
            detour,
        ],
        axis=1,
    )
    # Charge actions target the truck's own node, so a zero row there is the
    # truth rather than a missing value; an unreachable navigation target is
    # masked infeasible anyway.
    stacked[np.isnan(leg)] = 0.0
    return [tuple(float(value) for value in row) for row in stacked]


def extract_pairwise_relations(
    transport_graph,
    node_ids: dict[str, np.ndarray],
) -> dict[tuple[str, str], np.ndarray]:
    """Compute every typed source-target transport value exactly once.

    Values are resolved on the unique underlying network nodes and then
    gathered per relation, so identical node pairs always receive identical
    energy, travel-time, and reachability values regardless of node type.
    """
    if set(node_ids) != set(NODE_TYPES):
        raise ValueError("node_ids must contain truck, customer, and charger arrays")

    stacked = [
        np.asarray(node_ids[node_type], dtype=np.int64) for node_type in NODE_TYPES
    ]
    for values in stacked:
        if values.ndim != 1:
            raise ValueError("each node id array must be one-dimensional")
    position, dense = _dense_lookup(transport_graph, stacked)

    relations: dict[tuple[str, str], np.ndarray] = {}
    for source_type, target_type in RELATION_TYPES:
        rows = np.asarray(
            [position[int(node)] for node in node_ids[source_type].tolist()],
            dtype=np.int64,
        )
        columns = np.asarray(
            [position[int(node)] for node in node_ids[target_type].tolist()],
            dtype=np.int64,
        )
        if rows.size == 0 or columns.size == 0:
            relations[(source_type, target_type)] = np.zeros(
                (rows.size, columns.size, len(EDGE_FEATURES)), dtype=np.float32
            )
            continue
        relations[(source_type, target_type)] = dense[
            rows[:, None], columns[None, :]
        ].astype(np.float32, copy=True)
    return relations


def _dense_lookup(
    transport_graph,
    node_id_arrays: list[np.ndarray],
) -> tuple[dict[int, int], np.ndarray]:
    """Return a position map and dense value table covering the given nodes.

    Real transport graphs expose a cached all-pairs table; anything else (test
    doubles, reduced graphs) falls back to per-pair queries over the unique
    nodes actually present in the snapshot.
    """
    dense_matrix = getattr(transport_graph, "dense_transport_matrix", None)
    if callable(dense_matrix):
        return dense_matrix()

    unique_nodes = (
        np.unique(np.concatenate(node_id_arrays))
        if any(values.size for values in node_id_arrays)
        else np.zeros(0, dtype=np.int64)
    )
    position = {int(node): index for index, node in enumerate(unique_nodes.tolist())}
    dense = np.zeros(
        (len(unique_nodes), len(unique_nodes), len(EDGE_FEATURES)), dtype=np.float32
    )
    for source_index, source_node in enumerate(unique_nodes.tolist()):
        for target_index, target_node in enumerate(unique_nodes.tolist()):
            energy = _path_value(
                transport_graph.get_path_energy, int(source_node), int(target_node)
            )
            travel_hours = _path_value(
                transport_graph.get_time_distance, int(source_node), int(target_node)
            )
            if math.isfinite(energy) and math.isfinite(travel_hours):
                dense[source_index, target_index] = (energy, travel_hours, 1.0)
    return position, dense


def _path_value(function, source: int, target: int) -> float:
    """Return a nonnegative finite path value, or infinity when unreachable."""
    try:
        value = float(function(source, target))
    except (KeyError, TypeError, ValueError):
        return math.inf
    return value if math.isfinite(value) and value >= 0.0 else math.inf


def _as_rows(rows: list[list[float]], width: int) -> np.ndarray:
    if not rows:
        return np.zeros((0, width), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


def _action_metadata(env) -> tuple[list[int], list[float]]:
    if env.active_truck_id is None:
        return ([-1] * env.action_space.n, [0.0] * env.action_space.n)
    truck = env.trucks[env.active_truck_id]
    targets = [int(node) for node in env.charging_nodes]
    engine = getattr(env, "request_engine", None)
    for offset in range(env.fixed_num_stops):
        if engine is not None:
            target = engine.slot_target(truck, offset)
        else:
            target = (
                int(env.joint_instance.tasks[offset].node_id)
                if offset < len(env.joint_instance.tasks)
                else -1
            )
        targets.append(target)
    targets.append(int(env.joint_instance.depot_node))
    targets.extend([int(truck.current_node)] * env.num_charge_actions)
    extra = len(getattr(env, "extra_action_kinds", ()))
    targets.extend([int(truck.current_node)] * extra)
    values = (
        [0.0] * env.num_navigation_actions + list(env.charge_action_values) + [0.0] * extra
    )
    return targets, values


def _optional_value(value) -> float:
    return -1.0 if value is None else float(value)


def _finite_or_default(value: float | None, default: float) -> float:
    if value is None or not math.isfinite(value):
        return default
    return float(value)
