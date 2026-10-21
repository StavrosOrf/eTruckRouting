"""Hard feasibility rules for the primary joint fleet-routing model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class ActionKind(StrEnum):
    """Semantic action classes used by the feasibility engine."""

    CUSTOMER = "customer"
    CHARGER = "charger"
    DEPOT = "depot"
    CHARGE = "charge"
    UNKNOWN = "unknown"
    # Appended last: features derive kind codes from enumeration order.
    QUEUE = "queue"
    REST_BREAK = "rest_break"
    REST_DAILY = "rest_daily"
    WAIT = "wait"
    END_SHIFT = "end_shift"


class FeasibilityReason(StrEnum):
    """Stable reason codes for valid and rejected actions."""

    FEASIBLE = "feasible"
    NO_ACTIVE_TRUCK = "no_active_truck"
    INVALID_TRUCK_STATE = "invalid_truck_state"
    UNKNOWN_DESTINATION = "unknown_destination"
    CHARGER_UNAVAILABLE = "charger_unavailable"
    SAME_LOCATION = "same_location"
    TASK_UNAVAILABLE = "task_unavailable"
    PAYLOAD_EXCEEDED = "payload_exceeded"
    CUSTOMERS_REMAIN = "customers_remain"
    DEPOT_RETURN_NOT_REQUIRED = "depot_return_not_required"
    UNREACHABLE = "unreachable"
    INSUFFICIENT_ENERGY = "insufficient_energy"
    NOT_AT_CHARGER = "not_at_charger"
    CHARGER_MISMATCH = "charger_mismatch"
    BATTERY_FULL = "battery_full"
    INVALID_CHARGE_DURATION = "invalid_charge_duration"
    INVALID_TARGET_SOC = "invalid_target_soc"
    TARGET_SOC_NOT_ABOVE_CURRENT = "target_soc_not_above_current"
    TIME_WINDOW_EXPIRED = "time_window_expired"
    MUST_LEAVE_CHARGER = "must_leave_charger"
    EMPTY_ACTION_SLOT = "empty_action_slot"
    PREASSIGNED_TO_OTHER = "preassigned_to_other"
    OUT_OF_SEQUENCE = "out_of_sequence"
    # Appended last so existing reason codes keep their indices.
    QUEUE_NOT_NEEDED = "queue_not_needed"
    COMMITTED_TO_CHARGE = "committed_to_charge"
    HOS_DRIVING_LIMIT = "hos_driving_limit"
    HOS_SHIFT_WINDOW = "hos_shift_window"
    REST_NOT_REQUIRED = "rest_not_required"
    REQUEST_HIDDEN = "request_hidden"
    REQUEST_LOST = "request_lost"
    NOT_ON_BOARD = "not_on_board"
    MANIFEST_LOCKED = "manifest_locked"
    REQUESTS_ON_BOARD = "requests_on_board"
    NO_FUTURE_RELEASE = "no_future_release"
    CONTAINER_ON_BOARD = "container_on_board"
    PORTS_BUSY = "ports_busy"
    NOT_AT_DEPOT = "not_at_depot"
    DRAYAGE_ORIGIN_FIRST = "drayage_origin_first"
    STATION_HOP_LIMIT = "station_hop_limit"


@dataclass(frozen=True)
class FeasibilityResult:
    """Validity and diagnostic data for one candidate action."""

    feasible: bool
    reason: FeasibilityReason
    action_kind: ActionKind
    target_node: int | None = None
    required_energy: float | None = None

    @classmethod
    def allow(
        cls,
        action_kind: ActionKind,
        *,
        target_node: int | None = None,
        required_energy: float | None = None,
    ) -> FeasibilityResult:
        return cls(
            feasible=True,
            reason=FeasibilityReason.FEASIBLE,
            action_kind=action_kind,
            target_node=target_node,
            required_energy=required_energy,
        )

    @classmethod
    def reject(
        cls,
        reason: FeasibilityReason,
        action_kind: ActionKind,
        *,
        target_node: int | None = None,
        required_energy: float | None = None,
    ) -> FeasibilityResult:
        return cls(
            feasible=False,
            reason=reason,
            action_kind=action_kind,
            target_node=target_node,
            required_energy=required_energy,
        )


def evaluate_joint_route(
    *,
    truck: Any,
    truck_state: str,
    target_node: int,
    transport_graph: Any,
    charging_nodes: list[int],
    task_registry: Any,
    depot_node: int,
    energy_multiplier: float = 1.0,
    unavailable_charging_nodes: set[int] | None = None,
    current_time: float = 0.0,
    realized_leg: dict[str, float] | None = None,
) -> FeasibilityResult:
    """Evaluate a destination, optionally using a clairvoyant leg realization."""
    target_node = int(target_node)
    depot_node = int(depot_node)
    charging_set = {int(node) for node in charging_nodes}

    action_kind = _classify_destination(
        target_node,
        charging_set=charging_set,
        task_registry=task_registry,
        depot_node=depot_node,
    )
    if truck_state != "ready":
        return FeasibilityResult.reject(
            FeasibilityReason.INVALID_TRUCK_STATE,
            action_kind,
            target_node=target_node,
        )
    if not math.isfinite(current_time) or current_time < 0.0:
        raise ValueError("current_time must be finite and non-negative")
    if action_kind is ActionKind.UNKNOWN:
        return FeasibilityResult.reject(
            FeasibilityReason.UNKNOWN_DESTINATION,
            action_kind,
            target_node=target_node,
        )
    if (
        action_kind is ActionKind.CHARGER
        and unavailable_charging_nodes is not None
        and target_node in unavailable_charging_nodes
    ):
        return FeasibilityResult.reject(
            FeasibilityReason.CHARGER_UNAVAILABLE,
            action_kind,
            target_node=target_node,
        )
    if int(truck.current_node) == target_node and action_kind is not ActionKind.DEPOT:
        return FeasibilityResult.reject(
            FeasibilityReason.SAME_LOCATION,
            action_kind,
            target_node=target_node,
            required_energy=0.0,
        )

    if action_kind is ActionKind.CUSTOMER:
        task = task_registry.task_for_node(target_node)
        if task.preassigned_to is not None and task.preassigned_to != truck.truck_id:
            # eTFRP variant: this customer belongs to another truck, so it is
            # not merely taken but was never this truck's to take.
            return FeasibilityResult.reject(
                FeasibilityReason.PREASSIGNED_TO_OTHER,
                action_kind,
                target_node=target_node,
            )
        if not task.is_available:
            return FeasibilityResult.reject(
                FeasibilityReason.TASK_UNAVAILABLE,
                action_kind,
                target_node=target_node,
            )
        if not task_registry.is_next_customer(target_node, truck.truck_id):
            return FeasibilityResult.reject(
                FeasibilityReason.OUT_OF_SEQUENCE,
                action_kind,
                target_node=target_node,
            )
        if not truck.can_accept_demand(task.demand):
            return FeasibilityResult.reject(
                FeasibilityReason.PAYLOAD_EXCEEDED,
                action_kind,
                target_node=target_node,
            )
        if math.isfinite(task.latest_service):
            try:
                nominal_travel_time = (
                    realized_leg["travel_time_hours"]
                    if realized_leg is not None
                    else float(
                        transport_graph.get_time_distance(
                            int(truck.current_node), target_node
                        )
                    )
                )
            except (AttributeError, KeyError, TypeError, ValueError):
                nominal_travel_time = math.inf
            service_start = max(
                current_time + nominal_travel_time,
                float(task.earliest_service),
            )
            if not math.isfinite(service_start) or (
                service_start > float(task.latest_service) + 1e-9
            ):
                return FeasibilityResult.reject(
                    FeasibilityReason.TIME_WINDOW_EXPIRED,
                    action_kind,
                    target_node=target_node,
                )
    if not math.isfinite(energy_multiplier) or energy_multiplier < 1.0:
        raise ValueError("energy_multiplier must be finite and at least 1.0")
    try:
        # Selecting the depot closes the truck's route, even before service is
        # finished. A truck already there may retire without consuming energy.
        nominal_energy = (
            0.0
            if int(truck.current_node) == target_node
            else float(
                transport_graph.get_path_energy(int(truck.current_node), target_node)
            )
        )
    except (KeyError, ValueError):
        nominal_energy = math.inf

    if not math.isfinite(nominal_energy) or nominal_energy < 0.0:
        return FeasibilityResult.reject(
            FeasibilityReason.UNREACHABLE,
            action_kind,
            target_node=target_node,
            required_energy=math.inf,
        )
    required_energy = (
        float(realized_leg["energy_kwh"])
        if realized_leg is not None
        else nominal_energy * float(energy_multiplier)
    )
    if not math.isfinite(required_energy) or required_energy < 0:
        raise ValueError("realized energy must be finite and non-negative")
    if required_energy > float(truck.current_battery) + 1e-9:
        return FeasibilityResult.reject(
            FeasibilityReason.INSUFFICIENT_ENERGY,
            action_kind,
            target_node=target_node,
            required_energy=required_energy,
        )

    return FeasibilityResult.allow(
        action_kind,
        target_node=target_node,
        required_energy=required_energy,
    )


def evaluate_duration_charge(
    *,
    truck: Any,
    truck_state: str,
    charger_node: int,
    charging_nodes: list[int],
    charge_hours: float,
    station_available: bool = True,
) -> FeasibilityResult:
    """Evaluate a legacy duration charge without applying fallback behavior."""
    charger_node = int(charger_node)
    if truck_state != "ready":
        return FeasibilityResult.reject(
            FeasibilityReason.INVALID_TRUCK_STATE,
            ActionKind.CHARGE,
            target_node=charger_node,
        )
    if not math.isfinite(charge_hours) or charge_hours <= 0.0:
        return FeasibilityResult.reject(
            FeasibilityReason.INVALID_CHARGE_DURATION,
            ActionKind.CHARGE,
            target_node=charger_node,
        )
    if int(truck.current_node) not in {int(node) for node in charging_nodes}:
        return FeasibilityResult.reject(
            FeasibilityReason.NOT_AT_CHARGER,
            ActionKind.CHARGE,
            target_node=charger_node,
        )
    if charger_node != int(truck.current_node):
        return FeasibilityResult.reject(
            FeasibilityReason.CHARGER_MISMATCH,
            ActionKind.CHARGE,
            target_node=charger_node,
        )
    if not station_available:
        return FeasibilityResult.reject(
            FeasibilityReason.CHARGER_UNAVAILABLE,
            ActionKind.CHARGE,
            target_node=charger_node,
        )
    if bool(getattr(truck, "must_leave_charger", False)):
        return FeasibilityResult.reject(
            FeasibilityReason.MUST_LEAVE_CHARGER,
            ActionKind.CHARGE,
            target_node=charger_node,
        )
    if float(truck.current_battery) >= float(truck.battery_capacity) - 1e-9:
        return FeasibilityResult.reject(
            FeasibilityReason.BATTERY_FULL,
            ActionKind.CHARGE,
            target_node=charger_node,
        )
    return FeasibilityResult.allow(ActionKind.CHARGE, target_node=charger_node)


def evaluate_target_soc_charge(
    *,
    truck: Any,
    truck_state: str,
    charger_node: int,
    charging_nodes: list[int],
    target_soc: float,
    station_available: bool = True,
) -> FeasibilityResult:
    """Evaluate a target-SoC charge under hard location and state rules."""
    charger_node = int(charger_node)
    if truck_state != "ready":
        return FeasibilityResult.reject(
            FeasibilityReason.INVALID_TRUCK_STATE,
            ActionKind.CHARGE,
            target_node=charger_node,
        )
    if not math.isfinite(target_soc) or not 0.0 < target_soc <= 1.0:
        return FeasibilityResult.reject(
            FeasibilityReason.INVALID_TARGET_SOC,
            ActionKind.CHARGE,
            target_node=charger_node,
        )
    if int(truck.current_node) not in {int(node) for node in charging_nodes}:
        return FeasibilityResult.reject(
            FeasibilityReason.NOT_AT_CHARGER,
            ActionKind.CHARGE,
            target_node=charger_node,
        )
    if charger_node != int(truck.current_node):
        return FeasibilityResult.reject(
            FeasibilityReason.CHARGER_MISMATCH,
            ActionKind.CHARGE,
            target_node=charger_node,
        )
    if not station_available:
        return FeasibilityResult.reject(
            FeasibilityReason.CHARGER_UNAVAILABLE,
            ActionKind.CHARGE,
            target_node=charger_node,
        )
    if bool(getattr(truck, "must_leave_charger", False)):
        return FeasibilityResult.reject(
            FeasibilityReason.MUST_LEAVE_CHARGER,
            ActionKind.CHARGE,
            target_node=charger_node,
        )
    current_soc = float(truck.current_battery) / float(truck.battery_capacity)
    if target_soc <= current_soc + 1e-9:
        return FeasibilityResult.reject(
            FeasibilityReason.TARGET_SOC_NOT_ABOVE_CURRENT,
            ActionKind.CHARGE,
            target_node=charger_node,
        )
    return FeasibilityResult.allow(ActionKind.CHARGE, target_node=charger_node)


def realized_route_costs(
    env: Any, truck: Any, target_node: int
) -> dict[str, float] | None:
    """Query the next leg only on private perfect-information planning states.

    Masks and execution use the same query; evaluating alternatives never
    changes which traversal draw normal execution subsequently consumes.
    """
    if not getattr(env, "_perfect_information", False):
        return None
    source, target = int(truck.current_node), int(target_node)
    try:
        energy = float(env.transport_graph.get_path_energy(source, target))
        hours = float(env.transport_graph.get_time_distance(source, target))
    except (KeyError, ValueError):
        return None  # The ordinary reachability validator rejects this leg.
    if not math.isfinite(energy) or not math.isfinite(hours):
        return None
    return env.traffic_simulator.realize_journey(
        from_node=source,
        to_node=target,
        current_time=float(env.global_clock),
        travel_time=hours,
        base_energy=energy,
        occurrence=env.traffic_simulator._journey_counters.get((source, target), 0),
    )


def joint_action_feasibility(env: Any) -> list[FeasibilityResult]:
    """Evaluate every discrete action of a configured joint-routing environment."""
    if env.active_truck_id is None:
        return [
            FeasibilityResult.reject(
                FeasibilityReason.NO_ACTIVE_TRUCK,
                ActionKind.UNKNOWN,
            )
            for _ in range(env.action_space.n)
        ]
    if env.task_registry is None or env.joint_instance is None:
        raise RuntimeError("joint-routing environment has no task registry")

    truck = env.trucks[env.active_truck_id]
    truck_state = env.truck_states[truck.truck_id]
    energy_multiplier = 1.0
    if (
        env.traffic_config["enable_energy_uncertainty"]
        and env.traffic_config["enable_traffic"]
    ):
        energy_multiplier = max(
            1.0,
            float(env.traffic_config["max_energy_multiplier"]),
        )

    decisions: list[FeasibilityResult] = []
    unavailable_chargers = {
        int(node)
        for node, available in env.charging_station.station_available.items()
        if not available
    }
    for charger_node in env.charging_nodes:
        decisions.append(
            extension_route_check(env, truck, evaluate_joint_route(
                truck=truck,
                truck_state=truck_state,
                target_node=charger_node,
                realized_leg=realized_route_costs(env, truck, charger_node),
                transport_graph=env.transport_graph,
                charging_nodes=env.charging_nodes,
                task_registry=env.task_registry,
                depot_node=env.joint_instance.depot_node,
                energy_multiplier=energy_multiplier,
                current_time=float(env.global_clock),
                unavailable_charging_nodes=unavailable_chargers,
            ))
        )

    for offset in range(env.fixed_num_stops):
        if getattr(env, "dynamic_requests", False):
            decisions.append(evaluate_request_slot(env, truck, offset))
            continue
        if offset >= len(env.joint_instance.tasks):
            decisions.append(
                FeasibilityResult.reject(
                    FeasibilityReason.EMPTY_ACTION_SLOT,
                    ActionKind.CUSTOMER,
                )
            )
            continue
        decisions.append(
            extension_route_check(env, truck, evaluate_joint_route(
                truck=truck,
                truck_state=truck_state,
                target_node=env.joint_instance.tasks[offset].node_id,
                realized_leg=realized_route_costs(
                    env, truck, env.joint_instance.tasks[offset].node_id
                ),
                transport_graph=env.transport_graph,
                charging_nodes=env.charging_nodes,
                task_registry=env.task_registry,
                depot_node=env.joint_instance.depot_node,
                energy_multiplier=energy_multiplier,
                current_time=float(env.global_clock),
                unavailable_charging_nodes=unavailable_chargers,
            ))
        )

    decisions.append(evaluate_depot_action(env, truck))

    charger_nodes = env.charging_nodes
    station_available = env.charging_station.station_available.get(
        int(truck.current_node), False
    )
    if at_depot_charger(env, truck):
        charger_nodes = [*env.charging_nodes, int(truck.current_node)]
        station_available = True
    if env.charging_action_mode == "target_soc":
        for target_soc in env.charge_action_values:
            decisions.append(
                extension_charge_check(env, truck, evaluate_target_soc_charge(
                    truck=truck,
                    truck_state=truck_state,
                    charger_node=truck.current_node,
                    charging_nodes=charger_nodes,
                    target_soc=float(target_soc),
                    station_available=station_available,
                ))
            )
    else:
        for charge_hours in env.charge_action_values:
            decisions.append(
                extension_charge_check(env, truck, evaluate_duration_charge(
                    truck=truck,
                    truck_state=truck_state,
                    charger_node=truck.current_node,
                    charging_nodes=charger_nodes,
                    charge_hours=float(charge_hours),
                    station_available=station_available,
                ))
            )

    for kind in getattr(env, "extra_action_kinds", ()):
        decisions.append(evaluate_extra_action(env, truck, kind))

    if len(decisions) != env.action_space.n:
        raise RuntimeError(
            f"feasibility engine produced {len(decisions)} actions for "
            f"action space of size {env.action_space.n}"
        )
    return decisions


def _classify_destination(
    target_node: int,
    *,
    charging_set: set[int],
    task_registry: Any,
    depot_node: int,
) -> ActionKind:
    if target_node == depot_node:
        return ActionKind.DEPOT
    if target_node in charging_set:
        return ActionKind.CHARGER
    try:
        task_registry.task_for_node(target_node)
    except KeyError:
        return ActionKind.UNKNOWN
    return ActionKind.CUSTOMER


# ---------------------------------------------------------------------------
# Opt-in model components. Every rule of the public-station, hours-of-service
# and dynamic-request components lives here, so the policy mask, ``step``, the
# baselines and the planners all see one definition. Components on the
# environment hold state only (``env.public_stations``, ``env.hos``,
# ``env.request_engine``); each is None when its block is disabled, and then
# these functions return the legacy decision unchanged.
# ---------------------------------------------------------------------------


def _energy_multiplier(env: Any) -> float:
    if env.traffic_config["enable_traffic"] and env.traffic_config["enable_energy_uncertainty"]:
        return max(1.0, float(env.traffic_config["max_energy_multiplier"]))
    return 1.0


def at_depot_charger(env: Any, truck: Any) -> bool:
    """The private depot charger of Kullman et al. (2021) sec. 2.1."""
    engine = getattr(env, "request_engine", None)
    return (
        engine is not None
        and env.extensions.depot_charging.enabled
        and int(truck.current_node) == int(env.joint_instance.depot_node)
    )


def hos_leg_check(env: Any, truck: Any, target_node: int, kind: ActionKind):
    """Section 6.4/6.5: no driving over the allowance (stops-only: per leg)."""
    from EVRoutingEnv.models.core.hours_of_service import driving_allowance, window_left

    hos = getattr(env, "hos", None)
    if hos is None or int(truck.current_node) == int(target_node):
        return None
    clock = hos.clock(truck.truck_id)
    allowance = driving_allowance(hos.regime, clock)
    if hos.config.rest_locations == "anywhere":
        fits = allowance > 1e-9  # rests are inserted inside the leg (Goel)
    else:
        fits = hos.planned_leg_hours(env, truck, target_node) <= allowance + 1e-9
    if fits:
        return None
    window_binding = window_left(hos.regime, clock) <= allowance + 1e-9
    return FeasibilityResult.reject(
        FeasibilityReason.HOS_SHIFT_WINDOW if window_binding else FeasibilityReason.HOS_DRIVING_LIMIT,
        kind,
        target_node=int(target_node),
    )


def _component_leg_checks(env: Any, truck: Any, target_node: int, kind: ActionKind):
    """Rules shared by every action that drives the active truck somewhere."""
    stations = getattr(env, "public_stations", None)
    if stations is not None and stations.committed(truck.truck_id):
        # Kullman sec. 2.1: at the head of the queue "it must charge".
        return FeasibilityResult.reject(
            FeasibilityReason.COMMITTED_TO_CHARGE, kind, target_node=int(target_node)
        )
    if (
        stations is not None
        and kind is ActionKind.CHARGER
        and stations.config.max_stations_between_stops is not None
        and getattr(truck, "stations_since_stop", 0) >= stations.config.max_stations_between_stops
    ):
        return FeasibilityResult.reject(
            FeasibilityReason.STATION_HOP_LIMIT, kind, target_node=int(target_node)
        )
    engine = getattr(env, "request_engine", None)
    if engine is not None and engine.is_drayage and kind is ActionKind.CHARGER:
        if engine.container(truck.truck_id) is None and int(truck.current_node) == int(
            env.joint_instance.depot_node
        ):
            # Xiao et al. (2025) sec. 3.1: the first stop after the depot is an origin.
            return FeasibilityResult.reject(
                FeasibilityReason.DRAYAGE_ORIGIN_FIRST, kind, target_node=int(target_node)
            )
    return hos_leg_check(env, truck, target_node, kind)


def extension_route_check(env: Any, truck: Any, decision: FeasibilityResult) -> FeasibilityResult:
    """Apply the enabled components to a legacy navigation decision."""
    if not decision.feasible or decision.target_node is None:
        return decision
    if not getattr(env, "components_enabled", False):
        return decision
    objection = _component_leg_checks(env, truck, decision.target_node, decision.action_kind)
    return decision if objection is None else objection


def extension_charge_check(env: Any, truck: Any, decision: FeasibilityResult) -> FeasibilityResult:
    """Kullman's arrival epoch: a busy station offers queue or leave, not charge."""
    if not decision.feasible or not getattr(env, "components_enabled", False):
        return decision
    stations = getattr(env, "public_stations", None)
    if stations is not None and stations.awaiting_decision(env, truck):
        return FeasibilityResult.reject(
            FeasibilityReason.PORTS_BUSY, ActionKind.CHARGE, target_node=decision.target_node
        )
    return decision


def _leg_energy_check(env: Any, truck: Any, target_node: int, kind: ActionKind):
    """Reachability and worst-case energy of one leg (as in evaluate_joint_route)."""
    target_node = int(target_node)
    if int(truck.current_node) == target_node:
        return FeasibilityResult.allow(kind, target_node=target_node, required_energy=0.0)
    realized = realized_route_costs(env, truck, target_node)
    try:
        nominal = float(env.transport_graph.get_path_energy(int(truck.current_node), target_node))
    except (KeyError, ValueError):
        nominal = math.inf
    if not math.isfinite(nominal) or nominal < 0.0:
        return FeasibilityResult.reject(
            FeasibilityReason.UNREACHABLE, kind, target_node=target_node, required_energy=math.inf
        )
    required = (
        float(realized["energy_kwh"]) if realized is not None
        else nominal * _energy_multiplier(env)
    )
    engine = getattr(env, "request_engine", None)
    if engine is not None and realized is None:
        # Drayage with a load-dependent energy model scales loaded legs.
        required *= engine.energy_scale(truck.truck_id)
    if required > float(truck.current_battery) + 1e-9:
        return FeasibilityResult.reject(
            FeasibilityReason.INSUFFICIENT_ENERGY, kind, target_node=target_node,
            required_energy=required,
        )
    return FeasibilityResult.allow(kind, target_node=target_node, required_energy=required)


def _nominal_hours(env: Any, source: int, target: int) -> float:
    if int(source) == int(target):
        return 0.0
    try:
        value = float(env.transport_graph.get_time_distance(int(source), int(target)))
    except (KeyError, ValueError):
        return math.inf
    return value if math.isfinite(value) and value >= 0.0 else math.inf


def evaluate_depot_action(env: Any, truck: Any) -> FeasibilityResult:
    """Legacy: return and retire. Request engine: drive to the depot (no retire)."""
    depot = int(env.joint_instance.depot_node)
    engine = getattr(env, "request_engine", None)
    if engine is None:
        return extension_route_check(env, truck, evaluate_joint_route(
            truck=truck,
            truck_state=env.truck_states[truck.truck_id],
            target_node=depot,
            realized_leg=realized_route_costs(env, truck, depot),
            transport_graph=env.transport_graph,
            charging_nodes=env.charging_nodes,
            task_registry=env.task_registry,
            depot_node=depot,
            energy_multiplier=_energy_multiplier(env),
            current_time=float(env.global_clock),
            unavailable_charging_nodes={
                int(node)
                for node, available in env.charging_station.station_available.items()
                if not available
            },
        ))
    kind = ActionKind.DEPOT
    if env.truck_states[truck.truck_id] != "ready":
        return FeasibilityResult.reject(FeasibilityReason.INVALID_TRUCK_STATE, kind, target_node=depot)
    if int(truck.current_node) == depot:
        return FeasibilityResult.reject(
            FeasibilityReason.SAME_LOCATION, kind, target_node=depot, required_energy=0.0
        )
    if engine.is_drayage and engine.container(truck.truck_id) is not None:
        return FeasibilityResult.reject(FeasibilityReason.CONTAINER_ON_BOARD, kind, target_node=depot)
    if engine.registry.on_board(truck.truck_id) and not engine.allow_preemptive_returns:
        # Voccia et al. (2019): no preemptive depot returns.
        return FeasibilityResult.reject(FeasibilityReason.REQUESTS_ON_BOARD, kind, target_node=depot)
    energy = _leg_energy_check(env, truck, depot, kind)
    if not energy.feasible:
        return energy
    objection = _component_leg_checks(env, truck, depot, kind)
    return energy if objection is None else objection


def evaluate_request_slot(env: Any, truck: Any, slot: int) -> FeasibilityResult:
    """Slot ``r`` is request ``r`` (plan section 4.3)."""
    from EVRoutingEnv.models.core.requests import RequestStatus

    kind = ActionKind.CUSTOMER
    engine = env.request_engine
    registry = engine.registry
    if slot >= len(registry) or registry.get(slot).status is RequestStatus.HIDDEN:
        return FeasibilityResult.reject(FeasibilityReason.REQUEST_HIDDEN, kind)
    request = registry.get(slot)
    tid = truck.truck_id
    if env.truck_states[tid] != "ready":
        return FeasibilityResult.reject(FeasibilityReason.INVALID_TRUCK_STATE, kind)
    if request.status is RequestStatus.LOST:
        return FeasibilityResult.reject(FeasibilityReason.REQUEST_LOST, kind)
    if request.status is RequestStatus.ON_BOARD:
        if request.owner != tid:
            return FeasibilityResult.reject(FeasibilityReason.NOT_ON_BOARD, kind)
        target = int(request.delivery.node)
        if target == int(truck.current_node):
            # The truck is at the destination with this cargo: unload, no travel.
            return FeasibilityResult.allow(kind, target_node=target, required_energy=0.0)
        energy = _leg_energy_check(env, truck, target, kind)
        if not energy.feasible:
            return energy
        objection = _component_leg_checks(env, truck, target, kind)
        return energy if objection is None else objection
    if request.status is not RequestStatus.OPEN:
        return FeasibilityResult.reject(FeasibilityReason.TASK_UNAVAILABLE, kind)
    if request.pickup is None:
        raise RuntimeError("request-engine requests always carry a pickup stop")
    target = int(request.pickup.node)
    on_board = registry.on_board(tid)
    if engine.is_drayage and engine.container(tid) is not None:
        # One container at a time; only stations and its own destination.
        return FeasibilityResult.reject(FeasibilityReason.CONTAINER_ON_BOARD, kind, target_node=target)
    if (
        not engine.is_drayage
        and on_board
        and target != int(truck.current_node)
        and not engine.allow_preemptive_returns
    ):
        # Li et al. (2022) formulation (1): a trip carries what it loaded at
        # departure; reloading needs a (preemptive) return to the depot.
        return FeasibilityResult.reject(FeasibilityReason.MANIFEST_LOCKED, kind, target_node=target)
    if registry.on_board_demand(tid) + request.demand > float(truck.payload_capacity) + 1e-9:
        return FeasibilityResult.reject(FeasibilityReason.PAYLOAD_EXCEEDED, kind, target_node=target)
    now = float(env.global_clock)
    if engine.deadline_type == "hard" or engine.is_drayage:
        arrival = now + _nominal_hours(env, int(truck.current_node), target)
        if not math.isfinite(arrival) or arrival > request.pickup.latest + 1e-9:
            return FeasibilityResult.reject(
                FeasibilityReason.TIME_WINDOW_EXPIRED, kind, target_node=target
            )
    energy = _leg_energy_check(env, truck, target, kind)
    if not energy.feasible:
        return energy
    objection = _component_leg_checks(env, truck, target, kind)
    return energy if objection is None else objection


def evaluate_extra_action(env: Any, truck: Any, kind: ActionKind) -> FeasibilityResult:
    """Feasibility of the appended QUEUE, REST_*, WAIT and END_SHIFT actions."""
    from EVRoutingEnv.models.core.hours_of_service import TOLERANCE

    node = int(truck.current_node)
    if env.truck_states[truck.truck_id] != "ready":
        return FeasibilityResult.reject(FeasibilityReason.INVALID_TRUCK_STATE, kind, target_node=node)
    stations = getattr(env, "public_stations", None)
    if kind is ActionKind.QUEUE:
        if node not in env.charging_station.station_available:
            return FeasibilityResult.reject(FeasibilityReason.NOT_AT_CHARGER, kind, target_node=node)
        if stations is not None and stations.awaiting_decision(env, truck):
            return FeasibilityResult.allow(kind, target_node=node)
        return FeasibilityResult.reject(FeasibilityReason.QUEUE_NOT_NEEDED, kind, target_node=node)
    if stations is not None and stations.committed(truck.truck_id):
        return FeasibilityResult.reject(FeasibilityReason.COMMITTED_TO_CHARGE, kind, target_node=node)
    if kind in (ActionKind.REST_BREAK, ActionKind.REST_DAILY):
        hos = env.hos
        clock = hos.clock(truck.truck_id)
        if kind is ActionKind.REST_BREAK:
            needed = clock.driving_since_break > TOLERANCE
        else:
            needed = hos.regime.daily_rest_hours is not None and (
                clock.shift_start is not None or clock.driving_in_shift > TOLERANCE
            )
        if not needed:
            return FeasibilityResult.reject(FeasibilityReason.REST_NOT_REQUIRED, kind, target_node=node)
        return FeasibilityResult.allow(kind, target_node=node)
    engine = env.request_engine
    if kind is ActionKind.WAIT:
        if not engine.future_information_possible(float(env.global_clock)):
            return FeasibilityResult.reject(FeasibilityReason.NO_FUTURE_RELEASE, kind, target_node=node)
        return FeasibilityResult.allow(kind, target_node=node)
    if kind is ActionKind.END_SHIFT:
        if node != int(env.joint_instance.depot_node):
            return FeasibilityResult.reject(FeasibilityReason.NOT_AT_DEPOT, kind, target_node=node)
        if engine.registry.on_board(truck.truck_id) or engine.container(truck.truck_id) is not None:
            return FeasibilityResult.reject(FeasibilityReason.REQUESTS_ON_BOARD, kind, target_node=node)
        return FeasibilityResult.allow(kind, target_node=node)
    raise ValueError(f"unknown appended action kind {kind}")
