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
) -> FeasibilityResult:
    """Evaluate one routing destination using hard, observable constraints."""
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
    if int(truck.current_node) == target_node:
        return FeasibilityResult.reject(
            FeasibilityReason.SAME_LOCATION,
            action_kind,
            target_node=target_node,
            required_energy=0.0,
        )

    if action_kind is ActionKind.CUSTOMER:
        task = task_registry.task_for_node(target_node)
        if (
            task.preassigned_to is not None
            and task.preassigned_to != truck.truck_id
        ):
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
        if not truck.can_accept_demand(task.demand):
            return FeasibilityResult.reject(
                FeasibilityReason.PAYLOAD_EXCEEDED,
                action_kind,
                target_node=target_node,
            )
        if math.isfinite(task.latest_service):
            try:
                nominal_travel_time = float(
                    transport_graph.get_time_distance(
                        int(truck.current_node),
                        target_node,
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
    elif action_kind is ActionKind.DEPOT:
        # These rejections are reported with the return leg's energy attached.
        # The verdict is unchanged -- only the diagnostic is richer -- so that a
        # policy can price the mandatory depot return before the last customer
        # is served, rather than discovering the cost once it is too late.
        if not task_registry.all_served():
            return FeasibilityResult.reject(
                FeasibilityReason.CUSTOMERS_REMAIN,
                action_kind,
                target_node=target_node,
                required_energy=_nominal_leg_energy(
                    transport_graph, truck, target_node, energy_multiplier
                ),
            )
        if not truck.return_to_depot_pending:
            return FeasibilityResult.reject(
                FeasibilityReason.DEPOT_RETURN_NOT_REQUIRED,
                action_kind,
                target_node=target_node,
                required_energy=_nominal_leg_energy(
                    transport_graph, truck, target_node, energy_multiplier
                ),
            )

    if not math.isfinite(energy_multiplier) or energy_multiplier < 1.0:
        raise ValueError("energy_multiplier must be finite and at least 1.0")
    try:
        nominal_energy = float(
            transport_graph.get_path_energy(int(truck.current_node), target_node)
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
    required_energy = nominal_energy * float(energy_multiplier)
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
            evaluate_joint_route(
                truck=truck,
                truck_state=truck_state,
                target_node=charger_node,
                transport_graph=env.transport_graph,
                charging_nodes=env.charging_nodes,
                task_registry=env.task_registry,
                depot_node=env.joint_instance.depot_node,
                energy_multiplier=energy_multiplier,
                current_time=float(env.global_clock),
                unavailable_charging_nodes=unavailable_chargers,
            )
        )

    for offset in range(env.fixed_num_stops):
        if offset + 1 >= len(truck.delivery_sequence):
            decisions.append(
                FeasibilityResult.reject(
                    FeasibilityReason.EMPTY_ACTION_SLOT,
                    ActionKind.CUSTOMER,
                )
            )
            continue
        decisions.append(
            evaluate_joint_route(
                truck=truck,
                truck_state=truck_state,
                target_node=truck.delivery_sequence[offset + 1],
                transport_graph=env.transport_graph,
                charging_nodes=env.charging_nodes,
                task_registry=env.task_registry,
                depot_node=env.joint_instance.depot_node,
                energy_multiplier=energy_multiplier,
                current_time=float(env.global_clock),
                unavailable_charging_nodes=unavailable_chargers,
            )
        )

    decisions.append(
        evaluate_joint_route(
            truck=truck,
            truck_state=truck_state,
            target_node=env.joint_instance.depot_node,
            transport_graph=env.transport_graph,
            charging_nodes=env.charging_nodes,
            task_registry=env.task_registry,
            depot_node=env.joint_instance.depot_node,
            energy_multiplier=energy_multiplier,
            current_time=float(env.global_clock),
            unavailable_charging_nodes=unavailable_chargers,
        )
    )

    if env.charging_action_mode == "target_soc":
        for target_soc in env.charge_action_values:
            decisions.append(
                evaluate_target_soc_charge(
                    truck=truck,
                    truck_state=truck_state,
                    charger_node=truck.current_node,
                    charging_nodes=env.charging_nodes,
                    target_soc=float(target_soc),
                    station_available=env.charging_station.station_available.get(
                        int(truck.current_node),
                        False,
                    ),
                )
            )
    else:
        for charge_hours in env.charge_action_values:
            decisions.append(
                evaluate_duration_charge(
                    truck=truck,
                    truck_state=truck_state,
                    charger_node=truck.current_node,
                    charging_nodes=env.charging_nodes,
                    charge_hours=float(charge_hours),
                    station_available=env.charging_station.station_available.get(
                        int(truck.current_node),
                        False,
                    ),
                )
            )

    if len(decisions) != env.action_space.n:
        raise RuntimeError(
            f"feasibility engine produced {len(decisions)} actions for "
            f"action space of size {env.action_space.n}"
        )
    return decisions


def _nominal_leg_energy(
    transport_graph: Any,
    truck: Any,
    target_node: int,
    energy_multiplier: float,
) -> float | None:
    """Pessimistic energy for one leg, or ``None`` when the pair is unreachable."""
    try:
        nominal = float(
            transport_graph.get_path_energy(int(truck.current_node), int(target_node))
        )
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(nominal) or nominal < 0.0:
        return None
    return nominal * float(energy_multiplier)


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
