"""Canonical typed features shared by flat, set, and graph encoders."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from EVRoutingEnv.models.core.customer import TaskStatus
from EVRoutingEnv.state.feasibility import (
    ActionKind,
    FeasibilityReason,
    joint_action_feasibility,
)


SCHEMA_VERSION = "joint-fleet-v1"

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
)
ACTION_FEATURES = (
    "kind_code",
    "target_node",
    "charge_value",
    "customer_demand",
    "required_energy",
    "feasible",
    "reason_code",
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
)


@dataclass(frozen=True)
class CanonicalFleetFeatures:
    """One lossless, finite-valued feature snapshot for joint routing."""

    schema_version: str
    truck_features: np.ndarray
    customer_features: np.ndarray
    charger_features: np.ndarray
    action_features: np.ndarray
    global_features: np.ndarray
    truck_feature_names: tuple[str, ...] = TRUCK_FEATURES
    customer_feature_names: tuple[str, ...] = CUSTOMER_FEATURES
    charger_feature_names: tuple[str, ...] = CHARGER_FEATURES
    action_feature_names: tuple[str, ...] = ACTION_FEATURES
    global_feature_names: tuple[str, ...] = GLOBAL_FEATURES

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


def extract_canonical_features(env) -> CanonicalFleetFeatures:
    """Extract canonical joint-routing features from current environment state."""
    if not env.joint_routing or env.task_registry is None:
        raise ValueError("canonical joint features require problem.mode=joint_fleet")

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
            )
        )
    }
    truck_rows = []
    for truck in sorted(env.trucks, key=lambda item: item.truck_id):
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
            ]
        )

    task_status_codes = {status: index for index, status in enumerate(TaskStatus)}
    customer_rows = []
    for task in env.task_registry.tasks():
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
            ]
        )

    charger_type_codes = {"Level2": 0, "DCFast": 1}
    charger_rows = []
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
            ]
        )

    decisions = joint_action_feasibility(env)
    action_targets, charge_values = _action_metadata(env)
    kind_codes = {kind: index for index, kind in enumerate(ActionKind)}
    reason_codes = {
        reason: index for index, reason in enumerate(FeasibilityReason)
    }
    action_rows = []
    for decision, target, charge_value in zip(
        decisions,
        action_targets,
        charge_values,
        strict=True,
    ):
        demand = 0.0
        if decision.action_kind is ActionKind.CUSTOMER and target >= 0:
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
        ],
        dtype=np.float32,
    )

    result = CanonicalFleetFeatures(
        schema_version=SCHEMA_VERSION,
        truck_features=np.asarray(truck_rows, dtype=np.float32),
        customer_features=np.asarray(customer_rows, dtype=np.float32),
        charger_features=np.asarray(charger_rows, dtype=np.float32),
        action_features=np.asarray(action_rows, dtype=np.float32),
        global_features=global_features,
    )
    result.validate()
    return result


def _action_metadata(env) -> tuple[list[int], list[float]]:
    if env.active_truck_id is None:
        return ([-1] * env.action_space.n, [0.0] * env.action_space.n)
    truck = env.trucks[env.active_truck_id]
    targets = [int(node) for node in env.charging_nodes]
    for offset in range(env.fixed_num_stops):
        target = (
            int(truck.delivery_sequence[offset + 1])
            if offset + 1 < len(truck.delivery_sequence)
            else -1
        )
        targets.append(target)
    targets.append(int(env.joint_instance.depot_node))
    targets.extend([int(truck.current_node)] * env.num_charge_actions)
    values = [0.0] * env.num_navigation_actions + list(env.charge_action_values)
    return targets, values


def _optional_value(value) -> float:
    return -1.0 if value is None else float(value)


def _finite_or_default(value: float | None, default: float) -> float:
    if value is None or not math.isfinite(value):
        return default
    return float(value)
