"""Unit tests for hard joint-routing feasibility decisions."""

import math

from EVRoutingEnv.models.core.customer import CustomerTask, FleetTaskRegistry
from EVRoutingEnv.models.core.truck import Truck
from EVRoutingEnv.state.feasibility import (
    ActionKind,
    FeasibilityReason,
    evaluate_duration_charge,
    evaluate_joint_route,
    evaluate_target_soc_charge,
)


class _EnergyGraph:
    def __init__(self, energies: dict[tuple[int, int], float]) -> None:
        self.energies = energies

    def get_path_energy(self, origin: int, destination: int) -> float:
        return self.energies.get((origin, destination), math.inf)

    def get_time_distance(self, origin: int, destination: int) -> float:
        return 1.0 if (origin, destination) in self.energies else math.inf


def _truck(*, node: int = 1, battery: float = 50.0, payload: float = 5.0) -> Truck:
    return Truck(
        truck_id=0,
        truck_type="electric",
        delivery_sequence=[node, 10, 11],
        initial_battery=battery,
        battery_capacity=100.0,
        base_speed=40.0,
        enable_flexible_delivery_order=True,
        payload_capacity=payload,
    )


def _registry() -> FleetTaskRegistry:
    return FleetTaskRegistry(
        [
            CustomerTask(0, 10, demand=2.0, base_service_time=0.25),
            CustomerTask(1, 11, demand=6.0, base_service_time=0.25),
        ]
    )


def _route(target: int, **overrides):
    values = {
        "truck": _truck(),
        "truck_state": "ready",
        "target_node": target,
        "transport_graph": _EnergyGraph(
            {(1, 10): 20.0, (1, 11): 20.0, (1, 20): 30.0}
        ),
        "charging_nodes": [20],
        "task_registry": _registry(),
        "depot_node": 1,
        "energy_multiplier": 1.0,
    }
    values.update(overrides)
    return evaluate_joint_route(**values)


def test_available_customer_with_payload_and_energy_is_feasible() -> None:
    result = _route(10)
    assert result.feasible
    assert result.action_kind is ActionKind.CUSTOMER
    assert result.required_energy == 20.0


def test_claimed_customer_is_rejected() -> None:
    registry = _registry()
    registry.claim(10, truck_id=3, timestamp=0.0)
    result = _route(10, task_registry=registry)
    assert result.reason is FeasibilityReason.TASK_UNAVAILABLE


def test_payload_excess_is_rejected() -> None:
    result = _route(11)
    assert result.reason is FeasibilityReason.PAYLOAD_EXCEEDED


def test_conservative_energy_bound_is_enforced() -> None:
    result = _route(10, truck=_truck(battery=23.9), energy_multiplier=1.2)
    assert result.reason is FeasibilityReason.INSUFFICIENT_ENERGY
    assert result.required_energy == 24.0


def test_unreachable_unknown_and_same_location_are_distinct() -> None:
    unreachable = _route(10, transport_graph=_EnergyGraph({}))
    unknown = _route(99)
    same_location = _route(1)

    assert unreachable.reason is FeasibilityReason.UNREACHABLE
    assert unknown.reason is FeasibilityReason.UNKNOWN_DESTINATION
    assert same_location.reason is FeasibilityReason.SAME_LOCATION


def test_depot_is_only_available_after_service_and_when_return_required() -> None:
    registry = _registry()
    truck = _truck(node=5)
    before_service = _route(1, truck=truck, task_registry=registry)
    assert before_service.reason is FeasibilityReason.CUSTOMERS_REMAIN

    for task in registry.tasks():
        registry.claim(task.node_id, truck_id=0, timestamp=0.0)
        registry.start_service(task.node_id, truck_id=0, timestamp=0.0)
        registry.complete_service(task.node_id, truck_id=0, timestamp=0.0)

    no_return = _route(1, truck=truck, task_registry=registry)
    assert no_return.reason is FeasibilityReason.DEPOT_RETURN_NOT_REQUIRED

    truck.return_to_depot_pending = True
    feasible = _route(
        1,
        truck=truck,
        task_registry=registry,
        transport_graph=_EnergyGraph({(5, 1): 10.0}),
    )
    assert feasible.feasible
    assert feasible.action_kind is ActionKind.DEPOT


def test_duration_charge_has_no_hidden_location_or_full_battery_fallback() -> None:
    away = evaluate_duration_charge(
        truck=_truck(node=1),
        truck_state="ready",
        charger_node=20,
        charging_nodes=[20],
        charge_hours=1.0,
    )
    at_charger = _truck(node=20)
    full = _truck(node=20, battery=100.0)
    mismatch = evaluate_duration_charge(
        truck=at_charger,
        truck_state="ready",
        charger_node=21,
        charging_nodes=[20, 21],
        charge_hours=1.0,
    )

    assert away.reason is FeasibilityReason.NOT_AT_CHARGER
    assert mismatch.reason is FeasibilityReason.CHARGER_MISMATCH
    assert (
        evaluate_duration_charge(
            truck=full,
            truck_state="ready",
            charger_node=20,
            charging_nodes=[20],
            charge_hours=1.0,
        ).reason
        is FeasibilityReason.BATTERY_FULL
    )


def test_invalid_state_duration_and_must_leave_are_rejected() -> None:
    truck = _truck(node=20)
    invalid_state = evaluate_duration_charge(
        truck=truck,
        truck_state="routing",
        charger_node=20,
        charging_nodes=[20],
        charge_hours=1.0,
    )
    invalid_duration = evaluate_duration_charge(
        truck=truck,
        truck_state="ready",
        charger_node=20,
        charging_nodes=[20],
        charge_hours=0.0,
    )
    truck.must_leave_charger = True
    must_leave = evaluate_duration_charge(
        truck=truck,
        truck_state="ready",
        charger_node=20,
        charging_nodes=[20],
        charge_hours=1.0,
    )

    assert invalid_state.reason is FeasibilityReason.INVALID_TRUCK_STATE
    assert invalid_duration.reason is FeasibilityReason.INVALID_CHARGE_DURATION
    assert must_leave.reason is FeasibilityReason.MUST_LEAVE_CHARGER


def test_target_soc_must_be_valid_and_above_current_soc() -> None:
    truck = _truck(node=20, battery=50.0)
    below = evaluate_target_soc_charge(
        truck=truck,
        truck_state="ready",
        charger_node=20,
        charging_nodes=[20],
        target_soc=0.5,
    )
    invalid = evaluate_target_soc_charge(
        truck=truck,
        truck_state="ready",
        charger_node=20,
        charging_nodes=[20],
        target_soc=1.01,
    )
    feasible = evaluate_target_soc_charge(
        truck=truck,
        truck_state="ready",
        charger_node=20,
        charging_nodes=[20],
        target_soc=0.6,
    )

    assert below.reason is FeasibilityReason.TARGET_SOC_NOT_ABOVE_CURRENT
    assert invalid.reason is FeasibilityReason.INVALID_TARGET_SOC
    assert feasible.feasible


def test_closed_charger_is_rejected_for_routing_and_charging() -> None:
    route = _route(20, unavailable_charging_nodes={20})
    charge = evaluate_target_soc_charge(
        truck=_truck(node=20, battery=50.0),
        truck_state="ready",
        charger_node=20,
        charging_nodes=[20],
        target_soc=0.6,
        station_available=False,
    )

    assert route.reason is FeasibilityReason.CHARGER_UNAVAILABLE
    assert charge.reason is FeasibilityReason.CHARGER_UNAVAILABLE


def test_hard_time_window_allows_early_wait_but_rejects_late_arrival() -> None:
    early_registry = FleetTaskRegistry(
        [
            CustomerTask(
                0,
                10,
                demand=2.0,
                base_service_time=0.25,
                earliest_service=5.0,
                latest_service=6.0,
            )
        ]
    )
    late_registry = FleetTaskRegistry(
        [
            CustomerTask(
                0,
                10,
                demand=2.0,
                base_service_time=0.25,
                earliest_service=0.0,
                latest_service=1.5,
            )
        ]
    )

    early = _route(10, task_registry=early_registry, current_time=1.0)
    late = _route(10, task_registry=late_registry, current_time=1.0)

    assert early.feasible
    assert late.reason is FeasibilityReason.TIME_WINDOW_EXPIRED
