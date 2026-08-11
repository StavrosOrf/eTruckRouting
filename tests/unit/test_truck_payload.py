"""Payload-capacity tests for joint fleet routing."""

import pytest

from EVRoutingEnv.models.core.truck import Truck


def _truck(capacity: float = 10.0) -> Truck:
    return Truck(
        truck_id=0,
        truck_type="electric",
        delivery_sequence=[1],
        initial_battery=100.0,
        battery_capacity=100.0,
        base_speed=40.0,
        enable_flexible_delivery_order=True,
        payload_capacity=capacity,
    )


def test_customer_service_consumes_payload_once() -> None:
    truck = _truck()
    assert truck.can_accept_demand(6.0)

    truck.complete_customer_service(
        task_id=7,
        demand=6.0,
        timestamp=2.0,
        node_id=99,
    )

    assert truck.remaining_payload == 4.0
    assert truck.served_task_ids == [7]
    assert not truck.can_accept_demand(5.0)

    with pytest.raises(ValueError, match="already served"):
        truck.complete_customer_service(7, 1.0, 3.0, 99)


def test_invalid_payload_capacity_is_rejected() -> None:
    with pytest.raises(ValueError, match="payload_capacity"):
        _truck(capacity=0.0)

