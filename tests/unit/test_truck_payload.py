"""Payload-capacity tests for joint fleet routing."""

import numpy as np
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


@pytest.mark.parametrize("capacity", [float("inf"), float("nan")])
def test_nonfinite_payload_capacity_is_rejected(capacity: float) -> None:
    with pytest.raises(ValueError, match="payload_capacity"):
        _truck(capacity=capacity)


def test_empty_delivery_sequence_is_rejected() -> None:
    with pytest.raises(ValueError, match="delivery_sequence"):
        Truck(0, "electric", [], 100.0, 100.0, 40.0)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("initial_battery", -1.0),
        ("initial_battery", float("nan")),
        ("initial_battery", 101.0),
        ("battery_capacity", 0.0),
        ("battery_capacity", float("inf")),
        ("base_speed", 0.0),
        ("base_speed", float("nan")),
    ],
)
def test_invalid_physical_truck_parameters_are_rejected(
    field: str,
    value: float,
) -> None:
    values = {
        "truck_id": 0,
        "truck_type": "electric",
        "delivery_sequence": [1],
        "initial_battery": 100.0,
        "battery_capacity": 100.0,
        "base_speed": 40.0,
    }
    values[field] = value
    with pytest.raises(ValueError, match=field):
        Truck(**values)


def test_invalid_service_metadata_is_rejected_without_consuming_payload() -> None:
    truck = _truck()
    with pytest.raises(ValueError, match="timestamp"):
        truck.complete_customer_service(0, 1.0, -1.0, 10)
    with pytest.raises(ValueError, match="task_id"):
        truck.complete_customer_service(-1, 1.0, 0.0, 10)
    assert truck.remaining_payload == 10.0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("distance", -1.0),
        ("distance", float("nan")),
        ("travel_time", -0.1),
        ("travel_time", float("inf")),
        ("discharge", -1.0),
        ("discharge", float("nan")),
        ("discharge", 101.0),
    ],
)
def test_invalid_travel_values_are_rejected_without_mutation(
    field: str,
    value: float,
) -> None:
    truck = _truck()
    values = {
        "node": 2,
        "distance": 10.0,
        "travel_time": 0.25,
        "discharge": 5.0,
        "timestamp": 0.25,
        "mark_delivery_on_arrival": False,
    }
    values[field] = value

    with pytest.raises(ValueError, match=field):
        truck.move_to_node(**values)

    assert truck.current_node == 1
    assert truck.current_battery == 100.0
    assert truck.total_energy_consumed == 0.0


def test_charging_rejects_overfill_and_invalid_lifecycle() -> None:
    truck = _truck()
    with pytest.raises(RuntimeError, match="not charging"):
        truck.finish_charging(1.0, 0.1, timestamp=0.1)
    with pytest.raises(RuntimeError, match="already full"):
        truck.start_charging(0.0)

    truck.current_battery = 90.0
    truck.start_charging(0.0)
    with pytest.raises(ValueError, match="available battery capacity"):
        truck.finish_charging(11.0, 0.1, timestamp=0.1)
    assert truck.current_battery == 90.0
    assert truck.total_energy_charged == 0.0


def test_randomized_travel_and_charge_sequence_conserves_energy() -> None:
    rng = np.random.default_rng(20260811)
    truck = _truck()
    truck.current_battery = 60.0
    initial_battery = truck.current_battery
    clock = 0.0

    for step in range(100):
        if truck.current_battery < 20.0 or (
            truck.current_battery < 95.0 and rng.random() < 0.4
        ):
            amount = float(
                rng.uniform(0.1, truck.battery_capacity - truck.current_battery)
            )
            duration = amount / 100.0
            truck.start_charging(clock)
            clock += duration
            truck.finish_charging(amount, duration, timestamp=clock)
        else:
            discharge = float(
                rng.uniform(0.1, min(10.0, truck.current_battery - 0.01))
            )
            travel_time = float(rng.uniform(0.01, 1.0))
            clock += travel_time
            truck.move_to_node(
                node=step + 2,
                distance=travel_time * truck.base_speed,
                travel_time=travel_time,
                discharge=discharge,
                timestamp=clock,
                mark_delivery_on_arrival=False,
            )

        expected = (
            initial_battery
            + truck.total_energy_charged
            - truck.total_energy_consumed
        )
        assert 0.0 <= truck.current_battery <= truck.battery_capacity
        assert truck.current_battery == pytest.approx(expected, abs=1e-9)
