"""Tests for fleet-level customer ownership and service transitions."""

import pytest

from EVRoutingEnv.models.core.customer import (
    CustomerTask,
    FleetTaskRegistry,
    TaskStatus,
)


def _registry() -> FleetTaskRegistry:
    return FleetTaskRegistry(
        [
            CustomerTask(0, 101, demand=2.0, base_service_time=0.25),
            CustomerTask(1, 202, demand=5.0, base_service_time=0.50),
        ]
    )


def test_registry_rejects_duplicate_customer_nodes() -> None:
    with pytest.raises(ValueError, match="duplicate customer node"):
        FleetTaskRegistry(
            [
                CustomerTask(0, 101, demand=2.0, base_service_time=0.25),
                CustomerTask(1, 101, demand=3.0, base_service_time=0.25),
            ]
        )


def test_available_tasks_respect_remaining_payload() -> None:
    registry = _registry()
    assert [task.node_id for task in registry.available_tasks(3.0)] == [101]


def test_claim_is_atomic() -> None:
    registry = _registry()
    task = registry.claim(101, truck_id=4, timestamp=1.0)

    assert task.status is TaskStatus.CLAIMED
    assert task.claimed_by == 4
    with pytest.raises(ValueError, match="not available"):
        registry.claim(101, truck_id=5, timestamp=1.0)


def test_only_claiming_truck_can_service_task() -> None:
    registry = _registry()
    registry.claim(101, truck_id=4, timestamp=1.0)

    with pytest.raises(ValueError, match="not truck 5"):
        registry.start_service(101, truck_id=5, timestamp=2.0)


def test_service_lifecycle_and_counts() -> None:
    registry = _registry()
    registry.claim(101, truck_id=4, timestamp=1.0)
    registry.start_service(101, truck_id=4, timestamp=2.0)
    task = registry.complete_service(101, truck_id=4, timestamp=2.5)

    assert task.status is TaskStatus.SERVED
    assert task.served_by == 4
    assert task.served_at == 2.5
    assert registry.counts() == {
        "unassigned": 1,
        "claimed": 0,
        "in_service": 0,
        "served": 1,
    }
    assert not registry.all_served()


def test_claim_can_be_released_before_service_completion() -> None:
    registry = _registry()
    registry.claim(101, truck_id=4, timestamp=1.0)
    task = registry.release_claim(101, truck_id=4)

    assert task.status is TaskStatus.UNASSIGNED
    assert task.claimed_by is None
    assert [item.node_id for item in registry.available_tasks()] == [101, 202]


@pytest.mark.parametrize("demand", [0.0, -1.0, float("inf"), float("nan")])
def test_invalid_customer_demand_is_rejected(demand: float) -> None:
    with pytest.raises(ValueError, match="demand"):
        CustomerTask(0, 101, demand=demand, base_service_time=0.25)


@pytest.mark.parametrize("timestamp", [-1.0, float("inf"), float("nan")])
def test_invalid_transition_timestamp_is_rejected(timestamp: float) -> None:
    with pytest.raises(ValueError, match="timestamp"):
        _registry().claim(101, truck_id=0, timestamp=timestamp)


def test_service_timestamps_must_be_monotone() -> None:
    registry = _registry()
    registry.claim(101, truck_id=0, timestamp=2.0)
    with pytest.raises(ValueError, match="before task claim"):
        registry.start_service(101, truck_id=0, timestamp=1.0)

    registry.start_service(101, truck_id=0, timestamp=2.5)
    with pytest.raises(ValueError, match="before it starts"):
        registry.complete_service(101, truck_id=0, timestamp=2.0)


def test_negative_truck_id_is_rejected() -> None:
    with pytest.raises(ValueError, match="truck_id"):
        _registry().claim(101, truck_id=-1, timestamp=0.0)
