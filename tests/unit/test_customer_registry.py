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

