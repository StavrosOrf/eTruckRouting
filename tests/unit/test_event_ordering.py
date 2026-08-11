"""Tests for deterministic, monotone event scheduling."""

import heapq

import pytest

from EVRoutingEnv.models.environment.event_handlers import Event, EventType


def test_physical_completions_precede_new_decisions_at_same_time() -> None:
    events = [
        Event(
            time=2.0,
            truck_id=0,
            event_type=EventType.TRUCK_READY,
            data={"reason": "initial"},
        ),
        Event(
            time=2.0,
            truck_id=2,
            event_type=EventType.TRUCK_READY,
            data={"reason": "unloading_complete"},
        ),
        Event(
            time=2.0,
            truck_id=1,
            event_type=EventType.TRUCK_ROUTING,
        ),
    ]
    heapq.heapify(events)

    ordered = [heapq.heappop(events) for _ in range(3)]
    assert [event.event_type for event in ordered] == [
        EventType.TRUCK_ROUTING,
        EventType.TRUCK_READY,
        EventType.TRUCK_READY,
    ]
    assert [event.priority for event in ordered] == [0, 1, 2]


def test_equal_events_use_truck_id_as_deterministic_tie_breaker() -> None:
    events = [
        Event(1.0, truck_id, EventType.TRUCK_READY)
        for truck_id in [3, 1, 2, 0]
    ]
    heapq.heapify(events)

    assert [heapq.heappop(events).truck_id for _ in range(4)] == [0, 1, 2, 3]


@pytest.mark.parametrize("time", [-1.0, float("inf"), float("nan")])
def test_invalid_event_time_is_rejected(time: float) -> None:
    with pytest.raises(ValueError, match="event time"):
        Event(time, 0, EventType.TRUCK_READY)


def test_negative_truck_id_is_rejected() -> None:
    with pytest.raises(ValueError, match="truck_id"):
        Event(0.0, -1, EventType.TRUCK_READY)
