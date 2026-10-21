"""Request identities, validated transitions and the reveal-order envelope."""

import math

import pytest

from EVRoutingEnv.models.core.requests import (
    Request,
    RequestKind,
    RequestRegistry,
    RequestStatus,
    Stop,
)


def _request(request_id, node=10, demand=1.0, reveal=0.0, release=0.0, pickup=0):
    return Request(
        request_id=request_id,
        kind=RequestKind.SAME_DAY,
        pickup=None if pickup is None else Stop(pickup, release, math.inf, 0.2),
        delivery=Stop(node, 0.0, 10.0, 0.1),
        demand=demand,
        release_time=release,
        reveal_time=reveal,
    )


def test_full_lifecycle_through_pickup_and_delivery() -> None:
    registry = RequestRegistry(4)
    registry.add(_request(0))
    registry.reveal(0, 0.0)
    registry.claim_pickup(0, truck_id=2, timestamp=0.0)
    registry.start_loading(0, 2, 0.5)
    registry.finish_loading(0, 2)
    assert [r.request_id for r in registry.on_board(2)] == [0]
    registry.claim_delivery(0, 2, 1.0)
    registry.start_service(0, 2, 3.0)
    served = registry.complete_service(0, 2, 3.1, late=False)
    assert served.status is RequestStatus.SERVED and served.owner == 2
    assert registry.on_board(2) == []


def test_late_delivery_records_the_delay() -> None:
    registry = RequestRegistry(1)
    registry.add(_request(0))
    registry.reveal(0, 0.0)
    registry.claim_pickup(0, 1, 0.0)
    registry.start_loading(0, 1, 0.0)
    registry.finish_loading(0, 1)
    registry.claim_delivery(0, 1, 0.0)
    registry.start_service(0, 1, 12.5)
    request = registry.complete_service(0, 1, 12.6, late=True)
    assert request.status is RequestStatus.SERVED_LATE
    assert request.delay_hours == pytest.approx(2.5)


@pytest.mark.parametrize(
    "transition",
    [
        lambda r: r.start_loading(0, 1, 0.0),  # never claimed
        lambda r: r.claim_delivery(0, 1, 0.0),  # pickup first
        lambda r: r.finish_loading(0, 1),
        lambda r: r.start_service(0, 1, 0.0),
    ],
)
def test_illegal_transitions_raise(transition) -> None:
    registry = RequestRegistry(1)
    registry.add(_request(0))
    registry.reveal(0, 0.0)
    with pytest.raises(RuntimeError):
        transition(registry)


def test_hidden_requests_cannot_be_claimed_and_never_leave_the_registry() -> None:
    registry = RequestRegistry(2)
    registry.add(_request(0))
    registry.add(_request(1, reveal=5.0, release=5.0))
    registry.reveal(0, 0.0)
    with pytest.raises(RuntimeError, match="illegal transition"):
        registry.claim_pickup(1, 0, 1.0)
    assert [row["request_id"] for row in registry.snapshot()] == [0]
    assert len(registry.revealed()) == 1


def test_only_the_owner_moves_a_claimed_request() -> None:
    registry = RequestRegistry(1)
    registry.add(_request(0))
    registry.reveal(0, 0.0)
    registry.claim_pickup(0, 3, 0.0)
    with pytest.raises(RuntimeError, match="owned by 3"):
        registry.start_loading(0, 4, 0.0)
    registry.release_pickup(0, 3)
    assert registry.get(0).status is RequestStatus.OPEN and registry.get(0).owner is None


def test_open_and_claimed_requests_can_be_lost_but_loaded_ones_cannot() -> None:
    registry = RequestRegistry(3)
    for i in range(3):
        registry.add(_request(i))
        registry.reveal(i, 0.0)
    registry.lose(0, 1.0)
    registry.claim_pickup(1, 0, 0.0)
    registry.lose(1, 1.0)  # C3: lost while a truck drives to the pickup
    assert registry.get(1).owner is None
    registry.claim_pickup(2, 0, 0.0)
    registry.start_loading(2, 0, 0.0)
    with pytest.raises(RuntimeError):
        registry.lose(2, 1.0)


def test_ids_follow_reveal_order_and_are_never_reused() -> None:
    registry = RequestRegistry(3)
    registry.add(_request(0, reveal=0.0, release=0.0))
    with pytest.raises(RuntimeError, match="reveal-order"):
        registry.add(_request(2))
    with pytest.raises(RuntimeError, match="reveal order"):
        registry.add(Request(1, RequestKind.SAME_DAY, None, Stop(5), 1.0, 0.0, -1.0))


def test_C14_exceeding_the_envelope_raises_rather_than_truncating() -> None:
    registry = RequestRegistry(1)
    registry.add(_request(0))
    with pytest.raises(ValueError, match="max_requests"):
        registry.add(_request(1))


def test_C2_two_live_requests_at_one_node_keep_distinct_slots() -> None:
    registry = RequestRegistry(2)
    registry.add(_request(0, node=42))
    registry.add(_request(1, node=42))
    registry.reveal(0, 0.0)
    registry.reveal(1, 0.0)
    assert [r.request_id for r in registry.live_at_node(42)] == [0, 1]
    registry.lose(0, 1.0)
    assert [r.request_id for r in registry.live_at_node(42)] == [1]


def test_request_validation() -> None:
    with pytest.raises(ValueError, match="before it is revealed"):
        _request(0, reveal=2.0, release=1.0)
    with pytest.raises(ValueError, match="demand"):
        _request(0, demand=0.0)
    with pytest.raises(ValueError, match="latest"):
        Stop(1, 5.0, 4.0)
