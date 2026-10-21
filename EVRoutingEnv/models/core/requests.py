"""Requests with an identity separate from their stops (dynamic modes).

One ``Request`` has an optional pickup stop and a delivery stop (Savelsbergh &
Sol 1995; Berbeglia, Cordeau & Laporte 2010). Same-day freight picks up at the
depot (Li, Archetti & Ljubic 2024, formulation (1)); electric drayage moves one
container from an origin to a destination (Xiao et al. 2025). Several live
requests may share a node, so the registry keeps a node -> requests multimap.

Action slot ``r`` is request ``r``: identifiers are allocated in reveal order,
never reused, and bounded by the configured envelope. Every status change goes
through a registry method that checks the source status and raises
``RuntimeError`` on an illegal transition.

The legacy static mode keeps ``FleetTaskRegistry``, which is the
``static_delivery`` special case (no pickup, starts open, one task per node);
its trajectories are frozen by the golden traces.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from enum import StrEnum


class RequestKind(StrEnum):
    STATIC_DELIVERY = "static_delivery"
    SAME_DAY = "same_day"
    DRAYAGE_IMPORT = "drayage_import"
    DRAYAGE_EXPORT = "drayage_export"


class RequestStatus(StrEnum):
    """Lifecycle; append-only, codes are exposed as features."""

    HIDDEN = "hidden"
    OPEN = "open"
    CLAIMED_PICKUP = "claimed_pickup"
    LOADING = "loading"
    ON_BOARD = "on_board"
    CLAIMED_DELIVERY = "claimed_delivery"
    IN_SERVICE = "in_service"
    SERVED = "served"
    LOST = "lost"  # its pickup deadline passed while it was open
    SERVED_LATE = "served_late"  # delivered after its deadline


TERMINAL = frozenset({RequestStatus.SERVED, RequestStatus.SERVED_LATE, RequestStatus.LOST})

# Legal transitions. Releasing a claim returns a request to where it was.
_TRANSITIONS = {
    RequestStatus.HIDDEN: {RequestStatus.OPEN},
    RequestStatus.OPEN: {
        RequestStatus.CLAIMED_PICKUP,
        RequestStatus.CLAIMED_DELIVERY,  # static request: no pickup stop
        RequestStatus.LOST,
    },
    RequestStatus.CLAIMED_PICKUP: {
        RequestStatus.LOADING,
        RequestStatus.OPEN,
        RequestStatus.LOST,
    },
    RequestStatus.LOADING: {RequestStatus.ON_BOARD},
    RequestStatus.ON_BOARD: {RequestStatus.CLAIMED_DELIVERY},
    RequestStatus.CLAIMED_DELIVERY: {
        RequestStatus.IN_SERVICE,
        RequestStatus.ON_BOARD,
        RequestStatus.OPEN,
    },
    RequestStatus.IN_SERVICE: {RequestStatus.SERVED, RequestStatus.SERVED_LATE},
    RequestStatus.SERVED: set(),
    RequestStatus.SERVED_LATE: set(),
    RequestStatus.LOST: set(),
}


@dataclass(frozen=True)
class Stop:
    node: int
    earliest: float = 0.0
    latest: float = math.inf
    service_hours: float = 0.0

    def __post_init__(self) -> None:
        if int(self.node) < 0:
            raise ValueError("stop node must be non-negative")
        if not math.isfinite(self.earliest) or self.earliest < 0.0:
            raise ValueError("stop earliest time must be finite and non-negative")
        if math.isnan(self.latest) or self.latest < self.earliest - 1e-9:
            raise ValueError("stop latest time precedes its earliest time")
        if not math.isfinite(self.service_hours) or self.service_hours < 0.0:
            raise ValueError("stop service time must be finite and non-negative")


@dataclass
class Request:
    request_id: int
    kind: RequestKind
    pickup: Stop | None
    delivery: Stop
    demand: float
    release_time: float
    reveal_time: float
    status: RequestStatus = RequestStatus.HIDDEN
    owner: int | None = None
    claimed_at: float | None = None
    loaded_at: float | None = None
    service_started_at: float | None = None
    served_at: float | None = None
    lost_at: float | None = None
    delay_hours: float = 0.0
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not math.isfinite(self.demand) or self.demand <= 0.0:
            raise ValueError("request demand must be positive")
        if not math.isfinite(self.release_time) or self.release_time < 0.0:
            raise ValueError("release time must be finite and non-negative")
        if self.reveal_time > self.release_time + 1e-9:
            raise ValueError("a request cannot be released before it is revealed")

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL

    @property
    def is_served_on_time(self) -> bool:
        return self.status is RequestStatus.SERVED

    def as_dict(self) -> dict:
        result = asdict(self)
        result["kind"] = self.kind.value
        result["status"] = self.status.value
        return result


class RequestRegistry:
    """Request identities, ownership and validated lifecycle transitions."""

    def __init__(self, max_requests: int):
        if isinstance(max_requests, bool) or not isinstance(max_requests, int) or max_requests < 1:
            raise ValueError("max_requests must be a positive integer")
        self.max_requests = max_requests
        self._requests: list[Request] = []
        self._by_node: dict[int, set[int]] = defaultdict(set)

    # -- creation and lookup -------------------------------------------------

    def add(self, request: Request) -> Request:
        """Register a request in the next slot; ids follow reveal order."""
        if request.request_id != len(self._requests):
            raise RuntimeError(
                f"request id {request.request_id} breaks reveal-order allocation "
                f"(expected {len(self._requests)})"
            )
        if len(self._requests) >= self.max_requests:
            raise ValueError(
                f"requests.max_requests={self.max_requests} exceeded; raise the envelope "
                "rather than dropping requests"
            )
        if self._requests and request.reveal_time < self._requests[-1].reveal_time - 1e-9:
            raise RuntimeError("requests must be added in reveal order")
        self._requests.append(request)
        return request

    def reveal(self, request_id: int, timestamp: float) -> Request:
        request = self._transition(request_id, RequestStatus.OPEN)
        for stop in (request.pickup, request.delivery):
            if stop is not None:
                self._by_node[int(stop.node)].add(request.request_id)
        return request

    def __len__(self) -> int:
        return len(self._requests)

    def requests(self) -> list[Request]:
        return list(self._requests)

    def revealed(self) -> list[Request]:
        return [r for r in self._requests if r.status is not RequestStatus.HIDDEN]

    def get(self, request_id: int) -> Request:
        try:
            return self._requests[int(request_id)]
        except IndexError as error:
            raise KeyError(f"no request {request_id}") from error

    def live_at_node(self, node: int) -> list[Request]:
        return [
            self._requests[i]
            for i in sorted(self._by_node.get(int(node), ()))
            if not self._requests[i].is_terminal
        ]

    def on_board(self, truck_id: int) -> list[Request]:
        return [
            r for r in self._requests
            if r.owner == int(truck_id)
            and r.status in {
                RequestStatus.ON_BOARD,
                RequestStatus.CLAIMED_DELIVERY,
                RequestStatus.IN_SERVICE,
            }
        ]

    def on_board_demand(self, truck_id: int) -> float:
        return sum(r.demand for r in self.on_board(truck_id)) + sum(
            r.demand
            for r in self._requests
            if r.owner == int(truck_id) and r.status is RequestStatus.LOADING
        )

    # -- transitions ---------------------------------------------------------

    def _transition(self, request_id: int, target: RequestStatus) -> Request:
        request = self.get(request_id)
        if target not in _TRANSITIONS[request.status]:
            raise RuntimeError(
                f"request {request.request_id}: illegal transition "
                f"{request.status.value} -> {target.value}"
            )
        request.status = target
        return request

    def _require_owner(self, request: Request, truck_id: int) -> None:
        if request.owner != int(truck_id):
            raise RuntimeError(
                f"request {request.request_id} is owned by {request.owner}, not truck {truck_id}"
            )

    def claim_pickup(self, request_id: int, truck_id: int, timestamp: float) -> Request:
        request = self.get(request_id)
        if request.pickup is None:
            raise RuntimeError(f"request {request_id} has no pickup stop")
        self._transition(request_id, RequestStatus.CLAIMED_PICKUP)
        request.owner = int(truck_id)
        request.claimed_at = float(timestamp)
        return request

    def release_pickup(self, request_id: int, truck_id: int) -> Request:
        request = self.get(request_id)
        self._require_owner(request, truck_id)
        if request.status is not RequestStatus.CLAIMED_PICKUP:
            raise RuntimeError(f"request {request_id} has no pickup claim to release")
        self._transition(request_id, RequestStatus.OPEN)
        request.owner = None
        request.claimed_at = None
        return request

    def start_loading(self, request_id: int, truck_id: int, timestamp: float) -> Request:
        request = self.get(request_id)
        self._require_owner(request, truck_id)
        self._transition(request_id, RequestStatus.LOADING)
        request.loaded_at = float(timestamp)
        return request

    def finish_loading(self, request_id: int, truck_id: int) -> Request:
        request = self.get(request_id)
        self._require_owner(request, truck_id)
        return self._transition(request_id, RequestStatus.ON_BOARD)

    def claim_delivery(self, request_id: int, truck_id: int, timestamp: float) -> Request:
        request = self.get(request_id)
        if request.status is RequestStatus.OPEN:
            if request.pickup is not None:
                raise RuntimeError(f"request {request_id} must be picked up first")
            request.owner = int(truck_id)
            request.claimed_at = float(timestamp)
        else:
            self._require_owner(request, truck_id)
        return self._transition(request_id, RequestStatus.CLAIMED_DELIVERY)

    def release_delivery(self, request_id: int, truck_id: int) -> Request:
        request = self.get(request_id)
        self._require_owner(request, truck_id)
        if request.status is not RequestStatus.CLAIMED_DELIVERY:
            raise RuntimeError(f"request {request_id} has no delivery claim to release")
        if request.pickup is None:
            request.owner = None
            request.claimed_at = None
            return self._transition(request_id, RequestStatus.OPEN)
        return self._transition(request_id, RequestStatus.ON_BOARD)

    def start_service(self, request_id: int, truck_id: int, timestamp: float) -> Request:
        request = self.get(request_id)
        self._require_owner(request, truck_id)
        self._transition(request_id, RequestStatus.IN_SERVICE)
        request.service_started_at = float(timestamp)
        return request

    def complete_service(
        self, request_id: int, truck_id: int, timestamp: float, *, late: bool
    ) -> Request:
        request = self.get(request_id)
        self._require_owner(request, truck_id)
        self._transition(
            request_id, RequestStatus.SERVED_LATE if late else RequestStatus.SERVED
        )
        request.served_at = float(timestamp)
        request.delay_hours = max(
            0.0, float(request.service_started_at or timestamp) - request.delivery.latest
        ) if math.isfinite(request.delivery.latest) else 0.0
        return request

    def lose(self, request_id: int, timestamp: float) -> Request:
        request = self.get(request_id)
        self._transition(request_id, RequestStatus.LOST)
        request.owner = None
        request.lost_at = float(timestamp)
        return request

    # -- summaries -----------------------------------------------------------

    def counts(self) -> dict[str, int]:
        return {
            status.value: sum(r.status is status for r in self._requests)
            for status in RequestStatus
        }

    def snapshot(self) -> list[dict]:
        """Revealed requests only: hidden ones never leave the simulator."""
        return [r.as_dict() for r in self.revealed()]
