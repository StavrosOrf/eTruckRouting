"""Public DC-fast stations shared with outside traffic (Kullman et al. 2021).

Each station is a pooled FCFS M/M/c/inf queue: outside vehicles arrive as a
Poisson process with rate lambda = u c mu (Kullman sec. 7.1, p_arrive =
u psi_c p_depart), each with an Exp(mu) service time, and never balk or renege.
Fleet trucks join the same queue; their service is the deterministic charging
time they choose. Ports may fail and recover (Sweda et al. 2017; Rempel et al.
2023); a port that fails while busy finishes its session.

Exactness rules, fixed and tested:

* same-time events: departures, then port changes, then arrivals; a fleet
  truck arriving at the same instant as an outside vehicle goes first;
* a port is reserved for a queued fleet truck the instant it reaches the head;
* all randomness is keyed by station and event index (common random numbers),
  so fleet actions never change the outside realization.

``StationQueue`` advances lazily: ``advance(t)`` processes its own events up to
``t`` and never beyond the caller's clock. Stations interact only through fleet
trucks, so lazy advancement is exact. ``predict_reservation`` computes, on a
private copy, when the first queued fleet truck will get a port; every entry
ahead of it is already known and later arrivals cannot overtake it.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

from EVRoutingEnv.models.simulation import queueing


EPS = 1e-12
# Same-time processing order.
_DEPARTURE, _PORT_CHANGE, _ARRIVAL = 0, 1, 2


@dataclass
class QueueEntry:
    kind: str  # "outside" | "fleet"
    key: int  # outside arrival index k, or truck id
    arrival: float
    service: float | None  # outside: drawn at arrival; fleet: set when charging starts


@dataclass
class Port:
    index: int
    timeline_up: bool = True
    next_change: float = math.inf
    change_index: int = 0
    occupant: QueueEntry | None = None
    busy_until: float | None = None  # None while a fleet reservation awaits its decision

    @property
    def free(self) -> bool:
        return self.timeline_up and self.occupant is None


@dataclass
class StationParameters:
    node: int
    ports: int
    utilization: float
    service_mean_hours: float
    hourly_profile: tuple[float, ...] | None = None
    initial_state: str = "stationary"
    outages: bool = False
    availability: float = 1.0
    mean_repair_hours: float = 24.0

    @property
    def service_rate(self) -> float:
        return 1.0 / self.service_mean_hours

    @property
    def mean_arrival_rate(self) -> float:
        return self.utilization * self.ports * self.service_rate

    def arrival_rate(self, time: float) -> float:
        if self.hourly_profile is None:
            return self.mean_arrival_rate
        return self.mean_arrival_rate * self.hourly_profile[math.floor(time) % 24]

    @property
    def max_arrival_rate(self) -> float:
        if self.hourly_profile is None:
            return self.mean_arrival_rate
        return self.mean_arrival_rate * max(self.hourly_profile)


@dataclass
class StationLog:
    """Outside-traffic realization, for oracles and the eager reference."""

    arrivals: list[tuple[int, float, float]] = field(default_factory=list)  # (k, time, service)
    starts: list[tuple[str, int, float, int]] = field(default_factory=list)  # (kind, key, time, port)
    departures: list[tuple[str, int, float, int]] = field(default_factory=list)
    port_changes: list[tuple[int, float, bool]] = field(default_factory=list)
    balked_on_closure: int = 0


class StationQueue:
    """One public station with outside traffic and port outages."""

    def __init__(self, parameters: StationParameters, streams, *, record: bool = True):
        self.p = parameters
        self.streams = streams
        self.clock = 0.0
        self.closed = False
        self.queue: list[QueueEntry] = []
        self.ports = [Port(index=i) for i in range(int(parameters.ports))]
        self.log = StationLog() if record else None
        self.reservations: list[tuple[int, float]] = []  # fleet (truck, time) since last drain
        self._arrivals_k = 0  # accepted outside arrivals so far
        self._candidate_j = 0  # thinning candidates so far
        self._next_arrival: tuple[float, int] | None = None
        self._initialize()

    # -- keyed draws -----------------------------------------------------------

    def _exp(self, stream: str, key, mean: float) -> float:
        return self.streams.exponential(stream, (self.p.node, key), mean)

    def _initialize(self) -> None:
        node = self.p.node
        if self.p.outages:
            mtbf = (
                math.inf if self.p.availability >= 1.0
                else self.p.mean_repair_hours * self.p.availability / (1.0 - self.p.availability)
            )
            for port in self.ports:
                # Stationary start: up with probability a; residual times are
                # fresh exponentials by memorylessness.
                up = float(self.streams.uniforms("port_state", (node, port.index, 0))[0]) < self.p.availability
                port.timeline_up = up
                port.change_index = 1
                mean = mtbf if up else self.p.mean_repair_hours
                port.next_change = (
                    math.inf if not math.isfinite(mean)
                    else self._exp("port_state", (port.index, 1), mean)
                )
        if self.p.initial_state == "stationary":
            count = queueing.sample_stationary_count(
                int(self.p.ports),
                self.p.mean_arrival_rate,
                self.p.service_rate,
                tuple(self.streams.uniforms("station_initial", node, 2)),
            )
            for i in range(count):
                service = self._exp("station_service", ("initial", i), self.p.service_mean_hours)
                # Initial vehicles keep their FCFS order; residual services are
                # fresh exponentials by memorylessness.
                entry = QueueEntry("outside", -1 - i, -1.0 + i * 1e-6, service)
                if self.log is not None:
                    self.log.arrivals.append((entry.key, entry.arrival, service))
                port = next((port for port in self.ports if port.free), None)
                if port is None:
                    self.queue.append(entry)
                else:
                    port.occupant, port.busy_until = entry, service
                    if self.log is not None:
                        self.log.starts.append(("outside", entry.key, 0.0, port.index))
        self._schedule_next_arrival()

    def _schedule_next_arrival(self) -> None:
        if self.p.mean_arrival_rate <= 0.0:
            self._next_arrival = None
            return
        time = self._next_arrival[0] if self._next_arrival is not None else 0.0
        if self.p.hourly_profile is None:
            gap = self._exp("station_arrivals", self._arrivals_k, 1.0 / self.p.mean_arrival_rate)
            self._next_arrival = (time + gap, self._arrivals_k)
            return
        # Lewis & Shedler (1979) thinning at the profile's peak rate.
        peak = self.p.max_arrival_rate
        while True:
            draws = self.streams.uniforms("station_arrivals", (self.p.node, "thin", self._candidate_j), 2)
            self._candidate_j += 1
            time += -math.log1p(-float(draws[0])) / peak
            if float(draws[1]) * peak < self.p.arrival_rate(time):
                self._next_arrival = (time, self._arrivals_k)
                return

    # -- event loop ----------------------------------------------------------

    def _next_event(self):
        """Earliest pending event as (time, kind, index); ties by kind, then index."""
        best = None
        for port in self.ports:
            if port.occupant is not None and port.busy_until is not None:
                candidate = (port.busy_until, _DEPARTURE, port.index)
                best = candidate if best is None or candidate < best else best
            if math.isfinite(port.next_change):
                candidate = (port.next_change, _PORT_CHANGE, port.index)
                best = candidate if best is None or candidate < best else best
        if self._next_arrival is not None:
            time, k = self._next_arrival
            candidate = (time, _ARRIVAL, k)
            best = candidate if best is None or candidate < best else best
        return best

    def next_event_time(self) -> float:
        event = self._next_event()
        return math.inf if event is None else event[0]

    def advance(self, until: float, *, fleet_arrival: bool = False) -> list[tuple[int, float]]:
        """Process own events up to ``until``; return fleet reservations made.

        With ``fleet_arrival`` an outside arrival at exactly ``until`` is left
        for later, so a fleet truck arriving at the same instant goes first.
        """
        until = float(until)
        if until < self.clock - 1e-9:
            raise RuntimeError(f"station {self.p.node} cannot move back from {self.clock} to {until}")
        while True:
            event = self._next_event()
            if event is None or event[0] > until + EPS:
                break
            time, kind, index = event
            if kind == _ARRIVAL and fleet_arrival and time >= until - EPS:
                break
            self.clock = max(self.clock, time)
            if kind == _DEPARTURE:
                self._depart(self.ports[index], time)
            elif kind == _PORT_CHANGE:
                self._change_port(self.ports[index], time)
            else:
                self._outside_arrival(time)
            self._dispatch(time)
        self.clock = max(self.clock, until)
        made, self.reservations = self.reservations, []
        return made

    def _depart(self, port: Port, time: float) -> None:
        entry = port.occupant
        if self.log is not None:
            self.log.departures.append((entry.kind, entry.key, time, port.index))
        port.occupant, port.busy_until = None, None

    def _change_port(self, port: Port, time: float) -> None:
        port.timeline_up = not port.timeline_up
        port.change_index += 1
        mean = (
            self.p.mean_repair_hours * self.p.availability / (1.0 - self.p.availability)
            if port.timeline_up else self.p.mean_repair_hours
        )
        port.next_change = time + self._exp("port_state", (port.index, port.change_index), mean)
        if self.log is not None:
            self.log.port_changes.append((port.index, time, port.timeline_up))

    def _outside_arrival(self, time: float) -> None:
        _, k = self._next_arrival
        self._arrivals_k += 1
        service = self._exp("station_service", k, self.p.service_mean_hours)
        self._schedule_next_arrival()
        if self.log is not None:
            self.log.arrivals.append((k, time, service))
        if self.closed:
            if self.log is not None:
                self.log.balked_on_closure += 1
            return
        self.queue.append(QueueEntry("outside", k, time, service))

    def _dispatch(self, time: float) -> None:
        """Hand every free port to the head of the queue, in port order."""
        while self.queue:
            port = next((port for port in self.ports if port.free), None)
            if port is None:
                return
            entry = self.queue.pop(0)
            port.occupant = entry
            if entry.kind == "outside":
                port.busy_until = time + entry.service
            else:
                port.busy_until = None  # reserved until the truck decides
                self.reservations.append((entry.key, time))
            if self.log is not None:
                self.log.starts.append((entry.kind, entry.key, time, port.index))

    # -- fleet interface -----------------------------------------------------

    def _fleet_port(self, truck_id: int) -> Port | None:
        return next(
            (
                port for port in self.ports
                if port.occupant is not None
                and port.occupant.kind == "fleet"
                and port.occupant.key == truck_id
            ),
            None,
        )

    def has_free_port_for_newcomer(self) -> bool:
        return not self.queue and any(port.free for port in self.ports)

    def fleet_join(self, truck_id: int, time: float) -> None:
        """Join the FCFS queue at ``time`` (the station must be at ``time``)."""
        if self.closed:
            raise RuntimeError(f"station {self.p.node} is closed")
        if self.position(truck_id) is not None or self._fleet_port(truck_id) is not None:
            raise RuntimeError(f"truck {truck_id} is already at station {self.p.node}")
        self.queue.append(QueueEntry("fleet", int(truck_id), float(time), None))
        self._dispatch(time)

    def fleet_take_free_port(self, truck_id: int, time: float) -> None:
        """A truck that found a free port and an empty queue reserves it now."""
        if not self.has_free_port_for_newcomer():
            raise RuntimeError(f"station {self.p.node} has no free port for truck {truck_id}")
        self.fleet_join(truck_id, time)
        if self._fleet_port(truck_id) is None:
            raise RuntimeError("free-port reservation failed")
        self.reservations = [r for r in self.reservations if r[0] != truck_id]

    def fleet_start(self, truck_id: int, time: float, hours: float) -> None:
        port = self._fleet_port(truck_id)
        if port is None or port.busy_until is not None:
            raise RuntimeError(f"truck {truck_id} holds no reserved port at {self.p.node}")
        port.occupant.service = float(hours)
        port.busy_until = float(time) + float(hours)

    def fleet_release(self, truck_id: int, time: float) -> bool:
        """Leave the station: give up a reserved port or a queue place."""
        port = self._fleet_port(truck_id)
        if port is not None:
            if port.busy_until is not None and port.busy_until > time + 1e-9:
                raise RuntimeError(f"truck {truck_id} cannot leave a session in progress")
            port.occupant, port.busy_until = None, None
            if self.log is not None:
                self.log.departures.append(("fleet", int(truck_id), time, port.index))
            self._dispatch(time)
            return True
        before = len(self.queue)
        self.queue = [e for e in self.queue if not (e.kind == "fleet" and e.key == truck_id)]
        return len(self.queue) != before

    def position(self, truck_id: int) -> int | None:
        """Entries ahead of a queued fleet truck (Kullman's queue position)."""
        for index, entry in enumerate(self.queue):
            if entry.kind == "fleet" and entry.key == truck_id:
                return index
        return None

    def reserved_for(self, truck_id: int) -> bool:
        port = self._fleet_port(truck_id)
        return port is not None and port.busy_until is None

    def charging(self, truck_id: int) -> bool:
        port = self._fleet_port(truck_id)
        return port is not None and port.busy_until is not None

    def first_queued_fleet(self) -> int | None:
        return next((e.key for e in self.queue if e.kind == "fleet"), None)

    def predict_reservation(self, truck_id: int) -> float | None:
        """When will a port be reserved for this queued truck? None if not yet known.

        Runs on a private copy, so the live state never passes the caller's
        clock. Unknown only while a fleet truck ahead holds a reservation whose
        charging time it has not chosen yet.
        """
        if self.position(truck_id) is None:
            return None
        shadow = copy.copy(self)
        shadow.ports = [copy.copy(port) for port in self.ports]
        shadow.queue = list(self.queue)
        shadow.log = None
        shadow.reservations = []
        shadow._next_arrival = None  # later arrivals queue behind it
        while True:
            if shadow.reserved_for(truck_id):
                return next(time for key, time in shadow.reservations if key == truck_id)
            if any(
                port.occupant is not None
                and port.occupant.kind == "fleet"
                and port.busy_until is None
                for port in shadow.ports
            ):
                # Another truck holds a port for a charge it has not chosen yet;
                # it may release at once (case A5), so the time is not known.
                return None
            event = shadow._next_event()
            if event is None:
                return None  # blocked by an undecided fleet reservation, or ports never return
            time, kind, index = event
            if kind == _DEPARTURE:
                shadow._depart(shadow.ports[index], time)
            elif kind == _PORT_CHANGE:
                shadow._change_port(shadow.ports[index], time)
            shadow.clock = time
            shadow._dispatch(time)

    # -- closure and observation ---------------------------------------------

    def close(self, time: float) -> list[int]:
        """Station closure: queued vehicles leave; sessions in progress finish."""
        self.closed = True
        released = [e.key for e in self.queue if e.kind == "fleet"]
        if self.log is not None:
            self.log.balked_on_closure += sum(e.kind == "outside" for e in self.queue)
        self.queue = []
        for port in self.ports:
            if port.occupant is not None and port.busy_until is None:
                released.append(port.occupant.key)
                if self.log is not None:
                    self.log.departures.append(("fleet", port.occupant.key, time, port.index))
                port.occupant = None
        return released

    def reopen(self, time: float) -> None:
        self.closed = False
        self._dispatch(time)

    def busy_ports(self) -> int:
        return sum(port.occupant is not None for port in self.ports)

    def up_ports(self) -> int:
        """Ports that are working or still finishing a session."""
        return sum(port.timeline_up or port.occupant is not None for port in self.ports)

    def queue_length(self) -> int:
        return len(self.queue)

    def in_system(self) -> int:
        return self.busy_ports() + self.queue_length()

    def outside_busy(self) -> int:
        return sum(p.occupant is not None and p.occupant.kind == "outside" for p in self.ports)

    def fleet_busy(self) -> int:
        return sum(
            p.occupant is not None and p.occupant.kind == "fleet" and p.busy_until is not None
            for p in self.ports
        )

    def reserved(self) -> int:
        return sum(
            p.occupant is not None and p.occupant.kind == "fleet" and p.busy_until is None
            for p in self.ports
        )
