"""Fleet-facing facades for public stations with outside traffic.

``PublicChargingStation`` (queue_model ``dynamic_mmc``) puts every public
station's ``StationQueue`` behind the existing ``ChargingStation`` interface:
free ports are up ports minus fleet sessions, outside sessions and
reservations. ``check_charger_gating`` and ``wake_waiting_trucks`` remain the
entry points from the environment. Under ``queue_policy: kullman2021`` an
arriving truck gets a queue-or-leave epoch instead of joining automatically,
and a queued truck is committed until it reaches the head, where it must
charge (Kullman, Goodson & Mendoza 2021, sec. 2.1-2.2).

``StationaryWaitChargingStation`` (``iid_stationary_wait``) draws, on each
visit, an independent stationary M/M/c wait (Keskin, Akhavan-Tabatabaei &
Catay 2019, generalized to c ports) and then applies the legacy fleet FCFS.

The facades hold no reference to the environment, so certification and
perfect-information copies duplicate them safely.
"""

from __future__ import annotations

import math

from EVRoutingEnv.models.simulation import queueing
from EVRoutingEnv.models.simulation.charging_station import ChargingStation
from EVRoutingEnv.models.simulation.public_station_queue import (
    StationParameters,
    StationQueue,
)


class _PublicMixin:
    """Configuration shared by both public-station models."""

    def _init_public(self, config) -> None:
        self.config = config
        self.streams = None
        self.dirty_nodes: set[int] = set()
        self._parameters: dict[int, StationParameters] = {}
        self._long_run: dict[int, float] = {}

    def station_parameters(self, node: int) -> StationParameters:
        cached = self._parameters.get(int(node))
        if cached is None:
            cached = self._parameters[int(node)] = self._build_parameters(int(node))
        return cached

    def _build_parameters(self, node: int) -> StationParameters:
        cfg = self.config
        return StationParameters(
            node=int(node),
            ports=int(self.charger_capacity[node]),
            utilization=cfg.utilization,
            service_mean_hours=cfg.mean_service_hours,
            hourly_profile=cfg.hourly_profile,
            initial_state=cfg.initial_state,
            outages=cfg.outages.enabled,
            availability=cfg.outages.port_availability,
            mean_repair_hours=cfg.outages.mean_repair_hours,
        )

    def long_run_wait(self, node: int) -> float:
        cached = self._long_run.get(int(node))
        if cached is None:
            p = self.station_parameters(node)
            cached = self._long_run[int(node)] = queueing.mean_wait(
                p.ports, p.mean_arrival_rate, p.service_rate
            )
        return cached


class PublicChargingStation(_PublicMixin, ChargingStation):
    """``dynamic_mmc``: one lazy (or eager) ``StationQueue`` per public station."""

    def __init__(self, *args, public_config, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_public(public_config)
        self.queues: dict[int, StationQueue] = {}

    # -- episode -------------------------------------------------------------

    def begin_episode(self, streams) -> None:
        self.streams = streams
        self.queues = {
            node: StationQueue(self.station_parameters(node), streams)
            for node in self.charging_nodes
        }
        self.memory: dict = {}  # shared: node -> obs; per_truck: (truck, node) -> obs
        self.reservation_tokens: dict[int, int] = {}
        self.scheduled: dict[int, tuple[float, int]] = {}  # truck -> (time, token)
        self.eager_pending: dict[int, float] = {}
        self.released_without_charge: list[tuple[int, int, float]] = []
        self.dirty_nodes = set(self.charging_nodes) if self.config.debug_eager_public_queues else set()
        self._immediate: list[tuple[int, int, float]] = []
        self.now = 0.0
        # Truck -> public station where it is queued, reserved or charging.
        self.truck_station: dict[int, int] = {}

    @property
    def kullman(self) -> bool:
        return self.config.queue_policy == "kullman2021"

    @property
    def eager(self) -> bool:
        return self.config.debug_eager_public_queues

    def queue_at(self, node: int, clock: float, *, fleet_arrival: bool = False) -> StationQueue:
        """The station advanced to ``clock`` (never further)."""
        self.now = max(self.now, float(clock))
        station = self.queues[int(node)]
        if not self.eager:
            made = station.advance(clock, fleet_arrival=fleet_arrival)
            self._absorb(int(node), made)
        return station

    def _absorb(self, node: int, made) -> None:
        """Account for every port a station reserved for a fleet truck.

        A reservation with a scheduled wake-up must match its prediction (a
        mismatch would be a bug). Any other (two ports freeing at once, a
        released reservation passing on) is woken at once; one in the past
        would have been missed, which is also a bug.
        """
        for truck_id, time in made:
            scheduled = self.scheduled.get(truck_id)
            if scheduled is not None:
                if abs(scheduled[0] - time) > 1e-9:
                    raise RuntimeError(
                        f"lazy reservation for truck {truck_id} at {time} differs from "
                        f"its prediction {scheduled[0]}"
                    )
                continue
            if time < self.now - 1e-9:
                raise RuntimeError(f"reservation for truck {truck_id} at {time} was missed")
            self._immediate.append((int(truck_id), node, float(time)))

    def _collect(self, node: int, station: StationQueue) -> None:
        """Reservations made outside an advance (join, release, reopen)."""
        made, station.reservations = station.reservations, []
        self._absorb(node, made)

    def _unschedule(self, truck_id: int) -> None:
        """Drop a pending wake-up, scheduled or still to be, and invalidate its token."""
        self.scheduled.pop(truck_id, None)
        self._immediate = [entry for entry in self._immediate if entry[0] != truck_id]
        self.reservation_tokens[truck_id] = self.reservation_tokens.get(truck_id, 0) + 1

    def _station_of(self, truck_id: int) -> int | None:
        return self.truck_station.get(int(truck_id))

    # -- observation (Kullman sec. 2.2 memory; Dastpak et al. 2024) -----------

    def observe(self, truck_id: int, node: int, clock: float, station: StationQueue) -> None:
        record = (station.in_system(), station.busy_ports(), station.queue_length(),
                  station.up_ports(), float(clock))
        key = node if self.config.memory == "shared" else (truck_id, node)
        self.memory[key] = record

    # -- entry points --------------------------------------------------------

    def check_charger_gating(self, truck_id, charger_node, global_clock):
        node = int(charger_node)
        if node not in self.queues:
            return super().check_charger_gating(truck_id, charger_node, global_clock)
        station = self.queue_at(node, global_clock, fleet_arrival=True)
        self.observe(truck_id, node, global_clock, station)
        if station.reserved_for(truck_id) or station.charging(truck_id):
            return True, None
        if station.position(truck_id) is not None:
            return False, None
        if station.has_free_port_for_newcomer():
            return True, None
        if self.kullman:
            # Queue-or-leave epoch: the truck decides; nothing is joined here.
            return True, None
        self.join(truck_id, node, global_clock)
        return False, None

    def join(self, truck_id: int, node: int, clock: float) -> None:
        station = self.queue_at(node, clock, fleet_arrival=True)
        station.fleet_join(truck_id, clock)
        self.truck_station[int(truck_id)] = node
        self._record_queue_state(node, clock, truck_id, "arrive")
        self.waitlist_sequence_counter += 1
        self.charger_waitlist[node].append(
            {"truck_id": truck_id, "sequence": self.waitlist_sequence_counter}
        )
        self._collect(node, station)
        self.dirty_nodes.add(node)

    def start_charging(self, truck_id, charger_node, charge_hours, global_clock):
        node = int(charger_node)
        if node not in self.queues:
            return super().start_charging(truck_id, charger_node, charge_hours, global_clock)
        station = self.queue_at(node, global_clock, fleet_arrival=True)
        if not station.reserved_for(truck_id):
            station.fleet_take_free_port(truck_id, global_clock)
        station.fleet_start(truck_id, global_clock, charge_hours)
        self.truck_station[int(truck_id)] = node
        self.scheduled.pop(truck_id, None)
        self.dirty_nodes.add(node)
        super().start_charging(truck_id, charger_node, charge_hours, global_clock)

    def finish_charging(self, truck_id, charger_node, global_clock):
        node = int(charger_node)
        if node not in self.queues:
            return super().finish_charging(truck_id, charger_node, global_clock)
        # The fleet session ends at exactly this time; the station releases it.
        self.queue_at(node, global_clock)
        self.truck_station.pop(int(truck_id), None)
        super().finish_charging(truck_id, charger_node, global_clock)
        self.dirty_nodes.add(node)

    def wake_waiting_trucks(self, charger_node, global_clock, event_queue, EventType, Event,
                            truck_states=None):
        if int(charger_node) not in self.queues:
            return super().wake_waiting_trucks(
                charger_node, global_clock, event_queue, EventType, Event, truck_states
            )
        self.dirty_nodes.add(int(charger_node))

    def remove_from_waitlist(self, truck_id, charger_node):
        node = int(charger_node)
        station = self.queues.get(node)
        if station is not None and (
            station.position(truck_id) is not None or station.reserved_for(truck_id)
        ):
            station = self.queue_at(node, self.now, fleet_arrival=True)
            self.observe(truck_id, node, self.now, station)
            station.fleet_release(truck_id, self.now)
            self._unschedule(truck_id)
            self._collect(node, station)
            self.dirty_nodes.add(node)
            self.truck_station.pop(int(truck_id), None)
        super().remove_from_waitlist(truck_id, charger_node)

    def release_reservation(self, truck_id: int, node: int, clock: float) -> None:
        """Case A5: nothing to charge at the head; the next entry starts now."""
        station = self.queue_at(node, clock, fleet_arrival=True)
        station.fleet_release(truck_id, clock)
        self._unschedule(truck_id)
        self._collect(int(node), station)
        self.truck_station.pop(int(truck_id), None)
        self.released_without_charge.append((int(truck_id), int(node), float(clock)))
        super().remove_from_waitlist(truck_id, node)
        self.dirty_nodes.add(int(node))

    def set_station_available(self, charger_node, available, clock=None):
        node = int(charger_node)
        if node not in self.queues:
            return super().set_station_available(charger_node, available)
        clock = self.now if clock is None else float(clock)
        released = super().set_station_available(charger_node, available)
        station = self.queue_at(node, clock, fleet_arrival=True)
        if not available:
            released = sorted(set(released) | set(station.close(clock)))
            for truck_id in released:
                self._unschedule(truck_id)
                if self.truck_station.get(int(truck_id)) == node:
                    self.truck_station.pop(int(truck_id))
        else:
            station.reopen(clock)
            self._collect(node, station)
        self.dirty_nodes.add(node)
        return released

    def confirm_reservation(self, truck_id: int, node: int, token: int, clock: float) -> bool:
        """Whether a ``port_reserved`` event is current; advances the station."""
        if self.reservation_tokens.get(truck_id) != token:
            return False
        station = self.queue_at(node, clock)
        if not station.reserved_for(truck_id):
            raise RuntimeError(
                f"truck {truck_id} expected a port at {node} at t={clock} but none was reserved"
            )
        self.scheduled.pop(truck_id, None)
        return True

    # -- event scheduling ----------------------------------------------------

    def drain(self, clock: float, push_ready, push_system) -> None:
        """Schedule reservation events for dirty stations (and eager events)."""
        self.now = max(self.now, float(clock))
        immediate, self._immediate = list(self._immediate), []
        for truck_id, node, time in immediate:
            self._push_reservation(truck_id, node, time, push_ready)
        nodes, self.dirty_nodes = self.dirty_nodes, set()
        for node in sorted(nodes):
            station = self.queues[node]
            if self.eager:
                next_time = station.next_event_time()
                if math.isfinite(next_time) and self.eager_pending.get(node) != next_time:
                    self.eager_pending[node] = next_time
                    push_system(max(next_time, clock), "station_event", node=node)
                continue
            truck_id = station.first_queued_fleet()
            if truck_id is None or truck_id in self.scheduled:
                continue
            time = station.predict_reservation(truck_id)
            if time is not None:
                self._push_reservation(truck_id, node, max(time, clock), push_ready)

    def handle_eager_event(self, node: int, clock: float, push_ready) -> None:
        station = self.queues[int(node)]
        if self.eager_pending.get(int(node)) != clock:
            return  # superseded
        self.eager_pending.pop(int(node), None)
        for truck_id, time in station.advance(clock):
            self._push_reservation(truck_id, int(node), time, push_ready)
        self.dirty_nodes.add(int(node))

    def _push_reservation(self, truck_id: int, node: int, time: float, push_ready) -> None:
        token = self.reservation_tokens.get(truck_id, 0) + 1
        self.reservation_tokens[truck_id] = token
        self.scheduled[truck_id] = (time, token)
        push_ready(truck_id, time, {"reason": "port_reserved", "charger_node": node, "token": token})

    # -- queries for the feasibility engine and features ---------------------

    def committed(self, truck_id: int) -> bool:
        """Kullman: a truck at the head with a reserved port must charge."""
        if not self.kullman:
            return False
        node = self._station_of(truck_id)
        return node is not None and self.queues[node].reserved_for(truck_id)

    def awaiting_decision(self, env, truck) -> bool:
        """Kullman's busy-arrival epoch: queue or leave."""
        if not self.kullman:
            return False
        node = int(truck.current_node)
        if node not in self.queues or not self.station_available.get(node, False):
            return False
        if env.truck_states[truck.truck_id] != "ready" or truck.must_leave_charger:
            return False
        station = self.queue_at(node, env.global_clock, fleet_arrival=True)
        if (
            station.position(truck.truck_id) is not None
            or station.reserved_for(truck.truck_id)
            or station.charging(truck.truck_id)
        ):
            return False
        return not station.has_free_port_for_newcomer()

    def truck_flags(self, truck_id: int) -> list[float]:
        node = self._station_of(truck_id)
        if node is None:
            return [0.0, 0.0]
        station = self.queues[node]
        return [float(station.position(truck_id) is not None), float(self.committed(truck_id))]

    def queue_position(self, truck_id: int) -> int | None:
        node = self._station_of(truck_id)
        return None if node is None else self.queues[node].position(truck_id)

    def has_reservation(self, truck_id: int) -> bool:
        node = self._station_of(truck_id)
        return node is not None and self.queues[node].reserved_for(truck_id)

    def feature_row(self, node: int, active_truck: int | None, clock: float,
                    present: set[int]) -> list[float]:
        """Plan section 5.3; ``present`` are fleet trucks currently at the node."""
        cfg = self.config
        params = self.station_parameters(node)
        long_run = self.long_run_wait(node)
        live = (
            cfg.information == "realtime_queue"
            or (cfg.memory == "shared" and present)
            or (active_truck is not None and active_truck in present)
        )
        indicator = -1.0
        if cfg.information == "occupancy_indicator":
            station = self.queue_at(node, clock, fleet_arrival=True)
            indicator = float(station.up_ports() > 0 and not any(p.free for p in station.ports))
        if live:
            station = self.queue_at(node, clock, fleet_arrival=True)
            in_system, busy, queued, up, observed_at = (
                station.in_system(), station.busy_ports(), station.queue_length(),
                station.up_ports(), clock,
            )
        else:
            key = node if cfg.memory == "shared" else (active_truck, node)
            record = self.memory.get(key)
            if record is None:
                return [-1.0, -1.0, -1.0, -1.0, long_run, cfg.utilization, indicator]
            in_system, busy, queued, up, observed_at = record
        age = max(0.0, clock - observed_at)
        expected = queueing.expected_wait_given_memory(
            params.ports, params.mean_arrival_rate, params.service_rate, in_system, age
        )
        return [float(busy), float(queued), age, float(up), expected, cfg.utilization, indicator]


class StationaryWaitChargingStation(_PublicMixin, ChargingStation):
    """``iid_stationary_wait``: an independent stationary wait on every visit."""

    def __init__(self, *args, public_config, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_public(public_config)

    def begin_episode(self, streams) -> None:
        self.streams = streams
        self.visits = dict.fromkeys(self.charging_nodes, 0)
        self.outside_wait_until: dict[int, tuple[int, float]] = {}
        self.cleared: dict[int, int] = {}  # truck -> station whose outside wait it passed
        self.drawn_waits: list[tuple[int, int, float]] = []
        self.dirty_nodes = set()

    def draw_wait(self, node: int) -> float:
        p = self.station_parameters(node)
        index = self.visits[node]
        self.visits[node] += 1
        uniforms = tuple(self.streams.uniforms("iid_wait", (int(node), index), 2))
        return queueing.sample_stationary_wait(p.ports, p.mean_arrival_rate, p.service_rate, uniforms)

    def check_charger_gating(self, truck_id, charger_node, global_clock):
        node = int(charger_node)
        if node == self.private_node:
            return super().check_charger_gating(truck_id, charger_node, global_clock)
        if self.cleared.get(truck_id) != node:
            pending = self.outside_wait_until.get(truck_id)
            if pending is None or pending[0] != node:
                wait = self.draw_wait(node)
                self.drawn_waits.append((truck_id, node, wait))
                if wait > 0.0:
                    self.outside_wait_until[truck_id] = (node, global_clock + wait)
                    return False, global_clock + wait
            elif global_clock < pending[1] - 1e-9:
                return False, pending[1]
            self.outside_wait_until.pop(truck_id, None)
            self.cleared[truck_id] = node
        return super().check_charger_gating(truck_id, charger_node, global_clock)

    def remove_from_waitlist(self, truck_id, charger_node):
        pending = self.outside_wait_until.get(truck_id)
        if pending is not None and pending[0] == int(charger_node):
            self.outside_wait_until.pop(truck_id, None)
        if self.cleared.get(truck_id) == int(charger_node):
            self.cleared.pop(truck_id, None)
        super().remove_from_waitlist(truck_id, charger_node)

    def drain(self, clock, push_ready, push_system) -> None:
        return None

    def committed(self, truck_id: int) -> bool:
        return False

    def awaiting_decision(self, env, truck) -> bool:
        return False

    def truck_flags(self, truck_id: int) -> list[float]:
        waiting = truck_id in self.outside_wait_until or any(
            e["truck_id"] == truck_id for entries in self.charger_waitlist.values() for e in entries
        )
        return [float(waiting), 0.0]

    def queue_position(self, truck_id: int) -> int | None:
        return None

    def has_reservation(self, truck_id: int) -> bool:
        return truck_id in set().union(*self.pending_wake_trucks.values())

    def feature_row(self, node, active_truck, clock, present) -> list[float]:
        # No queue state exists: only the published prior and its long-run wait.
        return [-1.0, -1.0, -1.0, -1.0, self.long_run_wait(node), self.config.utilization, -1.0]
