"""Dynamic requests: same-day freight and electric drayage (plan Phase C).

Information process. ``euro_neurips_epochs`` reproduces the EURO Meets
NeurIPS 2022 DVRPTW environment (``ortec/euro-neurips-vrp-2022-quickstart``,
``VRPEnvironment._next_observation``, MIT licence): a static context of
customers with windows, demands and service times is drawn first; at every
epoch from ``start_epoch`` to ``end_epoch`` up to ``max_requests_per_epoch``
candidates draw customer, window, demand and service time independently with
replacement; a candidate is kept only if
``max(planning_start + d(depot, c), open) <= close`` and the return
``... + service + d(c, depot) <= depot_close``, with
``planning_start = epoch_start + margin``, which is also its release time.
Requests are revealed at the epoch start, as EURO observations are.

The documented extension keeps a candidate only if a fresh full-battery truck
at the depot at planning start also has a witness with at most one station per
segment, worst-case energy, the hours-of-service rules and the horizon. Drops
caused only by hours of service are counted (case X3).

``poisson`` draws arrival times as a Poisson process (Larsen et al. 2002).
Same-day requests pick up at the depot (Li, Archetti & Ljubic 2024, formulation
(1)); drayage requests move one container from an origin to a destination
(Xiao et al. 2025). Every draw is keyed (common random numbers), and the whole
stream is generated at reset but stays hidden until revealed.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from EVRoutingEnv.models.core.customer import TaskStatus
from EVRoutingEnv.models.core.hours_of_service import (
    DriverClock,
    HosActivity,
    apply,
    can_drive,
    insert_rests,
    rest_hours,
)
from EVRoutingEnv.models.core.requests import (
    Request,
    RequestKind,
    RequestRegistry,
    RequestStatus,
    Stop,
)
from EVRoutingEnv.utils.utils import resolve_network_data_dir


EARTH_RADIUS_KM = 6371.0088
GRAVITY = 9.81

# Kancharla & Ramadurai (2020) eq. (62) parameters, Xiao et al. (2025) Table 10.
KANCHARLA = {"empty_mass_kg": 7000.0, "rolling": 0.01, "drag": 0.8, "air_density": 1.2041,
             "frontal_area_m2": 7.0, "efficiency": 0.89}


def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(h))


def kancharla_power_ratio(container_kg: float, speed_kmh: float) -> float:
    """Loaded / empty tractive power at constant speed on flat ground (eq. 62)."""
    v = speed_kmh / 3.6
    aero = 0.5 * KANCHARLA["air_density"] * KANCHARLA["drag"] * KANCHARLA["frontal_area_m2"] * v * v
    empty = KANCHARLA["empty_mass_kg"] * GRAVITY * KANCHARLA["rolling"] + aero
    loaded = (KANCHARLA["empty_mass_kg"] + container_kg) * GRAVITY * KANCHARLA["rolling"] + aero
    return loaded / empty


def euro_filter(planning_start, depot_to_customer, customer_to_depot, open_, close, service,
                depot_close) -> np.ndarray:
    """EURO ``_next_observation`` feasibility, vectorized over candidates."""
    earliest_arrival = np.maximum(planning_start + depot_to_customer, open_)
    earliest_return = earliest_arrival + service + customer_to_depot
    return (earliest_arrival <= close) & (earliest_return <= depot_close)


@dataclass(frozen=True)
class StaticContext:
    """EURO's static instance / Voccia's known customer locations."""

    customers: np.ndarray
    open: np.ndarray
    close: np.ndarray
    demand: np.ndarray
    service: np.ndarray


class RequestTaskView:
    """Thin ``FleetTaskRegistry``-shaped view for generic consumers.

    It exposes revealed requests as customer tasks (identified by request id,
    located at their delivery node). ``task_for_node`` raises when a node hosts
    more than one live request: such code must be migrated to request ids.
    """

    _STATUS: ClassVar[dict] = {
        RequestStatus.OPEN: TaskStatus.UNASSIGNED,
        RequestStatus.CLAIMED_PICKUP: TaskStatus.CLAIMED,
        RequestStatus.LOADING: TaskStatus.CLAIMED,
        RequestStatus.ON_BOARD: TaskStatus.CLAIMED,
        RequestStatus.CLAIMED_DELIVERY: TaskStatus.CLAIMED,
        RequestStatus.IN_SERVICE: TaskStatus.IN_SERVICE,
        RequestStatus.SERVED: TaskStatus.SERVED,
        RequestStatus.SERVED_LATE: TaskStatus.SERVED,
        RequestStatus.LOST: TaskStatus.SERVED,
    }

    @dataclass(frozen=True)
    class TaskView:
        task_id: int
        node_id: int
        demand: float
        base_service_time: float
        earliest_service: float
        latest_service: float
        status: TaskStatus
        claimed_by: int | None
        claimed_at: float | None
        service_started_at: float | None
        served_at: float | None
        served_on_time: bool
        preassigned_to: None = None
        sequence_position: None = None

        @property
        def is_available(self) -> bool:
            return self.status is TaskStatus.UNASSIGNED

        @property
        def is_served(self) -> bool:
            # Hard deadlines: a late delivery is not served (Voccia et al. 2019).
            return self.served_on_time

    def __init__(self, engine: RequestEngine):
        self.engine = engine
        self.sequential = False

    def _view(self, request: Request) -> RequestTaskView.TaskView:
        return self.TaskView(
            task_id=request.request_id,
            node_id=int(request.delivery.node),
            demand=request.demand,
            base_service_time=request.delivery.service_hours,
            earliest_service=request.delivery.earliest,
            latest_service=request.delivery.latest,
            status=self._STATUS[request.status],
            claimed_by=request.owner,
            claimed_at=request.claimed_at,
            service_started_at=request.service_started_at,
            served_at=request.served_at,
            served_on_time=request.status is RequestStatus.SERVED,
        )

    def tasks(self):
        return [self._view(r) for r in self.engine.registry.revealed()]

    def __iter__(self):
        return iter(self.tasks())

    def __len__(self) -> int:
        return len(self.engine.registry.revealed())

    def task_for_node(self, node_id: int):
        live = self.engine.registry.live_at_node(node_id)
        if len(live) > 1:
            raise RuntimeError(
                f"node {node_id} hosts {len(live)} live requests; use request ids"
            )
        if not live:
            raise KeyError(f"node {node_id} hosts no live request")
        return self._view(live[0])

    def available_tasks(self, remaining_payload=None, *, truck_id=None):
        tasks = [view for view in self.tasks() if view.is_available]
        if remaining_payload is None:
            return tasks
        return [view for view in tasks if view.demand <= float(remaining_payload) + 1e-9]

    def pending_tasks(self):
        return [view for view in self.tasks() if view.status is not TaskStatus.SERVED]

    def all_served(self) -> bool:
        return self.engine.all_resolved_on_time()

    def is_next_customer(self, node_id: int, truck_id: int) -> bool:
        return True

    def route_complete_for(self, truck_id: int) -> bool:
        return self.engine.all_resolved_on_time()

    def counts(self) -> dict[str, int]:
        counts = {status.value: 0 for status in TaskStatus}
        for view in self.tasks():
            counts[view.status.value] += 1
        return counts

    def snapshot(self) -> list[dict]:
        return self.engine.registry.snapshot()


class RequestEngine:
    """Request stream, registry and request-related state of one environment."""

    def __init__(self, extensions, problem_config: dict, delivery_config: dict):
        self.config = extensions.requests
        self.drayage_config = extensions.drayage
        self.hos_config = extensions.hours_of_service
        self.loading = extensions.loading
        self.mode = self.config.mode if self.config.dynamic else "at_depot"
        self.problem_config = problem_config
        self.delivery_config = delivery_config
        self.registry = RequestRegistry(self.config.max_requests)
        self.terminals: tuple[int, ...] = ()
        self._drayage_pool: list[int] | None = None
        self._terminals_near: dict[int, tuple[int, ...]] = {}

    # -- properties used by the feasibility engine ----------------------------

    @property
    def is_drayage(self) -> bool:
        return self.mode == "drayage"

    @property
    def allow_preemptive_returns(self) -> bool:
        return self.config.allow_preemptive_returns

    @property
    def deadline_type(self) -> str:
        # Drayage always uses Xiao's soft destination deadline.
        return "soft" if self.is_drayage else self.config.deadline_type

    def container(self, truck_id: int) -> Request | None:
        if not self.is_drayage:
            return None
        for request in self.registry.requests():
            if request.owner == int(truck_id) and request.status in {
                RequestStatus.LOADING, RequestStatus.ON_BOARD,
                RequestStatus.CLAIMED_DELIVERY, RequestStatus.IN_SERVICE,
            }:
                return request
        return None

    def energy_scale(self, truck_id: int) -> float:
        """Kancharla & Ramadurai (2020) scaling of loaded drayage legs."""
        if not self.is_drayage or self.drayage_config.energy_model != "kancharla_ramadurai_2020":
            return 1.0
        container = self.container(truck_id)
        if container is None:
            return 1.0
        return kancharla_power_ratio(container.extra["container_kg"], self.base_speed_kmh)

    def next_information_time(self, now: float) -> float | None:
        """When the next reveal or release actually happens (wakes a WAIT).

        This is an interrupt, not an observation: a waiting truck wakes when the
        information arrives, and nothing about it is visible before then.
        """
        upcoming = [t for t in self._information_times if t > now and t < self.max_time]
        return min(upcoming) if upcoming else None

    def future_information_possible(self, now: float) -> bool:
        """Whether a later release can still occur, from known information only.

        EURO epochs follow a known schedule; Poisson arrivals can occur until the
        known cutoff; a revealed request may still be awaiting its release. The
        hidden stream itself is never consulted (non-anticipativity).
        """
        if any(t > now and t < self.max_time for t in self._known_schedule):
            return True
        return any(
            r.release_time > now and r.release_time < self.max_time
            for r in self.registry.revealed()
        )

    def slot_target(self, truck, slot: int) -> int:
        if slot >= len(self.registry):
            return -1
        request = self.registry.get(slot)
        if request.status is RequestStatus.HIDDEN:
            return -1
        if request.status is RequestStatus.OPEN and request.pickup is not None:
            return int(request.pickup.node)
        return int(request.delivery.node)

    def nominal_service_hours(self, node: int) -> float:
        live = self.registry.live_at_node(node)
        return max((r.delivery.service_hours for r in live if r.delivery.node == node), default=0.0)

    def all_resolved_on_time(self) -> bool:
        """Every revealed request served on time (vacuously true if none).

        Only revealed requests count: whether more are coming is not known to
        the fleet. At the end of an episode every request has been revealed.
        """
        return all(r.status is RequestStatus.SERVED for r in self.registry.revealed())

    # -- episode generation --------------------------------------------------

    def begin_episode(self, env) -> int:
        """Generate the static context and the whole hidden stream; return the depot."""
        self.streams = env.scenario_random_streams
        self.max_time = float(env.max_time)
        self.base_speed_kmh = float(env.config["truck"]["base_speed"])
        # The envelope: requests.max_requests, or num_stops for depot loading.
        self.registry = RequestRegistry(int(env.fixed_num_stops))
        self._drops: list[tuple[float, str]] = []  # (reveal time, filter)
        self._epoch_starts: list[float] = []
        self.lost_penalized: set[int] = set()
        index, energy, hours = env.transport_graph.cost_matrices()
        self._index, self._energy, self._hours = index, energy, hours
        self._env_static = env
        if self.mode == "at_depot":
            depot = int(env.joint_instance.depot_node)
            self._build_at_depot(env, depot)
        else:
            depot = self._draw_depot(env)
            context = self._static_context(env, depot)
            self.context = context
            candidates = self._candidates(env, depot, context)
            self._register(candidates)
        self.depot = depot
        self._known_schedule = self._schedule_times()
        self._information_times = sorted(
            {r.reveal_time for r in self.registry.requests()}
            | {r.release_time for r in self.registry.requests()}
        )
        return depot

    def _schedule_times(self) -> list[float]:
        """Information times the dispatcher knows in advance."""
        cfg = self.config
        if self.mode == "at_depot":
            return []
        if cfg.arrival_process == "euro_neurips_epochs":
            return list(self._epoch_starts)
        # Poisson: an arrival may come at any time before the cutoff.
        return [min(cfg.request_cutoff_hours, self.max_time)]

    def _draw_depot(self, env) -> int:
        from EVRoutingEnv.models.environment.route_walk import RouteWalkTopology

        topology = RouteWalkTopology.for_graph(
            env.transport_graph, env.charging_nodes,
            float(env.config["truck"]["battery_capacity"]), 1.0,
        )
        customers = [topology.nodes[i] for i in topology.customers]
        if self.is_drayage:
            # Terminals are neither depot nor customer; with a service radius
            # the pool was narrowed once at init.
            customers = self._drayage_pool if self._drayage_pool is not None else [
                node for node in customers if node not in self.terminals
            ]
        self._customer_pool = customers
        pick = int(self.streams.keyed_generator("request_static_context", "depot").integers(len(customers)))
        return int(customers[pick])

    def _static_context(self, env, depot: int) -> StaticContext:
        size = self.config.static_context_size
        pool = [node for node in self._customer_pool if node != depot]
        if len(pool) < size:
            raise ValueError(
                f"requests.static_context_size={size} exceeds the {len(pool)} customer "
                "nodes of this network"
            )
        rng = self.streams.keyed_generator("request_static_context", "customers")
        customers = np.asarray(rng.choice(pool, size=size, replace=False), dtype=np.int64)
        windows = dict(self.config.time_windows)
        enabled = windows.get("enabled", False)
        if enabled:
            rng = self.streams.keyed_generator("request_static_context", "time_windows")
            open_ = rng.uniform(float(windows.get("earliest_min", 0.0)),
                                float(windows.get("earliest_max", 0.0)), size)
            width_min = float(windows.get("window_width_min", 1.0))
            width = rng.uniform(width_min, float(windows.get("window_width_max", width_min)), size)
            close = open_ + width
        else:
            open_ = np.zeros(size)
            close = np.full(size, self.max_time)
        rng = self.streams.keyed_generator("request_static_context", "demand")
        if self.is_drayage:
            demand = np.ones(size)
        else:
            demand = rng.uniform(float(self.problem_config.get("min_customer_demand", 1.0)),
                                 float(self.problem_config.get("max_customer_demand", 1.0)), size)
        base_service = float(self.problem_config.get(
            "base_service_time", self.delivery_config["base_unloading_time"]
        ))
        service = np.full(size, base_service)
        return StaticContext(customers, open_, close, demand, service)

    def resolve_terminals(self, env) -> None:
        """Map each drayage terminal to the nearest non-station node (at init)."""
        if not self.is_drayage:
            return
        from EVRoutingEnv.models.environment.route_walk import RouteWalkTopology

        index, _, hours = env.transport_graph.cost_matrices()
        self._index = index
        self.terminals = self._resolve_terminals(env, depot=-1)
        self._drayage_pool = None
        radius = self.drayage_config.service_radius_hours
        if radius is not None:
            # Drayage is regional: depot and customers lie within a nominal
            # radius of some terminal, both ways. Static, so computed once.
            topology = RouteWalkTopology.for_graph(
                env.transport_graph, env.charging_nodes,
                float(env.config["truck"]["battery_capacity"]), 1.0,
            )
            self._drayage_pool = [
                int(topology.nodes[i]) for i in topology.customers
                if int(topology.nodes[i]) not in self.terminals
                and any(
                    hours[index[int(topology.nodes[i])], index[t]] <= radius
                    and hours[index[t], index[int(topology.nodes[i])]] <= radius
                    for t in self.terminals
                )
            ]
            needed = self.config.static_context_size + 1  # the context and the depot
            if len(self._drayage_pool) < needed:
                raise ValueError(
                    f"drayage.service_radius_hours={radius:g} leaves {len(self._drayage_pool)} "
                    f"network nodes around the terminals, but the static context and depot "
                    f"need {needed}; widen the radius or shrink requests.static_context_size"
                )
            # Each customer is paired only with terminals within the radius.
            self._terminals_near = {
                node: tuple(
                    t for t in self.terminals
                    if hours[index[node], index[t]] <= radius and hours[index[t], index[node]] <= radius
                )
                for node in self._drayage_pool
            }

    def _resolve_terminals(self, env, depot: int) -> tuple[int, ...]:
        path = os.path.join(resolve_network_data_dir(env.config["network"]), "vis_data", "nodes.json")
        with open(path) as handle:
            coordinates = {int(row["node"]): (float(row["lat"]), float(row["lon"])) for row in json.load(handle)}
        chargers = set(map(int, env.charging_nodes))
        candidates = [
            (node, coordinates[int(data["original_id"])])
            for node, data in env.transport_graph.graph.nodes(data=True)
            if node not in chargers and node != depot and int(data["original_id"]) in coordinates
            and node in self._index
        ]
        resolved = []
        for terminal in self.drayage_config.terminals:
            node, place = min(candidates, key=lambda item: haversine_km((terminal.lat, terminal.lon), item[1]))
            distance = haversine_km((terminal.lat, terminal.lon), place)
            if distance > self.drayage_config.max_terminal_distance_km:
                raise ValueError(
                    f"drayage terminal {terminal.name!r} is {distance:.2f} km from the nearest "
                    f"network node (limit {self.drayage_config.max_terminal_distance_km:g} km)"
                )
            resolved.append(int(node))
        return tuple(resolved)

    def _candidates(self, env, depot: int, context: StaticContext) -> list[dict]:
        cfg = self.config
        size = len(context.customers)
        index = self._index
        to_customer = np.array([self._hours[index[depot], index[int(c)]] for c in context.customers])
        to_depot = np.array([self._hours[index[int(c)], index[depot]] for c in context.customers])
        candidates: list[dict] = []
        batches: list[tuple[float, str, int, int]] = []  # (reveal time, source, key, count)
        if cfg.initial_requests:
            batches.append((0.0, "initial", 0, cfg.initial_requests))
        self._epoch_starts = []
        if cfg.arrival_process == "euro_neurips_epochs":
            margin = cfg.dispatch_margin_hours
            start = int(max((context.open.min() - margin) // cfg.epoch_hours, 0))
            end = int(max((context.open.max() - margin) // cfg.epoch_hours, 0))
            for epoch in range(start, end + 1):
                if epoch * cfg.epoch_hours > cfg.request_cutoff_hours + 1e-9:
                    break
                self._epoch_starts.append(epoch * cfg.epoch_hours)
                batches.append((epoch * cfg.epoch_hours, "epoch", epoch, cfg.max_requests_per_epoch))
        else:
            time, k = 0.0, 0
            while True:
                time += self.streams.exponential("request_poisson", k, 1.0 / cfg.poisson_rate_per_hour)
                if time > cfg.request_cutoff_hours:
                    break
                batches.append((time, "poisson", k, 1))
                k += 1
        for reveal, source, key, count in batches:
            draws = {
                field: self.streams.keyed_generator(
                    "request_epoch" if source != "poisson" else "request_poisson",
                    (source, key, field),
                ).integers(size, size=count)
                for field in ("customer", "time_window", "demand", "service")
            }
            release = reveal + (cfg.dispatch_margin_hours if source != "initial" else 0.0)
            customer_index = draws["customer"]
            window_index = draws["time_window"]
            keep = euro_filter(
                release,
                to_customer[customer_index],
                to_depot[customer_index],
                context.open[window_index],
                context.close[window_index],
                context.service[draws["service"]],
                self.max_time,
            )
            self._drops.extend([(reveal, "euro")] * int((~keep).sum()))
            for position in np.flatnonzero(keep):
                candidates.append(self._make_candidate(
                    env, depot, context, reveal, release, source, key, int(position),
                    int(customer_index[position]), int(window_index[position]),
                    int(draws["demand"][position]), int(draws["service"][position]),
                ))
        return [c for c in candidates if c is not None]

    def _make_candidate(self, env, depot, context, reveal, release, source, key, position,
                        customer_i, window_i, demand_i, service_i) -> dict | None:
        customer = int(context.customers[customer_i])
        open_, close = float(context.open[window_i]), float(context.close[window_i])
        demand = float(context.demand[demand_i])
        service = float(context.service[service_i])
        if self.is_drayage:
            u = self.streams.uniforms("request_epoch", (source, key, "drayage", position), 3)
            terminals = self._terminals_near.get(customer) or self.terminals
            terminal = terminals[int(u[1] * len(terminals)) % len(terminals)]
            importing = float(u[0]) < self.drayage_config.import_share
            origin, destination = (terminal, customer) if importing else (customer, terminal)
            low, high = self.drayage_config.container_mass_kg
            candidate = {
                "kind": RequestKind.DRAYAGE_IMPORT if importing else RequestKind.DRAYAGE_EXPORT,
                "pickup": Stop(origin, max(release, open_), math.inf, self.drayage_config.origin_service_hours),
                "delivery": Stop(destination, 0.0, close, self.drayage_config.destination_service_hours),
                "demand": 1.0, "reveal": reveal, "release": release,
                "extra": {"container_kg": float(low + (high - low) * float(u[2]))},
            }
        else:
            # The truck leaves only after loading, so the last pickup that can
            # still meet the window nominally starts loading_service_hours
            # before ``close - d(depot, customer)`` (plan 7.2: loads whose
            # window can no longer be met nominally are masked).
            latest_pickup = (
                close - self._hours[self._index[depot], self._index[customer]]
                - self.config.loading_service_hours
            )
            candidate = {
                "kind": RequestKind.SAME_DAY,
                "pickup": Stop(
                    depot, release,
                    max(release, latest_pickup) if self.deadline_type == "hard" else math.inf,
                    self.config.loading_service_hours,
                ),
                "delivery": Stop(customer, open_, close, service),
                "demand": demand, "reveal": reveal, "release": release, "extra": {},
            }
            if self.deadline_type == "hard" and latest_pickup < release - 1e-9:
                # EURO's filter has no loading time; this drop is the extension's.
                self._drops.append((reveal, "witness"))
                return None
        verdict = self._witness(env, depot, candidate)
        if verdict != "ok":
            self._drops.append((reveal, "hos" if verdict == "hos" else "witness"))
            return None
        return candidate

    def _register(self, candidates: list[dict]) -> None:
        candidates.sort(key=lambda c: c["reveal"])
        if len(candidates) > self.registry.max_requests:
            raise ValueError(
                f"the request stream has {len(candidates)} requests but the envelope holds "
                f"{self.registry.max_requests} (requests.max_requests); raise it: requests "
                "are never dropped to fit"
            )
        for request_id, candidate in enumerate(candidates):
            self.registry.add(Request(
                request_id=request_id,
                kind=candidate["kind"],
                pickup=candidate["pickup"],
                delivery=candidate["delivery"],
                demand=candidate["demand"],
                release_time=candidate["release"],
                reveal_time=candidate["reveal"],
                extra=candidate["extra"],
            ))

    def _build_at_depot(self, env, depot: int) -> None:
        """``problem.loading: at_depot``: the static instance through manifests."""
        candidates = []
        for task in env.joint_instance.tasks:
            latest = task.latest_service
            if math.isfinite(latest):
                latest -= self._hours[self._index[depot], self._index[int(task.node_id)]]
            candidates.append({
                "kind": RequestKind.STATIC_DELIVERY,
                "pickup": Stop(depot, 0.0, max(0.0, latest), 0.0),
                "delivery": Stop(int(task.node_id), task.earliest_service, task.latest_service,
                                 task.base_service_time),
                "demand": task.demand, "reveal": 0.0, "release": 0.0, "extra": {},
            })
        self._register(candidates)

    # -- witness (documented extension of the EURO filter) --------------------

    def _witness(self, env, depot: int, candidate: dict) -> str:
        """``ok``, ``hos`` (infeasible only because of HOS) or ``infeasible``."""
        stops = [candidate["pickup"], candidate["delivery"]]
        start = candidate["release"]
        hard = self.deadline_type == "hard"
        plain = self._walk(env, depot, stops, start, hard, with_hos=False)
        if plain is None:
            return "infeasible"
        if env.hos is None:
            return "ok"
        return "ok" if self._walk(env, depot, stops, start, hard, with_hos=True) is not None else "hos"

    def _walk(self, env, depot, stops, start, hard, *, with_hos) -> float | None:
        capacity = float(env.config["truck"]["battery_capacity"])
        multiplier = _energy_multiplier(env)
        battery, time, node = capacity, float(start), int(depot)
        regime = env.hos.regime if with_hos else None
        clock = DriverClock.fresh(time) if with_hos else None
        anywhere = with_hos and env.hos.config.rest_locations == "anywhere"
        sequence = [*stops, Stop(int(depot), 0.0, self.max_time, 0.0)]
        for position, stop in enumerate(sequence):
            legs = self._segment(env, node, int(stop.node), battery, multiplier, capacity,
                                 loaded=self.is_drayage and position == 1)
            if legs is None:
                return None
            for kind, _station, hours, after in legs:
                if kind == "drive" and with_hos:
                    driven = _hos_drive(regime, clock, hours, anywhere)
                    if driven is None:
                        return None
                    clock, elapsed = driven
                    time += elapsed
                elif kind == "drive":
                    time += hours
                else:  # charge
                    if with_hos:
                        clock = apply(regime, clock, HosActivity.CHARGE, hours)
                    time += hours
                battery = after
            node = int(stop.node)
            begin = max(time, stop.earliest)
            if with_hos and begin > time:
                clock = apply(regime, clock, HosActivity.WINDOW_WAIT, begin - time)
            if hard and position < len(stops) and begin > stop.latest + 1e-9:
                return None
            time = begin + stop.service_hours
            if with_hos and stop.service_hours > 0:
                clock = apply(regime, clock, HosActivity.SERVICE, stop.service_hours)
            if time > self.max_time + 1e-9:
                return None
        return time

    def _segment(self, env, source, target, battery, multiplier, capacity, *, loaded):
        """At most one station between two stops (Kullman eq. (5) restriction)."""
        if source == target:
            return []
        index = self._index
        scale = 1.0
        if loaded and self.drayage_config.energy_model == "kancharla_ramadurai_2020":
            scale = kancharla_power_ratio(sum(self.drayage_config.container_mass_kg) / 2, self.base_speed_kmh)
        direct = self._energy[index[source], index[target]] * multiplier * scale
        hours = float(self._hours[index[source], index[target]])
        if math.isfinite(direct) and direct <= battery + 1e-9:
            return [("drive", None, hours, battery - direct)]
        best = None
        for charger in env.charging_nodes:
            first = self._energy[index[source], index[charger]] * multiplier * scale
            if not math.isfinite(first) or first > battery + 1e-9 or charger in (source, target):
                continue
            after, charge_hours, _ = env._certificate_charge_transition(charger, battery - first)
            second = self._energy[index[charger], index[target]] * multiplier * scale
            if not math.isfinite(second) or second > after + 1e-9:
                continue
            total = (float(self._hours[index[source], index[charger]]) + charge_hours
                     + float(self._hours[index[charger], index[target]]))
            if best is None or total < best[0]:
                best = (total, charger, first, after, charge_hours, second)
        if best is None:
            return None
        total, charger, first, after, charge_hours, second = best
        return [
            ("drive", None, float(self._hours[index[source], index[charger]]), battery - first),
            ("charge", charger, charge_hours, after),
            ("drive", None, float(self._hours[index[charger], index[target]]), after - second),
        ]

    # -- features ------------------------------------------------------------

    def feature_rows(self, env, depot_node: int) -> list[list[float]]:
        from EVRoutingEnv.state.features import _depot_columns, _optional_value

        rows = []
        kinds = list(RequestKind)
        statuses = list(RequestStatus)
        task_codes = {status: index for index, status in enumerate(TaskStatus)}
        for request in self.registry.revealed():
            view = RequestTaskView._STATUS[request.status]
            delivery = request.delivery
            has_window = math.isfinite(delivery.latest)
            pickup = request.pickup
            rows.append([
                request.request_id,
                int(delivery.node),
                request.demand,
                delivery.service_hours,
                delivery.earliest,
                delivery.latest if has_window else env.max_time,
                int(has_window),
                -1 if request.status is RequestStatus.LOST else task_codes[view],
                _optional_value(request.owner),
                _optional_value(request.claimed_at),
                -1.0,
                -1.0,
                _optional_value(request.service_started_at),
                _optional_value(request.served_at),
                *_depot_columns(env, int(delivery.node), depot_node),
                float(kinds.index(request.kind)),
                float(statuses.index(request.status)),
                float(pickup is not None),
                -1.0 if pickup is None else float(pickup.node),
                -1.0 if pickup is None else pickup.earliest,
                -1.0 if pickup is None else min(pickup.latest, env.max_time),
                -1.0 if pickup is None else pickup.service_hours,
                request.release_time,
                request.delay_hours,
            ])
        return rows

    def global_columns(self) -> list[float]:
        counts = self.registry.counts()
        revealed = len(self.registry.revealed())
        return [
            float(revealed),
            float(counts[RequestStatus.SERVED.value]),
            float(counts[RequestStatus.SERVED_LATE.value]),
            float(counts[RequestStatus.LOST.value]),
        ]

    def metrics(self, now: float = math.inf) -> dict:
        """Voccia's objective first: requests served on time.

        Filter drops are counted only for epochs revealed by ``now``.
        """
        revealed = self.registry.revealed()
        counts = self.registry.counts()
        delays = [r.delay_hours for r in revealed if r.status is RequestStatus.SERVED_LATE]
        on_time = counts[RequestStatus.SERVED.value]
        return {
            "requests_released": len(revealed),
            "requests_served_on_time": on_time,
            "served_on_time_fraction": on_time / len(revealed) if revealed else 1.0,
            "requests_served_late": counts[RequestStatus.SERVED_LATE.value],
            "requests_lost": counts[RequestStatus.LOST.value],
            "requests_unserved": sum(not r.is_terminal for r in revealed),
            "total_delay_hours": float(sum(delays)),
            "dropped_by_euro_filter": self._dropped("euro", now),
            "dropped_by_witness": self._dropped("witness", now),
            "dropped_by_hours_of_service": self._dropped("hos", now),
        }

    def _dropped(self, reason: str, now: float) -> int:
        return sum(1 for time, kind in self._drops if kind == reason and time <= now + 1e-12)


def _hos_drive(regime, clock, hours, anywhere):
    """Drive one leg under HOS: rests inside it (anywhere) or before it (stops)."""
    if anywhere:
        try:
            segments, clock = insert_rests(regime, clock, hours)
        except ValueError:
            return None
        return clock, sum(h for _, h in segments)
    waited = 0.0
    if not can_drive(regime, clock, hours):
        for kind, activity in (("break", HosActivity.REST_BREAK), ("daily", HosActivity.REST_DAILY)):
            try:
                length = rest_hours(regime, clock, kind)
            except ValueError:
                continue
            rested = apply(regime, clock, activity, length)
            if can_drive(regime, rested, hours):
                clock, waited = rested, length
                break
        else:
            return None
    return apply(regime, clock, HosActivity.DRIVE, hours), waited + hours


def _energy_multiplier(env) -> float:
    traffic = env.traffic_config
    if traffic["enable_traffic"] and traffic["enable_energy_uncertainty"]:
        return max(1.0, float(traffic["max_energy_multiplier"]))
    return 1.0


def hindsight_problem(env) -> dict:
    """Full-information reference (EURO ``get_hindsight_problem``; Li et al. 2024).

    Every request with its release time and windows, including those not yet
    revealed, plus the scenario identity that fixes all realized travel,
    service and outside-traffic draws (common random numbers). A bound, not a
    policy: never pass it to a policy.
    """
    engine = env.request_engine
    if engine is None:
        raise ValueError("hindsight_problem requires a request-engine mode")
    requests = []
    for request in engine.registry.requests():
        requests.append({
            "request_id": request.request_id,
            "kind": request.kind.value,
            "reveal_time": request.reveal_time,
            "release_time": request.release_time,
            "pickup": None if request.pickup is None else vars(request.pickup).copy(),
            "delivery": vars(request.delivery).copy(),
            "demand": request.demand,
            "extra": dict(request.extra),
        })
    return {
        "mode": engine.mode,
        "depot": int(engine.depot),
        "scenario": env.scenario_random_streams.metadata(),
        "max_time": float(env.max_time),
        "num_trucks": int(env.num_trucks),
        "deadline_type": engine.deadline_type,
        "requests": requests,
    }
