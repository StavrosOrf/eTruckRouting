"""Rolling-horizon fleet re-optimization for the dynamic request modes.

A non-learned, non-clairvoyant online policy for same-day freight (Voccia,
Campbell & Thomas 2019 objective; trips with release dates as in formulation
(1) of Li, Archetti & Ljubic 2024) and electric drayage (Xiao et al. 2025 task
structure, US hours of service, public stations after Kullman et al. 2021).

At every decision epoch the policy

1. reads a non-anticipative snapshot of the fleet: the active truck exactly;
   every other truck at the end of its committed activity, with *nominal*
   travel and service times for work in progress (``nominal_eta``), never the
   realized draws, hidden requests or hidden queue state;
2. re-plans every revealed, unserved request over the whole fleet with an
   adaptive large neighbourhood search (Ropke & Pisinger 2006): random, worst,
   related and route removal; greedy and regret-2 insertion with noise;
   simulated-annealing acceptance; warm-started from the previous plan;
3. lets the active truck execute the first step of its route through the
   simulator's own action set, after a safety check that the step keeps a
   worst-case energy reserve to a charger or the depot.

Routes are evaluated forward with worst-case energy (the simulator's maximum
energy multiplier, so a planned leg is always feasible under any realization),
the simulator's CC-CV charging curve, at most one charge stop per segment,
payload, pickup deadlines, and the regime's hours of service (rests inserted
with the simulator's own ``insert_rests``). Travel times are nominal with the
simulator's Gaussian spread (rush hours included); the objective is the
expected number of on-time services (a normal approximation of the arrival
time) minus a small weight on route duration. End-of-shift reserves are part
of the policy, not of the simulator: a truck never leaves a node unless it can
still reach a charger or the depot afterwards with worst-case energy.

``clairvoyant=True`` turns the same planner into a hindsight reference: it then
also sees the requests the simulator has generated but not yet revealed, with
their true release times. That variant is a bound estimate, never a policy.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, replace

from EVRoutingEnv.baselines.heuristics.canonical_baselines import nominal_charge_hours
from EVRoutingEnv.models.core.hours_of_service import (
    HosActivity,
    advance_to,
    apply as hos_apply,
    driving_allowance,
    insert_rests,
    required_rest,
    window_left,
)
from EVRoutingEnv.models.core.requests import RequestStatus
from EVRoutingEnv.state.feasibility import ActionKind, joint_action_feasibility

INF = math.inf
EPS = 1e-9
REST = -1  # drayage route token: take the daily rest here
_SQRT2 = math.sqrt(2.0)


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / _SQRT2))


def on_time_probability(arrival: float, deadline: float, variance: float) -> float:
    """P(arrival + delay <= deadline) for a centred normal delay."""
    if not math.isfinite(deadline):
        return 1.0
    slack = deadline - arrival
    if variance <= 1e-12:
        return 1.0 if slack >= -1e-9 else 0.0
    return _phi(slack / math.sqrt(variance))


# ---------------------------------------------------------------------------
# Hours-of-service arithmetic for the planner
# ---------------------------------------------------------------------------


class HosOps:
    """The regime's clock rules through ``models.core.hours_of_service``.

    Clocks are ``DriverClock`` values. ``FastUsHosOps`` reproduces the same
    rules for the US property-carrying regime on plain tuples, which the
    planner evaluates hundreds of thousands of times per episode; a unit test
    checks the two against each other on random duty sequences.
    """

    def __init__(self, regime):
        self.regime = regime
        self.daily_rest = regime.daily_rest_hours

    def convert(self, clock):
        return clock

    def advance(self, clock, t):
        if clock.time >= t:
            return clock
        return advance_to(self.regime, clock, t, HosActivity.IDLE)

    def drive(self, clock, t, hours):
        clock = self.advance(clock, t)
        segments, clock = insert_rests(self.regime, clock, hours)
        return sum(h for _, h in segments), clock

    def work(self, clock, t, hours, activity=HosActivity.SERVICE):
        clock = self.advance(clock, t)
        return hos_apply(self.regime, clock, activity, hours) if hours > 0 else clock

    def rest_daily(self, clock, t):
        clock = self.advance(clock, t)
        return hos_apply(self.regime, clock, HosActivity.REST_DAILY, self.daily_rest)

    def started(self, clock) -> bool:
        return clock.shift_start is not None or clock.driving_in_shift > 0.0

    def window_left(self, clock) -> float:
        return window_left(self.regime, clock)


class FastUsHosOps(HosOps):
    """49 CFR 395.3 on tuples ``(time, since_break, in_shift, shift_start, streak, off_duty)``."""

    BEFORE_BREAK, BREAK, SHIFT, WINDOW, DAILY = 8.0, 0.5, 11.0, 14.0, 10.0
    TOL = 1e-9

    def convert(self, clock):
        return (float(clock.time), float(clock.driving_since_break), float(clock.driving_in_shift),
                None if clock.shift_start is None else float(clock.shift_start),
                float(clock.break_streak), float(clock.off_duty_streak))

    def _apply(self, c, hours, drive=False, off=False):
        if hours <= 0.0:
            return c
        time, dsb, dis, start, streak, offd = c
        if not off and start is None:
            start = time
        end = time + hours
        if drive:
            return (end, dsb + hours, dis + hours, start, 0.0, 0.0)
        streak += hours
        offd = offd + hours if off else 0.0
        if streak >= self.BREAK - self.TOL:
            dsb = 0.0
        if offd >= self.DAILY - self.TOL and (start is not None or dis > 0.0):
            dsb, dis, start = 0.0, 0.0, None
        return (end, dsb, dis, start, streak, offd)

    def _allowance(self, c):
        time, dsb, dis, start, _, _ = c
        window = self.WINDOW if start is None else start + self.WINDOW - time
        return max(0.0, min(self.BEFORE_BREAK - dsb, self.SHIFT - dis, window))

    def advance(self, c, t):
        if c[0] >= t:
            return c
        return self._apply(c, t - c[0])

    def drive(self, c, t, hours):
        c = self.advance(c, t)
        remaining, elapsed = float(hours), 0.0
        guard = 0
        while remaining > self.TOL:
            guard += 1
            if guard > 100:
                raise RuntimeError("rest insertion did not converge")
            allowance = self._allowance(c)
            if allowance >= remaining - self.TOL:
                c = self._apply(c, remaining, drive=True)
                elapsed += remaining
                break
            if allowance > self.TOL:
                c = self._apply(c, allowance, drive=True)
                remaining -= allowance
                elapsed += allowance
            time, dsb, dis, start, _, _ = c
            window = self.WINDOW if start is None else start + self.WINDOW - time
            spent = dis >= self.SHIFT - self.TOL or window <= self.TOL
            rest = self.DAILY
            if not spent and self._allowance(self._apply(c, self.BREAK, off=True)) > self.TOL:
                rest = self.BREAK
            c = self._apply(c, rest, off=True)
            elapsed += rest
        return elapsed, c

    def work(self, c, t, hours, activity=HosActivity.SERVICE):
        c = self.advance(c, t)
        off = activity in (HosActivity.REST_BREAK, HosActivity.REST_DAILY)
        return self._apply(c, hours, off=off)

    def rest_daily(self, c, t):
        return self._apply(self.advance(c, t), self.DAILY, off=True)

    def started(self, c) -> bool:
        return c[3] is not None or c[2] > 0.0

    def window_left(self, c) -> float:
        return self.WINDOW if c[3] is None else c[3] + self.WINDOW - c[0]


def hos_ops(regime) -> HosOps | None:
    if regime is None:
        return None
    if regime.name == "us_fmcsa_property":
        return FastUsHosOps(regime)
    return HosOps(regime)


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlannerParameters:
    """Search budget and objective weights (all fixed a priori or tuned on validation)."""

    iterations: int = 60  # ALNS iterations per re-plan
    min_iterations: int = 8
    destroy_fraction: float = 0.3
    cost_weight: float = 0.02  # objective per hour of route duration
    late_weight: float = 0.05  # objective per expected late service
    min_probability: float = 0.05  # requests planned below this count as unserved
    max_trips: int = 4  # same-day trips planned per truck
    wait_slack_hours: float = 0.75  # drayage: wait in place when this early
    idle_rest_window_hours: float = 4.0  # drayage: rest when idle and window short
    temperature: float = 0.05
    clairvoyant: bool = False
    seed: int = 0


# ---------------------------------------------------------------------------
# Network data
# ---------------------------------------------------------------------------


class _Network:
    """Nominal hours and energy as nested lists, plus per-depot refuges."""

    _cache: dict = {}

    def __init__(self, env):
        index, energy, hours = env.transport_graph.cost_matrices()
        self.index = {int(node): int(i) for node, i in index.items()}
        self.hours = hours.tolist()
        self.energy = energy.tolist()
        self.chargers = [int(node) for node in env.charging_nodes]
        self.charger_index = [self.index[node] for node in self.chargers]
        self._detours: dict = {}

    @classmethod
    def of(cls, env) -> _Network:
        key = (id(env.transport_graph), tuple(int(n) for n in env.charging_nodes))
        network = cls._cache.get(key)
        if network is None:
            network = cls._cache[key] = cls(env)
        return network

    def h(self, a: int, b: int) -> float:
        if a == b:
            return 0.0
        return self.hours[self.index[a]][self.index[b]]

    def e(self, a: int, b: int) -> float:
        if a == b:
            return 0.0
        return self.energy[self.index[a]][self.index[b]]

    def detour_chargers(self, a: int, b: int, refuges: tuple[int, ...]) -> list[int]:
        """Charge stops between a and b, by nominal detour time."""
        key = (a, b, refuges)
        cached = self._detours.get(key)
        if cached is None:
            options = [c for c in refuges if c != a and c != b]
            options.sort(key=lambda c: self.h(a, c) + self.h(c, b))
            cached = self._detours[key] = options
        return cached


# ---------------------------------------------------------------------------
# Snapshot of the planning problem
# ---------------------------------------------------------------------------


@dataclass
class Job:
    rid: int
    pickup: int
    delivery: int
    release: float  # earliest loading start (same-day) or pickup (drayage)
    pickup_latest: float
    earliest: float  # delivery window opening
    deadline: float  # delivery window closing
    service: float  # delivery service hours (nominal)
    load: float  # pickup service hours
    demand: float
    weight: float = 1.0
    revealed: bool = True
    reveal: float = 0.0


@dataclass
class TruckStart:
    tid: int
    t: float
    node: int
    battery: float  # worst-case kWh at time t
    var: float = 0.0  # variance of t
    clock: object | None = None
    onboard: tuple[int, ...] = ()  # same-day: loaded requests; drayage: the container
    away: bool = False  # same-day: a trip is in progress away from the depot
    first: int | None = None  # drayage: request the truck is committed to
    container_loaded: bool = False  # drayage: only the loaded leg remains
    available: bool = True
    activity: str = "ready"


@dataclass
class Snapshot:
    now: float
    jobs: dict[int, Job]
    trucks: list[TruckStart]
    future_information: bool
    active: int


# ---------------------------------------------------------------------------
# Route models
# ---------------------------------------------------------------------------


@dataclass
class RouteResult:
    value: float
    probs: dict
    end: float
    steps: list = field(default_factory=list)


class _BaseModel:
    """Shared leg arithmetic: worst-case energy, charging, spread of travel time."""

    def __init__(self, env, parameters: PlannerParameters):
        self.env = env
        self.parameters = parameters
        self.net = _Network.of(env)
        self.depot = int(env.joint_instance.depot_node)
        self.capacity = float(env.config["truck"]["battery_capacity"])
        self.payload = float(env.trucks[0].payload_capacity)
        traffic = env.traffic_config
        self.factor = (
            max(1.0, float(traffic["max_energy_multiplier"]))
            if traffic["enable_traffic"] and traffic["enable_energy_uncertainty"]
            else 1.0
        )
        self.traffic = bool(traffic["enable_traffic"])
        self.std_factor = float(traffic.get("std_dev_factor", 0.15))
        self.max_std = float(traffic.get("max_std_dev_hours", 1.0) or INF)
        self.rush = float(traffic.get("rush_hour_multiplier", 1.0))
        self.levels = sorted(float(v) for v in env.charge_action_values)
        self.depot_charging = bool(env.extensions.depot_charging.enabled)
        self.refuges = tuple(self.net.chargers) + ((self.depot,) if self.depot_charging else ())
        self._reserve: dict[int, float] = {}
        stations = getattr(env, "public_stations", None)
        self.public_wait = {}
        if stations is not None:
            for node in self.net.chargers:
                try:
                    self.public_wait[node] = float(stations.long_run_wait(node))
                except Exception:  # noqa: BLE001 - a station without parameters
                    self.public_wait[node] = 0.0
        self.max_time = float(env.max_time)
        self._charge_cache: dict = {}
        self.regime = None
        self.ops = None

    def _activity(self, clock, t, hours, activity):
        if self.ops is None or clock is None or hours <= 0.0:
            return clock
        return self.ops.work(clock, t, hours, activity)

    # -- legs ----------------------------------------------------------------

    def need(self, a: int, b: int) -> float:
        return self.factor * self.net.e(a, b)

    def reserve(self, node: int) -> float:
        """Worst-case energy to reach the nearest refuge from ``node``."""
        value = self._reserve.get(node)
        if value is None:
            if node in self.refuges:
                value = 0.0
            else:
                value = min(self.need(node, c) for c in self.refuges)
            self._reserve[node] = value
        return value

    def sigma2(self, hours: float, start: float) -> float:
        if not self.traffic or hours <= 0.0:
            return 0.0
        end = start + hours
        overlap = 0.0
        day = math.floor(start / 24.0) * 24.0
        while day < end:
            for lo, hi in ((7.0, 9.0), (17.0, 19.0)):
                a, b = max(start, day + lo), min(end, day + hi)
                if b > a:
                    overlap += b - a
            day += 24.0
        fraction = overlap / hours
        std = hours * self.std_factor * (1.0 + fraction * (self.rush - 1.0))
        std = min(std, self.max_std)
        return std * std

    def charge_hours(self, node: int, battery: float, target: float) -> float:
        soc = max(0.0, min(1.0, battery / self.capacity))
        if target <= soc + 1e-9:
            return 0.0
        key = (node in self.net.chargers, round(soc, 2), target)
        value = self._charge_cache.get(key)
        if value is None:
            value = self._charge_cache[key] = nominal_charge_hours(
                self.env, node, round(soc, 2), target, self.capacity
            )
        return value

    def target_for(self, energy: float, battery: float) -> float | None:
        """Smallest charge level covering ``energy``; None if none is above now."""
        soc = battery / self.capacity
        for level in self.levels:
            if level > soc + 1e-9 and level * self.capacity >= energy - 1e-6:
                return level
        top = self.levels[-1]
        return top if top > soc + 1e-9 else None

    def station_wait(self, node: int) -> float:
        return self.public_wait.get(node, 0.0)


class SameDayModel(_BaseModel):
    """Trips from the depot: load at release, deliver within hard windows, return.

    A route is a tuple of trips; each trip a tuple of request ids in delivery
    order. The first trip of a truck that is away is its trip in progress.
    """

    kind = "same_day"

    def __init__(self, env, parameters):
        super().__init__(env, parameters)
        self.load_hours = float(env.extensions.requests.loading_service_hours)

    # -- evaluation ------------------------------------------------------------

    def evaluate(self, ts: TruckStart, route, jobs, trace: bool = False) -> RouteResult:
        net, depot = self.net, self.depot
        t, node, battery, var = ts.t, ts.node, ts.battery, ts.var
        value, probs, steps = 0.0, {}, []
        start = ts.t
        onboard = set(ts.onboard)
        for j, trip in enumerate(route):
            current = j == 0 and ts.away
            if not current:
                if node != depot:
                    leg = self._leg(node, depot, t, battery, var, 0.0, steps if trace else None)
                    if leg is None:
                        return RouteResult(-1e9, {}, INF, steps)
                    t, battery, var = leg
                    node = depot
                    if trace:
                        steps.append(("arrive_depot", t))
                loads = [r for r in trip if not (j == 0 and r in onboard)]
                # Charge at the depot for the whole trip (worst case), if needed.
                sequence = [depot, *(jobs[r].delivery for r in trip), depot]
                trip_need = sum(self.need(a, b) for a, b in zip(sequence, sequence[1:]))
                charged = 0.0
                if battery < trip_need - 1e-6 and self.depot_charging:
                    target = self.target_for(trip_need, battery)
                    if target is not None:
                        charged = self.charge_hours(depot, battery, target)
                        if trace:
                            steps.append(("charge", depot, target, t, charged))
                        battery = max(battery, target * self.capacity)
                ready = t + charged
                skipped = set()
                for r in sorted(loads, key=lambda r: jobs[r].pickup_latest):
                    job = jobs[r]
                    begin = max(ready, job.release)
                    if begin > job.pickup_latest + 1e-9:
                        skipped.add(r)
                        continue
                    if trace:
                        steps.append(("load", r, begin))
                    ready = begin + self.load_hours
                if trace:
                    steps.append(("depart", ready))
                t, var = ready, 0.0
                trip = [r for r in trip if r not in skipped]
                for r in skipped:
                    probs[r] = 0.0
            # Deliveries in order; suffix energy decides when to charge.
            stops = [jobs[r].delivery for r in trip]
            suffix = [0.0] * (len(stops) + 1)
            suffix[-1] = 0.0
            nodes = [*stops, depot]
            for i in range(len(stops) - 1, -1, -1):
                suffix[i] = suffix[i + 1] + self.need(nodes[i], nodes[i + 1])
            for i, r in enumerate(trip):
                job = jobs[r]
                leg = self._leg(node, job.delivery, t, battery, var, suffix[i], steps if trace else None,
                                nodes[i + 1])
                if leg is None:
                    return RouteResult(-1e9, {}, INF, steps)
                t, battery, var = leg
                node = job.delivery
                p = on_time_probability(t, job.deadline, var)
                if p < self.parameters.min_probability:
                    p = 0.0
                probs[r] = p
                value += job.weight * (p - self.parameters.late_weight * (1.0 - p))
                begin = max(t, job.earliest)
                slack = begin - t
                if slack > 0.0 and var > 0.0:
                    sd = max(0.0, math.sqrt(var) - slack)
                    var = sd * sd
                if trace:
                    steps.append(("deliver", r, t, p))
                t = begin + job.service
        end = t
        if node != depot and route:
            back = self._leg(node, depot, t, battery, var, 0.0, None)
            end = back[0] if back is not None else t + net.h(node, depot)
        value -= self.parameters.cost_weight * max(0.0, end - start) * (1.0 if route else 0.0)
        return RouteResult(value, probs, end, steps)

    def _leg(self, a, b, t, battery, var, after, steps, onward=None):
        """Drive a -> b keeping ``after`` worst-case kWh for the rest of the trip.

        Charging is postponed as long as a station on the way to the stop after
        ``b`` (``onward``) stays reachable from ``b`` in the worst case.
        """
        if a == b:
            return t, battery, var
        direct = self.need(a, b)
        postpone = False
        if battery < direct + after - 1e-6 and onward is not None and b != self.depot:
            left = battery - direct
            if left >= 0.0:
                if b in self.refuges:
                    postpone = True
                else:
                    for c in self.net.detour_chargers(b, onward, self.refuges)[:6]:
                        if self.need(b, c) <= left + 1e-6:
                            postpone = True
                            break
        if battery >= direct + after - 1e-6 or postpone:
            hours = self.net.h(a, b)
            if steps is not None:
                steps.append(("drive", b, t))
            return t + hours, battery - direct, var + self.sigma2(hours, t)
        # Charge at a (it is a refuge) or detour through the best station.
        best = None
        if a in self.refuges and (a != self.depot or self.depot_charging):
            target = self.target_for(direct + after, battery)
            if target is not None:
                hours = self.charge_hours(a, battery, target) + self.station_wait(a)
                best = (hours, a, target)
        for c in self.net.detour_chargers(a, b, self.refuges)[:6]:
            first = self.need(a, c)
            if first > battery + 1e-6:
                continue
            left = battery - first
            target = self.target_for(self.need(c, b) + after, left)
            if target is None:
                continue
            if target * self.capacity < self.need(c, b) + self.reserve(b) - 1e-6:
                continue
            hours = (self.net.h(a, c) + self.station_wait(c) + self.charge_hours(c, left, target)
                     + self.net.h(c, b))
            if best is None or hours < best[0]:
                best = (hours, c, target)
            break_even = self.net.h(a, c) + self.net.h(c, b)
            if best is not None and break_even > best[0]:
                break
        if best is None:
            return None
        _, c, target = best
        if c == a:
            charge = self.charge_hours(a, battery, target)
            if steps is not None:
                steps.append(("charge", a, target, t, charge))
            t += charge + self.station_wait(a)
            battery = target * self.capacity
        else:
            hours = self.net.h(a, c)
            if steps is not None:
                steps.append(("drive", c, t))
            var += self.sigma2(hours, t)
            t += hours
            battery -= self.need(a, c)
            charge = self.charge_hours(c, battery, target)
            if steps is not None:
                steps.append(("charge", c, target, t, charge))
            t += charge + self.station_wait(c)
            battery = max(battery, target * self.capacity)
        hours = self.net.h(c, b)
        if steps is not None:
            steps.append(("drive", b, t))
        return t + hours, battery - self.need(c, b), var + self.sigma2(hours, t)

    # -- route editing -----------------------------------------------------------

    def empty_route(self, ts):
        return ((tuple(ts.onboard),) if ts.onboard else ())

    def locked(self, ts) -> set:
        return set(ts.onboard)

    def remove(self, route, r):
        trips = []
        for j, trip in enumerate(route):
            if r in trip:
                trip = tuple(x for x in trip if x != r)
                if not trip and j > 0:
                    continue
                if not trip and j == 0:
                    continue
            trips.append(trip)
        return tuple(trips)

    def insertions(self, ts, route, r, jobs):
        demand = jobs[r].demand
        closed0 = ts.away
        has_open0 = bool(ts.onboard) and not ts.away
        out = []
        for j, trip in enumerate(route):
            if j == 0 and closed0:
                continue
            load = sum(jobs[x].demand for x in trip)
            if load + demand > self.payload + 1e-9:
                continue
            for i in range(len(trip) + 1):
                out.append(route[:j] + (trip[:i] + (r,) + trip[i:],) + route[j + 1:])
        if len(route) < self.parameters.max_trips + (1 if (closed0 or has_open0) else 0):
            first = 1 if (closed0 or has_open0) else 0
            for j in range(first, len(route) + 1):
                out.append(route[:j] + ((r,),) + route[j:])
        return out

    def members(self, route):
        return [r for trip in route for r in trip]

    def reorders(self, ts, route):
        """Relocate one delivery within its trip (on-board requests included)."""
        out = []
        for j, trip in enumerate(route):
            fixed = 1 if (j == 0 and ts.away and ts.first is not None and trip and trip[0] == ts.first) else 0
            n = len(trip)
            for i in range(fixed, n):
                rest = trip[:i] + trip[i + 1:]
                for k in range(fixed, n):
                    if k == i:
                        continue
                    moved = rest[:k] + (trip[i],) + rest[k:]
                    out.append(route[:j] + (moved,) + route[j + 1:])
        return out


class DrayageModel(_BaseModel):
    """One container at a time: empty leg, pickup window, loaded leg, deadline.

    A route is a tuple of request ids. Hours of service follow the simulator's
    regime; rests are placed inside legs as late as possible (``insert_rests``,
    Goel 2009) or, when a long wait would absorb it, as a daily rest first.
    """

    kind = "drayage"

    def __init__(self, env, parameters):
        super().__init__(env, parameters)
        self.hos = env.hos
        self.regime = env.hos.regime if env.hos is not None else None
        self.ops = hos_ops(self.regime)

    def _drive(self, hours, t, clock):
        if self.ops is None or hours <= 0.0:
            return hours, clock
        return self.ops.drive(clock, t, hours)

    def _go(self, a, b, t, battery, var, clock, steps):
        hours = self.net.h(a, b)
        if steps is not None:
            steps.append(("drive", b, t))
        elapsed, clock = self._drive(hours, t, clock)
        return t + elapsed, battery - self.need(a, b), var + self.sigma2(hours, t), clock

    def _charge(self, node, t, battery, target, clock, steps):
        wait = self.station_wait(node)
        hours = self.charge_hours(node, battery, target)
        if steps is not None:
            steps.append(("charge", node, target, t, hours))
        clock = self._activity(clock, t, wait, HosActivity.QUEUE)
        clock = self._activity(clock, t + wait, hours, HosActivity.CHARGE)
        return t + wait + hours, max(battery, target * self.capacity), clock

    def rest_blocked(self, t: float, jobs, pending) -> bool:
        """A daily rest starting at ``t`` is not taken while a known job is due before it would end.

        ``pending`` are the jobs still to serve (route members, or every job of
        the snapshot when idle); a request released before ``t + daily rest``
        would wait out the rest. Known means revealed, or hidden too when the
        snapshot is clairvoyant.
        """
        if self.ops is None or not self.ops.daily_rest:
            return False
        end = t + self.ops.daily_rest
        return any(jobs[r].release < end - 1e-9 for r in pending if r != REST)

    def evaluate(self, ts: TruckStart, route, jobs, trace: bool = False) -> RouteResult:
        t, node, battery, var, clock = ts.t, ts.node, ts.battery, ts.var, ts.clock
        value, probs, steps = 0.0, {}, []
        start = t
        # Worst-case energy still to drive, for charge targets.
        legs = []
        position = node
        rested = 0.0
        for i, r in enumerate(route):
            if r == REST:
                continue
            job = jobs[r]
            if not (i == 0 and ts.container_loaded):
                legs.append(self.need(position, job.pickup))
            legs.append(self.need(job.pickup if not (i == 0 and ts.container_loaded) else position,
                                  job.delivery))
            position = job.delivery
        remaining = sum(legs)
        trace_steps = steps if trace else None
        for i, r in enumerate(route):
            if r == REST:
                if self.ops is not None and self.ops.daily_rest:
                    clock_now = self.ops.advance(clock, t)
                    if self.ops.started(clock_now) and not self.rest_blocked(t, jobs, route[i + 1:]):
                        if trace:
                            steps.append(("rest", "daily", t))
                        clock = self.ops.rest_daily(clock_now, t)
                        t += self.ops.daily_rest
                        rested += self.ops.daily_rest
                continue
            job = jobs[r]
            if i == 0 and ts.container_loaded:
                # Only stations may lie between origin and destination.
                need = self.need(node, job.delivery) + self.reserve(job.delivery)
                if battery < need - 1e-6:
                    result = self._via_station(node, job.delivery, t, battery, var, clock, need,
                                               trace_steps, allow_depot=False)
                    if result is None:
                        return RouteResult(-1e9, {}, INF, steps)
                    t, battery, var, clock = result
                else:
                    t, battery, var, clock = self._go(node, job.delivery, t, battery, var, clock, trace_steps)
                remaining -= self.need(node, job.delivery)
                node = job.delivery
            else:
                if job.reveal > t:
                    clock = self._activity(clock, t, job.reveal - t, HosActivity.IDLE)
                    t = job.reveal
                empty = self.need(node, job.pickup)
                loaded = self.need(job.pickup, job.delivery)
                need = empty + loaded + self.reserve(job.delivery)
                if battery < need - 1e-6:
                    cover = min(remaining + self.reserve(job.delivery), self.capacity)
                    result = self._recharge(node, job.pickup, t, battery, var, clock,
                                            max(need, cover), need, trace_steps)
                    if result is None:
                        return RouteResult(-1e9, {}, INF, steps)
                    t, battery, var, clock, node = result
                    empty = self.need(node, job.pickup)
                # Rest first when a long wait at the origin would absorb it.
                if self.ops is not None and node != job.pickup and self.ops.daily_rest:
                    clock_now = self.ops.advance(clock, t)
                    hours = self.net.h(node, job.pickup)
                    wait = job.release - (t + hours)
                    rest = self.ops.daily_rest
                    if (self.ops.started(clock_now) and wait >= rest - 0.75
                            and not self.rest_blocked(t, jobs, route[i:])):
                        if trace:
                            steps.append(("rest", "daily", t))
                        clock = self.ops.rest_daily(clock_now, t)
                        t += rest
                if node != job.pickup:
                    if trace and job.release - (t + self.net.h(node, job.pickup)) > 0:
                        steps.append(("early", job.release - (t + self.net.h(node, job.pickup))))
                    t, battery, var, clock = self._go(node, job.pickup, t, battery, var, clock, trace_steps)
                    node = job.pickup
                remaining -= empty
                begin = max(t, job.release)
                slack = begin - t
                if slack > 0.0:
                    clock = self._activity(clock, t, slack, HosActivity.WINDOW_WAIT)
                    if var > 0.0:
                        sd = max(0.0, math.sqrt(var) - slack)
                        var = sd * sd
                t = begin + job.load
                clock = self._activity(clock, begin, job.load, HosActivity.SERVICE)
                if battery < loaded + self.reserve(job.delivery) - 1e-6:
                    result = self._via_station(node, job.delivery, t, battery, var, clock,
                                               loaded + self.reserve(job.delivery), trace_steps,
                                               allow_depot=False)
                    if result is None:
                        return RouteResult(-1e9, {}, INF, steps)
                    t, battery, var, clock = result
                else:
                    t, battery, var, clock = self._go(node, job.delivery, t, battery, var, clock, trace_steps)
                remaining -= loaded
                node = job.delivery
            p = on_time_probability(t, job.deadline, var)
            if p < self.parameters.min_probability:
                p = 0.0
            probs[r] = p
            value += job.weight * (p - self.parameters.late_weight * (1.0 - p))
            if trace:
                steps.append(("deliver", r, t, p))
            clock = self._activity(clock, t, job.service, HosActivity.SERVICE)
            t += job.service
        busy = max(0.0, t - start - rested)
        value -= self.parameters.cost_weight * busy * (1.0 if route else 0.0)
        return RouteResult(value, probs, t, steps)

    def _recharge(self, a, b, t, battery, var, clock, want, need, steps):
        """Charge before the empty leg a -> b: here, at the depot, or at a station."""
        options = []
        if a == self.depot and self.depot_charging:
            target = self.target_for(want, battery)
            if target is not None and target * self.capacity >= need - 1e-6:
                options.append((self.charge_hours(a, battery, target), a, target))
        elif a in self.net.chargers:
            target = self.target_for(want, battery)
            if target is not None and target * self.capacity >= need - 1e-6:
                options.append((self.charge_hours(a, battery, target) + self.station_wait(a), a, target))
        if a != self.depot:
            candidates = list(self.net.detour_chargers(a, b, self.refuges)[:5])
            if self.depot_charging and self.depot not in candidates and self.depot != b:
                candidates.append(self.depot)
            for c in candidates:
                first = self.need(a, c)
                if first > battery + 1e-6:
                    continue
                left = battery - first
                target = self.target_for(want - 0.0, left)
                second_need = need - self.need(a, b) + self.need(c, b)
                if target is None or target * self.capacity < second_need - 1e-6:
                    continue
                hours = (self.net.h(a, c) + self.station_wait(c) + self.charge_hours(c, left, target)
                         + self.net.h(c, b) - self.net.h(a, b))
                options.append((hours, c, target))
        if not options:
            return None
        _, c, target = min(options, key=lambda item: item[0])
        node = a
        if c != a:
            t, battery, var, clock = self._go(a, c, t, battery, var, clock, steps)
            node = c
        t, battery, clock = self._charge(node, t, battery, target, clock, steps)
        return t, battery, var, clock, node

    def _via_station(self, a, b, t, battery, var, clock, need, steps, allow_depot):
        """A loaded leg that needs a charge: only stations between origin and destination."""
        options = []
        if a in self.net.chargers or (allow_depot and a == self.depot):
            target = self.target_for(need, battery)
            if target is not None and target * self.capacity >= need - 1e-6:
                options.append((self.charge_hours(a, battery, target) + self.station_wait(a), a, target))
        for c in self.net.detour_chargers(a, b, tuple(self.net.chargers))[:5]:
            first = self.need(a, c)
            if first > battery + 1e-6:
                continue
            left = battery - first
            second = self.need(c, b) + self.reserve(b)
            target = self.target_for(second, left)
            if target is None or target * self.capacity < second - 1e-6:
                continue
            hours = (self.net.h(a, c) + self.station_wait(c) + self.charge_hours(c, left, target)
                     + self.net.h(c, b))
            options.append((hours, c, target))
        if not options:
            return None
        _, c, target = min(options, key=lambda item: item[0])
        node = a
        if c != a:
            t, battery, var, clock = self._go(a, c, t, battery, var, clock, steps)
            node = c
        t, battery, clock = self._charge(node, t, battery, target, clock, steps)
        return self._go(node, b, t, battery, var, clock, steps)

    # -- route editing -----------------------------------------------------------

    def empty_route(self, ts):
        return (ts.first,) if ts.first is not None else ()

    def locked(self, ts) -> set:
        return {ts.first} if ts.first is not None else set()

    def remove(self, route, r):
        return tuple(x for x in route if x != r)

    def insertions(self, ts, route, r, jobs):
        begin = 1 if ts.first is not None else 0
        return [route[:i] + (r,) + route[i:] for i in range(begin, len(route) + 1)]

    def members(self, route):
        return [r for r in route if r != REST]

    def reorders(self, ts, route):
        fixed = 1 if ts.first is not None else 0
        out = []
        n = len(route)
        for i in range(fixed, n):
            rest = route[:i] + route[i + 1:]
            for k in range(fixed, n):
                if k != i:
                    out.append(rest[:k] + (route[i],) + rest[k:])
        if self.ops is not None and self.ops.daily_rest:
            if REST in route:
                out.append(tuple(x for x in route if x != REST))
            else:
                for k in range(fixed, n + 1):
                    out.append(route[:k] + (REST,) + route[k:])
        return out


# ---------------------------------------------------------------------------
# Non-anticipative snapshot
# ---------------------------------------------------------------------------

_DONE = frozenset({RequestStatus.SERVED, RequestStatus.SERVED_LATE, RequestStatus.LOST})
_CONTAINER = frozenset({
    RequestStatus.LOADING, RequestStatus.ON_BOARD, RequestStatus.CLAIMED_DELIVERY,
})


def _job(request, depot: int, drayage: bool, weight: float = 1.0) -> Job:
    pickup, delivery = request.pickup, request.delivery
    return Job(
        rid=int(request.request_id),
        pickup=int(pickup.node) if pickup is not None else depot,
        delivery=int(delivery.node),
        release=float(pickup.earliest) if pickup is not None else float(request.release_time),
        pickup_latest=float(pickup.latest) if pickup is not None else INF,
        earliest=float(delivery.earliest),
        deadline=float(delivery.latest),
        service=float(delivery.service_hours),
        load=float(pickup.service_hours) if pickup is not None else 0.0,
        demand=float(request.demand),
        weight=weight,
        revealed=request.status is not RequestStatus.HIDDEN,
        reveal=float(request.reveal_time),
    )


def _pending_event(env, truck_id: int):
    events = [event for event in env.event_queue if event.truck_id == truck_id]
    return min(events, key=lambda event: event.time) if events else None


def build_snapshot(env, model: _BaseModel, clairvoyant: bool = False) -> Snapshot:
    """What the dispatcher knows now (plus, if clairvoyant, the hidden stream)."""
    now = float(env.global_clock)
    engine = env.request_engine
    registry = engine.registry
    drayage = engine.is_drayage
    depot = model.depot
    jobs: dict[int, Job] = {}
    for request in registry.requests():
        if request.status in _DONE or request.status is RequestStatus.IN_SERVICE:
            continue
        if request.status is RequestStatus.HIDDEN and not clairvoyant:
            continue
        job = _job(request, depot, drayage)
        jobs[job.rid] = job
    trucks = [_truck_start(env, model, truck, now, jobs) for truck in env.trucks]
    return Snapshot(
        now=now,
        jobs=jobs,
        trucks=trucks,
        future_information=engine.future_information_possible(now),
        active=int(env.active_truck_id) if env.active_truck_id is not None else -1,
    )


def _truck_start(env, model: _BaseModel, truck, now: float, jobs) -> TruckStart:
    tid = int(truck.truck_id)
    state = env.truck_states[tid]
    engine = env.request_engine
    registry = engine.registry
    hos = getattr(env, "hos", None)
    clock = hos.clock(tid) if hos is not None else None
    node = int(truck.current_node)
    battery = float(truck.current_battery)
    if truck.is_complete or truck.failed or state in ("complete", "failed"):
        return TruckStart(tid, INF, node, battery, available=False, activity=state)
    owned = [r for r in registry.requests() if r.owner == tid and r.status not in _DONE]
    onboard = tuple(
        int(r.request_id) for r in owned
        if r.status in (RequestStatus.LOADING, RequestStatus.ON_BOARD, RequestStatus.CLAIMED_DELIVERY)
    )
    claimed = [int(r.request_id) for r in owned if r.status is RequestStatus.CLAIMED_PICKUP]
    delivering = [int(r.request_id) for r in owned if r.status is RequestStatus.CLAIMED_DELIVERY]
    t, var = now, 0.0
    event = _pending_event(env, tid)
    if state == "routing":
        destination = int(truck.route_destination)
        depart = float(truck.routing_start_time)
        hours = model.net.h(node, destination)
        elapsed = hours
        if hos is not None and model.ops is not None:
            elapsed, clock = model.ops.drive(model.ops.convert(clock), depart, hours)
        t = max(now, depart + elapsed)
        var = model.sigma2(hours, depart) if t > now + 1e-9 else 0.0
        battery = battery - model.need(node, destination)
        node = destination
    elif state == "unloading":
        start = float(truck.unloading_start_time or now)
        nominal = 0.0
        for r in owned:
            if r.status is RequestStatus.IN_SERVICE:
                nominal = float(r.delivery.service_hours)
        t = max(now, start + nominal)
        if clock is not None and model.ops is not None:
            clock = model.ops.work(model.ops.convert(clock), start, max(0.0, t - start))
    elif state in ("loading", "charging", "resting", "waiting_for_requests", "waiting_for_service"):
        if event is not None:
            t = max(now, float(event.time))
        ops = model.ops
        if clock is not None and ops is not None:
            clock = ops.convert(clock)
        if state == "charging" and event is not None:
            battery = min(model.capacity, battery + float(event.data.get("charge_amount", 0.0)))
            if clock is not None and ops is not None:
                clock = ops.work(clock, now, t - now, HosActivity.CHARGE)
        elif state == "resting" and clock is not None and ops is not None and event is not None:
            kind = event.data.get("kind", "daily")
            start = float(event.data.get("start", now))
            activity = HosActivity.REST_DAILY if kind == "daily" else HosActivity.REST_BREAK
            clock = ops.work(clock, start, max(0.0, t - start), activity)
        elif state == "waiting_for_service":
            # The service starts by itself at the window; plan from here.
            t = now
        elif clock is not None and ops is not None and t > now:
            activity = HosActivity.SERVICE if state == "loading" else HosActivity.IDLE
            clock = ops.work(clock, now, t - now, activity)
    elif state == "waiting_to_charge":
        t = now + model.station_wait(node)
    if clock is not None and model.ops is not None:
        clock = model.ops.advance(model.ops.convert(clock) if not isinstance(clock, tuple) else clock, t)
    start = TruckStart(tid, t, node, battery, var=var, clock=clock, activity=state)
    if engine.is_drayage:
        container = [r for r in onboard]
        if container:
            start.first = container[0]
            start.container_loaded = True
            start.onboard = (container[0],)
        elif claimed:
            start.first = claimed[0]
    else:
        start.onboard = tuple(sorted(onboard, key=lambda r: (r not in delivering, r)))
        start.away = bool(onboard) and not (node == model.depot and state in ("ready", "loading"))
        if delivering:
            start.first = delivering[0]
    for r in (*start.onboard, *(() if start.first is None else (start.first,))):
        if r not in jobs:
            request = registry.get(r)
            jobs[r] = _job(request, model.depot, engine.is_drayage)
    return start


# ---------------------------------------------------------------------------
# Fleet planner (ALNS)
# ---------------------------------------------------------------------------


def _signature(ts: TruckStart):
    clock = ts.clock
    clock_key = None
    if isinstance(clock, tuple):
        clock_key = tuple(None if v is None else round(v, 4) for v in clock)
    elif clock is not None:
        clock_key = clock
    return (round(ts.t, 4), ts.node, round(ts.battery, 2), round(ts.var, 4), clock_key, ts.onboard,
            ts.away, ts.first, ts.container_loaded)


class FleetPlanner:
    def __init__(self, model: _BaseModel, parameters: PlannerParameters):
        self.model = model
        self.parameters = parameters

    def plan(self, snapshot: Snapshot, warm: dict | None, rng: random.Random, iterations: int):
        model, jobs = self.model, snapshot.jobs
        trucks = [ts for ts in snapshot.trucks if ts.available]
        self.trucks = trucks
        self.jobs = jobs
        self.sig = {ts.tid: _signature(ts) for ts in trucks}
        self.memo: dict = {}
        self.ts_of = {ts.tid: ts for ts in trucks}
        locked = set()
        routes = {}
        for ts in trucks:
            routes[ts.tid] = model.empty_route(ts)
            locked |= model.locked(ts)
        # Warm start: keep previous orders of requests that are still plannable.
        assigned = set(locked)
        if warm:
            for ts in trucks:
                previous = warm.get(ts.tid)
                if not previous:
                    continue
                routes[ts.tid] = self._restore(ts, routes[ts.tid], previous, assigned)
        pool = [r for r in jobs if r not in assigned and self._plannable(r)]
        routes = {tid: self._polish(tid, route) for tid, route in routes.items()}
        routes, pool = self._repair(routes, pool, rng, noise=0.0, regret=False)
        current = (routes, pool)
        current_value = self._total(routes)
        best, best_value = current, current_value
        free = [r for r in jobs if r not in locked]
        if not free:
            return best[0], self._probs(best[0])
        temperature = self.parameters.temperature
        for iteration in range(iterations):
            routes, pool = current
            routes = dict(routes)
            pool = list(pool)
            removed = self._destroy(routes, locked, rng)
            if not removed and not pool:
                continue
            noise = 0.1 if rng.random() < 0.5 else 0.0
            regret = rng.random() < 0.3
            candidate, left = self._repair(routes, pool + removed, rng, noise=noise, regret=regret)
            if rng.random() < 0.2:
                tid = rng.choice(list(candidate))
                candidate[tid] = self._polish(tid, candidate[tid], passes=1)
            value = self._total(candidate)
            heat = temperature * (1.0 - iteration / max(1, iterations))
            if value > current_value + 1e-9 or (
                heat > 0 and rng.random() < math.exp((value - current_value) / heat)
            ):
                current, current_value = (candidate, left), value
            if value > best_value + 1e-9:
                best, best_value = (candidate, left), value
        routes = {tid: self._polish(tid, route) for tid, route in best[0].items()}
        routes, _ = self._repair(routes, list(best[1]), rng, noise=0.0, regret=False)
        return routes, self._probs(routes)

    def _polish(self, tid, route, passes: int = 3):
        """First-improvement relocation inside the route (intra-trip order)."""
        value = self._eval(tid, route).value
        for _ in range(passes):
            improved = False
            for candidate in self.model.reorders(self.ts_of[tid], route):
                candidate_value = self._eval(tid, candidate).value
                if candidate_value > value + 1e-9:
                    route, value, improved = candidate, candidate_value, True
                    break
            if not improved:
                break
        return route

    # -- helpers -----------------------------------------------------------------

    def _plannable(self, r) -> bool:
        return True

    def _restore(self, ts, base, previous, assigned):
        model, jobs = self.model, self.jobs
        if model.kind == "drayage":
            route = list(base)
            for r in previous:
                if r == REST and REST not in route:
                    route.append(r)
                elif r in jobs and r not in assigned:
                    route.append(r)
                    assigned.add(r)
            while route and route[-1] == REST:
                route.pop()
            return tuple(route)
        trips = [list(trip) for trip in base]
        onboard = set(ts.onboard)
        for j, trip in enumerate(previous):
            if j == 0 and trips and onboard:
                # Keep the previous delivery order of the trip being loaded or driven.
                order = [r for r in trip if r in onboard or (r in jobs and r not in assigned)]
                order += [r for r in trips[0] if r not in order]
                if ts.first is not None and ts.first in order:
                    order.remove(ts.first)
                    order.insert(0, ts.first)
                if ts.away:
                    extra = [r for r in order if r not in onboard]
                    trips[0] = [r for r in order if r in onboard]
                    if extra:
                        trips.append(extra)
                        assigned.update(extra)
                else:
                    trips[0] = order
                    assigned.update(order)
                continue
            keep = [r for r in trip if r in jobs and r not in assigned]
            if not keep:
                continue
            trips.append(keep)
            assigned.update(keep)
        load_ok = []
        for j, trip in enumerate(trips):
            load = 0.0
            kept = []
            for r in trip:
                if load + jobs[r].demand <= model.payload + 1e-9 or r in ts.onboard:
                    kept.append(r)
                    load += jobs[r].demand
                else:
                    assigned.discard(r)
            if kept:
                load_ok.append(tuple(kept))
        return tuple(load_ok[: model.parameters.max_trips + 1])

    def _eval(self, tid, route):
        key = (self.sig[tid], route)
        result = self.memo.get(key)
        if result is None:
            result = self.memo[key] = self.model.evaluate(self.ts_of[tid], route, self.jobs)
        return result

    def _total(self, routes) -> float:
        return sum(self._eval(tid, route).value for tid, route in routes.items())

    def _probs(self, routes) -> dict:
        probs = {}
        for tid, route in routes.items():
            probs.update(self._eval(tid, route).probs)
        return probs

    def _destroy(self, routes, locked, rng):
        model = self.model
        members = [(tid, r) for tid, route in routes.items() for r in model.members(route) if r not in locked]
        if not members:
            return []
        limit = max(1, int(round(self.parameters.destroy_fraction * len(members))))
        q = rng.randint(1, max(1, min(limit, 6)))
        operator = rng.random()
        chosen: list = []
        if operator < 0.35:
            chosen = rng.sample(members, min(q, len(members)))
        elif operator < 0.6:
            # Worst: lowest planned on-time probability.
            scored = []
            for tid, r in members:
                p = self._eval(tid, routes[tid]).probs.get(r, 0.0)
                scored.append((p + 0.3 * rng.random(), tid, r))
            scored.sort()
            chosen = [(tid, r) for _, tid, r in scored[:q]]
        elif operator < 0.85:
            # Related (Shaw): close in space and time to a seed request.
            seed_tid, seed = rng.choice(members)
            a = self.jobs[seed]
            net = model.net

            def relatedness(item):
                b = self.jobs[item[1]]
                return (net.h(a.delivery, b.delivery) + net.h(a.pickup, b.pickup)
                        + 0.5 * abs(a.deadline - b.deadline) + 0.5 * abs(a.release - b.release))

            members.sort(key=relatedness)
            chosen = members[:q]
        else:
            tids = sorted({tid for tid, _ in members})
            tid = rng.choice(tids)
            chosen = [(t2, r) for t2, r in members if t2 == tid]
        removed = []
        for tid, r in chosen:
            routes[tid] = model.remove(routes[tid], r)
            removed.append(r)
        return removed

    def _best_insertion(self, tid, route, r, base_value):
        best = None
        for candidate in self.model.insertions(self.ts_of[tid], route, r, self.jobs):
            delta = self._eval(tid, candidate).value - base_value
            if best is None or delta > best[0]:
                best = (delta, candidate)
        return best

    def _repair(self, routes, pool, rng, noise: float, regret: bool):
        routes = dict(routes)
        pool = list(dict.fromkeys(pool))
        if not pool:
            return routes, []
        base = {tid: self._eval(tid, route).value for tid, route in routes.items()}
        table = {r: {} for r in pool}
        for r in pool:
            for tid, route in routes.items():
                table[r][tid] = self._best_insertion(tid, route, r, base[tid])
        while pool:
            choice = None
            for r in pool:
                options = sorted(
                    (item for item in ((v[0], tid, v[1]) for tid, v in table[r].items() if v is not None)),
                    key=lambda item: -item[0],
                )
                if not options or options[0][0] <= 1e-9:
                    continue
                score = options[0][0]
                if regret:
                    second = options[1][0] if len(options) > 1 else 0.0
                    score = (options[0][0] - max(second, 0.0)) + 1e-3 * options[0][0]
                if noise:
                    score *= 1.0 + noise * (2.0 * rng.random() - 1.0)
                if choice is None or score > choice[0]:
                    choice = (score, r, options[0][1], options[0][2])
            if choice is None:
                break
            _, r, tid, route = choice
            routes[tid] = route
            base[tid] = self._eval(tid, route).value
            pool.remove(r)
            for other in pool:
                table[other][tid] = self._best_insertion(tid, route, other, base[tid])
        return routes, pool


# ---------------------------------------------------------------------------
# The policy
# ---------------------------------------------------------------------------


class RollingHorizonRequestPolicy:
    """Re-plan the fleet at every decision; the active truck takes its first step."""

    name = "rolling_horizon"

    def __init__(self, parameters: PlannerParameters | None = None):
        self.parameters = parameters or PlannerParameters()
        self._episode_key = None
        self._plan: dict = {}
        self._model = None
        self.decisions = 0
        self.plan_log: list = []

    # -- episode bookkeeping -------------------------------------------------------

    def _begin(self, env) -> None:
        key = (int(env.scenario_seed), id(env))
        if key != self._episode_key:
            self._episode_key = key
            self._plan = {}
            self.decisions = 0
            engine = env.request_engine
            model_type = DrayageModel if engine.is_drayage else SameDayModel
            self._model = model_type(env, self.parameters)

    def __call__(self, env, observation=None, info=None) -> int:
        if getattr(env, "request_engine", None) is None or not getattr(env, "dynamic_requests", False):
            raise ValueError("the rolling-horizon request policy needs a dynamic request mode")
        if env.episode_steps == 0:
            self._episode_key = None
        self._begin(env)
        model = self._model
        decisions = joint_action_feasibility(env)
        feasible = [index for index, decision in enumerate(decisions) if decision.feasible]
        if not feasible:
            raise RuntimeError("no feasible action; the policy never relaxes the mask")
        truck = env.trucks[env.active_truck_id]
        forced = self._forced(env, model, truck, decisions)
        if forced is not None:
            return forced
        snapshot = build_snapshot(env, model, self.parameters.clairvoyant)
        rng = random.Random(hash((self.parameters.seed, int(env.scenario_seed), self.decisions)) & 0xFFFFFFFF)
        planner = FleetPlanner(model, self.parameters)
        free = len(snapshot.jobs)
        iterations = max(self.parameters.min_iterations, min(self.parameters.iterations, 4 * free))
        routes, probs = planner.plan(snapshot, self._plan, rng, iterations)
        self._plan = routes
        self.decisions += 1
        ts = next(ts for ts in snapshot.trucks if ts.tid == truck.truck_id)
        route = routes.get(truck.truck_id, ())
        action = self._act(env, model, snapshot, ts, route, decisions)
        if action is None or not decisions[action].feasible:
            action = self._fallback(env, model, truck, decisions)
        return int(action)

    # -- action mapping --------------------------------------------------------------

    @staticmethod
    def _slot_action(env, rid: int) -> int:
        return env.num_charging_nodes + int(rid)

    @staticmethod
    def _depot_action(env) -> int:
        return env.num_charging_nodes + env.fixed_num_stops

    @staticmethod
    def _charger_action(env, node: int) -> int | None:
        try:
            return list(env.charging_nodes).index(int(node))
        except ValueError:
            return None

    @staticmethod
    def _extra_action(env, kind: ActionKind) -> int | None:
        kinds = list(getattr(env, "extra_action_kinds", ()))
        if kind not in kinds:
            return None
        return env.extra_action_offset + kinds.index(kind)

    def _charge_action(self, env, decisions, target: float | None) -> int | None:
        options = []
        for j, value in enumerate(env.charge_action_values):
            index = env.num_navigation_actions + j
            if decisions[index].feasible:
                options.append((float(value), index))
        if not options:
            return None
        if target is None:
            return max(options)[1]
        above = [item for item in options if item[0] >= target - 1e-9]
        return (min(above) if above else max(options))[1]

    def _feasible(self, decisions, action) -> int | None:
        return action if action is not None and decisions[action].feasible else None

    def _forced(self, env, model, truck, decisions) -> int | None:
        """Kullman's station epochs: queue-or-leave and the committed charge."""
        stations = getattr(env, "public_stations", None)
        if stations is None:
            return None
        if stations.committed(truck.truck_id):
            charge = self._charge_action(env, decisions, None)
            if charge is not None:
                plan_target = self._planned_target(env, model, truck)
                return self._charge_action(env, decisions, plan_target) or charge
            return None
        queue = self._extra_action(env, ActionKind.QUEUE)
        if queue is not None and decisions[queue].feasible:
            # Leave for the private depot charger when it is reachable and the
            # station's expected wait (as observed on arrival) is long.
            node = int(truck.current_node)
            row = stations.feature_row(node, truck.truck_id, env.global_clock, {truck.truck_id})
            expected = float(row[4])
            depot = self._depot_action(env)
            if (
                model.depot_charging
                and expected > 0.75
                and decisions[depot].feasible
                and env.request_engine.container(truck.truck_id) is None
            ):
                return depot
            return queue
        return None

    def _planned_target(self, env, model, truck) -> float | None:
        return None

    def _safe(self, model, truck, target_node: int) -> bool:
        node = int(truck.current_node)
        if target_node == node:
            return True
        left = float(truck.current_battery) - model.need(node, target_node)
        return left >= model.reserve(target_node) - 1e-6

    def _act(self, env, model, snapshot, ts, route, decisions) -> int | None:
        if model.kind == "same_day":
            return self._act_same_day(env, model, snapshot, ts, route, decisions)
        return self._act_drayage(env, model, snapshot, ts, route, decisions)

    def _navigate(self, env, model, truck, node, decisions, rid=None) -> int | None:
        if rid is not None:
            action = self._slot_action(env, rid)
        elif node == model.depot:
            action = self._depot_action(env)
        else:
            action = self._charger_action(env, node)
        if action is None or not decisions[action].feasible:
            return None
        if not self._safe(model, truck, node):
            return None
        return action

    def _act_same_day(self, env, model, snapshot, ts, route, decisions) -> int | None:
        truck = env.trucks[ts.tid]
        now = snapshot.now
        node = int(truck.current_node)
        jobs = snapshot.jobs
        wait = self._extra_action(env, ActionKind.WAIT)
        if route:
            steps = model.evaluate(ts, route, jobs, trace=True).steps
            loads = [step for step in steps if step[0] == "load"]
            for i, step in enumerate(steps):
                kind = step[0]
                if kind == "drive":
                    target = step[1]
                    rid = self._delivery_after(steps, i, jobs, target)
                    return self._navigate(env, model, truck, target, decisions, rid)
                if kind == "charge":
                    if step[1] == node:
                        action = self._charge_action(env, decisions, step[2])
                        if action is not None:
                            return action
                    continue
                if kind == "arrive_depot":
                    continue
                if kind == "load":
                    ready = [s for s in loads if s[2] <= now + 1e-6]
                    if ready:
                        return self._feasible(decisions, self._slot_action(env, ready[0][1]))
                    if wait is not None and decisions[wait].feasible:
                        return wait
                    return self._feasible(decisions, self._slot_action(env, loads[0][1]))
                if kind == "depart":
                    continue
                if kind == "deliver":
                    return self._feasible(decisions, self._slot_action(env, step[1]))
        return self._idle(env, model, snapshot, truck, decisions)

    @staticmethod
    def _delivery_after(steps, i, jobs, target):
        for step in steps[i + 1:]:
            if step[0] == "deliver":
                return step[1] if jobs[step[1]].delivery == target else None
            if step[0] in ("drive", "charge", "load"):
                return None
        return None

    def _idle(self, env, model, snapshot, truck, decisions) -> int | None:
        node = int(truck.current_node)
        wait = self._extra_action(env, ActionKind.WAIT)
        end = self._extra_action(env, ActionKind.END_SHIFT)
        if node == model.depot:
            if snapshot.future_information:
                soc = truck.current_battery / model.capacity
                if soc < 1.0 - 1e-6:
                    charge = self._charge_action(env, decisions, 1.0)
                    if charge is not None:
                        return charge
                if wait is not None and decisions[wait].feasible:
                    return wait
            if end is not None and decisions[end].feasible:
                return end
            if wait is not None and decisions[wait].feasible:
                return wait
            return None
        if model.kind == "drayage" and snapshot.future_information and wait is not None and decisions[wait].feasible:
            if truck.current_battery - model.reserve(node) > 0.25 * model.capacity:
                return wait
        action = self._navigate(env, model, truck, model.depot, decisions)
        if action is not None:
            return action
        return None

    def _act_drayage(self, env, model, snapshot, ts, route, decisions) -> int | None:
        truck = env.trucks[ts.tid]
        node = int(truck.current_node)
        jobs = snapshot.jobs
        wait = self._extra_action(env, ActionKind.WAIT)
        rest_daily = self._extra_action(env, ActionKind.REST_DAILY)
        if route:
            steps = model.evaluate(ts, route, jobs, trace=True).steps
            first = next((r for r in route if r != REST), None)
            for i, step in enumerate(steps):
                kind = step[0]
                if kind == "charge":
                    if step[1] == node:
                        action = self._charge_action(env, decisions, step[2])
                        if action is not None:
                            return action
                    continue
                if kind == "rest":
                    if rest_daily is not None and decisions[rest_daily].feasible:
                        return rest_daily
                    continue
                if kind == "early":
                    early = step[1]
                    if early > self.parameters.wait_slack_hours and wait is not None and decisions[wait].feasible:
                        return wait
                    continue
                if kind == "drive":
                    target = step[1]
                    if first is None:
                        return self._navigate(env, model, truck, target, decisions)
                    job = jobs[first]
                    rid = None
                    if target == job.pickup and not ts.container_loaded:
                        rid = first
                    elif target == job.delivery and ts.container_loaded:
                        rid = first
                    action = self._navigate(env, model, truck, target, decisions, rid)
                    if action is None:
                        action = self._hos_rest(env, truck, decisions)
                    return action
                if kind == "deliver":
                    return self._feasible(decisions, self._slot_action(env, step[1]))
        # Idle: rest if the shift window is nearly used and no known request is
        # due before the rest would end, else charge or wait.
        if model.regime is not None and rest_daily is not None and decisions[rest_daily].feasible:
            clock = env.hos.clock(truck.truck_id)
            clock = advance_to(model.regime, clock, max(clock.time, snapshot.now), HosActivity.IDLE)
            if (window_left(model.regime, clock) < self.parameters.idle_rest_window_hours
                    and snapshot.future_information
                    and not model.rest_blocked(snapshot.now, jobs, list(jobs))):
                return rest_daily
        return self._idle(env, model, snapshot, truck, decisions)

    def _hos_rest(self, env, truck, decisions) -> int | None:
        hos = getattr(env, "hos", None)
        if hos is None:
            return None
        try:
            kind, _ = required_rest(hos.regime, hos.clock(truck.truck_id))
        except ValueError:
            kind = None
        if kind is None:
            return None
        action = self._extra_action(env, ActionKind.REST_DAILY if kind == "daily" else ActionKind.REST_BREAK)
        return self._feasible(decisions, action)

    def _fallback(self, env, model, truck, decisions) -> int:
        """A safe feasible action: rest, refuge, wait, end shift, else the first."""
        rest = self._hos_rest(env, truck, decisions)
        if rest is not None:
            return rest
        node = int(truck.current_node)
        engine = env.request_engine
        container = engine.container(truck.truck_id) if engine.is_drayage else None
        onboard = engine.registry.on_board(truck.truck_id)
        targets = []
        if container is not None:
            targets.append((container.delivery.node, container.request_id))
        for request in onboard:
            targets.append((request.delivery.node, request.request_id))
        for target, rid in targets:
            action = self._slot_action(env, rid)
            if decisions[action].feasible and self._safe(model, truck, int(target)):
                return action
        charge = self._charge_action(env, decisions, None)
        if charge is not None and truck.current_battery < 0.5 * model.capacity:
            return charge
        if not onboard and container is None:
            action = self._navigate(env, model, truck, model.depot, decisions)
            if action is not None and node != model.depot:
                return action
        for kind in (ActionKind.WAIT, ActionKind.END_SHIFT):
            action = self._extra_action(env, kind)
            if action is not None and decisions[action].feasible:
                return action
        # Nearest safe refuge.
        best = None
        for c in model.refuges:
            action = self._depot_action(env) if c == model.depot else self._charger_action(env, c)
            if action is not None and decisions[action].feasible and self._safe(model, truck, c):
                hours = model.net.h(node, c)
                if best is None or hours < best[0]:
                    best = (hours, action)
        if best is not None:
            return best[1]
        for index, decision in enumerate(decisions):
            if decision.feasible and decision.action_kind is not ActionKind.QUEUE:
                return index
        return next(index for index, decision in enumerate(decisions) if decision.feasible)
