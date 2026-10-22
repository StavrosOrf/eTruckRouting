"""ALNS over customer routes for the static long-haul problems.

Routes are customer sequences; the legs between consecutive stops are the
waypoint closure of :mod:`network`, so a route's nominal cost is

    driving (closure hours) + charging (closure kWh x expected draw - start
    battery) / rate + service + a small charging-waste allowance
    [+ expected station waits per charging stop]
    [+ 10 h per extra hours-of-service window, with usable windows of
       ``hos_window`` on-duty and ``hos_driving`` driving hours]

which is what the simulator's operating hours count for one truck. Routes
must fit the payload and finish ``duration_margin`` hours before the horizon.
Unused trucks retire at the depot at time 0 (they cost nothing). Trucks may
start mid-route (rolling re-planning): each has a start node, time, battery
and payload left, and the customers it is already driving to stay first.

ALNS (Ropke & Pisinger 2006): random, worst, related (Shaw) and route
removal; greedy and regret-2/3 insertion with noise; simulated-annealing
acceptance with adaptive operator weights, then an intra-route 2-opt/or-opt
and inter-route relocate/swap descent.
"""

from __future__ import annotations

import math
import random
import time as _time
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class PlannerSettings:
    iterations: int = 6000
    time_limit_seconds: float = 30.0
    seed: int = 0
    duration_margin: float = 4.0
    min_destroy: float = 0.1
    max_destroy: float = 0.45
    start_temperature: float = 0.03
    cooling: float = 0.9993
    segment: int = 100
    reaction: float = 0.3
    noise: float = 0.05
    energy_expected: float = 1.02
    waste_hours: float = 0.4
    wait_per_stop: float = 0.0
    stop_energy_fraction: float = 0.6
    hos_window: float = 13.5
    hos_driving: float = 10.5
    rest_hours: float = 10.0
    infeasible_penalty: float = 1000.0


@dataclass
class TruckStart:
    """Where a truck's remaining route starts (rolling re-planning)."""

    truck_id: int
    node: int  # matrix index
    time: float
    battery: float
    payload: float
    fixed: tuple[int, ...] = ()  # customer positions it must serve first, in order
    window_used: float = 0.0  # HOS on-duty hours already used in the current window
    driving_used: float = 0.0
    fresh_window: bool = True


@dataclass
class PlanProblem:
    depot: int  # matrix index
    customers: list[int]  # matrix indices
    demand: list[float]
    service: list[float]
    starts: list[TruckStart]
    horizon: float
    capacity: float
    hos: bool = False
    # Closure tables (filled by build()).
    cost: np.ndarray = field(default=None, repr=False)
    hours: np.ndarray = field(default=None, repr=False)
    energy: np.ndarray = field(default=None, repr=False)


class LonghaulPlanner:
    """ALNS over the customers of one scenario."""

    def __init__(self, network, problem: PlanProblem, settings: PlannerSettings | None = None):
        self.net = network
        self.p = problem
        self.s = settings or PlannerSettings()
        n = len(problem.customers)
        self.n = n
        # Local index: 0..n-1 customers, n = depot, n+1+i = truck i's start.
        points = list(problem.customers) + [problem.depot] + [start.node for start in problem.starts]
        index = np.asarray(points, dtype=np.int64)
        self.H = network.closure_hours[np.ix_(index, index)].tolist()
        self.E = network.closure_energy[np.ix_(index, index)].tolist()
        self.C = network.closure_cost[np.ix_(index, index)].tolist()
        self.depot = n
        self.rate = network.rate
        self.demand = list(problem.demand)
        self.service = list(problem.service)
        self.related = [
            sorted(range(n), key=lambda j, i=i: self.C[i][j] + self.C[j][i]) for i in range(n)
        ]
        self.fixed_owner = {}
        for truck, start in enumerate(problem.starts):
            for customer in start.fixed:
                self.fixed_owner[customer] = truck
        self.points = index
        self._hops: dict[tuple[int, int], list[tuple[float, float, bool]]] = {}
        self._sim_cache: dict[tuple, float] = {}
        self.is_station = np.zeros(len(network.nodes), dtype=bool)
        self.is_station[network.chargers] = True

    def hops(self, a: int, b: int) -> list[tuple[float, float, bool]]:
        """(hours, kWh, starts at a station) of each hop of the closure path a -> b."""
        key = (a, b)
        cached = self._hops.get(key)
        if cached is None:
            net = self.net
            source, target = int(self.points[a]), int(self.points[b])
            path = net.path(source, target) or [target]
            out, previous = [], source
            for node in path:
                out.append((float(net.hours[previous, node]), float(net.energy[previous, node]),
                            bool(self.is_station[previous])))
                previous = node
            cached = self._hops[key] = out
        return cached

    def hos_duration(self, truck: int, route: list[int]) -> float:
        """Hop-level simulation of driving, charging and US-style rests.

        Charging is deferred to where it is needed, and a station where a
        daily rest falls is used to top up first (charging after the window
        closes costs no driving window). Mirrors what the executor does.
        """
        key = (truck, tuple(route))
        cached = self._sim_cache.get(key)
        if cached is not None:
            return cached
        s, start = self.s, self.p.starts[truck]
        margin, capacity, rate = self.net.margin, self.p.capacity, self.rate
        kappa = s.energy_expected
        sequence = list(route) + [self.depot]
        legs = []
        previous = self.start_point(truck)
        for stop in sequence:
            legs.append((stop, self.hops(previous, stop)))
            previous = stop
        remaining = sum(e for _, hops in legs for _, e, _ in hops) * kappa
        window, driving_limit, break_after = 14.0, 11.0, 8.0
        time = 0.0
        window_start = -start.window_used
        shift_driving = start.driving_used
        since_break = 0.0
        battery = start.battery
        for stop, hops in legs:
            for hours, energy, at_station in hops:
                needs_rest = shift_driving + hours > driving_limit + 1e-9 or time + hours - window_start > window + 1e-9
                if at_station and (needs_rest or battery < margin * energy - 1e-9):
                    if needs_rest:
                        target = min(capacity, max(remaining, margin * energy))
                    else:
                        target = min(capacity, max(margin * energy, min(remaining, capacity)))
                    if target > battery + 1e-6:
                        hours_charged = (target - battery) / rate
                        time += hours_charged
                        battery = target
                        if hours_charged >= 0.5:
                            since_break = 0.0
                if needs_rest:
                    time += s.rest_hours
                    window_start = time
                    shift_driving = 0.0
                    since_break = 0.0
                if since_break + hours > break_after + 1e-9:
                    time += 0.5
                    since_break = 0.0
                time += hours
                shift_driving += hours
                since_break += hours
                battery -= energy * kappa
                remaining -= energy * kappa
            if stop != self.depot:
                time += self.service[stop]
        if s.wait_per_stop > 0.0:
            charged = max(0.0, sum(e for _, hops in legs for _, e, _ in hops) * kappa - start.battery)
            time += math.ceil(charged / (s.stop_energy_fraction * capacity)) * s.wait_per_stop
        time += s.waste_hours
        self._sim_cache[key] = time
        return time

    # ------------------------------------------------------------------
    def start_point(self, truck: int) -> int:
        return self.n + 1 + truck

    def route_totals(self, truck: int, route: list[int]):
        previous = self.start_point(truck)
        hours = energy = service = 0.0
        for customer in route:
            hours += self.H[previous][customer]
            energy += self.E[previous][customer]
            service += self.service[customer]
            previous = customer
        hours += self.H[previous][self.depot]
        energy += self.E[previous][self.depot]
        return hours, energy, service

    def duration(self, truck: int, hours: float, energy: float, service: float) -> float:
        """Remaining nominal hours of a route from its start (0 if it stays home)."""
        s, start = self.s, self.p.starts[truck]
        if hours <= 0.0 and service <= 0.0:
            return 0.0
        charge_kwh = max(0.0, energy * s.energy_expected - start.battery)
        charge = charge_kwh / self.rate
        work = hours + charge + service
        if charge_kwh > 0.0:
            work += s.waste_hours
            if s.wait_per_stop > 0.0:
                work += math.ceil(charge_kwh / (s.stop_energy_fraction * self.p.capacity)) * s.wait_per_stop
        if not self.p.hos:
            return work
        window_left = max(0.0, s.hos_window - start.window_used)
        driving_left = max(0.0, s.hos_driving - start.driving_used)
        windows = max(
            math.ceil(max(0.0, work - window_left) / s.hos_window - 1e-9),
            math.ceil(max(0.0, hours - driving_left) / s.hos_driving - 1e-9),
        )
        return work + windows * s.rest_hours

    def route_cost(self, truck: int, route: list[int]) -> tuple[float, bool]:
        start = self.p.starts[truck]
        if not route and start.time <= 0.0 and start.node == self.p.depot:
            return 0.0, True
        load = sum(self.demand[c] for c in route)
        if self.p.hos:
            duration = self.hos_duration(truck, route)
        else:
            hours, energy, service = self.route_totals(truck, route)
            duration = self.duration(truck, hours, energy, service)
        feasible = (
            load <= start.payload + 1e-9
            and start.time + duration <= self.p.horizon - self.s.duration_margin + 1e-9
        )
        return duration, feasible

    def total(self, routes) -> float:
        value = 0.0
        for truck, route in enumerate(routes):
            duration, feasible = self.route_cost(truck, route)
            value += duration
            if not feasible:
                value += self.s.infeasible_penalty
        return value

    # ------------------------------------------------------------------
    def _insert_delta(self, truck, route, totals, customer):
        """(delta cost, position) of the best feasible insertion, or None."""
        start = self.p.starts[truck]
        load = totals[3]
        if load + self.demand[customer] > start.payload + 1e-9:
            return None
        hours, energy, service, _, base = totals
        H, E = self.H, self.E
        fixed = len(start.fixed)
        best = None
        previous = self.start_point(truck)
        sequence = route + [self.depot]
        horizon = self.p.horizon - self.s.duration_margin - start.time
        for position in range(len(sequence)):
            following = sequence[position]
            if position >= fixed:
                dh = H[previous][customer] + H[customer][following] - H[previous][following]
                de = E[previous][customer] + E[customer][following] - E[previous][following]
                duration = self.duration(truck, hours + dh, energy + de, service + self.service[customer])
                if duration <= horizon + 1e-9:
                    delta = duration - base
                    if best is None or delta < best[0]:
                        best = (delta, position)
            previous = following
        return best

    def _totals(self, truck, route):
        hours, energy, service = self.route_totals(truck, route)
        load = sum(self.demand[c] for c in route)
        base = self.duration(truck, hours, energy, service)
        start = self.p.starts[truck]
        if not route and start.time <= 0.0 and start.node == self.p.depot:
            base = 0.0
        return hours, energy, service, load, base

    def _repair(self, routes, removed, rng, regret_k=1, noise=0.0):
        totals = [self._totals(t, r) for t, r in enumerate(routes)]
        pending = list(removed)
        rng.shuffle(pending)
        unplaced = []
        while pending:
            choice = None
            for customer in pending:
                options = []
                owner = self.fixed_owner.get(customer)
                for truck, route in enumerate(routes):
                    if owner is not None and truck != owner:
                        continue
                    found = self._insert_delta(truck, route, totals[truck], customer)
                    if found is not None:
                        delta = found[0]
                        if noise:
                            delta += noise * rng.uniform(-1, 1) * 10.0
                        options.append((delta, truck, found[1]))
                if not options:
                    continue
                options.sort()
                if regret_k <= 1:
                    score = -options[0][0]
                else:
                    score = sum(
                        (options[min(k, len(options) - 1)][0] - options[0][0]) for k in range(1, regret_k)
                    )
                    if len(options) == 1:
                        score = 1e6 - options[0][0]
                if choice is None or score > choice[0]:
                    choice = (score, customer, options[0][1], options[0][2])
                if regret_k <= 1:
                    # Greedy: insert in random order, first customer wins.
                    break
            if choice is None:
                unplaced.extend(pending)
                break
            _, customer, truck, position = choice
            routes[truck].insert(position, customer)
            totals[truck] = self._totals(truck, routes[truck])
            pending.remove(customer)
        for customer in unplaced:
            # Keep every customer planned: the least-loaded truck takes it.
            owner = self.fixed_owner.get(customer)
            truck = owner if owner is not None else min(
                range(len(routes)), key=lambda t: sum(self.demand[c] for c in routes[t])
            )
            routes[truck].append(customer)

    # -- destroy operators ------------------------------------------------
    def _free(self, routes):
        return [
            (truck, customer)
            for truck, route in enumerate(routes)
            for position, customer in enumerate(route)
            if position >= len(self.p.starts[truck].fixed)
        ]

    def _destroy_random(self, routes, count, rng):
        assigned = self._free(routes)
        rng.shuffle(assigned)
        removed = []
        for truck, customer in assigned[:count]:
            routes[truck].remove(customer)
            removed.append(customer)
        return removed

    def _destroy_worst(self, routes, count, rng):
        scored = []
        for truck, route in enumerate(routes):
            fixed = len(self.p.starts[truck].fixed)
            previous = self.start_point(truck)
            sequence = route + [self.depot]
            for position, customer in enumerate(route):
                following = sequence[position + 1]
                if position >= fixed:
                    saving = self.C[previous][customer] + self.C[customer][following] - self.C[previous][following]
                    scored.append((saving * (1.0 + 0.3 * rng.random()), truck, customer))
                previous = customer
        scored.sort(reverse=True)
        removed = []
        for _, truck, customer in scored[:count]:
            routes[truck].remove(customer)
            removed.append(customer)
        return removed

    def _destroy_related(self, routes, count, rng):
        free = self._free(routes)
        if not free:
            return []
        where = {customer: truck for truck, customer in free}
        seed = rng.choice(free)[1]
        removed = []
        for customer in self.related[seed]:
            if len(removed) >= count:
                break
            if customer in where and rng.random() < 0.9:
                routes[where[customer]].remove(customer)
                removed.append(customer)
        return removed

    def _destroy_route(self, routes, count, rng):
        candidates = [t for t, r in enumerate(routes) if len(r) > len(self.p.starts[t].fixed)]
        if not candidates:
            return []
        truck = rng.choice(candidates)
        fixed = len(self.p.starts[truck].fixed)
        removed = routes[truck][fixed:]
        routes[truck] = routes[truck][:fixed]
        if len(removed) < count:
            removed += self._destroy_related(routes, count - len(removed), rng)
        return removed

    # ------------------------------------------------------------------
    def initial(self, rng) -> list[list[int]]:
        routes = [list(start.fixed) for start in self.p.starts]
        fixed = set(self.fixed_owner)
        pending = [c for c in range(self.n) if c not in fixed]
        self._repair(routes, pending, rng, regret_k=2)
        return routes

    def solve(self, initial=None) -> tuple[list[list[int]], float, dict]:
        s = self.s
        rng = random.Random(s.seed)
        started = _time.perf_counter()
        current = [list(r) for r in initial] if initial is not None else self.initial(rng)
        current_cost = self.total(current)
        best, best_cost = [list(r) for r in current], current_cost
        destroys = [self._destroy_random, self._destroy_worst, self._destroy_related, self._destroy_route]
        repairs = [(1, 0.0), (2, 0.0), (3, 0.0), (2, s.noise)]
        dw, rw = [1.0] * len(destroys), [1.0] * len(repairs)
        ds, rs = [0.0] * len(destroys), [0.0] * len(repairs)
        du, ru = [0] * len(destroys), [0] * len(repairs)
        temperature = max(1e-6, s.start_temperature * max(current_cost, 1.0) / 30.0)
        free_count = max(1, len(self._free(current)))
        iterations = 0
        for iteration in range(s.iterations):
            if _time.perf_counter() - started > s.time_limit_seconds:
                break
            iterations = iteration + 1
            d = _roulette(dw, rng)
            r = _roulette(rw, rng)
            du[d] += 1
            ru[r] += 1
            count = max(1, round(rng.uniform(s.min_destroy, s.max_destroy) * free_count))
            candidate = [list(route) for route in current]
            removed = destroys[d](candidate, count, rng)
            if not removed:
                continue
            regret, noise = repairs[r]
            self._repair(candidate, removed, rng, regret_k=regret, noise=noise)
            cost = self.total(candidate)
            score = 0.0
            if cost < best_cost - 1e-9:
                best, best_cost = [list(x) for x in candidate], cost
                current, current_cost = candidate, cost
                score = 33.0
            elif cost < current_cost - 1e-9:
                current, current_cost = candidate, cost
                score = 13.0
            elif rng.random() < math.exp(-(cost - current_cost) / temperature):
                current, current_cost = candidate, cost
                score = 9.0
            ds[d] += score
            rs[r] += score
            temperature *= s.cooling
            if iterations % s.segment == 0:
                for weights, scores, uses in ((dw, ds, du), (rw, rs, ru)):
                    for k in range(len(weights)):
                        if uses[k]:
                            weights[k] = (1 - s.reaction) * weights[k] + s.reaction * scores[k] / uses[k]
                        scores[k], uses[k] = 0.0, 0
        best = self.descent(best)
        best_cost = self.total(best)
        return best, best_cost, {
            "iterations": iterations,
            "seconds": _time.perf_counter() - started,
            "cost": best_cost,
        }

    # -- local search -----------------------------------------------------
    def descent(self, routes):
        routes = [list(r) for r in routes]
        improved = True
        rounds = 0
        while improved and rounds < 50:
            improved = False
            rounds += 1
            for truck in range(len(routes)):
                if self._two_opt(truck, routes):
                    improved = True
                if self._or_opt(truck, routes):
                    improved = True
            if self._relocate(routes):
                improved = True
            if self._swap(routes):
                improved = True
        return routes

    def _cost_one(self, truck, route):
        duration, feasible = self.route_cost(truck, route)
        return duration + (0.0 if feasible else self.s.infeasible_penalty)

    def _two_opt(self, truck, routes):
        route = routes[truck]
        fixed = len(self.p.starts[truck].fixed)
        base = self._cost_one(truck, route)
        for i in range(fixed, len(route) - 1):
            for j in range(i + 1, len(route)):
                candidate = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                value = self._cost_one(truck, candidate)
                if value < base - 1e-7:
                    routes[truck] = candidate
                    return True
        return False

    def _or_opt(self, truck, routes):
        route = routes[truck]
        fixed = len(self.p.starts[truck].fixed)
        base = self._cost_one(truck, route)
        for length in (1, 2, 3):
            for i in range(fixed, len(route) - length + 1):
                segment = route[i:i + length]
                rest = route[:i] + route[i + length:]
                for j in range(fixed, len(rest) + 1):
                    if j == i:
                        continue
                    for piece in (segment, segment[::-1]) if length > 1 else (segment,):
                        candidate = rest[:j] + piece + rest[j:]
                        value = self._cost_one(truck, candidate)
                        if value < base - 1e-7:
                            routes[truck] = candidate
                            return True
        return False

    def _relocate(self, routes):
        costs = [self._cost_one(t, r) for t, r in enumerate(routes)]
        for a in range(len(routes)):
            fixed_a = len(self.p.starts[a].fixed)
            for i in range(fixed_a, len(routes[a])):
                customer = routes[a][i]
                if customer in self.fixed_owner:
                    continue
                without = routes[a][:i] + routes[a][i + 1:]
                cost_a = self._cost_one(a, without)
                for b in range(len(routes)):
                    if b == a:
                        continue
                    fixed_b = len(self.p.starts[b].fixed)
                    for j in range(fixed_b, len(routes[b]) + 1):
                        candidate = routes[b][:j] + [customer] + routes[b][j:]
                        cost_b = self._cost_one(b, candidate)
                        if cost_a + cost_b < costs[a] + costs[b] - 1e-7:
                            routes[a], routes[b] = without, candidate
                            return True
        return False

    def _swap(self, routes):
        costs = [self._cost_one(t, r) for t, r in enumerate(routes)]
        for a in range(len(routes)):
            for b in range(a + 1, len(routes)):
                fa, fb = len(self.p.starts[a].fixed), len(self.p.starts[b].fixed)
                for i in range(fa, len(routes[a])):
                    for j in range(fb, len(routes[b])):
                        ra = routes[a][:i] + [routes[b][j]] + routes[a][i + 1:]
                        rb = routes[b][:j] + [routes[a][i]] + routes[b][j + 1:]
                        value = self._cost_one(a, ra) + self._cost_one(b, rb)
                        if value < costs[a] + costs[b] - 1e-7:
                            routes[a], routes[b] = ra, rb
                            return True
        return False


def _roulette(weights, rng) -> int:
    total = sum(weights)
    if total <= 0:
        return rng.randrange(len(weights))
    threshold = rng.random() * total
    cumulative = 0.0
    for index, weight in enumerate(weights):
        cumulative += weight
        if cumulative >= threshold:
            return index
    return len(weights) - 1
