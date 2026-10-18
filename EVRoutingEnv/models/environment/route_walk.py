"""Bounded route-first sampling on cached, directed energy lookups.

Customers are chosen along a feasible walk, rather than sampled before a
resource-constrained charger search. Reserve lookahead permits another delivery
on the way home. Every segment between depot/customer stops visits at most one
station, including the final depot return.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from EVRoutingEnv.models.environment.route_certificate import NoFeasibleCertificate


@dataclass
class _ReturnPaths:
    chargers: np.ndarray
    escape: np.ndarray


class RouteWalkTopology:
    """Static map/vehicle data; contains no scenario RNG or mutable task state."""

    PROPOSALS = 32
    DEPOT_CACHE_SIZE = 16

    @classmethod
    def for_graph(cls, graph, charging_nodes, capacity, multiplier):
        if not np.isfinite(multiplier) or multiplier < 1.0:
            raise ValueError("energy_multiplier must be finite and at least 1")
        matrices = graph.cost_matrices() if hasattr(graph, "cost_matrices") else None
        key = (
            tuple(sorted(map(int, charging_nodes))),
            float(capacity),
            float(multiplier),
        )
        cached = getattr(graph, "_route_walk_topology", None)
        # Matrix identity changes on graph scaling/cache invalidation.
        if cached is not None and cached[0] == key and cached[1] is matrices:
            return cached[2]
        result = cls(graph, key[0], capacity, multiplier, matrices)
        graph._route_walk_topology = (key, matrices, result)
        return result

    def __init__(self, graph, charging_nodes, capacity, multiplier, matrices):
        if matrices is None:
            nodes = tuple(map(int, graph.get_all_nodes()))
            self.index = {node: i for i, node in enumerate(nodes)}
            raw = np.asarray(
                [[graph.get_path_energy(a, b) for b in nodes] for a in nodes]
            )
        else:
            self.index, raw, _ = matrices
            nodes = tuple(sorted(self.index, key=self.index.__getitem__))
        self.nodes = nodes
        self.capacity = float(capacity)
        self.multiplier = float(multiplier)
        self.energy = np.where(
            np.isfinite(raw) & (raw >= 0.0), raw * multiplier, np.inf
        )
        self.chargers = np.asarray(
            [self.index[c] for c in charging_nodes], dtype=np.int64
        )
        charging_set = set(charging_nodes)
        self.customers = np.asarray(
            [
                self.index[n]
                for n in nodes
                if n not in charging_set
                and graph.graph.in_degree(n) > 0
                and graph.graph.out_degree(n) > 0
            ],
            dtype=np.int64,
        )
        self._returns = OrderedDict()
        self._two_delivery_bounds = {}

    def _return_paths(self, depot, floors):
        key = (depot, tuple(floors))
        if key in self._returns:
            self._returns.move_to_end(key)
            return self._returns[key]
        # Only stations that can reach the depot without another station are
        # safe return anchors. Affordability uses one configured charge even
        # when arriving empty; directed costs are checked in the return direction.
        safe = self.chargers[self.energy[self.chargers, depot] <= floors + 1e-9]
        exits = np.append(safe, depot)
        escape = self.energy[:, exits].min(axis=1)
        result = _ReturnPaths(safe, escape)
        self._returns[key] = result
        if len(self._returns) > self.DEPOT_CACHE_SIZE:
            self._returns.popitem(last=False)
        return result

    def walk(
        self,
        rng,
        *,
        num_customers,
        num_trucks,
        initial_batteries,
        charge_transition,
        min_hop_energy_kwh,
        max_hop_energy_kwh,
    ):
        depot = int(self.customers[int(rng.integers(len(self.customers)))])
        if num_customers == 2 * num_trucks:
            bound = self._two_delivery_customer_bound(depot)
            if num_customers > bound:
                raise NoFeasibleCertificate(
                    f"depot supports at most {bound} distinct customers in two-delivery "
                    "routes through at most one charging station per segment"
                )

        def charge(node, battery):
            if charge_transition is None:
                return self.capacity, 0.0, 1.0
            return charge_transition(self.nodes[node], battery)

        floors = np.asarray([charge(int(c), 0.0)[0] for c in self.chargers])
        home = self._return_paths(depot, floors)
        # A swap-delete pool makes unique customer removal O(1).
        pool = self.customers[self.customers != depot].copy()
        size = len(pool)
        minimum = min_hop_energy_kwh * self.multiplier
        maximum = max_hop_energy_kwh * self.multiplier
        visits, routes = [], []
        bounded_hops = minimum > 0.0 or math.isfinite(maximum)

        # These synchronous helpers share only the current truck state.
        def navigate(target):
            nonlocal node, battery
            required = float(self.energy[node, target])
            if not math.isfinite(required) or required > battery + 1e-9:
                raise NoFeasibleCertificate(
                    "walk cannot close its reserved return path"
                )
            battery = max(0.0, battery - required)
            node = int(target)
            actions.append((self.nodes[node], 0.0, False))

        def recharge():
            nonlocal battery
            charged, _, value = charge(node, battery)
            if charged > battery + 1e-9:
                battery = float(charged)
                actions.append((self.nodes[node], value, True))

        def feasible(offsets):
            candidates = pool[offsets]
            hop = self.energy[previous, candidates]
            return (
                (self.energy[node, candidates] + reserve[candidates] <= battery + 1e-9)
                & (hop >= minimum - 1e-9)
                & (hop <= maximum + 1e-9)
            )

        for tid in range(num_trucks):
            quota = num_customers // num_trucks + (tid < num_customers % num_trucks)
            node, previous = depot, depot
            battery = float(initial_batteries[tid])
            actions = []

            position = 0
            while position < quota:
                reserve = home.escape
                if position + 1 < quota:
                    # Bound the common lookahead to a random set of possible
                    # continuations, rather than allocate a full pool-square
                    # matrix on every delivery in a large fleet.
                    continuations = pool[:size]
                    if size > self.PROPOSALS:
                        continuations = pool[
                            rng.choice(size, self.PROPOSALS, replace=False)
                        ]
                    reserve = self._delivery_return_reserve(
                        pool[:size], home, floors, continuations
                    )
                # Most proposals fit. Try scalar lookups first so the common
                # case allocates no per-delivery candidate arrays.
                offset = -1
                for _ in range(4):
                    proposed = int(rng.random() * size)
                    candidate = int(pool[proposed])
                    if (
                        self.energy[node, candidate] + reserve[candidate]
                        > battery + 1e-9
                    ):
                        continue
                    if (
                        bounded_hops
                        and not minimum - 1e-9
                        <= self.energy[previous, candidate]
                        <= maximum + 1e-9
                    ):
                        continue
                    offset = proposed
                    break
                if offset < 0:
                    offsets = rng.integers(size, size=min(self.PROPOSALS, size))
                    choices = offsets[feasible(offsets)]
                    if not len(choices):
                        # Avoid rejecting a customer that a proposal batch missed.
                        offsets = np.arange(size)
                        choices = offsets[feasible(offsets)]
                    if len(choices):
                        offset = int(choices[int(rng.integers(len(choices)))])
                if offset >= 0:
                    target = int(pool[offset])
                    navigate(target)
                    previous = target
                    visits.append((self.nodes[target], tid, position))
                    position += 1
                    size -= 1
                    pool[offset] = pool[size]
                    continue

                # Select a station/customer pair together. Never navigate to a
                # connector station hoping that a second station will help.
                reachable = self.energy[node, self.chargers] <= battery + 1e-9
                candidates = self.chargers[reachable]
                if not len(candidates):
                    raise NoFeasibleCertificate(
                        "no customer fits directly or through one charging station"
                    )
                budgets = floors[reachable]

                def bridges(
                    offsets,
                    previous=previous,
                    candidates=candidates,
                    reserve=reserve,
                    budgets=budgets,
                ):
                    samples = pool[offsets]
                    hop = self.energy[previous, samples]
                    return (
                        self.energy[np.ix_(candidates, samples)] + reserve[samples]
                        <= budgets[:, None] + 1e-9
                    ) & ((hop >= minimum - 1e-9) & (hop <= maximum + 1e-9))

                offsets = rng.integers(size, size=min(self.PROPOSALS, size))
                useful = bridges(offsets)
                if not useful.any():
                    offsets = np.arange(size)
                    useful = bridges(offsets)
                stations = np.flatnonzero(useful.any(axis=1))
                if not len(stations):
                    raise NoFeasibleCertificate(
                        "no energy-safe unused customer fits through one charging station"
                    )
                station = int(stations[int(rng.integers(len(stations)))])
                choices = offsets[useful[station]]
                offset = int(choices[int(rng.integers(len(choices)))])
                navigate(int(candidates[station]))
                recharge()
                target = int(pool[offset])
                navigate(target)
                previous = target
                visits.append((self.nodes[target], tid, position))
                position += 1
                size -= 1
                pool[offset] = pool[size]

            # The final customer-to-depot segment has the same one-station
            # limit. Empty trucks retire at the depot without travel.
            if self.energy[node, depot] > battery + 1e-9:
                possible = home.chargers[
                    self.energy[node, home.chargers] <= battery + 1e-9
                ]
                if not len(possible):
                    raise NoFeasibleCertificate(
                        "walk has no depot return through at most one charging station"
                    )
                target = int(possible[np.argmin(self.energy[node, possible])])
                if target != node:
                    navigate(target)
                recharge()
            if node != depot or not actions:
                navigate(depot)
            routes.append(tuple(actions))
        return self.nodes[depot], visits, tuple(routes)

    def _two_delivery_customer_bound(self, depot):
        """Cached necessary bound, assuming full charging and ignoring time.

        Mark every possible ordered pair, then count distinct customers in at
        least one pair. This upper bound avoids costly partial fleet walks when
        even those relaxed assumptions cannot cover the requested customers.
        It never claims that a matching or event-engine solution exists.
        """
        if depot in self._two_delivery_bounds:
            return self._two_delivery_bounds[depot]
        customers = self.customers[self.customers != depot]
        home = self._return_paths(depot, np.full(len(self.chargers), self.capacity))
        arrival = self.capacity - self.energy[depot, customers]
        reachable = self.chargers[
            self.energy[depot, self.chargers] <= self.capacity + 1e-9
        ]
        if len(reachable):
            arrival = np.maximum(
                arrival,
                (self.capacity - self.energy[np.ix_(reachable, customers)]).max(axis=0),
            )
        pairs = (
            self.energy[np.ix_(customers, customers)] + home.escape[customers]
            <= arrival[:, None] + 1e-9
        )
        if len(self.chargers):
            to_station = (
                self.energy[np.ix_(customers, self.chargers)] <= arrival[:, None] + 1e-9
            )
            from_station = (
                self.energy[np.ix_(self.chargers, customers)] + home.escape[customers]
                <= self.capacity + 1e-9
            )
            pairs |= to_station @ from_station
        np.fill_diagonal(pairs, False)
        bound = int((pairs.any(axis=0) | pairs.any(axis=1)).sum())
        self._two_delivery_bounds[depot] = bound
        return bound

    def _delivery_return_reserve(self, available, home, floors, continuations=None):
        """Battery needed to serve a distinct remaining customer on the way home.

        That next customer must retain its own direct/one-station depot exit.
        Excluding the current candidate from both direct and station-assisted
        continuations prevents counting one delivery twice. No charger-chain
        search is performed, and no RNG is consumed by the lookahead.
        """
        continuations = available if continuations is None else continuations
        direct = (
            self.energy[np.ix_(available, continuations)] + home.escape[continuations]
        )
        same_customer = available[:, None] == continuations
        direct[same_customer] = np.inf
        required = direct.min(axis=1)
        if len(self.chargers):
            station_exits = (
                self.energy[np.ix_(self.chargers, continuations)]
                + home.escape[continuations]
                <= floors[:, None] + 1e-9
            )
            # A station needs a continuation other than the candidate itself.
            own_exit = same_customer @ station_exits.T
            usable = station_exits.sum(axis=1)[None, :] > own_exit
            via_station = np.where(
                usable, self.energy[np.ix_(available, self.chargers)], np.inf
            )
            required = np.minimum(required, via_station.min(axis=1))
        reserve = np.full(len(self.nodes), np.inf)
        reserve[available] = required
        return reserve
