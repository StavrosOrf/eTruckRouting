"""ALNS over customer-only routes (RoutingBlocks) with exact charging pricing.

The RoutingBlocks instance holds the depot and the customers only. Its
ADPTW evaluation checks payload, time windows and the horizon; battery is
switched off there. Arc cost and duration come from the *leg surrogate*: the
cheapest station detour of each leg when the truck leaves full. That makes the
RoutingBlocks cost an optimistic relaxation of the true route cost that sees
realistic detours, so its granular local search and destroy/repair operators
steer toward good customer sequences.

Acceptance, the incumbent and every reported number use the *exact* cost: each
route is priced by the native `Core` (optimal charger stops and charge levels
under the environment's model) and cached by customer sequence. Only routes a
move changed are re-priced. A candidate with an unpriceable route (the surrogate
cannot see carried-over SoC) is rejected.
"""

from __future__ import annotations

import copy
import math
import random
import time
from dataclasses import asdict, dataclass

import numpy as np
import routingblocks as rb
from routingblocks.operators import (
    BestInsertionOperator,
    RandomInsertionOperator,
    RandomRemovalOperator,
    WorstRemovalOperator,
    blink_selector_factory,
    first_move_selector,
)

from .problem import Problem

BIG = 1.0e6


@dataclass
class SearchParameters:
    time_limit_seconds: float = 60.0
    max_iterations: int = 10**9
    seed: int = 0
    surrogate_soc: float = 1.0
    min_removed_fraction: float = 0.05
    max_removed_fraction: float = 0.30
    max_removed: int = 20
    min_largest_destroy: int = 5  # small instances still need destroys big enough to restructure a route
    granularity: int = 30
    delta_local_search: float = 0.5
    local_search_operators: str = "extended"  # "extended" adds segment swaps 0-3, 1-2, 2-2
    local_search_scope: str = "changed"  # "changed" routes only, or "all"
    adaptive_smoothing: float = 0.2
    adaptive_period: int = 50
    penalty_period: int = 50
    initial_penalty: float = 100.0
    target_feasibility: float = 0.9
    start_temperature: float = 0.5  # annealing temperature, in units of the incumbent's cost per customer
    end_temperature: float = 0.02
    spare_routes: int = 2
    pool: bool = True
    pool_interval_seconds: float = 120.0
    adopt_fraction: float = 0.5  # parallel: share of workers that restart from the shared best each round
    pool_elite_tolerance: float = 0.03  # routes of candidates within this of the best join the pool
    pool_seconds: float = 20.0  # HiGHS budget per set-partitioning solve; the last one is carved out of the limit
    score_new_best: float = 9.0
    score_improved: float = 4.0
    score_accepted: float = 1.0
    verbose: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


class NonEmptyRouteRemoval(rb.DestroyOperator):
    """Remove whole non-empty routes, leaving an empty slot so repair can still open a route.

    The stock operator also deletes the spare empty routes and can leave a
    solution with no route at all, which makes the repair operators loop.
    """

    def __init__(self, evaluation, instance, rng):
        rb.DestroyOperator.__init__(self)  # not super(): pybind trampoline
        self._evaluation = evaluation
        self._instance = instance
        self._rng = rng

    def can_apply_to(self, solution) -> bool:
        return any(not route.empty for route in solution)

    def apply(self, evaluation, solution, number_of_removed_vertices: int) -> list[int]:
        removed: list[int] = []
        while len(removed) < number_of_removed_vertices:
            candidates = [k for k, route in enumerate(solution) if not route.empty]
            if len(candidates) <= 1 and removed:
                break
            if not candidates:
                break
            k = candidates[self._rng.randint(0, len(candidates) - 1)]
            removed.extend(node.vertex_id for node in solution[k] if not node.vertex.is_depot)
            del solution[k]
            solution.add_route(rb.create_route(self._evaluation, self._instance, []))
        return removed

    def name(self) -> str:
        return "NonEmptyRouteRemoval"


class NeighbourRemoval(rb.DestroyOperator):
    """Shaw-style removal: a random seed, then its nearest not-yet-removed customers.

    ``neighbours[v]`` lists the other customers by increasing relatedness distance.
    Each candidate is skipped with probability ``blink`` so repeated calls differ.
    The stock RelatedRemovalOperator re-sorts every customer per removal in Python,
    which was a third of the run time at 100 customers.
    """

    def __init__(self, neighbours: list[list[int]], rng, blink: float = 0.1):
        rb.DestroyOperator.__init__(self)
        self._neighbours = neighbours
        self._rng = rng
        self._blink = blink

    def can_apply_to(self, solution) -> bool:
        return len(solution) > 0

    def apply(self, evaluation, solution, number_of_removed_vertices: int) -> list[int]:
        present = [solution.lookup(location).vertex_id for location in solution.non_depot_nodes]
        if not present:
            return []
        wanted = min(number_of_removed_vertices, len(present))
        inside = set(present)
        removed = [present[self._rng.randint(0, len(present) - 1)]]
        taken = {removed[0]}
        while len(removed) < wanted:
            seed = removed[self._rng.randint(0, len(removed) - 1)]
            added = False
            for v in self._neighbours[seed]:
                if v in taken or v not in inside:
                    continue
                if self._rng.uniform(0, 1) < self._blink:
                    continue
                removed.append(v)
                taken.add(v)
                added = True
                break
            if not added:  # every neighbour of this seed is gone or skipped
                for v in self._neighbours[seed]:
                    if v not in taken and v in inside:
                        removed.append(v)
                        taken.add(v)
                        added = True
                        break
            if not added and len(taken) >= len(inside):
                break
        solution.remove_vertices([solution.find(v)[0] for v in removed])
        return removed

    def name(self) -> str:
        return "NeighbourRemoval"


class Pricer:
    """Exact route costs with a cache keyed by the customer sequence."""

    def __init__(self, problem: Problem, max_entries: int = 2_000_000):
        self.problem = problem
        self.cache: dict[tuple, float] = {}
        self.max_entries = max_entries
        self.calls = 0
        self.hits = 0

    def cost(self, route: tuple) -> float:
        self.calls += 1
        value = self.cache.get(route)
        if value is not None:
            self.hits += 1
            return value
        value = self.problem.price(route)
        if len(self.cache) >= self.max_entries:
            self.cache.clear()
        self.cache[route] = value
        return value

    def total(self, routes) -> float:
        total = 0.0
        for route in routes:
            value = self.cost(tuple(route))
            if not math.isfinite(value):
                return math.inf
            total += value
        return total


class Search:
    def __init__(self, problem: Problem, parameters: SearchParameters):
        self.problem = problem
        self.params = parameters
        self.pricer = Pricer(problem)
        self.n = problem.n_customers
        self.fleet = problem.trucks
        self.py_random = random.Random(parameters.seed)
        self.rb_random = rb.Random(parameters.seed)
        self._build_instance()
        self._build_operators()
        self.start = time.time()
        self.iterations = 0
        self.log: list[str] = []
        self.elite: dict[frozenset, tuple[float, tuple]] = {}
        self._elite_new: set[frozenset] = set()
        self._surrogate_cache: dict[tuple, float] = {}

    # ------------------------------------------------------------------ setup
    def _build_instance(self) -> None:
        P = self.problem
        cost, hours = P.core.leg_surrogate(self.params.surrogate_soc)
        self.surrogate_cost = np.where(np.isfinite(cost), cost, BIG)
        self.surrogate_time = np.where(np.isfinite(hours), hours, BIG)
        np.fill_diagonal(self.surrogate_cost, 0.0)
        np.fill_diagonal(self.surrogate_time, 0.0)
        builder = rb.utility.InstanceBuilder(
            create_vertex=rb.adptw.create_adptw_vertex, create_arc=rb.adptw.create_adptw_arc
        )
        for v in range(self.n + 1):
            ready = P.tw_open[v]
            due = P.tw_close[v] if math.isfinite(P.tw_close[v]) else BIG
            if v == 0:
                ready, due = 0.0, P.horizon
            data = rb.adptw.VertexData(0.0, 0.0, P.demand[v], ready, due, P.service[v])
            if v == 0:
                builder.set_depot("0", data)
            else:
                builder.add_customer(str(v), data)
        for i in range(self.n + 1):
            for j in range(self.n + 1):
                builder.add_arc(
                    str(i), str(j),
                    rb.adptw.ArcData(float(self.surrogate_cost[i, j]), 0.0, float(self.surrogate_time[i, j])),
                )
        builder.fleet_size = self.fleet
        self.instance = builder.build()
        self.evaluation = rb.adptw.Evaluation(BIG, P.payload)
        self._set_penalties(*(self.params.initial_penalty,) * 3)

    def _set_penalties(self, overload: float, overcharge: float, time_shift: float) -> None:
        self.evaluation.overload_penalty_factor = overload
        self.evaluation.resource_penalty_factor = overcharge
        self.evaluation.time_shift_penalty_factor = time_shift
        self.penalties = [overload, overcharge, time_shift]

    def _build_operators(self) -> None:
        P, p = self.problem, self.params
        arcs = rb.ArcSet(self.instance.number_of_vertices)
        keep = max(1, min(p.granularity, self.n - 1))
        for i in range(1, self.n + 1):
            ranked = sorted(
                (j for j in range(1, self.n + 1) if j != i),
                key=lambda j: self.surrogate_cost[i, j] + self.surrogate_cost[j, i],
            )
            for j in ranked[keep:]:
                arcs.forbid_arc(i, j)
        self.arc_set = arcs
        self.local_search = rb.LocalSearch(self.instance, self.evaluation, None, rb.BestImprovementPivotingRule())
        self.ls_operators = [
            rb.operators.SwapOperator_0_1(self.instance, arcs),
            rb.operators.SwapOperator_0_2(self.instance, arcs),
            rb.operators.SwapOperator_1_1(self.instance, arcs),
            rb.operators.InterRouteTwoOptOperator(self.instance, arcs),
        ]
        if p.local_search_operators == "extended":
            self.ls_operators += [
                rb.operators.SwapOperator_0_3(self.instance, arcs),
                rb.operators.SwapOperator_1_2(self.instance, arcs),
                rb.operators.SwapOperator_2_2(self.instance, arcs),
            ]
        self.alns = rb.AdaptiveLargeNeighborhood(self.rb_random, p.adaptive_smoothing)
        self.alns.add_destroy_operator(RandomRemovalOperator(self.rb_random))
        self.alns.add_destroy_operator(NonEmptyRouteRemoval(self.evaluation, self.instance, self.rb_random))
        self.alns.add_destroy_operator(
            WorstRemovalOperator(self.instance, blink_selector_factory(0.02, self.rb_random)))
        closeness = P.hours + P.hours.T
        neighbours = [[]] + [
            [int(j) for j in np.argsort(closeness[i, : self.n + 1]) if j not in (0, i)]
            for i in range(1, self.n + 1)
        ]
        self.alns.add_destroy_operator(NeighbourRemoval(neighbours, self.rb_random, 0.1))
        self.alns.add_repair_operator(RandomInsertionOperator(self.rb_random))
        self.alns.add_repair_operator(BestInsertionOperator(self.instance, first_move_selector))
        self.alns.add_repair_operator(
            BestInsertionOperator(self.instance, blink_selector_factory(0.15, self.rb_random)))

    # --------------------------------------------------------------- utilities
    @property
    def elapsed(self) -> float:
        return time.time() - self.start

    def to_rb(self, routes, spare: int = 0) -> rb.Solution:
        built = [rb.create_route(self.evaluation, self.instance, list(r)) for r in routes]
        built += [rb.create_route(self.evaluation, self.instance, []) for _ in range(spare)]
        return rb.Solution(self.evaluation, self.instance, built)

    @staticmethod
    def routes_of(solution: rb.Solution) -> list[tuple]:
        out = []
        for route in solution:
            ids = [node.vertex_id for node in route]
            if len(ids) > 2:  # depot, customers..., depot
                out.append(tuple(ids[1:-1]))
        return out

    def exact_cost(self, routes) -> float:
        if len(routes) > self.fleet:
            return math.inf
        return self.pricer.total(routes)

    # ----------------------------------------------------------- initial plan
    def greedy_routes(self, order=None) -> list[tuple] | None:
        """Cheapest-insertion construction priced exactly; None if a customer fits nowhere."""
        order = list(order if order is not None else self.problem.customers)
        self.py_random.shuffle(order)
        routes: list[list[int]] = []
        for c in order:
            best = None
            for r, route in enumerate(routes):
                base = self.pricer.cost(tuple(route))
                for pos in range(len(route) + 1):
                    cand = tuple(route[:pos] + [c] + route[pos:])
                    value = self.pricer.cost(cand)
                    if math.isfinite(value) and (best is None or value - base < best[0]):
                        best = (value - base, r, pos)
            if len(routes) < self.fleet:
                alone = self.pricer.cost((c,))
                if math.isfinite(alone) and (best is None or alone < best[0]):
                    best = (alone, len(routes), 0)
            if best is None:
                return None
            _, r, pos = best
            if r == len(routes):
                routes.append([c])
            else:
                routes[r].insert(pos, c)
        return [tuple(r) for r in routes]

    def initial_solutions(self, seeds: list[list[tuple]]) -> list[list[tuple]]:
        found = [r for r in seeds if r and self.exact_cost(r) < math.inf]
        for _ in range(3):
            built = self.greedy_routes()
            if built is not None:
                found.append(built)
        return found

    # -------------------------------------------------------------------- state
    def initialize(self, seeds: list[list[tuple]] | None = None) -> None:
        starts = self.initial_solutions(seeds or [])
        if not starts:
            raise RuntimeError("no exactly feasible starting solution found")
        starts.sort(key=self.exact_cost)
        self.best_routes = list(starts[0])
        self.best_cost = self.exact_cost(self.best_routes)
        self.current_routes, self.current_cost = list(self.best_routes), self.best_cost
        for routes in starts:
            self._elite_add(routes)
        self.history = [(self.elapsed, self.best_cost)]
        self.window: list[tuple] = []
        self.progress = 0.0  # fraction of the whole budget used; drives the temperature
        self._note(f"start exact cost {self.best_cost:.4f} with {len(self.best_routes)} routes")

    def set_current(self, routes) -> None:
        """Continue from ``routes`` (e.g. the master's best); adopts them as incumbent if better."""
        routes = [tuple(r) for r in routes]
        cost = self.exact_cost(routes)
        if not math.isfinite(cost):
            return
        self.current_routes, self.current_cost = routes, cost
        self._elite_add(routes)
        if cost < self.best_cost - 1e-9:
            self.best_routes, self.best_cost = list(routes), cost
            self.history.append((self.elapsed, cost))

    def advance(self, seconds: float, progress_at_start: float | None = None, progress_span: float = 0.0) -> None:
        """Run the ALNS for ``seconds`` of wall time."""
        p = self.params
        deadline = time.time() + seconds
        started = time.time()
        if progress_at_start is not None:
            self.progress = progress_at_start
        while time.time() < deadline and self.iterations < p.max_iterations:
            if progress_span > 0:
                self.progress = (progress_at_start or 0.0) + progress_span * (time.time() - started) / max(seconds, 1e-9)
            self.iterations += 1
            fraction = min(1.0, self.progress)
            temperature = self.best_cost / self.n * (p.start_temperature * (1 - fraction) + p.end_temperature * fraction)

            spare = max(0, min(p.spare_routes, self.fleet - len(self.current_routes)))
            solution = self.to_rb(self.current_routes, spare)
            q_hi = min(self.n, p.max_removed, max(p.min_largest_destroy, int(self.n * p.max_removed_fraction)))
            q_lo = min(q_hi, max(1, int(self.n * p.min_removed_fraction)))
            q = self.rb_random.randint(q_lo, q_hi)
            destroy, repair = self.alns.generate(self.evaluation, solution, q)

            repaired = self.routes_of(solution)
            candidates = [repaired]
            if solution.cost < self._surrogate(self.current_routes) * (1 + p.delta_local_search):
                if p.local_search_scope == "changed":
                    # Optimise only the routes destroy/repair touched: the stock search
                    # sweeps every route, which at 200 customers costs ~90 ms per call.
                    kept = set(self.current_routes)
                    changed = [r for r in repaired if r not in kept]
                    untouched = [r for r in repaired if r in kept]
                    sub = self.to_rb(changed, 1 if len(changed) < self.fleet else 0)
                    self.local_search.optimize(sub, self.ls_operators)
                    candidates.append(untouched + self.routes_of(sub))
                else:
                    self.local_search.optimize(solution, self.ls_operators)
                    candidates.append(self.routes_of(solution))
            self.window.append((solution.cost_components[1], solution.cost_components[2],
                                solution.cost_components[3]))

            cand_routes, cand_cost = None, math.inf
            for routes in candidates:
                value = self.exact_cost(routes)
                if value < cand_cost:
                    cand_routes, cand_cost = routes, value

            if cand_routes is not None and cand_cost <= self.best_cost * (1 + p.pool_elite_tolerance):
                self._elite_add(cand_routes)
            score = 0.0
            if cand_cost < self.best_cost - 1e-9:
                self.best_routes, self.best_cost = cand_routes, cand_cost
                score = p.score_new_best
                self.history.append((self.elapsed, self.best_cost))
                self._note(f"it {self.iterations} new best {self.best_cost:.4f} routes {len(self.best_routes)}")
            if cand_cost < math.inf:
                delta = cand_cost - self.current_cost
                if delta < -1e-9:
                    score = max(score, p.score_improved)
                    self.current_routes, self.current_cost = cand_routes, cand_cost
                elif delta <= 1e-9 or (temperature > 0 and self.py_random.random() < math.exp(-delta / temperature)):
                    score = max(score, p.score_accepted)
                    self.current_routes, self.current_cost = cand_routes, cand_cost
            self.alns.collect_score(destroy, repair, score)

            if self.iterations % p.adaptive_period == 0:
                self.alns.adapt_operator_weights()
            if self.iterations % p.penalty_period == 0 and self.window:
                self._adapt_penalties(self.window)
                self.window = []
            if self.iterations % 500 == 0 and self.current_cost > self.best_cost * 1.02:
                self.current_routes, self.current_cost = list(self.best_routes), self.best_cost

    def _elite_add(self, routes) -> None:
        for route in routes:
            route = tuple(route)
            value = self.pricer.cost(route)
            if not math.isfinite(value):
                continue
            key = frozenset(route)
            held = self.elite.get(key)
            if held is None or value < held[0] - 1e-12:
                self.elite[key] = (value, route)
                self._elite_new.add(key)

    def export_pool(self) -> dict[frozenset, tuple[float, tuple]]:
        """Elite columns added or improved since the last export."""
        fresh = {key: self.elite[key] for key in self._elite_new}
        self._elite_new = set()
        return fresh

    # -------------------------------------------------------------------- run
    def run(self, seeds: list[list[tuple]] | None = None) -> dict:
        """Single-process run: ALNS in slices with periodic set partitioning."""
        p = self.params
        self.initialize(seeds)
        reserve = min(p.pool_seconds, 0.15 * p.time_limit_seconds) if p.pool else 0.0
        budget = max(p.time_limit_seconds - reserve, 1.0)
        interval = p.pool_interval_seconds if p.pool else budget
        while self.elapsed < budget and self.iterations < p.max_iterations:
            span = min(interval, budget - self.elapsed)
            self.advance(span, progress_at_start=self.elapsed / budget, progress_span=span / budget)
            if p.pool:
                self._merge_partition()
        if p.pool:
            self._merge_partition()
        return {
            "routes": self.best_routes,
            "cost": self.best_cost,
            "iterations": self.iterations,
            "seconds": self.elapsed,
            "history": self.history,
            "pricer_calls": self.pricer.calls,
            "pricer_hits": self.pricer.hits,
        }

    def _merge_partition(self) -> None:
        merged = self._partition(self.best_routes, self.best_cost)
        if merged is not None:
            self.best_routes, self.best_cost = merged
            self.current_routes, self.current_cost = list(merged[0]), merged[1]
            self.history.append((self.elapsed, self.best_cost))

    def _partition(self, best_routes, best_cost):
        """Set partitioning over the route cache; returns (routes, cost) if it improves."""
        from .pool import solve_partition

        p = self.params
        self._elite_add(best_routes)
        found = solve_partition(self.elite, self.problem.customers, self.fleet, best_routes,
                                max(1.0, min(p.pool_seconds, 0.15 * p.time_limit_seconds)))
        if found is None:
            return None
        routes, cost = found
        self._note(f"partition timing {solve_partition.last_timing}")
        exact = self.exact_cost(routes)  # re-price: the cache is authoritative
        if exact < best_cost - 1e-9:
            self._note(f"set partitioning {best_cost:.4f} -> {exact:.4f} ({len(self.elite)} pool columns)")
            return [tuple(r) for r in routes], exact
        return None

    def _surrogate(self, routes) -> float:
        total = 0.0
        cache = self._surrogate_cache
        for route in routes:
            value = cache.get(route)
            if value is None:
                nodes = (0, *route, 0)
                value = float(sum(self.surrogate_cost[a, b] for a, b in zip(nodes, nodes[1:])))
                if len(cache) > 500_000:
                    cache.clear()
                cache[route] = value
            total += value
        return total

    def _adapt_penalties(self, window) -> None:
        p = self.params
        new = []
        for k in range(3):
            ok = sum(1 for w in window if w[k] <= 0.01) / len(window)
            factor = 1.2 if ok < p.target_feasibility else 0.8
            new.append(max(0.1, min(10000.0, self.penalties[k] * factor)))
        self._set_penalties(new[0], max(new[1], new[2]), max(new[1], new[2]))

    def _note(self, message: str) -> None:
        line = f"[{self.elapsed:7.1f}s] {message}"
        self.log.append(line)
        if self.params.verbose:
            print(line, flush=True)
