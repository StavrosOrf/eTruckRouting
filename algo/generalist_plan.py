"""Rolling-planner plan cache for plan-conditioned observations (plan 9.5).

The rolling request planner's live plan is advisory input to the generalist
policy on the dynamic problems: each truck's next planned request becomes an
observation feature (``generalist_observation.build_observation``), and the
policy keeps the right to deviate. One :class:`PlanTracker` lives on each
environment (:func:`tracker_for`); it re-plans only when a request has been
revealed since the last plan or when the active truck no longer follows its
planned next request, and reuses the cached routes in between. Per-episode
plan dropout zeroes the features for whole episodes so the plan-free mode
keeps working (``ObservationSpec.plan_dropout``).

Imports stay lazy: the tracker runs inside forked rollout workers and must not
pull torch; the planner modules are imported on first re-plan only.
"""

from __future__ import annotations

import random
import time


class PlanTracker:
    """Cached fleet plan of the rolling request planner for one environment."""

    def __init__(self) -> None:
        self._episode_key = None
        self._last_steps = 0
        self._routes: dict[int, tuple[int, ...]] = {}
        self._warm: dict = {}
        self._released = -1
        self._model = None
        self._parameters = None
        self._dropped = False
        self.plan_calls = 0
        self.plan_seconds = 0.0

    # -- episode bookkeeping ---------------------------------------------------

    def _begin(self, env, dropout: float) -> None:
        key = (int(env.scenario_seed), id(env))
        steps = int(env.episode_steps)
        if key != self._episode_key or steps < self._last_steps:
            self._episode_key = key
            self._routes, self._warm = {}, {}
            self._released = -1
            self._model = None
            self._dropped = dropout > 0.0 and random.random() < dropout
            self.plan_calls = 0
            self.plan_seconds = 0.0
        self._last_steps = steps

    # -- the cached plan ---------------------------------------------------------

    def routes_for(self, env, *, dropout: float, open_rids, active_id) -> dict[int, tuple[int, ...]]:
        """The fleet plan as ``{truck_id: (request ids, ...)}``; ``{}`` for dropped episodes.

        ``open_rids`` are the request ids still live in the observation and
        ``active_id`` is the deciding truck; a new reveal or the active truck
        exhausting its planned assignments forces a re-plan.
        """
        self._begin(env, dropout)
        if self._dropped:
            return {}
        released = int(env.request_engine.metrics(env.global_clock)["requests_released"])
        if self._released != released or self._deviated(env, open_rids, active_id):
            self._replan(env)
            self._released = released
        return self._routes

    def _next_open(self, truck_id: int, open_rids) -> int | None:
        for rid in self._routes.get(int(truck_id), ()):
            if rid in open_rids:
                return int(rid)
        return None

    def _deviated(self, env, open_rids, active_id) -> bool:
        # A route mismatch while merely en route (to a charger, say) is not a
        # deviation; re-planning then would thrash every step. The plan is
        # advisory, and reveals and exhausted assignments bound its staleness.
        if not self._routes:
            return True
        if active_id is None:
            return False
        planned = self._next_open(int(active_id), open_rids)
        # Everything the plan assigned to this truck is served or lost: re-plan.
        return planned is None and bool(self._routes.get(int(active_id), ()))

    def _replan(self, env) -> None:
        from EVRoutingEnv.baselines.heuristics.rolling_request_planner import (
            DrayageModel, FleetPlanner, SameDayModel, build_snapshot,
        )

        from algo.generalist_experts import ROLLING

        started = time.perf_counter()
        if self._model is None:
            self._parameters = ROLLING
            engine = env.request_engine
            model_type = DrayageModel if engine.is_drayage else SameDayModel
            self._model = model_type(env, self._parameters)
        snapshot = build_snapshot(env, self._model, False)
        rng = random.Random(hash((self._parameters.seed, int(env.scenario_seed), self.plan_calls)) & 0xFFFFFFFF)
        planner = FleetPlanner(self._model, self._parameters)
        free = len(snapshot.jobs)
        iterations = max(self._parameters.min_iterations, min(self._parameters.iterations, 4 * free))
        routes, _ = planner.plan(snapshot, self._warm, rng, iterations)
        self._warm = routes
        self._routes = {int(truck): _request_order(route) for truck, route in routes.items()}
        self.plan_calls += 1
        self.plan_seconds += time.perf_counter() - started


def _request_order(route) -> tuple[int, ...]:
    """Request ids in planned order: same-day routes nest trips of ids, drayage
    routes are flat and may hold the REST token (-1); both flatten to ids."""
    order: list[int] = []
    for item in route:
        if isinstance(item, (tuple, list)):
            order.extend(int(rid) for rid in item if int(rid) >= 0)
        elif int(item) >= 0:
            order.append(int(item))
    return tuple(order)


def tracker_for(env) -> PlanTracker:
    """The environment's tracker, created on first use."""
    tracker = getattr(env, "_generalist_plan_tracker", None)
    if tracker is None:
        tracker = PlanTracker()
        env._generalist_plan_tracker = tracker
    return tracker
