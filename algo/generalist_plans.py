"""Planner plans for the plan-conditioned generalist policy (design 4.4 / 9.5, `suite_v2`).

A plan is, per truck, the ordered list of task slots it is planned to serve
next (route-level: assignment and order; charging stays the policy's own
decision, unless the charging plan of ``suite_v4`` is on: see
:meth:`PlanProvider.moves`). :class:`PlanProvider` keeps one planner per
episode and reads its current plan at every decision:

* static long-haul problems: the BKS long-haul planner (``LonghaulPlanPolicy``
  at the BKS budget, 8,000 iterations / 40 s / 3 restarts). Its time-0 plan is
  read from a precomputed per-seed cache (``<cache>/<problem>/plans_<split>.json``,
  written by ``generalist_bks_static.py plans``) and re-planned cheaply on
  deviations and service events during the episode, as ``lh_rolling`` does;
* dynamic request problems: the rolling-horizon request planner of the BKS
  (``algo.generalist_experts.ROLLING``), which re-optimizes every revealed
  request at each decision and never sees hidden requests.

Settings come from the environment so that forked rollout workers inherit
them without any plumbing:

* ``EVRP_PLAN_CACHE``: the directory of per-problem plan caches (static);
* ``EVRP_PLAN_DROPOUT``: probability that an episode on a **train-namespace**
  seed is played without its plan (training only: validation and test seeds
  always keep the plan, so in-training validation, BC on train demonstrations
  and DAgger follow the training distribution while evaluation is plan-on);
* ``EVRP_PLAN_MODE``: ``on`` (default) or ``off``; ``off`` withholds the plan
  everywhere, for the plan-free evaluation.

Nothing here touches torch: providers run inside the forked rollout workers.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

TRAIN_NAMESPACE = 1_000_000_000  # seeds below this are train seeds
_CACHES: dict[str, dict[str, dict]] = {}


def static_settings():
    """The BKS long-haul planner, re-planning on events (``lh_rolling``); its tag matches ``lh_plan``'s."""
    from EVRoutingEnv.baselines.optimization.longhaul.planner import PlannerSettings
    from EVRoutingEnv.baselines.optimization.longhaul.policy import LonghaulSettings

    return LonghaulSettings(
        name="plan_provider", planner=PlannerSettings(iterations=8000, time_limit_seconds=40.0),
        restarts=3, replan="events",
    )


def plan_dropout() -> float:
    value = float(os.environ.get("EVRP_PLAN_DROPOUT", "0") or 0.0)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"EVRP_PLAN_DROPOUT must lie in [0, 1], not {value}")
    return value


def plan_mode() -> str:
    mode = os.environ.get("EVRP_PLAN_MODE", "on") or "on"
    if mode not in ("on", "off"):
        raise ValueError(f"EVRP_PLAN_MODE must be 'on' or 'off', not {mode!r}")
    return mode


def plan_cache(problem: str) -> dict[str, dict]:
    """Seed -> time-0 plan record for a static problem (all splits merged); empty without a cache."""
    directory = os.environ.get("EVRP_PLAN_CACHE")
    key = f"{directory}|{problem}"
    if key not in _CACHES:
        records: dict[str, dict] = {}
        if directory:
            for path in sorted((Path(directory) / problem).glob("plans_*.json")):
                records.update(json.loads(path.read_text()))
        _CACHES[key] = records
    return _CACHES[key]


def plan_withheld(seed: int, episode: int) -> bool:
    """Whether this episode is played without its plan (mode off, or dropout on a train seed)."""
    if plan_mode() == "off":
        return True
    probability = plan_dropout()
    if probability <= 0.0 or not 0 <= int(seed) < TRAIN_NAMESPACE:
        return False
    rng = np.random.default_rng([int(seed) % (2**32), int(episode) % (2**32), 0x504C414E])
    return bool(rng.random() < probability)


class PlanProvider:
    """One episode's planner; :meth:`routes` gives each truck's planned task slots, next first."""

    def __init__(self, problem: str, env):
        self.problem = problem
        self.dynamic = bool(getattr(env, "dynamic_requests", False)) and getattr(env, "request_engine", None) is not None
        if self.dynamic:
            from algo.generalist_experts import ROLLING
            from EVRoutingEnv.baselines.heuristics.rolling_request_planner import RollingHorizonRequestPolicy

            self.planner = RollingHorizonRequestPolicy(ROLLING)
        else:
            from EVRoutingEnv.baselines.optimization.longhaul.policy import LonghaulPlanPolicy

            self.planner = LonghaulPlanPolicy(static_settings(), plan_cache=plan_cache(problem))
            self.slot_of_node = {int(task.node_id): slot for slot, task in enumerate(env.joint_instance.tasks)}
        # Charging plan (suite_v4): the active truck's moves, solved once per decision.
        self._solved_at: tuple[int, int] | None = None
        self._solved: list[int] = []

    def routes(self, env) -> dict[int, list[int]]:
        if self.dynamic:
            return {int(tid): [int(r) for r in route] for tid, route in self.planner.planned_routes(env).items()}
        planned = self.planner.planned_routes(env)
        return {int(tid): [self.slot_of_node[int(node)] for node in nodes] for tid, nodes in planned.items()}

    def moves(self, env) -> list[tuple[int, float | None]] | None:
        """The active truck's planned moves to its next stop, next first: ``(action, level)`` (suite_v4).

        Static problems only (None for the rolling planner). The moves are the
        long-haul planner's continuation search along the truck's planned
        customers (waypoint stations, target SoCs, rests: what the route plan
        leaves open), solved from the truck's state at each of its decisions,
        as the planner itself does (user decision 2026-10-28: re-solving only
        when the plan was redone left stale charge levels that strand trucks).
        The list ends with the move to the next customer or the depot;
        ``level`` is the planned target SoC after a move to a station, else
        None. Call after :meth:`routes` at the same decision.
        """
        if self.dynamic or env.active_truck_id is None:
            return None
        planner = self.planner
        truck_id = int(env.active_truck_id)
        if self._solved_at != (truck_id, int(env.episode_steps)):
            self._solved = self._solve_moves(env, env.trucks[truck_id])
            self._solved_at = (truck_id, int(env.episode_steps))
        moves, charge = self._solved, planner.charge_offset
        leg: list[tuple[int, float | None]] = []
        for position, action in enumerate(moves):
            level = None
            following = moves[position + 1] if position + 1 < len(moves) else -1
            if action < planner.n_chargers and charge <= following < charge + len(planner.levels):
                level = planner.levels[following - charge]
            leg.append((action, level))
            if planner.n_chargers <= action <= planner.depot_action:
                break
        return leg

    def _solve_moves(self, env, truck) -> list[int]:
        """The planner's continuation search from the truck's state, as environment actions.

        The same searches as ``LonghaulPlanPolicy.__call__``: at a public queue
        it weighs joining (then charging here) against leaving; at the head of
        a queue it charges; elsewhere it may charge here only when it can.
        """
        from EVRoutingEnv.state.feasibility import ActionKind, joint_action_feasibility

        planner = self.planner
        here = planner.net.position[int(truck.current_node)]
        nodes = planner.remaining(env, truck)
        if not nodes and here == planner.depot:
            return [planner.depot_action]
        decisions = env.last_action_feasibility
        if decisions is None or len(decisions) != env.action_space.n:
            decisions = joint_action_feasibility(env)
        feasible = [decision.feasible for decision in decisions]
        charge_ok = any(feasible[planner.charge_offset + i] for i in range(len(planner.levels)))
        queue = planner.extra_index.get(ActionKind.QUEUE)
        queue_ok = queue is not None and feasible[queue]
        navigation_ok = any(feasible[:planner.depot_action + 1])
        unavailable = frozenset(
            planner.net.position[int(node)]
            for node, available in env.charging_station.station_available.items()
            if not available and int(node) in planner.net.position
        )
        hos = getattr(env, "hos", None)
        clock = hos.clock(int(truck.truck_id)) if hos is not None else None
        state = (here, float(truck.current_battery), float(env.global_clock), clock, planner._stops(nodes))
        head: list[int] = []
        if queue_ok:
            stay = planner.search.solve(*state, forced_charge=True, start_wait=planner._observed_wait(env, truck),
                                        unavailable=unavailable)
            leave = planner.search.solve(*state, can_charge_here=False, unavailable=unavailable)
            if stay is not None and (leave is None or stay.end_time <= leave.end_time):
                head, result = [int(queue)], stay
            else:
                result = leave
        elif charge_ok and not navigation_ok:
            result = planner.search.solve(*state, forced_charge=True, start_wait=0.0, unavailable=unavailable)
        else:
            result = planner.search.solve(*state, can_charge_here=charge_ok, start_wait=0.0 if charge_ok else None,
                                          unavailable=unavailable)
        moves = head
        for move in result.actions if result is not None else []:
            action = planner._action_index(env, move)
            if action is None:
                break
            moves.append(int(action))
        return moves


def problem_name(config) -> str | None:
    """The suite problem of a config path (``.../<problem>.yaml``); None for an in-memory config."""
    if isinstance(config, (str, Path)):
        return Path(config).stem
    return None
