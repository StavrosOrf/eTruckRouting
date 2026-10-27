"""Planner plans for the plan-conditioned generalist policy (design 4.4 / 9.5, `suite_v2`).

A plan is, per truck, the ordered list of task slots it is planned to serve
next (route-level: assignment and order; charging stays the policy's own
decision). :class:`PlanProvider` keeps one planner per episode and reads its
current plan at every decision:

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

    def routes(self, env) -> dict[int, list[int]]:
        if self.dynamic:
            return {int(tid): [int(r) for r in route] for tid, route in self.planner.planned_routes(env).items()}
        planned = self.planner.planned_routes(env)
        return {int(tid): [self.slot_of_node[int(node)] for node in nodes] for tid, nodes in planned.items()}


def problem_name(config) -> str | None:
    """The suite problem of a config path (``.../<problem>.yaml``); None for an in-memory config."""
    if isinstance(config, (str, Path)):
        return Path(config).stem
    return None
