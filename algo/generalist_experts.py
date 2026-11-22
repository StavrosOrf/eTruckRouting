"""Planner experts of the generalist suite: DAgger labels and BKS controllers.

Factories for ``--dagger-expert PROBLEM=algo.generalist_experts:<name>``
(:func:`algo.generalist_imitation.resolve_expert`). Each is called as
``factory(problem, env)`` after every episode reset and returns a controller
with ``act(env) -> environment action``; the rolling planner is a plain
callable, which ``resolve_expert`` wraps. Both re-plan from whatever state they
are handed, as DAgger requires when the learner chose the earlier actions:

* :func:`longhaul_plan` (static long-haul problems): the long-haul ALNS plan
  executed through the continuation search, re-planned on events and on any
  deviation from the plan (``LonghaulPlanPolicy._reconcile``). The planner runs
  at a reduced budget (:data:`DAGGER_PLANNER`); the best-known solutions keep
  the full one (``scripts/campaigns/generalist_bks_static.py``).
* :func:`longhaul_cached` (static long-haul problems): the same controller at
  the best-known budget with its first plan read from the plan cache: the
  teacher of the planner-free policy (``suite_v8``).
* :func:`rolling_request` (request problems): the rolling-horizon request
  planner, which rebuilds its snapshot at every decision. The same
  :data:`ROLLING` parameters build the ``rolling_request`` method of the
  dynamic best-known solutions, so the DAgger expert and the BKS method are
  one controller; ``clairvoyant=True`` turns it into the hindsight bound.

Imports stay lazy and nothing here touches torch: the factories run inside the
forked rollout workers, and ``check_experts`` imports this module in the parent.
"""

from __future__ import annotations

from dataclasses import replace

from EVRoutingEnv.baselines.heuristics.rolling_request_planner import PlannerParameters
from EVRoutingEnv.baselines.optimization.longhaul.planner import PlannerSettings


# Reduced planning budget for DAgger labels; the BKS planner runs 40 s x 3 restarts.
DAGGER_PLANNER = PlannerSettings(iterations=1500, time_limit_seconds=8.0)
DAGGER_RESTARTS = 1
# The rolling request planner of the dynamic BKS and of the DAgger expert.
ROLLING = PlannerParameters()


def is_dynamic(env) -> bool:
    """Whether ``env`` runs a dynamic request mode."""
    return bool(getattr(env, "dynamic_requests", False)) and getattr(env, "request_engine", None) is not None


def _name(problem) -> str:
    return str(getattr(problem, "name", problem))


def longhaul_plan(problem, env):
    """A fresh long-haul plan-and-continue controller for a static problem."""
    from EVRoutingEnv.baselines.optimization.longhaul.policy import LonghaulPlanPolicy, LonghaulSettings

    if is_dynamic(env):
        raise ValueError(f"{_name(problem)}: longhaul_plan needs a static problem")
    return LonghaulPlanPolicy(LonghaulSettings(
        name="dagger_lh", planner=DAGGER_PLANNER, restarts=DAGGER_RESTARTS, replan="events",
    ))


def longhaul_cached(problem, env):
    """The long-haul planner at the best-known budget, its time-0 plan read from the plan cache.

    The controller the plan-conditioned policy reads (``algo.generalist_plans.PlanProvider``): the cached
    full-budget plan of the episode's seed (``EVRP_PLAN_CACHE``, set by the trainer's ``--expert-plan-cache``),
    re-planned with a few hundred iterations on any deviation. As a DAgger expert it is a better teacher than
    :func:`longhaul_plan` (whose reduced first plan alone costs 0.1 to 3 % of operating hours against the full
    planner, measured on train seeds on 2026-11-21) and a cheaper one, because the first plan is read, not
    searched. A seed without a cached plan is planned at the full budget.
    """
    from algo.generalist_plans import plan_cache, static_settings
    from EVRoutingEnv.baselines.optimization.longhaul.policy import LonghaulPlanPolicy

    if is_dynamic(env):
        raise ValueError(f"{_name(problem)}: longhaul_cached needs a static problem")
    return LonghaulPlanPolicy(static_settings(), plan_cache=plan_cache(_name(problem)))


def rolling_request(problem, env, *, clairvoyant: bool = False):
    """A fresh rolling-horizon request planner for a dynamic problem."""
    from EVRoutingEnv.baselines.heuristics.rolling_request_planner import RollingHorizonRequestPolicy

    if not is_dynamic(env):
        raise ValueError(f"{_name(problem)}: rolling_request needs a dynamic request mode")
    return RollingHorizonRequestPolicy(replace(ROLLING, clairvoyant=clairvoyant))


def for_problem(problem, env):
    """The expert matching the environment's kind."""
    return rolling_request(problem, env) if is_dynamic(env) else longhaul_plan(problem, env)
