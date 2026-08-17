"""Shared, split-aware evaluation helpers for the canonical campaign.

Every method -- learned or not -- is scored through the same episode loop and
the same operational metrics, on seeds drawn from the disjoint namespaces in
:class:`CampaignSeedPlan`.  Tuning and architecture selection may only ever read
the validation split; the test split is reserved for the final campaign.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from EVRoutingEnv.evaluation.artifacts import CampaignSeedPlan
from EVRoutingEnv.evaluation.metrics import extract_operational_metrics


@dataclass(frozen=True)
class EpisodeOutcome:
    """One evaluated episode, retained whether or not it succeeded."""

    scenario_seed: int
    success: bool
    completed_fraction: float
    fleet_makespan: float | None
    total_travel_time: float
    total_distance: float
    total_operating_time: float
    total_energy_charged: float
    charging_sessions: int
    termination_reason: str | None
    policy_calls: int
    truncated: bool

    def as_dict(self) -> dict:
        return {
            "scenario_seed": self.scenario_seed,
            "success": self.success,
            "completed_fraction": self.completed_fraction,
            "fleet_makespan": self.fleet_makespan,
            "total_travel_time": self.total_travel_time,
            "total_distance": self.total_distance,
            "total_operating_time": self.total_operating_time,
            "total_energy_charged": self.total_energy_charged,
            "charging_sessions": self.charging_sessions,
            "termination_reason": self.termination_reason,
            "policy_calls": self.policy_calls,
            "truncated": self.truncated,
        }


def split_seeds(split: str, count: int, base_seed: int = 0) -> list[int]:
    """Return ``count`` seeds from the requested disjoint namespace."""
    plan = CampaignSeedPlan(base_seed=base_seed)
    return [plan.seed(split, index) for index in range(count)]


def evaluate_policy(
    env,
    policy: Callable,
    seeds: Iterable[int],
    max_steps: int = 5_000,
) -> list[EpisodeOutcome]:
    """Run one policy over the given scenarios on an already-built environment.

    Failures are recorded, never discarded: a policy that refuses to act still
    produces a row, so aggregates cannot be inflated by dropping hard scenarios.
    """
    outcomes: list[EpisodeOutcome] = []
    for seed in seeds:
        observation, info = env.reset(seed=int(seed))
        terminated = truncated = False
        calls = 0
        while not (terminated or truncated) and calls < max_steps:
            try:
                action = policy(env, observation, info)
            except (RuntimeError, ValueError):
                break
            observation, _, terminated, truncated, info = env.step(action)
            calls += 1
        metrics = extract_operational_metrics(env).as_dict()
        outcomes.append(
            EpisodeOutcome(
                scenario_seed=int(seed),
                success=bool(metrics["success"]),
                completed_fraction=float(metrics["completed_fraction"]),
                fleet_makespan=metrics["fleet_makespan"],
                total_travel_time=float(metrics["total_travel_time"]),
                total_distance=float(metrics["total_distance"]),
                total_operating_time=float(metrics["total_operating_time"]),
                total_energy_charged=float(metrics["total_energy_charged"]),
                charging_sessions=int(metrics["charging_sessions"]),
                termination_reason=info.get("termination_reason"),
                policy_calls=calls,
                truncated=bool(truncated),
            )
        )
    return outcomes


def summarize(outcomes: Sequence[EpisodeOutcome]) -> dict:
    """Feasibility-first aggregate that keeps failed episodes in the sample."""
    if not outcomes:
        raise ValueError("cannot summarize an empty set of outcomes")
    successes = [outcome for outcome in outcomes if outcome.success]
    makespans = [
        outcome.fleet_makespan
        for outcome in successes
        if outcome.fleet_makespan is not None
    ]
    return {
        "episodes": len(outcomes),
        "success_rate": float(np.mean([outcome.success for outcome in outcomes])),
        "mean_completed_fraction": float(
            np.mean([outcome.completed_fraction for outcome in outcomes])
        ),
        "mean_makespan_successful": float(np.mean(makespans)) if makespans else None,
        # The campaign objective: hours driven summed over the fleet, reported
        # only over plans that actually delivered everything, since travel time
        # on an abandoned route measures nothing.
        "mean_travel_time_successful": (
            float(np.mean([outcome.total_travel_time for outcome in successes]))
            if successes
            else None
        ),
        "mean_distance_successful": (
            float(np.mean([outcome.total_distance for outcome in successes]))
            if successes
            else None
        ),
        "mean_charging_sessions_successful": (
            float(np.mean([outcome.charging_sessions for outcome in successes]))
            if successes
            else None
        ),
        "mean_operating_time": float(
            np.mean([outcome.total_operating_time for outcome in outcomes])
        ),
        "mean_energy_charged": float(
            np.mean([outcome.total_energy_charged for outcome in outcomes])
        ),
        "mean_policy_calls": float(
            np.mean([outcome.policy_calls for outcome in outcomes])
        ),
        "truncation_rate": float(np.mean([outcome.truncated for outcome in outcomes])),
    }


OBJECTIVE_KEYS = {
    "travel_time": "mean_travel_time_successful",
    "makespan": "mean_makespan_successful",
    "operating_time": "mean_operating_time",
    "distance": "mean_distance_successful",
}


def selection_score(
    summary: dict, objective: str = "travel_time"
) -> tuple[float, float]:
    """Rank by feasibility first, then by the campaign objective. Lower is better.

    Success rate dominates because a plan that abandons deliveries is worthless
    however fast it looks; the objective -- by default fleet travel time -- only
    ever breaks ties between policies that complete the same share of routes.

    Conditioning the objective on a policy's own successes means a policy that
    solves strictly more scenarios is charged for the harder ones it took on.
    That bias is accepted here because feasibility already dominates the key;
    like-for-like speed claims come from the paired jointly-solved comparison in
    ``compare_campaign.py``, never from this ranking.
    """
    if objective not in OBJECTIVE_KEYS:
        raise ValueError(
            f"unknown objective {objective!r}; expected one of {sorted(OBJECTIVE_KEYS)}"
        )
    value = summary.get(OBJECTIVE_KEYS[objective])
    return (
        -float(summary["success_rate"]),
        float(value) if value is not None else float("inf"),
    )
