"""Failure-retaining campaign aggregation and paired uncertainty intervals."""

from __future__ import annotations

import math
from collections import Counter
from statistics import NormalDist

import numpy as np


def wilson_interval(
    successes: int,
    total: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Wilson score interval for a Bernoulli success probability."""
    if (
        not isinstance(successes, int)
        or isinstance(successes, bool)
        or not isinstance(total, int)
        or isinstance(total, bool)
        or total <= 0
        or not 0 <= successes <= total
    ):
        raise ValueError("require integer counts satisfying 0 <= successes <= total")
    if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    z_value = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    proportion = successes / total
    denominator = 1.0 + z_value**2 / total
    center = (proportion + z_value**2 / (2.0 * total)) / denominator
    half_width = (
        z_value
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z_value**2 / (4.0 * total**2)
        )
        / denominator
    )
    low = 0.0 if successes == 0 else max(0.0, center - half_width)
    high = 1.0 if successes == total else min(1.0, center + half_width)
    return low, high


def aggregate_episode_metrics(rows: list[dict]) -> dict:
    """Aggregate every episode, with explicit conditioning and missing counts."""
    if not rows:
        raise ValueError("at least one episode row is required")
    successes = [bool(row.get("success", False)) for row in rows]
    success_count = sum(successes)
    failure_causes = Counter(
        str(row.get("termination_reason") or "unspecified_failure")
        for row, success in zip(rows, successes, strict=True)
        if not success
    )
    metric_names = (
        "completed_fraction",
        "fleet_makespan",
        "total_operating_time",
        "total_travel_time",
        "total_charging_time",
        "total_queue_time",
        "total_time_window_waiting",
        "total_service_time",
        "total_distance",
        "total_energy_consumed",
        "total_energy_charged",
        "mean_terminal_soc",
        "minimum_terminal_soc",
        "vehicles_used",
        "charging_sessions",
        "invalid_actions",
    )
    metrics = {}
    for name in metric_names:
        eligible_rows = (
            [row for row, success in zip(rows, successes, strict=True) if success]
            if name == "fleet_makespan"
            else rows
        )
        values = [
            float(row[name])
            for row in eligible_rows
            if row.get(name) is not None and math.isfinite(float(row[name]))
        ]
        metrics[name] = _continuous_summary(values, len(eligible_rows))
        metrics[name]["conditioning"] = (
            "successful_episodes" if name == "fleet_makespan" else "all_episodes"
        )

    interval = wilson_interval(success_count, len(rows))
    return {
        "episode_count": len(rows),
        "success_count": success_count,
        "failure_count": len(rows) - success_count,
        "success_rate": success_count / len(rows),
        "success_wilson_95": {"low": interval[0], "high": interval[1]},
        "failure_causes": dict(sorted(failure_causes.items())),
        "metrics": metrics,
    }


def paired_bootstrap_difference(
    baseline: list[float] | np.ndarray,
    candidate: list[float] | np.ndarray,
    *,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 0,
    lower_is_better: bool = True,
) -> dict:
    """Paired bootstrap CI and paired standardized effect for candidate-baseline."""
    baseline_values = np.asarray(baseline, dtype=float)
    candidate_values = np.asarray(candidate, dtype=float)
    if (
        baseline_values.ndim != 1
        or candidate_values.ndim != 1
        or len(baseline_values) == 0
        or baseline_values.shape != candidate_values.shape
    ):
        raise ValueError("paired samples must be non-empty one-dimensional arrays")
    if not np.isfinite(baseline_values).all() or not np.isfinite(
        candidate_values
    ).all():
        raise ValueError("paired samples must be finite")
    if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if (
        not isinstance(resamples, int)
        or isinstance(resamples, bool)
        or resamples <= 0
    ):
        raise ValueError("resamples must be a positive integer")
    if not isinstance(lower_is_better, bool):
        raise TypeError("lower_is_better must be boolean")

    differences = candidate_values - baseline_values
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(resamples, dtype=float)
    for index in range(resamples):
        selection = rng.integers(0, len(differences), size=len(differences))
        bootstrap_means[index] = differences[selection].mean()
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(bootstrap_means, (alpha, 1.0 - alpha))
    standard_deviation = (
        float(differences.std(ddof=1)) if len(differences) > 1 else 0.0
    )
    return {
        "pair_count": len(differences),
        "mean_difference": float(differences.mean()),
        "median_difference": float(np.median(differences)),
        "confidence": confidence,
        "ci_low": float(low),
        "ci_high": float(high),
        "paired_effect_size": (
            float(differences.mean() / standard_deviation)
            if standard_deviation > 0.0
            else None
        ),
        "lower_is_better": lower_is_better,
        "candidate_win_fraction": float(
            np.mean(differences < 0.0)
            if lower_is_better
            else np.mean(differences > 0.0)
        ),
        "tie_fraction": float(np.mean(differences == 0.0)),
    }


def _continuous_summary(values: list[float], eligible_count: int) -> dict:
    if not values:
        return {
            "count": 0,
            "missing_count": eligible_count,
            "mean": None,
            "median": None,
            "p90": None,
        }
    array = np.asarray(values, dtype=float)
    return {
        "count": len(values),
        "missing_count": eligible_count - len(values),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
    }
