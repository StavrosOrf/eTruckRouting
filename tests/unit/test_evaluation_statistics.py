"""Statistical reporting tests that retain failures explicitly."""

import pytest

from EVRoutingEnv.evaluation.statistics import (
    aggregate_episode_metrics,
    paired_bootstrap_difference,
    wilson_interval,
)


def _row(success: bool, makespan: float | None, reason: str | None) -> dict:
    return {
        "success": success,
        "termination_reason": reason,
        "completed_fraction": 1.0 if success else 0.5,
        "fleet_makespan": makespan,
        "total_operating_time": 10.0,
        "total_travel_time": 6.0,
        "total_charging_time": 1.0,
        "total_queue_time": 1.0,
        "total_time_window_waiting": 0.0,
        "total_service_time": 2.0,
        "total_distance": 100.0,
        "total_energy_consumed": 80.0,
        "total_energy_charged": 20.0,
        "mean_terminal_soc": 0.3,
        "minimum_terminal_soc": 0.2,
        "vehicles_used": 2,
        "charging_sessions": 1,
        "invalid_actions": 0,
    }


def test_wilson_interval_handles_boundary_success_counts() -> None:
    none = wilson_interval(0, 10)
    all_success = wilson_interval(10, 10)

    assert none[0] == 0.0
    assert 0.0 < none[1] < 0.5
    assert 0.5 < all_success[0] < 1.0
    assert all_success[1] == 1.0


def test_aggregate_retains_failures_and_labels_success_conditioning() -> None:
    summary = aggregate_episode_metrics(
        [
            _row(True, 8.0, "success"),
            _row(False, None, "battery_depleted"),
            _row(False, None, "battery_depleted"),
            _row(False, None, "time_window_violation"),
        ]
    )

    assert summary["episode_count"] == 4
    assert summary["success_count"] == 1
    assert summary["failure_count"] == 3
    assert summary["failure_causes"] == {
        "battery_depleted": 2,
        "time_window_violation": 1,
    }
    makespan = summary["metrics"]["fleet_makespan"]
    assert makespan["count"] == 1
    assert makespan["conditioning"] == "successful_episodes"
    assert summary["metrics"]["completed_fraction"]["count"] == 4


def test_paired_bootstrap_is_reproducible_and_reports_effect() -> None:
    baseline = [10.0, 12.0, 11.0, 13.0, 9.0]
    candidate = [9.0, 11.0, 10.5, 12.0, 8.5]

    first = paired_bootstrap_difference(
        baseline,
        candidate,
        resamples=1_000,
        seed=44,
    )
    second = paired_bootstrap_difference(
        baseline,
        candidate,
        resamples=1_000,
        seed=44,
    )

    assert first == second
    assert first["mean_difference"] < 0.0
    assert first["ci_high"] < 0.0
    assert first["candidate_win_fraction"] == 1.0
    assert first["paired_effect_size"] is not None


@pytest.mark.parametrize(
    ("baseline", "candidate"),
    [([], []), ([1.0], [1.0, 2.0]), ([float("nan")], [1.0])],
)
def test_paired_bootstrap_rejects_invalid_pairs(baseline, candidate) -> None:
    with pytest.raises(ValueError):
        paired_bootstrap_difference(baseline, candidate, resamples=10)
