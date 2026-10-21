"""The Bai (2023) dynamic program equals exhaustive enumeration."""

import random

import pytest

from EVRoutingEnv.baselines.optimization.bai2023_dp import (
    FixedRoute,
    brute_force,
    linear_charge_hours,
    solve,
)


def test_dynamic_program_matches_enumeration() -> None:
    rng = random.Random(0)
    levels = (0.5, 0.7, 0.9, 1.0)
    charge = linear_charge_hours(150.0, 400.0)
    feasible = 0
    for _ in range(300):
        stops = rng.randint(1, 5)
        route = FixedRoute(
            tuple(rng.uniform(0.3, 2.6) for _ in range(stops)),
            tuple(rng.uniform(0.1, 0.55) for _ in range(stops)),
        )
        plan, exhaustive = solve(route, levels, charge), brute_force(route, levels, charge)
        assert (plan is None) == (exhaustive is None)
        if plan is not None:
            feasible += 1
            assert plan.total_hours == pytest.approx(exhaustive, abs=1e-9)
    assert feasible > 100


def test_rest_is_taken_only_where_driving_would_exceed_four_and_a_half_hours() -> None:
    route = FixedRoute((2.5, 2.5, 1.0), (0.2, 0.2, 0.1))
    plan = solve(route, (1.0,), linear_charge_hours(150.0, 400.0))
    assert plan.rest == (True, False)
    assert plan.total_hours == pytest.approx(6.0 + 0.75)


def test_trip_longer_than_nine_hours_has_no_plan() -> None:
    route = FixedRoute((4.0, 4.0, 1.5), (0.1, 0.1, 0.1))
    assert solve(route, (1.0,), linear_charge_hours(150.0, 400.0)) is None
