"""Charging-time estimation and the piecewise-linear comparison model."""

import os

import pytest


os.environ.setdefault("MPLCONFIGDIR", "/tmp/evrp_matplotlib")

from EVRoutingEnv.models.simulation.charging_curve import ChargingCurveModel


CONFIG = {
    "charge_rate": 750.0,
    "efficiency": 0.9,
    "use_realistic_curve": True,
    "taper_start_soc": 0.8,
    "taper_power_min": 150.0,
}
CAPACITY = 400.0


@pytest.mark.parametrize("target", [0.6, 0.8, 0.9, 0.99, 1.0])
def test_estimated_time_matches_the_integrator_including_a_full_charge(target) -> None:
    """A full charge is the case the old bisection could not solve.

    It returned the midpoint of its search range -- 10 hours for a charge that
    takes about half an hour -- because the integrator stops exactly at full and
    the search never saw an overshoot.
    """
    model = ChargingCurveModel()
    estimate = model.estimate_charge_time(0.2, target, CAPACITY, CONFIG)
    _, details = model.calculate_charge_to_target(
        initial_soc=0.2,
        target_soc=target,
        battery_capacity=CAPACITY,
        charger_config=CONFIG,
    )
    assert estimate == pytest.approx(details["actual_charge_hours"], rel=1e-9)
    assert estimate < 1.0


def test_no_charge_needed_when_the_target_is_already_reached() -> None:
    model = ChargingCurveModel()
    assert model.estimate_charge_time(0.8, 0.8, CAPACITY, CONFIG) == 0.0
    assert model.estimate_charge_time(0.9, 0.5, CAPACITY, CONFIG) == 0.0


def test_montoya_breakpoints_interpolate_the_curve_exactly_at_boundaries() -> None:
    model = ChargingCurveModel()
    breakpoints = model.montoya_breakpoints(
        battery_capacity=CAPACITY,
        peak_power=CONFIG["charge_rate"],
        efficiency=CONFIG["efficiency"],
        taper_start_soc=CONFIG["taper_start_soc"],
        taper_power_min=CONFIG["taper_power_min"],
    )
    assert breakpoints[0] == (0.0, 0.0)
    assert [soc for _, soc in breakpoints] == [0.0, 0.5, 0.8, 1.0]
    times = [time for time, _ in breakpoints]
    assert times == sorted(times)

    for time, soc in breakpoints[1:]:
        assert model.montoya_time_to_soc(breakpoints, 0.0, soc) == pytest.approx(time)


def test_this_curve_is_not_concave_which_montoya_style_models_assume() -> None:
    """A documented incompatibility, not a defect.

    Montoya et al. approximate charging with a *concave* piecewise-linear
    function, which is what lets the routing MILP stay tight. This simulator's
    curve ramps from 60% of peak power up to peak before tapering, so charge
    accrues faster in the middle than at the start: the function has an inflection
    and is not concave. Any concave approximation must therefore either lose the
    ramp or misprice the taper, which is why the comparison in
    scripts/analysis/compare_charging_models.py reports error rather than
    claiming equivalence.
    """
    model = ChargingCurveModel()
    breakpoints = model.montoya_breakpoints(
        battery_capacity=CAPACITY,
        peak_power=CONFIG["charge_rate"],
        efficiency=CONFIG["efficiency"],
        taper_start_soc=CONFIG["taper_start_soc"],
        taper_power_min=CONFIG["taper_power_min"],
    )
    slopes = [
        (breakpoints[index + 1][1] - breakpoints[index][1])
        / (breakpoints[index + 1][0] - breakpoints[index][0])
        for index in range(len(breakpoints) - 1)
    ]
    # Ramp, then plateau, then taper: the middle segment is the fastest.
    assert slopes[1] > slopes[0]
    assert slopes[1] > slopes[2]


def test_montoya_needs_breakpoints_in_both_curved_regions() -> None:
    """The three-segment form straddles the ramp and the taper.

    This is the comparison the review asked for: the classical piecewise-linear
    formulation is barely better than assuming constant power, and only becomes
    accurate once its breakpoints sit inside both curved regions.
    """
    model = ChargingCurveModel()

    def worst_error(boundaries) -> float:
        points = model.montoya_breakpoints(
            battery_capacity=CAPACITY,
            peak_power=CONFIG["charge_rate"],
            efficiency=CONFIG["efficiency"],
            taper_start_soc=CONFIG["taper_start_soc"],
            taper_power_min=CONFIG["taper_power_min"],
            boundaries=boundaries,
        )
        worst = 0.0
        for start in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            for target in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
                if target <= start:
                    continue
                _, details = model.calculate_charge_to_target(
                    initial_soc=start,
                    target_soc=target,
                    battery_capacity=CAPACITY,
                    charger_config=CONFIG,
                )
                exact = details["actual_charge_hours"]
                approximate = model.montoya_time_to_soc(points, start, target)
                worst = max(worst, abs(approximate - exact) / exact)
        return worst

    three_segment = worst_error((0.5, 0.8, 1.0))
    refined = worst_error((0.1, 0.5, 0.8, 0.9, 1.0))
    assert three_segment > 0.2
    assert refined < 0.05


def test_boundaries_must_be_increasing() -> None:
    model = ChargingCurveModel()
    with pytest.raises(ValueError, match="increasing"):
        model.montoya_breakpoints(
            battery_capacity=CAPACITY,
            peak_power=CONFIG["charge_rate"],
            efficiency=CONFIG["efficiency"],
            taper_start_soc=CONFIG["taper_start_soc"],
            taper_power_min=CONFIG["taper_power_min"],
            boundaries=(0.8, 0.5, 1.0),
        )
