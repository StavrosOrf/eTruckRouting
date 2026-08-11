"""Numerical and boundary tests for charging-to-target behavior."""

import pytest

from EVRoutingEnv.models.simulation.charging_curve import ChargingCurveModel


def _cccv_config() -> dict:
    return {
        "charge_rate": 350.0,
        "efficiency": 0.9,
        "use_realistic_curve": True,
        "taper_start_soc": 0.8,
        "taper_power_min": 150.0,
    }


def test_cccv_target_hits_requested_soc_without_overfill() -> None:
    model = ChargingCurveModel()
    charge, details = model.calculate_charge_to_target(
        initial_soc=0.2,
        target_soc=0.8,
        battery_capacity=400.0,
        charger_config=_cccv_config(),
        charger_type="DCFast",
    )

    assert charge == pytest.approx(240.0, abs=1e-8)
    assert details["final_soc"] == pytest.approx(0.8, abs=1e-12)
    assert details["target_soc"] == 0.8
    assert details["actual_charge_hours"] > 0.0
    assert all(0.2 <= sample[2] <= 0.8 for sample in details["power_curve"])


def test_higher_target_requires_more_energy_and_time() -> None:
    model = ChargingCurveModel()
    low_charge, low = model.calculate_charge_to_target(
        0.4,
        0.7,
        400.0,
        _cccv_config(),
        "DCFast",
    )
    high_charge, high = model.calculate_charge_to_target(
        0.4,
        0.9,
        400.0,
        _cccv_config(),
        "DCFast",
    )

    assert high_charge > low_charge
    assert high["actual_charge_hours"] > low["actual_charge_hours"]


def test_linear_target_duration_is_exact() -> None:
    model = ChargingCurveModel()
    config = {
        "charge_rate": 150.0,
        "efficiency": 0.8,
        "use_realistic_curve": False,
    }
    charge, details = model.calculate_charge_to_target(
        initial_soc=0.25,
        target_soc=0.55,
        battery_capacity=400.0,
        charger_config=config,
        charger_type="Level2",
    )

    assert charge == pytest.approx(120.0)
    assert details["actual_charge_hours"] == pytest.approx(1.0)
    assert details["final_soc"] == pytest.approx(0.55)


@pytest.mark.parametrize("target", [-0.1, 0.0, 1.1, float("nan")])
def test_invalid_target_soc_is_rejected(target: float) -> None:
    with pytest.raises(ValueError, match="target_soc"):
        ChargingCurveModel().calculate_charge_to_target(
            0.2,
            target,
            400.0,
            _cccv_config(),
            "DCFast",
        )


def test_target_at_or_below_initial_is_rejected() -> None:
    with pytest.raises(ValueError, match="above initial_soc"):
        ChargingCurveModel().calculate_charge_to_target(
            0.5,
            0.5,
            400.0,
            _cccv_config(),
            "DCFast",
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("initial_soc", float("nan"), "initial_soc"),
        ("charge_hours", float("inf"), "charge_hours"),
        ("battery_capacity", float("nan"), "battery_capacity"),
    ],
)
def test_duration_charge_rejects_nonfinite_physical_inputs(
    field: str,
    value: float,
    message: str,
) -> None:
    values = {
        "initial_soc": 0.2,
        "charge_hours": 1.0,
        "battery_capacity": 400.0,
        "charger_config": _cccv_config(),
        "charger_type": "DCFast",
    }
    values[field] = value

    with pytest.raises(ValueError, match=message):
        ChargingCurveModel().calculate_charge(**values)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("charge_rate", float("nan")),
        ("efficiency", float("nan")),
        ("taper_start_soc", float("nan")),
        ("taper_power_min", 351.0),
    ],
)
def test_target_charge_rejects_invalid_curve_parameters(
    field: str,
    value: float,
) -> None:
    config = _cccv_config()
    config[field] = value

    with pytest.raises(ValueError, match=field):
        ChargingCurveModel().calculate_charge_to_target(
            0.2,
            0.9,
            400.0,
            config,
            "DCFast",
        )


def test_curve_mode_and_charger_type_are_not_coerced_silently() -> None:
    config = _cccv_config()
    config["use_realistic_curve"] = "true"
    with pytest.raises(TypeError, match="use_realistic_curve"):
        ChargingCurveModel().calculate_charge_to_target(
            0.2,
            0.9,
            400.0,
            config,
            "DCFast",
        )

    config["use_realistic_curve"] = True
    with pytest.raises(ValueError, match="charger_type"):
        ChargingCurveModel().calculate_charge_to_target(
            0.2,
            0.9,
            400.0,
            config,
            "unknown",
        )
