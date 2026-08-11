"""Boundary validation for stochastic simulator inputs and parameters."""

import pytest

from EVRoutingEnv.models.simulation.delivery_simulator import DeliverySimulator
from EVRoutingEnv.models.simulation.traffic_simulation import TrafficSimulator


def _traffic(**overrides) -> TrafficSimulator:
    values = {
        "enable_traffic": True,
        "std_dev_factor": 0.15,
        "max_std_dev_hours": 1.0,
        "rush_hour_multiplier": 2.0,
        "enable_energy_uncertainty": True,
        "energy_uncertainty_factor": 0.05,
        "min_energy_multiplier": 0.9,
        "max_energy_multiplier": 1.2,
        "seed": 1,
    }
    values.update(overrides)
    return TrafficSimulator(**values)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("std_dev_factor", -0.1),
        ("std_dev_factor", float("nan")),
        ("rush_hour_multiplier", 0.5),
        ("energy_uncertainty_factor", -0.1),
        ("min_energy_multiplier", 0.0),
        ("max_energy_multiplier", 0.9),
    ],
)
def test_invalid_traffic_parameters_are_rejected(field: str, value: float) -> None:
    with pytest.raises(ValueError):
        _traffic(**{field: value})


def test_invalid_runtime_traffic_inputs_are_rejected() -> None:
    simulator = _traffic()
    with pytest.raises(ValueError, match="travel_time"):
        simulator.apply_traffic(-1.0, 0.0, 1, 2)
    with pytest.raises(ValueError, match="current_time"):
        simulator.apply_traffic(1.0, float("nan"), 1, 2)
    with pytest.raises(ValueError, match="base_energy"):
        simulator.apply_energy_uncertainty(-1.0)
    with pytest.raises(ValueError, match="traffic_multiplier"):
        simulator.apply_energy_uncertainty(1.0, traffic_multiplier=0.0)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("base_unloading_time", -0.1),
        ("std_dev_factor", -0.1),
        ("business_hours_multiplier", 0.5),
        ("min_unloading_multiplier", 0.0),
        ("max_unloading_multiplier", 0.5),
    ],
)
def test_invalid_delivery_parameters_are_rejected(field: str, value: float) -> None:
    values = {
        "enable_stochastic_unloading": True,
        "base_unloading_time": 0.5,
        "std_dev_factor": 0.2,
        "max_std_dev_hours": 0.25,
        "business_hours_multiplier": 1.5,
        "min_unloading_multiplier": 0.75,
        "max_unloading_multiplier": 1.5,
    }
    values[field] = value
    with pytest.raises(ValueError):
        DeliverySimulator(**values)


def test_zero_deterministic_service_is_allowed_but_zero_stochastic_is_not() -> None:
    deterministic = DeliverySimulator(
        enable_stochastic_unloading=False,
        base_unloading_time=0.0,
    )
    assert deterministic.apply_unloading_time(1, 0.0) == 0.0
    with pytest.raises(ValueError, match="requires positive"):
        DeliverySimulator(
            enable_stochastic_unloading=True,
            base_unloading_time=0.0,
        )


def test_invalid_delivery_runtime_inputs_are_rejected() -> None:
    simulator = DeliverySimulator()
    with pytest.raises(ValueError, match="delivery_node"):
        simulator.apply_unloading_time(-1, 0.0)
    with pytest.raises(ValueError, match="current_time"):
        simulator.apply_unloading_time(1, -1.0)
