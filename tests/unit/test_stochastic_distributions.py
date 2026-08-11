"""Deterministic Monte Carlo checks for configured uncertainty behavior."""

import numpy as np

from EVRoutingEnv.models.simulation.delivery_simulator import DeliverySimulator
from EVRoutingEnv.models.simulation.traffic_simulation import TrafficSimulator


SAMPLE_COUNT = 2_000


def _traffic() -> TrafficSimulator:
    return TrafficSimulator(
        enable_traffic=True,
        std_dev_factor=0.15,
        max_std_dev_hours=1.0,
        rush_hour_multiplier=2.0,
        enable_energy_uncertainty=True,
        energy_uncertainty_factor=0.05,
        min_energy_multiplier=0.9,
        max_energy_multiplier=1.2,
        seed=20260811,
    )


def test_travel_distribution_clipping_and_rush_hour_variance() -> None:
    simulator = _traffic()
    off_peak = []
    rush_hour = []
    for index in range(SAMPLE_COUNT):
        off_peak.append(
            simulator.apply_traffic(1.0, 12.0, index, index + 1)[0]
        )
        rush_hour.append(
            simulator.apply_traffic(
                1.0,
                8.0,
                index + SAMPLE_COUNT,
                index + SAMPLE_COUNT + 1,
            )[0]
        )

    off_peak = np.asarray(off_peak)
    rush_hour = np.asarray(rush_hour)
    combined = np.concatenate((off_peak, rush_hour))
    assert combined.min() >= 0.85
    assert combined.max() <= 2.5
    assert abs(off_peak.mean() - 1.0) < 0.03
    assert rush_hour.std() > 1.5 * off_peak.std()


def test_energy_is_bounded_and_positively_correlated_with_traffic() -> None:
    simulator = _traffic()
    multipliers = []
    energies = []
    for index in range(SAMPLE_COUNT):
        origin = index + 10_000
        destination = origin + 1
        _, multiplier = simulator.apply_traffic(
            1.0,
            8.0,
            origin,
            destination,
        )
        energy = simulator.apply_energy_uncertainty(
            100.0,
            multiplier,
            8.0,
            origin,
            destination,
        )
        multipliers.append(multiplier)
        energies.append(energy)

    multipliers = np.asarray(multipliers)
    energies = np.asarray(energies)
    assert energies.min() >= 90.0
    assert energies.max() <= 120.0
    assert np.corrcoef(multipliers, energies)[0, 1] > 0.7


def test_service_distribution_clipping_and_business_hour_variance() -> None:
    simulator = DeliverySimulator(
        enable_stochastic_unloading=True,
        base_unloading_time=0.5,
        std_dev_factor=0.2,
        max_std_dev_hours=0.25,
        business_hours_multiplier=1.5,
        min_unloading_multiplier=0.75,
        max_unloading_multiplier=1.5,
        seed=20260812,
    )
    off_hours = np.asarray(
        [
            simulator.apply_unloading_time(index, 2.0)
            for index in range(SAMPLE_COUNT)
        ]
    )
    business_hours = np.asarray(
        [
            simulator.apply_unloading_time(index + SAMPLE_COUNT, 12.0)
            for index in range(SAMPLE_COUNT)
        ]
    )
    combined = np.concatenate((off_hours, business_hours))

    assert combined.min() >= 0.5 * 0.75
    assert combined.max() <= 0.5 * 1.5
    assert abs(off_hours.mean() - 0.5) < 0.03
    assert business_hours.std() > 1.2 * off_hours.std()
