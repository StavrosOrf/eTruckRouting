"""Tests for scenario-scoped common random numbers."""

import numpy as np

from EVRoutingEnv.models.simulation.delivery_simulator import DeliverySimulator
from EVRoutingEnv.models.simulation.scenario import ScenarioRandomStreams
from EVRoutingEnv.models.simulation.traffic_simulation import TrafficSimulator


def _traffic_simulator(seed: int) -> TrafficSimulator:
    return TrafficSimulator(
        enable_traffic=True,
        std_dev_factor=0.15,
        max_std_dev_hours=1.0,
        rush_hour_multiplier=1.5,
        enable_energy_uncertainty=True,
        energy_uncertainty_factor=0.10,
        min_energy_multiplier=0.5,
        max_energy_multiplier=2.0,
        seed=seed,
    )


def test_keyed_samples_replay_for_same_scenario() -> None:
    first = ScenarioRandomStreams(1234)
    second = ScenarioRandomStreams(1234)

    key = (17, 29, 4, 0)
    assert first.standard_normal("travel_time", key) == second.standard_normal(
        "travel_time", key
    )
    assert first.standard_normal("energy", key) == second.standard_normal("energy", key)


def test_scenario_seed_changes_keyed_samples() -> None:
    first = ScenarioRandomStreams(1234)
    second = ScenarioRandomStreams(1235)

    key = (17, 29, 4, 0)
    assert first.standard_normal("travel_time", key) != second.standard_normal(
        "travel_time", key
    )


def test_named_sequential_streams_are_isolated() -> None:
    reference = ScenarioRandomStreams(77)
    expected = reference.generator("instance_generation").integers(0, 10_000, 5)

    candidate = ScenarioRandomStreams(77)
    candidate.generator("policy_noise").normal(size=100)
    actual = candidate.generator("instance_generation").integers(0, 10_000, 5)

    np.testing.assert_array_equal(actual, expected)


def test_travel_and_energy_share_one_traversal_index() -> None:
    simulator = _traffic_simulator(101)

    travel_time, multiplier = simulator.apply_traffic(
        travel_time=2.0,
        current_time=3.0,
        from_node=10,
        to_node=20,
    )
    energy = simulator.apply_energy_uncertainty(
        base_energy=100.0,
        traffic_multiplier=multiplier,
        current_time=3.0,
        from_node=10,
        to_node=20,
    )

    assert travel_time > 0.0
    assert energy > 0.0
    assert simulator._journey_counters[(10, 20)] == 1
    assert not simulator._pending_energy


def test_traffic_scenario_reset_replays_both_outcomes() -> None:
    simulator = _traffic_simulator(202)

    def traverse() -> tuple[float, float]:
        actual_time, multiplier = simulator.apply_traffic(
            travel_time=1.5,
            current_time=8.0,
            from_node=3,
            to_node=8,
        )
        actual_energy = simulator.apply_energy_uncertainty(
            base_energy=80.0,
            traffic_multiplier=multiplier,
            current_time=8.0,
            from_node=3,
            to_node=8,
        )
        return actual_time, actual_energy

    first = traverse()
    simulator.reset_scenario(202)
    second = traverse()

    assert first == second


def test_different_traffic_scenarios_change_outcomes() -> None:
    first = _traffic_simulator(303)
    second = _traffic_simulator(304)

    first_values = first._get_uncertainty_values(1, 2, 5.0)
    second_values = second._get_uncertainty_values(1, 2, 5.0)

    assert first_values != second_values


def test_delivery_scenario_reset_replays_service_time() -> None:
    simulator = DeliverySimulator(
        enable_stochastic_unloading=True,
        base_unloading_time=0.5,
        std_dev_factor=0.2,
        max_std_dev_hours=0.25,
        business_hours_multiplier=1.5,
        min_unloading_multiplier=0.5,
        max_unloading_multiplier=2.0,
        seed=404,
    )

    first = simulator.apply_unloading_time(delivery_node=12, current_time=10.0)
    simulator.reset_scenario(404)
    second = simulator.apply_unloading_time(delivery_node=12, current_time=10.0)

    assert first == second


def test_delivery_seed_changes_service_time() -> None:
    first = DeliverySimulator(enable_stochastic_unloading=True, seed=505)
    second = DeliverySimulator(enable_stochastic_unloading=True, seed=506)

    assert first._get_uncertainty_value(5, 11.0) != second._get_uncertainty_value(
        5, 11.0
    )
