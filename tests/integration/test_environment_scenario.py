"""Integration checks for scenario identity and instance replay."""

import os

import numpy as np
import pytest


os.environ.setdefault("MPLCONFIGDIR", "/tmp/evrp_matplotlib")

from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv


@pytest.mark.integration
def test_environment_reset_replays_instance_and_observation() -> None:
    env = EventDrivenTruckEnv(
        "EVRoutingEnv/config_files/config.yaml",
        verbose=False,
        enable_plotting=False,
    )
    try:
        first_observation, first_info = env.reset(seed=8123)
        first_plans = [tuple(truck.delivery_sequence) for truck in env.trucks]

        second_observation, second_info = env.reset(seed=8123)
        second_plans = [tuple(truck.delivery_sequence) for truck in env.trucks]

        assert first_info["scenario"] == {"seed": 8123, "version": "1"}
        assert second_info["scenario"] == first_info["scenario"]
        assert second_plans == first_plans
        np.testing.assert_array_equal(second_observation, first_observation)
    finally:
        env.close()


@pytest.mark.integration
def test_environment_scenario_controls_exogenous_draws() -> None:
    env = EventDrivenTruckEnv(
        "EVRoutingEnv/config_files/config.yaml",
        verbose=False,
        enable_plotting=False,
    )
    try:
        env.reset(seed=9201)
        first = env.traffic_simulator._get_uncertainty_values(10, 20, 7.5)

        env.reset(seed=9201)
        replay = env.traffic_simulator._get_uncertainty_values(10, 20, 7.5)

        env.reset(seed=9202)
        changed = env.traffic_simulator._get_uncertainty_values(10, 20, 7.5)

        assert replay == first
        assert changed != first
    finally:
        env.close()
