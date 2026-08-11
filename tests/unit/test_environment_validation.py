"""Boundary validation for the public environment configuration."""

import os
from copy import deepcopy

import pytest


os.environ.setdefault("MPLCONFIGDIR", "/tmp/evrp_matplotlib")

from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.utils.utils import load_config


def _config() -> dict:
    return deepcopy(load_config("EVRoutingEnv/config_files/config_vrp.yaml"))


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("num_trucks", 0, ValueError),
        ("num_trucks", True, ValueError),
        ("num_stops", 0, ValueError),
        ("allow_variable_num_stops", "false", TypeError),
        ("min_hop_distance", -1.0, ValueError),
        ("max_time", float("nan"), ValueError),
        ("max_episode_steps", 0, ValueError),
        ("verbose", "false", TypeError),
    ],
)
def test_invalid_environment_boundaries_fail_before_reset(
    field: str,
    value,
    error: type[Exception],
) -> None:
    config = _config()
    config["environment"][field] = value

    with pytest.raises(error, match=field):
        EventDrivenTruckEnv(config, enable_plotting=False)


def test_hop_distance_bounds_must_be_ordered() -> None:
    config = _config()
    config["environment"]["min_hop_distance"] = 10.0
    config["environment"]["max_hop_distance"] = 9.0

    with pytest.raises(ValueError, match="max_hop_distance"):
        EventDrivenTruckEnv(config, enable_plotting=False)


def test_configured_episode_step_limit_is_honored() -> None:
    config = _config()
    config["environment"]["max_episode_steps"] = 17
    env = EventDrivenTruckEnv(config, verbose=False, enable_plotting=False)
    try:
        assert env.max_episode_steps == 17
    finally:
        env.close()


@pytest.mark.parametrize(
    ("setting", "error"),
    [
        (-1.0, ValueError),
        (101.0, ValueError),
        (float("nan"), ValueError),
        (True, TypeError),
    ],
)
def test_invalid_numeric_initial_battery_percentage_is_rejected(
    setting,
    error,
) -> None:
    config = _config()
    config["truck"]["initial_battery"] = setting
    with pytest.raises(error, match="initial_battery"):
        EventDrivenTruckEnv(config, verbose=False, enable_plotting=False)


def test_nonfinite_duration_action_is_rejected() -> None:
    config = _config()
    config["charging"]["charge_durations"] = [0.25, float("nan")]

    with pytest.raises(ValueError, match="charge_durations"):
        EventDrivenTruckEnv(config, verbose=False, enable_plotting=False)
