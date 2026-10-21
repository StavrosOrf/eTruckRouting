"""Validation of the opt-in blocks: no silent relaxation (plan rule 3)."""

import copy

import pytest

from EVRoutingEnv.models.environment.extension_config import (
    KULLMAN_FAST_MINUTES,
    parse_extension_config,
)
from EVRoutingEnv.utils.utils import load_config


def _config(**blocks):
    config = copy.deepcopy(load_config("EVRoutingEnv/config_files/config_joint.yaml"))
    config.update(copy.deepcopy(blocks))
    return config


def _parse(config, max_time=48.0, sequential=False, preassigned=False):
    return parse_extension_config(config, max_time=max_time, sequential=sequential,
                                  preassigned=preassigned)


def test_absent_blocks_disable_everything() -> None:
    extensions = _parse(_config())
    assert not extensions.any_enabled
    assert not extensions.public_charging.enabled and not extensions.hours_of_service.enabled
    assert extensions.requests.mode == "static" and extensions.in_flight_information == "realized"


def test_A11_utilization_one_is_rejected() -> None:
    with pytest.raises(ValueError, match="< 1"):
        _parse(_config(public_charging={"enabled": True, "exogenous_traffic": {"utilization": 1.0}}))


def test_A10_utilization_099_is_allowed() -> None:
    parsed = _parse(_config(public_charging={"enabled": True, "exogenous_traffic": {"utilization": 0.99}}))
    assert parsed.public_charging.utilization == 0.99


def test_A19_iid_waits_reject_realtime_information() -> None:
    with pytest.raises(ValueError, match="information=on_arrival"):
        _parse(_config(public_charging={
            "enabled": True, "queue_model": "iid_stationary_wait", "queue_policy": "legacy",
            "information": "realtime_queue",
        }))


def test_kullman_policy_requires_the_dynamic_queue() -> None:
    with pytest.raises(ValueError, match="kullman2021 requires queue_model=dynamic_mmc"):
        _parse(_config(public_charging={"enabled": True, "queue_model": "iid_stationary_wait"}))


@pytest.mark.parametrize(
    ("traffic", "message"),
    [
        ({"service_preset": "kullman_fast", "mean_service_hours": 0.5}, "exactly one"),
        ({"arrival_model": "piecewise_poisson"}, "24 values"),
        ({"arrival_model": "piecewise_poisson", "hourly_profile": [2.0] * 24}, "mean 1"),
        ({"hourly_profile": [1.0] * 24}, "piecewise_poisson"),
        ({"service_preset": "nonsense"}, "service_preset"),
    ],
)
def test_exogenous_traffic_validation(traffic, message) -> None:
    with pytest.raises(ValueError, match=message):
        _parse(_config(public_charging={"enabled": True, "exogenous_traffic": traffic}))


def test_presets_resolve_to_published_values() -> None:
    parsed = _parse(_config(public_charging={"enabled": True}))
    assert parsed.public_charging.mean_service_hours == pytest.approx(KULLMAN_FAST_MINUTES / 60)
    keskin = _parse(_config(public_charging={
        "enabled": True, "exogenous_traffic": {"service_preset": "keskin_truck"}}))
    # 0.55 x 400 kWh / (150 kW x 0.9) = 1.63 h (plan section 5.1)
    assert keskin.public_charging.mean_service_hours == pytest.approx(1.6296, abs=1e-4)


@pytest.mark.parametrize("availability", [0.0, 1.5])
def test_port_availability_bounds(availability) -> None:
    with pytest.raises(ValueError, match="port_availability"):
        _parse(_config(public_charging={"enabled": True, "outages": {"port_availability": availability}}))


def test_interrupted_sessions_are_refused_explicitly() -> None:
    with pytest.raises(ValueError, match="interrupt_sessions"):
        _parse(_config(public_charging={"enabled": True, "outages": {"interrupt_sessions": True}}))


def test_B15_hours_of_service_cap_the_horizon() -> None:
    with pytest.raises(ValueError, match="weekly limits"):
        _parse(_config(hours_of_service={"enabled": True}), max_time=120.0)


def test_bai_regime_requires_stops_only() -> None:
    with pytest.raises(ValueError, match="stops_only"):
        _parse(_config(hours_of_service={"enabled": True, "regime": "bai2023_simplified"}))


def test_unknown_regime_and_unknown_keys_are_rejected() -> None:
    with pytest.raises(ValueError, match="regime"):
        _parse(_config(hours_of_service={"enabled": True, "regime": "mars"}))
    with pytest.raises(ValueError, match="unknown hours_of_service keys"):
        _parse(_config(hours_of_service={"enabled": True, "nap_hours": 1}))


def test_depot_charging_is_rejected_where_the_depot_retires_trucks() -> None:
    with pytest.raises(ValueError, match="depot_charging"):
        _parse(_config(depot_charging={"enabled": True}))


def test_dynamic_modes_must_state_depot_charging() -> None:
    with pytest.raises(ValueError, match="depot_charging.enabled explicitly"):
        _parse(_config(requests={"mode": "same_day"}))


def test_C21_dynamic_requests_reject_preassignment_and_sequential_mode() -> None:
    config = _config(requests={"mode": "same_day"}, depot_charging={"enabled": False})
    with pytest.raises(ValueError, match="assignment=fleet"):
        _parse(config, preassigned=True)
    with pytest.raises(ValueError, match="joint_fleet"):
        _parse(config, sequential=True)


def test_drayage_requires_terminals_and_only_drayage_takes_them() -> None:
    with pytest.raises(ValueError, match="drayage block"):
        _parse(_config(requests={"mode": "drayage"}, depot_charging={"enabled": False}))
    with pytest.raises(ValueError, match="only to requests.mode=drayage"):
        _parse(_config(drayage={"terminals": [{"name": "x", "lat": 0.0, "lon": 0.0}]}))


def test_poisson_arrivals_need_a_rate() -> None:
    with pytest.raises(ValueError, match="poisson_rate_per_hour"):
        _parse(_config(requests={"mode": "same_day", "arrival_process": "poisson"},
                       depot_charging={"enabled": False}))


def test_loading_option_and_in_flight_option() -> None:
    parsed = _parse(_config(problem={**_config()["problem"], "loading": "at_depot"},
                            depot_charging={"enabled": False},
                            observation={"in_flight_information": "nominal_eta"}))
    assert parsed.loading == "at_depot" and parsed.in_flight_information == "nominal_eta"
    with pytest.raises(ValueError, match="in_flight_information"):
        _parse(_config(observation={"in_flight_information": "psychic"}))


def test_request_windows_reject_unknown_keys_and_reversed_bounds() -> None:
    def windows(**block):
        return _config(requests={"mode": "same_day", "time_windows": {"enabled": True, **block}},
                       depot_charging={"enabled": False})

    with pytest.raises(ValueError, match="unknown requests.time_windows keys"):
        _parse(windows(window_width=4.0))  # a typo must not fall back to the default
    with pytest.raises(ValueError, match="min <= max"):
        _parse(windows(earliest_min=5.0, earliest_max=2.0))
    with pytest.raises(ValueError, match="window_width_min"):
        _parse(windows(window_width_min=0.0))
    parsed = _parse(windows(earliest_min=1.0, earliest_max=4.0, window_width_min=2.0, window_width_max=3.0))
    assert parsed.requests.time_windows["earliest_max"] == 4.0
