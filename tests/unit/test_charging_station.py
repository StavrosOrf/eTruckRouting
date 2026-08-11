"""Invariant tests for finite-port FCFS charging stations."""

import heapq
import json
import math

import pytest

from EVRoutingEnv.models.environment.event_handlers import Event, EventType
from EVRoutingEnv.models.simulation.charging_station import ChargingStation


class _ChargerGraph:
    def __init__(self, capacities: dict[int, int]) -> None:
        self.capacities = capacities

    def get_charger_capacity(self, node: int) -> int:
        return self.capacities[node]

    def get_charger_type(self, node: int) -> str:
        return "DCFast"


def _station(
    tmp_path,
    capacities: dict[int, int],
    charging_config: dict | None = None,
) -> ChargingStation:
    lookup = tmp_path / "wait.json"
    lookup.write_text(json.dumps({}), encoding="utf-8")
    return ChargingStation(
        charging_nodes=list(capacities),
        transport_graph=_ChargerGraph(capacities),
        waiting_time_lookup_path=str(lookup),
        charging_config=charging_config,
    )


def test_nonpositive_port_capacity_is_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="capacities must be positive"):
        _station(tmp_path, {10: 0})


def test_start_and_finish_enforce_port_and_session_invariants(tmp_path) -> None:
    station = _station(tmp_path, {10: 1, 20: 1})
    station.start_charging(0, 10, charge_hours=1.0, global_clock=0.0)

    with pytest.raises(RuntimeError, match="no free port"):
        station.start_charging(1, 10, charge_hours=1.0, global_clock=0.0)
    with pytest.raises(RuntimeError, match="already charging at 10"):
        station.start_charging(0, 10, charge_hours=1.0, global_clock=0.0)
    with pytest.raises(RuntimeError, match="already charging at 10"):
        station.start_charging(0, 20, charge_hours=1.0, global_clock=0.0)
    with pytest.raises(ValueError, match="charge_hours must be positive"):
        station.start_charging(2, 20, charge_hours=0.0, global_clock=0.0)

    station.finish_charging(0, 10, global_clock=1.0)
    with pytest.raises(RuntimeError, match="is not charging"):
        station.finish_charging(0, 10, global_clock=1.0)


def test_fcfs_wakes_exactly_one_waiter_per_free_port(tmp_path) -> None:
    station = _station(tmp_path, {10: 1})
    station.start_charging(0, 10, charge_hours=1.0, global_clock=0.0)
    assert station.check_charger_gating(1, 10, 0.1) == (False, None)
    assert station.check_charger_gating(2, 10, 0.2) == (False, None)

    station.finish_charging(0, 10, global_clock=1.0)
    events: list[Event] = []
    states = {1: "waiting_to_charge", 2: "waiting_to_charge"}
    station.wake_waiting_trucks(10, 1.0, events, EventType, Event, states)
    station.wake_waiting_trucks(10, 1.0, events, EventType, Event, states)

    assert len(events) == 1
    first = heapq.heappop(events)
    assert first.truck_id == 1
    assert station.check_charger_gating(1, 10, 1.0) == (True, None)
    station.start_charging(1, 10, charge_hours=1.0, global_clock=1.0)
    station.finish_charging(1, 10, global_clock=2.0)
    station.wake_waiting_trucks(10, 2.0, events, EventType, Event, states)

    assert heapq.heappop(events).truck_id == 2


def test_stale_waiter_is_removed_without_blocking_next_truck(tmp_path) -> None:
    station = _station(tmp_path, {10: 1})
    station.start_charging(0, 10, charge_hours=1.0, global_clock=0.0)
    station.check_charger_gating(1, 10, 0.1)
    station.check_charger_gating(2, 10, 0.2)
    station.finish_charging(0, 10, global_clock=1.0)

    events: list[Event] = []
    states = {1: "routing", 2: "waiting_to_charge"}
    station.wake_waiting_trucks(10, 1.0, events, EventType, Event, states)

    assert [entry["truck_id"] for entry in station.charger_waitlist[10]] == [2]
    assert heapq.heappop(events).truck_id == 2


def test_station_occupied_time_is_not_reset_by_second_port_start(tmp_path) -> None:
    station = _station(tmp_path, {10: 2})
    station.start_charging(0, 10, charge_hours=2.0, global_clock=1.0)
    station.start_charging(1, 10, charge_hours=2.0, global_clock=2.0)
    station.finish_charging(0, 10, global_clock=3.0)
    station.finish_charging(1, 10, global_clock=4.0)

    stats = station.get_utilization_stats(global_clock=4.0)
    assert station.charger_stats[10]["occupancy_time"] == 3.0
    assert stats["overall"]["avg_utilization"] == 0.75


def test_station_closure_releases_waiters_and_blocks_new_sessions(tmp_path) -> None:
    station = _station(tmp_path, {10: 1})
    station.start_charging(0, 10, charge_hours=1.0, global_clock=0.0)
    station.check_charger_gating(1, 10, 0.1)
    station.check_charger_gating(2, 10, 0.2)

    released = station.set_station_available(10, False)

    assert released == [1, 2]
    assert station.charger_waitlist[10] == []
    assert station.check_charger_gating(3, 10, 0.3) == (False, None)
    with pytest.raises(RuntimeError, match="unavailable"):
        station.start_charging(3, 10, charge_hours=1.0, global_clock=0.3)

    station.finish_charging(0, 10, global_clock=1.0)
    station.set_station_available(10, True)
    assert station.check_charger_gating(3, 10, 1.0) == (True, None)


def test_closed_station_wait_estimate_is_float_infinity(tmp_path) -> None:
    station = _station(tmp_path, {10: 1})
    station.set_station_available(10, False)

    assert math.isinf(station.get_waiting_time(10, 0.5))
    with pytest.raises(ValueError, match="current_utilization"):
        station.get_waiting_time(10, float("nan"))


def test_invalid_station_power_configuration_is_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="power classes"):
        _station(
            tmp_path,
            {10: 1},
            {"station_power_classes_kw": [float("nan")]},
        )
    with pytest.raises(ValueError, match="unknown nodes"):
        _station(
            tmp_path,
            {10: 1},
            {"station_power_overrides_kw": {99: 350.0}},
        )


def test_charging_session_rejects_nonfinite_and_early_times(tmp_path) -> None:
    station = _station(tmp_path, {10: 1})
    with pytest.raises(ValueError, match="charge_hours"):
        station.start_charging(0, 10, float("nan"), 0.0)
    with pytest.raises(ValueError, match="global_clock"):
        station.start_charging(0, 10, 1.0, -1.0)

    station.start_charging(0, 10, 1.0, 0.0)
    with pytest.raises(ValueError, match="scheduled end"):
        station.finish_charging(0, 10, global_clock=0.5)
    assert station.charger_occupancy[10] == [0]
