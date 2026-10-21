"""HosTracker leg planning and features (regressions from the local review)."""

import dataclasses

import pytest

from EVRoutingEnv.models.core import hours_of_service as h
from EVRoutingEnv.models.environment.extension_config import HoursOfServiceConfig
from EVRoutingEnv.models.environment.hos_tracker import HosTracker


class _Truck:
    truck_id = 0

    def _record_event(self, **kwargs):
        pass


def _tracker(**config):
    return HosTracker(HoursOfServiceConfig(enabled=True, **config), num_trucks=1)


def test_a_break_ending_after_the_window_is_replaced_by_the_daily_rest() -> None:
    us = h.HosRegime.build("us_fmcsa_property")
    clock = h.DriverClock(time=13.8, driving_since_break=8.0, driving_in_shift=9.0, shift_start=0.0)
    assert h.required_rest(us, clock) == ("daily", 10.0)


def test_features_use_the_extended_eu_limit_and_clip_the_window() -> None:
    tracker = _tracker(regime="eu_561_2006", eu_extended_daily_driving=True)
    tracker._clocks[0] = h.DriverClock(time=5.0, driving_in_shift=9.2, shift_start=0.0)
    row = tracker.feature_row(_Truck())
    assert row[1] == pytest.approx(0.8)
    tracker._clocks[0] = h.DriverClock(time=20.0, shift_start=0.0)
    assert tracker.feature_row(_Truck())[2] == 0.0


def test_bai_trip_cap_keeps_the_rests_already_placed() -> None:
    tracker = _tracker(regime="bai2023_simplified", rest_locations="stops_only")
    tracker._clocks[0] = h.DriverClock(time=4.0, driving_since_break=4.0, driving_in_shift=4.0)
    segments = tracker.plan_leg(_Truck(), 4.0, 6.0)
    assert segments == [("drive", 0.5), ("rest_break", 0.75), ("drive", 4.5), ("drive", 1.0)]
    assert tracker.unplanned_violations == 1


def test_planning_starts_at_the_departure_time() -> None:
    tracker = _tracker(regime="us_fmcsa_property")
    tracker._clocks[0] = h.DriverClock(time=0.0, driving_since_break=7.0, driving_in_shift=7.0, shift_start=0.0)
    # Ready 0.5 h later: the idle gap is a 30-minute break under US rules.
    segments = tracker.plan_leg(_Truck(), 0.5, 2.0)
    assert segments == [("drive", 2.0)]


def test_planning_rejects_bad_driving_times() -> None:
    tracker = _tracker()
    with pytest.raises(ValueError):
        tracker.plan_leg(_Truck(), 0.0, float("nan"))
    with pytest.raises(ValueError):
        tracker.plan_leg(_Truck(), 0.0, -1.0)


def test_handling_overhead_is_capped_by_the_session() -> None:
    tracker = _tracker(regime="eu_561_2006", handling_overhead_hours=1.0)
    tracker.record_charge(_Truck(), 0.0, 0.25)
    assert tracker.clock(0).time == pytest.approx(0.25)
    assert dataclasses.asdict(tracker.clock(0))["break_streak"] == 0.0
