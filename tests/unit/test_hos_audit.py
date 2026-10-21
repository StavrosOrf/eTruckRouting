"""The independent audit agrees with DriverClock on legal and illegal schedules."""

import random

import pytest

from EVRoutingEnv.models.core import hours_of_service as h
from EVRoutingEnv.models.core.hos_audit import (
    AuditSettings,
    hos_audit,
    intervals_from_event_log,
)


A = h.HosActivity
CONFIGS = [
    ("us_fmcsa_property", {}),
    ("eu_561_2006", {}),
    ("eu_561_2006", {"queue_wait_counts_as_break": True, "charging_counts_as_break": False}),
    ("eu_561_2006", {"eu_extended_daily_driving": True, "eu_reduced_daily_rest": True}),
    ("eu_561_2006", {"eu_split_break": False}),
    ("bai2023_simplified", {}),
]


def _random_schedule(rng, regime):
    clock, t, intervals, flagged = h.DriverClock.fresh(), 0.0, [], set()
    for _ in range(rng.randint(1, 40)):
        activity = rng.choice(list(A))
        if activity is A.DRIVE:
            hours = rng.choice([rng.uniform(0, 3), h.driving_allowance(regime, clock), rng.uniform(0, 12)])
            if hours <= 1e-9:
                continue
            if hours > h.driving_allowance(regime, clock) + 1e-9:
                flagged.add(round(t + hours, 6))
        elif activity is A.REST_DAILY:
            try:
                hours = rng.choice([h.rest_hours(regime, clock, "daily"), rng.uniform(0, 12)])
            except h.HosInfeasible:
                continue
        elif activity is A.REST_BREAK:
            hours = rng.choice([h.rest_hours(regime, clock, "break"), rng.uniform(0, 1)])
        else:
            hours = rng.choice([rng.uniform(0, 1), rng.uniform(0, 0.3), 0.25, 0.5, 0.75])
        if hours <= 1e-9:
            continue
        intervals.append((t, t + hours, activity))
        clock = h.apply(regime, clock, activity, hours)
        t += hours
    return intervals, flagged


def test_audit_agrees_with_driver_clock_on_ten_thousand_schedules() -> None:
    rng = random.Random(2026)
    checked = illegal = 0
    for index in range(10_000):
        name, options = CONFIGS[index % len(CONFIGS)]
        regime = h.HosRegime.build(name, **options)
        intervals, flagged = _random_schedule(rng, regime)
        audited = {round(v.time, 6) for v in hos_audit(intervals, AuditSettings(name, **options))}
        assert audited == flagged, (name, options, intervals)
        checked += 1
        illegal += bool(flagged)
    assert checked == 10_000 and 1000 < illegal < 9000  # both legal and illegal cases


def test_audit_names_the_rule_broken() -> None:
    rules = {v.rule for v in hos_audit([(0.0, 9.0, A.DRIVE)], AuditSettings("us_fmcsa_property"))}
    assert rules == {"break"}
    rules = {v.rule for v in hos_audit(
        [(0.0, 4.0, A.DRIVE), (4.0, 4.5, A.REST_BREAK), (4.5, 8.5, A.DRIVE),
         (8.5, 9.0, A.REST_BREAK), (9.0, 12.5, A.DRIVE)],
        AuditSettings("us_fmcsa_property"),
    )}
    assert rules == {"shift_driving"}
    rules = {v.rule for v in hos_audit(
        [(0.0, 13.0, A.IDLE), (13.0, 14.0, A.DRIVE)], AuditSettings("eu_561_2006"),
    )}
    assert rules == {"window"}


def test_event_log_round_trip_with_roadside_rests_and_overhead() -> None:
    log = {"events": [
        {"event_type": "ROUTING_START", "timestamp": 0.0, "details": {}},
        {"event_type": "HOS_REST", "timestamp": 8.5, "details": {"kind": "break", "start": 8.0, "end": 8.5}},
        {"event_type": "ROUTING_END", "timestamp": 10.0, "details": {}},
        {"event_type": "CHARGING_START", "timestamp": 10.0, "details": {}},
        {"event_type": "CHARGING_END", "timestamp": 11.0, "details": {}},
        {"event_type": "TIME_WINDOW_WAIT", "timestamp": 11.5, "details": {"wait_duration_hours": 0.5}},
        {"event_type": "UNLOADING_START", "timestamp": 11.5, "details": {}},
        {"event_type": "UNLOADING_END", "timestamp": 11.7, "details": {}},
    ]}
    intervals = sorted(intervals_from_event_log(log, handling_overhead_hours=0.1))
    assert intervals == [
        (0.0, 8.0, A.DRIVE),
        (8.0, 8.5, A.REST_BREAK),
        (8.5, 10.0, A.DRIVE),
        (10.0, 10.1, A.HANDLING),
        (10.1, 11.0, A.CHARGE),
        (11.0, 11.5, A.WINDOW_WAIT),
        (11.5, 11.7, A.SERVICE),
    ]


def test_overlapping_activities_are_rejected() -> None:
    with pytest.raises(ValueError, match="overlapping"):
        hos_audit([(0.0, 2.0, A.DRIVE), (1.0, 3.0, A.CHARGE)], AuditSettings("us_fmcsa_property"))
