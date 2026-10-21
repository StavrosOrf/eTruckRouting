"""Hand-computed DriverClock tables for the US, EU and Bai regimes (plan 8.3).

Each row is an activity sequence from a fresh clock (on duty at t = 0) and the
driving allowance that remains, computed by hand from the regulation text:
D drive, C charge, H handling, Q queue, S service, W window wait, I idle,
B rest break, R daily rest.
"""

import random

import pytest

from EVRoutingEnv.models.core import hours_of_service as h


A = h.HosActivity
CODES = {"D": A.DRIVE, "C": A.CHARGE, "H": A.HANDLING, "Q": A.QUEUE, "S": A.SERVICE,
         "W": A.WINDOW_WAIT, "I": A.IDLE, "B": A.REST_BREAK, "R": A.REST_DAILY}


def run(regime, steps):
    clock = h.DriverClock.fresh()
    for code, hours in steps:
        clock = h.apply(regime, clock, CODES[code], hours)
    return clock


US_TABLE = [
    ([], 8.0),
    ([("D", 3)], 5.0),
    ([("D", 8)], 0.0),  # B1: exactly the limit is legal, nothing remains
    ([("D", 8), ("B", 0.5)], 3.0),
    ([("D", 8), ("C", 0.5)], 3.0),  # on duty not driving qualifies
    ([("D", 8), ("Q", 0.2), ("C", 0.2), ("S", 0.1)], 3.0),  # B5: contiguous mix
    ([("D", 7), ("C", 0.2), ("D", 0.5), ("C", 0.4)], 0.5),  # only contiguous time counts
    ([("D", 5), ("W", 0.5)], 6.0),  # B14: a window wait completes a break
    ([("D", 8), ("B", 0.5), ("D", 3)], 0.0),  # 11 h shift driving
    ([("D", 8), ("B", 0.5), ("D", 2), ("I", 3)], 0.5),  # 14 h window binds
    ([("I", 13)], 1.0),
    ([("I", 14)], 0.0),
    ([("R", 10)], 8.0),
    ([("D", 5), ("R", 10)], 8.0),  # 10 h off duty resets the shift
    ([("D", 5), ("R", 10), ("I", 2)], 8.0),  # window restarts on duty at t = 15
    ([("D", 5), ("R", 10), ("I", 13.5)], 0.5),
    ([("D", 5), ("R", 9.9)], 0.0),  # no reset; the old window has expired
    ([("D", 5), ("B", 0.5), ("R", 9.5)], 8.0),  # contiguous off duty totals 10 h
    ([("D", 5), ("C", 0.5), ("R", 9.5)], 0.0),  # charging is on duty: no reset
    ([("D", 4), ("C", 0.25), ("D", 4)], 0.0),
    ([("D", 4), ("C", 0.25), ("B", 0.25), ("D", 4)], 3.0),
    ([("D", 6), ("S", 0.3), ("H", 0.2)], 5.0),
    ([("D", 10.9)], 0.0),
    ([("D", 3), ("B", 0.5), ("D", 3), ("B", 0.5), ("D", 3), ("B", 0.5)], 2.0),
    ([("D", 3), ("B", 0.5), ("D", 3), ("B", 0.5), ("D", 3), ("B", 0.5), ("D", 2)], 0.0),
    ([("R", 10), ("D", 8)], 0.0),
    ([("R", 10), ("I", 1), ("D", 2)], 6.0),
]

EU_TABLE = [
    ({}, [], 4.5),
    ({}, [("D", 4.5)], 0.0),  # B1
    ({}, [("D", 4.5), ("B", 0.75)], 4.5),
    ({}, [("D", 2), ("B", 0.25), ("D", 2.5), ("B", 0.5)], 4.5),  # B2: 15 then 30
    ({}, [("D", 2), ("B", 0.5), ("D", 2.5), ("B", 0.25)], 0.0),  # B3: 30 then 15
    ({}, [("D", 4.5), ("C", 44 / 60), ("Q", 1 / 60)], 0.0),  # B4
    ({"queue_wait_counts_as_break": True}, [("D", 4.5), ("C", 44 / 60), ("Q", 1 / 60)], 4.5),
    ({}, [("D", 4.5), ("S", 0.75)], 0.0),  # loading is work
    ({}, [("D", 4.5), ("W", 0.75)], 0.0),
    ({}, [("D", 4.5), ("I", 0.75)], 0.0),
    ({}, [("D", 4.5), ("H", 1 / 12), ("C", 0.75)], 4.5),
    ({}, [("D", 4.5), ("H", 0.5)], 0.0),  # B11: overhead longer than the session
    ({}, [("D", 2), ("B", 0.25), ("D", 2), ("B", 0.25), ("D", 0.5), ("B", 0.5)], 4.5),
    ({}, [("D", 4.5), ("B", 0.75), ("D", 4.5)], 0.0),  # 9 h daily driving
    ({}, [("D", 4.5), ("B", 0.75), ("D", 4.5), ("R", 11)], 4.5),
    ({}, [("D", 4.5), ("B", 0.75), ("D", 4.5), ("R", 10.9)], 0.0),
    ({}, [("I", 12)], 1.0),  # 13 h window
    ({}, [("I", 13)], 0.0),
    ({}, [("R", 11), ("I", 1)], 4.5),
    ({}, [("D", 3), ("Q", 1)], 1.5),  # queue waiting is not a break by default
    ({}, [("D", 3), ("C", 1)], 4.5),  # charging counts by default
    ({"charging_counts_as_break": False}, [("D", 3), ("C", 1)], 1.5),
    ({"eu_split_break": False}, [("D", 2), ("B", 0.25), ("D", 2.5), ("B", 0.5)], 0.0),
    ({}, [("D", 2), ("B", 0.3), ("D", 2.5), ("B", 0.5)], 4.5),
    ({}, [("D", 1), ("B", 1.0), ("D", 1)], 3.5),
    ({}, [("D", 1), ("B", 1.0), ("D", 3.5), ("B", 0.5)], 1.0),  # a full break is no first part
    ({"eu_extended_daily_driving": True}, [("D", 4.5), ("B", 0.75), ("D", 4.5), ("B", 0.75)], 1.0),
    ({"eu_reduced_daily_rest": True}, [("I", 14)], 1.0),  # 15 h window with a reduced rest
    ({"eu_reduced_daily_rest": True}, [("D", 4.5), ("B", 0.75), ("D", 4.5), ("R", 9)], 4.5),
]

BAI_TABLE = [
    ([], 4.5),
    ([("D", 4.5)], 0.0),
    ([("D", 4.5), ("C", 0.75)], 4.5),
    ([("D", 4.5), ("C", 0.5)], 0.0),
    ([("D", 4.5), ("C", 0.5), ("Q", 0.25)], 4.5),
    ([("D", 4.5), ("S", 0.75)], 4.5),
    ([("D", 4.5), ("C", 0.75), ("D", 4.5)], 0.0),
    ([("D", 4.5), ("C", 0.75), ("D", 4.0)], 0.5),
    ([("D", 2), ("C", 0.5), ("D", 2), ("C", 0.5)], 0.5),
    ([("D", 2), ("C", 0.5), ("D", 2), ("C", 0.25)], 0.5),
    ([("I", 100)], 4.5),  # no window
    ([("D", 3), ("R", 0.75)], 4.5),
    ([("D", 1), ("C", 0.74)], 3.5),
    ([("D", 1), ("C", 0.75)], 4.5),
    ([("D", 4), ("W", 0.75), ("D", 4)], 0.5),
    ([("D", 4.5), ("C", 0.75), ("D", 4.5), ("C", 0.75)], 0.0),  # 9 h per trip
    ([("D", 0.5), ("C", 0.1), ("D", 0.5)], 3.5),
    ([("D", 4.4)], 0.1),
    ([("D", 4.5), ("B", 0.75)], 4.5),
    ([("D", 4.5), ("R", 0.75)], 4.5),
    ([("D", 4.5), ("I", 0.75)], 4.5),
    ([("D", 4.5), ("H", 0.75)], 4.5),
    ([("D", 2), ("Q", 0.4), ("C", 0.35)], 4.5),
    ([("D", 8.9)], 0.0),
    ([("D", 4.5), ("C", 1.0), ("D", 4.4)], 0.1),
]


@pytest.mark.parametrize(("steps", "allowance"), US_TABLE)
def test_us_table(steps, allowance) -> None:
    regime = h.HosRegime.build("us_fmcsa_property")
    assert h.driving_allowance(regime, run(regime, steps)) == pytest.approx(allowance, abs=1e-9)


@pytest.mark.parametrize(("options", "steps", "allowance"), EU_TABLE)
def test_eu_table(options, steps, allowance) -> None:
    regime = h.HosRegime.build("eu_561_2006", **options)
    assert h.driving_allowance(regime, run(regime, steps)) == pytest.approx(allowance, abs=1e-9)


@pytest.mark.parametrize(("steps", "allowance"), BAI_TABLE)
def test_bai_table(steps, allowance) -> None:
    regime = h.HosRegime.build("bai2023_simplified")
    assert h.driving_allowance(regime, run(regime, steps)) == pytest.approx(allowance, abs=1e-9)


def test_tables_have_at_least_25_rows_per_regime() -> None:
    assert min(len(US_TABLE), len(EU_TABLE), len(BAI_TABLE)) >= 25


def test_rest_lengths_and_required_rest() -> None:
    us = h.HosRegime.build("us_fmcsa_property")
    eu = h.HosRegime.build("eu_561_2006")
    assert h.rest_hours(us, h.DriverClock.fresh(), "break") == 0.5
    assert h.rest_hours(us, h.DriverClock.fresh(), "daily") == 10.0
    assert h.rest_hours(eu, h.DriverClock.fresh(), "break") == 0.75
    assert h.rest_hours(eu, run(eu, [("D", 1), ("B", 0.25), ("D", 1)]), "break") == 0.5
    assert h.rest_hours(eu, h.DriverClock.fresh(), "daily") == 11.0
    assert h.required_rest(us, run(us, [("D", 8)])) == ("break", 0.5)
    assert h.required_rest(us, run(us, [("D", 8), ("B", 0.5), ("D", 3)])) == ("daily", 10.0)
    assert h.required_rest(us, run(us, [("D", 2)])) == (None, 0.0)
    bai = h.HosRegime.build("bai2023_simplified")
    with pytest.raises(h.HosInfeasible):
        h.required_rest(bai, run(bai, [("D", 4.5), ("C", 0.75), ("D", 4.5)]))


def test_B7_insert_rests_splits_a_leg_longer_than_a_shift_exactly() -> None:
    us = h.HosRegime.build("us_fmcsa_property")
    segments, clock = h.insert_rests(us, h.DriverClock.fresh(), 25.0)
    assert [(a.value, hours) for a, hours in segments] == [
        ("drive", 8.0), ("rest_break", 0.5), ("drive", 3.0), ("rest_daily", 10.0),
        ("drive", 8.0), ("rest_break", 0.5), ("drive", 3.0), ("rest_daily", 10.0), ("drive", 3.0),
    ]
    assert clock.time == pytest.approx(46.0, abs=1e-9)
    assert sum(hours for a, hours in segments if a is A.DRIVE) == pytest.approx(25.0)


def test_bai_clock_equals_bai_recursion() -> None:
    """Bai et al. (2023) eq. (4) with constraints 7-9: consecutive driving resets
    after a stay of at least T_r; shorter stays do not."""
    regime = h.HosRegime.build("bai2023_simplified")
    rng = random.Random(4)
    for _ in range(2000):
        clock, consecutive = h.DriverClock.fresh(), 0.0
        for _ in range(rng.randint(1, 12)):
            drive, stay = rng.uniform(0, 1.5), rng.choice([rng.uniform(0, 1.5), 0.75, 0.74])
            clock = h.apply(regime, clock, A.DRIVE, drive)
            consecutive += drive
            clock = h.apply(regime, clock, rng.choice([A.CHARGE, A.REST_BREAK, A.QUEUE]), stay)
            if stay >= 0.75 - 1e-12:
                consecutive = 0.0
            assert clock.driving_since_break == pytest.approx(consecutive, abs=1e-9)


def test_zero_length_activity_changes_nothing() -> None:
    eu = h.HosRegime.build("eu_561_2006")
    clock = run(eu, [("D", 2), ("B", 0.3)])
    assert h.apply(eu, clock, A.SERVICE, 0.0) == clock


def test_advance_never_moves_the_clock_back() -> None:
    us = h.HosRegime.build("us_fmcsa_property")
    with pytest.raises(ValueError):
        h.advance_to(us, run(us, [("D", 2)]), 1.0, A.IDLE)
