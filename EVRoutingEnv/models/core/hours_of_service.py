"""Driver hours-of-service (HOS) rules as a pure state machine.

Three regimes are reproduced one-to-one:

* ``us_fmcsa_property`` -- 49 CFR 395.3: a 10-hour off-duty reset, a 14-hour
  window from coming on duty, at most 11 hours driving in it, and no driving once
  8 hours of driving have passed without a 30-minute consecutive interruption
  (off duty, sleeper berth or on duty not driving).
* ``eu_561_2006`` -- Regulation (EC) 561/2006: after 4.5 hours driving a 45
  minute break, or 15 then 30 minutes in that order (Art. 7); at most 9 hours
  daily driving, 10 hours at most twice a week (Art. 6); an 11-hour daily rest,
  reduced to 9 hours at most three times, completed within 24 hours of the
  previous one (Art. 8), so a 13-hour (15-hour) window.
* ``bai2023_simplified`` -- Bai et al. (2023, arXiv:2303.08895): 4.5 hours
  consecutive driving, reset by a stay of at least 45 minutes, and 9 hours of
  driving per trip, with no daily reset.

Weekly limits are not modelled; the environment caps the horizon at 96 hours.
Only contiguous qualifying time earns break credit. ``DriverClock`` is immutable
and every function here is side-effect free, so the rules can be tested without
an environment; ``hos_audit`` is an independent re-implementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import StrEnum


TOLERANCE = 1e-9
REGIMES = ("us_fmcsa_property", "eu_561_2006", "bai2023_simplified")


class HosActivity(StrEnum):
    """Duty classes the rules distinguish."""

    DRIVE = "drive"
    CHARGE = "charge"
    HANDLING = "handling"  # plugging in/out: work, never break credit (EU)
    QUEUE = "queue"
    SERVICE = "service"  # loading or unloading
    WINDOW_WAIT = "window_wait"
    IDLE = "idle"  # waiting for requests, or any unaccounted on-duty time
    REST_BREAK = "rest_break"
    REST_DAILY = "rest_daily"


OFF_DUTY = frozenset({HosActivity.REST_BREAK, HosActivity.REST_DAILY})


class HosInfeasible(ValueError):
    """No rest can restore a driving allowance (e.g. Bai's per-trip cap)."""


@dataclass(frozen=True)
class HosRegime:
    """Static parameters of one regime, plus the configured EU options."""

    name: str
    driving_before_break: float
    break_hours: float
    split_first_hours: float | None
    split_second_hours: float | None
    shift_driving_hours: float
    extended_shift_driving_hours: float | None
    extended_shifts_allowed: int
    shift_window_hours: float | None
    reduced_shift_window_hours: float | None
    daily_rest_hours: float | None
    reduced_daily_rest_hours: float | None
    reduced_rests_allowed: int
    all_nondriving_counts: bool
    charging_counts_as_break: bool
    queue_wait_counts_as_break: bool

    @classmethod
    def build(
        cls,
        name: str,
        *,
        charging_counts_as_break: bool = True,
        queue_wait_counts_as_break: bool = False,
        eu_split_break: bool = True,
        eu_extended_daily_driving: bool = False,
        eu_reduced_daily_rest: bool = False,
    ) -> HosRegime:
        if name == "us_fmcsa_property":
            return cls(
                name, 8.0, 0.5, None, None, 11.0, None, 0, 14.0, None, 10.0, None, 0,
                all_nondriving_counts=True,
                charging_counts_as_break=True,
                queue_wait_counts_as_break=True,
            )
        if name == "eu_561_2006":
            return cls(
                name, 4.5, 0.75,
                0.25 if eu_split_break else None,
                0.5 if eu_split_break else None,
                9.0,
                10.0 if eu_extended_daily_driving else None,
                2 if eu_extended_daily_driving else 0,
                13.0,
                15.0 if eu_reduced_daily_rest else None,
                11.0,
                9.0 if eu_reduced_daily_rest else None,
                3 if eu_reduced_daily_rest else 0,
                all_nondriving_counts=False,
                charging_counts_as_break=bool(charging_counts_as_break),
                queue_wait_counts_as_break=bool(queue_wait_counts_as_break),
            )
        if name == "bai2023_simplified":
            return cls(
                name, 4.5, 0.75, None, None, 9.0, None, 0, None, None, None, None, 0,
                all_nondriving_counts=True,
                charging_counts_as_break=True,
                queue_wait_counts_as_break=True,
            )
        raise ValueError(f"unknown hours_of_service.regime {name!r}; expected one of {REGIMES}")

    def earns_break_credit(self, activity: HosActivity) -> bool:
        """Whether time in ``activity`` extends a contiguous break."""
        if activity is HosActivity.DRIVE:
            return False
        if activity in OFF_DUTY:
            return True
        if self.all_nondriving_counts:
            return True
        if activity is HosActivity.CHARGE:
            return self.charging_counts_as_break
        if activity is HosActivity.QUEUE:
            return self.queue_wait_counts_as_break
        # EU: loading, handling, window waits and idle availability are not
        # recuperation, so they interrupt a break.
        return False


@dataclass(frozen=True)
class DriverClock:
    """Duty counters of one driver at ``time``."""

    time: float = 0.0
    driving_since_break: float = 0.0
    driving_in_shift: float = 0.0
    shift_start: float | None = 0.0  # None: off duty since the last daily reset
    break_streak: float = 0.0  # contiguous time earning break credit
    off_duty_streak: float = 0.0  # contiguous off-duty time
    split_first_taken: bool = False
    streak_earned_break: bool = False  # this streak already completed a break
    extended_shifts_used: int = 0
    reduced_rests_used: int = 0
    # A reset was earned by a reduced (9 h) rest; it only counts as reduced if
    # the rest then ends before the full 11 h.
    reduced_rest_pending: bool = False

    @classmethod
    def fresh(cls, time: float = 0.0) -> DriverClock:
        """On duty at ``time`` after a full daily rest (``initial_state: fresh``)."""
        return cls(time=float(time), shift_start=float(time))


def _shift_driving_limit(regime: HosRegime, clock: DriverClock) -> float:
    if (
        regime.extended_shift_driving_hours is not None
        and clock.extended_shifts_used < regime.extended_shifts_allowed
    ):
        return regime.extended_shift_driving_hours
    return regime.shift_driving_hours


def _reduced_rest_available(regime: HosRegime, clock: DriverClock) -> bool:
    return (
        regime.reduced_daily_rest_hours is not None
        and clock.reduced_rests_used < regime.reduced_rests_allowed
    )


def shift_window_hours(regime: HosRegime, clock: DriverClock) -> float | None:
    """Window length of the current shift (EU: 15 h if a reduced rest follows)."""
    if regime.shift_window_hours is None:
        return None
    if _reduced_rest_available(regime, clock):
        return regime.reduced_shift_window_hours
    return regime.shift_window_hours


def rest_hours(regime: HosRegime, clock: DriverClock, kind: str) -> float:
    """Exact length of a ``REST_BREAK`` or ``REST_DAILY`` action."""
    if kind == "break":
        if regime.split_second_hours is not None and clock.split_first_taken:
            return regime.split_second_hours
        return regime.break_hours
    if kind == "daily":
        if regime.daily_rest_hours is None:
            raise HosInfeasible(f"{regime.name} has no daily rest")
        if _reduced_rest_available(regime, clock):
            return regime.reduced_daily_rest_hours
        return regime.daily_rest_hours
    raise ValueError(f"unknown rest kind {kind!r}")


def window_left(regime: HosRegime, clock: DriverClock) -> float:
    """Hours until driving must stop for the shift window (inf without one)."""
    window = shift_window_hours(regime, clock)
    if window is None:
        return math.inf
    if clock.shift_start is None:
        return window
    return clock.shift_start + window - clock.time


def driving_allowance(regime: HosRegime, clock: DriverClock) -> float:
    """Hours of driving permitted from now before some rest is required."""
    allowance = min(
        regime.driving_before_break - clock.driving_since_break,
        _shift_driving_limit(regime, clock) - clock.driving_in_shift,
        window_left(regime, clock),
    )
    return max(0.0, allowance)


def required_rest(regime: HosRegime, clock: DriverClock) -> tuple[str | None, float]:
    """The rest needed before driving may continue, as (kind, hours).

    ``(None, 0.0)`` while driving is allowed. A daily rest is required once the
    shift's driving or window is used up, else a break. Raises ``HosInfeasible``
    when the regime has no rest that restores driving (Bai's per-trip cap).
    """
    if driving_allowance(regime, clock) > TOLERANCE:
        return None, 0.0
    shift_spent = (
        clock.driving_in_shift >= _shift_driving_limit(regime, clock) - TOLERANCE
        or window_left(regime, clock) <= TOLERANCE
    )
    if shift_spent:
        if regime.daily_rest_hours is None:
            raise HosInfeasible(
                f"{regime.name}: the per-trip driving limit is exhausted and no rest resets it"
            )
        return "daily", rest_hours(regime, clock, "daily")
    return "break", rest_hours(regime, clock, "break")


def _close_streak(regime: HosRegime, clock: DriverClock) -> DriverClock:
    """Account for a break streak that just ended (EU split, first part)."""
    if clock.reduced_rest_pending:
        clock = replace(
            clock,
            reduced_rests_used=clock.reduced_rests_used + 1,
            reduced_rest_pending=False,
        )
    if (
        regime.split_first_hours is not None
        and not clock.split_first_taken
        and not clock.streak_earned_break
        and clock.break_streak >= regime.split_first_hours - TOLERANCE
    ):
        return replace(
            clock, split_first_taken=True, break_streak=0.0, streak_earned_break=False
        )
    return replace(clock, break_streak=0.0, streak_earned_break=False)


def _daily_reset(regime: HosRegime, clock: DriverClock) -> DriverClock:
    extended = clock.extended_shifts_used + int(
        clock.driving_in_shift > regime.shift_driving_hours + TOLERANCE
    )
    reduced_used = clock.reduced_rests_used
    pending = False
    if regime.reduced_daily_rest_hours is not None:
        # Art. 8(2): a rest is reduced when the part of it inside the 24 hours
        # after the previous rest is under 11 hours. A rest that starts after the
        # 13-hour mark is reduced outright; an earlier one is reduced only if it
        # ends before 11 hours, which is known when the rest ends.
        rest_start = clock.time - clock.off_duty_streak
        late = (
            clock.shift_start is not None
            and rest_start > clock.shift_start + regime.shift_window_hours + TOLERANCE
        )
        if late:
            reduced_used += 1
        else:
            pending = clock.off_duty_streak < regime.daily_rest_hours - TOLERANCE
    return replace(
        clock,
        driving_since_break=0.0,
        driving_in_shift=0.0,
        shift_start=None,
        split_first_taken=False,
        extended_shifts_used=extended,
        reduced_rests_used=reduced_used,
        reduced_rest_pending=pending,
    )


def apply(
    regime: HosRegime, clock: DriverClock, activity: HosActivity, hours: float
) -> DriverClock:
    """Return the clock after ``hours`` of ``activity`` starting at ``clock.time``."""
    activity = HosActivity(activity)
    hours = float(hours)
    if not math.isfinite(hours) or hours < 0.0:
        raise ValueError("activity hours must be finite and non-negative")
    if hours == 0.0:
        return clock
    if activity not in OFF_DUTY and clock.shift_start is None:
        clock = replace(clock, shift_start=clock.time)
    end = clock.time + hours

    if activity is HosActivity.DRIVE:
        clock = _close_streak(regime, clock)
        return replace(
            clock,
            time=end,
            driving_since_break=clock.driving_since_break + hours,
            driving_in_shift=clock.driving_in_shift + hours,
            off_duty_streak=0.0,
        )

    if not regime.earns_break_credit(activity):
        clock = _close_streak(regime, clock)
        return replace(clock, time=end, off_duty_streak=0.0)

    streak = clock.break_streak + hours
    off_duty = clock.off_duty_streak + hours if activity in OFF_DUTY else 0.0
    if activity not in OFF_DUTY and clock.reduced_rest_pending:
        # On-duty time that still earns break credit ends the rest period.
        clock = replace(
            clock,
            reduced_rests_used=clock.reduced_rests_used + 1,
            reduced_rest_pending=False,
        )
    clock = replace(clock, time=end, break_streak=streak, off_duty_streak=off_duty)
    completes = streak >= regime.break_hours - TOLERANCE or (
        regime.split_second_hours is not None
        and clock.split_first_taken
        and streak >= regime.split_second_hours - TOLERANCE
    )
    if completes:
        clock = replace(
            clock,
            driving_since_break=0.0,
            split_first_taken=False,
            streak_earned_break=True,
        )
    if regime.daily_rest_hours is not None:
        needed = (
            regime.reduced_daily_rest_hours
            if _reduced_rest_available(regime, clock)
            else regime.daily_rest_hours
        )
        if off_duty >= needed - TOLERANCE and (
            clock.shift_start is not None or clock.driving_in_shift > 0.0
        ):
            clock = _daily_reset(regime, clock)
        if (
            clock.reduced_rest_pending
            and off_duty >= regime.daily_rest_hours - TOLERANCE
        ):
            clock = replace(clock, reduced_rest_pending=False)
    return clock


def advance_to(
    regime: HosRegime, clock: DriverClock, time: float, activity: HosActivity
) -> DriverClock:
    """Fill the gap up to ``time`` with ``activity`` (never moves backwards)."""
    gap = float(time) - clock.time
    if gap < -TOLERANCE:
        raise ValueError(f"driver clock at {clock.time} cannot move back to {time}")
    return apply(regime, clock, activity, max(0.0, gap))


def insert_rests(
    regime: HosRegime, clock: DriverClock, driving_hours: float, *, max_segments: int = 1000
) -> tuple[list[tuple[HosActivity, float]], DriverClock]:
    """Split one leg into driving and rests taken as late as possible.

    Goel (2009, 2010)-style placement: drive until the allowance is used up,
    then take exactly the required break or daily rest, and repeat. Returns the
    segments and the clock at arrival.
    """
    remaining = float(driving_hours)
    if not math.isfinite(remaining) or remaining < 0.0:
        raise ValueError("driving_hours must be finite and non-negative")
    segments: list[tuple[HosActivity, float]] = []
    while remaining > TOLERANCE:
        if len(segments) >= max_segments:
            raise RuntimeError("rest insertion did not converge")
        allowance = driving_allowance(regime, clock)
        if allowance >= remaining - TOLERANCE:
            segments.append((HosActivity.DRIVE, remaining))
            clock = apply(regime, clock, HosActivity.DRIVE, remaining)
            break
        if allowance > TOLERANCE:
            segments.append((HosActivity.DRIVE, allowance))
            clock = apply(regime, clock, HosActivity.DRIVE, allowance)
            remaining -= allowance
        kind, hours = required_rest(regime, clock)
        activity = HosActivity.REST_DAILY if kind == "daily" else HosActivity.REST_BREAK
        segments.append((activity, hours))
        clock = apply(regime, clock, activity, hours)
    return segments, clock


def can_drive(regime: HosRegime, clock: DriverClock, driving_hours: float) -> bool:
    """Whether a leg of ``driving_hours`` fits the current allowance (stops-only)."""
    return float(driving_hours) <= driving_allowance(regime, clock) + TOLERANCE
