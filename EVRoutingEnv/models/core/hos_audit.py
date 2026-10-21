"""Independent hours-of-service compliance audit.

This is the second implementation the plan requires next to ``DriverClock``.
It shares no rule code with :mod:`hours_of_service`: the activity table, break
detection and shift boundaries are re-derived here from a timeline of
``(start, end, activity)`` intervals, and the timeline itself is rebuilt from
``Truck.export_event_log()`` rather than from the clock updates. A sweep over
maximal contiguous runs replaces the clock's incremental counters.
"""

from __future__ import annotations

from dataclasses import dataclass

from EVRoutingEnv.models.core.hours_of_service import HosActivity


EPS = 1e-9

# Plan section 6.2, kept literal and separate from HosRegime on purpose.
_LIMITS = {
    "us_fmcsa_property": {
        "drive_before_break": 8.0, "break": 0.5, "split": None,
        "shift_drive": 11.0, "window": 14.0, "daily_rest": 10.0,
    },
    "eu_561_2006": {
        "drive_before_break": 4.5, "break": 0.75, "split": (0.25, 0.5),
        "shift_drive": 9.0, "window": 13.0, "daily_rest": 11.0,
    },
    "bai2023_simplified": {
        "drive_before_break": 4.5, "break": 0.75, "split": None,
        "shift_drive": 9.0, "window": None, "daily_rest": None,
    },
}


@dataclass(frozen=True)
class AuditSettings:
    regime: str
    charging_counts_as_break: bool = True
    queue_wait_counts_as_break: bool = False
    eu_split_break: bool = True
    eu_extended_daily_driving: bool = False
    eu_reduced_daily_rest: bool = False


@dataclass(frozen=True)
class HosViolation:
    time: float
    rule: str
    detail: str


def _credit(settings: AuditSettings, activity: HosActivity) -> bool:
    """Plan section 6.3: does this activity count toward a break?"""
    if activity is HosActivity.DRIVE:
        return False
    if settings.regime in {"us_fmcsa_property", "bai2023_simplified"}:
        return True  # any non-driving time interrupts US driving; Bai: any stay
    return {
        HosActivity.REST_BREAK: True,
        HosActivity.REST_DAILY: True,
        HosActivity.CHARGE: settings.charging_counts_as_break,
        HosActivity.QUEUE: settings.queue_wait_counts_as_break,
    }.get(activity, False)


def _off_duty(activity: HosActivity) -> bool:
    return activity in (HosActivity.REST_BREAK, HosActivity.REST_DAILY)


def normalize(intervals, horizon_start: float = 0.0) -> list[tuple[float, float, HosActivity]]:
    """Sort, reject overlaps, and fill gaps with idle on-duty time."""
    ordered = sorted(
        ((float(s), float(e), HosActivity(a)) for s, e, a in intervals if e - s > EPS),
        key=lambda item: (item[0], item[1]),
    )
    timeline: list[tuple[float, float, HosActivity]] = []
    cursor = float(horizon_start)
    for start, end, activity in ordered:
        if start < cursor - 1e-7:
            raise ValueError(f"overlapping activities at t={start} (cursor {cursor})")
        if start > cursor + EPS:
            timeline.append((cursor, start, HosActivity.IDLE))
        timeline.append((max(start, cursor), end, activity))
        cursor = end
    return timeline


def _runs(timeline, predicate):
    """Maximal contiguous runs of intervals satisfying ``predicate``."""
    runs: list[list[float]] = []
    for start, end, activity in timeline:
        if not predicate(activity):
            continue
        if runs and abs(runs[-1][1] - start) <= EPS:
            runs[-1][1] = end
        else:
            runs.append([start, end])
    return runs


def hos_audit(intervals, settings: AuditSettings) -> list[HosViolation]:
    """Return every driving instant that breaks the configured regime."""
    limits = _LIMITS[settings.regime]
    timeline = normalize(intervals)
    eu = settings.regime == "eu_561_2006"
    split = limits["split"] if (eu and settings.eu_split_break) else None

    # Break completions: the instant a qualifying run (or EU split) is earned.
    completions: list[float] = []
    first_part = False
    for start, end in _runs(timeline, lambda a: _credit(settings, a)):
        length = end - start
        if split is not None and first_part and length >= split[1] - EPS:
            completions.append(start + split[1])
            first_part = False
        elif length >= limits["break"] - EPS:
            completions.append(start + limits["break"])
            first_part = False
        elif split is not None and length >= split[0] - EPS:
            first_part = True

    # Daily rests and shift starts, in time order.
    rest_ends: list[float] = []
    reduced_flags: list[bool] = []
    reduced_used = 0
    reduced_option = eu and settings.eu_reduced_daily_rest
    starts = [0.0]  # initial_state fresh: on duty at t = 0

    def first_on_duty(after: float) -> float:
        for start, end, activity in timeline:
            if end > after + EPS and not _off_duty(activity):
                return max(start, after)
        return after

    if limits["daily_rest"] is not None:
        for start, end in _runs(timeline, _off_duty):
            length = end - start
            needed = 9.0 if reduced_option and reduced_used < 3 else limits["daily_rest"]
            if length < needed - EPS:
                continue
            # Art. 8(2): reduced if under 11 h falls in the 24 h after the last rest.
            reduced = reduced_option and (
                start > starts[-1] + limits["window"] + EPS
                or length < limits["daily_rest"] - EPS
            )
            reduced_used += int(reduced)
            reduced_flags.append(reduced)
            rest_ends.append(end)
            starts.append(first_on_duty(end))
    violations: list[HosViolation] = []
    drives = [(s, e) for s, e, a in timeline if a is HosActivity.DRIVE]
    if not drives:
        return violations

    def shift_of(time: float) -> int:
        return sum(1 for value in rest_ends if value <= time + EPS)

    shift_driving: dict[int, float] = {}
    extended_used = 0
    extended_shifts: set[int] = set()
    total = 0.0
    for start, end in drives:
        hours = end - start
        index = shift_of(start)
        shift_driving[index] = shift_driving.get(index, 0.0) + hours
        total += hours
        # Driving since the latest completed break.
        last = max((c for c in completions if c <= start + EPS), default=0.0)
        since = sum(
            max(0.0, min(e, end) - max(s, last)) for s, e in drives if s < end and e > last
        )
        if since > limits["drive_before_break"] + EPS:
            violations.append(HosViolation(end, "break", f"{since:.6f} h without a break"))
        shift_limit = limits["shift_drive"]
        if eu and settings.eu_extended_daily_driving and (
            index in extended_shifts or extended_used < 2
        ):
            shift_limit = 10.0
        if settings.regime == "bai2023_simplified":
            if total > shift_limit + EPS:
                violations.append(HosViolation(end, "trip_driving", f"{total:.6f} h driving"))
        elif shift_driving[index] > shift_limit + EPS:
            violations.append(
                HosViolation(end, "shift_driving", f"{shift_driving[index]:.6f} h in shift")
            )
        if eu and shift_driving[index] > 9.0 + EPS and index not in extended_shifts:
            extended_shifts.add(index)
            extended_used += 1
        window = limits["window"]
        if window is not None:
            open_shift = index == len(rest_ends)
            if (not open_shift and reduced_flags[index]) or (
                open_shift and reduced_option and reduced_used < 3
            ):
                window = 15.0
            if end > starts[index] + window + EPS:
                violations.append(HosViolation(end, "window", f"driving ends at {end:.6f}"))
    return violations


def intervals_from_event_log(log: dict, *, handling_overhead_hours: float = 0.0):
    """Rebuild duty intervals from ``Truck.export_event_log()``.

    Routing legs are driving minus any roadside rests logged inside them;
    charging starts with ``handling_overhead_hours`` of handling work.
    """
    events = log["events"]
    intervals: list[tuple[float, float, HosActivity]] = []
    rests: list[tuple[float, float, HosActivity]] = []
    open_at: dict[str, float] = {}
    for event in events:
        kind = event["event_type"]
        time = float(event["timestamp"])
        details = event.get("details") or {}
        if kind == "HOS_REST":
            activity = (
                HosActivity.REST_DAILY if details["kind"] == "daily" else HosActivity.REST_BREAK
            )
            rests.append((float(details["start"]), float(details["end"]), activity))
        elif kind.endswith("_START") and kind != "ROUTING_START":
            open_at[kind[: -len("_START")]] = time
        elif kind == "ROUTING_START":
            open_at["ROUTING"] = time
        elif kind.endswith("_END"):
            label = kind[: -len("_END")]
            start = open_at.pop(label, None)
            if start is None:
                continue
            if label == "ROUTING":
                intervals.append((start, time, HosActivity.DRIVE))
            elif label == "CHARGING":
                overhead = min(max(0.0, handling_overhead_hours), time - start)
                if overhead > 0.0:
                    intervals.append((start, start + overhead, HosActivity.HANDLING))
                intervals.append((start + overhead, time, HosActivity.CHARGE))
            elif label == "WAITING":
                intervals.append((start, time, HosActivity.QUEUE))
            elif label in {"UNLOADING", "LOADING"}:
                intervals.append((start, time, HosActivity.SERVICE))
            elif label == "IDLE":
                intervals.append((start, time, HosActivity.IDLE))
        elif kind == "TIME_WINDOW_WAIT":
            duration = float(details.get("wait_duration_hours", 0.0))
            intervals.append((time - duration, time, HosActivity.WINDOW_WAIT))
    # Roadside rests split the driving interval that contains them.
    ordered_rests = sorted(rests)
    result: list[tuple[float, float, HosActivity]] = []
    for start, end, activity in intervals:
        if activity is not HosActivity.DRIVE:
            result.append((start, end, activity))
            continue
        cursor = start
        for rest_start, rest_end, _ in ordered_rests:
            if rest_start >= cursor - EPS and rest_end <= end + EPS:
                if rest_start > cursor + EPS:
                    result.append((cursor, rest_start, HosActivity.DRIVE))
                cursor = rest_end
        if end > cursor + EPS:
            result.append((cursor, end, HosActivity.DRIVE))
    result.extend(rests)
    return result
