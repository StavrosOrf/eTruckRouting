"""Per-truck hours-of-service state for the event-driven environment.

The tracker owns one ``DriverClock`` per truck and applies each activity when
it ends, so a clock never reflects time that has not passed. Driving legs are
split into driving and rests when they are dispatched (the whole leg is
realized at departure), and committed on arrival. Every rest is written to the
truck's event log as ``HOS_REST`` so that ``hos_audit`` can rebuild the duty
timeline independently.
"""

from __future__ import annotations

import math

from EVRoutingEnv.models.core.hours_of_service import (
    TOLERANCE,
    DriverClock,
    HosActivity,
    advance_to,
    apply,
    driving_allowance,
    insert_rests,
    required_rest,
    rest_hours,
    window_left,
)
from EVRoutingEnv.models.environment.extension_config import HoursOfServiceConfig


# TrafficSimulator clips realized leg time to [0.85, 2.5] x nominal.
WORST_CASE_TRAVEL_FACTOR = 2.5


class HosTracker:
    def __init__(self, config: HoursOfServiceConfig, num_trucks: int):
        self.config = config
        self.regime = config.regime_model()
        self.num_trucks = int(num_trucks)
        self.reset()

    def reset(self) -> None:
        self._clocks = {tid: DriverClock.fresh(0.0) for tid in range(self.num_trucks)}
        self.unplanned_breaks = 0
        self.unplanned_violations = 0
        self.rest_hours_total = {tid: 0.0 for tid in range(self.num_trucks)}

    # -- queries -------------------------------------------------------------

    def clock(self, truck_id: int) -> DriverClock:
        return self._clocks[int(truck_id)]

    def planned_leg_hours(self, env, truck, target_node: int) -> float:
        """Driving hours the mask assumes for a leg (section 6.5 time guard)."""
        try:
            hours = float(env.transport_graph.get_time_distance(int(truck.current_node), int(target_node)))
        except (KeyError, ValueError):
            return math.inf
        if not math.isfinite(hours):
            return math.inf
        if self.config.time_guard == "worst_case":
            return hours * WORST_CASE_TRAVEL_FACTOR
        return hours

    def feature_row(self, truck) -> list[float]:
        clock = self.clock(truck.truck_id)
        regime = self.regime
        try:
            kind, _ = required_rest(regime, clock)
        except ValueError:
            kind = "blocked"
        return [
            max(0.0, regime.driving_before_break - clock.driving_since_break),
            max(0.0, regime.shift_driving_hours - clock.driving_in_shift),
            min(window_left(regime, clock), 1e6),
            clock.break_streak,
            float(clock.split_first_taken),
            float(kind is not None),
        ]

    def rest_action_hours(self, truck_id: int, kind: str) -> float:
        return rest_hours(self.regime, self.clock(truck_id), kind)

    # -- updates -------------------------------------------------------------

    def record(self, truck, activity: HosActivity, start: float, end: float) -> None:
        """Apply ``activity`` over [start, end], filling any gap with idle duty."""
        tid = int(truck.truck_id)
        clock = advance_to(self.regime, self._clocks[tid], start, HosActivity.IDLE)
        self._clocks[tid] = apply(self.regime, clock, activity, max(0.0, end - start))

    def record_charge(self, truck, start: float, end: float) -> None:
        """Handling overhead is work (no break credit); the rest is charging."""
        overhead = min(self.config.handling_overhead_hours, max(0.0, end - start))
        if overhead > 0.0:
            self.record(truck, HosActivity.HANDLING, start, start + overhead)
        self.record(truck, HosActivity.CHARGE, start + overhead, end)

    def record_rest(self, truck, kind: str, start: float, end: float, *, roadside: bool) -> None:
        activity = HosActivity.REST_DAILY if kind == "daily" else HosActivity.REST_BREAK
        self.record(truck, activity, start, end)
        self.rest_hours_total[int(truck.truck_id)] += end - start
        truck._record_event(
            event_type="HOS_REST",
            timestamp=float(end),
            details={"kind": kind, "start": float(start), "end": float(end), "roadside": roadside},
        )

    def plan_leg(self, truck, driving_hours: float) -> list[tuple[str, float]]:
        """Split a realized leg into driving and rests, as late as possible.

        ``anywhere`` (Goel 2009/2010) always inserts. ``stops_only`` only inserts
        when the realized driving overruns the allowance the mask granted on
        nominal time: an unplanned roadside break, counted and reported.
        """
        clock = self.clock(truck.truck_id)
        if (
            self.config.rest_locations == "stops_only"
            and driving_hours <= driving_allowance(self.regime, clock) + TOLERANCE
        ):
            return [(HosActivity.DRIVE.value, float(driving_hours))]
        try:
            segments, _ = insert_rests(self.regime, clock, driving_hours)
        except ValueError:
            # Bai's per-trip cap has no reset: drive on and report the violation.
            self.unplanned_violations += 1
            return [(HosActivity.DRIVE.value, float(driving_hours))]
        if self.config.rest_locations == "stops_only":
            self.unplanned_breaks += sum(
                activity is not HosActivity.DRIVE for activity, _ in segments
            )
        return [(activity.value, float(hours)) for activity, hours in segments]

    def commit_leg(self, truck, departure: float, segments) -> None:
        """Apply a planned leg on arrival and log its roadside rests."""
        time = float(departure)
        for activity, hours in segments:
            activity = HosActivity(activity)
            if activity is HosActivity.DRIVE:
                self.record(truck, activity, time, time + hours)
            else:
                kind = "daily" if activity is HosActivity.REST_DAILY else "break"
                self.record_rest(truck, kind, time, time + hours, roadside=True)
            time += hours

    @staticmethod
    def rest_hours_in(segments) -> float:
        return sum(hours for activity, hours in segments if activity != HosActivity.DRIVE.value)

    @staticmethod
    def elapsed_split(segments, elapsed: float) -> tuple[float, float]:
        """(driving, resting) hours within the first ``elapsed`` hours of a leg."""
        driving = resting = 0.0
        remaining = max(0.0, float(elapsed))
        for activity, hours in segments:
            used = min(hours, remaining)
            if activity == HosActivity.DRIVE.value:
                driving += used
            else:
                resting += used
            remaining -= used
            if remaining <= 0.0:
                break
        return driving, resting
