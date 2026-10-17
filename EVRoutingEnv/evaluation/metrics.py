"""Operational episode metrics independent of shaped training reward."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class OperationalMetrics:
    """Feasibility-first metrics for one simulator episode."""

    success: bool
    termination_reason: str | None
    customers_total: int
    customers_served: int
    completed_fraction: float
    fleet_makespan: float | None
    total_operating_time: float
    total_travel_time: float
    total_charging_time: float
    total_queue_time: float
    total_time_window_waiting: float
    total_service_time: float
    total_distance: float
    total_energy_consumed: float
    total_energy_charged: float
    mean_terminal_soc: float
    minimum_terminal_soc: float
    vehicles_used: int
    charging_sessions: int
    invalid_actions: int

    def as_dict(self) -> dict:
        return asdict(self)


def extract_operational_metrics(env) -> OperationalMetrics:
    """Extract metrics from simulator state without consulting episode reward."""
    trucks = list(env.trucks)
    tasks = env.task_registry.tasks()
    customers_total = len(tasks)
    customers_served = sum(task.is_served for task in tasks)

    completed_fraction = customers_served / customers_total if customers_total else 1.0
    success = bool(
        customers_served == customers_total
        and all(truck.is_complete for truck in trucks)
        and not any(truck.failed for truck in trucks)
        and getattr(env, "_truncation_reason", None) is None
        and float(env.global_clock) <= float(getattr(env, "max_time", float("inf")))
    )
    terminal_socs = [truck.current_battery / truck.battery_capacity for truck in trucks]
    vehicles_used = sum(
        bool(
            truck.served_task_ids
            or truck.total_distance_traveled > 0.0
            or truck.num_charging_sessions > 0
            or truck.route_destination is not None
            or truck.charge_start_time is not None
        )
        for truck in trucks
    )

    # Truck counters commit at completion events. Include elapsed portions of
    # in-flight work in snapshots, especially when a horizon cuts an event off.
    # This is a read-only view: queued events and physical endpoint state remain
    # untouched, so repeated metric extraction cannot double-count work.
    partial = {
        "travel": 0.0,
        "charging": 0.0,
        "queue": 0.0,
        "window": 0.0,
        "service": 0.0,
    }
    for truck in trucks:
        state = env.truck_states[truck.truck_id]
        start = None
        kind = None
        if state == "routing":
            start, kind = truck.routing_start_time, "travel"
        elif state == "charging":
            start, kind = truck.charge_start_time, "charging"
        elif state == "unloading":
            start, kind = truck.unloading_start_time, "service"
        elif state == "waiting_to_charge":
            start, kind = env.waiting_start_times.get(truck.truck_id), "queue"
        elif state == "waiting_for_service":
            task = next(
                (
                    task
                    for task in tasks
                    if task.claimed_by == truck.truck_id and not task.is_served
                ),
                None,
            )
            if task is not None:
                # Arrival precedes the queued service-window-opening event.
                event = next(
                    (
                        event
                        for event in env.event_queue
                        if event.truck_id == truck.truck_id
                        and event.data.get("reason") == "service_window_open"
                    ),
                    None,
                )
                if event is not None:
                    start = event.time - float(event.data["wait_duration"])
                    kind = "window"
        if start is not None and kind is not None:
            partial[kind] += max(0.0, float(env.global_clock) - start)

    return OperationalMetrics(
        success=success,
        termination_reason=env.termination_reason,
        customers_total=customers_total,
        customers_served=customers_served,
        completed_fraction=completed_fraction,
        fleet_makespan=float(env.global_clock) if success else None,
        total_operating_time=sum(truck.total_time_elapsed for truck in trucks)
        + sum(partial.values()),
        total_travel_time=sum(truck.total_routing_time for truck in trucks)
        + partial["travel"],
        total_charging_time=sum(truck.total_charging_time for truck in trucks)
        + partial["charging"],
        total_queue_time=sum(truck.waiting_time for truck in trucks) + partial["queue"],
        total_time_window_waiting=sum(
            truck.time_window_waiting_time for truck in trucks
        )
        + partial["window"],
        total_service_time=sum(truck.total_unloading_time for truck in trucks)
        + partial["service"],
        total_distance=sum(truck.total_distance_traveled for truck in trucks),
        total_energy_consumed=sum(truck.total_energy_consumed for truck in trucks),
        total_energy_charged=sum(truck.total_energy_charged for truck in trucks),
        mean_terminal_soc=(
            sum(terminal_socs) / len(terminal_socs) if terminal_socs else 0.0
        ),
        minimum_terminal_soc=min(terminal_socs, default=0.0),
        vehicles_used=vehicles_used,
        charging_sessions=sum(truck.num_charging_sessions for truck in trucks),
        invalid_actions=int(env.invalid_action_count),
    )
