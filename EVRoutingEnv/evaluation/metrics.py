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
    if env.task_registry is not None:
        tasks = env.task_registry.tasks()
        customers_total = len(tasks)
        customers_served = sum(task.is_served for task in tasks)
    else:
        customers_total = sum(len(truck.delivery_sequence) - 1 for truck in trucks)
        customers_served = sum(
            (len(truck.delivery_sequence) - 1) - len(truck.get_remaining_deliveries())
            for truck in trucks
        )

    completed_fraction = (
        customers_served / customers_total if customers_total else 1.0
    )
    success = bool(
        customers_served == customers_total
        and all(truck.is_complete for truck in trucks)
        and not any(truck.failed for truck in trucks)
    )
    terminal_socs = [
        truck.current_battery / truck.battery_capacity for truck in trucks
    ]
    vehicles_used = sum(
        bool(
            truck.served_task_ids
            or truck.total_distance_traveled > 0.0
            or truck.num_charging_sessions > 0
        )
        for truck in trucks
    )

    return OperationalMetrics(
        success=success,
        termination_reason=env.termination_reason,
        customers_total=customers_total,
        customers_served=customers_served,
        completed_fraction=completed_fraction,
        fleet_makespan=float(env.global_clock) if success else None,
        total_operating_time=sum(truck.total_time_elapsed for truck in trucks),
        total_travel_time=sum(truck.total_routing_time for truck in trucks),
        total_charging_time=sum(truck.total_charging_time for truck in trucks),
        total_queue_time=sum(truck.waiting_time for truck in trucks),
        total_time_window_waiting=sum(
            truck.time_window_waiting_time for truck in trucks
        ),
        total_service_time=sum(truck.total_unloading_time for truck in trucks),
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
