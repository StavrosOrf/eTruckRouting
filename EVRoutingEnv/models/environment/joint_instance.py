"""Generation of fleet-owned customer tasks for joint routing instances."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from EVRoutingEnv.models.core.customer import CustomerTask, FleetTaskRegistry
from EVRoutingEnv.models.environment.route_certificate import (
    NoFeasibleCertificate,
)
from EVRoutingEnv.models.environment.route_walk import RouteWalkTopology


GENERATOR_VERSION = "route-walk-v2-single-charger"


@dataclass(frozen=True)
class JointRoutingInstance:
    """Static customer/depot data shared by all trucks in an episode."""

    depot_node: int
    tasks: tuple[CustomerTask, ...]
    witness_owners: tuple[int, ...] = ()
    witness_routes: tuple = ()
    generation_attempts_used: int = 0

    def create_registry(self, *, sequential: bool = False) -> FleetTaskRegistry:
        # Tasks are mutable during an episode, so return fresh copies.
        return FleetTaskRegistry(
            (
                CustomerTask(
                    task_id=task.task_id,
                    node_id=task.node_id,
                    demand=task.demand,
                    base_service_time=task.base_service_time,
                    earliest_service=task.earliest_service,
                    latest_service=task.latest_service,
                    preassigned_to=task.preassigned_to,
                    sequence_position=task.sequence_position,
                )
                for task in self.tasks
            ),
            sequential=sequential,
        )


def generate_joint_routing_instance(
    transport_graph,
    charging_nodes,
    rng,
    *,
    num_customers,
    num_trucks,
    battery_capacity,
    payload_capacity,
    min_customer_demand,
    max_customer_demand,
    base_service_time,
    time_window_config=None,
    preassign=False,
    energy_multiplier=1.0,
    sequential=False,
    initial_batteries=None,
    charge_transition=None,
    min_hop_energy_kwh=0.0,
    max_hop_energy_kwh=math.inf,
    min_depot_energy_kwh=0.0,
    max_depot_energy_kwh=math.inf,
    generation_attempts=64,
    certify=None,
) -> JointRoutingInstance:
    """Sample customers along battery-safe closed walks, retaining their witness.

    The environment supplies a certifier that replays the configured event
    engine (windows, queues, initial SoC, charging actions, stochastic seed,
    horizon and action budget). Standalone callers receive an energy/capacity
    certificate under full-battery charging, without an execution-time claim.
    Each depot/customer-to-customer/depot segment visits at most one charger.
    Cached map lookups and bounded proposals replace charger label searches.
    No constraint is relaxed on failure. Bounded construction failure raises clearly;
    it is not a mathematical proof that the configuration is infeasible.
    """
    _validate_inputs(
        num_customers=num_customers,
        num_trucks=num_trucks,
        battery_capacity=battery_capacity,
        payload_capacity=payload_capacity,
        min_customer_demand=min_customer_demand,
        max_customer_demand=max_customer_demand,
        base_service_time=base_service_time,
    )
    if (
        isinstance(generation_attempts, bool)
        or not isinstance(generation_attempts, int)
        or generation_attempts < 1
    ):
        raise ValueError("generation_attempts must be a positive integer")
    if (
        not math.isfinite(min_hop_energy_kwh)
        or min_hop_energy_kwh < 0
        or math.isnan(max_hop_energy_kwh)
        or max_hop_energy_kwh < min_hop_energy_kwh
    ):
        raise ValueError("hop energy bounds must satisfy 0 <= min <= max")
    if (
        not math.isfinite(min_depot_energy_kwh)
        or min_depot_energy_kwh < 0
        or math.isnan(max_depot_energy_kwh)
        or max_depot_energy_kwh < min_depot_energy_kwh
    ):
        raise ValueError("depot energy bounds must satisfy 0 <= min <= max")
    initial_batteries = (
        [battery_capacity] * num_trucks
        if initial_batteries is None
        else list(initial_batteries)
    )
    if len(initial_batteries) != num_trucks or any(
        not math.isfinite(value) or not 0 <= value <= battery_capacity
        for value in initial_batteries
    ):
        raise ValueError("initial_batteries must contain one valid kWh value per truck")
    topology = RouteWalkTopology.for_graph(
        transport_graph,
        charging_nodes,
        battery_capacity,
        energy_multiplier,
    )
    if len(topology.customers) < num_customers + 1:
        raise ValueError("road graph has too few eligible depot/customer nodes")
    _generate_time_windows(rng, count=0, config=time_window_config)
    last_error = "no candidate"
    for attempt in range(1, generation_attempts + 1):
        try:
            depot, visits, routes = topology.walk(
                rng,
                num_customers=num_customers,
                num_trucks=num_trucks,
                initial_batteries=initial_batteries,
                charge_transition=charge_transition,
                min_hop_energy_kwh=min_hop_energy_kwh,
                max_hop_energy_kwh=max_hop_energy_kwh,
                min_depot_energy_kwh=min_depot_energy_kwh,
                max_depot_energy_kwh=max_depot_energy_kwh,
            )
            # Flexible-mode IDs must not reveal the hidden visit order.
            # Sequential mode exposes its prescribed order by definition and
            # retains task IDs in that order for existing consumers.
            if not sequential:
                visits = [visits[int(i)] for i in rng.permutation(num_customers)]
            owners = [owner for _, owner, _ in visits]
            demands = _capacity_feasible_demands(
                rng,
                count=num_customers,
                num_trucks=num_trucks,
                payload_capacity=payload_capacity,
                min_demand=min_customer_demand,
                max_demand=max_customer_demand,
                assignment=owners,
            )
            windows = _generate_time_windows(
                rng, count=num_customers, config=time_window_config
            )
            tasks = tuple(
                CustomerTask(
                    task_id=i,
                    node_id=node,
                    demand=demands[i],
                    base_service_time=base_service_time,
                    earliest_service=windows[i][0],
                    latest_service=windows[i][1],
                    preassigned_to=owner if preassign or sequential else None,
                    sequence_position=position if sequential else None,
                )
                for i, (node, owner, position) in enumerate(visits)
            )
            instance = JointRoutingInstance(
                depot_node=depot,
                tasks=tasks,
                witness_owners=tuple(owners),
                witness_routes=routes,
                generation_attempts_used=attempt,
            )
            if certify is not None:
                certify(instance)
            return instance
        except NoFeasibleCertificate as error:
            last_error = str(error)
    raise ValueError(
        f"could not certify a solvable instance after {generation_attempts} attempts: "
        f"{last_error}. Check battery/initial SoC, payload, hop-energy bounds, "
        "depot-energy ring, the one-charger-per-segment limit, time windows, hours of "
        "service, public-queue waits, max_time and "
        "max_episode_steps; no constraints were relaxed."
    )


def _preassign_to_trucks(
    tasks: tuple[CustomerTask, ...],
    *,
    num_trucks: int,
    payload_capacity: float,
    assignment: list[int] | None = None,
) -> tuple[CustomerTask, ...]:
    """Keep the generator's feasible partition; pack explicit tasks exactly."""
    ordered = sorted(range(len(tasks)), key=lambda i: -tasks[i].demand)
    if assignment is None:
        assignment = [-1] * len(tasks)
        remaining = [payload_capacity] * num_trucks
        failed_states: set[tuple[int, tuple[float, ...]]] = set()

        def place(position: int) -> bool:
            if position == len(ordered):
                return True
            key = (position, tuple(sorted(remaining)))
            if key in failed_states:
                return False
            index = ordered[position]
            demand = tasks[index].demand
            seen_capacities: set[float] = set()
            for owner in sorted(range(num_trucks), key=lambda i: -remaining[i]):
                capacity = remaining[owner]
                if capacity in seen_capacities or demand > capacity + 1e-9:
                    continue
                seen_capacities.add(capacity)
                remaining[owner] -= demand
                assignment[index] = owner
                if place(position + 1):
                    return True
                remaining[owner] = capacity
            failed_states.add(key)
            return False

        if not place(0):
            raise ValueError(
                "customer demands cannot be assigned within truck payloads"
            )
    if len(assignment) != len(tasks) or any(
        owner < 0 or owner >= num_trucks for owner in assignment
    ):
        raise ValueError("invalid capacity assignment")
    if any(
        sum(
            task.demand
            for task, owner in zip(tasks, assignment, strict=True)
            if owner == truck
        )
        > payload_capacity + 1e-9
        for truck in range(num_trucks)
    ):
        raise ValueError("customer assignment exceeds truck payload capacity")
    return tuple(
        CustomerTask(
            task_id=task.task_id,
            node_id=task.node_id,
            demand=task.demand,
            base_service_time=task.base_service_time,
            earliest_service=task.earliest_service,
            latest_service=task.latest_service,
            preassigned_to=owner,
        )
        for task, owner in zip(tasks, assignment, strict=True)
    )


def _capacity_feasible_demands(
    rng: np.random.Generator,
    *,
    count: int,
    num_trucks: int,
    payload_capacity: float,
    min_demand: float,
    max_demand: float,
    return_assignment: bool = False,
    assignment: list[int] | None = None,
) -> list[float] | tuple[list[float], list[int]]:
    """Sample demands with a guaranteed feasible multi-bin partition."""
    total_capacity = num_trucks * payload_capacity
    if count * min_demand > total_capacity + 1e-9:
        raise ValueError("minimum customer demand exceeds total fleet capacity")

    if assignment is None:
        assignments = np.arange(count, dtype=int) % num_trucks
        rng.shuffle(assignments)
    else:
        assignments = np.asarray(assignment, dtype=int)
    remaining_capacity = np.full(num_trucks, payload_capacity, dtype=float)
    remaining_tasks = np.bincount(assignments, minlength=num_trucks)
    demands: list[float] = [0.0] * count
    for index, truck_id in enumerate(assignments):
        remaining_tasks[truck_id] -= 1
        maximum_here = min(
            max_demand,
            remaining_capacity[truck_id] - remaining_tasks[truck_id] * min_demand,
        )
        if maximum_here < min_demand - 1e-9:
            raise ValueError("could not allocate capacity-feasible customer demands")
        demand = float(rng.uniform(min_demand, maximum_here))
        demands[index] = demand
        remaining_capacity[truck_id] -= demand
    return (demands, assignments.tolist()) if return_assignment else demands


def _generate_time_windows(
    rng: np.random.Generator,
    *,
    count: int,
    config: dict | None,
) -> list[tuple[float, float]]:
    """Generate the optional controlled hard-time-window variant."""
    settings = {} if config is None else dict(config)
    enabled = settings.get("enabled", False)
    if not isinstance(enabled, bool):
        raise TypeError("time_windows.enabled must be boolean")
    if not enabled:
        return [(0.0, math.inf)] * count

    earliest_min = float(settings.get("earliest_min", 0.0))
    earliest_max = float(settings.get("earliest_max", 0.0))
    width_min = float(settings.get("window_width_min", 1.0))
    width_max = float(settings.get("window_width_max", width_min))
    if (
        not math.isfinite(earliest_min)
        or not math.isfinite(earliest_max)
        or earliest_min < 0.0
        or earliest_max < earliest_min
    ):
        raise ValueError("time-window earliest bounds must satisfy 0 <= min <= max")
    if (
        not math.isfinite(width_min)
        or not math.isfinite(width_max)
        or width_min <= 0.0
        or width_max < width_min
    ):
        raise ValueError("time-window widths must satisfy 0 < min <= max")

    earliest_values = rng.uniform(earliest_min, earliest_max, size=count)
    widths = rng.uniform(width_min, width_max, size=count)
    return [
        (float(earliest), float(earliest + width))
        for earliest, width in zip(earliest_values, widths, strict=True)
    ]


def _validate_inputs(
    *,
    num_customers: int,
    num_trucks: int,
    battery_capacity: float,
    payload_capacity: float,
    min_customer_demand: float,
    max_customer_demand: float,
    base_service_time: float,
) -> None:
    if num_customers <= 0 or num_trucks <= 0:
        raise ValueError("num_customers and num_trucks must be positive")
    if battery_capacity <= 0.0 or payload_capacity <= 0.0:
        raise ValueError("battery and payload capacities must be positive")
    if min_customer_demand <= 0.0:
        raise ValueError("minimum customer demand must be positive")
    if max_customer_demand < min_customer_demand:
        raise ValueError("maximum demand is below minimum demand")
    if max_customer_demand > payload_capacity:
        raise ValueError("one customer demand may exceed truck payload capacity")
    tasks_per_truck = math.ceil(num_customers / num_trucks)
    if tasks_per_truck * min_customer_demand > payload_capacity + 1e-9:
        raise ValueError("minimum demands cannot be partitioned across truck payloads")
    if base_service_time < 0.0:
        raise ValueError("base service time cannot be negative")
