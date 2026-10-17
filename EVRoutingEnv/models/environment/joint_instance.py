"""Generation of fleet-owned customer tasks for joint routing instances."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from EVRoutingEnv.models.core.customer import CustomerTask, FleetTaskRegistry
from EVRoutingEnv.models.environment.energy_reachability import EnergyReachability
from EVRoutingEnv.models.environment.route_certificate import (
    NoFeasibleCertificate,
    construct_energy_routes,
)


@dataclass(frozen=True)
class JointRoutingInstance:
    """Static customer/depot data shared by all trucks in an episode."""

    depot_node: int
    tasks: tuple[CustomerTask, ...]
    witness_owners: tuple[int, ...] = ()
    witness_routes: tuple = ()

    def create_registry(self, *, sequential: bool = False) -> FleetTaskRegistry:
        # Tasks are mutable during an episode, so return fresh copies.
        return FleetTaskRegistry(
            (CustomerTask(
                task_id=task.task_id,
                node_id=task.node_id,
                demand=task.demand,
                base_service_time=task.base_service_time,
                earliest_service=task.earliest_service,
                latest_service=task.latest_service,
                preassigned_to=task.preassigned_to,
                sequence_position=task.sequence_position,
            ) for task in self.tasks),
            sequential=sequential,
        )


def generate_joint_routing_instance(
    transport_graph, charging_nodes, rng, *, num_customers, num_trucks,
    battery_capacity, payload_capacity, min_customer_demand, max_customer_demand,
    base_service_time, time_window_config=None, preassign=False,
    energy_multiplier=1.0, sequential=False, initial_batteries=None,
    charge_transition=None, min_hop_energy_kwh=0.0, max_hop_energy_kwh=math.inf,
    generation_attempts=64, certify=None,
) -> JointRoutingInstance:
    """Emit only candidates with a complete constructive fleet witness.

    The environment supplies a certifier that replays the configured event
    engine (windows, queues, initial SoC, charging actions, stochastic seed,
    horizon and action budget). Standalone callers receive an energy/capacity
    certificate under full-battery charging, without an execution-time claim.
    No constraint is relaxed on failure. Bounded search failure raises clearly;
    it is not a mathematical proof that the configuration is infeasible.
    """
    _validate_inputs(
        num_customers=num_customers, num_trucks=num_trucks,
        battery_capacity=battery_capacity, payload_capacity=payload_capacity,
        min_customer_demand=min_customer_demand, max_customer_demand=max_customer_demand,
        base_service_time=base_service_time,
    )
    if isinstance(generation_attempts, bool) or not isinstance(generation_attempts, int) or generation_attempts < 1:
        raise ValueError("generation_attempts must be a positive integer")
    if (not math.isfinite(min_hop_energy_kwh) or min_hop_energy_kwh < 0
            or math.isnan(max_hop_energy_kwh) or max_hop_energy_kwh < min_hop_energy_kwh):
        raise ValueError("hop energy bounds must satisfy 0 <= min <= max")
    initial_batteries = ([battery_capacity] * num_trucks if initial_batteries is None
                         else list(initial_batteries))
    if len(initial_batteries) != num_trucks or any(
        not math.isfinite(value) or not 0 <= value <= battery_capacity
        for value in initial_batteries
    ):
        raise ValueError("initial_batteries must contain one valid kWh value per truck")
    envelope = EnergyReachability.build(
        transport_graph, charging_nodes, battery_capacity, energy_multiplier,
    )
    last_error = "no candidate"
    for _ in range(generation_attempts):
        instance = _sample_joint_routing_instance(
            transport_graph, charging_nodes, rng, num_customers=num_customers,
            num_trucks=num_trucks, battery_capacity=battery_capacity,
            payload_capacity=payload_capacity, min_customer_demand=min_customer_demand,
            max_customer_demand=max_customer_demand, base_service_time=base_service_time,
            time_window_config=time_window_config, preassign=preassign,
            energy_multiplier=energy_multiplier, sequential=sequential,
            envelope=envelope,
        )
        try:
            routes = construct_energy_routes(
                instance, transport_graph, charging_nodes, num_trucks=num_trucks,
                battery_capacity=battery_capacity, initial_batteries=initial_batteries,
                energy_multiplier=energy_multiplier, charge_transition=charge_transition,
                min_hop_energy_kwh=min_hop_energy_kwh,
                max_hop_energy_kwh=max_hop_energy_kwh,
            )
            instance = replace(instance, witness_routes=routes)
            if certify is not None:
                certify(instance)
            return instance
        except NoFeasibleCertificate as error:
            last_error = str(error)
    raise ValueError(
        f"could not certify a solvable instance after {generation_attempts} attempts: "
        f"{last_error}. Check battery/initial SoC, payload, hop-energy bounds, "
        "time windows, max_time and max_episode_steps; no constraints were relaxed."
    )


def _sample_joint_routing_instance(
    transport_graph,
    charging_nodes: list[int],
    rng: np.random.Generator,
    *,
    num_customers: int,
    num_trucks: int,
    battery_capacity: float,
    payload_capacity: float,
    min_customer_demand: float,
    max_customer_demand: float,
    base_service_time: float,
    time_window_config: dict | None = None,
    preassign: bool = False,
    energy_multiplier: float = 1.0,
    sequential: bool = False,
    envelope: EnergyReachability | None = None,
) -> JointRoutingInstance:
    """Generate one feasible-capacity joint-routing instance.

    Every unique customer has an affordable approach AND exit on one battery,
    through chargers connected to the depot in both directions. The multiplier
    matches execution's stochastic reserve. This certifies individual energy
    reachability; a full capacity/time-feasible routing policy is still needed.
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

    graph = transport_graph.graph
    charging_set = {int(node) for node in charging_nodes}
    eligible_nodes = [
        int(node)
        for node in transport_graph.get_all_nodes()
        if int(node) not in charging_set
        and graph.out_degree(node) > 0
        and graph.in_degree(node) > 0
    ]
    if len(eligible_nodes) < num_customers + 1:
        raise ValueError("road graph has too few eligible depot/customer nodes")

    depot_candidates = eligible_nodes.copy()
    rng.shuffle(depot_candidates)

    envelope = envelope or EnergyReachability.build(
        transport_graph, charging_nodes, battery_capacity, energy_multiplier
    )
    selected_depot: int | None = None
    selected_customers: list[int] | None = None
    for depot in depot_candidates:
        feasible_customers = envelope.customers(depot, eligible_nodes)
        if len(feasible_customers) < num_customers:
            continue
        chosen = rng.choice(feasible_customers, size=num_customers, replace=False)
        selected_depot = int(depot)
        selected_customers = [int(node) for node in chosen]
        break

    if selected_depot is None or selected_customers is None:
        raise ValueError(
            "could not generate enough energy-safe customers from any depot"
        )

    demands, capacity_assignment = _capacity_feasible_demands(
        rng,
        count=num_customers,
        num_trucks=num_trucks,
        payload_capacity=payload_capacity,
        min_demand=min_customer_demand,
        max_demand=max_customer_demand,
        return_assignment=True,
    )
    time_windows = _generate_time_windows(
        rng,
        count=num_customers,
        config=time_window_config,
    )
    tasks = tuple(
        CustomerTask(
            task_id=task_id,
            node_id=node_id,
            demand=demand,
            base_service_time=base_service_time,
            earliest_service=earliest_service,
            latest_service=latest_service,
        )
        for task_id, (node_id, demand, (earliest_service, latest_service)) in enumerate(
            zip(selected_customers, demands, time_windows, strict=True)
        )
    )
    if preassign or sequential:
        tasks = _preassign_to_trucks(
            tasks,
            num_trucks=num_trucks,
            payload_capacity=payload_capacity,
            assignment=capacity_assignment,
        )
    if sequential:
        positions: dict[int, int] = {}
        ordered = []
        for task in tasks:
            owner = task.preassigned_to
            position = positions.get(owner, 0)
            ordered.append(replace(task, sequence_position=position))
            positions[owner] = position + 1
        tasks = tuple(ordered)
    return JointRoutingInstance(
        depot_node=selected_depot, tasks=tasks,
        witness_owners=tuple(capacity_assignment),
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
) -> list[float] | tuple[list[float], list[int]]:
    """Sample demands with a guaranteed feasible multi-bin partition."""
    total_capacity = num_trucks * payload_capacity
    if count * min_demand > total_capacity + 1e-9:
        raise ValueError("minimum customer demand exceeds total fleet capacity")

    assignments = np.arange(count, dtype=int) % num_trucks
    rng.shuffle(assignments)
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


def _leg_is_feasible(
    origin: int,
    destination: int,
    battery_capacity: float,
    transport_graph,
    charging_nodes: set[int],
) -> bool:
    direct_energy = transport_graph.get_path_energy(origin, destination)
    if math.isfinite(direct_energy) and direct_energy <= battery_capacity:
        return True

    for charger in charging_nodes:
        first_energy = transport_graph.get_path_energy(origin, charger)
        second_energy = transport_graph.get_path_energy(charger, destination)
        if (
            math.isfinite(first_energy)
            and math.isfinite(second_energy)
            and first_energy <= battery_capacity
            and second_energy <= battery_capacity
        ):
            return True
    return False


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
