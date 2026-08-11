"""Generation of fleet-owned customer tasks for joint routing instances."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from EVRoutingEnv.models.core.customer import CustomerTask, FleetTaskRegistry


@dataclass(frozen=True)
class JointRoutingInstance:
    """Static customer/depot data shared by all trucks in an episode."""

    depot_node: int
    tasks: tuple[CustomerTask, ...]

    def create_registry(self) -> FleetTaskRegistry:
        # Tasks are mutable during an episode, so return fresh copies.
        return FleetTaskRegistry(
            CustomerTask(
                task_id=task.task_id,
                node_id=task.node_id,
                demand=task.demand,
                base_service_time=task.base_service_time,
                earliest_service=task.earliest_service,
                latest_service=task.latest_service,
            )
            for task in self.tasks
        )


def generate_joint_routing_instance(
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
) -> JointRoutingInstance:
    """Generate one feasible-capacity joint-routing instance.

    Every customer is placed at a unique non-charging node that can be reached
    from and returned to the selected depot, possibly through charging stations,
    with a full battery. This validates static reachability only; online energy
    feasibility remains a policy responsibility.
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

    selected_depot: int | None = None
    selected_customers: list[int] | None = None
    for depot in depot_candidates:
        feasible_customers = [
            node
            for node in eligible_nodes
            if node != depot
            and _leg_is_feasible(
                depot,
                node,
                battery_capacity,
                transport_graph,
                charging_set,
            )
            and _leg_is_feasible(
                node,
                depot,
                battery_capacity,
                transport_graph,
                charging_set,
            )
        ]
        if len(feasible_customers) < num_customers:
            continue
        chosen = rng.choice(feasible_customers, size=num_customers, replace=False)
        selected_depot = int(depot)
        selected_customers = [int(node) for node in chosen]
        break

    if selected_depot is None or selected_customers is None:
        raise ValueError(
            "could not generate enough battery-reachable customers from any depot"
        )

    demands = _capacity_feasible_demands(
        rng,
        count=num_customers,
        total_capacity=num_trucks * payload_capacity,
        min_demand=min_customer_demand,
        max_demand=max_customer_demand,
    )
    tasks = tuple(
        CustomerTask(
            task_id=task_id,
            node_id=node_id,
            demand=demand,
            base_service_time=base_service_time,
        )
        for task_id, (node_id, demand) in enumerate(
            zip(selected_customers, demands, strict=True)
        )
    )
    return JointRoutingInstance(depot_node=selected_depot, tasks=tasks)


def _capacity_feasible_demands(
    rng: np.random.Generator,
    *,
    count: int,
    total_capacity: float,
    min_demand: float,
    max_demand: float,
) -> list[float]:
    """Sample demands while reserving minimum capacity for remaining tasks."""
    if count * min_demand > total_capacity + 1e-9:
        raise ValueError("minimum customer demand exceeds total fleet capacity")

    remaining_capacity = float(total_capacity)
    demands: list[float] = []
    for index in range(count):
        remaining_tasks = count - index - 1
        maximum_here = min(
            max_demand,
            remaining_capacity - remaining_tasks * min_demand,
        )
        if maximum_here < min_demand - 1e-9:
            raise ValueError("could not allocate capacity-feasible customer demands")
        demand = float(rng.uniform(min_demand, maximum_here))
        demands.append(demand)
        remaining_capacity -= demand
    return demands


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
    if base_service_time < 0.0:
        raise ValueError("base service time cannot be negative")
