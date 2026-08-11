"""Tests for joint fleet-routing instance generation."""

import networkx as nx
import numpy as np

from EVRoutingEnv.models.environment.joint_instance import (
    generate_joint_routing_instance,
)


class _ToyTransportationGraph:
    def __init__(self) -> None:
        self.graph = nx.complete_graph(8, create_using=nx.DiGraph)

    def get_all_nodes(self) -> list[int]:
        return list(self.graph.nodes)

    def get_path_energy(self, origin: int, destination: int) -> float:
        if origin == destination:
            return 0.0
        return 20.0 + abs(origin - destination)


def _generate(seed: int):
    return generate_joint_routing_instance(
        _ToyTransportationGraph(),
        charging_nodes=[7],
        rng=np.random.default_rng(seed),
        num_customers=4,
        num_trucks=2,
        battery_capacity=100.0,
        payload_capacity=10.0,
        min_customer_demand=1.0,
        max_customer_demand=4.0,
        base_service_time=0.25,
    )


def test_joint_instance_replays_for_same_rng_seed() -> None:
    first = _generate(10)
    second = _generate(10)

    assert first.depot_node == second.depot_node
    assert [task.as_dict() for task in first.tasks] == [
        task.as_dict() for task in second.tasks
    ]


def test_joint_instance_has_unique_customers_and_capacity_feasible_demand() -> None:
    instance = _generate(11)
    customer_nodes = [task.node_id for task in instance.tasks]

    assert len(customer_nodes) == len(set(customer_nodes)) == 4
    assert instance.depot_node not in customer_nodes
    assert 7 not in customer_nodes
    assert sum(task.demand for task in instance.tasks) <= 20.0 + 1e-9
    assert all(task.demand <= 10.0 for task in instance.tasks)


def test_instance_registry_is_fresh() -> None:
    instance = _generate(12)
    first = instance.create_registry()
    second = instance.create_registry()

    node = first.tasks()[0].node_id
    first.claim(node, truck_id=0, timestamp=0.0)
    assert second.task_for_node(node).is_available

