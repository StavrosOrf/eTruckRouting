"""Tests for joint fleet-routing instance generation."""

from itertools import product

import networkx as nx
import numpy as np
import pytest

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


def test_total_capacity_is_not_mistaken_for_bin_pack_feasibility() -> None:
    with pytest.raises(ValueError, match="cannot be partitioned"):
        generate_joint_routing_instance(
            _ToyTransportationGraph(),
            charging_nodes=[7],
            rng=np.random.default_rng(13),
            num_customers=3,
            num_trucks=2,
            battery_capacity=100.0,
            payload_capacity=10.0,
            min_customer_demand=6.0,
            max_customer_demand=6.0,
            base_service_time=0.25,
        )


def test_generated_demands_have_a_feasible_vehicle_partition() -> None:
    for seed in range(20):
        instance = _generate(seed)
        demands = [task.demand for task in instance.tasks]
        feasible = any(
            all(
                sum(
                    demand
                    for demand, owner in zip(demands, assignment, strict=True)
                    if owner == truck_id
                )
                <= 10.0 + 1e-9
                for truck_id in range(2)
            )
            for assignment in product(range(2), repeat=len(demands))
        )
        assert feasible, f"seed {seed} produced infeasible demands {demands}"


def test_controlled_time_windows_are_seeded_and_bounded() -> None:
    values = {
        "transport_graph": _ToyTransportationGraph(),
        "charging_nodes": [7],
        "num_customers": 4,
        "num_trucks": 2,
        "battery_capacity": 100.0,
        "payload_capacity": 10.0,
        "min_customer_demand": 1.0,
        "max_customer_demand": 4.0,
        "base_service_time": 0.25,
        "time_window_config": {
            "enabled": True,
            "earliest_min": 2.0,
            "earliest_max": 4.0,
            "window_width_min": 5.0,
            "window_width_max": 7.0,
        },
    }
    first = generate_joint_routing_instance(
        rng=np.random.default_rng(44),
        **values,
    )
    second = generate_joint_routing_instance(
        rng=np.random.default_rng(44),
        **values,
    )

    assert [task.as_dict() for task in first.tasks] == [
        task.as_dict() for task in second.tasks
    ]
    for task in first.tasks:
        assert 2.0 <= task.earliest_service <= 4.0
        assert 5.0 <= task.latest_service - task.earliest_service <= 7.0


@pytest.mark.parametrize(
    "settings",
    [
        {"enabled": "yes"},
        {"enabled": True, "earliest_min": -1.0},
        {
            "enabled": True,
            "window_width_min": 2.0,
            "window_width_max": 1.0,
        },
    ],
)
def test_invalid_time_window_settings_are_rejected(settings) -> None:
    with pytest.raises((TypeError, ValueError), match="time.window|time_windows"):
        generate_joint_routing_instance(
            _ToyTransportationGraph(),
            charging_nodes=[7],
            rng=np.random.default_rng(45),
            num_customers=4,
            num_trucks=2,
            battery_capacity=100.0,
            payload_capacity=10.0,
            min_customer_demand=1.0,
            max_customer_demand=4.0,
            base_service_time=0.25,
            time_window_config=settings,
        )
