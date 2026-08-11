"""Integration checks for fleet-owned customer generation and service events."""

import heapq
import os
from copy import deepcopy

import pytest


os.environ.setdefault("MPLCONFIGDIR", "/tmp/evrp_matplotlib")

from EVRoutingEnv.models.core.customer import (
    CustomerTask,
    FleetTaskRegistry,
    TaskStatus,
)
from EVRoutingEnv.models.core.truck import Truck
from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.models.environment.event_handlers import (
    Event,
    EventHandler,
    EventType,
)
from EVRoutingEnv.models.simulation.delivery_simulator import DeliverySimulator
from EVRoutingEnv.utils.utils import load_config


def _joint_config() -> dict:
    config = deepcopy(load_config("EVRoutingEnv/config_files/config_vrp.yaml"))
    config["environment"].update(
        {
            "num_trucks": 2,
            "num_stops": 4,
            "allow_variable_num_stops": False,
        }
    )
    config["truck"]["battery_capacity"] = 10_000.0
    config["problem"] = {
        "mode": "joint_fleet",
        "payload_capacity": 10.0,
        "min_customer_demand": 1.0,
        "max_customer_demand": 3.0,
        "base_service_time": 0.1,
    }
    return config


@pytest.mark.integration
def test_joint_reset_creates_one_shared_fleet_instance() -> None:
    env = EventDrivenTruckEnv(
        _joint_config(),
        verbose=False,
        enable_plotting=False,
    )
    try:
        _, info = env.reset(seed=31415)

        assert env.joint_instance is not None
        assert env.task_registry is not None
        depot = env.joint_instance.depot_node
        customer_nodes = [task.node_id for task in env.task_registry.tasks()]
        assert len(customer_nodes) == len(set(customer_nodes)) == 4
        assert all(truck.current_node == depot for truck in env.trucks)
        assert all(
            truck.delivery_sequence == [depot, *customer_nodes]
            for truck in env.trucks
        )
        assert sum(task.demand for task in env.task_registry.tasks()) <= 20.0
        assert info["problem_mode"] == "joint_fleet"
        assert info["task_counts"]["unassigned"] == 4
        assert not info["successful"]
    finally:
        env.close()


def test_joint_arrival_starts_but_does_not_complete_service() -> None:
    registry = FleetTaskRegistry(
        [CustomerTask(0, 5, demand=2.0, base_service_time=0.25)]
    )
    registry.claim(5, truck_id=0, timestamp=0.0)
    truck = Truck(
        truck_id=0,
        truck_type="electric",
        delivery_sequence=[1, 5],
        initial_battery=100.0,
        battery_capacity=100.0,
        base_speed=40.0,
        enable_flexible_delivery_order=True,
        payload_capacity=5.0,
    )
    event_queue: list[Event] = []
    event = Event(
        time=1.0,
        truck_id=0,
        event_type=EventType.TRUCK_ROUTING,
        data={
            "destination": 5,
            "distance": 40.0,
            "travel_time": 1.0,
            "discharge": 10.0,
            "unloading_time": 0.25,
            "task_id": 0,
        },
    )

    EventHandler().handle_truck_routing(
        event=event,
        trucks=[truck],
        truck_states={0: "routing"},
        truck_routes={0: []},
        event_queue=event_queue,
        global_clock=1.0,
        enable_plotting=False,
        delivery_simulator=DeliverySimulator(enable_stochastic_unloading=False),
        task_registry=registry,
    )

    task = registry.task_for_node(5)
    assert task.status is TaskStatus.IN_SERVICE
    assert not task.is_served
    assert 5 not in truck.delivered_nodes
    assert truck.remaining_payload == 5.0
    ready_event = heapq.heappop(event_queue)
    assert ready_event.time == 1.25
    assert ready_event.data["reason"] == "unloading_complete"
    assert ready_event.data["task_id"] == 0
    assert ready_event.data["customer_node"] == 5


@pytest.mark.integration
def test_joint_episode_serves_each_customer_once_and_returns_to_depot() -> None:
    env = EventDrivenTruckEnv(
        _joint_config(),
        verbose=False,
        enable_plotting=False,
    )
    try:
        env.reset(seed=31415)
        terminated = False
        truncated = False
        info = {}

        for _step in range(30):
            if terminated or truncated:
                break
            assert env.active_truck_id is not None
            truck = env.trucks[env.active_truck_id]
            available = env.task_registry.available_tasks(truck.remaining_payload)
            target = (
                available[0].node_id
                if available
                else env.joint_instance.depot_node
            )
            _, _, terminated, truncated, info = env.step((target, 0.0, False))

        depot = env.joint_instance.depot_node
        served_task_ids = [
            task_id
            for truck in env.trucks
            for task_id in truck.served_task_ids
        ]
        assert terminated and not truncated
        assert info["successful"]
        assert env.task_registry.all_served()
        assert len(served_task_ids) == len(set(served_task_ids)) == 4
        assert all(truck.current_node == depot for truck in env.trucks)
        assert all(truck.remaining_payload >= 0.0 for truck in env.trucks)
    finally:
        env.close()


@pytest.mark.integration
def test_joint_mask_uses_hard_engine_without_graph_dependencies() -> None:
    env = EventDrivenTruckEnv(
        _joint_config(),
        verbose=False,
        enable_plotting=False,
    )
    try:
        env.reset(seed=31415)
        mask = env.mask_fn()
        customer_start = env.num_charging_nodes
        customer_end = customer_start + env.fixed_num_stops
        depot_action = customer_end

        assert mask.shape == (env.action_space.n,)
        assert mask[customer_start:customer_end].all()
        assert not mask[depot_action]
        assert not mask[env.num_navigation_actions :].any()
        assert env.last_action_feasibility is not None
    finally:
        env.close()


@pytest.mark.integration
def test_premature_depot_action_fails_instead_of_rerouting() -> None:
    env = EventDrivenTruckEnv(
        _joint_config(),
        verbose=False,
        enable_plotting=False,
    )
    try:
        env.reset(seed=31415)
        acting_truck = env.active_truck_id
        depot = env.joint_instance.depot_node
        _, reward, _, _, info = env.step((depot, 0.0, False))

        assert reward == env.reward_config["failure_penalty"]
        assert env.trucks[acting_truck].failed
        assert (
            env.trucks[acting_truck].failure_reason
            == "invalid_action:same_location"
        )
        assert env.task_registry.counts()["unassigned"] == 4
        assert info["invalid_action_count"] == 1
    finally:
        env.close()


@pytest.mark.integration
def test_charge_away_from_station_fails_without_navigation_fallback() -> None:
    env = EventDrivenTruckEnv(
        _joint_config(),
        verbose=False,
        enable_plotting=False,
    )
    try:
        env.reset(seed=27182)
        acting_truck = env.active_truck_id
        origin = env.trucks[acting_truck].current_node
        _, reward, _, _, _ = env.step((origin, 1.0, True))

        truck = env.trucks[acting_truck]
        assert reward == env.reward_config["failure_penalty"]
        assert truck.failed
        assert truck.failure_reason == "invalid_action:not_at_charger"
        assert truck.current_node == origin
        assert truck.total_distance_traveled == 0.0
        assert env.task_registry.counts()["unassigned"] == 4
    finally:
        env.close()
