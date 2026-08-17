"""Integration checks for fleet-owned customer generation and service events."""

import heapq
import os
from copy import deepcopy

import pytest
from gymnasium.utils.env_checker import check_env


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
from EVRoutingEnv.state.features import SCHEMA_VERSION
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
def test_primary_joint_yaml_is_directly_runnable() -> None:
    env = EventDrivenTruckEnv(
        "EVRoutingEnv/config_files/config_joint.yaml",
        verbose=False,
        enable_plotting=False,
    )
    try:
        observation, info = env.reset(seed=12345)
        assert observation.shape == env.observation_space.shape
        assert info["problem_mode"] == "joint_fleet"
        assert info["feature_schema_version"] == SCHEMA_VERSION
        assert env.charging_action_mode == "target_soc"
        assert len(env.task_registry) == 10
        assert env.mask_fn().any()
    finally:
        env.close()


@pytest.mark.integration
def test_primary_joint_environment_satisfies_gymnasium_contract() -> None:
    env = EventDrivenTruckEnv(
        "EVRoutingEnv/config_files/config_joint.yaml",
        verbose=False,
        enable_plotting=False,
    )
    try:
        check_env(env, skip_render_check=True)
    finally:
        env.close()


@pytest.mark.integration
def test_time_window_variant_yaml_is_directly_runnable() -> None:
    env = EventDrivenTruckEnv(
        "EVRoutingEnv/config_files/config_joint_time_windows.yaml",
        verbose=False,
        enable_plotting=False,
    )
    try:
        env.reset(seed=12345)
        assert all(
            task.latest_service > task.earliest_service > 0.0
            for task in env.task_registry.tasks()
        )
        assert env.mask_fn().any()
    finally:
        env.close()


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
        assert set(env.charging_station.charger_power_kw.values()) == {
            150.0,
            350.0,
            750.0,
        }
        assert set(env.charging_station.charger_type.values()) == {
            "Level2",
            "DCFast",
        }
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
def test_early_arrival_waits_until_window_before_service() -> None:
    config = _joint_config()
    config["environment"]["num_trucks"] = 1
    config["problem"]["payload_capacity"] = 20.0
    env = EventDrivenTruckEnv(config, verbose=False, enable_plotting=False)
    try:
        env.reset(seed=6071)
        truck = env.trucks[0]
        task = env.task_registry.tasks()[0]
        nominal_time = env.transport_graph.get_time_distance(
            truck.current_node,
            task.node_id,
        )
        task.earliest_service = 3.0 * nominal_time
        task.latest_service = task.earliest_service + 20.0

        _, _, terminated, truncated, info = env.step(
            (task.node_id, 0.0, False)
        )

        assert not terminated and not truncated
        assert task.status is TaskStatus.SERVED
        assert task.service_started_at == pytest.approx(task.earliest_service)
        assert task.served_at > task.service_started_at
        assert truck.time_window_waiting_time > 0.0
        assert info["operational_metrics"]["total_time_window_waiting"] == (
            pytest.approx(truck.time_window_waiting_time)
        )
    finally:
        env.close()


@pytest.mark.integration
def test_realized_late_arrival_fails_and_releases_customer_claim() -> None:
    config = _joint_config()
    config["environment"]["num_trucks"] = 1
    config["problem"]["payload_capacity"] = 20.0
    env = EventDrivenTruckEnv(config, verbose=False, enable_plotting=False)
    try:
        env.reset(seed=6072)
        truck = env.trucks[0]
        task = env.task_registry.tasks()[0]
        nominal_time = env.transport_graph.get_time_distance(
            truck.current_node,
            task.node_id,
        )
        task.earliest_service = 0.0
        task.latest_service = env.global_clock + nominal_time + 0.1
        env.traffic_simulator.apply_traffic = (
            lambda **kwargs: (kwargs["travel_time"] + 1.0, 1.0)
        )
        env.traffic_simulator.apply_energy_uncertainty = (
            lambda **kwargs: kwargs["base_energy"]
        )

        _, reward, terminated, truncated, info = env.step(
            (task.node_id, 0.0, False)
        )

        assert terminated and not truncated
        assert truck.failed
        assert truck.failure_reason == "time_window_violation_after_realization"
        assert task.status is TaskStatus.UNASSIGNED
        assert reward <= env.reward_config["failure_penalty"]
        assert info["failure_causes"] == {
            "time_window_violation_after_realization": 1
        }
    finally:
        env.close()


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
        metrics = info["operational_metrics"]
        assert metrics["success"]
        assert metrics["customers_served"] == metrics["customers_total"] == 4
        assert metrics["completed_fraction"] == 1.0
        assert metrics["fleet_makespan"] == env.global_clock
        assert metrics["total_energy_consumed"] > 0.0
        assert metrics["fleet_makespan"] is not None
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
        assert not info["operational_metrics"]["success"]
        assert info["operational_metrics"]["fleet_makespan"] is None
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


@pytest.mark.integration
def test_joint_target_soc_mask_and_execution_reach_exact_target() -> None:
    config = _joint_config()
    config["environment"]["num_trucks"] = 1
    config["problem"]["payload_capacity"] = 20.0
    env = EventDrivenTruckEnv(
        config,
        verbose=False,
        enable_plotting=False,
    )
    try:
        env.reset(seed=16180)
        truck = env.trucks[0]
        charger = env.charging_nodes[0]
        truck.current_node = charger
        truck.current_battery = 0.5 * truck.battery_capacity
        env.active_truck_id = 0
        env.truck_states[0] = "ready"
        env.truck_ready_times[0] = env.global_clock
        env.event_queue.clear()

        initial_mask = env.mask_fn()
        charge_mask = initial_mask[env.num_navigation_actions :]
        assert env.charging_action_mode == "target_soc"
        assert env.charge_action_values == [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        assert charge_mask.tolist() == [False, True, True, True, True, True]

        _, _, terminated, truncated, _ = env.step((charger, 0.6, True))

        assert not terminated and not truncated
        assert truck.current_battery / truck.battery_capacity == pytest.approx(
            0.6,
            abs=1e-9,
        )
        assert truck.must_leave_charger
        after_mask = env.mask_fn()[env.num_navigation_actions :]
        assert not after_mask.any()
    finally:
        env.close()


@pytest.mark.integration
def test_empty_feasible_set_terminates_with_explicit_failure() -> None:
    config = _joint_config()
    config["environment"]["num_trucks"] = 1
    config["problem"]["payload_capacity"] = 20.0
    env = EventDrivenTruckEnv(config, verbose=False, enable_plotting=False)
    try:
        env.reset(seed=14142)
        truck = env.trucks[0]
        truck.current_node = env.task_registry.tasks()[0].node_id
        truck.current_battery = 0.0
        env.active_truck_id = None
        env.truck_states[0] = "ready"
        env.event_queue = [
            Event(
                time=env.global_clock,
                truck_id=0,
                event_type=EventType.TRUCK_READY,
                data={"reason": "edge_case_probe"},
            )
        ]

        env._advance_to_next_decision()
        info = env._get_info()

        assert env.active_truck_id is None
        assert truck.failed
        assert truck.failure_reason == "no_feasible_action"
        assert info["termination_reason"] == "no_feasible_action"
        assert not info["successful"]
    finally:
        env.close()


@pytest.mark.integration
def test_payload_deadlock_has_distinct_terminal_cause() -> None:
    env = EventDrivenTruckEnv(
        _joint_config(),
        verbose=False,
        enable_plotting=False,
    )
    try:
        env.reset(seed=17320)
        env.active_truck_id = None
        env.event_queue.clear()
        for truck in env.trucks:
            truck.remaining_payload = 0.5
            env.truck_states[truck.truck_id] = "ready"
            heapq.heappush(
                env.event_queue,
                Event(
                    time=env.global_clock,
                    truck_id=truck.truck_id,
                    event_type=EventType.TRUCK_READY,
                    data={"reason": "edge_case_probe"},
                ),
            )

        env._advance_to_next_decision()
        info = env._get_info()

        assert env.active_truck_id is None
        assert info["termination_reason"] == "payload_capacity_deadlock"
        assert info["failure_causes"] == {"payload_capacity_deadlock": 2}
        assert not info["all_customers_served"]
    finally:
        env.close()


@pytest.mark.integration
def test_joint_invariants_hold_across_multiple_stochastic_scenarios() -> None:
    env = EventDrivenTruckEnv(
        _joint_config(),
        verbose=False,
        enable_plotting=False,
    )
    try:
        for seed in range(20):
            env.reset(seed=10_000 + seed)
            terminated = False
            truncated = False
            info = {}

            for _step in range(40):
                if terminated or truncated:
                    break
                truck = env.trucks[env.active_truck_id]
                available = env.task_registry.available_tasks(
                    truck.remaining_payload
                )
                target = (
                    available[0].node_id
                    if available
                    else env.joint_instance.depot_node
                )
                _, _, terminated, truncated, info = env.step(
                    (target, 0.0, False)
                )

            served_ids = [
                task_id
                for truck in env.trucks
                for task_id in truck.served_task_ids
            ]
            assert terminated and not truncated, f"scenario seed {seed} did not end"
            assert info["successful"], (
                f"scenario seed {seed} failed: {info['termination_reason']}"
            )
            assert len(served_ids) == len(set(served_ids)) == 4
            assert all(
                truck.remaining_payload >= -1e-9 for truck in env.trucks
            )
            assert all(
                truck.current_node == env.joint_instance.depot_node
                for truck in env.trucks
            )
            for truck in env.trucks:
                expected_battery = (
                    truck.battery_capacity
                    + truck.total_energy_charged
                    - truck.total_energy_consumed
                )
                assert truck.current_battery == pytest.approx(
                    expected_battery,
                    abs=1e-7,
                )
    finally:
        env.close()


@pytest.mark.integration
def test_two_trucks_cannot_claim_the_same_customer() -> None:
    env = EventDrivenTruckEnv(
        _joint_config(),
        verbose=False,
        enable_plotting=False,
    )
    try:
        env.reset(seed=22360)
        first_truck = env.active_truck_id
        task = env.task_registry.tasks()[0]
        task_slot = next(
            index
            for index, node in enumerate(
                env.trucks[first_truck].delivery_sequence[1:]
            )
            if node == task.node_id
        )

        env.step((task.node_id, 0.0, False))
        second_truck = env.active_truck_id
        assert second_truck != first_truck
        mask = env.mask_fn()
        assert not mask[env.num_charging_nodes + task_slot]

        env.step((task.node_id, 0.0, False))

        assert env.trucks[second_truck].failure_reason == (
            "invalid_action:task_unavailable"
        )
        assert env.task_registry.task_for_node(task.node_id).served_by == first_truck
        served_ids = [
            task_id
            for truck in env.trucks
            for task_id in truck.served_task_ids
        ]
        assert served_ids.count(task.task_id) == 1
    finally:
        env.close()


@pytest.mark.integration
def test_full_station_fcfs_transfer_runs_through_environment_events() -> None:
    env = EventDrivenTruckEnv(
        _joint_config(),
        verbose=False,
        enable_plotting=False,
    )
    try:
        env.reset(seed=24494)
        charger = env.charging_nodes[0]
        env.charging_station.charger_capacity[charger] = 1
        env.event_queue.clear()
        env.active_truck_id = 0
        for truck in env.trucks:
            truck.current_node = charger
            truck.current_battery = 0.5 * truck.battery_capacity
            env.truck_states[truck.truck_id] = "ready"
            env.truck_ready_times[truck.truck_id] = env.global_clock
        heapq.heappush(
            env.event_queue,
            Event(
                time=env.global_clock,
                truck_id=1,
                event_type=EventType.TRUCK_READY,
                data={"reason": "simultaneous_arrival_probe"},
            ),
        )

        env.step((charger, 0.6, True))

        assert env.active_truck_id == 0
        assert [
            entry["truck_id"]
            for entry in env.charging_station.charger_waitlist[charger]
        ] == [1]
        assert 1 in env.charging_station.pending_wake_trucks[charger]
        assert env.truck_states[1] == "waiting_to_charge"

        customer = env.task_registry.available_tasks(
            env.trucks[0].remaining_payload
        )[0]
        env.step((customer.node_id, 0.0, False))

        assert env.active_truck_id == 1
        assert env.trucks[1].waiting_time > 0.0
        env.step((charger, 0.6, True))
        assert env.charging_station.charger_occupancy[charger] == [1]
        assert env.charging_station.charger_waitlist[charger] == []
        assert len(env.charging_station.charger_occupancy[charger]) <= 1
    finally:
        env.close()


@pytest.mark.integration
def test_closed_station_is_masked_and_cannot_be_selected() -> None:
    env = EventDrivenTruckEnv(
        _joint_config(),
        verbose=False,
        enable_plotting=False,
    )
    try:
        env.reset(seed=26457)
        acting_truck = env.active_truck_id
        charger = env.charging_nodes[0]
        charger_action = env.charging_nodes.index(charger)
        env.set_charger_available(charger, False)

        assert not env.mask_fn()[charger_action]
        _, reward, _, _, info = env.step((charger, 0.0, False))

        assert reward == env.reward_config["failure_penalty"]
        assert env.trucks[acting_truck].failure_reason == (
            "invalid_action:charger_unavailable"
        )
        assert not info["charger_utilization"]["all_chargers"][
            charger_action
        ]["available"]
    finally:
        env.close()
