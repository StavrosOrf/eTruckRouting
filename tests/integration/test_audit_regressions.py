"""Behavioral regressions for the 2026-10-17 environment audit."""

from __future__ import annotations

import heapq
from copy import deepcopy

import networkx as nx
import numpy as np
import pytest

from EVRoutingEnv.models.core.customer import CustomerTask
from EVRoutingEnv.models.core.transportation_graph import TransportationGraph
from EVRoutingEnv.models.environment.curriculum_env import (
    CurriculumEnvWrapper,
    UniformRandomStrategy,
)
from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.models.environment.event_handlers import Event, EventType
from EVRoutingEnv.models.environment.joint_instance import _preassign_to_trucks
from EVRoutingEnv.models.simulation.delivery_simulator import DeliverySimulator
from EVRoutingEnv.utils.utils import load_config


@pytest.fixture
def make_env():
    environments = []

    def build(*, legacy=None, trucks=1, stops=2, seed=7, **sections):
        name = legacy or "config_joint.yaml"
        config = deepcopy(load_config("EVRoutingEnv/config_files/" + name))
        config["environment"].update(num_trucks=trucks, num_stops=stops)
        config["truck"]["battery_capacity"] = 10_000.0
        config["traffic"].update(enable_traffic=False, enable_energy_uncertainty=False)
        config["delivery"]["enable_stochastic_unloading"] = False
        for section, values in sections.items():
            config.setdefault(section, {}).update(values)
        env = EventDrivenTruckEnv(config, enable_plotting=False)
        environments.append(env)
        env.reset(seed=seed)
        return env

    yield build
    for env in environments:
        env.close()


@pytest.mark.parametrize("integer", [True, False])
def test_depot_retirement_has_same_meaning_for_both_action_formats(make_env, integer):
    env = make_env()
    action = (
        env.num_charging_nodes + env.fixed_num_stops
        if integer
        else (env.joint_instance.depot_node, 0.0, False)
    )
    _, reward, terminated, _, info = env.step(action)
    assert terminated
    assert reward == 0.0
    assert info["invalid_action_count"] == 0
    assert info["task_counts"]["served"] == 0
    assert info["all_complete"] and not info["successful"]
    assert info["termination_reason"] == "fleet_stopped_with_unserved_customers"


@pytest.mark.parametrize("trucks", [1, 2])
def test_refusal_retries_same_truck_without_processing_other_events(make_env, trucks):
    env = make_env(trucks=trucks, environment={"invalid_action_mode": "penalize"})
    truck = env.active_truck_id
    clock = env.global_clock
    events = list(env.event_queue)
    observation = env._get_observation().copy()
    _, reward, terminated, truncated, _ = env.step(env.num_navigation_actions)
    assert not terminated and not truncated
    assert env.active_truck_id == truck
    assert env.global_clock == clock
    assert env.event_queue == events
    # Physical/task state is unchanged, but spent action budgets are visible.
    assert not np.array_equal(env._get_observation(), observation)
    assert reward == -env.reward_config["invalid_action_penalty"]
    env.step(env.num_charging_nodes)
    assert env.task_registry.tasks()[0].claimed_by == truck
    assert not env.trucks[truck].failed


@pytest.mark.parametrize("trucks", [1, 2])
def test_retry_budget_exhaustion_charges_failure_once(make_env, trucks):
    env = make_env(
        trucks=trucks,
        environment={"invalid_action_mode": "penalize", "invalid_action_budget": 3},
    )
    for _ in range(2):
        assert not env.step(env.num_navigation_actions)[2]
    _, reward, terminated, _, info = env.step(env.num_navigation_actions)
    assert terminated
    assert info["termination_reason"] == "invalid_action_budget_exhausted"
    assert (
        reward
        == -env.reward_config["invalid_action_penalty"]
        + env.reward_config["failure_penalty"]
    )
    assert env.step(0)[1] == 0.0


def test_explicit_feasible_partition_is_found_instead_of_greedy_overload():
    tasks = tuple(
        CustomerTask(i, i + 1, demand, 0.2) for i, demand in enumerate([5, 5, 4, 3, 3])
    )
    tasks = _preassign_to_trucks(tasks, num_trucks=2, payload_capacity=10.0)
    assert sorted(
        sum(task.demand for task in tasks if task.preassigned_to == i) for i in range(2)
    ) == [10.0, 10.0]
    impossible = tuple(CustomerTask(i, i + 1, 6, 0.2) for i in range(3))
    with pytest.raises(ValueError, match="cannot be assigned"):
        _preassign_to_trucks(impossible, num_trucks=2, payload_capacity=10.0)


def test_generated_preassignments_preserve_capacity_under_tight_loads(make_env):
    env = make_env(
        trucks=2,
        stops=5,
        problem={
            "assignment": "preassigned",
            "payload_capacity": 10,
            "max_customer_demand": 6,
        },
    )
    for seed in range(20):
        env.reset(seed=seed)
        for truck in env.trucks:
            demand = sum(
                task.demand
                for task in env.task_registry.tasks()
                if task.preassigned_to == truck.truck_id
            )
            assert demand <= truck.payload_capacity + 1e-9


def test_horizon_stops_return_and_never_reports_success(make_env):
    env = make_env(stops=1)
    env.max_time = 30.0  # Inject a cutoff after generating a certified instance.
    assert not env.step(env.num_charging_nodes)[3]
    _, _, terminated, truncated, info = env.step(
        env.num_charging_nodes + env.fixed_num_stops
    )
    assert not terminated and truncated
    assert env.global_clock == 30.0
    assert not info["successful"] and not info["operational_metrics"]["success"]
    assert info["termination_reason"] == "time_limit_exhausted"
    assert env.event_queue and env.trucks[0].route_destination is not None
    assert not env.trucks[0].failed
    assert env.step(0)[2:4] == (False, True)


def test_horizon_does_not_serve_a_customer_after_cutoff(make_env):
    env = make_env(stops=1)
    env.max_time = 0.01
    _, _, terminated, truncated, info = env.step(env.num_charging_nodes)
    assert not terminated and truncated
    assert env.global_clock == 0.01
    assert info["task_counts"]["served"] == 0
    assert env.trucks[0].total_distance_traveled == 0.0


def test_completion_exactly_at_horizon_is_success(make_env):
    env = make_env(legacy="config.yaml", stops=1)
    target = env.trucks[0].get_next_delivery_target()[0]
    travel = env.transport_graph.get_time_distance(env.trucks[0].current_node, target)
    depot = env.joint_instance.depot_node
    env.max_time = travel + env.problem_config["base_service_time"] + env.transport_graph.get_time_distance(target, depot)
    env.step((target, 0.0, False))
    _, _, terminated, truncated, info = env.step((depot, 0.0, False))
    assert terminated and not truncated
    assert info["successful"] and info["termination_reason"] == "success"


def test_departing_for_another_station_releases_the_source_queue(make_env):
    env = make_env(trucks=2)
    station = env.charging_station
    source, target = env.charging_nodes[:2]
    env.event_queue.clear()
    env.active_truck_id = None
    for truck in env.trucks:
        truck.current_node = source
        truck.current_battery = 5000.0
        env.truck_states[truck.truck_id] = "waiting_to_charge"
    station.charger_capacity[source] = 1
    station.charger_waitlist[source] = [
        {"truck_id": 0, "sequence": 1},
        {"truck_id": 1, "sequence": 2},
    ]
    station.wake_waiting_trucks(
        source, env.global_clock, env.event_queue, EventType, Event, env.truck_states
    )
    env._advance_to_next_decision()
    env.step((target, 0.0, False))
    assert env.active_truck_id == 1
    assert env.truck_states[1] == "ready"
    assert all(entry["truck_id"] != 0 for entry in station.charger_waitlist[source])


@pytest.mark.parametrize("reason", ["charger_port_available", "charger_closed"])
def test_stale_station_wake_cannot_make_a_routing_truck_ready(make_env, reason):
    env = make_env()
    target = env.task_registry.tasks()[0].node_id
    env._execute_navigation_action(env.trucks[0], target)
    heapq.heappush(
        env.event_queue, Event(0.0, 0, EventType.TRUCK_READY, {"reason": reason})
    )
    env._advance_to_next_decision()
    assert env.global_clock > 0.0
    assert env.trucks[0].route_destination is None
    env.step((env.task_registry.tasks()[1].node_id, 0.0, False))


def test_task_service_time_drives_execution_and_observation(make_env):
    env = make_env(
        problem={"base_service_time": 3.0}, delivery={"base_unloading_time": 0.2}
    )
    task = env.task_registry.tasks()[0]
    env.step((task.node_id, 0.0, False))
    assert task.served_at - task.service_started_at == pytest.approx(3.0)
    assert env.trucks[0].total_unloading_time == pytest.approx(3.0)


def test_task_service_override_does_not_change_sampler_default():
    simulator = DeliverySimulator(
        enable_stochastic_unloading=False, base_unloading_time=0.2
    )
    assert simulator.apply_unloading_time(1, 0, base_service_time=3.0) == 3.0
    assert simulator.apply_unloading_time(2, 0) == 0.2


def test_automatic_dead_end_receives_failure_penalty_once(make_env):
    env = make_env()
    truck = env.trucks[0]
    target = env.task_registry.tasks()[0].node_id
    truck.current_battery = (
        env.transport_graph.get_path_energy(truck.current_node, target) + 0.0001
    )
    _, reward, terminated, _, info = env.step(env.num_charging_nodes)
    assert terminated and info["termination_reason"] == "no_feasible_action"
    assert reward < 0.0
    assert env.episode_reward == reward
    assert env.step(0)[1] == 0.0


def test_charge_and_travel_use_the_same_time_coefficient(make_env):
    env = make_env(rewards={"time_multiplier": 5.0})
    truck = env.trucks[0]
    truck.current_node = env.charging_nodes[0]
    truck.current_battery = 4000.0
    _, reward, _, _, _ = env.step((truck.current_node, 0.5, True))
    assert reward == pytest.approx(-5 * truck.total_charging_time)


def test_zero_battery_arrival_at_charger_can_recharge(make_env):
    env = make_env()
    truck = env.trucks[0]
    truck.current_battery = env.transport_graph.get_path_energy(
        truck.current_node, env.charging_nodes[0]
    )
    assert env.mask_fn()[0]
    env.step(0)
    assert not truck.failed and truck.current_battery == 0.0
    env.step((truck.current_node, 0.5, True))
    assert truck.current_battery == pytest.approx(0.5 * truck.battery_capacity)


def test_curriculum_starts_with_a_valid_size_and_uses_first_sample_once():
    config = deepcopy(load_config("EVRoutingEnv/config_files/config_joint.yaml"))
    strategy = UniformRandomStrategy((1, 2), (1, 3), seed=7)
    control = UniformRandomStrategy((1, 2), (1, 3), seed=7)
    expected = control.sample(0)
    env = CurriculumEnvWrapper(config, strategy)
    try:
        _, info = env.reset(seed=7)
        assert (
            info["curriculum"]["num_trucks"],
            info["curriculum"]["num_stops"],
        ) == expected
        assert strategy.get_stats()["total_samples"] == 1
    finally:
        env.close()


@pytest.mark.parametrize("name", ["config.yaml", "config_joint.yaml"])
def test_legacy_yaml_default_constructor_is_runnable(name):
    env = EventDrivenTruckEnv("EVRoutingEnv/config_files/" + name)
    try:
        observation, _ = env.reset(seed=7)
        assert env.observation_space.contains(observation)
    finally:
        env.close()


def test_network_caches_are_isolated_and_scaling_invalidates_time_lookup():
    first = nx.DiGraph()
    first.add_edge(0, 1, distance=999.0, time=2.0)
    second = nx.DiGraph()
    second.add_edge(0, 1, distance=7.0, time=2.0)
    a, b = TransportationGraph(first), TransportationGraph(second)
    assert a.get_path_energy(0, 1) == 999.0
    assert b.get_path_energy(0, 1) == 7.0
    assert a._cache_file != b._cache_file
    assert a.get_time_distance(0, 1) == 2.0
    a.scale_network(time_scale=3)
    assert a.get_time_distance(0, 1) == 6.0
    assert not a.load_distance_cache() and not a.save_distance_cache()
    assert not a.get_cache_stats()["cache_file_exists"]
    reloaded = TransportationGraph(second.copy())
    assert reloaded.get_path_energy(0, 1) == 7.0


def test_legacy_flat_state_includes_every_fleet_customer(make_env):
    env = make_env(legacy="config_joint.yaml", trucks=2, stops=3, environment={"observation_mode": "legacy_flat"})
    unique = set().union(*(set(truck.delivery_sequence[1:]) for truck in env.trucks))
    assert len(unique) == env.state_space_manager.num_stops == 3


@pytest.mark.parametrize("legacy", ["config.yaml", "config_joint.yaml"])
def test_legacy_service_completion_and_depot_reward(make_env, legacy):
    env = make_env(legacy=legacy, stops=1)
    truck = env.trucks[0]
    target = truck.get_remaining_deliveries()[0]
    _, _, terminated, _, info = env.step((target, 0.0, False))
    assert truck.total_unloading_time == env.delivery_simulator.base_unloading_time
    if not terminated:
        before = truck.total_routing_time
        _, reward, terminated, _, info = env.step(
            (truck.delivery_sequence[0], 0.0, False)
        )
        assert reward == pytest.approx(
            -(truck.total_routing_time - before) * env.reward_config["time_multiplier"]
        )
    assert terminated and info["successful"]
    assert info["termination_reason"] == "success"


def test_stale_event_after_completion_cannot_move_makespan_past_horizon(make_env):
    env = make_env(legacy="config.yaml", stops=1)
    heapq.heappush(
        env.event_queue,
        Event(
            env.max_time + 50,
            0,
            EventType.TRUCK_READY,
            {"reason": "charger_port_available"},
        ),
    )
    env.step((env.trucks[0].get_next_delivery_target()[0], 0.0, False))
    _, _, terminated, truncated, info = env.step(
        (env.joint_instance.depot_node, 0.0, False)
    )
    assert terminated and not truncated
    assert info["successful"] and info["global_clock"] < env.max_time


def test_corrupt_or_foreign_cache_payload_is_not_loaded(tmp_path):
    import pickle

    graph = nx.DiGraph()
    graph.add_edge(0, 1, distance=999.0, time=2.0)
    transport = TransportationGraph(graph, precompute_distances=False)
    transport._cache_file = str(tmp_path / "wrong.pkl")
    with open(transport._cache_file, "wb") as file:
        pickle.dump({"fingerprint": "other-network", "distances": {(0, 1): 7.0}}, file)
    assert not transport.load_distance_cache()
    assert transport.get_path_energy(0, 1) == 999.0
    with open(transport._cache_file, "wb") as file:
        file.write(b"incomplete pickle")
    assert not transport.load_distance_cache()


def test_service_draw_uses_window_opening_and_task_duration(make_env, monkeypatch):
    env = make_env(problem={"base_service_time": 3.0})
    task = env.task_registry.tasks()[0]
    task.earliest_service, task.latest_service = 100.0, 150.0
    calls = []
    sample = env.delivery_simulator.apply_unloading_time

    def recorded_sample(**kwargs):
        calls.append(kwargs)
        return sample(**kwargs)

    monkeypatch.setattr(env.delivery_simulator, "apply_unloading_time", recorded_sample)
    env.step((task.node_id, 0.0, False))
    assert len(calls) == 1
    assert calls[0]["current_time"] == 100.0
    assert calls[0]["base_service_time"] == 3.0
    assert task.served_at == 103.0


def test_horizon_is_not_reported_as_a_battery_failure(make_env):
    env = make_env()
    task = env.task_registry.tasks()[0]
    truck = env.trucks[0]
    travel = env.transport_graph.get_time_distance(truck.current_node, task.node_id)
    truck.current_battery = (
        env.transport_graph.get_path_energy(truck.current_node, task.node_id) + 0.0001
    )
    env.max_time = travel + task.base_service_time
    _, _, terminated, truncated, info = env.step((task.node_id, 0.0, False))
    assert truncated and not terminated
    assert info["termination_reason"] == "time_limit_exhausted"
    assert not truck.failed
