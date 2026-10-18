"""Full-route solvability and P1/P2 regressions from the Gym environment audit."""

import heapq
import json
from copy import deepcopy

import networkx as nx
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from algo.canonical_policy import CanonicalPolicyConfig
from EVRoutingEnv.evaluation.actions import validate_discrete_action
from EVRoutingEnv.models.environment import event_driven_env
from EVRoutingEnv.models.environment.curriculum_env import (
    CurriculumEnvWrapper,
    UniformRandomStrategy,
)
from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.models.environment.event_handlers import EventType
from EVRoutingEnv.state.features import GLOBAL_FEATURES, SCHEMA_VERSION, TRUCK_FEATURES
from EVRoutingEnv.utils.utils import load_config


def road(nodes=8, chargers=(6, 7), energy=20.0, hours=1.0):
    graph = nx.complete_graph(nodes, create_using=nx.DiGraph)
    for node in graph:
        graph.nodes[node].update(
            has_charger=node in chargers, charger_type={"DCFast": 1}
        )
    for _, _, data in graph.edges(data=True):
        data.update(distance=energy, time=hours)
    return graph


@pytest.fixture
def build_env(monkeypatch):
    environments = []

    def build(
        variant="joint",
        trucks=1,
        stops=2,
        seed=7,
        stochastic=False,
        graph=None,
        reset=True,
        **sections,
    ):
        config = deepcopy(load_config("EVRoutingEnv/config_files/config_joint.yaml"))
        config["environment"].update(
            num_trucks=trucks, num_stops=stops, generation_attempts=8
        )
        config["problem"].update(
            mode="sequential" if variant == "sequential" else "joint_fleet",
            assignment="fleet" if variant == "joint" else "preassigned",
            payload_capacity=40.0,
            min_customer_demand=1.0,
            max_customer_demand=2.0,
        )
        config["truck"].update(battery_capacity=120.0, initial_battery=50.0)
        config["traffic"].update(
            enable_traffic=stochastic, enable_energy_uncertainty=stochastic
        )
        config["delivery"]["enable_stochastic_unloading"] = stochastic
        for section, values in sections.items():
            config[section].update(values)
        network = road() if graph is None else graph
        monkeypatch.setattr(event_driven_env, "get_graph", lambda _: network.copy())
        env = EventDrivenTruckEnv(config, verbose=False, enable_plotting=False)
        environments.append(env)
        if reset:
            env.reset(seed=seed)
        return env

    yield build
    for env in environments:
        env.close()


def replay_certificate(env):
    trace = env.feasibility_witness
    assert trace
    for truck, action in trace:
        assert env.active_truck_id == truck
        assert env.mask_fn()[action]
        result = env.step(action)
    assert result[2:4] == (True, False)
    assert result[4]["successful"] and result[4]["all_customers_served"]
    assert not result[4]["any_failed"]
    return result


@pytest.mark.parametrize("variant", ["joint", "preassigned", "sequential"])
@pytest.mark.parametrize("trucks", [1, 2, 4])
@pytest.mark.parametrize("stops", [1, 2, 5])
@pytest.mark.parametrize("stochastic", [False, True])
@pytest.mark.parametrize("charging", ["target_soc", "duration"])
def test_every_accepted_instance_has_replayable_public_action_witness(
    build_env,
    variant,
    trucks,
    stops,
    stochastic,
    charging,
):
    env = build_env(
        variant,
        trucks,
        stops,
        stochastic=stochastic,
        charging={"action_mode": charging},
    )
    assert env.global_clock == env.episode_steps == env.invalid_action_count == 0
    assert not env.traffic_simulator._journey_counters
    assert not env.delivery_simulator._delivery_counters
    assert not env.scenario_random_streams._normal_cache
    assert all(not t.is_charging and not t.served_task_ids for t in env.trucks)
    assert env._get_info()["instance_feasibility"]["certified"]
    first = deepcopy(env.scenario_descriptor)
    trace = env.feasibility_witness
    replay_certificate(env)
    env.reset(seed=7)
    assert env.feasibility_witness == trace
    assert env.scenario_descriptor == first


@pytest.mark.parametrize(
    "sections",
    [
        {"environment": {"max_time": 2.0}},
        {"environment": {"max_episode_steps": 2}},
        {"truck": {"initial_battery": 0.0}},
        {"problem": {"time_windows": {"enabled": True, "window_width_min": 0.01}}},
        {"charging": {"target_soc_levels": [0.1]}, "truck": {"initial_battery": 20.0}},
        {"environment": {"max_hop_energy_kwh": 19.0}},
        {"environment": {"min_hop_energy_kwh": 21.0}},
    ],
)
def test_impossible_settings_raise_instead_of_emitting_unsolvable_instance(
    build_env, sections
):
    env = build_env(reset=False, **sections)
    with pytest.raises(ValueError, match="could not certify a solvable instance"):
        env.reset(seed=7)


def test_individually_reachable_customers_do_not_certify_a_whole_route(build_env):
    env = build_env(
        reset=False, graph=road(3, (), energy=60.0), truck={"initial_battery": "full"}
    )
    # Each customer has a 120 kWh round trip; visiting both requires 180 kWh.
    with pytest.raises(ValueError, match="could not certify"):
        env.reset(seed=7)


@pytest.mark.parametrize("variant", ["joint", "preassigned", "sequential"])
def test_certificate_rejects_routes_that_require_consecutive_chargers(
    build_env, variant
):
    graph = nx.DiGraph()
    for node in range(4):
        graph.add_node(node, has_charger=node in (1, 2), charger_type={"DCFast": 1})
    for a, b, energy in ((0, 1, 60), (1, 2, 60), (2, 3, 40)):
        graph.add_edge(a, b, distance=energy, time=0.1)
        graph.add_edge(b, a, distance=energy, time=0.1)
    env = build_env(
        variant,
        stops=1,
        graph=graph,
        truck={"battery_capacity": 100.0, "initial_battery": "full"},
        environment={"max_episode_steps": 10},
        reset=False,
    )
    with pytest.raises(ValueError, match="one.charg"):
        env.reset(seed=7)


@pytest.mark.parametrize("variant", ["joint", "preassigned", "sequential"])
def test_single_station_segments_replay_with_tight_action_limit(build_env, variant):
    graph = nx.DiGraph()
    for node in range(3):
        graph.add_node(node, has_charger=node == 1, charger_type={"DCFast": 1})
    for a, b, energy in ((0, 1, 60), (1, 2, 40)):
        graph.add_edge(a, b, distance=energy, time=0.1)
        graph.add_edge(b, a, distance=energy, time=0.1)
    env = build_env(
        variant,
        stops=1,
        graph=graph,
        truck={"battery_capacity": 100.0, "initial_battery": "full"},
        environment={"max_episode_steps": 6},
    )
    assert len(env.feasibility_witness) == 6
    replay_certificate(env)
    env.config["environment"]["max_episode_steps"] = 5
    env.max_episode_steps = 5
    with pytest.raises(ValueError, match="could not certify"):
        env.reset(seed=7)


def test_energy_bounds_are_enforced_without_restricting_early_depot_return(build_env):
    env = build_env(
        environment={"min_hop_energy_kwh": 20.0, "max_hop_energy_kwh": 20.0}
    )
    for route in env.joint_instance.witness_routes:
        previous = env.joint_instance.depot_node
        for target, _, charging in route:
            if (
                not charging
                and target not in env.charging_nodes
                and target != env.joint_instance.depot_node
            ):
                assert env.transport_graph.get_path_energy(previous, target) == 20.0
                previous = target
    assert env.step(env.num_charging_nodes + env.fixed_num_stops)[4]["all_complete"]


@pytest.mark.parametrize("old_key", ["min_hop_distance", "max_hop_distance"])
def test_removed_distance_settings_fail_with_a_migration_message(build_env, old_key):
    with pytest.raises(ValueError, match="were removed.*kWh|were removed"):
        build_env(environment={old_key: 10.0})


@pytest.mark.parametrize("mode,budget", [("terminate", 100), ("penalize", 1)])
def test_failing_queue_head_releases_next_truck(build_env, mode, budget):
    env = build_env(
        trucks=3,
        stops=3,
        environment={"invalid_action_mode": mode, "invalid_action_budget": budget},
    )
    node = env.charging_nodes[0]
    for _ in range(3):
        env.step(0)
    env.step((node, 0.9, True))
    env.step(env.num_charging_nodes + env.fixed_num_stops)
    assert env.active_truck_id == 1
    env.step((node, 0.1, True))
    assert env.trucks[1].failed and not env.trucks[2].failed
    assert not any(
        e["truck_id"] == 1 for e in env.charging_station.charger_waitlist[node]
    )
    if mode == "terminate":
        assert env.active_truck_id == 2
        env.step((node, 0.9, True))
        assert not env.trucks[2].failed


@pytest.mark.parametrize("observation", ["canonical_flat", "legacy_flat"])
def test_budget_and_pending_event_history_are_observable(build_env, observation):
    env = build_env(
        trucks=2,
        environment={
            "observation_mode": observation,
            "invalid_action_mode": "penalize",
            "invalid_action_budget": 3,
        },
    )
    before = env._get_observation()
    after, _, _, _, _ = env.step(env.num_navigation_actions)
    assert not np.array_equal(before, after)
    features = env.get_canonical_features()
    assert (
        features.global_features[GLOBAL_FEATURES.index("remaining_invalid_actions")]
        == 2
    )
    assert (
        features.global_features[GLOBAL_FEATURES.index("remaining_episode_steps")]
        == env.max_episode_steps - 1
    )
    env.step(0)
    before = env._get_observation()
    env.trucks[0].route_arrival_time += 0.5
    for event in env.event_queue:
        if event.truck_id == 0 and event.event_type is EventType.TRUCK_ROUTING:
            event.time += 0.5
    heapq.heapify(env.event_queue)
    after = env._get_observation()
    assert not np.array_equal(before, after)
    assert env.observation_space.contains(after)


def test_fcfs_order_and_reservations_are_observable(build_env):
    env = build_env(trucks=3, stops=3)
    node = env.charging_nodes[0]
    for _ in range(3):
        env.step(0)
    env.step((node, 0.9, True))
    features = env.get_canonical_features()
    column = TRUCK_FEATURES.index("queue_position")
    assert features.truck_features[1, column] == 0
    assert features.truck_features[2, column] == 1
    before = features.truck_features.copy()
    env.charging_station.charger_waitlist[node].reverse()
    assert not np.array_equal(before, env.get_canonical_features().truck_features)


def test_curriculum_records_real_success_once_and_preserves_explicit_budget(build_env):
    base = build_env(stops=1).config
    base["environment"]["max_episode_steps"] = 200
    wrapper = CurriculumEnvWrapper(base, UniformRandomStrategy([1, 1], [1, 1], seed=7))
    try:
        wrapper.reset(seed=7)
        assert wrapper.env.max_episode_steps == 200
        for _ in range(3):
            wrapper.step(wrapper.env.num_charging_nodes + 1)
        stats = wrapper.get_curriculum_stats()
        assert stats["total_episodes"] == 1
        assert stats["performance_by_size"]["1t_1s"]["success_rate"] == 0.0
        wrapper.reset(seed=7)
        for _, action in wrapper.env.feasibility_witness:
            wrapper.step(action)
        assert wrapper.episode_count == 2
        assert (
            wrapper.get_curriculum_stats()["performance_by_size"]["1t_1s"][
                "success_rate"
            ]
            == 0.5
        )
    finally:
        wrapper.close()


def test_curriculum_uses_base_default_and_counts_truncation_once(build_env):
    base = build_env(stops=1).config
    base["environment"].pop("max_episode_steps", None)
    wrapper = CurriculumEnvWrapper(base, UniformRandomStrategy([1, 1], [1, 1], seed=7))
    try:
        wrapper.reset(seed=7)
        assert wrapper.env.max_episode_steps >= 200
        wrapper.env.max_episode_steps = 1  # Cut off a certified scenario.
        result = wrapper.step(wrapper.env.num_charging_nodes)
        assert result[2:4] == (False, True)
        for _ in range(2):
            assert wrapper.step(0)[2:4] == (False, True)
        assert wrapper.episode_count == 1
        assert (
            wrapper.get_curriculum_stats()["performance_by_size"]["1t_1s"][
                "num_episodes"
            ]
            == 1
        )
    finally:
        wrapper.close()


@pytest.mark.parametrize("variant", ["joint", "preassigned", "sequential"])
@pytest.mark.parametrize("observation", ["canonical_flat", "legacy_flat"])
def test_certified_environment_still_passes_gym_contract(
    build_env, variant, observation
):
    env = build_env(
        variant,
        trucks=2,
        stops=2,
        stochastic=True,
        environment={"observation_mode": observation},
    )
    check_env(env, skip_render_check=True)


@pytest.mark.parametrize("value", [-1, 8, 9, 0.5, "1", True, np.bool_(False)])
def test_policy_action_ids_are_never_silently_wrapped(value):
    with pytest.raises(ValueError):
        validate_discrete_action(value, 8)


def test_numpy_integer_policy_action_is_preserved():
    assert validate_discrete_action(np.int64(7), 8) == 7


def test_legacy_gnn_policies_observe_budget_and_activity_history(build_env):
    from EVRoutingEnv.state.gnn_utils import create_default_gnn_space

    env = build_env(trucks=2, environment={"invalid_action_mode": "penalize"})
    adapter = create_default_gnn_space(env)
    before = adapter.get_state_GNN(env)
    env.step(env.num_navigation_actions)
    after = adapter.get_state_GNN(env)
    assert before.feature_schema_version == SCHEMA_VERSION
    assert not np.array_equal(before["truck"].x.numpy(), after["truck"].x.numpy())
    env.step(0)
    before = adapter.get_state_GNN(env)
    env.trucks[0].route_arrival_time += 0.5
    for event in env.event_queue:
        if event.truck_id == 0 and event.event_type is EventType.TRUCK_ROUTING:
            event.time += 0.5
    heapq.heapify(env.event_queue)
    after = adapter.get_state_GNN(env)
    assert not np.array_equal(before["truck"].x.numpy(), after["truck"].x.numpy())


def test_terminal_charger_metrics_exclude_unexecuted_future_charging(build_env):
    env = build_env(trucks=2, stops=2)
    env.step(0)
    env.step(env.num_charging_nodes)
    assert env.active_truck_id == 0
    env.max_time = env.global_clock + 0.1
    result = env.step((env.charging_nodes[0], 0.9, True))
    assert result[2:4] == (False, True)
    info = result[4]
    assert info["charger_utilization"]["overall"]["total_charge_time"] == pytest.approx(
        0.1
    )
    assert info["operational_metrics"]["total_charging_time"] == pytest.approx(0.1)
    assert env.step(0)[4]["charger_utilization"] == info["charger_utilization"]


@pytest.mark.parametrize("variant", ["joint", "preassigned", "sequential"])
@pytest.mark.parametrize("trucks,stops", [(1, 1), (1, 5), (4, 1), (4, 5)])
def test_certificates_replay_on_real_network_at_standard_battery(
    variant, trucks, stops
):
    config = deepcopy(load_config("EVRoutingEnv/config_files/config_joint.yaml"))
    config["environment"].update(num_trucks=trucks, num_stops=stops)
    config["problem"].update(
        mode="sequential" if variant == "sequential" else "joint_fleet",
        assignment="fleet" if variant == "joint" else "preassigned",
    )
    env = EventDrivenTruckEnv(config, verbose=False, enable_plotting=False)
    try:
        env.reset(seed=59)
        replay_certificate(env)
    finally:
        env.close()


def test_old_checkpoint_schema_is_rejected_before_loading_weights(tmp_path):
    path = tmp_path / "policy_config.json"
    config = {"max_trucks": 1, "max_customers": 2, "max_chargers": 2, "max_actions": 8}
    for version in (None, "joint-fleet-v5"):
        payload = dict(config)
        if version is not None:
            payload["feature_schema_version"] = version
        path.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match="incompatible.*retrain"):
            CanonicalPolicyConfig.load(path)
    config["feature_schema_version"] = SCHEMA_VERSION
    path.write_text(json.dumps(config))
    assert CanonicalPolicyConfig.load(path).feature_schema_version == SCHEMA_VERSION
