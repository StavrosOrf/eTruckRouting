"""Perfect-information correctness against independent public-API enumeration."""

import itertools
import json
from copy import deepcopy
from dataclasses import replace

import networkx as nx
import numpy as np
import pytest

from EVRoutingEnv.baselines.optimization.perfect_information import (
    PerfectInformationParameters,
    replay_perfect_information_plan,
    solve_perfect_information_plan,
)
from EVRoutingEnv.evaluation.artifacts import (
    build_scenario_descriptor,
    scenario_static_inputs,
)
from EVRoutingEnv.evaluation.metrics import extract_operational_metrics
from EVRoutingEnv.models.environment import event_driven_env
from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.utils.utils import load_config


@pytest.fixture
def make_env(monkeypatch):
    environments = []

    def make(customers=3, trucks=2, charging=False, variant="joint", seed=17):
        config = deepcopy(load_config("EVRoutingEnv/config_files/config_joint.yaml"))
        config["environment"].update(
            num_trucks=trucks,
            num_stops=customers,
            max_episode_steps=8,
            generation_attempts=32,
        )
        config["problem"].update(
            assignment="fleet" if variant == "joint" else "preassigned",
            mode="sequential" if variant == "sequential" else "joint_fleet",
        )
        if charging:
            config["truck"].update(battery_capacity=120.0, initial_battery=50.0)
            config["charging"]["target_soc_levels"] = [1.0]
            config["problem"].update(
                payload_capacity=4.0, min_customer_demand=4.0, max_customer_demand=4.0
            )
        else:
            config["truck"]["initial_battery"] = "random"
        graph = nx.complete_graph(
            customers + 1 + int(charging), create_using=nx.DiGraph
        )
        for node in graph:
            graph.nodes[node].update(
                has_charger=charging and node == customers + 1,
                charger_type={"DCFast": 1},
            )
        for a, b, data in graph.edges(data=True):
            data.update(
                distance=50.0 if charging else 1.0, time=0.1 + 0.03 * ((a + 2 * b) % 5)
            )
        monkeypatch.setattr(event_driven_env, "get_graph", lambda _: graph.copy())
        env = EventDrivenTruckEnv(config, verbose=False, enable_plotting=False)
        environments.append(env)
        env.reset(seed=seed)
        return env

    yield make
    for env in environments:
        env.close()


def enumerate_customer_routes(env, objective):
    """Independent enumeration of assignments/orders on charger-free networks."""
    customers = [task.node_id for task in env.task_registry.tasks()]
    optimum = float("inf")
    for order in itertools.permutations(customers):
        for owners in itertools.product(range(env.num_trucks), repeat=len(customers)):
            routes = {
                truck: iter(
                    [
                        node
                        for node, owner in zip(order, owners, strict=True)
                        if owner == truck
                    ]
                )
                for truck in range(env.num_trucks)
            }
            candidate = deepcopy(env)
            candidate._perfect_information = True
            for _ in range(len(customers) + env.num_trucks):
                truck = candidate.active_truck_id
                if truck is None:
                    break
                target = next(routes[truck], candidate.joint_instance.depot_node)
                candidate.step((target, 0.0, False))
            metrics = extract_operational_metrics(candidate)
            if metrics.success:
                value = (
                    metrics.fleet_makespan
                    if objective == "makespan"
                    else metrics.total_operating_time
                )
                optimum = min(optimum, value)
    return optimum


@pytest.mark.parametrize("objective", ["makespan", "total_time"])
@pytest.mark.parametrize("variant", ["joint", "preassigned", "sequential"])
def test_proven_optimum_matches_independent_stochastic_enumeration(
    make_env, objective, variant
):
    env = make_env(variant=variant)
    expected = enumerate_customer_routes(env, objective)
    solution = solve_perfect_information_plan(
        env, PerfectInformationParameters(objective=objective)
    )
    assert solution.proven_optimal
    assert solution.objective_hours == pytest.approx(expected, abs=1e-12)
    assert solution.best_bound_hours == solution.objective_hours
    replay = replay_perfect_information_plan(env, solution)
    assert replay["operational_metrics"]["success"]
    # Searching and replaying cannot consume the live scenario.
    assert env.episode_steps == env.global_clock == 0
    assert not env.traffic_simulator._journey_counters
    assert not env.delivery_simulator._delivery_counters
    assert not env.scenario_random_streams._normal_cache


def test_limits_report_incumbent_without_claiming_optimality(make_env):
    env = make_env()
    solution = solve_perfect_information_plan(
        env, PerfectInformationParameters(max_nodes=1)
    )
    assert solution.status == "FEASIBLE"
    assert solution.stop_reason == "node_limit"
    assert not solution.proven_optimal
    assert solution.best_bound_hours <= solution.objective_hours
    assert replay_perfect_information_plan(env, solution)["operational_metrics"][
        "success"
    ]
    env.feasibility_witness = ()
    no_incumbent = solve_perfect_information_plan(
        env, PerfectInformationParameters(max_nodes=1)
    )
    assert no_incumbent.status == "UNKNOWN"
    assert no_incumbent.objective_hours is None
    with pytest.raises(ValueError, match="no feasible incumbent"):
        replay_perfect_information_plan(env, no_incumbent)


def test_deterministic_node_budget_and_full_replay(make_env):
    env = make_env(customers=2, trucks=2, charging=True)
    parameters = PerfectInformationParameters(max_nodes=100_000)
    first = solve_perfect_information_plan(env, parameters)
    assert first.proven_optimal
    replay = replay_perfect_information_plan(env, first)
    assert replay["operational_metrics"]["total_charging_time"] > 0
    assert replay["operational_metrics"]["total_queue_time"] > 0
    assert replay["charger_queue_history"]
    np.random.seed(999)
    np.random.normal(size=1000)
    env.reset(seed=17)
    second = solve_perfect_information_plan(env, parameters)
    assert first.actions == second.actions
    assert first.nodes_expanded == second.nodes_expanded
    assert first.objective_hours == second.objective_hours
    assert replay == replay_perfect_information_plan(env, second)
    json.dumps(replay, allow_nan=False)


def test_replay_rejects_wrong_seed_objective_and_active_truck(make_env):
    env = make_env()
    solution = solve_perfect_information_plan(env)
    with pytest.raises(ValueError, match="objective differs"):
        replay_perfect_information_plan(env, replace(solution, objective_hours=999.0))
    actions = ((99, solution.actions[0][1]), *solution.actions[1:])
    with pytest.raises(ValueError, match="active truck"):
        replay_perfect_information_plan(env, replace(solution, actions=actions))
    env.reset(seed=18)
    with pytest.raises(ValueError, match="different scenario"):
        replay_perfect_information_plan(env, solution)
    env.step(env.num_charging_nodes)
    with pytest.raises(ValueError, match="freshly reset"):
        solve_perfect_information_plan(env)


def test_external_data_changes_identity_and_scenario_cache(make_env, tmp_path):
    env = make_env(customers=2, charging=True)
    env.config["environment"]["scenario_cache_dir"] = str(tmp_path)
    batteries = [t.current_battery for t in env.trucks]
    before = scenario_static_inputs(env)
    old_path = env._scenario_cache_path(batteries)
    station = env.charging_station
    station.waiting_time_lookup["test_entry"] = {"1": {"0.50": 99.0}}
    after = scenario_static_inputs(env)
    assert after["queue_lookup_sha256"] != before["queue_lookup_sha256"]
    assert env._scenario_cache_path(batteries) != old_path
    old_path = env._scenario_cache_path(batteries)
    station.charger_efficiency[env.charging_nodes[0]] = 0.8
    assert env._scenario_cache_path(batteries) != old_path
    old_network = build_scenario_descriptor(env)["static_inputs"]["network_sha256"]
    env.transport_graph.scale_network(time_scale=2.0)
    assert (
        build_scenario_descriptor(env)["static_inputs"]["network_sha256"] != old_network
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_nodes": 0},
        {"max_nodes": True},
        {"time_limit_seconds": float("nan")},
        {"objective": "reward"},
        {"information_mode": "bad"},
    ],
)
def test_parameters_reject_invalid_budgets(kwargs):
    with pytest.raises(ValueError):
        PerfectInformationParameters(**kwargs)


def test_charging_optimum_matches_unpruned_public_action_tree(make_env):
    env = make_env(customers=2, trucks=2, charging=True)
    root = deepcopy(env)
    root._perfect_information = True
    pending = [root]
    best = float("inf")
    while pending:
        state = pending.pop()
        metrics = extract_operational_metrics(state)
        if state._episode_ended:
            if metrics.success:
                best = min(best, metrics.fleet_makespan)
            continue
        for action, allowed in enumerate(state.mask_fn()):
            if allowed:
                child = deepcopy(state)
                child.step(action)
                pending.append(child)
    solution = solve_perfect_information_plan(env)
    assert solution.proven_optimal
    assert solution.objective_hours == pytest.approx(best, abs=1e-12)


def test_known_energy_allows_a_trip_refused_by_worst_case_reserve(make_env):
    env = make_env(customers=1, trucks=1)
    target = env.joint_instance.tasks[0].node_id
    truck = env.trucks[0]
    values = env.traffic_simulator.realize_journey(
        from_node=truck.current_node,
        to_node=target,
        current_time=0.0,
        travel_time=env.transport_graph.get_time_distance(truck.current_node, target),
        base_energy=1.0,
    )
    assert values["energy_kwh"] < 1.2
    truck.current_battery = (values["energy_kwh"] + 1.2) / 2
    assert not env.mask_fn()[0]
    candidate = deepcopy(env)
    candidate._perfect_information = True
    for _ in range(2):
        assert candidate.mask_fn()[0]
        assert not candidate.traffic_simulator._journey_counters
    candidate.step(0)
    assert candidate.invalid_action_count == 0
    assert candidate.trucks[0].total_energy_consumed == values["energy_kwh"]
    assert not getattr(env, "_perfect_information", False)


def test_node_only_budget_is_supported(make_env):
    env = make_env()
    parameters = PerfectInformationParameters(time_limit_seconds=None, max_nodes=1)
    first = solve_perfect_information_plan(env, parameters)
    second = solve_perfect_information_plan(env, parameters)
    assert first.actions == second.actions
    assert first.nodes_expanded == second.nodes_expanded == 1
    assert first.stop_reason == second.stop_reason == "node_limit"
