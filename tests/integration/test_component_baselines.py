"""Baselines under the components: Kullman's myopic rules, queue-aware planning,
and the scale and stress cases of plan section 8.6."""

import copy

import numpy as np
import pytest
from extension_harness import PUBLIC, SAME_DAY, InvariantChecker, build, choose, rollout

from EVRoutingEnv.baselines.heuristics.component_baselines import KullmanMyopicPolicy
from EVRoutingEnv.models.simulation import queueing
from EVRoutingEnv.models.simulation.public_station_queue import QueueEntry
from EVRoutingEnv.state.feasibility import ActionKind, joint_action_feasibility


pytestmark = pytest.mark.integration

KULLMAN = {"enabled": True, "exogenous_traffic": {"utilization": 0.65}}


def _at_station(queue_length):
    env = build("seattle", trucks=1, stops=3, public_charging=KULLMAN)
    env.reset(seed=2)
    decisions = joint_action_feasibility(env)
    node_of = {i: d.target_node for i, d in enumerate(decisions[: env.num_charging_nodes]) if d.feasible}
    index = min(node_of, key=lambda i: env.transport_graph.get_time_distance(
        int(env.trucks[0].current_node), node_of[i]))
    station = env.public_stations.queues[node_of[index]]
    station._next_arrival = None
    station.queue = []
    for port in station.ports:
        port.occupant = QueueEntry("outside", 10_000 + port.index, -1.0, 50.0) if queue_length is not None else None
        port.busy_until = 50.0 if queue_length is not None else None
        port.timeline_up, port.next_change = True, float("inf")
    for i in range(queue_length or 0):
        station.queue.append(QueueEntry("outside", 20_000 + i, -0.5 + i * 1e-3, 1.0))
    env.step(index)
    return env, node_of[index]


def test_kullman_queues_only_behind_a_shorter_than_expected_queue() -> None:
    env, node = _at_station(0)
    params = env.public_stations.station_parameters(node)
    expected = queueing.mean_queue_length(params.ports, params.mean_arrival_rate, params.service_rate)
    queue_index = env.extra_action_offset + env.extra_action_kinds.index(ActionKind.QUEUE)
    assert 0 < expected  # an empty queue is shorter than expected
    assert KullmanMyopicPolicy()(env) == queue_index
    env, node = _at_station(int(np.ceil(expected)) + 1)
    action = KullmanMyopicPolicy()(env)
    assert action != queue_index
    assert joint_action_feasibility(env)[action].action_kind in {
        ActionKind.CUSTOMER, ActionKind.CHARGER, ActionKind.DEPOT}


def test_kullman_charges_to_full_at_an_available_station() -> None:
    env, node = _at_station(None)
    action = KullmanMyopicPolicy()(env)
    decisions = joint_action_feasibility(env)
    assert decisions[action].action_kind is ActionKind.CHARGE
    assert env.charge_action_values[action - env.num_navigation_actions] == max(
        env.charge_action_values[i - env.num_navigation_actions]
        for i in range(env.num_navigation_actions, env.extra_action_offset) if decisions[i].feasible
    )


def test_kullman_prefers_the_nearest_reachable_customer() -> None:
    env = build("seattle", trucks=1, stops=3, public_charging=KULLMAN)
    env.reset(seed=0)
    decisions = joint_action_feasibility(env)
    action = KullmanMyopicPolicy()(env)
    customers = [i for i, d in enumerate(decisions) if d.feasible and d.action_kind is ActionKind.CUSTOMER]
    node = int(env.trucks[0].current_node)
    assert action == min(customers, key=lambda i: env.transport_graph.get_time_distance(
        node, decisions[i].target_node))


def test_kullman_returns_once_everything_is_served() -> None:
    env = build("seattle", trucks=1, stops=3, public_charging=KULLMAN)
    observation, info = env.reset(seed=0)
    policy = KullmanMyopicPolicy()
    while not env._episode_ended and not env.task_registry.all_served():
        observation, _, *_ , info = env.step(policy(env, observation, info))
    if not env._episode_ended:
        decisions = joint_action_feasibility(env)
        if decisions[env.num_charging_nodes + env.fixed_num_stops].feasible:
            assert policy(env) == env.num_charging_nodes + env.fixed_num_stops


def test_kullman_visits_at_most_two_stations_between_customers() -> None:
    for seed in range(3):
        env = build("california", trucks=1, stops=4, public_charging=KULLMAN)
        observation, info = env.reset(seed=seed)
        policy = KullmanMyopicPolicy()
        check = InvariantChecker(env)  # includes the station-hop counter
        while not env._episode_ended:
            decisions = joint_action_feasibility(env)
            action = policy(env, observation, info)
            alternatives = any(
                d.feasible and d.action_kind in {ActionKind.CUSTOMER, ActionKind.DEPOT} for d in decisions)
            if decisions[action].action_kind is ActionKind.CHARGER and alternatives:
                assert env.trucks[0].stations_since_stop < 2
            observation, _, terminated, truncated, info = env.step(action)
            check(env)


def test_queue_aware_planning_prices_the_expected_wait() -> None:
    from EVRoutingEnv.baselines.heuristics.alns import ALNSParameters, _NominalProblem
    from EVRoutingEnv.baselines.optimization.exact_optimization import (
        expected_station_wait,
    )

    env = build("seattle", trucks=2, stops=4, public_charging={**KULLMAN, "exogenous_traffic": {"utilization": 0.9}})
    env.reset(seed=0)
    plain = _NominalProblem(env, ALNSParameters())
    aware = _NominalProblem(env, ALNSParameters(queue_aware=True))
    wait = expected_station_wait(env)
    assert wait > 0.0
    a, b = env.joint_instance.depot_node, env.joint_instance.tasks[0].node_id
    energy = env.transport_graph.get_path_energy(a, b)
    capacity = float(env.config["truck"]["battery_capacity"])
    assert aware.arc(a, b) - plain.arc(a, b) == pytest.approx(energy * wait / capacity)
    no_traffic = build("seattle", trucks=2, stops=4)
    no_traffic.reset(seed=0)
    assert expected_station_wait(no_traffic) == 0.0


def test_scale_128_trucks_share_one_public_station() -> None:
    from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
    from EVRoutingEnv.utils.utils import load_config

    config = copy.deepcopy(load_config("EVRoutingEnv/config_files/config_joint.yaml"))
    config["environment"].update(num_trucks=128, num_stops=25, max_time=60.0)
    config["truck"]["initial_battery"] = 60.0  # everyone needs the station
    config["public_charging"] = {**PUBLIC, "exogenous_traffic": {"utilization": 0.9}}
    env = EventDrivenTruckEnv(config, verbose=False, enable_plotting=False)
    observation, info = env.reset(seed=0)
    depot = env.joint_instance.depot_node
    target = min(env.charging_nodes, key=lambda node: env.transport_graph.get_time_distance(depot, node))
    for node in env.charging_nodes:
        if node != target:
            env.set_charger_available(node, False)
    checker = InvariantChecker(env)
    target_index = env.charging_nodes.index(target)
    steps = 0
    rng = np.random.default_rng(0)
    while not env._episode_ended and steps < 4000:
        decisions = joint_action_feasibility(env)
        feasible = [i for i, d in enumerate(decisions) if d.feasible]
        action = target_index if decisions[target_index].feasible else int(rng.choice(feasible))
        if decisions[target_index].feasible is False:
            queue = [i for i in feasible if decisions[i].action_kind is ActionKind.QUEUE]
            action = queue[0] if queue else action
        observation, _, terminated, truncated, info = env.step(action)
        checker(env)
        steps += 1
    assert env.termination_reason != "no_events_with_live_truck"
    log = env.public_stations.queues[target].log
    assert sum(kind == "fleet" for kind, *_ in log.starts) >= 10


def test_the_request_envelope_can_be_filled_exactly() -> None:
    probe = build("seattle", trucks=1, requests=SAME_DAY)
    probe.reset(seed=3)
    count = len(probe.request_engine.registry)
    exact = build("seattle", trucks=1, requests={**SAME_DAY, "max_requests": count})
    exact.reset(seed=3)
    assert len(exact.request_engine.registry) == count == exact.fixed_num_stops
    short = build("seattle", trucks=1, requests={**SAME_DAY, "max_requests": count - 1})
    with pytest.raises(ValueError, match="envelope"):
        short.reset(seed=3)


@pytest.mark.statistical
@pytest.mark.parametrize("regime", ["us_fmcsa_property", "eu_561_2006"])
@pytest.mark.parametrize("policy", ["greedy", "random"])
def test_mixed_components_on_texas_longhaul_fifty_seeds(regime, policy) -> None:
    blocks = {"public_charging": PUBLIC,
              "hours_of_service": {"enabled": True, "regime": regime},
              "requests": {**SAME_DAY, "static_context_size": 30}}
    env = build("texas_longhaul", trucks=10, max_time=48.0, **blocks)
    for seed in range(50):
        rollout(env, seed, choose(env, policy, seed), check=InvariantChecker(env))
