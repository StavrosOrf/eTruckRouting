"""Second-audit regressions; all environment batteries stay at the default 400 kWh."""

from copy import deepcopy

import networkx as nx
import numpy as np
import pytest

from EVRoutingEnv.models.core.customer import CustomerTask, FleetTaskRegistry
from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.models.environment.joint_instance import (
    generate_joint_routing_instance,
)
from EVRoutingEnv.models.simulation.charging_curve import ChargingCurveModel
from EVRoutingEnv.models.simulation.delivery_simulator import DeliverySimulator
from EVRoutingEnv.utils.utils import load_config


@pytest.fixture
def make_env():
    environments = []

    def build(trucks=1, customers=2, **sections):
        config = deepcopy(load_config("EVRoutingEnv/config_files/config_joint.yaml"))
        config["environment"].update(num_trucks=trucks, num_stops=customers)
        for section, values in sections.items():
            config.setdefault(section, {}).update(values)
        assert config["truck"]["battery_capacity"] == 400.0
        env = EventDrivenTruckEnv(config, verbose=False, enable_plotting=False)
        environments.append(env)
        env.reset(seed=7)
        # Controlled short legs isolate scheduling from geography; stochastic
        # mechanisms remain enabled and physical battery capacity is unchanged.
        env.transport_graph.get_path_energy = lambda a, b: float(a != b)
        env.transport_graph.get_time_distance = lambda a, b: float(a != b)
        return env

    yield build
    for env in environments:
        env.close()


def test_unfinished_delivery_earns_no_bonus_and_reports_elapsed_work(make_env):
    env = make_env(customers=1)
    env.max_time = 0.01
    _, reward, terminated, truncated, info = env.step(env.num_charging_nodes)
    assert truncated and not terminated
    assert reward == pytest.approx(-0.01 * env.reward_config["time_multiplier"])
    assert info["operational_metrics"]["vehicles_used"] == 1
    assert info["task_counts"]["served"] == 0
    assert info["operational_metrics"]["total_operating_time"] == pytest.approx(0.01)
    assert info["operational_metrics"]["total_travel_time"] == pytest.approx(0.01)
    assert env._get_info()["operational_metrics"] == info["operational_metrics"]


def test_service_bonus_paid_once_when_service_really_finishes(make_env):
    env = make_env(customers=1)
    _, reward, _, _, info = env.step(env.num_charging_nodes)
    truck = env.trucks[0]
    assert info["task_counts"]["served"] == 1
    assert reward == pytest.approx(500 - truck.total_time_elapsed)
    _, reward, terminated, truncated, _ = env.step(env.num_charging_nodes + 1)
    assert terminated and not truncated
    assert reward < 0
    assert env.step(0)[1] == 0.0


def test_preassigned_finished_owner_can_return_while_other_owner_works(make_env):
    env = make_env(trucks=2, customers=4, problem={"assignment": "preassigned"})
    owners = {task.node_id: task.preassigned_to for task in env.task_registry.tasks()}
    env.transport_graph.get_time_distance = lambda a, b: (
        1.0 if owners.get(b) == 0 else 20.0
    )
    for node in env.charging_nodes:
        env.set_charger_available(node, False)
    returned_early = False
    for _ in range(7):
        truck_id = env.active_truck_id
        own = env.task_registry.available_tasks(truck_id=truck_id)
        if own:
            result = env.step((own[0].node_id, 0.0, False))
        else:
            returned_early |= not env.task_registry.all_served()
            result = env.step((env.joint_instance.depot_node, 0.0, False))
        if result[2] or result[3]:
            break
    assert returned_early
    assert result[2:4] == (True, False) and result[4]["successful"]
    assert env.task_registry.all_served()
    assert not any(truck.failed for truck in env.trucks)


@pytest.mark.parametrize("action", [-1, -1000, 10_000, 0.5, True, np.bool_(False)])
def test_discrete_out_of_space_actions_do_not_mutate_state(make_env, action):
    env = make_env()
    before = (env.global_clock, env.episode_steps, env.trucks[0].current_node)
    with pytest.raises(ValueError, match="discrete action"):
        env.step(action)
    assert before == (env.global_clock, env.episode_steps, env.trucks[0].current_node)


def test_step_limit_is_latched_and_reset_clears_it(make_env):
    env = make_env()
    default_step_limit = env.max_episode_steps
    env.max_episode_steps = 1
    observation, _, terminated, truncated, info = env.step(env.num_charging_nodes)
    assert truncated and not terminated
    for _ in range(2):
        again, reward, ended, cutoff, next_info = env.step(0)
        assert reward == 0.0
        assert (ended, cutoff) == (terminated, truncated)
        assert np.array_equal(again, observation)
        assert info["global_clock"] == next_info["global_clock"]
    env.max_episode_steps = default_step_limit
    env.reset(seed=8)
    assert not env._episode_ended
    assert env.episode_steps == 0


@pytest.mark.parametrize("base", [0.0001, 0.02, 0.2])
def test_short_service_keeps_consistent_bounds_and_randomness(base):
    simulator = DeliverySimulator(
        enable_stochastic_unloading=True, base_unloading_time=base, seed=7
    )
    samples = np.array([simulator.apply_unloading_time(i, 12.0) for i in range(200)])
    assert np.ptp(samples) > 0
    assert samples.min() >= base * simulator.min_unloading_multiplier
    assert samples.max() <= base * simulator.max_unloading_multiplier


def test_very_weak_charger_finishes_without_time_step_stagnation():
    amount, details = ChargingCurveModel().calculate_charge_to_target(
        0.5,
        0.6,
        400.0,
        {
            "charge_rate": 1e-12,
            "efficiency": 0.9,
            "use_realistic_curve": True,
            "taper_start_soc": 0.8,
            "taper_power_min": 1e-12,
        },
    )
    assert amount == pytest.approx(40)
    assert details["actual_charge_hours"] == pytest.approx(40 / 0.9e-12)
    assert len(details["power_curve"]) < 100


def test_zero_service_can_complete_multiple_customers(make_env):
    env = make_env(problem={"base_service_time": 0.0})
    for offset in range(2):
        env.step(env.num_charging_nodes + offset)
        assert env.trucks[0].unloading_start_time is None
    assert env.task_registry.all_served()
    assert env.trucks[0].total_unloading_time == 0.0


def test_owner_constraint_is_enforced_at_registry_boundary():
    registry = FleetTaskRegistry([CustomerTask(0, 1, 1, 0.2, preassigned_to=1)])
    assert registry.available_tasks(truck_id=0) == []
    with pytest.raises(ValueError, match="preassigned"):
        registry.claim(1, 0, 0.0)
    assert registry.task_for_node(1).is_available
    registry.claim(1, 1, 0.0)


@pytest.mark.parametrize(("leg", "multiplier"), [(250.0, 1.0), (180.0, 1.2)])
def test_generator_rejects_customer_without_affordable_approach_and_exit(
    leg, multiplier
):
    class Graph:
        graph = nx.complete_graph(3, create_using=nx.DiGraph)

        def get_all_nodes(self):
            return list(self.graph)

        def get_path_energy(self, a, b):
            return leg if a != b else 0.0

    with pytest.raises(ValueError, match="energy-safe"):
        generate_joint_routing_instance(
            Graph(),
            [2],
            np.random.default_rng(7),
            num_customers=1,
            num_trucks=1,
            battery_capacity=400.0,
            payload_capacity=20.0,
            min_customer_demand=1.0,
            max_customer_demand=4.0,
            base_service_time=0.2,
            energy_multiplier=multiplier,
        )


@pytest.mark.parametrize("activity", ["charging", "unloading", "waiting_to_charge"])
def test_pending_activity_is_counted_once_in_metrics(make_env, activity):
    env = make_env()
    truck = env.trucks[0]
    env.truck_states[0] = activity
    env.global_clock = 0.1
    if activity == "charging":
        truck.charge_start_time = 0.0
    elif activity == "unloading":
        truck.unloading_start_time = 0.0
    else:
        env.waiting_start_times[0] = 0.0
    first = env._get_info()["operational_metrics"]
    assert first["total_operating_time"] == pytest.approx(0.1)
    assert first == env._get_info()["operational_metrics"]


def test_generator_never_uses_negative_net_energy_as_a_charging_exit():
    from EVRoutingEnv.models.environment.energy_reachability import EnergyReachability

    class Graph:
        def get_all_nodes(self):
            return [0, 1, 2]

        def get_path_energy(self, a, b):
            if a == b:
                return 0.0
            return {(0, 1): 350.0, (1, 0): 100.0, (1, 2): -100.0}.get((a, b), 450.0)

    envelope = EnergyReachability.build(Graph(), [2], 400)
    assert np.isinf(envelope.energy[1, 2])
    assert envelope.customers(0, [0, 1]) == []


def test_default_action_budget_scales_with_customer_count(make_env):
    env = make_env(trucks=10, customers=40)
    assert env.max_episode_steps == 2 * (env.num_charging_nodes + 2) * (40 + 10)


@pytest.mark.parametrize("seed", [1_000_000_000, 1_000_000_001, 1_000_000_002])
def test_energy_path_reference_finishes_real_stochastic_400kwh_instances(seed):
    from EVRoutingEnv.baselines.heuristics.energy_path import EnergyPathHeuristicPolicy

    config = deepcopy(load_config("EVRoutingEnv/config_files/config_joint.yaml"))
    config["environment"].update(num_trucks=2, num_stops=8)
    env = EventDrivenTruckEnv(config, verbose=False, enable_plotting=False)
    try:
        observation, info = env.reset(seed=seed)
        policy = EnergyPathHeuristicPolicy()
        reward_sum = 0.0
        for _ in range(env.max_episode_steps):
            action = policy(env, observation, info)
            assert env.mask_fn()[action]
            observation, reward, terminated, truncated, info = env.step(action)
            reward_sum += reward
            if terminated or truncated:
                break
        assert info["successful"], info["termination_reason"]
        assert env.task_registry.all_served()
        assert reward_sum == pytest.approx(
            len(env.task_registry.tasks()) * env.reward_config["delivery_bonus"]
            - env.reward_config["time_multiplier"]
            * info["operational_metrics"]["total_operating_time"]
        )
        assert env.episode_reward == pytest.approx(reward_sum)
        assert all(
            truck.current_node == env.joint_instance.depot_node for truck in env.trucks
        )
    finally:
        env.close()


def test_task_availability_matches_truck_payload_roundoff_tolerance():
    from EVRoutingEnv.models.core.truck import Truck

    remaining = 1.0 - 5e-10
    registry = FleetTaskRegistry([CustomerTask(0, 1, 1.0, 0.2)])
    truck = Truck(0, "electric", [0, 1], 400.0, 400.0, 40.0, payload_capacity=20.0)
    truck.remaining_payload = remaining
    assert truck.can_accept_demand(1.0)
    assert registry.available_tasks(remaining, truck_id=0) == registry.tasks()


def test_leftover_battery_penalty_waits_for_physical_completion(make_env):
    env = make_env(
        customers=1,
        rewards={
            "enable_leftover_battery_penalty": True,
            "leftover_battery_penalty_coef": 2.0,
        },
    )
    env.max_time = 0.01
    customer = env.task_registry.tasks()[0].node_id

    _, reward, terminated, truncated, info = env.step((customer, 0.0, False))

    assert truncated and not terminated
    assert info["task_counts"]["served"] == 0
    assert reward == pytest.approx(-0.01 * env.reward_config["time_multiplier"])
    assert env._penalized_completed_trucks == set()


@pytest.mark.parametrize("enabled", [False, True])
def test_leftover_battery_penalty_prices_used_truck_once(make_env, enabled):
    coefficient = 2.0
    env = make_env(
        trucks=2,
        customers=1,
        rewards={
            "enable_leftover_battery_penalty": enabled,
            "leftover_battery_penalty_coef": coefficient,
        },
    )
    customer = env.task_registry.tasks()[0].node_id
    depot = env.joint_instance.depot_node

    _, first_reward, terminated, truncated, first_info = env.step(
        (customer, 0.0, False)
    )
    assert not terminated and not truncated
    assert env.active_truck_id == 1
    _, retirement_reward, terminated, truncated, first_info = env.step(
        (depot, 0.0, False)
    )
    first_reward += retirement_reward
    assert not terminated and not truncated
    assert env.trucks[1].is_complete  # Explicitly retire the unused truck.
    assert env._penalized_completed_trucks == {1}
    assert env.trucks[1].total_distance_traveled == 0.0
    assert env.trucks[1].completed_service_count == 0
    assert first_reward == pytest.approx(
        env.reward_config["delivery_bonus"]
        - env.reward_config["time_multiplier"]
        * first_info["operational_metrics"]["total_operating_time"]
    )

    _, second_reward, terminated, truncated, final_info = env.step(
        (depot, 0.0, False)
    )
    assert terminated and not truncated
    assert final_info["successful"]
    assert env._penalized_completed_trucks == {0, 1}
    terminal_penalty = (
        coefficient
        * 100.0
        * env.trucks[0].battery_at_completion
        / env.trucks[0].battery_capacity
        if enabled
        else 0.0
    )
    assert first_reward + second_reward == pytest.approx(
        env.reward_config["delivery_bonus"]
        - env.reward_config["time_multiplier"]
        * final_info["operational_metrics"]["total_operating_time"]
        - terminal_penalty
    )
    assert env.step(0)[1] == 0.0


def test_failure_at_exact_time_horizon_terminates(make_env):
    env = make_env(
        customers=1,
        traffic={"enable_traffic": False, "enable_energy_uncertainty": False},
    )
    env.max_time = 2.0  # Inject the failure boundary after certifying the instance.
    task = env.task_registry.tasks()[0]
    task.earliest_service = 0.0
    task.latest_service = 1.5
    env.traffic_simulator.apply_traffic = lambda **kwargs: (2.0, 2.0)

    _, _, terminated, truncated, info = env.step((task.node_id, 0.0, False))

    assert env.global_clock == pytest.approx(env.max_time)
    assert terminated and not truncated
    assert info["termination_reason"] == "all_trucks_failed"
    assert info["failure_causes"] == {"time_window_violation_after_realization": 1}


def _legacy_flexible_env(customers):
    config = deepcopy(load_config("EVRoutingEnv/config_files/config_joint.yaml"))
    config["environment"].update(num_trucks=1, num_stops=customers)
    config["truck"]["battery_capacity"] = 10_000.0
    config["traffic"].update(
        enable_traffic=False, enable_energy_uncertainty=False
    )
    config["delivery"]["enable_stochastic_unloading"] = False
    env = EventDrivenTruckEnv(config, verbose=False, enable_plotting=False)
    env.reset(seed=7)
    env.transport_graph.get_path_energy = lambda a, b: float(a != b)
    env.transport_graph.get_time_distance = lambda a, b: float(a != b)
    return env


def test_legacy_flexible_does_not_count_unfinished_service():
    env = _legacy_flexible_env(customers=2)
    try:
        truck = env.trucks[0]
        customer = truck.get_remaining_deliveries()[0]
        env.max_time = 1.0 + env.delivery_simulator.base_unloading_time / 2

        _, _, terminated, truncated, info = env.step((customer, 0.0, False))

        assert truncated and not terminated
        assert env.truck_states[0] == "unloading"
        assert truck.completed_service_count == 0
        assert customer not in truck.delivered_nodes
        assert customer in truck.get_remaining_deliveries()
        assert info["operational_metrics"]["customers_served"] == 0
    finally:
        env.close()


def test_legacy_flexible_counts_served_customer_before_depot_return():
    env = _legacy_flexible_env(customers=1)
    try:
        truck = env.trucks[0]
        customer = truck.get_remaining_deliveries()[0]

        _, _, terminated, truncated, info = env.step((customer, 0.0, False))

        assert not terminated and not truncated
        assert not truck.is_complete
        assert truck.completed_service_count == 1
        assert customer in truck.delivered_nodes
        assert info["operational_metrics"]["customers_served"] == 1
        assert info["operational_metrics"]["completed_fraction"] == 1.0
        assert info["all_customers_served"]
    finally:
        env.close()
