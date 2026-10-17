"""Routing invariants across all variants, including fleets larger than workloads.

The sparse synthetic road graph keeps the fleet-size matrix inexpensive and
makes failures reproducible. Separate cases exercise the repository road data.
The action oracle uses task state and prescribed order, not the mask evaluator.
"""

from copy import deepcopy

import networkx as nx
import numpy as np
import pytest

from EVRoutingEnv.models.core.customer import TaskStatus
from EVRoutingEnv.models.environment import event_driven_env
from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.utils.utils import load_config


VARIANTS = ("joint", "preassigned", "sequential")
FLEET_SIZES = (1, 3, 10, 100, 128)
CUSTOMER_COUNTS = (1, 2, 5, 20, 25)


@pytest.fixture(scope="module")
def road_graph():
    # Any two leaves require two directed legs through hub 0. Customers and
    # stations are disjoint, while 150 customers still fit in this graph.
    graph = nx.DiGraph()
    for node in range(163):
        graph.add_node(
            node, has_charger=node >= 161, station_type="DCFast", total_capacity=2
        )
        if node:
            graph.add_edge(0, node, distance=2.0, time=0.125)
            graph.add_edge(node, 0, distance=2.0, time=0.125)
    return graph


@pytest.fixture
def make_routing_env(monkeypatch, road_graph):
    opened = []
    original_get_graph = event_driven_env.get_graph

    def build(
        variant,
        trucks=3,
        stops=5,
        seed=23,
        *,
        real_network=False,
        stochastic=False,
        **overrides,
    ):
        config = deepcopy(load_config("EVRoutingEnv/config_files/config_joint.yaml"))
        config["problem"].update(
            mode="sequential" if variant == "sequential" else "joint_fleet",
            assignment="fleet" if variant == "joint" else "preassigned",
            payload_capacity=max(20.0, stops * 4.0),
            min_customer_demand=1.0,
            max_customer_demand=4.0,
            base_service_time=0.125,
        )
        config["environment"].update(
            num_trucks=trucks,
            num_stops=stops,
            max_time=10000.0,
            invalid_action_mode="penalize",
        )
        config["truck"].update(battery_capacity=10000.0, initial_battery="full")
        config["traffic"].update(
            enable_traffic=stochastic, enable_energy_uncertainty=stochastic
        )
        config["delivery"]["enable_stochastic_unloading"] = stochastic
        for section, values in overrides.items():
            config[section].update(values)
        monkeypatch.setattr(
            event_driven_env,
            "get_graph",
            original_get_graph if real_network else lambda _: road_graph.copy(),
        )
        env = EventDrivenTruckEnv(config, verbose=False, enable_plotting=False)
        opened.append(env)
        env.reset(seed=seed)
        return env

    yield build
    for env in opened:
        env.close()


def expected_customers(env, variant):
    """Independent eligibility oracle for the currently active truck."""
    truck = env.trucks[env.active_truck_id]
    tasks = env.task_registry.tasks()
    owned = [task for task in tasks if task.preassigned_to == truck.truck_id]
    first = (
        min(
            (task.sequence_position for task in owned if not task.is_served),
            default=None,
        )
        if variant == "sequential"
        else None
    )
    return [
        task
        for task in tasks
        if task.status == TaskStatus.UNASSIGNED
        and (variant == "joint" or task.preassigned_to == truck.truck_id)
        and (variant != "sequential" or task.sequence_position == first)
        and task.demand <= truck.remaining_payload + 1e-9
        and task.node_id != truck.current_node
    ]


def assert_fleet_invariants(env, observation):
    assert env.observation_space.contains(observation)
    assert np.isfinite(observation).all()
    tasks = env.task_registry.tasks()
    served = [task for task in tasks if task.is_served]
    ids = [task_id for truck in env.trucks for task_id in truck.served_task_ids]
    assert len(ids) == len(set(ids)) == len(served)
    assert set(ids) == {task.task_id for task in served}
    assert sum(truck.completed_service_count for truck in env.trucks) == len(served)
    for truck in env.trucks:
        deliveries = [task for task in served if task.served_by == truck.truck_id]
        assert truck.payload_capacity - truck.remaining_payload == pytest.approx(
            sum(task.demand for task in deliveries)
        )
        assert truck.current_battery == pytest.approx(
            truck.battery_capacity
            + truck.total_energy_charged
            - truck.total_energy_consumed
        )
        assert -1e-8 <= truck.current_battery <= truck.battery_capacity + 1e-8
        assert 0 <= truck.remaining_payload <= truck.payload_capacity
        assert not (truck.is_complete and truck.failed)
    for index, event in enumerate(env.event_queue):
        assert event.time >= env.global_clock
        if index:
            assert env.event_queue[(index - 1) // 2] <= event


def drive_deliveries(env, variant, seed=99):
    rng = np.random.default_rng(seed)
    previous_time = env.global_clock
    rewards = 0.0
    trace = []
    for step in range(env.num_stops + env.num_trucks + 1):
        assert env.active_truck_id is not None
        truck = env.trucks[env.active_truck_id]
        mask = env.mask_fn()
        eligible = expected_customers(env, variant)
        customer_indices = np.flatnonzero(
            mask[env.num_charging_nodes : env.num_charging_nodes + env.num_stops]
        )
        assert set(customer_indices) == {task.task_id for task in eligible}
        if eligible:
            task = eligible[int(rng.integers(len(eligible)))]
            slot, node = task.task_id, task.node_id
        else:
            slot, node = env.fixed_num_stops, env.joint_instance.depot_node
            if variant == "sequential":
                assert all(
                    task.is_served
                    for task in env.task_registry.tasks()
                    if task.preassigned_to == truck.truck_id
                )
            else:
                assert env.task_registry.all_served()
        index = env.num_charging_nodes + slot
        assert mask[index]
        trace.append((env.active_truck_id, index, env.global_clock))
        action = index if step % 2 == 0 else (node, 0.0, False)
        observation, reward, terminated, truncated, info = env.step(action)
        rewards += reward
        assert not truncated and env.global_clock >= previous_time
        previous_time = env.global_clock
        assert_fleet_invariants(env, observation)
        assert info["invalid_action_count"] == 0
        if terminated:
            assert info["successful"] and info["termination_reason"] == "success"
            assert info["task_counts"]["served"] == env.num_stops
            assert all(
                truck.is_complete
                and truck.current_node == env.joint_instance.depot_node
                for truck in env.trucks
            )
            metrics = info["operational_metrics"]
            assert (
                metrics["customers_total"]
                == metrics["customers_served"]
                == env.num_stops
            )
            assert 1 <= metrics["vehicles_used"] <= min(env.num_trucks, env.num_stops)
            assert metrics["fleet_makespan"] == env.global_clock
            assert rewards == pytest.approx(env.episode_reward)
            assert rewards == pytest.approx(
                env.num_stops * env.reward_config["delivery_bonus"]
                - metrics["total_operating_time"] * env.reward_config["time_multiplier"]
            )
            for _ in range(2):
                after = env.step(0)
                assert after[1:4] == (0.0, True, False)
                assert after[4]["operational_metrics"] == metrics
            return trace, info
    pytest.fail("episode exceeded the number of customer and depot decisions")


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("trucks", FLEET_SIZES)
@pytest.mark.parametrize("stops", CUSTOMER_COUNTS)
@pytest.mark.parametrize("seed,stochastic", [(23, False), (71, True)])
def test_routing_variant_fleet_and_customer_matrix(
    make_routing_env, variant, trucks, stops, seed, stochastic
):
    env = make_routing_env(variant, trucks, stops, seed, stochastic=stochastic)
    assert len(env.task_registry) == stops
    assert len({task.node_id for task in env.task_registry.tasks()}) == stops
    drive_deliveries(env, variant, seed)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("trucks,stops", [(1, 25), (100, 25), (128, 150)])
def test_many_deliveries_and_large_fleets_replay_exactly(
    make_routing_env, variant, trucks, stops
):
    env = make_routing_env(variant, trucks, stops, stochastic=True)
    first_scenario = deepcopy(env.scenario_descriptor)
    first_trace, first_info = drive_deliveries(env, variant)
    env.reset(seed=23)
    assert env.scenario_descriptor == first_scenario
    second_trace, second_info = drive_deliveries(env, variant)
    assert second_trace == first_trace
    assert second_info["customer_tasks"] == first_info["customer_tasks"]
    assert second_info["operational_metrics"] == first_info["operational_metrics"]


@pytest.mark.integration
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("trucks,stops", [(1, 1), (3, 25), (128, 25)])
def test_variants_on_repository_road_network(make_routing_env, variant, trucks, stops):
    env = make_routing_env(variant, trucks, stops, real_network=True, stochastic=True)
    drive_deliveries(env, variant)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("trucks", [3, 128])
@pytest.mark.parametrize("integer", [False, True])
def test_ownership_and_order_rejections_preserve_the_decision(
    make_routing_env, variant, trucks, integer
):
    env = make_routing_env(variant, trucks, 25)
    truck_id = env.active_truck_id
    tasks = env.task_registry.tasks()
    if variant == "joint":
        # A served task may not be visited again, even by its serving truck.
        first = tasks[0]
        env.step((first.node_id, 0.0, False))
        rejected = first
    else:
        rejected = next(task for task in tasks if task.preassigned_to != truck_id)
    observation = env._get_observation().copy()
    events = deepcopy(env.event_queue)
    snapshot = env.task_registry.snapshot()
    active, clock = env.active_truck_id, env.global_clock
    index = env.num_charging_nodes + rejected.task_id
    assert not env.mask_fn()[index]
    action = index if integer else (rejected.node_id, 0.0, False)
    after, reward, terminated, truncated, info = env.step(action)
    assert not terminated and not truncated
    assert reward == -env.reward_config["invalid_action_penalty"]
    assert np.array_equal(observation, after)
    assert env.event_queue == events and env.task_registry.snapshot() == snapshot
    assert (env.active_truck_id, env.global_clock) == (active, clock)
    assert info["invalid_action_count"] == 1


@pytest.mark.parametrize("variant", VARIANTS)
def test_only_sequential_mode_restricts_customer_order(make_routing_env, variant):
    env = make_routing_env(variant, 1, 25)
    last = env.task_registry.tasks()[-1]
    action = env.num_charging_nodes + last.task_id
    if variant == "sequential":
        assert not env.mask_fn()[action]
        env.step((last.node_id, 0.0, False))
        assert not last.is_served and last.claimed_by is None
    else:
        assert env.mask_fn()[action]
        env.step((last.node_id, 0.0, False))
        assert last.is_served and last.served_by == 0


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("trucks", [1, 128])
@pytest.mark.parametrize("horizon", [0.125, 0.3125, 0.375, 0.5, 0.625])
def test_horizon_during_travel_service_and_exact_depot_arrival(
    make_routing_env, variant, trucks, horizon
):
    env = make_routing_env(variant, trucks, 1, environment={"max_time": horizon})
    task = expected_customers(env, variant)[0]
    truck = env.trucks[env.active_truck_id]
    # Seed 23 chooses two leaves, so the independent known road cost is 0.25h.
    assert (
        env.transport_graph.get_time_distance(truck.current_node, task.node_id) == 0.25
    )
    observation, _, terminated, truncated, info = env.step(
        env.num_charging_nodes + task.task_id
    )
    if not (terminated or truncated):
        observation, _, terminated, truncated, info = env.step(
            env.num_charging_nodes + env.fixed_num_stops
        )
    assert env.global_clock == horizon
    assert_fleet_invariants(env, observation)
    assert info["task_counts"]["served"] == int(horizon >= 0.375)
    assert info["successful"] == (horizon == 0.625)
    assert (terminated, truncated) == (
        (True, False) if horizon == 0.625 else (False, True)
    )
    if truncated:
        assert info["termination_reason"] == "time_limit_exhausted"
        assert env.event_queue or horizon == 0.375
    frozen = deepcopy(info["operational_metrics"])
    again = env.step(0)
    assert again[1:4] == (0.0, terminated, truncated)
    assert again[4]["operational_metrics"] == frozen


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("limit", ["steps", "invalid_actions"])
def test_large_fleet_limits_latch_and_reset_cleanly(make_routing_env, variant, limit):
    setting = (
        {"max_episode_steps": 1} if limit == "steps" else {"invalid_action_budget": 2}
    )
    env = make_routing_env(variant, 128, 25, environment=setting)
    if limit == "steps":
        task = expected_customers(env, variant)[0]
        result = env.step(env.num_charging_nodes + task.task_id)
        assert result[2:4] == (False, True)
        reason = "step_limit_exhausted"
    else:
        env.step(env.num_navigation_actions)
        result = env.step(env.num_navigation_actions)
        assert result[2:4] == (True, False)
        reason = "invalid_action_budget_exhausted"
    assert result[4]["termination_reason"] == reason
    assert not result[4]["successful"]
    for _ in range(2):
        again = env.step(0)
        assert again[1] == 0.0 and again[2:4] == result[2:4]
        assert np.array_equal(again[0], result[0])
    env.reset(seed=23)
    assert env.episode_steps == env.invalid_action_count == 0
    assert env.global_clock == 0 and env.termination_reason is None
    assert all(
        task.status == TaskStatus.UNASSIGNED for task in env.task_registry.tasks()
    )


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("trucks", [1, 3, 128])
def test_tight_unit_demands_fit_without_payload_deadlock(
    make_routing_env, variant, trucks
):
    stops = 25
    capacity = float((stops + trucks - 1) // trucks)
    env = make_routing_env(
        variant,
        trucks,
        stops,
        problem={
            "payload_capacity": capacity,
            "min_customer_demand": 1.0,
            "max_customer_demand": 1.0,
        },
    )
    drive_deliveries(env, variant)


@pytest.mark.parametrize("variant", VARIANTS)
def test_large_fleet_rejects_impossible_total_demand(make_routing_env, variant):
    with pytest.raises(ValueError, match="capacity|partition"):
        make_routing_env(
            variant,
            128,
            129,
            problem={
                "payload_capacity": 1.0,
                "min_customer_demand": 1.0,
                "max_customer_demand": 1.0,
            },
        )


@pytest.mark.parametrize("variant", VARIANTS)
def test_hundred_trucks_wait_for_windows_without_early_service(
    make_routing_env, variant
):
    env = make_routing_env(variant, 128, 25)
    depot = env.joint_instance.depot_node
    tasks = env.task_registry.tasks()
    expected_wait = sum(
        5.0 - env.transport_graph.get_time_distance(depot, task.node_id)
        for task in tasks
    )
    for task in tasks:
        task.earliest_service, task.latest_service = 5.0, 10.0
    _, info = drive_deliveries(env, variant)
    assert all(task.service_started_at == 5.0 for task in tasks)
    assert all(task.served_at == 5.125 for task in tasks)
    assert info["operational_metrics"]["total_time_window_waiting"] == pytest.approx(
        expected_wait
    )


@pytest.mark.parametrize("variant", VARIANTS)
def test_unused_trucks_do_not_receive_terminal_battery_penalties(
    make_routing_env, variant
):
    env = make_routing_env(
        variant,
        128,
        1,
        rewards={
            "enable_leftover_battery_penalty": True,
            "leftover_battery_penalty_coef": 2.0,
        },
    )
    task = expected_customers(env, variant)[0]
    used = env.trucks[env.active_truck_id]
    first = env.step(env.num_charging_nodes + task.task_id)
    final = env.step(env.num_charging_nodes + env.fixed_num_stops)
    assert final[2:4] == (True, False) and final[4]["successful"]
    expected = (
        env.reward_config["delivery_bonus"]
        - final[4]["operational_metrics"]["total_operating_time"]
    )
    expected -= 2.0 * 100.0 * used.current_battery / used.battery_capacity
    assert first[1] + final[1] == pytest.approx(expected)
    unused = [truck for truck in env.trucks if truck is not used]
    assert all(
        truck.completed_service_count == 0 and truck.total_distance_traveled == 0
        for truck in unused
    )
    assert final[4]["operational_metrics"]["vehicles_used"] == 1


def assert_station_invariants(env):
    for node in env.charging_nodes:
        occupancy = env.charging_station.charger_occupancy[node]
        waiting = [
            entry["truck_id"] for entry in env.charging_station.charger_waitlist[node]
        ]
        assert (
            len(occupancy)
            == len(set(occupancy))
            <= env.charging_station.charger_capacity[node]
        )
        assert len(waiting) == len(set(waiting))
        assert not set(occupancy).intersection(waiting)
        assert env.charging_station.pending_wake_trucks[node] <= set(waiting)
        assert all(env.truck_states[truck_id] == "charging" for truck_id in occupancy)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize(
    "trucks,ports,charging",
    [(3, 1, "target_soc"), (128, 1, "target_soc"), (128, 3, "duration")],
)
def test_simultaneous_charger_arrivals_obey_fcfs_and_port_capacity(
    make_routing_env, variant, trucks, ports, charging
):
    env = make_routing_env(
        variant,
        trucks,
        trucks,
        truck={"battery_capacity": 12.0},
        charging={
            "action_mode": charging,
            "charge_durations": [0.05],
            "station_power_classes_kw": [100.0],
        },
        problem={
            "payload_capacity": 1.0,
            "min_customer_demand": 1.0,
            "max_customer_demand": 1.0,
        },
    )
    charger = env.charging_nodes[0]
    env.charging_station.charger_capacity[charger] = ports
    # Every truck reaches the same station at the same virtual time. No direct
    # mutation of truck state or insertion of synthetic readiness events.
    for truck_id in range(trucks):
        assert env.active_truck_id == truck_id
        result = env.step(0)
        assert not result[2] and not result[3]
    peak_waiting = 0
    for _ in range(3 * trucks + 1):
        assert_station_invariants(env)
        peak_waiting = max(
            peak_waiting, len(env.charging_station.charger_waitlist[charger])
        )
        truck = env.trucks[env.active_truck_id]
        if truck.current_node == charger and truck.num_charging_sessions == 0:
            action = env.num_navigation_actions + (len(env.charge_action_values) - 1)
        else:
            eligible = expected_customers(env, variant)
            action = env.num_charging_nodes + (
                eligible[0].task_id if eligible else env.fixed_num_stops
            )
        assert env.mask_fn()[action]
        observation, _, terminated, truncated, info = env.step(action)
        assert not truncated and info["invalid_action_count"] == 0
        assert_fleet_invariants(env, observation)
        if terminated:
            assert info["successful"]
            break
    else:
        pytest.fail("charger queue or route did not drain")
    assert peak_waiting >= trucks - ports
    starts = [
        truck_id
        for _, truck_id, kind in env.charging_station.queue_history[charger][
            "truck_events"
        ]
        if kind == "start"
    ]
    assert starts == list(range(trucks))
    assert all(truck.num_charging_sessions == 1 for truck in env.trucks)
    assert info["operational_metrics"]["charging_sessions"] == trucks
    assert info["operational_metrics"]["total_queue_time"] > 0
    assert all(
        not env.charging_station.charger_waitlist[node]
        and not env.charging_station.charger_occupancy[node]
        and not env.charging_station.pending_wake_trucks[node]
        for node in env.charging_nodes
    )


@pytest.mark.parametrize("variant", VARIANTS)
def test_station_closure_releases_a_hundred_waiting_trucks(make_routing_env, variant):
    from EVRoutingEnv.baselines.energy_path import EnergyPathHeuristicPolicy

    env = make_routing_env(
        variant,
        128,
        128,
        truck={"battery_capacity": 16.0},
        charging={"station_power_classes_kw": [100.0]},
    )
    charger = env.charging_nodes[0]
    env.charging_station.charger_capacity[charger] = 1
    for _ in range(128):
        env.step(0)
    env.step(env.num_navigation_actions + len(env.charge_action_values) - 1)
    assert len(env.charging_station.charger_waitlist[charger]) == 127
    env.set_charger_available(charger, False)
    assert not env.charging_station.charger_waitlist[charger]
    policy = EnergyPathHeuristicPolicy()
    for _ in range(1024):
        assert not env.mask_fn()[0]
        action = policy(env)
        assert env.mask_fn()[action]
        observation, _, terminated, truncated, info = env.step(action)
        assert not truncated and info["invalid_action_count"] == 0
        assert_fleet_invariants(env, observation)
        assert_station_invariants(env)
        if terminated:
            assert info["successful"]
            break
    else:
        pytest.fail("released trucks did not finish their routes")
    assert not env.waiting_start_times


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("trucks", [1, 128])
def test_low_battery_routes_require_and_execute_recharging(
    make_routing_env, variant, trucks
):
    from EVRoutingEnv.baselines.energy_path import EnergyPathHeuristicPolicy

    env = make_routing_env(variant, trucks, 25, truck={"battery_capacity": 12.0})
    # One truck must recharge repeatedly. With 128 trucks, start below full SoC
    # so the return cannot fit without replenishing. This also checks initial
    # battery percentages independently of fleet capacity.
    if trucks == 128:
        env.config["truck"]["initial_battery"] = 50.0
        env.reset(seed=23)
    initial = {truck.truck_id: truck.current_battery for truck in env.trucks}
    policy = EnergyPathHeuristicPolicy()
    for _ in range(1024):
        action = policy(env)
        assert env.mask_fn()[action]
        observation, _, terminated, truncated, info = env.step(action)
        assert not truncated and info["invalid_action_count"] == 0
        assert env.observation_space.contains(observation)
        for truck in env.trucks:
            assert truck.current_battery == pytest.approx(
                initial[truck.truck_id]
                + truck.total_energy_charged
                - truck.total_energy_consumed
            )
        assert_station_invariants(env)
        if terminated:
            assert info["successful"]
            assert info["operational_metrics"]["charging_sessions"] > 0
            break
    else:
        pytest.fail("energy-feasible policy did not finish within its charging budget")


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("trucks", [1, 128])
def test_zero_duration_service_with_simultaneous_fleet_events(
    make_routing_env, variant, trucks
):
    env = make_routing_env(variant, trucks, 25, problem={"base_service_time": 0.0})
    _, info = drive_deliveries(env, variant)
    assert info["operational_metrics"]["total_service_time"] == 0.0
    assert all(
        task.served_at == task.service_started_at for task in env.task_registry.tasks()
    )


@pytest.mark.parametrize("variant", VARIANTS)
def test_variable_customer_counts_preserve_large_fleet_action_slots(
    make_routing_env, variant
):
    env = make_routing_env(
        variant, 128, 25, environment={"allow_variable_num_stops": True}
    )
    shape, actions = env.observation_space.shape, env.action_space.n
    counts = set()
    for seed in range(5):
        env.reset(seed=seed)
        counts.add(env.num_stops)
        assert env.observation_space.shape == shape and env.action_space.n == actions
        assert len(env.task_registry) == env.num_stops
        padding = env.mask_fn()[
            env.num_charging_nodes + env.num_stops : env.num_charging_nodes
            + env.fixed_num_stops
        ]
        assert not padding.any()
        drive_deliveries(env, variant)
    assert len(counts) > 1


@pytest.mark.parametrize("variant", VARIANTS)
def test_exact_zero_energy_arrival_at_charger_can_recover(make_routing_env, variant):
    from EVRoutingEnv.baselines.energy_path import EnergyPathHeuristicPolicy

    env = make_routing_env(
        variant, 1, 5, truck={"battery_capacity": 12.0, "initial_battery": 100.0 / 3}
    )
    initial = env.trucks[0].current_battery
    assert initial == pytest.approx(4.0)
    assert env.mask_fn()[0]
    _, _, terminated, truncated, _ = env.step(0)
    truck = env.trucks[0]
    assert not terminated and not truncated
    assert truck.current_battery == pytest.approx(0.0) and not truck.failed
    assert env.mask_fn()[env.num_navigation_actions :].any()
    policy = EnergyPathHeuristicPolicy()
    for _ in range(64):
        action = policy(env)
        assert env.mask_fn()[action]
        _, _, terminated, truncated, info = env.step(action)
        assert not truncated
        assert truck.current_battery == pytest.approx(
            initial + truck.total_energy_charged - truck.total_energy_consumed
        )
        if terminated:
            assert info["successful"] and truck.num_charging_sessions > 0
            break
    else:
        pytest.fail("truck did not recover from exact-zero charger arrival")


@pytest.mark.parametrize("variant", VARIANTS)
def test_charge_cutoff_never_commits_future_energy_and_reset_clears_station(
    make_routing_env, variant
):
    env = make_routing_env(variant, 1, 5, truck={"battery_capacity": 12.0})
    env.step(0)
    truck = env.trucks[0]
    battery = truck.current_battery
    env.max_time = env.global_clock + 0.001
    result = env.step(env.num_navigation_actions + len(env.charge_action_values) - 1)
    assert result[2:4] == (False, True)
    assert truck.is_charging and truck.num_charging_sessions == 0
    assert truck.current_battery == battery and truck.total_energy_charged == 0
    metrics = result[4]["operational_metrics"]
    assert metrics["total_charging_time"] == pytest.approx(0.001)
    assert env._get_info()["operational_metrics"] == metrics
    env.set_charger_available(env.charging_nodes[0], False)
    env.max_time = 10000.0
    env.reset(seed=23)
    assert env.global_clock == 0.0 and not env.waiting_start_times
    assert not any(truck.is_charging for truck in env.trucks)
    assert all(env.charging_station.station_available.values())
    assert_station_invariants(env)
    assert all(not queue for queue in env.charging_station.charger_occupancy.values())


@pytest.mark.parametrize("variant", VARIANTS)
def test_realized_lateness_releases_claim_without_service_or_payload_use(
    make_routing_env, variant, monkeypatch
):
    env = make_routing_env(variant, 1, 5)
    truck = env.trucks[0]
    task = expected_customers(env, variant)[0]
    task.latest_service = 1.0
    monkeypatch.setattr(env.traffic_simulator, "apply_traffic", lambda **_: (5.0, 1.0))
    assert env.mask_fn()[env.num_charging_nodes + task.task_id]
    _, reward, terminated, truncated, info = env.step(
        env.num_charging_nodes + task.task_id
    )
    assert terminated and not truncated and not info["successful"]
    assert truck.failure_reason == "time_window_violation_after_realization"
    assert task.status == TaskStatus.UNASSIGNED and task.claimed_by is None
    assert task.served_at is None and truck.remaining_payload == truck.payload_capacity
    assert info["task_counts"]["served"] == 0
    assert reward == pytest.approx(
        env.reward_config["failure_penalty"]
        - 5.0 * env.reward_config["time_multiplier"]
    )
    assert env.step(0)[1] == 0.0


@pytest.mark.parametrize("variant", VARIANTS)
def test_large_fleet_canonical_encoders_and_gnn_masks_remain_consistent(
    make_routing_env, variant
):
    import torch
    from algo.canonical_encoders import STATE_ENCODER_TYPES, build_state_encoder
    from algo.canonical_state import unpack_flat_observation
    from EVRoutingEnv.state.gnn_utils import create_default_gnn_space

    env = make_routing_env(variant, 128, 150)
    tensors = unpack_flat_observation(
        torch.from_numpy(env._get_observation()).unsqueeze(0), env.canonical_shape
    )
    for name in STATE_ENCODER_TYPES:
        encoder = build_state_encoder(name, env.canonical_shape, 16, 24).eval()
        encoder.observe(tensors)
        with torch.no_grad():
            encoded = encoder(tensors)
        assert encoded.shape == (1, 24) and torch.isfinite(encoded).all()
    graph = create_default_gnn_space(env).get_state_GNN(env)
    assert len(graph.action_to_node_map) == env.action_space.n
    assert np.array_equal(graph.feasible_action_mask.cpu().numpy(), env.mask_fn())
    assert graph.action_to_node_map[
        env.num_charging_nodes : env.num_charging_nodes + env.num_stops
    ] == [(task.node_id, False) for task in env.joint_instance.tasks]
    assert torch.isfinite(graph.action_graph_features).all()


@pytest.mark.integration
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("trucks", [1, 128])
def test_repository_network_at_standard_battery_capacity(
    make_routing_env, variant, trucks
):
    from EVRoutingEnv.baselines.energy_path import EnergyPathHeuristicPolicy

    env = make_routing_env(
        variant,
        trucks,
        25,
        real_network=True,
        stochastic=True,
        truck={"battery_capacity": 400.0},
    )
    policy = EnergyPathHeuristicPolicy()
    for _ in range(env.max_episode_steps):
        action = policy(env)
        assert env.mask_fn()[action]
        observation, _, terminated, truncated, info = env.step(action)
        assert not truncated and info["invalid_action_count"] == 0
        assert_fleet_invariants(env, observation)
        assert_station_invariants(env)
        if terminated:
            assert info["successful"]
            assert info["task_counts"]["served"] == 25
            assert all(
                truck.current_node == env.joint_instance.depot_node
                for truck in env.trucks
            )
            assert info["operational_metrics"]["charging_sessions"] > 0
            return
    pytest.fail("reference dispatcher did not complete on the repository network")


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("trucks", [1, 128])
def test_failed_trucks_never_receive_tasks_or_duplicate_failure_penalties(
    make_routing_env, variant, trucks
):
    env = make_routing_env(
        variant, trucks, 25, environment={"invalid_action_mode": "terminate"}
    )
    total_reward = 0.0
    failed = set()
    for _ in range(trucks):
        actor = env.active_truck_id
        assert actor not in failed
        failed.add(actor)
        _, reward, terminated, truncated, info = env.step(env.num_navigation_actions)
        total_reward += reward
        assert not truncated and env.trucks[actor].failed
        assert info["task_counts"]["served"] == 0
        if terminated:
            assert not info["successful"]
            break
    else:
        pytest.fail("fleet with no surviving owners did not terminate")
    # When every assigned owner has failed, the scheduler also marks the
    # remaining waiting fleet as deadlocked. Price every actual failure once.
    actual_failures = sum(truck.failed for truck in env.trucks)
    assert actual_failures >= len(failed)
    assert total_reward == actual_failures * env.reward_config["failure_penalty"]
    assert all(
        not task.is_served and task.claimed_by is None
        for task in env.task_registry.tasks()
    )
    assert env.step(0)[1] == 0.0
    env.reset(seed=23)
    assert not any(truck.failed for truck in env.trucks)
    assert env.episode_reward == 0.0


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize(
    "malformation",
    [
        "fractional_node",
        "boolean_node",
        "string_node",
        "string_flag",
        "integer_flag",
        "missing_value",
        "boolean_value",
        "empty",
        "short",
        "long",
    ],
)
def test_malformed_tuple_actions_cannot_be_coerced_into_other_actions(
    make_routing_env, variant, malformation
):
    env = make_routing_env(variant)
    task = expected_customers(env, variant)[0]
    node, value, charge = task.node_id, 0.0, False
    if malformation == "fractional_node":
        node += 0.5
    elif malformation == "boolean_node":
        node = True
    elif malformation == "string_node":
        node = str(node)
    elif malformation == "string_flag":
        charge = "False"
    elif malformation == "integer_flag":
        charge = 0
    elif malformation == "missing_value":
        value = None
    else:
        value = True
    action = (node, value, charge)
    if malformation == "empty":
        action = ()
    elif malformation == "short":
        action = (node, value)
    elif malformation == "long":
        action = (node, value, charge, None)
    before = env._get_observation().copy()
    events, tasks = deepcopy(env.event_queue), env.task_registry.snapshot()
    with pytest.raises(ValueError):
        env.step(action)
    assert np.array_equal(before, env._get_observation())
    assert events == env.event_queue and tasks == env.task_registry.snapshot()
    assert env.episode_steps == env.invalid_action_count == 0


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("charging", ["target_soc", "duration"])
@pytest.mark.parametrize("value", [-1.0, 0.0, float("nan"), float("inf")])
def test_invalid_charge_values_are_refused_without_progress(
    make_routing_env, variant, charging, value
):
    # Use a single truck to reach a decision at the station immediately.
    env = make_routing_env(variant, 1, 5, charging={"action_mode": charging})
    env.step(0)
    before = env._get_observation().copy()
    truck = env.trucks[0]
    clock, events = env.global_clock, deepcopy(env.event_queue)
    after, reward, terminated, truncated, info = env.step(
        (truck.current_node, value, True)
    )
    assert not terminated and not truncated
    assert reward == -env.reward_config["invalid_action_penalty"]
    assert np.array_equal(before, after)
    assert env.global_clock == clock and env.event_queue == events
    assert info["invalid_action_count"] == 1 and truck.num_charging_sessions == 0


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("numpy_scalars", [False, True])
def test_well_typed_tuple_actions_accept_python_and_numpy_scalars(
    make_routing_env, variant, numpy_scalars
):
    env = make_routing_env(variant, 1, 5)
    task = expected_customers(env, variant)[0]
    action = (
        (np.int64(task.node_id), np.float32(0.0), np.bool_(False))
        if numpy_scalars
        else (task.node_id, 0, False)
    )
    _, _, terminated, truncated, info = env.step(action)
    assert not terminated and not truncated
    assert task.is_served and info["invalid_action_count"] == 0
