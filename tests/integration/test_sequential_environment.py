"""Behavioral contracts for fixed routes on the shared fleet engine."""

from copy import deepcopy

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from EVRoutingEnv.models.core.customer import CustomerTask, FleetTaskRegistry, TaskStatus
from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.state.features import CUSTOMER_FEATURES, extract_canonical_features
from EVRoutingEnv.state.feasibility import FeasibilityReason, joint_action_feasibility
from EVRoutingEnv.state.gnn_utils import create_default_gnn_space
from EVRoutingEnv.utils.utils import load_config


@pytest.fixture
def environment():
    opened = []

    def build(mode="sequential", trucks=1, stops=4, seed=17, charging="target_soc", stochastic=False):
        config = deepcopy(load_config("EVRoutingEnv/config_files/config_sequential.yaml"))
        config["problem"]["mode"] = mode
        if mode == "joint_fleet":
            config["problem"]["assignment"] = "fleet"
        config["environment"].update(num_trucks=trucks, num_stops=stops, max_time=10000, invalid_action_mode="penalize")
        config["truck"]["battery_capacity"] = 100000.0
        config["traffic"].update(enable_traffic=stochastic, enable_energy_uncertainty=stochastic)
        config["delivery"]["enable_stochastic_unloading"] = stochastic
        config["charging"]["action_mode"] = charging
        env = EventDrivenTruckEnv(config, enable_plotting=False)
        opened.append(env)
        env.reset(seed=seed)
        return env

    yield build
    for env in opened:
        env.close()


def finish_routes(env):
    for _ in range(env.fixed_num_stops + env.num_trucks + 1):
        truck_id = env.active_truck_id
        assert truck_id is not None
        if env.sequential_routing:
            task = env.task_registry.next_task_for_truck(truck_id)
        else:
            tasks = env.task_registry.available_tasks(env.trucks[truck_id].remaining_payload, truck_id=truck_id)
            task = tasks[0] if tasks else None
        action = env.num_charging_nodes + (task.task_id if task else env.fixed_num_stops)
        assert env.mask_fn()[action]
        _, _, terminated, truncated, info = env.step(action)
        assert not truncated
        if terminated:
            assert info["successful"]
            return info
    raise AssertionError("route did not finish within customer and return decisions")


@pytest.mark.parametrize("mode", ["joint_fleet", "sequential"])
@pytest.mark.parametrize("trucks,stops", [(1, 4), (3, 7), (4, 1)])
@pytest.mark.parametrize("stochastic", [False, True])
def test_shared_engine_finishes_routes_and_returns_to_depot(environment, mode, trucks, stops, stochastic):
    env = environment(mode, trucks, stops, stochastic=stochastic)
    assert len(env.task_registry) == stops
    info = finish_routes(env)
    assert info["task_counts"]["served"] == stops
    assert sum(truck.completed_service_count for truck in env.trucks) == stops
    assert all(truck.is_complete and truck.current_node == env.joint_instance.depot_node for truck in env.trucks)
    if env.sequential_routing:
        for truck in env.trucks:
            tasks = [task for task in env.task_registry.tasks() if task.preassigned_to == truck.truck_id]
            assert sum(task.demand for task in tasks) <= truck.payload_capacity
            assert [task.sequence_position for task in tasks] == list(range(len(tasks)))
            assert all(task.served_by == truck.truck_id for task in tasks)
            assert [task.served_at for task in tasks] == sorted(task.served_at for task in tasks)


@pytest.mark.parametrize("integer", [False, True])
def test_later_customer_cannot_bypass_sequence_or_mutate_state(environment, integer):
    env = environment()
    task = env.task_registry.tasks()[1]
    index = env.num_charging_nodes + task.task_id
    assert joint_action_feasibility(env)[index].reason == FeasibilityReason.OUT_OF_SEQUENCE
    snapshot = env._get_observation().copy()
    events = list(env.event_queue)
    clock = env.global_clock
    _, reward, terminated, truncated, _ = env.step(index if integer else (task.node_id, 0.0, False))
    assert not terminated and not truncated and reward < 0
    assert env.event_queue == events and env.global_clock == clock
    assert not np.array_equal(snapshot, env._get_observation())  # Budgets changed.
    assert task.status == TaskStatus.UNASSIGNED
    with pytest.raises(ValueError, match="out of sequence"):
        env.task_registry.claim(task.node_id, env.active_truck_id, env.global_clock)


def test_sequential_ownership_and_order_are_observable_and_seeded(environment):
    env = environment(trucks=2, stops=5)
    first = [task.as_dict() for task in env.task_registry.tasks()]
    features = extract_canonical_features(env)
    for row, task in zip(features.customer_features, env.task_registry.tasks(), strict=True):
        assert row[CUSTOMER_FEATURES.index("preassigned_to")] == task.preassigned_to
        assert row[CUSTOMER_FEATURES.index("sequence_position")] == task.sequence_position
    env.reset(seed=17)
    assert [task.as_dict() for task in env.task_registry.tasks()] == first


@pytest.mark.parametrize("mode", ["joint_fleet", "sequential"])
def test_single_truck_mode_satisfies_gym_contract(environment, mode):
    check_env(environment(mode), skip_render_check=True)


@pytest.mark.parametrize("charging", ["target_soc", "duration"])
def test_sequential_gnn_global_slots_and_masks_follow_active_truck(environment, charging):
    env = environment(trucks=2, stops=5, charging=charging)
    adapter = create_default_gnn_space(env)
    for _ in range(3):
        graph = adapter.get_state_GNN(env)
        assert graph.action_to_node_map[env.num_charging_nodes:env.num_charging_nodes + env.num_stops] == [(task.node_id, False) for task in env.joint_instance.tasks]
        assert np.array_equal(graph.feasible_action_mask.cpu().numpy(), env.mask_fn())
        task = env.task_registry.next_task_for_truck(env.active_truck_id)
        assert task is not None
        env.step(env.num_charging_nodes + task.task_id)


def test_finishing_truck_can_return_before_other_routes_are_served(environment):
    env = environment(trucks=2, stops=4)
    task = env.task_registry.next_task_for_truck(0)
    task.base_service_time = 500.0
    observed_independent_return = False
    for _ in range(8):
        truck_id = env.active_truck_id
        task = env.task_registry.next_task_for_truck(truck_id)
        if task is None and not env.task_registry.all_served():
            observed_independent_return = True
            assert env.mask_fn()[env.num_charging_nodes + env.fixed_num_stops]
        action = env.num_charging_nodes + (task.task_id if task else env.fixed_num_stops)
        _, _, terminated, truncated, info = env.step(action)
        assert not truncated
        if terminated:
            assert info["successful"]
            break
    assert observed_independent_return


def test_truncated_service_does_not_unlock_next_customer(environment):
    env = environment(stops=2)
    task = env.task_registry.tasks()[0]
    travel = env.transport_graph.get_time_distance(env.trucks[0].current_node, task.node_id)
    task.base_service_time = 10.0
    env.max_time = travel + 1.0
    _, _, terminated, truncated, _ = env.step(env.num_charging_nodes)
    assert truncated and not terminated
    assert task.status == TaskStatus.IN_SERVICE
    assert env.task_registry.next_task_for_truck(0) is task
    assert not env.task_registry.available_tasks(truck_id=0)


@pytest.mark.parametrize("positions", [[0, 0], [0, 2]])
def test_sequential_registry_rejects_ambiguous_routes(positions):
    tasks = [CustomerTask(i, i + 1, 1, 0.1, preassigned_to=0, sequence_position=position) for i, position in enumerate(positions)]
    with pytest.raises(ValueError, match="unique and contiguous"):
        FleetTaskRegistry(tasks, sequential=True)


def test_removed_flexible_flag_has_clear_migration_error():
    config = load_config("EVRoutingEnv/config_files/config_joint.yaml")
    config["delivery"]["enable_flexible_delivery_order"] = True
    with pytest.raises(ValueError, match="was removed; use problem.mode"):
        EventDrivenTruckEnv(config, enable_plotting=False)


def test_one_truck_joint_keeps_every_customer_available(environment):
    env = environment("joint_fleet")
    assert env.mask_fn()[env.num_charging_nodes:env.num_charging_nodes + env.num_stops].all()
    last = env.task_registry.tasks()[-1]
    _, _, terminated, truncated, _ = env.step(env.num_charging_nodes + last.task_id)
    assert not terminated and not truncated
    assert last.is_served and last.served_by == 0


@pytest.mark.parametrize("charging", ["target_soc", "duration"])
def test_sequential_charging_execution_uses_configured_semantics(environment, charging):
    env = environment(charging=charging)
    truck = env.trucks[0]
    truck.current_node = env.charging_nodes[0]
    truck.current_battery = 0.4 * truck.battery_capacity
    initial = truck.current_battery
    value = 0.6 if charging == "target_soc" else 0.5
    _, _, terminated, truncated, _ = env.step((truck.current_node, value, True))
    assert not terminated and not truncated
    assert truck.current_battery > initial
    assert truck.total_charging_time > 0
    if charging == "target_soc":
        assert truck.current_battery / truck.battery_capacity == pytest.approx(value)
    else:
        assert truck.total_charging_time == pytest.approx(value)
    assert not env.mask_fn()[env.num_navigation_actions:].any()
    finish_routes(env)


def test_sequential_time_window_wait_commits_before_next_task(environment):
    env = environment()
    task = env.task_registry.tasks()[0]
    travel = env.transport_graph.get_time_distance(env.trucks[0].current_node, task.node_id)
    task.earliest_service = travel + 5.0
    task.latest_service = task.earliest_service + 10.0
    env.step(env.num_charging_nodes)
    assert task.is_served
    assert task.service_started_at == pytest.approx(task.earliest_service)
    assert env.trucks[0].time_window_waiting_time == pytest.approx(5.0)
    assert env.task_registry.next_task_for_truck(0).sequence_position == 1


def test_energy_path_dispatcher_finishes_sequential_routes(environment):
    from EVRoutingEnv.baselines.heuristics.energy_path import EnergyPathHeuristicPolicy

    env = environment(trucks=2, stops=4)
    env.task_registry.next_task_for_truck(0).base_service_time = 500.0
    policy = EnergyPathHeuristicPolicy()
    for _ in range(20):
        action = policy(env)
        assert env.mask_fn()[action]
        _, _, terminated, truncated, info = env.step(action)
        assert not truncated
        if terminated:
            assert info["successful"]
            return
    raise AssertionError("dispatcher did not finish prescribed routes")


@pytest.mark.parametrize("planner", ["cpsat", "alns"])
def test_nominal_planners_preserve_prescribed_routes(environment, planner):
    env = environment(trucks=2, stops=6)
    if planner == "cpsat":
        from EVRoutingEnv.baselines.optimization.exact_optimization import ExactPlannerParameters, solve_nominal_plan
        solution = solve_nominal_plan(env, ExactPlannerParameters(time_limit_seconds=2, workers=1))
    else:
        from EVRoutingEnv.baselines.heuristics.alns import ALNSParameters, solve_alns_plan
        solution = solve_alns_plan(env, ALNSParameters(iterations=50))
    for truck in env.trucks:
        assert solution.routes[truck.truck_id] == truck.delivery_sequence[1:]


def test_public_heuristic_entrypoint_handles_sequential_fleet(environment):
    from EVRoutingEnv.baselines.heuristics.heuristic_policy import HeuristicPolicy

    env = environment(trucks=2, stops=4)
    policy = HeuristicPolicy()
    for _ in range(12):
        action, explanation = policy.get_action_with_explanations(env)
        assert explanation and env.mask_fn()[action]
        _, _, terminated, truncated, info = env.step(action)
        assert not truncated
        if terminated:
            assert info["successful"]
            assert policy.decision_history
            return
    raise AssertionError("public heuristic did not finish sequential routes")
