"""Contract tests for the non-learning baselines.

Every baseline must obey the same hard feasibility mask as the learned policies,
otherwise a comparison measures rule-breaking rather than decision quality.
"""

import os
from copy import deepcopy

import pytest


os.environ.setdefault("MPLCONFIGDIR", "/tmp/evrp_matplotlib")

from EVRoutingEnv.baselines.canonical_baselines import (
    GreedyHeuristicPolicy,
    HeuristicParameters,
    MPCParameters,
    RandomFeasiblePolicy,
    RollingHorizonMPCPolicy,
    decode_feasible_actions,
)
from EVRoutingEnv.baselines.exact_optimization import (
    ExactPlannerParameters,
    MathematicalProgrammingPolicy,
    solve_nominal_plan,
)
from EVRoutingEnv.models.core.customer import TaskStatus
from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.utils.utils import load_config


def _config() -> dict:
    config = deepcopy(load_config("EVRoutingEnv/config_files/config_joint.yaml"))
    config["environment"]["num_stops"] = 6
    return config


@pytest.fixture(scope="module")
def env():
    instance = EventDrivenTruckEnv(_config(), verbose=False, enable_plotting=False)
    try:
        yield instance
    finally:
        instance.close()


def _policies():
    return [
        RandomFeasiblePolicy(seed=1),
        GreedyHeuristicPolicy(HeuristicParameters(energy_safety_factor=1.15)),
        RollingHorizonMPCPolicy(MPCParameters(horizon=3, branching=2)),
        MathematicalProgrammingPolicy(
            ExactPlannerParameters(time_limit_seconds=5.0, workers=1)
        ),
    ]


@pytest.mark.parametrize("policy", _policies(), ids=lambda item: item.name)
def test_baselines_only_ever_emit_hard_feasible_actions(env, policy) -> None:
    for seed in (11, 12):
        observation, info = env.reset(seed=seed)
        terminated = truncated = False
        steps = 0
        while not (terminated or truncated) and steps < 120:
            feasible = {action.index for action in decode_feasible_actions(env)}
            action = policy(env, observation, info)
            assert action in feasible, (
                f"{policy.name} chose action {action} outside the feasible set"
            )
            assert bool(env.mask_fn()[action])
            observation, _, terminated, truncated, info = env.step(action)
            steps += 1


def test_decode_refuses_an_empty_feasible_set(env, monkeypatch) -> None:
    env.reset(seed=21)
    features = env.get_canonical_features()
    features.action_features[:, 5] = 0.0
    monkeypatch.setattr(env, "get_canonical_features", lambda: features)

    with pytest.raises(RuntimeError, match="no feasible action"):
        decode_feasible_actions(env)


def test_heuristic_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="target_soc"):
        GreedyHeuristicPolicy(HeuristicParameters(target_soc=0.0))
    with pytest.raises(ValueError, match="energy_safety_factor"):
        GreedyHeuristicPolicy(HeuristicParameters(energy_safety_factor=0.5))


def test_nominal_plan_covers_every_pending_customer_exactly_once(env) -> None:
    env.reset(seed=31)
    plan = solve_nominal_plan(env, ExactPlannerParameters(time_limit_seconds=15.0))

    assert plan.status in {"OPTIMAL", "FEASIBLE"}
    assigned = [node for route in plan.routes.values() for node in route]
    pending = [
        int(task.node_id)
        for task in env.task_registry.tasks()
        if task.status in (TaskStatus.UNASSIGNED, TaskStatus.CLAIMED)
    ]
    assert sorted(assigned) == sorted(pending)
    assert len(assigned) == len(set(assigned))


def test_nominal_plan_respects_payload_capacity(env) -> None:
    env.reset(seed=32)
    plan = solve_nominal_plan(env, ExactPlannerParameters(time_limit_seconds=15.0))

    for truck_id, route in plan.routes.items():
        truck = env.trucks[truck_id]
        load = sum(env.task_registry.task_for_node(node).demand for node in route)
        assert load <= float(truck.payload_capacity) + 1e-6


def test_nominal_plan_reports_its_own_optimality_evidence(env) -> None:
    env.reset(seed=33)
    plan = solve_nominal_plan(env, ExactPlannerParameters(time_limit_seconds=20.0))

    assert plan.objective_hours is not None
    assert plan.best_bound_hours is not None
    assert plan.best_bound_hours <= plan.objective_hours + 1e-6
    assert plan.proven_optimal == (plan.status == "OPTIMAL")


def test_mpc_lookahead_changes_the_committed_goal(env) -> None:
    """A horizon-1 and a deep-horizon controller must not be trivially identical."""
    shallow = RollingHorizonMPCPolicy(MPCParameters(horizon=1, branching=1))
    deep = RollingHorizonMPCPolicy(MPCParameters(horizon=6, branching=4))

    differences = 0
    for seed in (41, 42, 43):
        observation, info = env.reset(seed=seed)
        for _ in range(25):
            if shallow(env, observation, info) != deep(env, observation, info):
                differences += 1
            observation, _, terminated, truncated, info = env.step(
                deep(env, observation, info)
            )
            if terminated or truncated:
                break
    assert differences > 0, "lookahead depth never changed a decision"
