"""Properties the ALNS baseline has to hold to be a fair comparison."""

import os

import pytest


os.environ.setdefault("MPLCONFIGDIR", "/tmp/evrp_matplotlib")

from EVRoutingEnv.baselines.alns import (
    ALNSParameters,
    ALNSPolicy,
    solve_alns_plan,
)
from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv


def _env() -> EventDrivenTruckEnv:
    return EventDrivenTruckEnv(
        "EVRoutingEnv/config_files/config_joint.yaml",
        verbose=False,
        enable_plotting=False,
    )


def test_parameters_reject_an_unknown_objective() -> None:
    with pytest.raises(ValueError, match="objective"):
        ALNSParameters(objective="fuel")


def test_parameters_reject_inverted_destroy_fractions() -> None:
    with pytest.raises(ValueError, match="destroy fractions"):
        ALNSParameters(min_destroy_fraction=0.5, max_destroy_fraction=0.2)


def test_plan_serves_every_customer_exactly_once() -> None:
    env = _env()
    try:
        env.reset(seed=31337)
        plan = solve_alns_plan(env, ALNSParameters(iterations=500))
        planned = [node for route in plan.routes.values() for node in route]
        expected = {int(task.node_id) for task in env.task_registry.tasks()}

        assert len(planned) == len(set(planned)), "a customer appears twice"
        assert set(planned) == expected
    finally:
        env.close()


def test_search_never_reports_a_bound() -> None:
    """A metaheuristic proves nothing: the artifact must not imply otherwise."""
    env = _env()
    try:
        env.reset(seed=99)
        plan = solve_alns_plan(env, ALNSParameters(iterations=200))
        assert plan.best_bound_hours is None
        assert not plan.proven_optimal
        assert plan.status.startswith("HEURISTIC_")
    finally:
        env.close()


def test_more_iterations_never_return_a_worse_plan() -> None:
    """The incumbent is monotone in the budget for a fixed scenario and seed."""
    env = _env()
    try:
        env.reset(seed=4242)
        short = solve_alns_plan(env, ALNSParameters(iterations=200, seed=0))
        long = solve_alns_plan(env, ALNSParameters(iterations=5000, seed=0))
        assert long.objective_hours <= short.objective_hours + 1e-6
    finally:
        env.close()


def test_plan_is_reproducible_for_a_scenario() -> None:
    env = _env()
    try:
        env.reset(seed=777)
        first = solve_alns_plan(env, ALNSParameters(iterations=1000, seed=3))
        second = solve_alns_plan(env, ALNSParameters(iterations=1000, seed=3))
        assert first.routes == second.routes
        assert first.objective_hours == pytest.approx(second.objective_hours)
    finally:
        env.close()


def test_policy_respects_payload_capacity_in_its_plan() -> None:
    env = _env()
    try:
        env.reset(seed=1234)
        plan = solve_alns_plan(env, ALNSParameters(iterations=2000))
        for truck_id, route in plan.routes.items():
            truck = env.trucks[truck_id]
            load = sum(
                float(env.task_registry.task_for_node(int(node)).demand)
                for node in route
            )
            assert load <= float(truck.payload_capacity) + 1e-6
    finally:
        env.close()


@pytest.mark.integration
def test_policy_drives_a_full_episode_through_the_shared_execution_layer() -> None:
    env = _env()
    try:
        policy = ALNSPolicy(ALNSParameters(iterations=1000))
        observation, info = env.reset(seed=2468)
        terminated = truncated = False
        steps = 0
        while not (terminated or truncated) and steps < 400:
            action = policy(env, observation, info)
            assert bool(env.mask_fn()[action]), "ALNS proposed an infeasible action"
            observation, _, terminated, truncated, info = env.step(action)
            steps += 1
        assert steps > 0
    finally:
        env.close()
