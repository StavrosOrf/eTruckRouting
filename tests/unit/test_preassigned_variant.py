"""The eTFRP-style variant: customers belong to one truck from the start.

This is the setting the paper's main benchmark models. Expressing it inside the
canonical stack is what lets the mask ablation, the seed study, and the artifact
contract apply to it as well as to the joint formulation.
"""

import os
from copy import deepcopy

import pytest


os.environ.setdefault("MPLCONFIGDIR", "/tmp/evrp_matplotlib")

from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.state.feasibility import FeasibilityReason, joint_action_feasibility
from EVRoutingEnv.utils.utils import load_config


def _config(assignment: str, **overrides) -> dict:
    config = deepcopy(load_config("EVRoutingEnv/config_files/config_joint.yaml"))
    config["problem"]["assignment"] = assignment
    config["environment"].update(overrides)
    return config


def _env(assignment: str, **overrides) -> EventDrivenTruckEnv:
    return EventDrivenTruckEnv(
        _config(assignment, **overrides), verbose=False, enable_plotting=False
    )


def test_unknown_assignment_is_rejected() -> None:
    with pytest.raises(ValueError, match="problem.assignment"):
        _env("sometimes").close()


def test_fleet_assignment_leaves_customers_unowned() -> None:
    env = _env("fleet")
    try:
        env.reset(seed=11)
        assert all(
            task.preassigned_to is None for task in env.task_registry.tasks()
        )
    finally:
        env.close()


def test_every_customer_gets_exactly_one_owner() -> None:
    env = _env("preassigned")
    try:
        for seed in (1, 2, 3):
            env.reset(seed=seed)
            owners = [task.preassigned_to for task in env.task_registry.tasks()]
            assert all(owner is not None for owner in owners)
            assert set(owners) <= {truck.truck_id for truck in env.trucks}
    finally:
        env.close()


def test_each_truck_can_carry_its_own_assignment() -> None:
    """Assignment must not hand a truck more than its payload capacity."""
    env = _env("preassigned")
    try:
        for seed in (4, 5, 6, 7):
            env.reset(seed=seed)
            loads: dict[int, float] = {}
            for task in env.task_registry.tasks():
                loads[task.preassigned_to] = (
                    loads.get(task.preassigned_to, 0.0) + float(task.demand)
                )
            for truck in env.trucks:
                assert loads.get(truck.truck_id, 0.0) <= float(
                    truck.payload_capacity
                ) + 1e-9
    finally:
        env.close()


def test_another_trucks_customer_is_rejected_with_its_own_reason() -> None:
    env = _env("preassigned")
    try:
        env.reset(seed=21)
        active = env.active_truck_id
        foreign = [
            task
            for task in env.task_registry.tasks()
            if task.preassigned_to != active
        ]
        assert foreign, "scenario gave every customer to the active truck"

        decisions = joint_action_feasibility(env)
        reasons = {
            decision.target_node: decision.reason
            for decision in decisions
            if decision.target_node is not None
        }
        for task in foreign:
            reason = reasons.get(int(task.node_id))
            if reason is not None:
                assert reason is FeasibilityReason.PREASSIGNED_TO_OTHER
    finally:
        env.close()


def test_preassignment_does_not_change_the_observation_shape() -> None:
    """The same policy architecture must apply to both settings."""
    fleet, preassigned = _env("fleet"), _env("preassigned")
    try:
        a, _ = fleet.reset(seed=31)
        b, _ = preassigned.reset(seed=31)
        assert a.shape == b.shape
        assert fleet.action_space.n == preassigned.action_space.n
    finally:
        fleet.close()
        preassigned.close()


def test_preassignment_narrows_the_feasible_set() -> None:
    fleet, preassigned = _env("fleet"), _env("preassigned")
    try:
        fleet.reset(seed=41)
        preassigned.reset(seed=41)
        assert int(preassigned.mask_fn().sum()) <= int(fleet.mask_fn().sum())
    finally:
        fleet.close()
        preassigned.close()
