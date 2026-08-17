"""Semantics of the no-mask ablation.

The ablation is only meaningful if removing the feasibility mask changes what
the policy may select and nothing else: the same observation, the same candidate
slots, the same simulator dynamics.  These checks pin that down, and pin down
what executing an infeasible action then does under each rejection mode.
"""

import os
from copy import deepcopy

import numpy as np
import pytest


os.environ.setdefault("MPLCONFIGDIR", "/tmp/evrp_matplotlib")

from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.state.action_mask import policy_action_mask
from EVRoutingEnv.state.feasibility import (
    FeasibilityReason,
    joint_action_feasibility,
)
from EVRoutingEnv.utils.utils import load_config


def _config(**environment_overrides) -> dict:
    config = deepcopy(load_config("EVRoutingEnv/config_files/config_joint.yaml"))
    config["environment"].update(environment_overrides)
    return config


def _first_infeasible_defined_action(env) -> int | None:
    """An action the feasibility engine rejects for a dynamic reason."""
    hard = env.mask_fn()
    structural = env.structural_mask_fn()
    candidates = np.flatnonzero(structural & ~hard)
    return int(candidates[0]) if candidates.size else None


@pytest.mark.integration
def test_structural_mask_only_hides_slots_that_denote_no_action() -> None:
    env = EventDrivenTruckEnv(_config(), verbose=False, enable_plotting=False)
    try:
        env.reset(seed=4242)
        decisions = joint_action_feasibility(env)
        structural = env.structural_mask_fn()

        expected = np.asarray(
            [
                item.reason is not FeasibilityReason.EMPTY_ACTION_SLOT
                for item in decisions
            ],
            dtype=bool,
        )
        assert np.array_equal(structural, expected)

        # Every feasible action stays selectable, and the structural mask is a
        # strict superset whenever anything is infeasible at all.
        hard = env.mask_fn()
        assert np.all(structural[hard])
        assert structural.sum() >= hard.sum()
    finally:
        env.close()


@pytest.mark.integration
def test_mask_mode_selects_which_mask_a_policy_receives() -> None:
    hard_env = EventDrivenTruckEnv(_config(), verbose=False, enable_plotting=False)
    soft_env = EventDrivenTruckEnv(
        _config(policy_action_mask="structural"), verbose=False, enable_plotting=False
    )
    try:
        hard_observation, _ = hard_env.reset(seed=99)
        soft_observation, _ = soft_env.reset(seed=99)

        # Same scenario, same observation: only the mask differs.
        assert np.array_equal(hard_observation, soft_observation)
        assert np.array_equal(policy_action_mask(hard_env), hard_env.mask_fn())
        assert np.array_equal(
            policy_action_mask(soft_env), soft_env.structural_mask_fn()
        )
    finally:
        hard_env.close()
        soft_env.close()


@pytest.mark.integration
def test_terminate_mode_strands_the_truck_on_an_infeasible_action() -> None:
    env = EventDrivenTruckEnv(
        _config(policy_action_mask="structural"), verbose=False, enable_plotting=False
    )
    try:
        env.reset(seed=7)
        action = _first_infeasible_defined_action(env)
        assert action is not None, "scenario has no infeasible-but-defined action"

        _, reward, _, _, info = env.step(action)

        assert reward <= env.reward_config["failure_penalty"]
        assert env.invalid_action_count == 1
        assert info["termination_reason"].startswith("invalid_action:")
    finally:
        env.close()


@pytest.mark.integration
def test_penalize_mode_refuses_the_action_and_keeps_the_episode_alive() -> None:
    env = EventDrivenTruckEnv(
        _config(
            policy_action_mask="structural",
            invalid_action_mode="penalize",
            invalid_action_budget=3,
        ),
        verbose=False,
        enable_plotting=False,
    )
    try:
        env.reset(seed=7)
        action = _first_infeasible_defined_action(env)
        assert action is not None

        clock_before = float(env.global_clock)
        _, reward, terminated, truncated, info = env.step(action)

        # Refused, not executed: no failure, no progress, and nothing in the
        # simulator moved.
        assert not terminated and not truncated
        assert info["termination_reason"] is None
        assert env.invalid_action_count == 1
        assert float(env.global_clock) == pytest.approx(clock_before)
        assert reward == pytest.approx(
            -abs(env.reward_config["invalid_action_penalty"])
        )

        # The refusal must not strand the truck: it stays alive and is asked
        # again, which is the whole point of the charitable variant.
        assert env.truck_states[env.trucks[0].truck_id] != "failed"

        # Nothing can spin forever here.  A refusal schedules no event, so once
        # every truck refuses in turn the queue empties and the simulator ends
        # the episode on its own deadlock check rather than on the budget.
        for _ in range(20):
            repeat = _first_infeasible_defined_action(env)
            if repeat is None:
                break
            _, _, terminated, truncated, info = env.step(repeat)
            if terminated or truncated:
                break

        assert terminated or truncated
        assert info["termination_reason"] is not None
        assert not info["termination_reason"].startswith("invalid_action:")
    finally:
        env.close()


@pytest.mark.integration
def test_penalize_mode_lets_a_truck_recover_from_a_refused_action() -> None:
    """A refused action costs reward but does not end the truck's episode."""
    env = EventDrivenTruckEnv(
        _config(policy_action_mask="structural", invalid_action_mode="penalize"),
        verbose=False,
        enable_plotting=False,
    )
    try:
        env.reset(seed=11)
        refused = _first_infeasible_defined_action(env)
        assert refused is not None

        _, _, terminated, truncated, _ = env.step(refused)
        assert not (terminated or truncated)

        feasible = np.flatnonzero(env.mask_fn())
        assert feasible.size > 0
        _, reward, terminated, truncated, info = env.step(int(feasible[0]))

        # The follow-up action executes normally: no lingering failure state.
        assert reward > -abs(env.reward_config["failure_penalty"])
        assert all(state != "failed" for state in env.truck_states.values())
        assert env.invalid_action_count == 1
    finally:
        env.close()


@pytest.mark.integration
def test_removing_the_mask_does_not_change_the_dynamics() -> None:
    """A feasible action produces identical transitions under either mask."""
    hard_env = EventDrivenTruckEnv(_config(), verbose=False, enable_plotting=False)
    soft_env = EventDrivenTruckEnv(
        _config(policy_action_mask="structural", invalid_action_mode="penalize"),
        verbose=False,
        enable_plotting=False,
    )
    try:
        hard_observation, _ = hard_env.reset(seed=2024)
        soft_observation, _ = soft_env.reset(seed=2024)

        for _ in range(25):
            feasible = np.flatnonzero(hard_env.mask_fn())
            if feasible.size == 0:
                break
            action = int(feasible[0])
            hard_observation, hard_reward, hard_done, hard_trunc, _ = hard_env.step(
                action
            )
            soft_observation, soft_reward, soft_done, soft_trunc, _ = soft_env.step(
                action
            )

            assert np.array_equal(hard_observation, soft_observation)
            assert hard_reward == pytest.approx(soft_reward)
            assert hard_done == soft_done and hard_trunc == soft_trunc
            if hard_done or hard_trunc:
                break
    finally:
        hard_env.close()
        soft_env.close()
