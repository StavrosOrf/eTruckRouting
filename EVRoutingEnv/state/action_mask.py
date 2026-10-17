"""
Action feasibility mask generation for the truck routing environment.

This module provides functionality to determine which actions are feasible
for the active truck based on battery constraints, location, and state.
"""

from typing import TYPE_CHECKING

import numpy as np

from EVRoutingEnv.state.feasibility import (
    FeasibilityReason,
    joint_action_feasibility,
)


# Slots that denote no action at all rather than an action that happens to be
# infeasible right now.  An unmasked policy still may not select these: there is
# no customer behind an empty slot to route to, and with no active truck there is
# no decision to make.
STRUCTURALLY_UNDEFINED = frozenset(
    {
        FeasibilityReason.EMPTY_ACTION_SLOT,
        FeasibilityReason.NO_ACTIVE_TRUCK,
    }
)


if TYPE_CHECKING:
    from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv


def get_action_mask(env: "EventDrivenTruckEnv") -> np.ndarray:
    """
    Generate the hard mask from the shared routing feasibility engine.
    
    Args:
        env: EventDrivenTruckEnv instance
    
    Returns:
        np.ndarray: Boolean array where True indicates feasible actions.
                   Shape: (action_space.n,)
                   Order: chargers, global customer slots, depot, configured charge values.
    """
    decisions = joint_action_feasibility(env)
    env.last_action_feasibility = decisions
    return np.asarray([item.feasible for item in decisions], dtype=bool)


def get_structural_action_mask(env: "EventDrivenTruckEnv") -> np.ndarray:
    """Return the candidate set without any feasibility filtering.

    This is the mask for the no-mask ablation.  It differs from
    :func:`get_action_mask` in exactly one respect: an action that the
    feasibility engine rejects for a *dynamic* reason -- not enough energy, the
    task is already claimed, the truck is not at a charger -- stays selectable,
    and the environment rejects it at execution time under
    ``environment.invalid_action_mode``.  Slots that denote no action at all are
    still hidden, so both arms score the same number of real candidates and the
    comparison isolates the mask rather than the action space.
    """
    decisions = joint_action_feasibility(env)
    env.last_action_feasibility = decisions
    return np.asarray(
        [item.reason not in STRUCTURALLY_UNDEFINED for item in decisions],
        dtype=bool,
    )


def policy_action_mask(env: "EventDrivenTruckEnv") -> np.ndarray:
    """Return whichever mask ``env`` hands to a learning policy.

    Environments that predate the mask ablation -- and the lightweight stubs in
    the test suite -- only expose ``mask_fn``, so fall back to it.
    """
    accessor = getattr(env, "policy_mask_fn", None)
    if accessor is None:
        return env.mask_fn()
    return accessor()
