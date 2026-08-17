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
from EVRoutingEnv.state.gnn_utils import create_default_gnn_space


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
    Generate feasibility mask for actions using the same logic as GNN state space.
    
    Args:
        env: EventDrivenTruckEnv instance
    
    Returns:
        np.ndarray: Boolean array where True indicates feasible actions.
                   Shape: (action_space.n,)
                   Order: [charger_0, ..., charger_N-1, next_delivery, charge_1h, ..., charge_4h]
    """
    # Initialize all actions as infeasible
    feasible_mask = np.zeros(env.action_space.n, dtype=bool)

    # If no active truck, keep all infeasible
    if env.active_truck_id is None:
        return feasible_mask

    # The primary joint model uses the centralized hard-feasibility engine.
    # Legacy route-execution modes retain their historical graph-derived mask
    # until they are migrated to the canonical feature/action schema.
    if getattr(env, "joint_routing", False):
        decisions = joint_action_feasibility(env)
        env.last_action_feasibility = decisions
        return np.asarray([item.feasible for item in decisions], dtype=bool)

    # Cache a default GNN state space on the env to avoid re-instantiation
    cached_space = getattr(env, "_default_gnn_state_space", None)
    if cached_space is None:
        mode = "vrp" if getattr(env, "enable_flexible_delivery_order", False) else "nonflex"
        use_detour = bool(getattr(env, "use_detour_mask", False))
        cached_space = create_default_gnn_space(env, mode=mode, use_detour=use_detour)
        env._default_gnn_state_space = cached_space

    action_graph = cached_space.get_action_graph(env)
    
    # Handle both ActionGraph dataclass and dict (for compatibility)
    if hasattr(action_graph, 'feasible_action_mask'):
        # ActionGraph dataclass
        mask_np = action_graph.feasible_action_mask
    elif isinstance(action_graph, dict):
        # Dictionary format (legacy)
        mask_np = action_graph.get("feasible_action_mask", feasible_mask)
    else:
        # Fallback
        mask_np = feasible_mask

    # Align length defensively in case action spaces diverge
    if len(mask_np) != env.action_space.n:
        clipped = np.zeros(env.action_space.n, dtype=bool)
        limit = min(len(mask_np), env.action_space.n)
        clipped[:limit] = mask_np[:limit]
        mask_np = clipped

    return mask_np


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
    if not getattr(env, "joint_routing", False):
        raise NotImplementedError(
            "the structural mask is defined for joint-fleet routing only"
        )

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
