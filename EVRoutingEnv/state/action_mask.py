"""
Action feasibility mask generation for the truck routing environment.

This module provides functionality to determine which actions are feasible
for the active truck based on battery constraints, location, and state.
"""

from typing import TYPE_CHECKING

import numpy as np

from EVRoutingEnv.state.feasibility import joint_action_feasibility
from EVRoutingEnv.state.gnn_utils import create_default_gnn_space


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
