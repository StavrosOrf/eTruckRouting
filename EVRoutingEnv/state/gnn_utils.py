"""
Shared helpers for GNN state-space builders.

This module centralizes small utilities used by GNN state-space variants
(non-flex, detour, and VRP) and by auxiliary helpers such as the action
mask. The helpers intentionally stay lightweight to avoid pulling in heavy
framework dependencies where not required.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

try:  # Torch is optional for callers that only need numpy masks
    import torch
except ImportError:  # pragma: no cover - torch not required for pure numpy use
    torch = None


@dataclass
class ActionGraph:
    """Lightweight container for action graph metadata."""

    data: Any
    feasible_action_mask: np.ndarray
    action_to_node_map: Any


def feasible_mask_to_numpy(mask: Any) -> np.ndarray:
    """Convert a torch or numpy mask to a numpy boolean array."""
    if mask is None:
        return np.zeros(0, dtype=bool)
    if torch is not None and isinstance(mask, torch.Tensor):
        return mask.detach().cpu().numpy().astype(bool)
    return np.asarray(mask, dtype=bool)


def extract_action_graph(state_space, env) -> ActionGraph:
    """Build the action graph via the provided state-space object.

    This calls ``state_space.get_state_GNN`` once and returns a small
    dataclass that includes the PyG data object plus a numpy mask for SB3
    maskable policies.
    """
    data = state_space.get_state_GNN(env)
    mask_np = feasible_mask_to_numpy(getattr(data, "feasible_action_mask", None))
    return ActionGraph(
        data=data,
        feasible_action_mask=mask_np,
        action_to_node_map=getattr(data, "action_to_node_map", []),
    )


def create_default_gnn_space(
    env,
    *,
    mode: str = "nonflex",
    use_detour: bool = False,
    device: Optional[str] = None,
    vrp_top_k_deliveries: int = 5,
):
    """Instantiate a GNN state-space object from an environment.

    Args:
        env: Environment instance with expected attributes (num_trucks, num_stops, max_time, charging_nodes).
        mode: "nonflex" or "vrp".
        use_detour: Toggle detour-based charger restriction for non-flex runs.
        device: Optional torch device string.
    """
    device = device or getattr(env, "device", "cpu")

    if mode == "vrp":
        from EVRoutingEnv.state.gnn_state_space_vrp import GNNStateSpaceVRP

        return GNNStateSpaceVRP(
            num_trucks=env.num_trucks,
            num_stops=env.num_stops,
            max_time=env.max_time,
            num_charging_nodes=len(env.charging_nodes),
            device=device,
            vrp_top_k_deliveries=vrp_top_k_deliveries,
        )

    from EVRoutingEnv.state.gnn_state_space_nonflex import GNNStateSpaceNonFlex

    return GNNStateSpaceNonFlex(
        num_trucks=env.num_trucks,
        num_stops=env.num_stops,
        max_time=env.max_time,
        num_charging_nodes=len(env.charging_nodes),
        device=device,
        use_detour=use_detour,
    )
