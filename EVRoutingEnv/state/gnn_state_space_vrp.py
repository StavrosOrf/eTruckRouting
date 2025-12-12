"""
GNN State Representation for flexible-delivery (VRP-style) control.

Use this when ``enable_flexible_delivery_order`` is enabled; otherwise prefer
``GNNStateSpaceNonFlex`` (with or without detour masking) for sequential runs.
"""
from __future__ import annotations

from typing import Optional

from EVRoutingEnv.state.gnn_state_space import GNNStateSpace
from EVRoutingEnv.state.gnn_utils import extract_action_graph


class GNNStateSpaceVRP(GNNStateSpace):
    """GNN state space specialized for flexible delivery ordering (VRP)."""

    def __init__(
        self,
        num_trucks: int,
        num_stops: int,
        max_time: float,
        num_charging_nodes: int,
        max_nodes_in_graph: int = 500,
        device: str = "cpu",
        verbose: bool = False,
    ):
        super().__init__(
            num_trucks=num_trucks,
            num_stops=num_stops,
            max_time=max_time,
            num_charging_nodes=num_charging_nodes,
            max_nodes_in_graph=max_nodes_in_graph,
            device=device,
            verbose=verbose,
        )

    def get_state_GNN(self, env):  # type: ignore[override]
        data = super().get_state_GNN(env)

        active_truck = None
        if getattr(env, "active_truck_id", None) is not None and env.active_truck_id < len(env.trucks):
            active_truck = env.trucks[env.active_truck_id]

        is_flexible_env = bool(getattr(env, "enable_flexible_delivery_order", False))
        is_flexible_truck = bool(active_truck and getattr(active_truck, "enable_flexible_delivery_order", False))
        if not (is_flexible_env or is_flexible_truck):
            raise ValueError(
                "GNNStateSpaceVRP requires flexible delivery ordering. "
                "Use GNNStateSpaceNonFlex for sequential runs."
            )

        expected_actions: Optional[int] = getattr(getattr(env, "action_space", None), "n", None)
        action_count = len(getattr(data, "action_to_node_map", []))
        if expected_actions is not None and action_count != expected_actions:
            raise ValueError(
                f"VRP action map size {action_count} does not match environment action space {expected_actions}."
            )

        # Attach small bits of metadata for downstream debuggers/agents.
        data.delivery_mode = "flexible"
        data.action_space_n = expected_actions
        return data

    def get_action_graph(self, env):
        return extract_action_graph(self, env)


def get_action_graph(env, state_space: Optional[GNNStateSpaceVRP] = None):
    """Module-level helper for convenience when only an env is available."""
    space = state_space
    if space is None:
        space = GNNStateSpaceVRP(
            num_trucks=env.num_trucks,
            num_stops=env.num_stops,
            max_time=env.max_time,
            num_charging_nodes=len(env.charging_nodes),
            device=getattr(env, "device", "cpu"),
        )
    return extract_action_graph(space, env)
