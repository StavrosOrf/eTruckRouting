"""
Unified non-flexible GNN state space.

This module exposes a single entry point for non-flexible delivery runs.
It wraps the existing full navigation state space as well as the detour-
based variant behind a shared interface controlled by ``use_detour``.
"""
from __future__ import annotations

from typing import Optional

from EVRoutingEnv.state.gnn_state_space import GNNStateSpace
from EVRoutingEnv.state.gnn_state_space_detour import GNNStateSpaceDetourBased
from EVRoutingEnv.state.gnn_utils import extract_action_graph


class GNNStateSpaceNonFlex(GNNStateSpace):
    """Non-flexible GNN state space with optional detour masking."""

    def __init__(
        self,
        num_trucks: int,
        num_stops: int,
        max_time: float,
        num_charging_nodes: int,
        max_nodes_in_graph: int = 500,
        device: str = "cpu",
        verbose: bool = False,
        use_detour: bool = False,
        route_delivery_after_charge_only: bool = True,
        detour_num_chargers_to_keep: int = 2,
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
        self.use_detour = use_detour
        self.route_delivery_after_charge_only = route_delivery_after_charge_only
        self.detour_num_chargers_to_keep = detour_num_chargers_to_keep
        self._detour_space: Optional[GNNStateSpaceDetourBased] = None

    def _ensure_detour_space(self):
        if self._detour_space is None:
            self._detour_space = GNNStateSpaceDetourBased(
                num_trucks=self.num_trucks,
                num_stops=self.num_stops,
                max_time=self.max_time,
                num_charging_nodes=self.num_charging_nodes,
                max_nodes_in_graph=self.max_nodes_in_graph,
                device=self.device,
                verbose=self.verbose,
                route_delivery_after_charge_only=self.route_delivery_after_charge_only,
                num_chargers_to_keep=self.detour_num_chargers_to_keep,
            )

    def get_state_GNN(self, env):
        if not self.use_detour:
            return super().get_state_GNN(env)
        self._ensure_detour_space()
        return self._detour_space.get_state_GNN(env)

    def get_action_graph(self, env):
        if not self.use_detour:
            return super().get_action_graph(env)
        self._ensure_detour_space()
        return self._detour_space.get_action_graph(env)


def get_action_graph(env, state_space: Optional[GNNStateSpaceNonFlex] = None):
    """Module-level helper mirroring the class method for convenience."""
    space = state_space
    if space is None:
        space = GNNStateSpaceNonFlex(
            num_trucks=env.num_trucks,
            num_stops=env.num_stops,
            max_time=env.max_time,
            num_charging_nodes=len(env.charging_nodes),
            device=getattr(env, "device", "cpu"),
        )
    return extract_action_graph(space, env)
