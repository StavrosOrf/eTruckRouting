"""
GNN State Representation for flexible-delivery (VRP-style) control.

This thin wrapper enforces flexible delivery ordering and validates that the
GNN action mapping aligns with the environment's flexible action space.
Use this when `enable_flexible_delivery_order` is enabled; otherwise prefer
`GNNStateSpaceDetourBased` for sequential runs.
"""
from typing import Optional

from EVRoutingEnv.state.gnn_state_space import GNNStateSpace


class GNNStateSpaceVRP(GNNStateSpace):
    """GNN state space specialized for flexible delivery ordering (VRP).

    Behavior matches ``GNNStateSpace`` but adds:
    - Validation that flexible ordering is active in the environment/truck.
    - A sanity check that the generated action map matches the environment's
      discrete action space size, to catch indexing mismatches early.
    """

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

    def get_state_GNN(self, env):
        """Build the flexible-order graph state and validate action mapping."""
        data = super().get_state_GNN(env)

        active_truck = None
        if getattr(env, "active_truck_id", None) is not None and env.active_truck_id < len(env.trucks):
            active_truck = env.trucks[env.active_truck_id]

        is_flexible_env = bool(getattr(env, "enable_flexible_delivery_order", False))
        is_flexible_truck = bool(active_truck and getattr(active_truck, "enable_flexible_delivery_order", False))
        if not (is_flexible_env or is_flexible_truck):
            raise ValueError(
                "GNNStateSpaceVRP requires flexible delivery ordering. "
                "Use GNNStateSpaceDetourBased for sequential runs."
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
