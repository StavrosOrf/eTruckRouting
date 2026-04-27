"""
Linear (flattened) state space helper that mirrors the GNN feature layout.

This wraps the existing ``StateSpace`` and adds a convenience method that
pulls values directly from an environment instance, making it usable for
both non-flexible and VRP-style runs.
"""
from __future__ import annotations

from EVRoutingEnv.state.state_space import StateSpace


class LinearStateSpace(StateSpace):
    """Flattened observation helper with env-aware entry point."""

    def get_state_from_env(self, env):
        return self.get_state(
            trucks=env.trucks,
            active_truck_id=env.active_truck_id,
            transport_graph=env.transport_graph,
            charging_nodes=env.charging_nodes,
            truck_states=env.truck_states,
            event_queue=env.event_queue,
            global_clock=env.global_clock,
            charging_station=getattr(env, "charging_station", None),
        )


def get_state(env, state_space: LinearStateSpace) -> object:
    """Module-level convenience wrapper."""
    return state_space.get_state_from_env(env)
