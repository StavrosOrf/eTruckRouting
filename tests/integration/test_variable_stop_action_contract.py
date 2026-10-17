"""Action-index contract for variable-size legacy VRP episodes."""

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/evrp_matplotlib")

from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.utils.utils import load_config
from EVRoutingEnv.state.gnn_utils import create_default_gnn_space


def test_flexible_variable_stops_keep_fixed_action_indices() -> None:
    config = load_config("EVRoutingEnv/config_files/config_joint.yaml")
    config["environment"].update(
        num_trucks=1,
        num_stops=4,
        allow_variable_num_stops=True,
    )
    config["truck"]["battery_capacity"] = 10_000.0
    env = EventDrivenTruckEnv(config, verbose=False, enable_plotting=False)
    try:
        for seed in range(10):
            env.reset(seed=seed)
            if env.num_stops < env.fixed_num_stops:
                break
        else:
            raise AssertionError("no smaller variable-stop episode was sampled")

        mask = env.mask_fn()
        state = create_default_gnn_space(env).get_state_GNN(env)
        action_map = state.action_to_node_map
        assert len(mask) == len(action_map) == env.action_space.n

        for slot in range(env.num_stops, env.fixed_num_stops):
            index = env.num_charging_nodes + slot
            assert action_map[index] == (-1, False)
            assert not mask[index]

        depot_index = env.num_charging_nodes + env.fixed_num_stops
        truck = env.trucks[env.active_truck_id]
        assert action_map[depot_index] == (truck.delivery_sequence[0], False)
        assert all(is_charging for _, is_charging in action_map[env.num_navigation_actions :])
    finally:
        env.close()
