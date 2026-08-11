import copy

import pytest

from algo.action_heads import ACTION_HEAD_TYPES
from algo.PPO_VariableActionGNN import PPOVariableActionGNN
from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.state.gnn_utils import create_default_gnn_space
from EVRoutingEnv.utils.utils import load_config


@pytest.fixture(scope="module")
def seeded_gnn_state():
    config = copy.deepcopy(load_config("EVRoutingEnv/config_files/config_vrp.yaml"))
    config["environment"]["num_trucks"] = 1
    config["environment"]["num_stops"] = 3
    config["environment"]["max_episode_steps"] = 50
    config["delivery"]["enable_flexible_delivery_order"] = True
    environment = EventDrivenTruckEnv(
        config=config,
        verbose=False,
        enable_plotting=False,
        run_id="action_head_integration_test",
    )
    try:
        environment.reset(seed=19)
        state_space = create_default_gnn_space(
            environment,
            mode="vrp",
            vrp_top_k_deliveries=5,
            device="cpu",
        )
        state = state_space.get_state_GNN(environment)
        node_feature_dims = {
            node_type: int(state[node_type].x.shape[-1])
            for node_type in state.node_types
        }
        yield environment, state, node_feature_dims
    finally:
        environment.close()


@pytest.mark.integration
@pytest.mark.parametrize("head_type", ACTION_HEAD_TYPES)
def test_every_action_head_selects_a_hard_feasible_environment_action(
    head_type, seeded_gnn_state
):
    environment, state, node_feature_dims = seeded_gnn_state
    agent = PPOVariableActionGNN(
        action_dim=environment.action_space.n,
        node_feature_dims=node_feature_dims,
        hidden_dim=8,
        num_layers=1,
        mlp_dim=16,
        action_head_type=head_type,
        action_head_layers=2,
        action_attention_heads=4,
        device="cpu",
    )

    action, log_probability, value = agent.act(state, deterministic=True)

    assert bool(state.feasible_action_mask[action])
    assert isinstance(log_probability, float)
    assert isinstance(value, float)
