from types import SimpleNamespace

import pytest
import torch
from torch_geometric.data import Batch, HeteroData

from algo.action_heads import ACTION_HEAD_TYPES
from algo.PPO_VariableActionGNN import (
    PPOVariableActionGNN,
    PPOVariableActorCritic,
)


def _graph(offset: float) -> HeteroData:
    graph = HeteroData()
    graph["truck"].x = torch.tensor([[offset, 1.0]], dtype=torch.float32)
    graph["delivery"].x = torch.tensor([[offset, 2.0]], dtype=torch.float32)
    graph["charger"].x = torch.tensor([[offset, 3.0]], dtype=torch.float32)
    return graph


@pytest.mark.parametrize("head_type", ACTION_HEAD_TYPES)
def test_actor_critic_runs_each_action_head_on_a_real_heterogeneous_batch(head_type):
    torch.manual_seed(11)
    policy = PPOVariableActorCritic(
        action_dim=8,
        node_feature_dims={"truck": 2, "delivery": 2, "charger": 2},
        action_feature_dim=3,
        hidden_dim=4,
        num_layers=1,
        mlp_dim=8,
        action_head_type=head_type,
        action_head_layers=2,
        action_attention_heads=2,
    ).eval()
    batch = Batch.from_data_list([_graph(0.0), _graph(10.0)])
    actions = torch.randn(5, 3)
    ptr = torch.tensor([0, 2, 5])

    with torch.no_grad():
        output, values = policy(batch, actions, ptr)

    assert output.logits.shape == (5,)
    assert values.shape == (2,)
    assert torch.isfinite(output.logits).all()
    assert torch.isfinite(values).all()


def test_policy_refuses_an_empty_hard_mask():
    agent = PPOVariableActionGNN.__new__(PPOVariableActionGNN)
    state = SimpleNamespace(feasible_action_mask=torch.tensor([False, False]))

    with pytest.raises(RuntimeError, match="refusing to relax"):
        agent._prepare_feasible_actions(state)


def test_policy_refuses_disjoint_hard_masks():
    agent = PPOVariableActionGNN.__new__(PPOVariableActionGNN)
    state = SimpleNamespace(feasible_action_mask=torch.tensor([True, False]))

    with pytest.raises(RuntimeError, match="refusing to relax"):
        agent._prepare_feasible_actions(state, torch.tensor([False, True]))


def test_policy_refuses_mask_length_mismatch():
    agent = PPOVariableActionGNN.__new__(PPOVariableActionGNN)
    state = SimpleNamespace(feasible_action_mask=torch.tensor([True, False]))

    with pytest.raises(ValueError, match="length mismatch"):
        agent._prepare_feasible_actions(state, torch.tensor([True]))


def test_policy_maps_global_actions_to_the_correct_feature_rows():
    agent = PPOVariableActionGNN.__new__(PPOVariableActionGNN)
    agent.action_feature_dim = 3
    state = SimpleNamespace(
        feasible_action_mask=torch.tensor([True, False, True, False]),
        action_graph_features=torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    )

    features, ptr = agent._build_action_graph_inputs(
        [state], [torch.tensor([2, 0])], device=torch.device("cpu")
    )

    torch.testing.assert_close(
        features, torch.tensor([[4.0, 5.0, 6.0], [1.0, 2.0, 3.0]])
    )
    assert torch.equal(ptr, torch.tensor([0, 2]))


def test_policy_refuses_action_feature_width_mismatch():
    agent = PPOVariableActionGNN.__new__(PPOVariableActionGNN)
    agent.action_feature_dim = 3
    state = SimpleNamespace(
        feasible_action_mask=torch.tensor([True]),
        action_graph_features=torch.ones((1, 2)),
    )

    with pytest.raises(ValueError, match="width mismatch"):
        agent._build_action_graph_inputs(
            [state], [torch.tensor([0])], device=torch.device("cpu")
        )


def test_policy_refuses_a_stored_action_outside_the_feasible_set():
    with pytest.raises(ValueError, match="not in feasible set"):
        PPOVariableActionGNN._locate_action_index(torch.tensor([1, 3]), torch.tensor(2))
