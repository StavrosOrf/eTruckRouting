"""Sanity checks for PPOVariableActionGNN action mapping and batching."""

import unittest
from typing import Dict, Iterable

import torch
from torch_geometric.data import HeteroData

from algo.PPO_VariableActionGNN import (
    ActionGraphHead,
    PPOVariableActionGNN,
)


NODE_DIMS = {
    "truck": 4,
    "delivery": 3,
    "charger": 2,
}


def _make_state(
    num_actions: int,
    valid_indices: Iterable[int],
    node_dims: Dict[str, int] = NODE_DIMS,
) -> HeteroData:
    data = HeteroData()
    for node_type, dim in node_dims.items():
        x = torch.randn(1, dim)
        data[node_type].x = x
        data[node_type].batch = torch.zeros(x.size(0), dtype=torch.long)
    mask = torch.zeros(num_actions, dtype=torch.bool)
    for idx in valid_indices:
        mask[idx] = True
    data.feasible_action_mask = mask
    data.num_actions = torch.tensor(num_actions)
    return data


class VariableActionGNNTests(unittest.TestCase):
    def setUp(self):
        self.policy = PPOVariableActionGNN(
            action_dim=8,
            node_feature_dims=NODE_DIMS,
            hidden_dim=16,
            num_layers=1,
            mlp_dim=32,
            device="cpu",
            minibatch_size=2,
        )

    def test_action_selection_respects_mask(self):
        state = _make_state(num_actions=5, valid_indices=[2, 4])
        action = self.policy.select_action(state, deterministic=True)
        self.assertEqual(action, 2)

    def test_action_graph_head_ptr_builder(self):
        head = ActionGraphHead(encoder_dim=12, mlp_dim=8)
        embedding = torch.randn(2, 12)
        counts = [2, 3]
        output = head(embedding, counts)
        self.assertEqual(output.ptr.tolist(), [0, 2, 5])
        self.assertEqual(output.logits.numel(), sum(counts))

    def test_update_handles_variable_batches(self):
        masks = [
            [True, False, True],
            [False, True, True],
            [True, True, False],
        ]
        for idx, mask_list in enumerate(masks):
            state = _make_state(num_actions=3, valid_indices=[i for i, v in enumerate(mask_list) if v])
            action = state.feasible_action_mask.nonzero(as_tuple=False)[0].item()
            done = idx == len(masks) - 1
            self.policy.store_transition(
                state,
                action=action,
                logprob=0.0,
                reward=1.0,
                done=done,
                value=0.0,
                action_mask=state.feasible_action_mask,
            )
        stats = self.policy.update(last_value=0.0)
        self.assertIn("policy_loss", stats)
        self.assertEqual(len(self.policy.buffer.states), 0)


if __name__ == "__main__":
    unittest.main()
