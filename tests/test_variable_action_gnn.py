"""Sanity checks for PPOVariableActionGNN action mapping and batching."""

import unittest
from typing import Dict, Iterable

import torch
from torch_geometric.data import HeteroData

from algo.PPO_VariableActionGNN import ActionGraphHead, PPOVariableActionGNN


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
        x = torch.randn(2, dim)
        data[node_type].x = x
        data[node_type].batch = torch.zeros(x.size(0), dtype=torch.long)
    mask = torch.zeros(num_actions, dtype=torch.bool)
    for idx in valid_indices:
        mask[idx] = True
    data.feasible_action_mask = mask
    data.num_actions = torch.tensor(num_actions)
    data.action_is_charging = torch.zeros(num_actions, dtype=torch.bool)
    if num_actions > 0:
        data.action_is_charging[-1] = True
    data.action_local_index = torch.arange(num_actions, dtype=torch.long)
    data.action_to_node_map = [(idx, bool(data.action_is_charging[idx].item())) for idx in range(num_actions)]
    data.action_charge_durations = torch.ones(num_actions, dtype=torch.float32)
    data.node_id_to_type = {idx: ('delivery', 0) for idx in range(num_actions)}
    data.active_truck_id = torch.tensor([0])
    # Add precomputed action graph features ONLY for feasible actions
    action_graph_features = []
    for idx in valid_indices:
        is_charging = bool(data.action_is_charging[idx].item())
        if is_charging:
            action_graph_features.append([1.0, 1.0, 1.0])  # charging: type=3/3, soc=1.0, duration=1.0
        else:
            action_graph_features.append([0.333, 0.9, 0.0])  # delivery: type=1/3, soc=0.9, duration=0.0
    data.action_graph_features = torch.tensor(action_graph_features, dtype=torch.float32)
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
        self.assertIn(action, [2, 4])

    def test_action_graph_head_ptr_builder(self):
        head = ActionGraphHead(
            encoder_dim=12,
            mlp_dim=8,
            action_feature_dim=self.policy.action_feature_dim,
        )
        embedding = torch.randn(2, 12)
        action_features = torch.randn(5, self.policy.action_feature_dim)
        ptr = torch.tensor([0, 2, 5])
        output = head(embedding, action_features, ptr)
        self.assertEqual(output.ptr.tolist(), [0, 2, 5])
        self.assertEqual(output.logits.numel(), 5)

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
