"""
PPO agent with a GNN backbone for the EV truck routing environment.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch_geometric.data import Batch, HeteroData
from torch_geometric.nn import global_mean_pool

from algo.networks import HeteroInteractionLayer


BATCH_EXCLUDE_KEYS = [
    "action_to_node_map",
    "node_id_to_type",
    "node_type_offsets",
    "can_charge_here",
]


class GNNFeatureEncoder(nn.Module):
    """Shared GNN encoder that produces a graph level embedding."""

    def __init__(
        self,
        node_feature_dims: Dict[str, int],
        edge_dim: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 3,
        device: str = "cpu",
    ):
        super().__init__()
        self.node_types = list(node_feature_dims.keys())
        self.device = torch.device(device)
        self.hidden_dim = hidden_dim

        self.input_projections = nn.ModuleDict(
            {
                node_type: nn.Linear(in_dim, hidden_dim)
                for node_type, in_dim in node_feature_dims.items()
            }
        )

        self.layers = nn.ModuleList()
        node_channels = {nt: hidden_dim for nt in self.node_types}
        for _ in range(num_layers):
            self.layers.append(
                HeteroInteractionLayer(node_channels, edge_dim, hidden_dim)
            )

    def forward(self, data: HeteroData) -> torch.Tensor:
        batch_size = self._get_batch_size(data)

        # Project inputs
        x_dict = {}
        for node_type in self.node_types:
            if node_type in data.node_types:
                node_store = data[node_type]
                x = node_store.x
                if x.numel() > 0:
                    x_dict[node_type] = F.relu(self.input_projections[node_type](x))
                    continue
                device = x.device
            else:
                device = self.device
            x_dict[node_type] = torch.zeros((0, self.hidden_dim), device=device)

        edge_index_dict = {edge_type: data[edge_type].edge_index for edge_type in data.edge_types}
        edge_attr_dict = {edge_type: data[edge_type].edge_attr for edge_type in data.edge_types}

        for layer in self.layers:
            x_dict = layer(x_dict, edge_index_dict, edge_attr_dict)
            x_dict = {k: F.relu(v) for k, v in x_dict.items()}

        pooled_features = []
        for node_type in self.node_types:
            features = x_dict.get(node_type)
            if (features is None) or (features.numel() == 0):
                device = features.device if features is not None else self.device
                pooled = torch.zeros((batch_size, self.hidden_dim), device=device)
            else:
                store = data[node_type]
                if hasattr(store, "batch"):
                    batch_idx = store.batch
                    pooled = global_mean_pool(features, batch_idx, size=batch_size)
                else:
                    pooled = features.mean(dim=0, keepdim=True)
                    if batch_size > 1:
                        pooled = pooled.repeat(batch_size, 1)
            pooled_features.append(pooled)

        return torch.cat(pooled_features, dim=-1)

    def _get_batch_size(self, data: HeteroData) -> int:
        if hasattr(data, "num_graphs"):
            return int(data.num_graphs)

        max_batch = -1
        for node_type in self.node_types:
            if node_type in data.node_types and hasattr(data[node_type], "batch"):
                batch_tensor = data[node_type].batch
                if batch_tensor.numel() > 0:
                    max_batch = max(max_batch, int(batch_tensor.max().item()))
        return max_batch + 1 if max_batch >= 0 else 1


class PPOActorCritic(nn.Module):
    """Combined actor critic network built on the GNN encoder."""

    def __init__(
        self,
        action_dim: int,
        node_feature_dims: Dict[str, int],
        edge_dim: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 3,
        mlp_dim: int = 128,
        device: str = "cpu",
    ):
        super().__init__()
        self.encoder = GNNFeatureEncoder(
            node_feature_dims=node_feature_dims,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            device=device,
        )
        encoder_dim = hidden_dim * len(node_feature_dims)

        self.policy_head = nn.Sequential(
            nn.Linear(encoder_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, action_dim),
        )

        self.value_head = nn.Sequential(
            nn.Linear(encoder_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, 1),
        )

    def forward(self, data: HeteroData):
        embedding = self.encoder(data)
        logits = self.policy_head(embedding)
        value = self.value_head(embedding).squeeze(-1)
        return logits, value


@dataclass
class RolloutBatch:
    states: List[HeteroData]
    actions: torch.Tensor
    logprobs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor


class RolloutBuffer:
    """Simple on-policy buffer for PPO."""

    def __init__(self):
        self.clear()

    def add(
        self,
        state: HeteroData,
        action: int,
        logprob: float,
        reward: float,
        done: bool,
        value: float,
    ):
        self.states.append(state.to("cpu"))
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def compute_returns_and_advantages(
        self, gamma: float, gae_lambda: float, last_value: float
    ):
        values = self.values + [last_value]
        gae = 0.0
        returns = []
        advantages = []

        for step in reversed(range(len(self.rewards))):
            mask = 0.0 if self.dones[step] else 1.0
            delta = (
                self.rewards[step]
                + gamma * values[step + 1] * mask
                - values[step]
            )
            gae = delta + gamma * gae_lambda * mask * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[step])

        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)
        actions = torch.tensor(self.actions, dtype=torch.long)
        logprobs = torch.tensor(self.logprobs, dtype=torch.float32)

        return returns, advantages, actions, logprobs

    def get_minibatch(self, indices: Sequence[int]) -> RolloutBatch:
        subset_states = [self.states[i] for i in indices]
        return RolloutBatch(
            states=subset_states,
            actions=torch.tensor([self.actions[i] for i in indices], dtype=torch.long),
            logprobs=torch.tensor([self.logprobs[i] for i in indices], dtype=torch.float32),
            returns=torch.tensor([self.returns[i] for i in indices], dtype=torch.float32),
            advantages=torch.tensor([self.advantages[i] for i in indices], dtype=torch.float32),
        )

    def finalize(self, returns: torch.Tensor, advantages: torch.Tensor):
        self.returns = returns
        self.advantages = advantages

    def clear(self):
        self.states: List[HeteroData] = []
        self.actions: List[int] = []
        self.logprobs: List[float] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.values: List[float] = []


class PPOActionGNN:
    """Main PPO agent class used by the training loop."""

    def __init__(
        self,
        action_dim: int,
        node_feature_dims: Dict[str, int],
        edge_dim: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 3,
        mlp_dim: int = 128,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_coef: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 10,
        minibatch_size: int = 128,
        device: str = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size

        self.policy = PPOActorCritic(
            action_dim=action_dim,
            node_feature_dims=node_feature_dims,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            mlp_dim=mlp_dim,
            device=self.device,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.buffer = RolloutBuffer()

    def act(self, state: HeteroData):
        state = state.to(self.device)
        logits, value = self.policy(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        logprob = dist.log_prob(action)
        return action.item(), logprob.item(), value.squeeze().item()

    def select_action(
        self,
        state: HeteroData,
        deterministic: bool = True,
        expl_noise: float = None,
        **kwargs,
    ):
        if expl_noise is not None:
            deterministic = expl_noise == 0
        state = state.to(self.device)
        logits, _ = self.policy(state)
        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            dist = Categorical(logits=logits)
            action = dist.sample()
        return action.squeeze().item()

    def store_transition(self, *args, **kwargs):
        self.buffer.add(*args, **kwargs)

    def update(self, last_value: float):
        if len(self.buffer.rewards) == 0:
            return {}

        returns, advantages, actions, logprobs = self.buffer.compute_returns_and_advantages(
            self.gamma, self.gae_lambda, last_value
        )
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        self.buffer.finalize(returns, advantages)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_items = len(self.buffer.states) * self.ppo_epochs

        indices = np.arange(len(self.buffer.states))
        for _ in range(self.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, len(indices), self.minibatch_size):
                mb_idx = indices[start : start + self.minibatch_size]
                state_batch = Batch.from_data_list(
                    [self.buffer.states[i] for i in mb_idx], exclude_keys=BATCH_EXCLUDE_KEYS
                ).to(self.device)

                mb_actions = actions[mb_idx].to(self.device)
                mb_old_logprobs = logprobs[mb_idx].to(self.device)
                mb_returns = returns[mb_idx].to(self.device)
                mb_advantages = advantages[mb_idx].to(self.device)

                logits, values = self.policy(state_batch)
                dist = Categorical(logits=logits)
                new_logprobs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                ratio = (new_logprobs - mb_old_logprobs).exp()
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values, mb_returns)

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()

        self.buffer.clear()
        updates = len(indices) / self.minibatch_size if self.minibatch_size > 0 else 1
        norm = max(total_items, 1)
        return {
            "policy_loss": total_policy_loss / max(total_items, 1),
            "value_loss": total_value_loss / max(total_items, 1),
            "entropy": total_entropy / max(total_items, 1),
        }

    def value(self, state: HeteroData) -> float:
        state = state.to(self.device)
        with torch.no_grad():
            _, value = self.policy(state)
        return value.squeeze().item()

    def save(self, path: str):
        torch.save(self.policy.state_dict(), f"{path}_actor.pt")

    def load(self, path: str):
        self.policy.load_state_dict(torch.load(f"{path}_actor.pt", map_location=self.device))
