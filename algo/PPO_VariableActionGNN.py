"""PPO variant with variable-sized discrete action spaces via an action graph head."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch_geometric.data import Batch, HeteroData
from torch_geometric.nn import GCNConv

from algo.PPO_actionGNN import GNNFeatureEncoder, BATCH_EXCLUDE_KEYS


VARIABLE_BATCH_EXCLUDE_KEYS = BATCH_EXCLUDE_KEYS + [
    "feasible_action_mask",
    "action_node_type",
    "action_local_index",
    "action_is_charging",
    "action_charge_durations",
    "num_actions",
]


@dataclass
class ActionGraphOutput:
    """Holds logits and prefix sums for variable action batches."""

    logits: torch.Tensor
    ptr: torch.Tensor


class ActionGraphHead(nn.Module):
    """Two-layer GCN over fully connected feasible action graphs."""

    def __init__(self, encoder_dim: int, mlp_dim: int, action_feature_dim: int):
        super().__init__()
        self.state_proj = nn.Linear(encoder_dim, mlp_dim)
        self.action_proj = nn.Linear(action_feature_dim, mlp_dim)
        self.gcn_layers = nn.ModuleList([
            GCNConv(mlp_dim, mlp_dim, add_self_loops=True),
            GCNConv(mlp_dim, mlp_dim, add_self_loops=True),
        ])

    def forward(
        self,
        embedding: torch.Tensor,
        action_features: torch.Tensor,
        ptr: torch.Tensor,
    ) -> ActionGraphOutput:
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        state_repr = F.relu(self.state_proj(embedding))
        if action_features.numel() == 0:
            return ActionGraphOutput(logits=action_features.new_zeros((0,)), ptr=ptr)

        x = F.relu(self.action_proj(action_features))
        edge_index = self._build_fully_connected_edges(ptr, action_features.device)
        for conv in self.gcn_layers:
            x = F.relu(conv(x, edge_index)) if x.numel() > 0 else x

        logits = x.new_zeros(x.size(0))
        batch_size = state_repr.size(0)
        for i in range(batch_size):
            start = ptr[i].item()
            end = ptr[i + 1].item()
            if end <= start:
                continue
            logits[start:end] = (x[start:end] * state_repr[i]).sum(dim=-1)

        return ActionGraphOutput(logits=logits, ptr=ptr)

    @staticmethod
    def _build_fully_connected_edges(ptr: torch.Tensor, device: torch.device) -> torch.Tensor:
        if ptr.numel() <= 1:
            return torch.zeros((2, 0), dtype=torch.long, device=device)
        edges = []
        for i in range(ptr.numel() - 1):
            start = int(ptr[i].item())
            end = int(ptr[i + 1].item())
            count = end - start
            if count <= 1:
                continue
            node_range = torch.arange(start, end, device=device)
            src = node_range[:-1]
            dst = node_range[1:]
            edges.append(torch.stack([src, dst], dim=0))
            edges.append(torch.stack([dst, src], dim=0))
        if not edges:
            return torch.zeros((2, 0), dtype=torch.long, device=device)
        return torch.cat(edges, dim=1)


class PPOVariableActorCritic(nn.Module):
    """Actor-Critic network with shared encoder and action graph head."""

    def __init__(
        self,
        action_dim: int,
        node_feature_dims: Dict[str, int],
        action_feature_dim: int,
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
        self.action_dim = action_dim
        encoder_dim = hidden_dim * len(node_feature_dims)
        self.action_head = ActionGraphHead(encoder_dim, mlp_dim, action_feature_dim)
        self.value_head = nn.Sequential(
            nn.Linear(encoder_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, 1),
        )

    def forward(
        self,
        data: HeteroData,
        action_features: torch.Tensor,
        ptr: torch.Tensor,
    ) -> Tuple[ActionGraphOutput, torch.Tensor]:
        embedding = self.encoder(data)
        action_output = self.action_head(embedding, action_features, ptr)
        value = self.value_head(embedding).squeeze(-1)
        return action_output, value


class VariableRolloutBuffer:
    """Rollout buffer tailored for variable-sized action spaces."""

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
        action_mask: torch.Tensor,
    ):
        self.states.append(state.to("cpu"))
        self.actions.append(int(action))
        self.logprobs.append(float(logprob))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.values.append(float(value))
        mask_tensor = torch.as_tensor(action_mask, dtype=torch.bool)
        self.action_masks.append(mask_tensor.cpu())

    def compute_returns_and_advantages(
        self, gamma: float, gae_lambda: float, last_value: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        self.action_masks: List[torch.Tensor] = []


class PPOVariableActionGNN:
    """PPO agent variant that handles variable action counts per timestep."""

    def __init__(
        self,
        action_dim: int,
        node_feature_dims: Dict[str, int],
        edge_dim: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 3,
        mlp_dim: int = 128,
        charge_durations: Optional[List[float]] = None,
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
        self.action_dim = action_dim
        self.node_feature_dims = node_feature_dims
        self.charge_durations = list(charge_durations) if charge_durations is not None else []
        # [normalized_action_type, resulting_soc, charge_duration_norm]
        self.action_feature_dim = 3

        self.policy = PPOVariableActorCritic(
            action_dim=action_dim,
            node_feature_dims=node_feature_dims,
            action_feature_dim=self.action_feature_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            mlp_dim=mlp_dim,
            device=self.device,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.buffer = VariableRolloutBuffer()

    def act(self, state: HeteroData, action_mask: torch.Tensor = None, **_):
        _, feasible_idx = self._prepare_feasible_actions(state, action_mask, device=self.device)
        action_features, ptr = self._build_action_graph_inputs([state], [feasible_idx], device=self.device)
        data = state.to(self.device)
        action_output, value = self.policy(data, action_features, ptr)
        start = action_output.ptr[0].item()
        end = action_output.ptr[1].item()
        logits = action_output.logits[start:end]
        if logits.numel() == 0:
            raise RuntimeError("No feasible actions available for the current state.")
        dist = Categorical(logits=logits)
        action_choice = dist.sample()
        logprob = dist.log_prob(action_choice)
        global_action_idx = feasible_idx[action_choice]
        return int(global_action_idx.item()), float(logprob.item()), float(value.squeeze().item())

    def to_env_action(self, state: HeteroData, action_idx: int) -> Tuple[int, float, bool]:
        """Map a chosen action index to the environment tuple API."""
        if not hasattr(state, "action_to_node_map"):
            raise ValueError("State missing action_to_node_map needed for env action translation.")
        actions = state.action_to_node_map
        if action_idx < 0 or action_idx >= len(actions):
            raise IndexError(f"action_idx {action_idx} out of bounds for action_to_node_map (size {len(actions)})")
        node_id, is_charging = actions[action_idx]

        # Pull matching charge duration if it exists
        duration = 0.0
        if is_charging:
            if hasattr(state, "action_charge_durations") and len(state.action_charge_durations) > action_idx:
                duration = float(state.action_charge_durations[action_idx])
            elif self.charge_durations:
                duration = float(self.charge_durations[0])
            else:
                duration = 1.0

        # Convert possible tensors to python types
        if hasattr(node_id, "item"):
            node_id = int(node_id.item())
        else:
            node_id = int(node_id)

        return node_id, float(duration), bool(is_charging)

    def select_action(
        self,
        state: HeteroData,
        deterministic: bool = True,
        expl_noise: float = None,
        action_mask: torch.Tensor = None,
        **_,
    ) -> int:
        if expl_noise is not None:
            deterministic = expl_noise == 0
        _, feasible_idx = self._prepare_feasible_actions(state, action_mask, device=self.device)
        action_features, ptr = self._build_action_graph_inputs([state], [feasible_idx], device=self.device)
        data = state.to(self.device)
        action_output, _ = self.policy(data, action_features, ptr)
        start, end = action_output.ptr[0].item(), action_output.ptr[1].item()
        logits = action_output.logits[start:end]
        if logits.numel() == 0:
            raise RuntimeError("No feasible actions available for the current state.")
        if deterministic:
            action_choice = torch.argmax(logits)
        else:
            dist = Categorical(logits=logits)
            action_choice = dist.sample()
        global_action_idx = feasible_idx[action_choice]
        return int(global_action_idx.item())

    def store_transition(
        self,
        state: HeteroData,
        action: int,
        logprob: float,
        reward: float,
        done: bool,
        value: float,
        action_mask: Optional[torch.Tensor] = None,
    ):
        # Always ensure state is on CPU for storage (act() moves it to GPU)
        # Use detach().clone() to create a separate copy on CPU to avoid any reference issues
        state_cpu = HeteroData()
        
        # Copy node features
        for node_type in state.node_types:
            if hasattr(state[node_type], 'x'):
                state_cpu[node_type].x = state[node_type].x.detach().cpu()
            # Copy any other node-level attributes
            for key in state[node_type].keys():
                if key != 'x':
                    attr = state[node_type][key]
                    if isinstance(attr, torch.Tensor):
                        state_cpu[node_type][key] = attr.detach().cpu()
                    else:
                        state_cpu[node_type][key] = attr
        
        # Copy edge features
        for edge_type in state.edge_types:
            if hasattr(state[edge_type], 'edge_index'):
                state_cpu[edge_type].edge_index = state[edge_type].edge_index.detach().cpu()
            if hasattr(state[edge_type], 'edge_attr'):
                state_cpu[edge_type].edge_attr = state[edge_type].edge_attr.detach().cpu()
            # Copy any other edge-level attributes
            for key in state[edge_type].keys():
                if key not in ['edge_index', 'edge_attr']:
                    attr = state[edge_type][key]
                    if isinstance(attr, torch.Tensor):
                        state_cpu[edge_type][key] = attr.detach().cpu()
                    else:
                        state_cpu[edge_type][key] = attr
        
        # Copy global attributes
        for key in state.keys():
            if key not in ['_node_store_dict', '_edge_store_dict']:
                attr = state[key]
                if isinstance(attr, torch.Tensor):
                    state_cpu[key] = attr.detach().cpu()
                else:
                    state_cpu[key] = attr
        
        combined_mask, _ = self._prepare_feasible_actions(state_cpu, action_mask, device="cpu")
        self.buffer.add(state_cpu, action, logprob, reward, done, value, combined_mask)

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
                if len(mb_idx) == 0:
                    continue

                subset_states = [self.buffer.states[i] for i in mb_idx]
                state_batch = Batch.from_data_list(
                    subset_states, exclude_keys=VARIABLE_BATCH_EXCLUDE_KEYS
                ).to(self.device)

                mb_actions = actions[mb_idx].to(self.device)
                mb_old_logprobs = logprobs[mb_idx].to(self.device)
                mb_returns = returns[mb_idx].to(self.device)
                mb_advantages = advantages[mb_idx].to(self.device)

                feasible_info = [
                    self._mask_to_indices(self.buffer.action_masks[i].to(self.device))
                    for i in mb_idx
                ]
                feasible_indices = [info[1] for info in feasible_info]
                action_positions = torch.tensor(
                    [self._locate_action_index(feasible_indices[i], mb_actions[i]) for i in range(len(mb_idx))],
                    dtype=torch.long,
                    device=self.device,
                )

                action_features, ptr = self._build_action_graph_inputs(subset_states, feasible_indices, device=self.device)
                action_output, values = self.policy(state_batch, action_features, ptr)
                new_logprobs, entropy = self._gather_logprobs(
                    action_output.logits, action_output.ptr, action_positions
                )

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
        norm = max(total_items, 1)
        return {
            "policy_loss": total_policy_loss / norm,
            "value_loss": total_value_loss / norm,
            "entropy": total_entropy / norm,
        }

    def value(self, state: HeteroData) -> float:
        data = state.to(self.device)
        with torch.no_grad():
            embedding = self.policy.encoder(data)
            value = self.policy.value_head(embedding).squeeze(-1)
        return float(value.mean().item())

    def save(self, path: str):
        torch.save(self.policy.state_dict(), f"{path}_actor.pt")

    def load(self, path: str):
        self.policy.load_state_dict(torch.load(f"{path}_actor.pt", map_location=self.device))

    def _build_action_graph_inputs(
        self,
        states: List[HeteroData],
        feasible_indices: List[torch.Tensor],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        features = []
        ptr = [0]
        for state, global_indices in zip(states, feasible_indices):
            # action_graph_features contains features for feasible actions in state.feasible_action_mask
            # global_indices are the indices we actually want to use (subset of feasible actions)
            # We need to map global_indices to positions in action_graph_features
            
            # Get the indices of feasible actions from the state's mask
            feasible_mask = state.feasible_action_mask
            state_feasible_indices = torch.nonzero(feasible_mask, as_tuple=False).view(-1)
            
            # Create mapping from global action index to local index in action_graph_features
            global_to_local = {int(g.item()): i for i, g in enumerate(state_feasible_indices)}
            
            # Map the requested global indices to local indices
            # All global_indices must be in the state's feasible set
            local_indices = torch.tensor(
                [global_to_local[int(g.item())] for g in global_indices],
                dtype=torch.long
            )
            
            node_features = self._action_node_features(state, local_indices)
            features.append(node_features)
            ptr.append(ptr[-1] + node_features.size(0))

        if features:
            action_features = torch.cat(features, dim=0)
        else:
            action_features = torch.zeros((0, self.action_feature_dim), dtype=torch.float32)

        action_features = action_features.to(device)
        ptr_tensor = torch.tensor(ptr, dtype=torch.long, device=device)
        return action_features, ptr_tensor

    def _action_node_features(self, state: HeteroData, local_indices: torch.Tensor) -> torch.Tensor:
        """
        Extract action graph features from state for the given local action indices.
        The state contains precomputed action_graph_features for feasible actions only.
        The local_indices are 0-based indices within the feasible action set.
        """
        if local_indices.numel() == 0:
            return torch.zeros((0, self.action_feature_dim), dtype=torch.float32)
        
        # State contains precomputed action graph features for feasible actions only
        action_graph_features = state.action_graph_features
        
        # Extract features for the requested local indices
        selected_features = action_graph_features[local_indices.to(action_graph_features.device)]
        
        return selected_features.cpu()

    def _prepare_feasible_actions(
        self,
        state: HeteroData,
        action_mask: torch.Tensor = None,
        device: Optional[Union[torch.device, str]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not hasattr(state, "feasible_action_mask"):
            raise ValueError("State is missing feasible_action_mask attribute.")
        mask = torch.as_tensor(state.feasible_action_mask, dtype=torch.bool)
        if action_mask is not None:
            env_mask = torch.as_tensor(action_mask, dtype=torch.bool)
            mask = self._align_and_combine_masks(mask, env_mask)
        if not mask.any():
            mask = torch.ones_like(mask, dtype=torch.bool)
        indices = torch.nonzero(mask, as_tuple=False).view(-1)
        if indices.numel() == 0:
            indices = torch.arange(mask.numel(), dtype=torch.long)
        if device is not None:
            mask = mask.to(device)
            indices = indices.to(device)
        return mask, indices

    @staticmethod
    def _align_and_combine_masks(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if a.numel() == b.numel():
            return a & b.to(device=a.device)
        min_len = min(a.numel(), b.numel())
        a_trim = a[:min_len]
        b_trim = b[:min_len].to(device=a.device)
        combined = a_trim & b_trim
        if combined.numel() < a.numel():
            pad = torch.zeros(a.numel() - combined.numel(), dtype=torch.bool, device=a.device)
            combined = torch.cat([combined, pad], dim=0)
        return combined

    @staticmethod
    def _mask_to_indices(mask: torch.Tensor) -> Tuple[int, torch.Tensor]:
        mask = mask.to(torch.bool)
        if not mask.any():
            mask = torch.ones_like(mask, dtype=torch.bool)
        indices = torch.nonzero(mask, as_tuple=False).view(-1)
        if indices.numel() == 0:
            indices = torch.arange(mask.numel(), device=mask.device)
        return int(indices.numel()), indices

    @staticmethod
    def _locate_action_index(indices: torch.Tensor, action_value: torch.Tensor) -> int:
        if indices.numel() == 0:
            return 0
        matches = torch.nonzero(indices == int(action_value.item()), as_tuple=False).view(-1)
        if matches.numel() == 0:
            return int(torch.clamp(action_value, 0, indices.numel() - 1).item())
        return int(matches[0].item())

    def _gather_logprobs(
        self,
        logits: torch.Tensor,
        ptr: torch.Tensor,
        positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logprob_list = []
        entropy_list = []
        for idx in range(len(positions)):
            start = ptr[idx].item()
            end = ptr[idx + 1].item()
            slice_logits = logits[start:end]
            if slice_logits.numel() == 0:
                raise RuntimeError("Encountered empty logits slice during PPO update.")
            dist = Categorical(logits=slice_logits)
            logprob_list.append(dist.log_prob(positions[idx]))
            entropy_list.append(dist.entropy())
        new_logprobs = torch.stack(logprob_list)
        entropy = torch.stack(entropy_list).mean()
        return new_logprobs, entropy
