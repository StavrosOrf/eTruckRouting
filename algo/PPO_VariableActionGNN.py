"""PPO variant with variable-sized discrete action spaces via an action graph head."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch_geometric.data import Batch, HeteroData

from algo.PPO_actionGNN import BATCH_EXCLUDE_KEYS, GNNFeatureEncoder
from algo.action_heads import ActionHeadOutput, build_action_head


VARIABLE_BATCH_EXCLUDE_KEYS = BATCH_EXCLUDE_KEYS + [
    "feasible_action_mask",
    "action_node_type",
    "action_local_index",
    "action_is_charging",
    "action_charge_durations",
    "num_actions",
]


class PPOVariableActorCritic(nn.Module):
    """Actor-Critic network with a shared encoder and selectable action head."""

    def __init__(
        self,
        action_dim: int,
        node_feature_dims: Dict[str, int],
        action_feature_dim: int,
        edge_dim: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 3,
        mlp_dim: int = 128,
        action_head_type: str = "independent",
        action_head_layers: int = 2,
        action_attention_heads: int = 4,
        action_head_dropout: float = 0.0,
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
        self.action_head = build_action_head(
            action_head_type,
            encoder_dim,
            mlp_dim,
            action_feature_dim,
            num_layers=action_head_layers,
            attention_heads=action_attention_heads,
            dropout=action_head_dropout,
        )
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
    ) -> Tuple[ActionHeadOutput, torch.Tensor]:
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
        action_head_type: str = "independent",
        action_head_layers: int = 2,
        action_attention_heads: int = 4,
        action_head_dropout: float = 0.0,
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
        self.action_head_type = action_head_type
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
            action_head_type=action_head_type,
            action_head_layers=action_head_layers,
            action_attention_heads=action_attention_heads,
            action_head_dropout=action_head_dropout,
            device=self.device,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.buffer = VariableRolloutBuffer()

    def act(self, state: HeteroData, action_mask: torch.Tensor = None, deterministic: bool = False, **_):
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
        if deterministic:
            action_choice = torch.argmax(logits)
            logprob = dist.log_prob(action_choice)
        else:
            action_choice = dist.sample()
            logprob = dist.log_prob(action_choice)
        global_action_idx = feasible_idx[action_choice]
        return int(global_action_idx.item()), float(logprob.item()), float(value.squeeze().item())
    
    def act_batch(self, states: List[HeteroData], action_masks: Optional[List[torch.Tensor]] = None, 
                  deterministic: bool = False) -> Tuple[List[int], List[float], List[float]]:
        """Batch action selection for multiple states.
        
        Args:
            states: List of HeteroData states
            action_masks: Optional list of action masks
            deterministic: Whether to select actions deterministically
            
        Returns:
            Tuple of (actions, logprobs, values) lists
        """
        if action_masks is None:
            action_masks = [None] * len(states)
        if len(action_masks) != len(states):
            raise ValueError("action_masks must contain one entry per state")
        
        # Prepare feasible actions for all states
        feasible_indices = []
        for state, mask in zip(states, action_masks):
            _, feasible_idx = self._prepare_feasible_actions(state, mask, device=self.device)
            feasible_indices.append(feasible_idx)
        
        # Build batch action graph inputs
        action_features, ptr = self._build_action_graph_inputs(states, feasible_indices, device=self.device)
        
        # Batch the states
        batch_data = Batch.from_data_list([s.to(self.device) for s in states], exclude_keys=VARIABLE_BATCH_EXCLUDE_KEYS)
        
        # Forward pass
        action_output, values = self.policy(batch_data, action_features, ptr)
        
        # Process outputs for each state
        actions = []
        logprobs = []
        value_list = []
        
        for i in range(len(states)):
            start = action_output.ptr[i].item()
            end = action_output.ptr[i + 1].item()
            logits = action_output.logits[start:end]
            
            if logits.numel() == 0:
                raise RuntimeError(f"No feasible actions available for state {i}.")
            
            dist = Categorical(logits=logits)
            if deterministic:
                action_choice = torch.argmax(logits)
                logprob = dist.log_prob(action_choice)
            else:
                action_choice = dist.sample()
                logprob = dist.log_prob(action_choice)
            
            global_action_idx = feasible_indices[i][action_choice]
            actions.append(int(global_action_idx.item()))
            logprobs.append(float(logprob.item()))
            value_list.append(float(values[i].item()))
        
        return actions, logprobs, value_list

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
    
    def value_batch(self, states: List[HeteroData]) -> List[float]:
        """Batch value estimation for multiple states.
        
        Args:
            states: List of HeteroData states
            
        Returns:
            List of value estimates
        """
        batch_data = Batch.from_data_list([s.to(self.device) for s in states], exclude_keys=VARIABLE_BATCH_EXCLUDE_KEYS)
        with torch.no_grad():
            embedding = self.policy.encoder(batch_data)
            values = self.policy.value_head(embedding).squeeze(-1)
        return [float(v.item()) for v in values]

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
        if len(states) != len(feasible_indices):
            raise ValueError("feasible_indices must contain one tensor per state")
        features = []
        ptr = [0]
        for state, global_indices in zip(states, feasible_indices):
            feasible_mask = state.feasible_action_mask
            state_feasible_indices = torch.nonzero(feasible_mask, as_tuple=False).view(-1)

            if state_feasible_indices.numel() == 0:
                raise RuntimeError("State has no feasible actions in feasible_action_mask")

            global_to_local = {int(g.item()): i for i, g in enumerate(state_feasible_indices)}
            local_indices = []
            for g in global_indices:
                g_int = int(g.item())
                if g_int in global_to_local:
                    local_indices.append(global_to_local[g_int])
                else:
                    raise KeyError(
                        f"Action index {g_int} is absent from the hard feasible set "
                        f"{state_feasible_indices.tolist()}"
                    )

            if not local_indices:
                raise RuntimeError("No feasible action rows were selected")

            local_indices = torch.tensor(local_indices, dtype=torch.long)
            
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
        if not hasattr(state, "action_graph_features"):
            raise ValueError("State is missing action_graph_features")
        action_graph_features = state.action_graph_features
        if action_graph_features.ndim != 2:
            raise ValueError("action_graph_features must be two-dimensional")
        if action_graph_features.shape[1] != self.action_feature_dim:
            raise ValueError(
                "action_graph_features width mismatch: expected "
                f"{self.action_feature_dim}, got {action_graph_features.shape[1]}"
            )
        if local_indices.min() < 0 or local_indices.max() >= action_graph_features.shape[0]:
            raise IndexError("local action feature index is out of range")
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
        if mask.ndim != 1:
            raise ValueError("feasible_action_mask must be one-dimensional")

        if action_mask is not None:
            env_mask = torch.as_tensor(action_mask, dtype=torch.bool)
            mask = self._align_and_combine_masks(mask, env_mask)

        if not mask.any():
            raise RuntimeError(
                "No feasible actions are available; refusing to relax the hard mask."
            )

        indices = torch.nonzero(mask, as_tuple=False).view(-1)
        if device is not None:
            mask = mask.to(device)
            indices = indices.to(device)
        return mask, indices

    @staticmethod
    def _align_and_combine_masks(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if a.ndim != 1 or b.ndim != 1:
            raise ValueError("action masks must be one-dimensional")
        if a.numel() != b.numel():
            raise ValueError(
                "action mask length mismatch: "
                f"state has {a.numel()} actions, environment has {b.numel()}"
            )
        return a & b.to(device=a.device)

    @staticmethod
    def _mask_to_indices(mask: torch.Tensor) -> Tuple[int, torch.Tensor]:
        mask = mask.to(torch.bool)
        if mask.ndim != 1:
            raise ValueError("action mask must be one-dimensional")
        if not mask.any():
            raise RuntimeError(
                "No feasible actions are available; refusing to relax the hard mask."
            )
        indices = torch.nonzero(mask, as_tuple=False).view(-1)
        return int(indices.numel()), indices

    @staticmethod
    def _locate_action_index(indices: torch.Tensor, action_value: torch.Tensor) -> int:
        if indices.numel() == 0:
            raise RuntimeError("Cannot locate an action in an empty feasible set")
        matches = torch.nonzero(indices == int(action_value.item()), as_tuple=False).view(-1)
        if matches.numel() == 0:
            raise ValueError(
                f"Stored action {int(action_value.item())} is not in feasible set "
                f"{indices.tolist()}"
            )
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
