"""
models/ppo_policy.py
====================
Proximal Policy Optimisation (PPO) policy network for EVRPTW routing.

Implements Equations (12)–(14) from:
  "Uncertainty-Aware Deep Reinforcement Learning for Sustainable Electric
   Vehicle Routing: A Hybrid Optimization Framework"

Policy (Eq. 12):
    π_θ(a_j | s) = softmax_j( w^T h^(L)_j / τ )

PPO clipped surrogate objective (Eq. 13):
    L^CLIP(θ) = E_t[ min( r_t(θ) Â_t, clip(r_t(θ), 1-ε, 1+ε) Â_t ) ]

The policy is conditioned on GAT node embeddings H^(L), so routing
decisions naturally incorporate battery-aware spatial-temporal structure.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical


class PPOPolicy(nn.Module):
    """
    PPO actor network that selects routing actions from GAT embeddings.

    At each routing step t, the policy receives:
      - Node embeddings H^(L) from the GAT encoder (N × d)
      - A feasibility mask indicating which nodes are currently actionable
    and returns a probability distribution over feasible next nodes.

    Parameters
    ----------
    hidden_dim : int
        Node embedding dimension d (must match GATEncoder.hidden_dim = 128).
    temperature : float
        Softmax temperature τ controlling exploration (Eq. 12).
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.temperature = temperature

        # Learned projection vector w (Eq. 12)
        self.projection = nn.Linear(hidden_dim, 1, bias=False)

        # Context encoder: compresses graph embedding + vehicle state into
        # a context query used to modulate node scores
        self.context_encoder = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),  # h_graph + [b, w, t]
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.projection.weight)

    def forward(
        self,
        node_embeddings: Tensor,
        graph_embedding: Tensor,
        vehicle_state: Tensor,
        feasibility_mask: Optional[Tensor] = None,
    ) -> Tuple[Categorical, Tensor]:
        """
        Compute action distribution over feasible next nodes.

        Parameters
        ----------
        node_embeddings : Tensor, shape (N, hidden_dim)
            H^(L) from GATEncoder.
        graph_embedding : Tensor, shape (hidden_dim,)
            h_graph from GATEncoder (used for context modulation).
        vehicle_state : Tensor, shape (3,)
            [b_cur/B, w_load/Q, t_cur/T_max].
        feasibility_mask : Tensor, shape (N,) of bool, optional
            True for nodes that are feasible (time windows + capacity).
            Infeasible nodes receive -inf logit before softmax.

        Returns
        -------
        dist : Categorical
            Probability distribution over nodes (Eq. 12).
        logits : Tensor, shape (N,)
            Raw logits before masking, useful for entropy computation.
        """
        # Context query from graph summary + vehicle state
        context_input = torch.cat(
            [graph_embedding, vehicle_state], dim=-1
        )  # (hidden_dim + 3,)
        context = self.context_encoder(context_input)  # (hidden_dim,)

        # Score each node: w^T h^(L)_j / τ  (Eq. 12)
        # Modulate by context via dot product
        modulated = node_embeddings + context.unsqueeze(0)  # (N, hidden_dim)
        logits = self.projection(modulated).squeeze(-1) / self.temperature  # (N,)

        # Apply feasibility mask
        if feasibility_mask is not None:
            logits = logits.masked_fill(~feasibility_mask, float("-inf"))

        dist = Categorical(logits=logits)
        return dist, logits

    def select_action(
        self,
        node_embeddings: Tensor,
        graph_embedding: Tensor,
        vehicle_state: Tensor,
        feasibility_mask: Optional[Tensor] = None,
        greedy: bool = False,
    ) -> Tuple[int, Tensor]:
        """
        Sample or greedily select the next routing action.

        Parameters
        ----------
        greedy : bool
            If True, select argmax (used during evaluation / deployment).

        Returns
        -------
        action : int
            Index of the selected node.
        log_prob : Tensor
            Log-probability of the selected action.
        """
        dist, _ = self.forward(
            node_embeddings, graph_embedding, vehicle_state, feasibility_mask
        )
        if greedy:
            action = dist.probs.argmax().item()
            action_tensor = torch.tensor(action, device=node_embeddings.device)
        else:
            action_tensor = dist.sample()
            action = action_tensor.item()

        log_prob = dist.log_prob(action_tensor)
        return action, log_prob


class PPOUpdater:
    """
    PPO training update logic (Eq. 13 — clipped surrogate objective).

    Manages the on-policy buffer, advantage estimation via Generalised
    Advantage Estimation (GAE), and the clipped policy gradient update.

    Parameters
    ----------
    policy : PPOPolicy
        The actor network θ.
    value_net : nn.Module
        The critic network V_φ (see value_network.py).
    lr : float
        Learning rate for both actor and critic.
    clip_epsilon : float
        ε in the clipped surrogate objective (default 0.2).
    gae_lambda : float
        λ for GAE advantage estimation (default 0.95).
    discount_gamma : float
        Discount factor γ (default 0.99).
    value_coef : float
        Critic loss coefficient in total loss.
    entropy_coef : float
        Entropy bonus coefficient for exploration.
    max_grad_norm : float
        Gradient clipping threshold.
    n_epochs : int
        Number of gradient steps per PPO update.
    """

    def __init__(
        self,
        policy: PPOPolicy,
        value_net: nn.Module,
        lr: float = 3e-4,
        clip_epsilon: float = 0.2,
        gae_lambda: float = 0.95,
        discount_gamma: float = 0.99,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_epochs: int = 4,
    ) -> None:
        self.policy = policy
        self.value_net = value_net
        self.clip_epsilon = clip_epsilon
        self.gae_lambda = gae_lambda
        self.discount_gamma = discount_gamma
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs

        self.optimizer = torch.optim.Adam(
            list(policy.parameters()) + list(value_net.parameters()),
            lr=lr,
        )

    def compute_gae(
        self,
        rewards: Tensor,
        values: Tensor,
        dones: Tensor,
        next_value: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Generalised Advantage Estimation.

        Parameters
        ----------
        rewards : Tensor, shape (T,)
        values  : Tensor, shape (T,)
        dones   : Tensor, shape (T,)  — 1.0 at episode end
        next_value : Tensor, shape ()  — V(s_{T+1})

        Returns
        -------
        advantages : Tensor, shape (T,)
        returns    : Tensor, shape (T,)   (= advantages + values)
        """
        T = rewards.size(0)
        advantages = torch.zeros(T, device=rewards.device)
        gae = torch.tensor(0.0, device=rewards.device)

        for t in reversed(range(T)):
            next_val = next_value if t == T - 1 else values[t + 1]
            delta = rewards[t] + self.discount_gamma * next_val * (1 - dones[t]) \
                    - values[t]
            gae = delta + self.discount_gamma * self.gae_lambda * \
                  (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns

    def update(
        self,
        states: list,
        actions: Tensor,
        old_log_probs: Tensor,
        returns: Tensor,
        advantages: Tensor,
    ) -> dict:
        """
        Perform n_epochs PPO update steps on collected rollout data.

        Returns
        -------
        dict with keys: policy_loss, value_loss, entropy, total_loss
        """
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        metrics = {"policy_loss": 0.0, "value_loss": 0.0,
                   "entropy": 0.0, "total_loss": 0.0}

        for _ in range(self.n_epochs):
            # Recompute log probs and values from current policy
            # NOTE: states must be re-encoded through GAT at each step
            # (placeholder — actual implementation feeds batched graph data)
            new_log_probs, entropy, new_values = self._evaluate_actions(
                states, actions
            )

            # Probability ratio r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
            ratio = torch.exp(new_log_probs - old_log_probs.detach())

            # Clipped surrogate objective (Eq. 13)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon,
                                1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Critic loss
            value_loss = F.mse_loss(new_values, returns)

            # Total loss
            loss = (
                policy_loss
                + self.value_coef * value_loss
                - self.entropy_coef * entropy.mean()
            )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.policy.parameters()) +
                list(self.value_net.parameters()),
                self.max_grad_norm,
            )
            self.optimizer.step()

            metrics["policy_loss"] += policy_loss.item()
            metrics["value_loss"] += value_loss.item()
            metrics["entropy"] += entropy.mean().item()
            metrics["total_loss"] += loss.item()

        for k in metrics:
            metrics[k] /= self.n_epochs

        return metrics

    def _evaluate_actions(
        self,
        states: list,
        actions: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Re-evaluate log-probabilities, entropy, and values for collected
        transitions.  Placeholder — concrete implementation processes
        batched graph observations through GATEncoder → PPOPolicy → ValueNet.
        """
        # TODO: batch-process graph states through GATEncoder
        raise NotImplementedError(
            "Implement batched GAT encoding + policy evaluation here. "
            "See training/trainer.py for the full training loop integration."
        )
