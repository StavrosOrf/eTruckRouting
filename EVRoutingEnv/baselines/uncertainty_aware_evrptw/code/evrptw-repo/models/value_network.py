"""
models/value_network.py
=======================
Critic (Value) network V_φ(s) for advantage estimation in PPO training.

Takes the graph-level embedding h_graph produced by GATEncoder's
attention-weighted global pooling (Algorithm 1, line 14) and outputs a
scalar value estimate of the expected cumulative reward from state s.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class ValueNetwork(nn.Module):
    """
    MLP critic network mapping graph embedding + vehicle state → V(s).

    Parameters
    ----------
    hidden_dim : int
        Graph embedding dimension d (must match GATEncoder, default 128).
    vehicle_state_dim : int
        Vehicle context dimension [b/B, w/Q, t/T_max] = 3.
    mlp_dims : tuple of int
        Hidden layer widths for the MLP critic.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        vehicle_state_dim: int = 3,
        mlp_dims: tuple = (256, 128, 64),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        input_dim = hidden_dim + vehicle_state_dim
        layers: list[nn.Module] = []
        in_dim = input_dim
        for out_dim in mlp_dims:
            layers += [
                nn.Linear(in_dim, out_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 1))   # scalar output V(s)
        self.net = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        graph_embedding: Tensor,
        vehicle_state: Tensor,
    ) -> Tensor:
        """
        Parameters
        ----------
        graph_embedding : Tensor, shape (hidden_dim,) or (B, hidden_dim)
            h_graph from GATEncoder.
        vehicle_state : Tensor, shape (3,) or (B, 3)
            Current vehicle context.

        Returns
        -------
        Tensor, shape () or (B,)
            Scalar value estimate V(s).
        """
        if graph_embedding.dim() == 1:
            x = torch.cat([graph_embedding, vehicle_state], dim=-1)
        else:
            x = torch.cat([graph_embedding, vehicle_state], dim=-1)
        return self.net(x).squeeze(-1)
