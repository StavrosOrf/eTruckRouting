"""
models/gat_encoder.py
=====================
Graph Attention Network (GAT) encoder for battery-aware EVRPTW state
representation.

Implements Algorithm 1 and Equations (5)–(9) from:
  "Uncertainty-Aware Deep Reinforcement Learning for Sustainable Electric
   Vehicle Routing: A Hybrid Optimization Framework"

Architecture summary
--------------------
  • L = 4 GAT layers, each with K = 8 attention heads
  • Node features: 7-dimensional type-specific vectors (Eq. 5)
  • Edge features: 4-dimensional arc descriptors including μ_E, σ_E (Eq. 6)
  • Vehicle state (battery, load, time) injected at each layer (Alg. 1, lines 3–4)
  • Output: node embeddings H^(L) ∈ R^{|V| × d} and graph-level embedding h_graph
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class EdgeFeatureAttention(nn.Module):
    """
    Single multi-head attention layer with explicit edge feature integration.

    Implements Eqs. (7)–(9):
        e^k_ij = LeakyReLU( a_k^T [ W_k h_i || W_k h_j || φ(e_ij) ] )
        α^k_ij = softmax_j( e^k_ij )
        h^(ℓ,k)_i = σ( Σ_{j∈N(i)} α^k_ij W_k h_j )

    Parameters
    ----------
    in_dim : int
        Input node embedding dimension.
    out_dim : int
        Per-head output dimension. Total output = n_heads × out_dim.
    edge_dim : int
        Edge feature dimension.
    n_heads : int
        Number of attention heads K.
    dropout : float
        Dropout probability on attention weights.
    leaky_relu_slope : float
        Negative slope for LeakyReLU in attention score computation.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        edge_dim: int,
        n_heads: int = 8,
        dropout: float = 0.1,
        leaky_relu_slope: float = 0.2,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.out_dim = out_dim

        # Linear transforms W_k for each head (stacked for efficiency)
        self.W_node = nn.Linear(in_dim, n_heads * out_dim, bias=False)

        # Edge feature projection φ(e_ij) = W_e · e_ij
        self.W_edge = nn.Linear(edge_dim, n_heads * out_dim, bias=False)

        # Attention vector a_k (one per head)
        self.attn_vec = nn.Parameter(torch.empty(n_heads, 3 * out_dim))

        self.leaky_relu = nn.LeakyReLU(negative_slope=leaky_relu_slope)
        self.dropout = nn.Dropout(p=dropout)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.W_node.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.W_edge.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.attn_vec.unsqueeze(0))

    def forward(
        self,
        node_embeddings: Tensor,
        edge_index: Tensor,
        edge_features: Tensor,
    ) -> Tensor:
        """
        Parameters
        ----------
        node_embeddings : Tensor, shape (N, in_dim)
            Current node embeddings h^(ℓ-1).
        edge_index : Tensor, shape (2, E)
            COO-format edge indices [source_nodes; target_nodes].
        edge_features : Tensor, shape (E, edge_dim)
            Arc feature vectors e_ij.

        Returns
        -------
        Tensor, shape (N, n_heads × out_dim)
            Updated node embeddings (pre-LayerNorm, pre-residual).
        """
        N = node_embeddings.size(0)
        src, dst = edge_index  # (E,), (E,)

        # Project node features: (N, n_heads × out_dim)
        h = self.W_node(node_embeddings)
        h = h.view(N, self.n_heads, self.out_dim)

        # Project edge features: (E, n_heads × out_dim)
        e_feat = self.W_edge(edge_features)
        e_feat = e_feat.view(-1, self.n_heads, self.out_dim)

        # Compute attention scores e^k_ij (Eq. 7–8)
        # Concatenate: [W_k h_i || W_k h_j || φ(e_ij)]
        attn_input = torch.cat(
            [h[src], h[dst], e_feat], dim=-1
        )  # (E, n_heads, 3 × out_dim)
        attn_score = (attn_input * self.attn_vec).sum(dim=-1)  # (E, n_heads)
        attn_score = self.leaky_relu(attn_score)

        # Softmax over neighbours (Eq. 8)
        # NOTE: replace with sparse softmax for large graphs
        attn_weight = self._sparse_softmax(attn_score, dst, N)
        attn_weight = self.dropout(attn_weight)  # (E, n_heads)

        # Aggregate (Eq. 9): h^(ℓ,k)_i = Σ α^k_ij · W_k h_j
        # Expand h[src] → (E, n_heads, out_dim)
        msg = h[src] * attn_weight.unsqueeze(-1)  # (E, n_heads, out_dim)
        out = torch.zeros(N, self.n_heads, self.out_dim, device=node_embeddings.device)
        out.scatter_add_(0, dst.view(-1, 1, 1).expand_as(msg), msg)

        return F.elu(out).view(N, self.n_heads * self.out_dim)

    @staticmethod
    def _sparse_softmax(scores: Tensor, index: Tensor, n_nodes: int) -> Tensor:
        """Numerically-stable softmax over variable-size neighbourhoods."""
        # Subtract per-node max for stability
        max_scores = torch.zeros(n_nodes, scores.size(-1), device=scores.device)
        max_scores.scatter_reduce_(0, index.unsqueeze(-1).expand_as(scores),
                                   scores, reduce="amax", include_self=True)
        exp_scores = torch.exp(scores - max_scores[index])
        denom = torch.zeros_like(max_scores)
        denom.scatter_add_(0, index.unsqueeze(-1).expand_as(exp_scores), exp_scores)
        return exp_scores / (denom[index] + 1e-9)


class GATLayer(nn.Module):
    """
    Single GAT layer: multi-head attention + residual + LayerNorm (Eq. 10).

    h^(ℓ)_i = LayerNorm( Concat_{k=1}^K h^(ℓ,k)_i  +  h^(ℓ-1)_i )
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        n_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        head_dim = hidden_dim // n_heads
        self.attention = EdgeFeatureAttention(
            in_dim=hidden_dim,
            out_dim=head_dim,
            edge_dim=edge_dim,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h: Tensor,
        edge_index: Tensor,
        edge_features: Tensor,
    ) -> Tensor:
        # Multi-head attention with residual (Eq. 10)
        h_attn = self.attention(h, edge_index, edge_features)
        h = self.norm(h + h_attn)
        # Feed-forward sublayer
        h = self.norm2(h + self.ff(h))
        return h


class GATEncoder(nn.Module):
    """
    Full L-layer Graph Attention Network encoder (Algorithm 1).

    Takes a heterogeneous EVRPTW graph with node type-specific features
    (depot, customers, charging stations — Eq. 5) and arc energy features
    (Eq. 6), and produces:
      - Per-node embeddings H^(L) ∈ R^{N × d}  (used by PPO policy)
      - Graph-level embedding h_graph ∈ R^d     (used by Value Network)

    Parameters
    ----------
    node_feature_dim : int
        Raw node feature dimension (7 per Eq. 5).
    edge_feature_dim : int
        Raw edge feature dimension (4 per Eq. 6).
    hidden_dim : int
        Embedding dimension d (default 128).
    n_layers : int
        Number of GAT layers L (default 4).
    n_heads : int
        Attention heads K (default 8).
    vehicle_state_dim : int
        Vehicle context vector dimension [b/B, w/Q, t/T_max] = 3.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        node_feature_dim: int = 7,
        edge_feature_dim: int = 4,
        hidden_dim: int = 128,
        n_layers: int = 4,
        n_heads: int = 8,
        vehicle_state_dim: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        # Input projection (Alg. 1, line 4):
        # Project [x_v ⊕ x_vehicle] → d-dimensional embedding
        self.input_proj = nn.Linear(
            node_feature_dim + vehicle_state_dim, hidden_dim
        )

        # L GAT layers
        self.layers = nn.ModuleList(
            [
                GATLayer(
                    hidden_dim=hidden_dim,
                    edge_dim=edge_feature_dim,
                    n_heads=n_heads,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

        # Global pooling MLP for graph-level embedding (Alg. 1, line 14)
        # h_graph = Σ_i softmax(MLP(h^(L)_i)) · h^(L)_i
        self.pool_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        edge_features: Tensor,
        vehicle_state: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        node_features : Tensor, shape (N, node_feature_dim)
            Type-specific node features x_i (Eq. 5).
        edge_index : Tensor, shape (2, E)
            COO edge indices.
        edge_features : Tensor, shape (E, edge_feature_dim)
            Arc features e_ij including LSTM energy predictions (Eq. 6).
        vehicle_state : Tensor, shape (3,) or (N, 3)
            Current vehicle context [b_cur/B, w_load/Q, t_cur/T_max].
            Broadcast to all nodes if 1-D.

        Returns
        -------
        H_L : Tensor, shape (N, hidden_dim)
            Final node embeddings H^(L).
        h_graph : Tensor, shape (hidden_dim,)
            Graph-level embedding for value network.
        """
        N = node_features.size(0)

        # Alg. 1, lines 3–4: inject vehicle state into every node
        if vehicle_state.dim() == 1:
            vehicle_state = vehicle_state.unsqueeze(0).expand(N, -1)
        x = torch.cat([node_features, vehicle_state], dim=-1)  # (N, 7+3)

        # Alg. 1, line 4: initial projection to d dimensions
        h = F.gelu(self.input_proj(x))  # (N, d)

        # Alg. 1, lines 5–13: L-layer GAT
        for layer in self.layers:
            h = layer(h, edge_index, edge_features)  # (N, d)

        H_L = h  # final node embeddings

        # Alg. 1, line 14: attention-weighted global pooling
        pool_weights = torch.softmax(self.pool_mlp(H_L), dim=0)  # (N, 1)
        h_graph = (pool_weights * H_L).sum(dim=0)  # (d,)

        return H_L, h_graph
