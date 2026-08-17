"""Permutation-equivariant heads for variable-size feasible action sets.

All heads share the same contract: a batch of state embeddings, concatenated
action features, and a CSR-style pointer vector describing which actions belong
to each state.  No head may exchange information across pointer segments.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


ACTION_HEAD_TYPES = ("independent", "complete_gcn", "self_attention")


@dataclass(frozen=True)
class ActionHeadOutput:
    """Action logits and the unchanged variable-batch pointer vector."""

    logits: torch.Tensor
    ptr: torch.Tensor


class _BaseActionHead(nn.Module):
    """Validation and state-conditioned scoring shared by every action head."""

    head_type: str

    def __init__(self, encoder_dim: int, hidden_dim: int, action_feature_dim: int):
        super().__init__()
        for name, value in (
            ("encoder_dim", encoder_dim),
            ("hidden_dim", hidden_dim),
            ("action_feature_dim", action_feature_dim),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.encoder_dim = encoder_dim
        self.hidden_dim = hidden_dim
        self.action_feature_dim = action_feature_dim
        self.state_proj = nn.Linear(encoder_dim, hidden_dim)
        self.score = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _validate_inputs(
        self,
        embedding: torch.Tensor,
        action_features: torch.Tensor,
        ptr: torch.Tensor,
    ) -> torch.Tensor:
        if embedding.ndim == 1:
            embedding = embedding.unsqueeze(0)
        if embedding.ndim != 2 or embedding.shape[1] != self.encoder_dim:
            raise ValueError(
                "embedding must have shape "
                f"[batch, {self.encoder_dim}], got {tuple(embedding.shape)}"
            )
        if (
            action_features.ndim != 2
            or action_features.shape[1] != self.action_feature_dim
        ):
            raise ValueError(
                "action_features must have shape "
                f"[actions, {self.action_feature_dim}], got "
                f"{tuple(action_features.shape)}"
            )
        if ptr.ndim != 1 or ptr.numel() != embedding.shape[0] + 1:
            raise ValueError("ptr must be one-dimensional with batch_size + 1 entries")
        if ptr.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise TypeError("ptr must use an integer dtype")
        if embedding.device != action_features.device or ptr.device != embedding.device:
            raise ValueError("embedding, action_features, and ptr must share a device")
        if not embedding.is_floating_point() or not action_features.is_floating_point():
            raise TypeError(
                "embedding and action_features must be floating-point tensors"
            )
        if not torch.isfinite(embedding).all():
            raise ValueError("embedding contains non-finite values")
        if not torch.isfinite(action_features).all():
            raise ValueError("action_features contain non-finite values")

        if int(ptr[0].item()) != 0:
            raise ValueError("ptr must start at zero")
        if int(ptr[-1].item()) != action_features.shape[0]:
            raise ValueError("ptr must end at the number of action rows")
        if ptr.numel() > 1 and bool((ptr[1:] < ptr[:-1]).any().item()):
            raise ValueError("ptr must be nondecreasing")
        return embedding

    def _score_actions(
        self,
        embedding: torch.Tensor,
        action_embeddings: torch.Tensor,
        ptr: torch.Tensor,
    ) -> ActionHeadOutput:
        if action_embeddings.shape[0] == 0:
            return ActionHeadOutput(action_embeddings.new_zeros((0,)), ptr)
        counts = ptr[1:] - ptr[:-1]
        batch_index = torch.repeat_interleave(
            torch.arange(embedding.shape[0], device=embedding.device), counts
        )
        states = F.relu(self.state_proj(embedding))[batch_index]
        logits = self.score(torch.cat((action_embeddings, states), dim=-1)).squeeze(-1)
        return ActionHeadOutput(logits, ptr)


class IndependentActionHead(_BaseActionHead):
    """Score each action independently after conditioning on the state."""

    head_type = "independent"

    def __init__(self, encoder_dim: int, hidden_dim: int, action_feature_dim: int):
        super().__init__(encoder_dim, hidden_dim, action_feature_dim)
        self.action_proj = nn.Linear(action_feature_dim, hidden_dim)

    def forward(
        self,
        embedding: torch.Tensor,
        action_features: torch.Tensor,
        ptr: torch.Tensor,
    ) -> ActionHeadOutput:
        embedding = self._validate_inputs(embedding, action_features, ptr)
        actions = F.relu(self.action_proj(action_features))
        return self._score_actions(embedding, actions, ptr)


class _CompleteGraphConvolution(nn.Module):
    """One directed complete-graph message-passing layer without self edges."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.self_linear = nn.Linear(hidden_dim, hidden_dim)
        self.neighbor_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        neighbor_sum = torch.zeros_like(x)
        degree = x.new_zeros((x.shape[0], 1))
        if edge_index.shape[1] > 0:
            source, target = edge_index
            neighbor_sum.index_add_(0, target, x[source])
            degree.index_add_(0, target, x.new_ones((target.numel(), 1)))
        neighbor_mean = neighbor_sum / degree.clamp_min(1.0)
        update = self.self_linear(x) + self.neighbor_linear(neighbor_mean)
        return self.norm(x + F.relu(update))


class CompleteGraphGCNActionHead(_BaseActionHead):
    """Message passing on every ordered pair of distinct actions per state."""

    head_type = "complete_gcn"

    def __init__(
        self,
        encoder_dim: int,
        hidden_dim: int,
        action_feature_dim: int,
        *,
        num_layers: int = 2,
    ):
        super().__init__(encoder_dim, hidden_dim, action_feature_dim)
        if (
            isinstance(num_layers, bool)
            or not isinstance(num_layers, int)
            or num_layers <= 0
        ):
            raise ValueError("num_layers must be a positive integer")
        self.action_proj = nn.Linear(action_feature_dim, hidden_dim)
        self.layers = nn.ModuleList(
            _CompleteGraphConvolution(hidden_dim) for _ in range(num_layers)
        )

    def forward(
        self,
        embedding: torch.Tensor,
        action_features: torch.Tensor,
        ptr: torch.Tensor,
    ) -> ActionHeadOutput:
        embedding = self._validate_inputs(embedding, action_features, ptr)
        actions = F.relu(self.action_proj(action_features))
        edge_index = self.build_complete_edge_index(ptr)
        for layer in self.layers:
            actions = layer(actions, edge_index)
        return self._score_actions(embedding, actions, ptr)

    @staticmethod
    def build_complete_edge_index(ptr: torch.Tensor) -> torch.Tensor:
        """Return all within-segment ordered pairs ``source != target``.

        Built as one masked grid rather than a Python loop over segments; the
        edge ordering matches the per-segment construction exactly.
        """
        if ptr.ndim != 1:
            raise ValueError("ptr must be one-dimensional")
        empty = torch.zeros((2, 0), dtype=torch.long, device=ptr.device)
        if ptr.numel() < 2:
            return empty
        counts = (ptr[1:] - ptr[:-1]).to(torch.long)
        keep = counts > 1
        if not bool(keep.any().item()):
            return empty
        counts = counts[keep]
        starts = ptr[:-1].to(torch.long)[keep]
        width = int(counts.max().item())

        positions = torch.arange(width, device=ptr.device)
        rows = positions.view(1, width, 1)
        columns = positions.view(1, 1, width)
        limit = counts.view(-1, 1, 1)
        valid = (rows < limit) & (columns < limit) & (rows != columns)
        source = starts.view(-1, 1, 1) + rows
        target = starts.view(-1, 1, 1) + columns
        return torch.stack(
            (source.expand_as(valid)[valid], target.expand_as(valid)[valid])
        )


class _SelfAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attended, _ = self.attention(
            x, x, x, need_weights=False, key_padding_mask=key_padding_mask
        )
        x = self.attention_norm(x + attended)
        return self.output_norm(x + self.feed_forward(x))


class SelfAttentionActionHead(_BaseActionHead):
    """Encode each feasible action set with shared self-attention blocks."""

    head_type = "self_attention"

    def __init__(
        self,
        encoder_dim: int,
        hidden_dim: int,
        action_feature_dim: int,
        *,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__(encoder_dim, hidden_dim, action_feature_dim)
        if (
            isinstance(num_layers, bool)
            or not isinstance(num_layers, int)
            or num_layers <= 0
        ):
            raise ValueError("num_layers must be a positive integer")
        if (
            isinstance(num_heads, bool)
            or not isinstance(num_heads, int)
            or num_heads <= 0
        ):
            raise ValueError("num_heads must be a positive integer")
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if not isinstance(dropout, (int, float)) or isinstance(dropout, bool):
            raise TypeError("dropout must be numeric")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.action_proj = nn.Linear(action_feature_dim, hidden_dim)
        self.layers = nn.ModuleList(
            _SelfAttentionBlock(hidden_dim, num_heads, float(dropout))
            for _ in range(num_layers)
        )

    def forward(
        self,
        embedding: torch.Tensor,
        action_features: torch.Tensor,
        ptr: torch.Tensor,
    ) -> ActionHeadOutput:
        embedding = self._validate_inputs(embedding, action_features, ptr)
        actions = F.relu(self.action_proj(action_features))
        if actions.shape[0] == 0:
            return self._score_actions(
                embedding, actions.new_zeros((0, self.hidden_dim)), ptr
            )

        # Pad the ragged segments into one batch so that every state's action set
        # is attended in a single kernel launch. `key_padding_mask` keeps each
        # segment isolated exactly as the per-segment loop did.
        counts = ptr[1:] - ptr[:-1]
        nonempty = counts > 0
        segment_counts = counts[nonempty]
        width = int(segment_counts.max().item())
        positions = torch.arange(width, device=actions.device)
        valid = positions.unsqueeze(0) < segment_counts.unsqueeze(1)
        starts = ptr[:-1][nonempty]
        gather = (starts.unsqueeze(1) + positions.unsqueeze(0)).clamp(
            max=actions.shape[0] - 1
        )

        padded = actions[gather] * valid.unsqueeze(-1)
        for layer in self.layers:
            padded = layer(padded, key_padding_mask=~valid)
        encoded = padded[valid]
        return self._score_actions(embedding, encoded, ptr)


def build_action_head(
    head_type: str,
    encoder_dim: int,
    hidden_dim: int,
    action_feature_dim: int,
    *,
    num_layers: int = 2,
    attention_heads: int = 4,
    dropout: float = 0.0,
) -> _BaseActionHead:
    """Construct one of the three approved action-head ablations."""
    if not isinstance(head_type, str):
        raise TypeError("head_type must be a string")
    normalized = head_type.strip().lower().replace("-", "_")
    if normalized == "independent":
        return IndependentActionHead(encoder_dim, hidden_dim, action_feature_dim)
    if normalized == "complete_gcn":
        return CompleteGraphGCNActionHead(
            encoder_dim,
            hidden_dim,
            action_feature_dim,
            num_layers=num_layers,
        )
    if normalized == "self_attention":
        return SelfAttentionActionHead(
            encoder_dim,
            hidden_dim,
            action_feature_dim,
            num_layers=num_layers,
            num_heads=attention_heads,
            dropout=dropout,
        )
    raise ValueError(
        f"unknown action head {head_type!r}; expected one of {ACTION_HEAD_TYPES}"
    )
