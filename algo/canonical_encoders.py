"""Canonical state encoders: flat, DeepSets, and heterogeneous graph.

All three consume the identical :class:`CanonicalTensors` unpacked from the
environment's flat observation, and all three emit a fixed-width state
embedding for the approved action heads.  They differ only in inductive bias.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from algo.canonical_state import (
    ACTION_FEATURE_DIM,
    EDGE_FEATURE_DIM,
    GLOBAL_FEATURE_DIM,
    NODE_FEATURE_DIMS,
    CanonicalTensors,
)
from EVRoutingEnv.state.features import NODE_TYPES, RELATION_TYPES
from EVRoutingEnv.state.representations import CanonicalShapeSpec


STATE_ENCODER_TYPES = ("flat", "deep_sets", "hetero_graph")
_CLIP = 10.0
_EPSILON = 1e-6


class CanonicalNormalizer(nn.Module):
    """Shared running standardization applied identically to every encoder.

    Statistics are accumulated only over unmasked entries so that padding never
    shifts the mean.  Because the module is shared by all encoders, feature
    scaling can never become a confound in the architecture comparison.
    """

    def __init__(self) -> None:
        super().__init__()
        widths = {
            **{
                f"node_{node_type}": NODE_FEATURE_DIMS[node_type]
                for node_type in NODE_TYPES
            },
            "action": ACTION_FEATURE_DIM,
            "pairwise": EDGE_FEATURE_DIM,
            "global": GLOBAL_FEATURE_DIM,
        }
        for name, width in widths.items():
            self.register_buffer(f"{name}_mean", torch.zeros(width))
            self.register_buffer(f"{name}_var", torch.ones(width))
            self.register_buffer(f"{name}_count", torch.zeros(()))
        self._widths = widths

    @torch.no_grad()
    def observe(self, tensors: CanonicalTensors) -> None:
        """Update running statistics from one batch of canonical tensors."""
        for node_type in NODE_TYPES:
            self._update(
                f"node_{node_type}",
                tensors.nodes[node_type][tensors.node_masks[node_type]],
            )
        self._update("action", tensors.action[tensors.action_padding_mask])
        pairwise_rows = [
            tensors.pairwise[relation][tensors.pairwise_mask[relation]]
            for relation in RELATION_TYPES
        ]
        self._update("pairwise", torch.cat(pairwise_rows, dim=0))
        self._update("global", tensors.global_features)

    @torch.no_grad()
    def _update(self, name: str, values: torch.Tensor) -> None:
        if values.numel() == 0:
            return
        values = values.reshape(-1, self._widths[name]).to(torch.float32)
        batch_count = torch.tensor(
            float(values.shape[0]), device=values.device, dtype=torch.float32
        )
        batch_mean = values.mean(dim=0)
        batch_var = values.var(dim=0, unbiased=False)

        mean = getattr(self, f"{name}_mean")
        var = getattr(self, f"{name}_var")
        count = getattr(self, f"{name}_count")

        total = count + batch_count
        delta = batch_mean - mean
        new_mean = mean + delta * (batch_count / total)
        m_a = var * count
        m_b = batch_var * batch_count
        new_var = (m_a + m_b + delta.pow(2) * count * batch_count / total) / total

        mean.copy_(new_mean)
        var.copy_(torch.clamp(new_var, min=_EPSILON))
        count.copy_(total)

    def _standardize(self, name: str, values: torch.Tensor) -> torch.Tensor:
        count = getattr(self, f"{name}_count")
        if float(count.item()) < 2.0:
            return values
        mean = getattr(self, f"{name}_mean")
        var = getattr(self, f"{name}_var")
        return torch.clamp((values - mean) / torch.sqrt(var + _EPSILON), -_CLIP, _CLIP)

    def forward(self, tensors: CanonicalTensors) -> CanonicalTensors:
        """Return standardized tensors with masks and semantics untouched."""
        nodes = {
            node_type: self._standardize(f"node_{node_type}", tensors.nodes[node_type])
            * tensors.node_masks[node_type].unsqueeze(-1)
            for node_type in NODE_TYPES
        }
        pairwise = {
            relation: self._standardize("pairwise", tensors.pairwise[relation])
            * tensors.pairwise_mask[relation].unsqueeze(-1)
            for relation in RELATION_TYPES
        }
        return CanonicalTensors(
            nodes=nodes,
            node_masks=tensors.node_masks,
            action=self._standardize("action", tensors.action)
            * tensors.action_padding_mask.unsqueeze(-1),
            action_padding_mask=tensors.action_padding_mask,
            pairwise=pairwise,
            pairwise_mask=tensors.pairwise_mask,
            global_features=self._standardize("global", tensors.global_features),
        )


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
        nn.ReLU(),
    )


def _masked_pool(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Concatenate masked mean and masked max pooling over dimension one."""
    weights = mask.unsqueeze(-1).to(values.dtype)
    total = weights.sum(dim=1).clamp_min(1.0)
    mean = (values * weights).sum(dim=1) / total
    neutral = torch.finfo(values.dtype).min / 2
    maximum = torch.where(mask.unsqueeze(-1), values, torch.full_like(values, neutral))
    maximum = maximum.max(dim=1).values
    maximum = torch.where(
        mask.any(dim=1, keepdim=True), maximum, torch.zeros_like(maximum)
    )
    return torch.cat((mean, maximum), dim=-1)


class _BaseStateEncoder(nn.Module):
    """Shared normalization and output-dimension contract."""

    encoder_type: str

    def __init__(self, shape: CanonicalShapeSpec, hidden_dim: int, output_dim: int):
        super().__init__()
        for name, value in (("hidden_dim", hidden_dim), ("output_dim", output_dim)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.shape = shape
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.normalizer = CanonicalNormalizer()

    def observe(self, tensors: CanonicalTensors) -> None:
        self.normalizer.observe(tensors)

    def forward(self, tensors: CanonicalTensors) -> torch.Tensor:
        raise NotImplementedError


class FlatStateEncoder(_BaseStateEncoder):
    """Multilayer perceptron over the whole canonical observation."""

    encoder_type = "flat"

    def __init__(
        self,
        shape: CanonicalShapeSpec,
        hidden_dim: int,
        output_dim: int,
        *,
        num_layers: int = 2,
    ):
        super().__init__(shape, hidden_dim, output_dim)
        if (
            isinstance(num_layers, bool)
            or not isinstance(num_layers, int)
            or num_layers <= 0
        ):
            raise ValueError("num_layers must be a positive integer")
        layers: list[nn.Module] = [nn.Linear(shape.flat_size, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.extend((nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))
        layers.extend((nn.Linear(hidden_dim, output_dim), nn.ReLU()))
        self.network = nn.Sequential(*layers)

    def forward(self, tensors: CanonicalTensors) -> torch.Tensor:
        normalized = self.normalizer(tensors)
        batch = normalized.batch_size
        blocks = [
            normalized.nodes[node_type].reshape(batch, -1) for node_type in NODE_TYPES
        ]
        blocks.append(normalized.action.reshape(batch, -1))
        blocks.extend(
            normalized.node_masks[node_type]
            .reshape(batch, -1)
            .to(normalized.action.dtype)
            for node_type in NODE_TYPES
        )
        blocks.append(
            normalized.action_padding_mask.reshape(batch, -1).to(
                normalized.action.dtype
            )
        )
        blocks.extend(
            normalized.pairwise[relation].reshape(batch, -1)
            for relation in RELATION_TYPES
        )
        blocks.extend(
            normalized.pairwise_mask[relation]
            .reshape(batch, -1)
            .to(normalized.action.dtype)
            for relation in RELATION_TYPES
        )
        blocks.append(normalized.global_features)
        return self.network(torch.cat(blocks, dim=-1))


class DeepSetsStateEncoder(_BaseStateEncoder):
    """Permutation-invariant set encoder with a pairwise relation aggregator."""

    encoder_type = "deep_sets"

    def __init__(
        self,
        shape: CanonicalShapeSpec,
        hidden_dim: int,
        output_dim: int,
        *,
        num_layers: int = 2,
    ):
        super().__init__(shape, hidden_dim, output_dim)
        if (
            isinstance(num_layers, bool)
            or not isinstance(num_layers, int)
            or num_layers <= 0
        ):
            raise ValueError("num_layers must be a positive integer")
        self.node_encoders = nn.ModuleDict(
            {
                node_type: _mlp(NODE_FEATURE_DIMS[node_type], hidden_dim, hidden_dim)
                for node_type in NODE_TYPES
            }
        )
        # One aggregator per typed relation keeps relations distinguishable while
        # staying invariant to the order of sources and of targets. The pair
        # tensor is the dominant cost, so the aggregator runs at a reduced width
        # and its first layer is applied factorwise (see forward).
        relation_dim = max(8, hidden_dim // 4)
        self.relation_dim = relation_dim
        self.relation_source = nn.ModuleDict(
            {
                _relation_key(relation): nn.Linear(hidden_dim, relation_dim)
                for relation in RELATION_TYPES
            }
        )
        self.relation_target = nn.ModuleDict(
            {
                _relation_key(relation): nn.Linear(hidden_dim, relation_dim, bias=False)
                for relation in RELATION_TYPES
            }
        )
        self.relation_edge = nn.ModuleDict(
            {
                _relation_key(relation): nn.Linear(
                    EDGE_FEATURE_DIM, relation_dim, bias=False
                )
                for relation in RELATION_TYPES
            }
        )
        self.relation_output = nn.ModuleDict(
            {
                _relation_key(relation): nn.Linear(relation_dim, relation_dim)
                for relation in RELATION_TYPES
            }
        )
        self.global_encoder = _mlp(GLOBAL_FEATURE_DIM, hidden_dim, hidden_dim)
        summary_dim = (
            2 * hidden_dim * len(NODE_TYPES)
            + 2 * relation_dim * len(RELATION_TYPES)
            + hidden_dim
        )
        layers: list[nn.Module] = [nn.Linear(summary_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.extend((nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))
        layers.extend((nn.Linear(hidden_dim, output_dim), nn.ReLU()))
        self.readout = nn.Sequential(*layers)

    def forward(self, tensors: CanonicalTensors) -> torch.Tensor:
        normalized = self.normalizer(tensors)
        node_embeddings = {
            node_type: self.node_encoders[node_type](normalized.nodes[node_type])
            * normalized.node_masks[node_type].unsqueeze(-1)
            for node_type in NODE_TYPES
        }
        summary = [
            _masked_pool(node_embeddings[node_type], normalized.node_masks[node_type])
            for node_type in NODE_TYPES
        ]
        for relation in RELATION_TYPES:
            key = _relation_key(relation)
            source, target = relation
            mask = normalized.pairwise_mask[relation]
            # phi([h_i, h_j, e_ij]) whose first layer is affine, so the three
            # terms are projected separately and broadcast together instead of
            # building the [batch, sources, targets, 2*hidden+3] concatenation.
            projected_source = self.relation_source[key](node_embeddings[source])
            projected_target = self.relation_target[key](node_embeddings[target])
            projected_edge = self.relation_edge[key](normalized.pairwise[relation])
            pairs = F.relu(
                projected_source.unsqueeze(2)
                + projected_target.unsqueeze(1)
                + projected_edge
            )
            messages = F.relu(self.relation_output[key](pairs)) * mask.unsqueeze(-1)
            summary.append(
                _masked_pool(
                    messages.reshape(messages.shape[0], -1, messages.shape[-1]),
                    mask.reshape(mask.shape[0], -1),
                )
            )
        summary.append(self.global_encoder(normalized.global_features))
        return self.readout(torch.cat(summary, dim=-1))


class _TypedMessagePassingLayer(nn.Module):
    """One round of dense typed message passing over all nine relations.

    The per-edge message is affine in the source embedding and the edge
    features, and aggregation is a masked sum, so the sum is folded *through*
    the linear map instead of materializing a ``[batch, sources, targets,
    hidden]`` tensor.  The result is algebraically identical to scoring every
    edge and summing, but costs ``O(sources * targets)`` instead of
    ``O(sources * targets * hidden)``.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.messages = nn.ModuleDict(
            {
                _relation_key(relation): nn.Linear(
                    hidden_dim + EDGE_FEATURE_DIM, hidden_dim
                )
                for relation in RELATION_TYPES
            }
        )
        self.updates = nn.ModuleDict(
            {
                node_type: nn.Linear(2 * hidden_dim, hidden_dim)
                for node_type in NODE_TYPES
            }
        )
        self.norms = nn.ModuleDict(
            {node_type: nn.LayerNorm(hidden_dim) for node_type in NODE_TYPES}
        )

    def forward(
        self,
        node_embeddings: dict[str, torch.Tensor],
        pairwise: dict[tuple[str, str], torch.Tensor],
        pairwise_mask: dict[tuple[str, str], torch.Tensor],
        node_masks: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        aggregated = {
            node_type: torch.zeros_like(values)
            for node_type, values in node_embeddings.items()
        }
        counts = {
            node_type: torch.zeros(
                values.shape[0],
                values.shape[1],
                1,
                device=values.device,
                dtype=values.dtype,
            )
            for node_type, values in node_embeddings.items()
        }
        for relation in RELATION_TYPES:
            source, target = relation
            source_embed = node_embeddings[source]
            mask = pairwise_mask[relation]
            linear = self.messages[_relation_key(relation)]
            weight = linear.weight
            source_weight = weight[:, : source_embed.shape[-1]]
            edge_weight = weight[:, source_embed.shape[-1] :]

            counted = mask.sum(dim=1).unsqueeze(-1).to(source_embed.dtype)
            # sum_i mask_ij * h_i  and  sum_i mask_ij * e_ij, then map once.
            pooled_source = torch.einsum(
                "bij,bih->bjh", mask.to(source_embed.dtype), source_embed
            )
            pooled_edge = (pairwise[relation] * mask.unsqueeze(-1)).sum(dim=1)
            summed = (
                pooled_source @ source_weight.t()
                + pooled_edge @ edge_weight.t()
                + linear.bias * counted
            )
            aggregated[target] = aggregated[target] + summed
            counts[target] = counts[target] + counted

        updated: dict[str, torch.Tensor] = {}
        for node_type, values in node_embeddings.items():
            neighborhood = aggregated[node_type] / counts[node_type].clamp_min(1.0)
            residual = F.relu(
                self.updates[node_type](torch.cat((values, neighborhood), dim=-1))
            )
            updated[node_type] = self.norms[node_type](values + residual) * node_masks[
                node_type
            ].unsqueeze(-1)
        return updated


class HeteroGraphStateEncoder(_BaseStateEncoder):
    """Typed message passing that uses the canonical pairwise values as edges."""

    encoder_type = "hetero_graph"

    def __init__(
        self,
        shape: CanonicalShapeSpec,
        hidden_dim: int,
        output_dim: int,
        *,
        num_layers: int = 2,
    ):
        super().__init__(shape, hidden_dim, output_dim)
        if (
            isinstance(num_layers, bool)
            or not isinstance(num_layers, int)
            or num_layers <= 0
        ):
            raise ValueError("num_layers must be a positive integer")
        self.input_projections = nn.ModuleDict(
            {
                node_type: nn.Linear(NODE_FEATURE_DIMS[node_type], hidden_dim)
                for node_type in NODE_TYPES
            }
        )
        self.layers = nn.ModuleList(
            _TypedMessagePassingLayer(hidden_dim) for _ in range(num_layers)
        )
        self.global_encoder = _mlp(GLOBAL_FEATURE_DIM, hidden_dim, hidden_dim)
        summary_dim = 2 * hidden_dim * len(NODE_TYPES) + hidden_dim
        self.readout = nn.Sequential(
            nn.Linear(summary_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, tensors: CanonicalTensors) -> torch.Tensor:
        normalized = self.normalizer(tensors)
        node_embeddings = {
            node_type: F.relu(
                self.input_projections[node_type](normalized.nodes[node_type])
            )
            * normalized.node_masks[node_type].unsqueeze(-1)
            for node_type in NODE_TYPES
        }
        for layer in self.layers:
            node_embeddings = layer(
                node_embeddings,
                normalized.pairwise,
                normalized.pairwise_mask,
                normalized.node_masks,
            )
        summary = [
            _masked_pool(node_embeddings[node_type], normalized.node_masks[node_type])
            for node_type in NODE_TYPES
        ]
        summary.append(self.global_encoder(normalized.global_features))
        return self.readout(torch.cat(summary, dim=-1))


def _relation_key(relation: tuple[str, str]) -> str:
    return f"{relation[0]}__{relation[1]}"


def build_state_encoder(
    encoder_type: str,
    shape: CanonicalShapeSpec,
    hidden_dim: int,
    output_dim: int,
    *,
    num_layers: int = 2,
) -> _BaseStateEncoder:
    """Construct one of the three canonical state encoders."""
    if not isinstance(encoder_type, str):
        raise TypeError("encoder_type must be a string")
    normalized = encoder_type.strip().lower().replace("-", "_")
    builders = {
        "flat": FlatStateEncoder,
        "deep_sets": DeepSetsStateEncoder,
        "hetero_graph": HeteroGraphStateEncoder,
    }
    if normalized not in builders:
        raise ValueError(
            f"unknown state encoder {encoder_type!r}; expected one of "
            f"{STATE_ENCODER_TYPES}"
        )
    return builders[normalized](shape, hidden_dim, output_dim, num_layers=num_layers)
