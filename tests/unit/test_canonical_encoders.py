"""Invariance, masking, and equivalence checks for the canonical encoders."""

import os

import pytest
import torch


os.environ.setdefault("MPLCONFIGDIR", "/tmp/evrp_matplotlib")

from algo.canonical_encoders import (
    STATE_ENCODER_TYPES,
    CanonicalNormalizer,
    _TypedMessagePassingLayer,
    build_state_encoder,
)
from algo.canonical_state import unpack_flat_observation
from EVRoutingEnv.state.features import NODE_TYPES, RELATION_TYPES
from EVRoutingEnv.state.representations import CanonicalShapeSpec


SHAPE = CanonicalShapeSpec(max_trucks=2, max_customers=4, max_chargers=3, max_actions=8)


def _tensors(batch: int = 3, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    observation = torch.rand(
        batch, SHAPE.flat_size, generator=generator, dtype=torch.float32
    )
    tensors = unpack_flat_observation(observation, SHAPE)
    # Force a realistic padding pattern: the last charger row is padding.
    masks = dict(tensors.node_masks)
    for node_type in NODE_TYPES:
        masks[node_type] = torch.ones_like(masks[node_type], dtype=torch.bool)
    masks["charger"][:, -1] = False
    pairwise_mask = {
        relation: masks[relation[0]].unsqueeze(2) & masks[relation[1]].unsqueeze(1)
        for relation in RELATION_TYPES
    }
    nodes = {
        node_type: tensors.nodes[node_type] * masks[node_type].unsqueeze(-1)
        for node_type in NODE_TYPES
    }
    pairwise = {
        relation: tensors.pairwise[relation] * pairwise_mask[relation].unsqueeze(-1)
        for relation in RELATION_TYPES
    }
    return type(tensors)(
        nodes=nodes,
        node_masks=masks,
        action=tensors.action,
        action_padding_mask=torch.ones_like(tensors.action_padding_mask),
        pairwise=pairwise,
        pairwise_mask=pairwise_mask,
        global_features=tensors.global_features,
    )


def _permute(tensors, node_type: str, order: torch.Tensor):
    nodes = dict(tensors.nodes)
    masks = dict(tensors.node_masks)
    nodes[node_type] = nodes[node_type][:, order]
    masks[node_type] = masks[node_type][:, order]
    pairwise = dict(tensors.pairwise)
    pairwise_mask = dict(tensors.pairwise_mask)
    for relation in RELATION_TYPES:
        source, target = relation
        values = pairwise[relation]
        mask = pairwise_mask[relation]
        if source == node_type:
            values = values[:, order]
            mask = mask[:, order]
        if target == node_type:
            values = values[:, :, order]
            mask = mask[:, :, order]
        pairwise[relation] = values
        pairwise_mask[relation] = mask
    return type(tensors)(
        nodes=nodes,
        node_masks=masks,
        action=tensors.action,
        action_padding_mask=tensors.action_padding_mask,
        pairwise=pairwise,
        pairwise_mask=pairwise_mask,
        global_features=tensors.global_features,
    )


@pytest.mark.parametrize("encoder_type", STATE_ENCODER_TYPES)
def test_encoders_emit_the_requested_width_and_finite_values(encoder_type) -> None:
    encoder = build_state_encoder(encoder_type, SHAPE, 16, 24)
    tensors = _tensors()
    encoder.observe(tensors)
    embedding = encoder(tensors)

    assert embedding.shape == (3, 24)
    assert torch.isfinite(embedding).all()


@pytest.mark.parametrize("encoder_type", ["deep_sets", "hetero_graph", "attention"])
@pytest.mark.parametrize("node_type", NODE_TYPES)
def test_set_and_graph_encoders_are_permutation_invariant(
    encoder_type, node_type
) -> None:
    encoder = build_state_encoder(encoder_type, SHAPE, 16, 16).eval()
    tensors = _tensors(seed=4)
    count = tensors.nodes[node_type].shape[1]
    if count < 2:
        pytest.skip("set too small to permute")
    order = torch.randperm(count, generator=torch.Generator().manual_seed(1))

    with torch.no_grad():
        baseline = encoder(tensors)
        permuted = encoder(_permute(tensors, node_type, order))
    torch.testing.assert_close(baseline, permuted, rtol=1e-4, atol=1e-5)


def test_folded_message_passing_equals_dense_edge_scoring() -> None:
    """The fast aggregation must be exactly the slow one, not an approximation."""
    hidden = 12
    layer = _TypedMessagePassingLayer(hidden).eval()
    tensors = _tensors(batch=2, seed=7)
    embeddings = {
        node_type: torch.randn(2, tensors.nodes[node_type].shape[1], hidden)
        * tensors.node_masks[node_type].unsqueeze(-1)
        for node_type in NODE_TYPES
    }

    with torch.no_grad():
        fast = layer(
            embeddings, tensors.pairwise, tensors.pairwise_mask, tensors.node_masks
        )

    # Reference: build every edge message explicitly and sum over sources.
    aggregated = {key: torch.zeros_like(value) for key, value in embeddings.items()}
    counts = {
        key: torch.zeros(value.shape[0], value.shape[1], 1)
        for key, value in embeddings.items()
    }
    with torch.no_grad():
        for relation in RELATION_TYPES:
            source, target = relation
            mask = tensors.pairwise_mask[relation]
            source_embed = embeddings[source]
            inputs = torch.cat(
                (
                    source_embed.unsqueeze(2).expand(-1, -1, mask.shape[2], -1),
                    tensors.pairwise[relation],
                ),
                dim=-1,
            )
            linear = layer.messages[f"{source}__{target}"]
            messages = linear(inputs) * mask.unsqueeze(-1)
            aggregated[target] = aggregated[target] + messages.sum(dim=1)
            counts[target] = counts[target] + mask.sum(dim=1).unsqueeze(-1).float()

        expected = {}
        for node_type, values in embeddings.items():
            neighborhood = aggregated[node_type] / counts[node_type].clamp_min(1.0)
            residual = torch.relu(
                layer.updates[node_type](torch.cat((values, neighborhood), dim=-1))
            )
            expected[node_type] = layer.norms[node_type](
                values + residual
            ) * tensors.node_masks[node_type].unsqueeze(-1)

    for node_type in NODE_TYPES:
        torch.testing.assert_close(
            fast[node_type], expected[node_type], rtol=1e-5, atol=1e-5
        )


@pytest.mark.parametrize("encoder_type", STATE_ENCODER_TYPES)
def test_padded_entities_cannot_change_the_embedding(encoder_type) -> None:
    encoder = build_state_encoder(encoder_type, SHAPE, 16, 16).eval()
    tensors = _tensors(seed=11)

    polluted_nodes = dict(tensors.nodes)
    polluted_nodes["charger"] = tensors.nodes["charger"].clone()
    polluted_nodes["charger"][:, -1] = 12345.0
    polluted = type(tensors)(
        nodes=polluted_nodes,
        node_masks=tensors.node_masks,
        action=tensors.action,
        action_padding_mask=tensors.action_padding_mask,
        pairwise=tensors.pairwise,
        pairwise_mask=tensors.pairwise_mask,
        global_features=tensors.global_features,
    )

    with torch.no_grad():
        baseline = encoder(tensors)
        contaminated = encoder(polluted)
    if encoder_type == "flat":
        # The flat baseline sees the raw vector, so it legitimately reads the
        # padded row; the mask channel is what tells it the row is padding.
        assert contaminated.shape == baseline.shape
    else:
        torch.testing.assert_close(baseline, contaminated, rtol=1e-4, atol=1e-5)


def test_normalizer_ignores_padding_when_accumulating_statistics() -> None:
    normalizer = CanonicalNormalizer()
    tensors = _tensors(batch=6, seed=13)
    for _ in range(4):
        normalizer.observe(tensors)

    normalized = normalizer(tensors)
    for node_type in NODE_TYPES:
        values = normalized.nodes[node_type]
        assert torch.isfinite(values).all()
        assert values.abs().max() <= 10.0 + 1e-6
        padded = ~tensors.node_masks[node_type]
        assert torch.equal(values[padded], torch.zeros_like(values[padded]))


def test_unknown_encoder_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="unknown state encoder"):
        build_state_encoder("transformer", SHAPE, 16, 16)
