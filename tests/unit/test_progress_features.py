"""Charging connectivity must distinguish useful travel from a short cycle."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from algo.canonical_policy import CanonicalActorCritic, CanonicalPolicyConfig
from algo.canonical_state import CanonicalTensors
from algo.progress_features import ProgressActionFeatures
from EVRoutingEnv.state.features import (
    ACTION_FEATURES,
    CHARGER_FEATURES,
    CUSTOMER_FEATURES,
    EDGE_FEATURES,
    GLOBAL_FEATURES,
    RELATION_TYPES,
    SCHEMA_VERSION,
    TRUCK_FEATURES,
    CanonicalFleetFeatures,
)
from EVRoutingEnv.state.representations import pad_canonical_features


IDS = (10, 20, 30)
HOPS = ((0, 1, 2), (1, 0, 1), (2, 1, 0))


def snapshot():
    names = {
        "truck": TRUCK_FEATURES,
        "customer": CUSTOMER_FEATURES,
        "charger": CHARGER_FEATURES,
    }
    counts = {"truck": 1, "customer": 2, "charger": 3}
    nodes = {key: torch.zeros(1, counts[key], len(names[key])) for key in counts}
    for name, value in {
        "is_active": 1,
        "current_node": 10,
        "battery_kwh": 10,
        "battery_capacity_kwh": 10,
        "remaining_payload": 20,
    }.items():
        nodes["truck"][..., TRUCK_FEATURES.index(name)] = value
    nodes["customer"][0, :, CUSTOMER_FEATURES.index("node_id")] = torch.tensor(
        [100, 200]
    )
    nodes["customer"][..., CUSTOMER_FEATURES.index("demand")] = 2
    nodes["customer"][..., CUSTOMER_FEATURES.index("preassigned_to")] = -1
    nodes["customer"][0, 1, CUSTOMER_FEATURES.index("status_code")] = 3
    nodes["customer"][..., CUSTOMER_FEATURES.index("depot_energy_kwh")] = 22
    nodes["customer"][..., CUSTOMER_FEATURES.index("depot_reachable")] = 1
    nodes["charger"][0, :, CHARGER_FEATURES.index("node_id")] = torch.tensor(IDS)
    nodes["charger"][..., CHARGER_FEATURES.index("available")] = 1
    nodes["charger"][..., CHARGER_FEATURES.index("depot_energy_kwh")] = 20
    nodes["charger"][..., CHARGER_FEATURES.index("depot_reachable")] = 1
    pairwise = {
        relation: torch.zeros(
            1, counts[relation[0]], counts[relation[1]], len(EDGE_FEATURES)
        )
        for relation in RELATION_TYPES
    }
    for source, target, energy in (
        ("charger", "customer", [[15, 1], [10, 1], [4, 1]]),
        ("customer", "charger", [[13, 8, 2], [1, 1, 1]]),
        ("truck", "charger", [[0, 5, 11]]),
        ("truck", "customer", [[15, 1]]),
    ):
        pairwise[(source, target)][0, ..., 0] = torch.tensor(energy)
        pairwise[(source, target)][..., 1] = pairwise[(source, target)][..., 0] / 10
        pairwise[(source, target)][..., 2] = 1
    actions = torch.zeros(1, 6, len(ACTION_FEATURES))
    actions[0, :, ACTION_FEATURES.index("target_node")] = torch.tensor(
        [10, 20, 30, 100, 200, 999]
    )
    actions[0, :, ACTION_FEATURES.index("kind_code")] = torch.tensor([1, 1, 1, 0, 0, 2])
    actions[0, 1, ACTION_FEATURES.index("feasible")] = 1
    global_features = torch.zeros(1, len(GLOBAL_FEATURES))
    global_features[..., GLOBAL_FEATURES.index("max_time")] = 200
    return CanonicalTensors(
        nodes=nodes,
        node_masks={
            key: torch.ones(1, count, dtype=torch.bool) for key, count in counts.items()
        },
        action=actions,
        action_padding_mask=torch.ones(1, 6, dtype=torch.bool),
        global_features=global_features,
        pairwise=pairwise,
        pairwise_mask={
            relation: torch.ones(
                1, counts[relation[0]], counts[relation[1]], dtype=torch.bool
            )
            for relation in RELATION_TYPES
        },
    )


def test_station_chain_points_to_remaining_customer():
    features = ProgressActionFeatures(IDS, HOPS, 1)(snapshot())
    torch.testing.assert_close(features[0, :3, 0], torch.tensor([0.0, 0.0, 1.0]))
    torch.testing.assert_close(features[0, :3, 1], torch.tensor([0.5, 0.25, 0.0]))
    assert features[0, 1, 5] > features[0, 0, 5], (
        "moving toward work must improve on returning to the same loop"
    )
    assert torch.isfinite(features).all()


def test_features_follow_identity_when_station_rows_are_permuted():
    original = snapshot()
    order = torch.tensor([2, 0, 1])
    nodes = {**original.nodes, "charger": original.nodes["charger"][:, order]}
    pairwise = {}
    for (source, target), values in original.pairwise.items():
        if source == "charger":
            values = values[:, order]
        if target == "charger":
            values = values[:, :, order]
        pairwise[(source, target)] = values
    changed = replace(original, nodes=nodes, pairwise=pairwise)
    module = ProgressActionFeatures(IDS, HOPS, 1)
    torch.testing.assert_close(module(changed), module(original))


def test_unknown_station_map_is_rejected():
    data = snapshot()
    data.nodes["charger"][0, 0, CHARGER_FEATURES.index("node_id")] = 9999
    with pytest.raises(ValueError, match="map does not match"):
        ProgressActionFeatures(IDS, HOPS, 1)(data)


def test_progress_policy_has_finite_gradients_and_round_trips(tmp_path):
    data = snapshot()
    config = CanonicalPolicyConfig(
        max_trucks=1,
        max_customers=2,
        max_chargers=3,
        max_actions=6,
        state_encoder="hetero_graph",
        action_head="complete_gcn",
        hidden_dim=16,
        encoder_output_dim=16,
        normalize_action_features=True,
        progress_action_features=True,
        progress_station_ids=IDS,
        progress_station_hops=HOPS,
    )
    features = CanonicalFleetFeatures(
        schema_version=SCHEMA_VERSION,
        truck_features=data.nodes["truck"][0].numpy(),
        customer_features=data.nodes["customer"][0].numpy(),
        charger_features=data.nodes["charger"][0].numpy(),
        action_features=data.action[0].numpy(),
        global_features=data.global_features[0].numpy(),
        pairwise_features={
            key: value[0].numpy() for key, value in data.pairwise.items()
        },
    )
    observation = torch.from_numpy(
        np.stack([pad_canonical_features(features, config.shape).flatten()])
    )
    policy = CanonicalActorCritic(config)
    policy.observe(observation)
    result = policy(observation)
    assert result.logits.argmax(-1).item() == 1
    (result.values.sum() + result.distribution().entropy().sum()).backward()
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in policy.parameters()
        if parameter.grad is not None
    )
    policy.save(tmp_path)
    restored = CanonicalActorCritic.load(tmp_path)
    torch.testing.assert_close(restored(observation).logits, result.logits)
    torch.testing.assert_close(restored.predict_values(observation), result.values)
