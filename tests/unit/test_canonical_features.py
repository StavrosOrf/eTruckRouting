"""Semantic parity tests for canonical joint-fleet features."""

import os
from copy import deepcopy

import numpy as np
import pytest


os.environ.setdefault("MPLCONFIGDIR", "/tmp/evrp_matplotlib")

from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.state.features import (
    ACTION_FEATURES,
    CUSTOMER_FEATURES,
    SCHEMA_VERSION,
    TRUCK_FEATURES,
)
from EVRoutingEnv.state.representations import (
    CanonicalShapeSpec,
    canonical_flat_observation,
    pad_canonical_features,
)
from EVRoutingEnv.utils.utils import load_config


def _config() -> dict:
    config = deepcopy(load_config("EVRoutingEnv/config_files/config_vrp.yaml"))
    config["environment"].update(
        {"num_trucks": 2, "num_stops": 4, "allow_variable_num_stops": False}
    )
    config["truck"]["battery_capacity"] = 10_000.0
    config["problem"] = {
        "mode": "joint_fleet",
        "payload_capacity": 10.0,
        "min_customer_demand": 1.0,
        "max_customer_demand": 3.0,
        "base_service_time": 0.1,
    }
    return config


@pytest.fixture
def env():
    instance = EventDrivenTruckEnv(
        _config(),
        verbose=False,
        enable_plotting=False,
    )
    instance.reset(seed=4242)
    try:
        yield instance
    finally:
        instance.close()


def test_canonical_shapes_are_finite_and_versioned(env) -> None:
    features = env.get_canonical_features()
    features.validate()

    assert features.schema_version == SCHEMA_VERSION
    assert features.truck_features.shape == (2, len(TRUCK_FEATURES))
    assert features.customer_features.shape == (4, len(CUSTOMER_FEATURES))
    assert features.charger_features.shape[0] == env.num_charging_nodes
    assert features.action_features.shape == (
        env.action_space.n,
        len(ACTION_FEATURES),
    )
    assert np.isfinite(features.global_features).all()


def test_customer_rows_match_registry_semantics(env) -> None:
    features = env.get_canonical_features()
    node_column = CUSTOMER_FEATURES.index("node_id")
    demand_column = CUSTOMER_FEATURES.index("demand")
    status_column = CUSTOMER_FEATURES.index("status_code")

    for row, task in zip(
        features.customer_features,
        env.task_registry.tasks(),
        strict=True,
    ):
        assert int(row[node_column]) == task.node_id
        assert row[demand_column] == pytest.approx(task.demand)
        assert row[status_column] == 0.0


def test_action_feasibility_column_equals_environment_mask(env) -> None:
    features = env.get_canonical_features()
    feasible_column = ACTION_FEATURES.index("feasible")

    np.testing.assert_array_equal(
        features.action_features[:, feasible_column].astype(bool),
        env.mask_fn(),
    )


def test_claimed_task_update_is_visible_to_all_encoders(env) -> None:
    task = env.task_registry.tasks()[0]
    env.step((task.node_id, 0.0, False))
    features = env.get_canonical_features()
    task_id_column = CUSTOMER_FEATURES.index("task_id")
    status_column = CUSTOMER_FEATURES.index("status_code")
    feasible_column = ACTION_FEATURES.index("feasible")
    target_column = ACTION_FEATURES.index("target_node")

    row = features.customer_features[
        features.customer_features[:, task_id_column] == task.task_id
    ][0]
    assert row[status_column] == 1.0
    action_rows = features.action_features[
        features.action_features[:, target_column] == task.node_id
    ]
    assert len(action_rows) == 1
    assert action_rows[0, feasible_column] == 0.0


def test_active_truck_flag_is_unique(env) -> None:
    features = env.get_canonical_features()
    active_column = TRUCK_FEATURES.index("is_active")
    assert features.truck_features[:, active_column].sum() == 1.0


def test_primary_joint_observation_is_exact_canonical_flat_view(env) -> None:
    expected = canonical_flat_observation(
        env.get_canonical_features(),
        env.canonical_shape,
    )

    np.testing.assert_array_equal(env._get_observation(), expected)
    assert env.observation_mode == "canonical_flat"
    assert expected.shape == env.observation_space.shape
    assert env.observation_space.contains(expected)


def test_padded_sets_are_lossless_and_masks_distinguish_padding(env) -> None:
    canonical = env.get_canonical_features()
    shape = CanonicalShapeSpec(
        max_trucks=len(env.trucks) + 1,
        max_customers=len(env.task_registry) + 2,
        max_chargers=len(env.charging_nodes) + 1,
        max_actions=env.action_space.n + 3,
    )
    padded = pad_canonical_features(canonical, shape)

    np.testing.assert_array_equal(
        padded.truck_features[padded.truck_mask], canonical.truck_features
    )
    np.testing.assert_array_equal(
        padded.customer_features[padded.customer_mask],
        canonical.customer_features,
    )
    np.testing.assert_array_equal(
        padded.charger_features[padded.charger_mask],
        canonical.charger_features,
    )
    np.testing.assert_array_equal(
        padded.action_features[padded.action_mask], canonical.action_features
    )
    assert not padded.truck_mask[-1]
    assert not padded.customer_mask[-1]
    assert not padded.charger_mask[-1]
    assert not padded.action_mask[-1]
    assert padded.flatten().shape == (shape.flat_size,)


def test_padding_rejects_any_shape_that_would_drop_entities(env) -> None:
    canonical = env.get_canonical_features()
    too_small = CanonicalShapeSpec(
        max_trucks=len(env.trucks),
        max_customers=len(env.task_registry) - 1,
        max_chargers=len(env.charging_nodes),
        max_actions=env.action_space.n,
    )

    with pytest.raises(ValueError, match="customers count"):
        pad_canonical_features(canonical, too_small)


def test_graph_view_preserves_all_semantic_rows_and_has_complete_relations(env) -> None:
    canonical = env.get_canonical_features()
    graph = env.get_canonical_graph()

    np.testing.assert_array_equal(
        graph.node_features["truck"], canonical.truck_features
    )
    np.testing.assert_array_equal(
        graph.node_features["customer"], canonical.customer_features
    )
    np.testing.assert_array_equal(
        graph.node_features["charger"], canonical.charger_features
    )
    np.testing.assert_array_equal(graph.action_features, canonical.action_features)
    np.testing.assert_array_equal(graph.global_features, canonical.global_features)

    node_counts = {
        node_type: len(values)
        for node_type, values in graph.node_features.items()
    }
    assert len(graph.edge_indices) == 9
    for (source, target), edge_index in graph.edge_indices.items():
        assert edge_index.shape == (
            2,
            node_counts[source] * node_counts[target],
        )
        assert graph.edge_features[(source, target)].shape[0] == edge_index.shape[1]
        assert np.isfinite(graph.edge_features[(source, target)]).all()


@pytest.mark.parametrize("observation_mode", ["unknown", "canonical_graph"])
def test_unknown_observation_modes_are_rejected(observation_mode) -> None:
    config = _config()
    config["environment"]["observation_mode"] = observation_mode

    with pytest.raises(ValueError, match="observation_mode"):
        EventDrivenTruckEnv(config, verbose=False, enable_plotting=False)
