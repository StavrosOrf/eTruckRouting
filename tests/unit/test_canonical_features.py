"""Semantic parity tests for canonical joint-fleet features."""

import os
from copy import deepcopy

import numpy as np
import pytest


os.environ.setdefault("MPLCONFIGDIR", "/tmp/evrp_matplotlib")

from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.state.features import (
    ACTION_FEATURES,
    CHARGER_FEATURES,
    CUSTOMER_FEATURES,
    EDGE_FEATURES,
    NODE_TYPES,
    RELATION_TYPES,
    SCHEMA_VERSION,
    TRUCK_FEATURES,
    extract_pairwise_relations,
)
from EVRoutingEnv.state.representations import (
    CanonicalShapeSpec,
    canonical_flat_observation,
    canonical_graph_features,
    pad_canonical_features,
    relation_matrix,
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
        node_type: len(values) for node_type, values in graph.node_features.items()
    }
    assert len(graph.edge_indices) == 9
    for (source, target), edge_index in graph.edge_indices.items():
        assert edge_index.shape == (
            2,
            node_counts[source] * node_counts[target],
        )
        assert graph.edge_features[(source, target)].shape[0] == edge_index.shape[1]
        assert np.isfinite(graph.edge_features[(source, target)]).all()


def _node_ids(canonical) -> dict:
    return {
        "truck": canonical.truck_features[
            :, TRUCK_FEATURES.index("current_node")
        ].astype(int),
        "customer": canonical.customer_features[
            :, CUSTOMER_FEATURES.index("node_id")
        ].astype(int),
        "charger": canonical.charger_features[
            :, CHARGER_FEATURES.index("node_id")
        ].astype(int),
    }


def test_pairwise_relations_are_canonical_and_complete(env) -> None:
    canonical = env.get_canonical_features()
    counts = canonical.node_counts()

    assert set(canonical.pairwise_features) == set(RELATION_TYPES)
    assert len(RELATION_TYPES) == 9
    for source, target in RELATION_TYPES:
        values = canonical.pairwise_features[(source, target)]
        assert values.shape == (counts[source], counts[target], len(EDGE_FEATURES))
        assert np.isfinite(values).all()


def test_pairwise_values_equal_the_transport_graph_for_reachable_pairs(env) -> None:
    canonical = env.get_canonical_features()
    node_ids = _node_ids(canonical)
    reachable_column = EDGE_FEATURES.index("reachable")

    for source, target in RELATION_TYPES:
        values = canonical.pairwise_features[(source, target)]
        for row, source_node in enumerate(node_ids[source]):
            for column, target_node in enumerate(node_ids[target]):
                if values[row, column, reachable_column] != 1.0:
                    continue
                assert values[row, column, 0] == pytest.approx(
                    env.transport_graph.get_path_energy(
                        int(source_node), int(target_node)
                    ),
                    rel=1e-6,
                )
                assert values[row, column, 1] == pytest.approx(
                    env.transport_graph.get_time_distance(
                        int(source_node), int(target_node)
                    ),
                    rel=1e-6,
                )


def test_identical_node_pairs_share_values_across_relations(env) -> None:
    canonical = env.get_canonical_features()
    node_ids = _node_ids(canonical)
    seen: dict[tuple[int, int], np.ndarray] = {}

    for source, target in RELATION_TYPES:
        values = canonical.pairwise_features[(source, target)]
        for row, source_node in enumerate(node_ids[source]):
            for column, target_node in enumerate(node_ids[target]):
                key = (int(source_node), int(target_node))
                if key in seen:
                    np.testing.assert_array_equal(seen[key], values[row, column])
                else:
                    seen[key] = values[row, column]


def test_flat_set_and_graph_expose_identical_pairwise_semantics(env) -> None:
    canonical = env.get_canonical_features()
    padded = env.get_canonical_sets()
    graph = env.get_canonical_graph()
    flat = env._get_observation()
    counts = canonical.node_counts()

    for relation in RELATION_TYPES:
        source, target = relation
        expected = canonical.pairwise_features[relation]
        np.testing.assert_array_equal(
            padded.pairwise_features[relation][: counts[source], : counts[target]],
            expected,
        )
        np.testing.assert_array_equal(relation_matrix(graph, relation), expected)

    # The flat vector is the padded view, so a matching padded flatten proves the
    # flat baseline observes exactly the same pairwise values.
    np.testing.assert_array_equal(flat, padded.flatten())


def test_padded_pairwise_masks_are_entity_mask_outer_products(env) -> None:
    canonical = env.get_canonical_features()
    shape = CanonicalShapeSpec(
        max_trucks=len(env.trucks) + 1,
        max_customers=len(env.task_registry) + 2,
        max_chargers=len(env.charging_nodes) + 1,
        max_actions=env.action_space.n + 3,
    )
    padded = pad_canonical_features(canonical, shape)
    masks = padded.entity_masks()

    for relation in RELATION_TYPES:
        source, target = relation
        mask = padded.pairwise_mask[relation]
        np.testing.assert_array_equal(mask, np.outer(masks[source], masks[target]))
        values = padded.pairwise_features[relation]
        assert np.isfinite(values).all()
        assert not values[~mask].any()
        np.testing.assert_array_equal(
            values[mask],
            canonical.pairwise_features[relation].reshape(-1, len(EDGE_FEATURES)),
        )


def test_flat_size_accounts_for_the_pairwise_block(env) -> None:
    shape = env.canonical_shape
    observation = canonical_flat_observation(env.get_canonical_features(), shape)

    limits = shape.node_limits
    cells = sum(limits[source] * limits[target] for source, target in RELATION_TYPES)
    assert shape.pairwise_size == cells * len(EDGE_FEATURES) + cells
    assert observation.shape == (shape.flat_size,)
    assert env.observation_space.shape == (shape.flat_size,)
    assert env.observation_space.contains(observation)


@pytest.mark.parametrize("permuted_type", NODE_TYPES)
def test_pairwise_extraction_is_source_and_target_permutation_covariant(
    env, permuted_type
) -> None:
    canonical = env.get_canonical_features()
    node_ids = _node_ids(canonical)
    order = np.arange(len(node_ids[permuted_type]))
    if len(order) < 2:
        pytest.skip(f"{permuted_type} set is too small to permute")
    order = order[::-1]

    permuted_ids = dict(node_ids)
    permuted_ids[permuted_type] = node_ids[permuted_type][order]
    permuted = extract_pairwise_relations(env.transport_graph, permuted_ids)
    baseline = extract_pairwise_relations(env.transport_graph, node_ids)

    for source, target in RELATION_TYPES:
        expected = baseline[(source, target)]
        if source == permuted_type:
            expected = expected[order]
        if target == permuted_type:
            expected = expected[:, order]
        np.testing.assert_array_equal(permuted[(source, target)], expected)


def test_graph_adapter_never_queries_the_transport_graph(env) -> None:
    canonical = env.get_canonical_features()
    graph = canonical_graph_features(canonical)

    for relation in RELATION_TYPES:
        np.testing.assert_array_equal(
            relation_matrix(graph, relation), canonical.pairwise_features[relation]
        )


def test_unreachable_pairs_are_finite_zero_and_flagged() -> None:
    class _BrokenGraph:
        def get_path_energy(self, source, target):
            if source != target:
                raise ValueError("no path")
            return 0.0

        def get_time_distance(self, source, target):
            if source != target:
                return float("inf")
            return 0.0

    relations = extract_pairwise_relations(
        _BrokenGraph(),
        {
            "truck": np.asarray([0, 1]),
            "customer": np.asarray([2]),
            "charger": np.asarray([3]),
        },
    )
    reachable_column = EDGE_FEATURES.index("reachable")
    for values in relations.values():
        assert np.isfinite(values).all()
    off_diagonal = relations[("truck", "customer")]
    assert off_diagonal[:, :, reachable_column].sum() == 0.0
    assert not off_diagonal.any()
    assert relations[("truck", "truck")][0, 0, reachable_column] == 1.0


def test_depot_is_observable_from_every_node_type(env) -> None:
    """The mandatory return destination must be visible before it is actionable.

    The depot is not a truck, customer, or charger, so it appears in none of the
    nine typed relations. Without these columns a policy cannot price the return
    leg until the last customer is served, which is far too late to reserve
    energy for it.
    """
    from EVRoutingEnv.state.features import CHARGER_FEATURES, DEPOT_FEATURES

    features = env.get_canonical_features()
    depot = int(env.joint_instance.depot_node)

    for names, rows, id_column in (
        (TRUCK_FEATURES, features.truck_features, "current_node"),
        (CUSTOMER_FEATURES, features.customer_features, "node_id"),
        (CHARGER_FEATURES, features.charger_features, "node_id"),
    ):
        for column in DEPOT_FEATURES:
            assert column in names
        energy_column = names.index("depot_energy_kwh")
        hours_column = names.index("depot_travel_hours")
        reach_column = names.index("depot_reachable")
        for row in rows:
            node = int(row[names.index(id_column)])
            reachable = row[reach_column] == 1.0
            assert np.isfinite(row[energy_column])
            assert np.isfinite(row[hours_column])
            if reachable:
                assert row[energy_column] == pytest.approx(
                    env.transport_graph.get_path_energy(node, depot), rel=1e-5
                )
                assert row[hours_column] == pytest.approx(
                    env.transport_graph.get_time_distance(node, depot), rel=1e-5
                )
            else:
                # Unreachable pairs are zeroed with the flag cleared, never inf.
                assert row[energy_column] == 0.0
                assert row[hours_column] == 0.0


def test_truck_rows_expose_return_energy_headroom(env) -> None:
    features = env.get_canonical_features()
    depot = int(env.joint_instance.depot_node)
    headroom_column = TRUCK_FEATURES.index("battery_minus_depot_energy")
    battery_column = TRUCK_FEATURES.index("battery_kwh")

    for row, truck in zip(
        features.truck_features,
        sorted(env.trucks, key=lambda item: item.truck_id),
        strict=True,
    ):
        expected = float(truck.current_battery) - env.transport_graph.get_path_energy(
            int(truck.current_node), depot
        )
        assert row[headroom_column] == pytest.approx(expected, rel=1e-5)
        assert row[battery_column] == pytest.approx(float(truck.current_battery))


def test_depot_action_reports_its_energy_even_while_customers_remain(env) -> None:
    """A rejected depot action must still price the return leg."""
    from EVRoutingEnv.state.feasibility import ActionKind, joint_action_feasibility

    decisions = joint_action_feasibility(env)
    depot_indices = [
        index
        for index, decision in enumerate(decisions)
        if decision.action_kind is ActionKind.DEPOT
    ]
    assert depot_indices

    energy_column = ACTION_FEATURES.index("required_energy")
    features = env.get_canonical_features()
    for index in depot_indices:
        decision = decisions[index]
        if decision.reason.value in {"customers_remain", "depot_return_not_required"}:
            # -1.0 is the sentinel for "no value"; the whole point of the fix is
            # that this rejection now carries a real number.
            assert features.action_features[index, energy_column] >= 0.0


@pytest.mark.parametrize("observation_mode", ["unknown", "canonical_graph"])
def test_unknown_observation_modes_are_rejected(observation_mode) -> None:
    config = _config()
    config["environment"]["observation_mode"] = observation_mode

    with pytest.raises(ValueError, match="observation_mode"):
        EventDrivenTruckEnv(config, verbose=False, enable_plotting=False)
