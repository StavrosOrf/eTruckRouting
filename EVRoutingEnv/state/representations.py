"""Lossless representations derived from canonical joint-fleet features."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral

import numpy as np

from EVRoutingEnv.state.features import (
    ACTION_FEATURES,
    CHARGER_FEATURES,
    CUSTOMER_FEATURES,
    GLOBAL_FEATURES,
    SCHEMA_VERSION,
    TRUCK_FEATURES,
    CanonicalFleetFeatures,
)


EDGE_FEATURES = ("nominal_energy_kwh", "nominal_travel_hours", "reachable")


@dataclass(frozen=True)
class CanonicalShapeSpec:
    """Static padding limits required by vectorized learning libraries."""

    max_trucks: int
    max_customers: int
    max_chargers: int
    max_actions: int

    def __post_init__(self) -> None:
        for name, value in (
            ("max_trucks", self.max_trucks),
            ("max_customers", self.max_customers),
            ("max_chargers", self.max_chargers),
            ("max_actions", self.max_actions),
        ):
            if (
                not isinstance(value, Integral)
                or isinstance(value, bool)
                or value < 0
            ):
                raise ValueError(f"{name} must be a nonnegative integer")
            object.__setattr__(self, name, int(value))

    @property
    def flat_size(self) -> int:
        """Number of scalars in the canonical padded flat observation."""
        feature_values = (
            self.max_trucks * len(TRUCK_FEATURES)
            + self.max_customers * len(CUSTOMER_FEATURES)
            + self.max_chargers * len(CHARGER_FEATURES)
            + self.max_actions * len(ACTION_FEATURES)
        )
        validity_masks = (
            self.max_trucks
            + self.max_customers
            + self.max_chargers
            + self.max_actions
        )
        return feature_values + validity_masks + len(GLOBAL_FEATURES)


@dataclass(frozen=True)
class PaddedCanonicalFeatures:
    """Typed fixed-shape sets with explicit padding masks."""

    schema_version: str
    truck_features: np.ndarray
    truck_mask: np.ndarray
    customer_features: np.ndarray
    customer_mask: np.ndarray
    charger_features: np.ndarray
    charger_mask: np.ndarray
    action_features: np.ndarray
    action_mask: np.ndarray
    global_features: np.ndarray

    def validate(self, shape: CanonicalShapeSpec) -> None:
        expected = (
            (
                self.truck_features,
                (shape.max_trucks, len(TRUCK_FEATURES)),
                "truck_features",
            ),
            (self.truck_mask, (shape.max_trucks,), "truck_mask"),
            (
                self.customer_features,
                (shape.max_customers, len(CUSTOMER_FEATURES)),
                "customer_features",
            ),
            (self.customer_mask, (shape.max_customers,), "customer_mask"),
            (
                self.charger_features,
                (shape.max_chargers, len(CHARGER_FEATURES)),
                "charger_features",
            ),
            (self.charger_mask, (shape.max_chargers,), "charger_mask"),
            (
                self.action_features,
                (shape.max_actions, len(ACTION_FEATURES)),
                "action_features",
            ),
            (self.action_mask, (shape.max_actions,), "action_mask"),
            (self.global_features, (len(GLOBAL_FEATURES),), "global_features"),
        )
        for values, expected_shape, label in expected:
            if values.shape != expected_shape:
                raise ValueError(
                    f"{label} shape {values.shape} does not match {expected_shape}"
                )
            if not np.isfinite(values).all():
                raise ValueError(f"{label} contains non-finite values")
        for label, mask in (
            ("truck_mask", self.truck_mask),
            ("customer_mask", self.customer_mask),
            ("charger_mask", self.charger_mask),
            ("action_mask", self.action_mask),
        ):
            if mask.dtype != np.bool_:
                raise ValueError(f"{label} must have boolean dtype")

    def flatten(self) -> np.ndarray:
        """Return one deterministic vector without dropping type masks."""
        return np.concatenate(
            (
                self.truck_features.ravel(),
                self.customer_features.ravel(),
                self.charger_features.ravel(),
                self.action_features.ravel(),
                self.truck_mask.astype(np.float32),
                self.customer_mask.astype(np.float32),
                self.charger_mask.astype(np.float32),
                self.action_mask.astype(np.float32),
                self.global_features,
            )
        ).astype(np.float32, copy=False)


@dataclass(frozen=True)
class CanonicalGraphFeatures:
    """Heterogeneous graph view that preserves every canonical node row."""

    schema_version: str
    node_features: dict[str, np.ndarray]
    edge_indices: dict[tuple[str, str], np.ndarray]
    edge_features: dict[tuple[str, str], np.ndarray]
    action_features: np.ndarray
    global_features: np.ndarray

    def validate(self) -> None:
        expected_widths = {
            "truck": len(TRUCK_FEATURES),
            "customer": len(CUSTOMER_FEATURES),
            "charger": len(CHARGER_FEATURES),
        }
        if set(self.node_features) != set(expected_widths):
            raise ValueError("graph must contain truck, customer, and charger nodes")
        for node_type, width in expected_widths.items():
            values = self.node_features[node_type]
            if values.ndim != 2 or values.shape[1] != width:
                raise ValueError(f"invalid {node_type} node feature shape")
            if not np.isfinite(values).all():
                raise ValueError(f"{node_type} node features contain non-finite values")
        if set(self.edge_indices) != set(self.edge_features):
            raise ValueError("edge index and feature relations do not match")
        for relation, edge_index in self.edge_indices.items():
            values = self.edge_features[relation]
            if edge_index.ndim != 2 or edge_index.shape[0] != 2:
                raise ValueError(f"invalid edge index for relation {relation}")
            if values.shape != (edge_index.shape[1], len(EDGE_FEATURES)):
                raise ValueError(f"invalid edge features for relation {relation}")
            if not np.isfinite(values).all():
                raise ValueError(f"edge features contain non-finite values: {relation}")
        if self.action_features.ndim != 2 or self.action_features.shape[1] != len(
            ACTION_FEATURES
        ):
            raise ValueError("invalid graph action feature shape")
        if self.global_features.shape != (len(GLOBAL_FEATURES),):
            raise ValueError("invalid graph global feature shape")


def pad_canonical_features(
    features: CanonicalFleetFeatures,
    shape: CanonicalShapeSpec,
) -> PaddedCanonicalFeatures:
    """Pad canonical sets to fixed limits, rejecting information loss."""
    features.validate()
    if features.schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported feature schema {features.schema_version!r}; "
            f"expected {SCHEMA_VERSION!r}"
        )

    trucks, truck_mask = _pad_rows(
        features.truck_features, shape.max_trucks, "trucks"
    )
    customers, customer_mask = _pad_rows(
        features.customer_features, shape.max_customers, "customers"
    )
    chargers, charger_mask = _pad_rows(
        features.charger_features, shape.max_chargers, "chargers"
    )
    actions, action_mask = _pad_rows(
        features.action_features, shape.max_actions, "actions"
    )
    result = PaddedCanonicalFeatures(
        schema_version=features.schema_version,
        truck_features=trucks,
        truck_mask=truck_mask,
        customer_features=customers,
        customer_mask=customer_mask,
        charger_features=chargers,
        charger_mask=charger_mask,
        action_features=actions,
        action_mask=action_mask,
        global_features=features.global_features.astype(np.float32, copy=True),
    )
    result.validate(shape)
    return result


def canonical_flat_observation(
    features: CanonicalFleetFeatures,
    shape: CanonicalShapeSpec,
) -> np.ndarray:
    """Create the fair fixed-size flat baseline observation."""
    observation = pad_canonical_features(features, shape).flatten()
    if observation.shape != (shape.flat_size,):
        raise RuntimeError("canonical flat observation size mismatch")
    return observation


def canonical_graph_observation(env) -> CanonicalGraphFeatures:
    """Create a complete typed state graph from the canonical snapshot."""
    canonical = env.get_canonical_features()
    node_features = {
        "truck": canonical.truck_features.copy(),
        "customer": canonical.customer_features.copy(),
        "charger": canonical.charger_features.copy(),
    }
    node_ids = {
        "truck": canonical.truck_features[
            :, TRUCK_FEATURES.index("current_node")
        ].astype(np.int64),
        "customer": canonical.customer_features[
            :, CUSTOMER_FEATURES.index("node_id")
        ].astype(np.int64),
        "charger": canonical.charger_features[
            :, CHARGER_FEATURES.index("node_id")
        ].astype(np.int64),
    }
    edge_indices: dict[tuple[str, str], np.ndarray] = {}
    edge_features: dict[tuple[str, str], np.ndarray] = {}
    for source_type, source_nodes in node_ids.items():
        for target_type, target_nodes in node_ids.items():
            relation = (source_type, target_type)
            index, values = _complete_relation(
                source_nodes,
                target_nodes,
                env.transport_graph,
            )
            edge_indices[relation] = index
            edge_features[relation] = values

    result = CanonicalGraphFeatures(
        schema_version=canonical.schema_version,
        node_features=node_features,
        edge_indices=edge_indices,
        edge_features=edge_features,
        action_features=canonical.action_features.copy(),
        global_features=canonical.global_features.copy(),
    )
    result.validate()
    return result


def _pad_rows(
    values: np.ndarray,
    maximum: int,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    count, width = values.shape
    if count > maximum:
        raise ValueError(
            f"{label} count {count} exceeds configured maximum {maximum}"
        )
    padded = np.zeros((maximum, width), dtype=np.float32)
    mask = np.zeros(maximum, dtype=bool)
    if count:
        padded[:count] = values
        mask[:count] = True
    return padded, mask


def _complete_relation(
    source_nodes: np.ndarray,
    target_nodes: np.ndarray,
    transport_graph,
) -> tuple[np.ndarray, np.ndarray]:
    edge_count = len(source_nodes) * len(target_nodes)
    index = np.empty((2, edge_count), dtype=np.int64)
    values = np.zeros((edge_count, len(EDGE_FEATURES)), dtype=np.float32)
    offset = 0
    for source_index, source_node in enumerate(source_nodes):
        for target_index, target_node in enumerate(target_nodes):
            index[:, offset] = (source_index, target_index)
            energy = _path_value(
                transport_graph.get_path_energy,
                int(source_node),
                int(target_node),
            )
            travel_time = _path_value(
                transport_graph.get_time_distance,
                int(source_node),
                int(target_node),
            )
            reachable = math.isfinite(energy) and math.isfinite(travel_time)
            if reachable:
                values[offset] = (energy, travel_time, 1.0)
            offset += 1
    return index, values


def _path_value(function, source: int, target: int) -> float:
    try:
        value = float(function(source, target))
    except (KeyError, TypeError, ValueError):
        return math.inf
    return value if math.isfinite(value) and value >= 0.0 else math.inf
