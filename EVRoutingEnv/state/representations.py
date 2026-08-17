"""Lossless representations derived from canonical joint-fleet features."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np

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


__all__ = [
    "EDGE_FEATURES",
    "RELATION_TYPES",
    "CanonicalGraphFeatures",
    "CanonicalShapeSpec",
    "PaddedCanonicalFeatures",
    "canonical_flat_observation",
    "canonical_graph_observation",
    "pad_canonical_features",
]


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
            if not isinstance(value, Integral) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
            object.__setattr__(self, name, int(value))

    @property
    def node_limits(self) -> dict[str, int]:
        """Return the padded row limit for each canonical node type."""
        return {
            "truck": self.max_trucks,
            "customer": self.max_customers,
            "charger": self.max_chargers,
        }

    @property
    def pairwise_size(self) -> int:
        """Number of scalars used by the padded pairwise relation block."""
        limits = self.node_limits
        cells = sum(
            limits[source] * limits[target] for source, target in RELATION_TYPES
        )
        return cells * len(EDGE_FEATURES) + cells

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
            self.max_trucks + self.max_customers + self.max_chargers + self.max_actions
        )
        return (
            feature_values + validity_masks + self.pairwise_size + len(GLOBAL_FEATURES)
        )


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
    pairwise_features: dict[tuple[str, str], np.ndarray]
    pairwise_mask: dict[tuple[str, str], np.ndarray]

    def entity_masks(self) -> dict[str, np.ndarray]:
        """Return the padding mask of each canonical node type."""
        return {
            "truck": self.truck_mask,
            "customer": self.customer_mask,
            "charger": self.charger_mask,
        }

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
        self._validate_pairwise(shape)

    def _validate_pairwise(self, shape: CanonicalShapeSpec) -> None:
        if set(self.pairwise_features) != set(RELATION_TYPES):
            raise ValueError("padded pairwise features must cover all nine relations")
        if set(self.pairwise_mask) != set(RELATION_TYPES):
            raise ValueError("padded pairwise masks must cover all nine relations")
        limits = shape.node_limits
        entity_masks = self.entity_masks()
        for relation in RELATION_TYPES:
            source, target = relation
            values = self.pairwise_features[relation]
            mask = self.pairwise_mask[relation]
            expected = (limits[source], limits[target], len(EDGE_FEATURES))
            if values.shape != expected:
                raise ValueError(
                    f"padded relation {relation} shape {values.shape} does not "
                    f"match {expected}"
                )
            if not np.isfinite(values).all():
                raise ValueError(f"padded relation {relation} has non-finite values")
            if mask.shape != expected[:2]:
                raise ValueError(f"padded relation mask {relation} has the wrong shape")
            if mask.dtype != np.bool_:
                raise ValueError(f"padded relation mask {relation} must be boolean")
            expected_mask = np.outer(entity_masks[source], entity_masks[target])
            if not np.array_equal(mask, expected_mask):
                raise ValueError(
                    f"padded relation mask {relation} must be the outer product of "
                    "its source and target entity masks"
                )
            if values[~mask].any():
                raise ValueError(
                    f"padded relation {relation} must be zero outside its mask"
                )

    def flatten(self) -> np.ndarray:
        """Return one deterministic vector without dropping type or pair masks."""
        blocks = [
            self.truck_features.ravel(),
            self.customer_features.ravel(),
            self.charger_features.ravel(),
            self.action_features.ravel(),
            self.truck_mask.astype(np.float32),
            self.customer_mask.astype(np.float32),
            self.charger_mask.astype(np.float32),
            self.action_mask.astype(np.float32),
        ]
        blocks.extend(
            self.pairwise_features[relation].ravel() for relation in RELATION_TYPES
        )
        blocks.extend(
            self.pairwise_mask[relation].ravel().astype(np.float32)
            for relation in RELATION_TYPES
        )
        blocks.append(self.global_features)
        return np.concatenate(blocks).astype(np.float32, copy=False)


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
        if set(self.edge_indices) != set(RELATION_TYPES):
            raise ValueError("graph must contain exactly the nine typed relations")
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

    trucks, truck_mask = _pad_rows(features.truck_features, shape.max_trucks, "trucks")
    customers, customer_mask = _pad_rows(
        features.customer_features, shape.max_customers, "customers"
    )
    chargers, charger_mask = _pad_rows(
        features.charger_features, shape.max_chargers, "chargers"
    )
    actions, action_mask = _pad_rows(
        features.action_features, shape.max_actions, "actions"
    )
    entity_masks = {
        "truck": truck_mask,
        "customer": customer_mask,
        "charger": charger_mask,
    }
    pairwise_features: dict[tuple[str, str], np.ndarray] = {}
    pairwise_mask: dict[tuple[str, str], np.ndarray] = {}
    limits = shape.node_limits
    for relation in RELATION_TYPES:
        source, target = relation
        values = features.pairwise_features[relation]
        padded = np.zeros(
            (limits[source], limits[target], len(EDGE_FEATURES)), dtype=np.float32
        )
        rows, columns = values.shape[0], values.shape[1]
        padded[:rows, :columns] = values
        pairwise_features[relation] = padded
        pairwise_mask[relation] = np.outer(entity_masks[source], entity_masks[target])

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
        pairwise_features=pairwise_features,
        pairwise_mask=pairwise_mask,
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
    return canonical_graph_features(env.get_canonical_features())


def canonical_graph_features(
    features: CanonicalFleetFeatures,
) -> CanonicalGraphFeatures:
    """Expand the canonical pairwise tensors into dense typed edge lists.

    The adapter never queries the transport graph itself; it consumes exactly
    the pairwise values the flat and padded-set adapters receive.
    """
    features.validate()
    node_features = {
        "truck": features.truck_features.copy(),
        "customer": features.customer_features.copy(),
        "charger": features.charger_features.copy(),
    }
    counts = features.node_counts()
    edge_indices: dict[tuple[str, str], np.ndarray] = {}
    edge_features: dict[tuple[str, str], np.ndarray] = {}
    for relation in RELATION_TYPES:
        source, target = relation
        source_count, target_count = counts[source], counts[target]
        rows = np.repeat(np.arange(source_count, dtype=np.int64), target_count)
        columns = np.tile(np.arange(target_count, dtype=np.int64), source_count)
        edge_indices[relation] = np.stack((rows, columns))
        edge_features[relation] = (
            features.pairwise_features[relation]
            .reshape(source_count * target_count, len(EDGE_FEATURES))
            .astype(np.float32, copy=True)
        )

    result = CanonicalGraphFeatures(
        schema_version=features.schema_version,
        node_features=node_features,
        edge_indices=edge_indices,
        edge_features=edge_features,
        action_features=features.action_features.copy(),
        global_features=features.global_features.copy(),
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
        raise ValueError(f"{label} count {count} exceeds configured maximum {maximum}")
    padded = np.zeros((maximum, width), dtype=np.float32)
    mask = np.zeros(maximum, dtype=bool)
    if count:
        padded[:count] = values
        mask[:count] = True
    return padded, mask


def relation_matrix(
    graph: CanonicalGraphFeatures,
    relation: tuple[str, str],
) -> np.ndarray:
    """Fold one typed edge list back into its dense source-target matrix."""
    if relation not in graph.edge_indices:
        raise KeyError(f"graph has no relation {relation}")
    source, target = relation
    counts = {
        node_type: int(values.shape[0])
        for node_type, values in graph.node_features.items()
    }
    return graph.edge_features[relation].reshape(
        counts[source], counts[target], len(EDGE_FEATURES)
    )
