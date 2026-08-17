"""Torch view of the canonical joint-fleet observation.

Every canonical policy consumes the *same* flat observation vector emitted by
the environment and unpacks it here.  Representation-specific inductive bias
therefore never changes what a policy can observe: flat, DeepSets, and
heterogeneous-graph encoders all start from byte-identical inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from EVRoutingEnv.state.features import (
    ACTION_FEATURES,
    CHARGER_FEATURES,
    CUSTOMER_FEATURES,
    EDGE_FEATURES,
    GLOBAL_FEATURES,
    NODE_TYPES,
    RELATION_TYPES,
    TRUCK_FEATURES,
)
from EVRoutingEnv.state.representations import CanonicalShapeSpec


NODE_FEATURE_NAMES = {
    "truck": TRUCK_FEATURES,
    "customer": CUSTOMER_FEATURES,
    "charger": CHARGER_FEATURES,
}
NODE_FEATURE_DIMS = {
    node_type: len(names) for node_type, names in NODE_FEATURE_NAMES.items()
}
ACTION_FEATURE_DIM = len(ACTION_FEATURES)
GLOBAL_FEATURE_DIM = len(GLOBAL_FEATURES)
EDGE_FEATURE_DIM = len(EDGE_FEATURES)
FEASIBLE_COLUMN = ACTION_FEATURES.index("feasible")


@dataclass(frozen=True)
class CanonicalTensors:
    """Batched typed blocks recovered from the canonical flat observation."""

    nodes: dict[str, torch.Tensor]
    node_masks: dict[str, torch.Tensor]
    action: torch.Tensor
    action_padding_mask: torch.Tensor
    pairwise: dict[tuple[str, str], torch.Tensor]
    pairwise_mask: dict[tuple[str, str], torch.Tensor]
    global_features: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.global_features.shape[0])

    @property
    def device(self) -> torch.device:
        return self.global_features.device

    def feasible_action_mask(self) -> torch.Tensor:
        """Hard feasibility mask: a real action row that the simulator allows."""
        feasible = self.action[..., FEASIBLE_COLUMN] > 0.5
        return feasible & self.action_padding_mask

    def ragged_actions(
        self,
        override_mask: torch.Tensor | None = None,
        allow_infeasible: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return concatenated selectable action rows and their CSR pointers.

        ``override_mask`` may narrow the hard mask (for example with the mask a
        vectorized environment reports), never widen it.  A state with no
        feasible action is refused rather than silently relaxed.

        ``allow_infeasible`` is the one declared exception, used by the no-mask
        ablation: the override then defines the selectable set directly and may
        include actions the simulator would reject, which is exactly the
        capability under test.  Padding rows stay excluded either way -- they
        carry no action to score.
        """
        mask = self.feasible_action_mask()
        if override_mask is not None:
            override = override_mask.to(dtype=torch.bool, device=mask.device)
            if override.shape != mask.shape:
                raise ValueError(
                    f"override mask shape {tuple(override.shape)} does not match "
                    f"the canonical action mask {tuple(mask.shape)}"
                )
            if allow_infeasible:
                mask = override & self.action_padding_mask
            else:
                if (override & ~mask).any():
                    raise ValueError(
                        "override mask enables actions the simulator marked infeasible"
                    )
                mask = mask & override
        elif allow_infeasible:
            mask = self.action_padding_mask
        counts = mask.sum(dim=1)
        if int(counts.min().item()) == 0:
            raise RuntimeError(
                "a state has no feasible action; refusing to relax the hard mask"
            )
        rows = self.action[mask]
        ptr = torch.zeros(
            mask.shape[0] + 1, dtype=torch.long, device=self.action.device
        )
        torch.cumsum(counts, dim=0, out=ptr[1:])
        return rows, ptr


def flat_observation_slices(shape: CanonicalShapeSpec) -> dict[str, slice]:
    """Return the byte layout of :meth:`PaddedCanonicalFeatures.flatten`."""
    limits = shape.node_limits
    offset = 0
    layout: dict[str, slice] = {}

    def take(name: str, size: int) -> None:
        nonlocal offset
        layout[name] = slice(offset, offset + size)
        offset += size

    for node_type in NODE_TYPES:
        take(f"{node_type}_features", limits[node_type] * NODE_FEATURE_DIMS[node_type])
    take("action_features", shape.max_actions * ACTION_FEATURE_DIM)
    for node_type in NODE_TYPES:
        take(f"{node_type}_mask", limits[node_type])
    take("action_mask", shape.max_actions)
    for source, target in RELATION_TYPES:
        take(
            f"pairwise_{source}_{target}",
            limits[source] * limits[target] * EDGE_FEATURE_DIM,
        )
    for source, target in RELATION_TYPES:
        take(f"pairwise_mask_{source}_{target}", limits[source] * limits[target])
    take("global_features", GLOBAL_FEATURE_DIM)

    if offset != shape.flat_size:
        raise RuntimeError(
            f"flat layout covers {offset} scalars but flat_size is {shape.flat_size}"
        )
    return layout


def unpack_flat_observation(
    observation: torch.Tensor,
    shape: CanonicalShapeSpec,
) -> CanonicalTensors:
    """Recover the typed canonical blocks from a batch of flat observations."""
    if observation.ndim == 1:
        observation = observation.unsqueeze(0)
    if observation.ndim != 2:
        raise ValueError("observation must be one- or two-dimensional")
    if observation.shape[1] != shape.flat_size:
        raise ValueError(
            f"observation width {observation.shape[1]} does not match the canonical "
            f"flat size {shape.flat_size}"
        )
    if not torch.isfinite(observation).all():
        raise ValueError("canonical observation contains non-finite values")

    layout = flat_observation_slices(shape)
    limits = shape.node_limits
    batch = observation.shape[0]

    nodes: dict[str, torch.Tensor] = {}
    node_masks: dict[str, torch.Tensor] = {}
    for node_type in NODE_TYPES:
        nodes[node_type] = observation[:, layout[f"{node_type}_features"]].reshape(
            batch, limits[node_type], NODE_FEATURE_DIMS[node_type]
        )
        node_masks[node_type] = (
            observation[:, layout[f"{node_type}_mask"]] > 0.5
        ).reshape(batch, limits[node_type])

    action = observation[:, layout["action_features"]].reshape(
        batch, shape.max_actions, ACTION_FEATURE_DIM
    )
    action_padding_mask = (observation[:, layout["action_mask"]] > 0.5).reshape(
        batch, shape.max_actions
    )

    pairwise: dict[tuple[str, str], torch.Tensor] = {}
    pairwise_mask: dict[tuple[str, str], torch.Tensor] = {}
    for relation in RELATION_TYPES:
        source, target = relation
        pairwise[relation] = observation[
            :, layout[f"pairwise_{source}_{target}"]
        ].reshape(batch, limits[source], limits[target], EDGE_FEATURE_DIM)
        pairwise_mask[relation] = (
            observation[:, layout[f"pairwise_mask_{source}_{target}"]] > 0.5
        ).reshape(batch, limits[source], limits[target])

    return CanonicalTensors(
        nodes=nodes,
        node_masks=node_masks,
        action=action,
        action_padding_mask=action_padding_mask,
        pairwise=pairwise,
        pairwise_mask=pairwise_mask,
        global_features=observation[:, layout["global_features"]],
    )


def scatter_logits(
    logits: torch.Tensor,
    ptr: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Place ragged feasible-action logits back into a fixed-width matrix.

    Infeasible positions receive a large negative constant rather than ``-inf``
    so that downstream log-softmax stays finite under autograd.
    """
    if logits.ndim != 1:
        raise ValueError("logits must be one-dimensional")
    if int(ptr[-1].item()) != logits.numel():
        raise ValueError("ptr does not terminate at the number of logits")
    if int(mask.sum().item()) != logits.numel():
        raise ValueError("mask cardinality does not match the number of logits")
    dense = torch.full(
        mask.shape,
        torch.finfo(logits.dtype).min / 2,
        dtype=logits.dtype,
        device=logits.device,
    )
    return dense.masked_scatter(mask, logits)
