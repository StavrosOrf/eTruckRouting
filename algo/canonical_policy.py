"""Canonical actor-critic: one state encoder plus one approved action head.

The policy is fully described by ``(state_encoder, action_head)``.  Every
combination sees the identical canonical observation and the identical hard
feasibility mask, so a campaign that varies one factor at a time measures
architecture rather than information.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.distributions import Categorical

from algo.action_heads import ACTION_HEAD_TYPES, build_action_head
from algo.canonical_encoders import STATE_ENCODER_TYPES, build_state_encoder
from algo.canonical_state import (
    ACTION_FEATURE_DIM,
    CanonicalTensors,
    scatter_logits,
    unpack_flat_observation,
)
from algo.progress_features import (
    PROGRESS_FEATURE_DIM,
    ProgressActionFeatures,
    station_progress_config,
)
from EVRoutingEnv.state.features import (
    ACTION_FEATURES,
    CHARGER_FEATURES,
    CUSTOMER_FEATURES,
    SCHEMA_VERSION,
    TRUCK_FEATURES,
)
from EVRoutingEnv.state.representations import CanonicalShapeSpec


@dataclass(frozen=True)
class CanonicalPolicyConfig:
    """Everything needed to rebuild a canonical policy from disk."""

    max_trucks: int
    max_customers: int
    max_chargers: int
    max_actions: int
    state_encoder: str = "hetero_graph"
    action_head: str = "self_attention"
    hidden_dim: int = 128
    encoder_output_dim: int = 128
    encoder_layers: int = 2
    action_head_layers: int = 2
    action_attention_heads: int = 4
    action_head_dropout: float = 0.0
    # Set only by the no-mask ablation.  The policy then scores every candidate
    # the environment declares selectable, including ones the simulator would
    # reject, and has to learn feasibility itself.  Persisted with the
    # checkpoint so a run cannot be evaluated under a mask it never trained on.
    allow_infeasible_actions: bool = False
    # Component ablation: withhold the pooled state embedding from the action
    # head. Persisted with the checkpoint so an ablated run is never reloaded as
    # if it were the full architecture.
    ablate_state_pooling: bool = False
    normalize_action_features: bool = False
    action_entity_links: bool = False
    nonlinear_edge_messages: bool = False
    progress_action_features: bool = False
    progress_station_ids: tuple[int, ...] = ()
    progress_station_hops: tuple[tuple[float, ...], ...] = ()
    progress_energy_multiplier: float = 1.0
    feature_schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "progress_station_ids", tuple(self.progress_station_ids)
        )
        object.__setattr__(
            self,
            "progress_station_hops",
            tuple(tuple(row) for row in self.progress_station_hops),
        )
        if self.progress_action_features and (
            len(self.progress_station_ids) != self.max_chargers
            or self.max_chargers == 0
            or len(self.progress_station_hops) != self.max_chargers
            or any(len(row) != self.max_chargers for row in self.progress_station_hops)
        ):
            raise ValueError(
                "progress features require the fixed station map and hop matrix"
            )
        if self.progress_action_features and (
            not math.isfinite(self.progress_energy_multiplier)
            or self.progress_energy_multiplier < 1
            or len(set(self.progress_station_ids)) != self.max_chargers
            or any(
                not math.isfinite(value) or value < 0
                for row in self.progress_station_hops
                for value in row
            )
        ):
            raise ValueError(
                "progress feature map costs and energy factor must be valid"
            )
        if self.action_entity_links and self.state_encoder != "hetero_graph":
            raise ValueError(
                "action entity links require the heterogeneous graph encoder"
            )
        if self.feature_schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"checkpoint feature schema {self.feature_schema_version!r} is incompatible "
                f"with {SCHEMA_VERSION!r}; retrain using the current observation layout"
            )
        if self.state_encoder not in STATE_ENCODER_TYPES:
            raise ValueError(
                f"state_encoder must be one of {STATE_ENCODER_TYPES}, got "
                f"{self.state_encoder!r}"
            )
        if self.action_head not in ACTION_HEAD_TYPES:
            raise ValueError(
                f"action_head must be one of {ACTION_HEAD_TYPES}, got "
                f"{self.action_head!r}"
            )

    @property
    def shape(self) -> CanonicalShapeSpec:
        return CanonicalShapeSpec(
            max_trucks=self.max_trucks,
            max_customers=self.max_customers,
            max_chargers=self.max_chargers,
            max_actions=self.max_actions,
        )

    @classmethod
    def from_env(cls, env, **overrides) -> CanonicalPolicyConfig:
        shape = env.canonical_shape
        if overrides.get("progress_action_features"):
            overrides.update(station_progress_config(env))
        return cls(
            max_trucks=shape.max_trucks,
            max_customers=shape.max_customers,
            max_chargers=shape.max_chargers,
            max_actions=shape.max_actions,
            **overrides,
        )

    def save(self, path: str | Path) -> None:
        from algo.checkpoint_io import write_text_atomic

        write_text_atomic(json.dumps(asdict(self), indent=2, sort_keys=True), path)

    @classmethod
    def load(cls, path: str | Path) -> CanonicalPolicyConfig:
        payload = json.loads(Path(path).read_text())
        payload.setdefault("feature_schema_version", "unversioned")
        return cls(**payload)


@dataclass(frozen=True)
class PolicyOutput:
    """Dense masked logits, state values, and the hard mask that produced them."""

    logits: torch.Tensor
    values: torch.Tensor
    feasible_mask: torch.Tensor

    def distribution(self) -> Categorical:
        return Categorical(logits=self.logits)


class CanonicalActorCritic(nn.Module):
    """Shared-encoder actor-critic over variable-size feasible action sets."""

    def __init__(self, config: CanonicalPolicyConfig):
        super().__init__()
        self.config = config
        self.encoder = build_state_encoder(
            config.state_encoder,
            config.shape,
            config.hidden_dim,
            config.encoder_output_dim,
            num_layers=config.encoder_layers,
            nonlinear_edge_messages=config.nonlinear_edge_messages,
        )
        self.action_head = build_action_head(
            config.action_head,
            config.encoder_output_dim,
            config.hidden_dim,
            ACTION_FEATURE_DIM
            + (PROGRESS_FEATURE_DIM if config.progress_action_features else 0)
            + (2 * config.hidden_dim if config.action_entity_links else 0),
            num_layers=config.action_head_layers,
            attention_heads=config.action_attention_heads,
            dropout=config.action_head_dropout,
        )
        self.progress_features = (
            ProgressActionFeatures(
                config.progress_station_ids,
                config.progress_station_hops,
                config.progress_energy_multiplier,
            )
            if config.progress_action_features
            else None
        )
        self.value_head = nn.Sequential(
            nn.Linear(config.encoder_output_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1),
        )

    def unpack(self, observation: torch.Tensor) -> CanonicalTensors:
        return unpack_flat_observation(observation, self.config.shape)

    def observe(self, observation: torch.Tensor) -> None:
        """Fold a batch of observations into the shared normalizer statistics."""
        self.encoder.observe(self.unpack(observation))

    def forward(
        self,
        observation: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> PolicyOutput:
        tensors = self.unpack(observation)
        unmasked = self.config.allow_infeasible_actions
        action_rows, ptr = tensors.ragged_actions(
            action_mask, allow_infeasible=unmasked
        )
        if unmasked:
            # The selectable set is whatever the environment offered, minus
            # padding; feasibility is the policy's problem, not the mask's.
            feasible = tensors.action_padding_mask
            if action_mask is not None:
                feasible = feasible & action_mask.to(
                    dtype=torch.bool, device=feasible.device
                )
        else:
            feasible = tensors.feasible_action_mask()
            if action_mask is not None:
                feasible = feasible & action_mask.to(
                    dtype=torch.bool, device=feasible.device
                )
        if self.config.normalize_action_features:
            action_rows = self.encoder.normalizer._standardize(
                "action", tensors.action
            )[feasible]
        if self.progress_features is not None:
            action_rows = torch.cat(
                (action_rows, self.progress_features(tensors)[feasible]), dim=-1
            )
        if self.config.action_entity_links:
            embedding, nodes = self.encoder(tensors, return_nodes=True)
            active = tensors.nodes["truck"][..., TRUCK_FEATURES.index("is_active")]
            active_embedding = (nodes["truck"] * active.unsqueeze(-1)).sum(dim=1)
            target_ids = tensors.action[..., ACTION_FEATURES.index("target_node")]
            target_embedding = nodes["truck"].new_zeros(
                (*target_ids.shape, self.config.hidden_dim)
            )
            # Customer/station identity is used only for exact matching, never
            # as an ordinal neural feature. California uses small integer IDs.
            for kind, names in (
                ("customer", CUSTOMER_FEATURES),
                ("charger", CHARGER_FEATURES),
            ):
                ids = tensors.nodes[kind][..., names.index("node_id")]
                matches = target_ids.unsqueeze(-1) == ids.unsqueeze(1)
                matches = matches & tensors.node_masks[kind].unsqueeze(1)
                target_embedding = target_embedding + torch.bmm(
                    matches.to(embedding.dtype), nodes[kind]
                )
            action_rows = torch.cat(
                (
                    action_rows,
                    active_embedding.unsqueeze(1).expand(
                        -1, self.config.max_actions, -1
                    )[feasible],
                    target_embedding[feasible],
                ),
                dim=-1,
            )
        else:
            embedding = self.encoder(tensors)
        head_input = embedding
        if self.config.ablate_state_pooling:
            # Component ablation: the action head scores candidates without the
            # pooled fleet embedding, so a candidate is judged only on its own
            # row. The critic keeps the real embedding, so this isolates what
            # pooling contributes to the *policy* rather than crippling value
            # estimation as well.
            head_input = torch.zeros_like(embedding)
        head_output = self.action_head(head_input, action_rows, ptr)
        logits = scatter_logits(head_output.logits, head_output.ptr, feasible)
        values = self.value_head(embedding).squeeze(-1)
        return PolicyOutput(logits=logits, values=values, feasible_mask=feasible)

    def predict_values(self, observation: torch.Tensor) -> torch.Tensor:
        """Value estimate that never touches the action head.

        Terminal observations can legitimately expose an empty feasible set, so
        bootstrapping must not require a valid action distribution.
        """
        embedding = self.encoder(self.unpack(observation))
        return self.value_head(embedding).squeeze(-1)

    @torch.no_grad()
    def act(
        self,
        observation: torch.Tensor,
        action_mask: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return sampled actions, their log probabilities, and state values."""
        output = self(observation, action_mask)
        distribution = output.distribution()
        if deterministic:
            actions = torch.argmax(output.logits, dim=-1)
        else:
            actions = distribution.sample()
        return actions, distribution.log_prob(actions), output.values

    def evaluate_actions(
        self,
        observation: torch.Tensor,
        actions: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Score stored actions; refuses any action outside the hard mask."""
        output = self(observation, action_mask)
        chosen = output.feasible_mask.gather(1, actions.view(-1, 1).long()).squeeze(1)
        if not bool(chosen.all().item()):
            offending = int(torch.nonzero(~chosen).view(-1)[0].item())
            raise ValueError(
                f"stored action {int(actions[offending].item())} is outside the "
                f"feasible set of batch element {offending}"
            )
        distribution = output.distribution()
        return (
            distribution.log_prob(actions),
            distribution.entropy(),
            output.values,
        )

    def save(self, directory: str | Path, prefix: str = "policy") -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        from algo.checkpoint_io import save_torch_atomic

        save_torch_atomic(self.state_dict(), directory / f"{prefix}.pt")
        self.config.save(directory / f"{prefix}_config.json")

    @classmethod
    def load(
        cls,
        directory: str | Path,
        prefix: str = "policy",
        map_location: str | torch.device = "cpu",
    ) -> CanonicalActorCritic:
        directory = Path(directory)
        config = CanonicalPolicyConfig.load(directory / f"{prefix}_config.json")
        policy = cls(config)
        policy.load_state_dict(
            torch.load(directory / f"{prefix}.pt", map_location=map_location)
        )
        return policy
