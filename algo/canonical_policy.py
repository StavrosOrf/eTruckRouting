"""Canonical actor-critic: one state encoder plus one approved action head.

The policy is fully described by ``(state_encoder, action_head)``.  Every
combination sees the identical canonical observation and the identical hard
feasibility mask, so a campaign that varies one factor at a time measures
architecture rather than information.
"""

from __future__ import annotations

import json
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

    def __post_init__(self) -> None:
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
        return cls(
            max_trucks=shape.max_trucks,
            max_customers=shape.max_customers,
            max_chargers=shape.max_chargers,
            max_actions=shape.max_actions,
            **overrides,
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: str | Path) -> CanonicalPolicyConfig:
        return cls(**json.loads(Path(path).read_text()))


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
        )
        self.action_head = build_action_head(
            config.action_head,
            config.encoder_output_dim,
            config.hidden_dim,
            ACTION_FEATURE_DIM,
            num_layers=config.action_head_layers,
            attention_heads=config.action_attention_heads,
            dropout=config.action_head_dropout,
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
        embedding = self.encoder(tensors)
        head_output = self.action_head(embedding, action_rows, ptr)
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
        torch.save(self.state_dict(), directory / f"{prefix}.pt")
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
