"""Generalist GraphPPO actor-critic: no parameter depends on a problem's size.

The canonical actor-critic is built for one :class:`CanonicalShapeSpec`; its
flat observation fixes the fleet, task and charger envelopes, so a checkpoint
cannot even read another problem. Here every learned map acts on one row (a
truck, a task, a charger, an action) or one typed edge, and every aggregation
is a masked mean or max, so the same weights score a 3-truck, 25-charger state
and a 10-truck, 432-charger state. Batches are padded to their own largest
graph and padding never reaches a valid row.

Architecture (plan section 4, kept): typed message passing over the
heterogeneous state graph, a complete-graph GCN over the feasible actions, and
a masked categorical policy. Each action row is linked to the embedding of its
target task or charger and to the active truck's embedding, instead of a node
id, and every node is conditioned on the context row.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical

from algo.action_heads import CompleteGraphGCNActionHead
from algo.canonical_encoders import (
    RunningStandardizer,
    _masked_pool,
    _mlp,
    _TypedMessagePassingLayer,
)
from algo.canonical_policy import PolicyOutput
from algo.canonical_state import scatter_logits
from algo.generalist_observation import (
    FEATURE_DIMS,
    GENERALIST_SCHEMA_VERSION,
    NODE_TYPES,
    PLAN_ACTION_DIM,
    PLAN_TASK_DIM,
    PLAN_TRUCK_DIM,
    PRIVILEGED_DIM,
    RELATION_TYPES,
    GraphBatch,
    ObservationSpec,
)


@dataclass(frozen=True)
class GeneralistPolicyConfig:
    """Everything needed to rebuild a generalist policy from disk."""

    hidden_dim: int = 128
    encoder_layers: int = 3
    action_head_layers: int = 2
    nonlinear_edge_messages: bool = True
    charger_budget: int = 32
    fleet_assignment: bool = False
    # Plan 9.4: the value head also reads the hindsight stream summary; the
    # actor path never does, so the policy stays causal.
    privileged_critic: bool = False
    # Plan 9.5: the rolling planner's live plan conditions tasks, trucks and
    # action candidates on the dynamic problems; the policy may deviate.
    plan_conditioning: bool = False
    feature_schema_version: str = GENERALIST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("hidden_dim", "encoder_layers", "action_head_layers"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        ObservationSpec(self.charger_budget, self.fleet_assignment, self.privileged_critic, self.plan_conditioning)
        if self.feature_schema_version != GENERALIST_SCHEMA_VERSION:
            raise ValueError(
                f"checkpoint feature schema {self.feature_schema_version!r} is incompatible "
                f"with {GENERALIST_SCHEMA_VERSION!r}; retrain using the current observation"
            )

    @property
    def observation_spec(self) -> ObservationSpec:
        return ObservationSpec(
            self.charger_budget, self.fleet_assignment, self.privileged_critic, self.plan_conditioning,
        )

    def save(self, path: str | Path) -> None:
        from algo.checkpoint_io import write_text_atomic

        write_text_atomic(json.dumps(asdict(self), indent=2, sort_keys=True), path)

    @classmethod
    def load(cls, path: str | Path) -> GeneralistPolicyConfig:
        return cls(**json.loads(Path(path).read_text()))


class GeneralistGraphEncoder(nn.Module):
    """Context-conditioned typed message passing with a size-free readout."""

    def __init__(self, hidden_dim: int, num_layers: int, nonlinear_edge_messages: bool):
        super().__init__()
        self.normalizer = RunningStandardizer(dict(FEATURE_DIMS))
        self.context_encoder = _mlp(FEATURE_DIMS["context"], hidden_dim, hidden_dim)
        self.input_projections = nn.ModuleDict(
            {node_type: nn.Linear(FEATURE_DIMS[node_type], hidden_dim) for node_type in NODE_TYPES}
        )
        self.context_projections = nn.ModuleDict(
            {node_type: nn.Linear(hidden_dim, hidden_dim, bias=False) for node_type in NODE_TYPES}
        )
        self.layers = nn.ModuleList(
            _TypedMessagePassingLayer(
                hidden_dim,
                nonlinear_edge_messages=nonlinear_edge_messages,
                edge_dim=FEATURE_DIMS["edge"],
                node_types=NODE_TYPES,
            )
            for _ in range(num_layers)
        )
        self.readout = nn.Sequential(
            nn.Linear(2 * hidden_dim * len(NODE_TYPES) + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    @torch.no_grad()
    def observe(self, batch: GraphBatch) -> None:
        """Fold valid rows into the running statistics; padding never counts."""
        for node_type in NODE_TYPES:
            self.normalizer._update(node_type, batch.nodes[node_type][batch.node_masks[node_type]])
        self.normalizer._update("action", batch.actions[batch.action_mask])
        self.normalizer._update("context", batch.context)
        self.normalizer._update(
            "edge",
            torch.cat([
                batch.pairwise[relation][batch.pairwise_mask[relation]].float()
                for relation in RELATION_TYPES
            ]),
        )

    def standardized_actions(self, batch: GraphBatch) -> torch.Tensor:
        return self.normalizer._standardize("action", batch.actions) * batch.action_mask.unsqueeze(-1)

    def forward(
        self, batch: GraphBatch, extra: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """``extra`` adds gated per-row terms (plan 9.5) before the projection's ReLU."""
        context = self.context_encoder(self.normalizer._standardize("context", batch.context))
        embeddings = {}
        for node_type in NODE_TYPES:
            mask = batch.node_masks[node_type].unsqueeze(-1)
            values = self.normalizer._standardize(node_type, batch.nodes[node_type]) * mask
            projected = (
                self.input_projections[node_type](values)
                + self.context_projections[node_type](context).unsqueeze(1)
            )
            if extra is not None and node_type in extra:
                projected = projected + extra[node_type]
            embeddings[node_type] = F.relu(projected) * mask
        pairwise = {
            relation: self.normalizer._standardize("edge", batch.pairwise[relation].float())
            * batch.pairwise_mask[relation].unsqueeze(-1)
            for relation in RELATION_TYPES
        }
        for layer in self.layers:
            embeddings = layer(embeddings, pairwise, batch.pairwise_mask, batch.node_masks)
        summary = [
            _masked_pool(embeddings[node_type], batch.node_masks[node_type])
            for node_type in NODE_TYPES
        ]
        pooled = self.readout(torch.cat([*summary, context], dim=-1))
        return pooled, embeddings


def _gather_rows(embeddings: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """``embeddings[b, rows[b, a]]``, with zeros wherever ``rows`` is -1."""
    valid = rows >= 0
    index = rows.clamp_min(0).unsqueeze(-1).expand(-1, -1, embeddings.shape[-1])
    return torch.gather(embeddings, 1, index) * valid.unsqueeze(-1)


class GeneralistActorCritic(nn.Module):
    """Shared-encoder actor-critic over variable state graphs and action sets."""

    def __init__(self, config: GeneralistPolicyConfig):
        super().__init__()
        self.config = config
        hidden = config.hidden_dim
        self.encoder = GeneralistGraphEncoder(
            hidden, config.encoder_layers, config.nonlinear_edge_messages
        )
        self.action_head = CompleteGraphGCNActionHead(
            hidden,
            hidden,
            FEATURE_DIMS["action"] + 2 * hidden + (PLAN_ACTION_DIM if config.plan_conditioning else 0),
            num_layers=config.action_head_layers,
        )
        value_input = hidden + (hidden if config.privileged_critic else 0)
        self.value_head = nn.Sequential(
            nn.Linear(value_input, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )
        if config.privileged_critic:
            # Plan 9.4, asymmetric critic: these modules exist only with the
            # flag, so a flag-off checkpoint keeps today's state dict exactly.
            self.privileged_normalizer = RunningStandardizer({"privileged": PRIVILEGED_DIM})
            self.privileged_encoder = _mlp(PRIVILEGED_DIM, hidden, hidden)
        if config.plan_conditioning:
            # Plan 9.5: bias-free so an all-zero plan row (static problem,
            # dropped episode, flag-free batch) injects exactly nothing; the
            # modules exist only with the flag, keeping old state dicts intact.
            self.plan_task_encoder = nn.Linear(PLAN_TASK_DIM, hidden, bias=False)
            self.plan_truck_encoder = nn.Linear(PLAN_TRUCK_DIM, hidden, bias=False)

    def observe(self, batch: GraphBatch) -> None:
        self.encoder.observe(batch)
        if self.config.privileged_critic and batch.privileged is not None:
            self.privileged_normalizer._update("privileged", batch.privileged)

    def _values(self, pooled: torch.Tensor, batch: GraphBatch) -> torch.Tensor:
        """The value estimate; only here may the privileged rows be read."""
        if not self.config.privileged_critic:
            return self.value_head(pooled).squeeze(-1)
        rows = batch.privileged
        if rows is None:
            rows = pooled.new_zeros(pooled.shape[0], PRIVILEGED_DIM)
        hindsight = self.privileged_encoder(self.privileged_normalizer._standardize("privileged", rows))
        return self.value_head(torch.cat([pooled, hindsight], dim=-1)).squeeze(-1)

    def _plan_extra(self, batch: GraphBatch) -> dict[str, torch.Tensor] | None:
        """Gated per-row plan terms for the encoder (plan 9.5); zero rows add nothing."""
        if not self.config.plan_conditioning or batch.plan_tasks is None:
            return None
        return {
            "task": self.plan_task_encoder(batch.plan_tasks),
            "truck": self.plan_truck_encoder(batch.plan_trucks),
        }

    def forward(self, batch: GraphBatch) -> PolicyOutput:
        mask = batch.action_mask
        counts = mask.sum(dim=1)
        if int(counts.min().item()) == 0:
            raise RuntimeError("a state has no selectable action; refusing to relax the hard mask")
        pooled, nodes = self.encoder(batch, self._plan_extra(batch))
        targets = _gather_rows(nodes["task"], batch.action_task) + _gather_rows(
            nodes["charger"], batch.action_charger
        )
        active = _gather_rows(nodes["truck"], batch.active_truck.unsqueeze(1))
        columns = [
            self.encoder.standardized_actions(batch),
            targets,
            active.expand(-1, mask.shape[1], -1),
        ]
        if self.config.plan_conditioning:
            plan_column = batch.plan_actions
            if plan_column is None:
                plan_column = mask.new_zeros((*mask.shape, PLAN_ACTION_DIM), dtype=columns[0].dtype)
            columns.append(plan_column)
        rows = torch.cat(columns, dim=-1)[mask]
        ptr = torch.zeros(mask.shape[0] + 1, dtype=torch.long, device=mask.device)
        torch.cumsum(counts, dim=0, out=ptr[1:])
        head = self.action_head(pooled, rows, ptr)
        return PolicyOutput(
            logits=scatter_logits(head.logits, head.ptr, mask),
            values=self._values(pooled, batch),
            feasible_mask=mask,
        )

    def predict_values(self, batch: GraphBatch) -> torch.Tensor:
        """Value estimate that never touches the action head (terminal states)."""
        pooled, _ = self.encoder(batch, self._plan_extra(batch))
        return self._values(pooled, batch)

    @torch.no_grad()
    def act(self, batch: GraphBatch, deterministic: bool = False):
        """Candidate positions, their log probabilities, and state values."""
        output = self(batch)
        distribution = Categorical(logits=output.logits)
        actions = output.logits.argmax(dim=-1) if deterministic else distribution.sample()
        return actions, distribution.log_prob(actions), output.values

    def evaluate_actions(self, batch: GraphBatch, actions: torch.Tensor):
        """Score stored candidate positions; refuses any outside the mask."""
        output = self(batch)
        chosen = output.feasible_mask.gather(1, actions.view(-1, 1).long()).squeeze(1)
        if not bool(chosen.all().item()):
            offending = int(torch.nonzero(~chosen).view(-1)[0].item())
            raise ValueError(
                f"stored action {int(actions[offending].item())} is outside the "
                f"candidate set of batch element {offending}"
            )
        distribution = Categorical(logits=output.logits)
        return distribution.log_prob(actions), distribution.entropy(), output.values

    def save(self, directory: str | Path, prefix: str = "policy") -> None:
        from algo.checkpoint_io import save_torch_atomic

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        save_torch_atomic(self.state_dict(), directory / f"{prefix}.pt")
        self.config.save(directory / f"{prefix}_config.json")

    @classmethod
    def load(
        cls, directory: str | Path, prefix: str = "policy", map_location: str | torch.device = "cpu"
    ) -> GeneralistActorCritic:
        directory = Path(directory)
        policy = cls(GeneralistPolicyConfig.load(directory / f"{prefix}_config.json"))
        policy.load_state_dict(torch.load(directory / f"{prefix}.pt", map_location=map_location))
        return policy
