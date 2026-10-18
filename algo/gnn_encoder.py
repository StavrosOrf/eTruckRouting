"""Shared heterogeneous-GNN layers used by the variable-action policy."""

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import MessagePassing, global_mean_pool


BATCH_EXCLUDE_KEYS = [
    "action_to_node_map",
    "node_id_to_type",
    "node_type_offsets",
    "can_charge_here",
]


class MLP(nn.Module):
    """Multi-layer perceptron with optional layer normalization."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = in_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class EdgeConditionedConv(MessagePassing):
    """Message-passing layer conditioned on source, target, and edge features."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int,
        aggr: str = "add",
    ) -> None:
        super().__init__(aggr=aggr)
        self.message_mlp = MLP(
            in_dim=2 * in_channels + edge_dim,
            hidden_dim=out_channels,
            out_dim=out_channels,
            num_layers=2,
        )
        self.update_mlp = MLP(
            in_dim=in_channels + out_channels,
            hidden_dim=out_channels,
            out_dim=out_channels,
            num_layers=1,
        )

    def forward(self, inputs, edge_index, edge_attr, size=None):
        if isinstance(inputs, tuple):
            source, target = inputs
        else:
            source = target = inputs
        return self.propagate(
            edge_index,
            x=(source, target),
            edge_attr=edge_attr,
            size=size,
        )

    def message(self, x_i, x_j, edge_attr):
        return self.message_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))

    def update(self, aggr_out, x):
        target = x[1] if isinstance(x, tuple) else x
        return self.update_mlp(torch.cat([target, aggr_out], dim=-1))


class HeteroInteractionLayer(nn.Module):
    """One heterogeneous graph-interaction layer."""

    def __init__(
        self,
        node_channels: dict[str, int],
        edge_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.node_types = list(node_channels)
        self.edge_types = [
            ("truck", "to", "delivery"),
            ("delivery", "to", "truck"),
            ("truck", "to", "charger"),
            ("charger", "to", "truck"),
            ("charger", "to", "charger"),
            ("truck", "to", "truck"),
            ("delivery", "to", "delivery"),
            ("charger", "to", "delivery"),
        ]
        self.convs = nn.ModuleDict()
        for source, relation, target in self.edge_types:
            self.convs[f"{source}__{relation}__{target}"] = EdgeConditionedConv(
                node_channels[source], hidden_dim, edge_dim
            )

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[Any, torch.Tensor],
        edge_attr_dict: dict[Any, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        incoming: dict[str, list[torch.Tensor]] = {
            node_type: [] for node_type in self.node_types
        }
        for edge_type in self.edge_types:
            if edge_type not in edge_index_dict:
                continue
            source, relation, target = edge_type
            edge_index = edge_index_dict[edge_type]
            if edge_index.shape[1] == 0:
                continue
            source_features = x_dict[source]
            target_features = x_dict[target]
            incoming[target].append(
                self.convs[f"{source}__{relation}__{target}"](
                    (source_features, target_features),
                    edge_index,
                    edge_attr_dict[edge_type],
                    size=(source_features.shape[0], target_features.shape[0]),
                )
            )
        return {
            node_type: (
                torch.stack(messages).mean(dim=0)
                if messages
                else x_dict[node_type]
            )
            for node_type, messages in incoming.items()
        }


class GNNFeatureEncoder(nn.Module):
    """Shared GNN encoder that produces a graph-level embedding."""

    def __init__(
        self,
        node_feature_dims: dict[str, int],
        edge_dim: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 3,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.node_types = list(node_feature_dims)
        self.device = torch.device(device)
        self.hidden_dim = hidden_dim
        self.input_projections = nn.ModuleDict(
            {
                node_type: nn.Linear(input_dim, hidden_dim)
                for node_type, input_dim in node_feature_dims.items()
            }
        )
        node_channels = dict.fromkeys(self.node_types, hidden_dim)
        self.layers = nn.ModuleList(
            HeteroInteractionLayer(node_channels, edge_dim, hidden_dim)
            for _ in range(num_layers)
        )

    def forward(self, data: HeteroData) -> torch.Tensor:
        batch_size = self._get_batch_size(data)
        x_dict: dict[str, torch.Tensor] = {}
        for node_type in self.node_types:
            if node_type in data.node_types:
                store = data[node_type]
                features = store.x
                if features.numel() > 0:
                    x_dict[node_type] = F.relu(
                        self.input_projections[node_type](features)
                    )
                    continue
                device = features.device
            else:
                device = self.device
            x_dict[node_type] = torch.zeros(
                (0, self.hidden_dim), device=device
            )

        edge_indices = {
            edge_type: data[edge_type].edge_index for edge_type in data.edge_types
        }
        edge_attributes = {
            edge_type: data[edge_type].edge_attr for edge_type in data.edge_types
        }
        for layer in self.layers:
            x_dict = layer(x_dict, edge_indices, edge_attributes)
            x_dict = {key: F.relu(value) for key, value in x_dict.items()}

        pooled_features = []
        for node_type in self.node_types:
            features = x_dict.get(node_type)
            if features is None or features.numel() == 0:
                device = features.device if features is not None else self.device
                pooled = torch.zeros(
                    (batch_size, self.hidden_dim), device=device
                )
            else:
                store = data[node_type]
                if hasattr(store, "batch"):
                    pooled = global_mean_pool(
                        features, store.batch, size=batch_size
                    )
                else:
                    pooled = features.mean(dim=0, keepdim=True)
                    if batch_size > 1:
                        pooled = pooled.repeat(batch_size, 1)
            pooled_features.append(pooled)
        return torch.cat(pooled_features, dim=-1)

    def _get_batch_size(self, data: HeteroData) -> int:
        if hasattr(data, "num_graphs"):
            return int(data.num_graphs)
        max_batch = -1
        for node_type in self.node_types:
            if node_type in data.node_types and hasattr(data[node_type], "batch"):
                batch = data[node_type].batch
                if batch.numel() > 0:
                    max_batch = max(max_batch, int(batch.max().item()))
        return max_batch + 1 if max_batch >= 0 else 1
