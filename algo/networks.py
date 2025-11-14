"""
Neural Network architectures for TD3 Action-GNN agent.
Uses Heterogeneous Graph Neural Networks to handle different node types
(trucks, deliveries, chargers) and edge features (energy, time).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.data import HeteroData
from torch_geometric.utils import softmax


class MLP(nn.Module):
    """Multi-layer perceptron with LayerNorm."""
    
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, use_layer_norm=True):
        super().__init__()
        layers = []
        current_dim = in_dim
        
        for i in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim
            
        layers.append(nn.Linear(current_dim, out_dim))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)


class EdgeConditionedConv(MessagePassing):
    """
    Message passing layer that conditions on edge features.
    For heterogeneous graphs: takes (x_src, x_dst) and edge features.
    Message from node j to node i: MLP(concat(h_i, h_j, edge_attr))
    """
    
    def __init__(self, in_channels, out_channels, edge_dim, aggr='add'):
        super().__init__(aggr=aggr)
        
        # MLP for message computation: takes [h_i, h_j, edge_attr]
        self.message_mlp = MLP(
            in_dim=2 * in_channels + edge_dim,
            hidden_dim=out_channels,
            out_dim=out_channels,
            num_layers=2
        )
        
        # Update MLP: takes [h_i, aggregated_messages]
        self.update_mlp = MLP(
            in_dim=in_channels + out_channels,
            hidden_dim=out_channels,
            out_dim=out_channels,
            num_layers=1
        )
        
    def forward(self, x, edge_index, edge_attr, size=None):
        """
        Args:
            x: Tuple of (x_src, x_dst) for heterogeneous, or single tensor for homogeneous
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
            size: Tuple (num_src_nodes, num_dst_nodes) for heterogeneous graphs
        """
        # Handle both homogeneous and heterogeneous cases
        if isinstance(x, tuple):
            x_src, x_dst = x
        else:
            x_src = x_dst = x
            
        return self.propagate(edge_index, x=(x_src, x_dst), edge_attr=edge_attr, size=size)
    
    def message(self, x_i, x_j, edge_attr):
        """Compute messages from j to i."""
        # Concatenate: target node (i), source node (j), edge features
        msg_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.message_mlp(msg_input)
    
    def update(self, aggr_out, x):
        """Update node features."""
        # x here is x_dst (target nodes)
        if isinstance(x, tuple):
            x = x[1]  # Get destination features
        update_input = torch.cat([x, aggr_out], dim=-1)
        return self.update_mlp(update_input)


class HeteroInteractionLayer(nn.Module):
    """
    One layer of heterogeneous graph interaction.
    Processes different edge types with separate EdgeConditionedConv layers.
    """
    
    def __init__(self, node_channels_dict, edge_dim, hidden_dim):
        """
        Args:
            node_channels_dict: Dict mapping node type -> input feature dim
                                e.g., {'truck': 64, 'delivery': 64, 'charger': 64}
            edge_dim: Dimension of edge features (2 for energy, time)
            hidden_dim: Output dimension for all node types
        """
        super().__init__()
        
        self.node_types = list(node_channels_dict.keys())
        self.convs = nn.ModuleDict()
        
        # Define edge types (bidirectional)
        self.edge_types = [
            ('truck', 'to', 'delivery'),
            ('delivery', 'to', 'truck'),
            ('truck', 'to', 'charger'),
            ('charger', 'to', 'truck'),
            ('charger', 'to', 'charger'),
            ('truck', 'to', 'truck'),
            ('delivery', 'to', 'delivery'),
            ('charger', 'to', 'delivery'),
        ]
        
        # Create conv layer for each edge type
        for src, rel, dst in self.edge_types:
            in_channels = node_channels_dict[src]
            conv = EdgeConditionedConv(in_channels, hidden_dim, edge_dim)
            self.convs[f'{src}__{rel}__{dst}'] = conv
            
    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        """
        Args:
            x_dict: Dict mapping node type -> features
            edge_index_dict: Dict mapping edge type -> edge_index
            edge_attr_dict: Dict mapping edge type -> edge_attr
            
        Returns:
            Updated x_dict with same keys
        """
        out_dict = {node_type: [] for node_type in self.node_types}
        
        # Apply convolutions for each edge type
        for edge_type in self.edge_types:
            src, rel, dst = edge_type
            conv_key = f'{src}__{rel}__{dst}'
            
            if edge_type not in edge_index_dict:
                continue
                
            edge_index = edge_index_dict[edge_type]
            edge_attr = edge_attr_dict[edge_type]
            
            # Skip empty edge types
            if edge_index.shape[1] == 0:
                continue
            
            # Get source and destination node features
            x_src = x_dict[src]
            x_dst = x_dict[dst]
            
            # Apply edge-conditioned convolution with both src and dst features
            # Pass size to specify (num_src_nodes, num_dst_nodes)
            size = (x_src.shape[0], x_dst.shape[0])
            out = self.convs[conv_key]((x_src, x_dst), edge_index, edge_attr, size=size)
            
            # Messages target the destination node type
            out_dict[dst].append(out)
        
        # Aggregate messages for each node type (mean aggregation)
        x_dict_out = {}
        for node_type in self.node_types:
            if len(out_dict[node_type]) > 0:
                # Average all incoming messages
                x_dict_out[node_type] = torch.stack(out_dict[node_type]).mean(dim=0)
            else:
                # No incoming messages, keep original features (projected to hidden_dim)
                x_dict_out[node_type] = x_dict[node_type]
                
        return x_dict_out


class HeteroGNN_Actor(nn.Module):
    """
    Heterogeneous GNN Actor for action selection.
    Outputs:
    - Node selection scores (discrete action over nodes)
    - Charging duration (continuous action, used only when charger selected)
    """
    
    def __init__(self, 
                 node_feature_dims={'truck': 13, 'delivery': 2, 'charger': 5},
                 edge_dim=2,
                 hidden_dim=64,
                 num_layers=3,
                 min_charging_duration=0.5,
                 max_charging_duration=10.0,
                 device='cpu'):
        super().__init__()
        
        self.device = device
        self.node_types = list(node_feature_dims.keys())
        self.min_charging_duration = min_charging_duration
        self.max_charging_duration = max_charging_duration
        self.node_type_to_code = {
            node_type: idx for idx, node_type in enumerate(self.node_types)
        }
        
        # Input projection for each node type (to hidden_dim)
        self.input_projections = nn.ModuleDict({
            node_type: nn.Linear(dim, hidden_dim)
            for node_type, dim in node_feature_dims.items()
        })
        
        # Heterogeneous interaction layers
        self.layers = nn.ModuleList()
        node_channels = {nt: hidden_dim for nt in self.node_types}
        
        for _ in range(num_layers):
            self.layers.append(
                HeteroInteractionLayer(node_channels, edge_dim, hidden_dim)
            )
            
        # Output heads for each node type (score per node)
        self.output_heads = nn.ModuleDict({
            node_type: nn.Linear(hidden_dim, 1)
            for node_type in self.node_types
        })
        
        # Charging duration head (continuous action)
        # Input: pooled graph features
        self.charging_duration_head = nn.Sequential(
            nn.Linear(hidden_dim * len(self.node_types), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Output in [0, 1], will scale to [0, max_charging_duration]
        )
        # Charging action head uses global context (routing heads operate per-node)
        self.charge_action_head = nn.Sequential(
            nn.Linear(hidden_dim * len(self.node_types), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
    def forward(self, data, return_mapper=False, apply_mask=True):
        """
        Args:
            data: HeteroData object
            return_mapper: Compatibility flag
            apply_mask: If True, apply feasibility mask to action scores
            
        Returns:
            Tuple of (action_scores, charging_duration)
            - action_scores: Scores for discrete actions [next_delivery, charger_0, ..., charger_N, charge_here]
            - charging_duration: Continuous value in [0, max_charging_duration]
        """
        # Input projection
        x_dict = {
            node_type: F.relu(self.input_projections[node_type](data[node_type].x))
            for node_type in self.node_types
            if node_type in data.node_types
        }
        
        # Extract edge indices and attributes
        edge_index_dict = {
            edge_type: data[edge_type].edge_index
            for edge_type in data.edge_types
        }
        
        edge_attr_dict = {
            edge_type: data[edge_type].edge_attr
            for edge_type in data.edge_types
        }
        
        # Apply heterogeneous layers
        for layer in self.layers:
            x_dict = layer(x_dict, edge_index_dict, edge_attr_dict)
            x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        
        # Compute charging duration (global graph feature)
        # Pool features from all node types
        # Determine the batch size from the batch attribute - use max across all node types
        batch_size = None
        if hasattr(data, 'batch'):
            # Single graph case
            batch_size = 1
        else:
            # Batched graphs - determine batch size from all node types
            max_batch_idx = -1
            for node_type in self.node_types:
                if node_type in x_dict and hasattr(data[node_type], 'batch'):
                    batch_tensor = data[node_type].batch
                    # Only compute max if batch tensor is non-empty
                    if batch_tensor.numel() > 0:
                        node_max = int(batch_tensor.max().item())
                        max_batch_idx = max(max_batch_idx, node_max)
            
            if max_batch_idx >= 0:
                batch_size = max_batch_idx + 1
        
        if batch_size is None:
            batch_size = 1
        
        pooled_features = []
        for node_type in self.node_types:
            if node_type in x_dict:
                # Mean pooling over nodes of this type
                if hasattr(data[node_type], 'batch'):
                    batch = data[node_type].batch
                    # Only pool if there are nodes of this type
                    if x_dict[node_type].shape[0] > 0:
                        pooled = global_mean_pool(x_dict[node_type], batch, size=batch_size)
                    else:
                        # No nodes of this type - create zero pooling
                        pooled = torch.zeros(batch_size, x_dict[node_type].shape[1], device=self.device)
                else:
                    # Single graph - just mean over all nodes
                    if x_dict[node_type].shape[0] > 0:
                        pooled = x_dict[node_type].mean(dim=0, keepdim=True)
                    else:
                        pooled = torch.zeros(1, x_dict[node_type].shape[1], device=self.device)
                    # Ensure correct batch size
                    if pooled.shape[0] < batch_size:
                        pooled = pooled.repeat(batch_size, 1)
                pooled_features.append(pooled)
        
        graph_embedding = None
        if pooled_features:
            graph_embedding = torch.cat(pooled_features, dim=-1)
            # Predict charging duration: map sigmoid [0,1] to [min_charging_duration, max_charging_duration]
            sigmoid_output = self.charging_duration_head(graph_embedding)
            charging_duration = self.min_charging_duration + sigmoid_output * (self.max_charging_duration - self.min_charging_duration)
        else:
            charging_duration = torch.full((batch_size, 1), self.min_charging_duration, device=self.device)
        
        # Prepare score for charging actions per graph (fall back to zeros if embedding missing)
        if graph_embedding is not None and graph_embedding.numel() > 0:
            raw_charge_score = self.charge_action_head(graph_embedding).squeeze(-1)
            charge_action_values = raw_charge_score
        else:
            charge_action_values = torch.zeros(batch_size, device=self.device)
        
        if not (hasattr(data, 'action_node_type') and hasattr(data, 'action_local_index') and hasattr(data, 'num_actions') and hasattr(data, 'action_is_charging') and hasattr(data, 'feasible_action_mask')):
            raise ValueError("State missing action metadata required by actor")

        scores_dict = {
            node_type: self.output_heads[node_type](x_dict[node_type]).squeeze(-1)
            for node_type in x_dict.keys()
        }

        num_actions_tensor = data.num_actions
        if num_actions_tensor.dim() == 0:
            num_actions_tensor = num_actions_tensor.unsqueeze(0)
        num_actions_tensor = num_actions_tensor.to(torch.long)
        num_graphs = num_actions_tensor.shape[0]
        action_counts = num_actions_tensor
        total_actions = int(action_counts.sum().item())
        if total_actions == 0:
            return torch.empty((0,), device=self.device), charging_duration
        action_scores = torch.full((total_actions,), -1e9, device=self.device)

        action_node_type = data.action_node_type.to(torch.long)
        action_local_index = data.action_local_index.to(torch.long)
        action_graph_ids = torch.repeat_interleave(
            torch.arange(num_graphs, device=self.device), action_counts
        ) if num_graphs > 0 else None

        valid_nodes = action_local_index >= 0
        for node_type, scores in scores_dict.items():
            node_code = self.node_type_to_code.get(node_type, None)
            if node_code is None:
                continue
            mask = (action_node_type == node_code) & valid_nodes
            if mask.any():
                local_indices = action_local_index[mask]
                action_scores[mask] = scores[local_indices]

        charge_mask = data.action_is_charging
        if action_graph_ids is not None and charge_mask.numel() == action_scores.numel() and charge_action_values.numel() > 0:
            action_scores[charge_mask] = charge_action_values[action_graph_ids[charge_mask]]

        if apply_mask:
            mask = data.feasible_action_mask
            action_scores = torch.where(mask, action_scores, torch.tensor(-1e9, device=self.device))

        action_scores = torch.tanh(action_scores)

        if return_mapper:
            return (action_scores, charging_duration), None, torch.arange(len(action_scores), device=self.device)
        return action_scores, charging_duration


class HeteroGNN_Critic(nn.Module):
    """
    Heterogeneous GNN Critic for Q-value estimation.
    Takes state and action (node_index + charging_duration), outputs Q-value.
    """
    
    def __init__(self,
                 node_feature_dims={'truck': 13, 'delivery': 2, 'charger': 5},
                 edge_dim=2,
                 hidden_dim=64,
                 num_layers=3,
                 mlp_hidden_dim=256,
                 device='cpu'):
        super().__init__()
        
        self.device = device
        self.node_types = list(node_feature_dims.keys())
        
        # Input projection for each node type (feature + action_indicator + charging_duration)
        # Additional scalars:
        #  - action_indicator: 1 if node selected, else 0
        #  - charging_duration: duration suggested/executed for action
        #  - is_charging flag: indicates charging vs routing action
        self.input_projections = nn.ModuleDict({
            node_type: nn.Linear(dim + 3, hidden_dim)
            for node_type, dim in node_feature_dims.items()
        })
        self.node_type_to_code = {
            node_type: idx for idx, node_type in enumerate(self.node_types)
        }
        
        # Heterogeneous interaction layers
        self.layers = nn.ModuleList()
        node_channels = {nt: hidden_dim for nt in self.node_types}
        
        for _ in range(num_layers):
            self.layers.append(
                HeteroInteractionLayer(node_channels, edge_dim, hidden_dim)
            )
            
        # MLP for Q-value estimation from graph embedding
        self.q_net = MLP(
            in_dim=hidden_dim * len(self.node_types),  # Pooled features from all types
            hidden_dim=mlp_hidden_dim,
            out_dim=1,
            num_layers=3
        )
        
    def forward(self, data, action, charging_duration=None):
        """
        Args:
            data: HeteroData object (possibly batched)
            action: Action tensor - [batch_size] of action indices (per-graph)
            charging_duration: Optional charging duration [batch_size, 1]
            
        Returns:
            Q-value estimate [batch_size, 1]
        """
        if not hasattr(data, 'num_actions'):
            raise ValueError("Missing num_actions metadata in state for critic forward pass")
        if not (hasattr(data, 'action_node_type') and hasattr(data, 'action_local_index') and hasattr(data, 'action_is_charging')):
            raise ValueError("Missing action metadata tensors required by the critic")

        num_actions_tensor = data.num_actions
        if isinstance(num_actions_tensor, torch.Tensor) and num_actions_tensor.dim() == 0:
            num_actions_tensor = num_actions_tensor.unsqueeze(0)
        num_actions_tensor = num_actions_tensor.to(torch.long)
        batch_size = num_actions_tensor.shape[0]

        if isinstance(action, int):
            action = torch.tensor([action], dtype=torch.long, device=self.device)
        elif not isinstance(action, torch.Tensor):
            action = torch.tensor(action, device=self.device)
        action = action.long()
        if action.dim() == 0:
            action = action.unsqueeze(0)
        if action.shape[0] != batch_size:
            if action.shape[0] == 1:
                action = action.repeat(batch_size)

        if charging_duration is None:
            charging_duration = torch.zeros(batch_size, 1, device=self.device)
        elif charging_duration.dim() == 1:
            charging_duration = charging_duration.unsqueeze(-1)
        elif charging_duration.dim() == 0:
            charging_duration = charging_duration.unsqueeze(0).unsqueeze(0)
        if charging_duration.shape[0] != batch_size:
            if charging_duration.shape[0] == 1:
                charging_duration = charging_duration.repeat(batch_size, 1)

        graph_has_actions = num_actions_tensor > 0
        safe_counts = torch.where(graph_has_actions, num_actions_tensor, torch.ones_like(num_actions_tensor))
        clamped_action = torch.minimum(action, safe_counts - 1).clamp(min=0)
        action_offsets = torch.cumsum(
            torch.cat([
                num_actions_tensor.new_zeros(1),
                num_actions_tensor[:-1]
            ]), dim=0
        )
        global_action_idx = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        if graph_has_actions.any():
            valid_idx = torch.nonzero(graph_has_actions, as_tuple=False).squeeze(-1)
            global_action_idx[valid_idx] = action_offsets[valid_idx] + clamped_action[valid_idx]
        total_actions = int(num_actions_tensor.sum().item())
        if total_actions == 0:
            raise ValueError("Sampled state with zero available actions")

        action_node_type = data.action_node_type.to(torch.long)
        action_local_index = data.action_local_index.to(torch.long)
        action_is_charging = data.action_is_charging.float().unsqueeze(-1)
        selected_node_types = torch.full((batch_size,), -1, dtype=torch.long, device=self.device)
        selected_local_idx = torch.full((batch_size,), -1, dtype=torch.long, device=self.device)
        selected_charge_flags = torch.zeros((batch_size, 1), dtype=torch.float32, device=self.device)
        if graph_has_actions.any():
            valid_idx = torch.nonzero(graph_has_actions, as_tuple=False).squeeze(-1)
            valid_global = global_action_idx[valid_idx]
            selected_node_types[valid_idx] = action_node_type[valid_global]
            selected_local_idx[valid_idx] = action_local_index[valid_global]
            selected_charge_flags[valid_idx] = action_is_charging[valid_global]
        graph_indices = torch.arange(batch_size, device=self.device)

        def _build_ptr(store):
            if hasattr(store, 'ptr') and store.ptr is not None:
                return store.ptr.to(torch.long).to(self.device)
            if hasattr(store, 'batch'):
                counts = torch.bincount(store.batch, minlength=batch_size)
                ptr = torch.zeros(batch_size + 1, dtype=torch.long, device=self.device)
                ptr[1:] = torch.cumsum(counts, dim=0)
                return ptr
            num_nodes = store.x.shape[0]
            ptr = torch.zeros(batch_size + 1, dtype=torch.long, device=self.device)
            ptr[1] = num_nodes
            return ptr

        def _node_batch_index(store, num_nodes):
            if hasattr(store, 'batch'):
                return store.batch.to(torch.long)
            return torch.zeros(num_nodes, dtype=torch.long, device=self.device)

        x_dict = {}
        for node_type in self.node_types:
            if node_type not in data.node_types:
                continue

            node_store = data[node_type]
            num_nodes = node_store.x.shape[0]
            if num_nodes == 0:
                x_dict[node_type] = torch.empty((0, self.input_projections[node_type].out_features), device=self.device)
                continue

            node_indicator = torch.zeros((num_nodes, 1), device=self.device)
            node_code = self.node_type_to_code.get(node_type, -1)
            mask = (selected_node_types == node_code) & (selected_local_idx >= 0)
            if mask.any():
                ptr = _build_ptr(node_store)
                node_counts = ptr[1:] - ptr[:-1]
                selected_graphs = graph_indices[mask]
                local_positions = selected_local_idx[mask]
                if selected_graphs.numel() > 0:
                    valid = node_counts[selected_graphs] > local_positions
                    if valid.any():
                        selected_graphs = selected_graphs[valid]
                        local_positions = local_positions[valid]
                        global_node_idx = ptr[selected_graphs] + local_positions
                        node_indicator[global_node_idx] = 1.0

            node_batch = _node_batch_index(node_store, num_nodes)
            node_charging = charging_duration[node_batch]
            node_is_charging = selected_charge_flags[node_batch]
            x_with_action = torch.cat([
                node_store.x,
                node_indicator,
                node_charging,
                node_is_charging
            ], dim=-1)
            x_dict[node_type] = F.relu(self.input_projections[node_type](x_with_action))

        edge_index_dict = {
            edge_type: data[edge_type].edge_index
            for edge_type in data.edge_types
        }

        edge_attr_dict = {
            edge_type: data[edge_type].edge_attr
            for edge_type in data.edge_types
        }

        for layer in self.layers:
            x_dict = layer(x_dict, edge_index_dict, edge_attr_dict)
            x_dict = {k: F.relu(v) for k, v in x_dict.items()}

        pooled_features = []
        for node_type in self.node_types:
            if node_type in x_dict:
                if hasattr(data[node_type], 'batch'):
                    batch_idx = data[node_type].batch
                    pooled = global_mean_pool(x_dict[node_type], batch_idx, size=batch_size)
                else:
                    batch_idx = torch.zeros(x_dict[node_type].shape[0], dtype=torch.long, device=self.device)
                    pooled = global_mean_pool(x_dict[node_type], batch_idx, size=batch_size)
                pooled_features.append(pooled)

        graph_embedding = torch.cat(pooled_features, dim=-1)
        q_value = self.q_net(graph_embedding)
        return q_value


class Actor(nn.Module):
    """Wrapper for HeteroGNN_Actor to maintain compatibility with TD3."""
    
    def __init__(self, max_action=1.0, device='cpu', **kwargs):
        super().__init__()
        self.actor = HeteroGNN_Actor(device=device, **kwargs)
        self.max_action = max_action
        self.device = device
        
    def forward(self, state, return_mapper=False, apply_mask=True):
        scores, charging_duration = self.actor(state, return_mapper=return_mapper, apply_mask=apply_mask)
        if return_mapper:
            (scores, charging_duration), _, mapper = (scores, charging_duration), None, torch.arange(len(scores), device=self.device)
            return (self.max_action * scores, charging_duration), None, mapper
        return self.max_action * scores, charging_duration


class Critic(nn.Module):
    """Twin critics for TD3."""
    
    def __init__(self, device='cpu', **kwargs):
        super().__init__()
        self.q1 = HeteroGNN_Critic(device=device, **kwargs)
        self.q2 = HeteroGNN_Critic(device=device, **kwargs)
        self.device = device
        
    def forward(self, state, action, charging_duration=None):
        return self.q1(state, action, charging_duration), self.q2(state, action, charging_duration)
    
    def Q1(self, state, action, charging_duration=None):
        return self.q1(state, action, charging_duration)
