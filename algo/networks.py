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
        
        # Prepare scalar score for charging actions (fall back to zero if embedding missing)
        if graph_embedding is not None and graph_embedding.numel() > 0:
            raw_charge_score = self.charge_action_head(graph_embedding).squeeze()
            if raw_charge_score.ndim == 0:
                charge_action_value = raw_charge_score
            else:
                charge_action_value = raw_charge_score.reshape(-1)[0]
        else:
            charge_action_value = torch.tensor(0.0, device=self.device)
        
        # Build action scores based on action_to_node_map
        # Strategy: Compute scores for all nodes first, then index by action_to_node_map
        if hasattr(data, 'action_to_node_map'):
            # Compute scores for all nodes
            scores_dict = {}
            for node_type in x_dict.keys():
                scores_dict[node_type] = self.output_heads[node_type](x_dict[node_type]).squeeze(-1)
            
            # Build a global node indexing: node_id -> (node_type, local_idx)
            # This requires node_id_to_type mapping from data
            if hasattr(data, 'node_id_to_type'):
                num_actions = len(data.action_to_node_map)
                action_scores = torch.zeros(num_actions, device=self.device)
                
                for action_idx, (node_id, is_charging) in enumerate(data.action_to_node_map):
                    # Handle dummy nodes (e.g., node_id = -1 for infeasible actions)
                    if node_id == -1 or node_id not in data.node_id_to_type:
                        # Use a low score for dummy/infeasible nodes
                        action_scores[action_idx] = -1.0
                        continue
                    
                    # Get node type and local index for this node_id
                    node_type, local_idx = data.node_id_to_type[node_id]
                    
                    if is_charging:
                        # For charging action, use global charge head instead of node score
                        action_scores[action_idx] = charge_action_value
                    else:
                        # Routing action - use the node's score
                        action_scores[action_idx] = scores_dict[node_type][local_idx]
            else:
                # Fallback: Simple concatenation (assume delivery nodes come first, then chargers)
                # This won't distinguish routing vs charging properly but will work for initial testing
                scores_list = []
                for node_type in self.node_types:
                    if node_type in scores_dict:
                        scores_list.append(scores_dict[node_type])
                all_node_scores = torch.cat(scores_list, dim=0)
                
                # Map actions to nodes
                num_actions = len(data.action_to_node_map)
                action_scores = torch.zeros(num_actions, device=self.device)
                
                for action_idx, (node_id, is_charging) in enumerate(data.action_to_node_map):
                    if node_id == -1:
                        action_scores[action_idx] = -1.0
                        continue
                    
                    if is_charging:
                        # Use global charge score for charging actions
                        action_scores[action_idx] = charge_action_value
                    else:
                        # Use node score if in range
                        if node_id < len(all_node_scores):
                            action_scores[action_idx] = all_node_scores[node_id]
            
            # Apply masking if feasible_action_mask is provided AND apply_mask is True
            if apply_mask and hasattr(data, 'feasible_action_mask'):
                mask = data.feasible_action_mask
                action_scores = torch.where(mask, action_scores, torch.tensor(-1e9, device=self.device))
            
            # Apply tanh for bounded actions
            action_scores = torch.tanh(action_scores)
        else:
            # Fallback: old behavior with all nodes
            scores_dict = {
                node_type: self.output_heads[node_type](x_dict[node_type])
                for node_type in x_dict.keys()
            }
            
            scores_list = []
            for node_type in self.node_types:
                if node_type in scores_dict:
                    scores_list.append(scores_dict[node_type])
                    
            action_scores = torch.cat(scores_list, dim=0).squeeze(-1)
            
            # Apply masking if feasible_action_mask is provided AND apply_mask is True
            if apply_mask and hasattr(data, 'feasible_action_mask'):
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
        # action_indicator: 1 if this node is selected, 0 otherwise
        # charging_duration: global feature added to all nodes
        self.input_projections = nn.ModuleDict({
            node_type: nn.Linear(dim + 2, hidden_dim)  # +2 for action_indicator and charging_duration
            for node_type, dim in node_feature_dims.items()
        })
        
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
            action: Action tensor - [batch_size] of action indices
            charging_duration: Optional charging duration [batch_size, 1]
            
        Returns:
            Q-value estimate [batch_size, 1]
        """
        # Get batch information - use max batch index across all node types
        has_batch = hasattr(data['truck'], 'batch')
        
        if has_batch:
            # Find max batch index across all node types
            max_batch_idx = -1
            for node_type in self.node_types:
                if node_type in data.node_types and hasattr(data[node_type], 'batch'):
                    batch_tensor = data[node_type].batch
                    if batch_tensor.numel() > 0:
                        node_max = int(batch_tensor.max().item())
                        max_batch_idx = max(max_batch_idx, node_max)
            batch_size = max_batch_idx + 1 if max_batch_idx >= 0 else 1
        else:
            batch_size = 1
        
        # Handle action format - convert to tensor if needed
        if isinstance(action, int):
            action = torch.tensor([action], dtype=torch.long, device=self.device)
        elif not isinstance(action, torch.Tensor):
            action = torch.tensor(action, device=self.device)
        
        # Ensure action is the right shape [batch_size]
        if action.dim() == 0:
            action = action.unsqueeze(0)
        if action.shape[0] != batch_size:
            # If single action given, repeat for batch
            if action.shape[0] == 1:
                action = action.repeat(batch_size)
        
        # Normalize action indices to [0, 1] range for feature concatenation
        action_feature = action.float().unsqueeze(-1) / 100.0  # Rough normalization
        
        # Handle charging duration (default to 0 if not provided)
        if charging_duration is None:
            charging_duration = torch.zeros(batch_size, 1, device=self.device)
        elif charging_duration.dim() == 1:
            charging_duration = charging_duration.unsqueeze(-1)
        elif charging_duration.dim() == 0:
            charging_duration = charging_duration.unsqueeze(0).unsqueeze(0)
        
        # Ensure charging_duration is [batch_size, 1]
        if charging_duration.shape[0] != batch_size:
            if charging_duration.shape[0] == 1:
                charging_duration = charging_duration.repeat(batch_size, 1)
        
        # Replicate action and charging duration to all nodes in each graph
        x_dict = {}
        for node_type in self.node_types:
            if node_type not in data.node_types:
                continue
            
            num_nodes = data[node_type].x.shape[0]
            
            if has_batch:
                # Expand action to match number of nodes
                node_batch = data[node_type].batch
                # For each node, get its corresponding action and charging duration from the batch
                node_actions = action_feature[node_batch]
                node_charging = charging_duration[node_batch]
            else:
                # Single graph - repeat action and charging duration for all nodes
                node_actions = action_feature.repeat(num_nodes, 1)
                node_charging = charging_duration.repeat(num_nodes, 1)
            
            # Concatenate node features with action indicator and charging duration
            x_with_action = torch.cat([data[node_type].x, node_actions, node_charging], dim=-1)
            x_dict[node_type] = F.relu(self.input_projections[node_type](x_with_action))
            
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
            
        # Global pooling for each node type
        pooled_features = []
        for node_type in self.node_types:
            if node_type in x_dict:
                # Mean pooling over nodes of this type
                if has_batch:
                    batch_idx = data[node_type].batch
                    pooled = global_mean_pool(x_dict[node_type], batch_idx, size=batch_size)
                else:
                    batch_idx = torch.zeros(x_dict[node_type].shape[0], 
                                      dtype=torch.long, device=self.device)
                    pooled = global_mean_pool(x_dict[node_type], batch_idx, size=batch_size)
                pooled_features.append(pooled)
                
        # Concatenate pooled features from all node types
        graph_embedding = torch.cat(pooled_features, dim=-1)
        
        # Estimate Q-value
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
