"""
Neural Network architectures for TD3 Action-GNN agent.
Includes Actor and Critic networks using Graph Convolutional Networks.
Works with PyTorch Geometric Data objects from GNNStateSpace.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data


class Actor(nn.Module):
    """
    Actor network using GCN for graph-based state representation.
    Outputs action logits for each node (action is selecting a node).
    """
    
    def __init__(self,
                 max_action: float,
                 feature_dim: int = 8,
                 GNN_hidden_dim: int = 32,
                 num_gcn_layers: int = 3,
                 discrete_actions: int = 1,
                 device: torch.device = torch.device('cpu'),
                 **kwargs):  # Accept fx_node_sizes for compatibility but don't use it
        """
        Initialize Actor network.
        
        Args:
            max_action: Maximum action value
            feature_dim: Dimension of node feature embeddings
            GNN_hidden_dim: Hidden dimension for GNN layers
            num_gcn_layers: Number of GCN layers (3, 4, 5, or 6)
            discrete_actions: Number of discrete actions
            device: Device to run network on
        """
        super(Actor, self).__init__()

        self.device = device
        self.feature_dim = feature_dim
        self.discrete_actions = discrete_actions
        self.num_gcn_layers = num_gcn_layers

        # Input embedding layer (input features are variable but we'll use max)
        # Max features = 13 (truck nodes have most features)
        self.input_embedding = nn.Linear(13, feature_dim)

        # GCN layers to extract features
        self.gcn_conv = GCNConv(feature_dim, GNN_hidden_dim)
        
        if num_gcn_layers == 3:
            self.gcn_layers = nn.ModuleList([
                GCNConv(GNN_hidden_dim, feature_dim)
            ])
        elif num_gcn_layers == 4:
            self.gcn_layers = nn.ModuleList([
                GCNConv(GNN_hidden_dim, 2*GNN_hidden_dim),
                GCNConv(2*GNN_hidden_dim, feature_dim)
            ])
        elif num_gcn_layers == 5:
            self.gcn_layers = nn.ModuleList([
                GCNConv(GNN_hidden_dim, 2*GNN_hidden_dim),
                GCNConv(2*GNN_hidden_dim, GNN_hidden_dim),
                GCNConv(GNN_hidden_dim, feature_dim)
            ])
        elif num_gcn_layers == 6:
            self.gcn_layers = nn.ModuleList([
                GCNConv(GNN_hidden_dim, 2*GNN_hidden_dim),
                GCNConv(2*GNN_hidden_dim, 3*GNN_hidden_dim),
                GCNConv(3*GNN_hidden_dim, 2*GNN_hidden_dim),
                GCNConv(2*GNN_hidden_dim, feature_dim)
            ])
        else:
            raise ValueError(
                f"Number of Actor GCN layers not supported, use 3, 4, 5, or 6!")

        self.gcn_last = GCNConv(feature_dim, 1)  # Output score for each node
        self.max_action = max_action

    def forward(self, state: Data, return_mapper: bool = False):
        """
        Forward pass through the actor network.
        
        Args:
            state: PyTorch Geometric Data object with graph structure
            return_mapper: Whether to return action mapper (for compatibility)
            
        Returns:
            Action values (logits for each node)
        """
        x = state.x
        edge_index = state.edge_index
        
        # Embed input features
        embedded_x = self.input_embedding(x)
        embedded_x = F.relu(embedded_x)

        # Apply GCN layers
        x = self.gcn_conv(embedded_x, edge_index)
        x = F.relu(x)

        for layer in self.gcn_layers:
            x = layer(x, edge_index)
            x = F.relu(x)

        x = self.gcn_last(x, edge_index)

        # Bound output to action space
        x = self.max_action * torch.tanh(x)

        # Flatten to get action scores for each node
        x = x.reshape(-1)
        
        if return_mapper:
            # For compatibility with old interface
            # Return dummy values for valid_action_indexes and ev_indexes
            return x, None, torch.arange(len(x), device=x.device)
        else:
            return x


class Critic_GNN(nn.Module):
    """
    Critic network using GCN for graph-based state-action value estimation.
    """
    
    def __init__(self,
                 feature_dim: int = 8,
                 GNN_hidden_dim: int = 32,
                 mlp_hidden_dim: int = 256,
                 discrete_actions: int = 1,
                 num_gcn_layers: int = 3,
                 device: torch.device = torch.device('cpu'),
                 **kwargs):  # Accept fx_node_sizes for compatibility
        """
        Initialize Critic network.
        
        Args:
            feature_dim: Dimension of node feature embeddings
            GNN_hidden_dim: Hidden dimension for GNN layers
            mlp_hidden_dim: Hidden dimension for MLP layers
            discrete_actions: Number of discrete actions
            num_gcn_layers: Number of GCN layers (3, 4, or 5)
            device: Device to run network on
        """
        super(Critic_GNN, self).__init__()

        self.device = device
        self.feature_dim = feature_dim
        self.discrete_actions = discrete_actions
        
        # Input embedding (13 max features + 1 for action)
        self.input_embedding = nn.Linear(14, feature_dim)

        # GCN layers (state + action concatenated at input level)
        self.gcn_conv = GCNConv(feature_dim, GNN_hidden_dim)
        
        if num_gcn_layers == 3:
            self.gcn_layers = nn.ModuleList([
                GCNConv(GNN_hidden_dim, 2*GNN_hidden_dim),
                GCNConv(2*GNN_hidden_dim, 3*GNN_hidden_dim)
            ])
            mlp_layer_features = 3*GNN_hidden_dim
            
        elif num_gcn_layers == 4:
            self.gcn_layers = nn.ModuleList([
                GCNConv(GNN_hidden_dim, 2*GNN_hidden_dim),
                GCNConv(2*GNN_hidden_dim, 3*GNN_hidden_dim),
                GCNConv(3*GNN_hidden_dim, 2*GNN_hidden_dim)
            ])            
            mlp_layer_features = 2*GNN_hidden_dim
            
        elif num_gcn_layers == 5:
            self.gcn_layers = nn.ModuleList([
                GCNConv(GNN_hidden_dim, 2*GNN_hidden_dim),
                GCNConv(2*GNN_hidden_dim, 3*GNN_hidden_dim),
                GCNConv(3*GNN_hidden_dim, 4*GNN_hidden_dim),
                GCNConv(4*GNN_hidden_dim, 3*GNN_hidden_dim)
            ])
            mlp_layer_features = 3*GNN_hidden_dim
            
        else:
            raise ValueError(
                f"Number of Critic GCN layers not supported, use 3, 4, or 5!")

        # MLP for Q-value estimation
        self.l1 = nn.Linear(mlp_layer_features, mlp_hidden_dim)
        self.l2 = nn.Linear(mlp_hidden_dim, mlp_hidden_dim)
        self.l3 = nn.Linear(mlp_hidden_dim, 1)

    def forward(self, state: Data, action: torch.Tensor):
        """
        Forward pass through critic network.
        
        Args:
            state: PyTorch Geometric Data object with graph structure
            action: Action tensor (node action scores)
            
        Returns:
            Q-value estimate
        """
        x = state.x
        edge_index = state.edge_index

        # Concatenate action to state features
        # Action should have same number of nodes
        if action.dim() == 1:
            action = action.unsqueeze(1)
        
        # Pad action to match number of nodes if needed
        if action.shape[0] != x.shape[0]:
            # Repeat action for all nodes (broadcast)
            action = action.repeat(x.shape[0], 1)[:x.shape[0], :]
        
        state_action = torch.cat([x, action], dim=1)

        # Embed concatenated features
        embedded_x = self.input_embedding(state_action)
        embedded_x = F.relu(embedded_x)

        # Apply GCN layers
        x = self.gcn_conv(embedded_x, edge_index)
        x = F.relu(x)

        for layer in self.gcn_layers:
            x = layer(x, edge_index)
            x = F.relu(x)

        # Create batch mask for pooling
        if hasattr(state, 'batch'):
            batch = state.batch
        else:
            # Single graph - all nodes belong to batch 0
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=self.device)

        # Graph pooling
        pooled_embedding = global_mean_pool(x, batch=batch)

        # MLP for Q-value
        x = F.relu(self.l1(pooled_embedding))
        x = F.relu(self.l2(x))
        x = self.l3(x)

        return x


class Critic(nn.Module):
    """
    Twin Critic networks for TD3 (Q1 and Q2).
    """
    
    def __init__(self,
                 feature_dim: int = 8,
                 GNN_hidden_dim: int = 32,
                 mlp_hidden_dim: int = 256,
                 discrete_actions: int = 1,
                 num_gcn_layers: int = 3,
                 device: torch.device = torch.device('cpu'),
                 **kwargs):
        """
        Initialize twin critic networks.
        
        Args:
            feature_dim: Dimension of node feature embeddings
            GNN_hidden_dim: Hidden dimension for GNN layers
            mlp_hidden_dim: Hidden dimension for MLP layers
            discrete_actions: Number of discrete actions
            num_gcn_layers: Number of GCN layers
            device: Device to run network on
        """
        super(Critic, self).__init__()

        self.device = device
        self.feature_dim = feature_dim

        self.q1 = Critic_GNN(
            feature_dim=feature_dim,
            GNN_hidden_dim=GNN_hidden_dim,
            mlp_hidden_dim=mlp_hidden_dim,
            discrete_actions=discrete_actions,
            num_gcn_layers=num_gcn_layers,
            device=device
        )

        self.q2 = Critic_GNN(
            feature_dim=feature_dim,
            GNN_hidden_dim=GNN_hidden_dim,
            mlp_hidden_dim=mlp_hidden_dim,
            discrete_actions=discrete_actions,
            num_gcn_layers=num_gcn_layers,
            device=device
        )

    def forward(self, state: Data, action: torch.Tensor):
        """Forward pass through both critics."""
        return self.q1(state, action), self.q2(state, action)

    def Q1(self, state: Data, action: torch.Tensor):
        """Forward pass through first critic only."""
        return self.q1(state, action)
