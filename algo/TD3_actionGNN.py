"""
TD3 Action-GNN Algorithm.
Twin Delayed Deep Deterministic Policy Gradient with Graph Neural Networks.
"""

import copy
import numpy as np
import torch
import torch.nn.functional as F

from algo.networks import Actor, Critic


class TD3_ActionGNN(object):
    """
    TD3 (Twin Delayed Deep Deterministic Policy Gradient) algorithm with GNN-based actor and critic.
    
    Key features:
    - Twin critics to reduce overestimation bias
    - Delayed policy updates
    - Target policy smoothing
    - Graph Neural Networks for structured state representation
    """
    
    def __init__(
            self,
            action_dim,
            max_action,
            fx_node_sizes,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            fx_dim=8,
            fx_GNN_hidden_dim=32,
            mlp_hidden_dim=256,
            lr=3e-4,
            discrete_actions=1,
            actor_num_gcn_layers=3,
            critic_num_gcn_layers=3,
            **kwargs
    ):
        """
        Initialize TD3 agent.
        
        Args:
            action_dim: Dimension of action space
            max_action: Maximum action value
            fx_node_sizes: Dictionary of feature sizes for different node types
            discount: Discount factor (gamma)
            tau: Target network update rate
            policy_noise: Std of Gaussian noise for target policy smoothing
            noise_clip: Range to clip target policy noise
            policy_freq: Frequency of delayed policy updates
            fx_dim: Feature dimension for node embeddings
            fx_GNN_hidden_dim: Hidden dimension for GNN layers
            mlp_hidden_dim: Hidden dimension for MLP layers in critic
            lr: Learning rate for both actor and critic
            discrete_actions: Number of discrete actions
            actor_num_gcn_layers: Number of GCN layers in actor
            critic_num_gcn_layers: Number of GCN layers in critic
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.discrete_actions = discrete_actions

        # Initialize actor network
        self.actor = Actor(
            max_action,
            hidden_dim=fx_GNN_hidden_dim,
            num_layers=actor_num_gcn_layers,
            device=self.device
        ).to(self.device)

        self.actor_target = copy.deepcopy(self.actor).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        # Initialize critic network
        self.critic = Critic(
            hidden_dim=fx_GNN_hidden_dim,
            mlp_hidden_dim=mlp_hidden_dim,
            num_layers=critic_num_gcn_layers,
            device=self.device
        ).to(self.device)

        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # Hyperparameters
        self.action_dim = action_dim
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

    def select_action(self, state, expl_noise=0, **kwargs):
        """
        Select action using current policy with softmax over feasible actions.
        
        Args:
            state: GNN state object (HeteroData with feasible_action_mask and action_to_node_map)
            expl_noise: Exploration noise (temperature for softmax)
            
        Returns:
            Tuple of (node_id, charging_duration, is_charging_action) or (action_idx, charging_duration) if no map
        """
        state = state.to(self.device)

        with torch.no_grad():
            # Apply mask during exploration
            action_logits, charging_duration = self.actor(state, apply_mask=True)

        # Apply softmax over feasible actions only
        if hasattr(state, 'feasible_action_mask'):
            mask = state.feasible_action_mask
            
            # Check if there are any feasible actions
            if not mask.any():
                # No feasible actions - return first action as fallback
                print("Warning: No feasible actions found in select_action")
                raise ValueError("No feasible actions found in select_action")
                return 0, 0.0, False
            
            # Set infeasible actions to -inf before softmax
            masked_logits = torch.where(
                mask,
                action_logits,
                torch.tensor(float('-inf'), device=self.device)
            )
            
            # Softmax with temperature (expl_noise acts as inverse temperature)
            if expl_noise > 0:
                temperature = 1.0 + expl_noise
                probs = F.softmax(masked_logits / temperature, dim=0)
                # Sample from distribution
                action_idx = torch.multinomial(probs, 1).item()
            else:
                # Greedy selection
                action_idx = torch.argmax(masked_logits).item()
        else:
            # Fallback: no masking
            if expl_noise > 0:
                noise = torch.randn_like(action_logits) * expl_noise
                action_logits = action_logits + noise
            action_idx = torch.argmax(action_logits).item()
        
        # Extract charging duration (scalar value)
        charge_dur = charging_duration.squeeze().item()
        
        # If action_to_node_map exists, translate action_idx to (node_id, is_charging_action)
        if hasattr(state, 'action_to_node_map'):
            if action_idx < len(state.action_to_node_map):
                node_id, is_charging = state.action_to_node_map[action_idx]
                return node_id, charge_dur, is_charging
            else:
                # Fallback if index out of range
                print(f"Warning: action_idx {action_idx} out of range for action_to_node_map (size {len(state.action_to_node_map)})")
                return action_idx, charge_dur, False
        
        # Legacy return format (action_idx, charging_duration)
        return action_idx, charge_dur

    def train(self, replay_buffer, batch_size=256):
        """
        Train actor and critic networks using replay buffer.
        Handles variable-sized actions per graph in batch.
        
        Args:
            replay_buffer: Replay buffer with experience tuples
            batch_size: Batch size for training
            
        Returns:
            Tuple of (critic_loss, actor_loss) or (critic_loss, None) if not updating actor
        """
        self.total_it += 1

        # Sample replay buffer
        # Now returns: (state, action_idx, charging_duration, next_state, reward, not_done)
        state, action, charging_duration, next_state, reward, not_done = replay_buffer.sample(
            batch_size, device=self.device)

        with torch.no_grad():
            # Select next action with target policy (no mask for continuous values)
            next_action_logits, next_charging_duration = self.actor_target(next_state, apply_mask=False)
            
            # Add clipped noise for target policy smoothing
            noise = (torch.randn_like(next_action_logits) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip)
            next_action_logits = next_action_logits + noise
            
            # Also add noise to charging duration
            charge_noise = (torch.randn_like(next_charging_duration) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip)
            next_charging_duration = (next_charging_duration + charge_noise).clamp(0, 10.0)
            
            # Convert logits to action indices for target Q
            # For batched states, we need to handle variable action dimensions
            # Since we can't batch variable-sized tensors, we process individually
            if hasattr(next_state, 'ptr'):
                # Batched graph
                batch_size = len(next_state.ptr) - 1
                next_action_idx = torch.zeros(batch_size, dtype=torch.long, device=self.device)
                
                # For batched graphs, we need action indices per graph
                # This is complex with variable actions - simplified approach
                # Assume action_logits are already properly sized
                if next_action_logits.dim() == 1:
                    # Single action vector - assume batch_size repeats
                    next_action_idx = torch.argmax(next_action_logits).unsqueeze(0).repeat(batch_size)
                else:
                    # Multiple action vectors
                    next_action_idx = torch.argmax(next_action_logits, dim=-1)
            else:
                # Single graph
                next_action_idx = torch.argmax(next_action_logits).unsqueeze(0)

            # Compute target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action_idx, next_charging_duration)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action, charging_duration)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor loss (no mask for gradient flow)
            actor_logits, actor_charging = self.actor(state, apply_mask=False)
            
            # Convert logits to action indices
            # Handle both single and batched states
            if hasattr(state, 'ptr'):
                # Batched graph
                batch_size = len(state.ptr) - 1
                if actor_logits.dim() == 1:
                    actor_action_idx = torch.argmax(actor_logits).unsqueeze(0).repeat(batch_size)
                else:
                    actor_action_idx = torch.argmax(actor_logits, dim=-1)
            else:
                # Single graph
                actor_action_idx = torch.argmax(actor_logits).unsqueeze(0)
            
            actor_loss = -self.critic.Q1(state, actor_action_idx, actor_charging).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data)

            return critic_loss.item(), actor_loss.item()

        return critic_loss.item(), None

    def save(self, filename):
        """
        Save actor and critic networks.
        
        Args:
            filename: Base filename for saving (without extension)
        """
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(),
                   filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(),
                   filename + "_actor_optimizer")

    def load(self, filename):
        """
        Load actor and critic networks.
        
        Args:
            filename: Base filename for loading (without extension)
        """
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(
            torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(
            torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
