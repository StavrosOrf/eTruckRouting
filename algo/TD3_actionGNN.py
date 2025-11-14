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
            min_charging_duration=0.5,
            max_charging_duration=10.0,
            target_action_temperature=1.0,
            device=None,
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
            min_charging_duration: Minimum charging duration (hours)
            max_charging_duration: Maximum charging duration (hours)
            target_action_temperature: Temperature for sampling target actions (>0)
            device: Device to use (cuda/cpu). If None, auto-detect.
        """
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.discrete_actions = discrete_actions

        # Initialize actor network
        self.actor = Actor(
            max_action,
            hidden_dim=fx_GNN_hidden_dim,
            num_layers=actor_num_gcn_layers,
            min_charging_duration=min_charging_duration,
            max_charging_duration=max_charging_duration,
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
        self.min_charging_duration = min_charging_duration
        self.max_charging_duration = max_charging_duration
        self.target_action_temperature = target_action_temperature

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

        # Exploration noise directly on logits
        if expl_noise > 0:
            action_logits = action_logits + torch.randn_like(action_logits) * expl_noise

        # Apply softmax over feasible actions only
        mask = state.feasible_action_mask
        
        # Check if there are any feasible actions
        if not mask.any():
            raise ValueError("No feasible actions found in select_action - state construction error")
        
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
        # else:
        #     # Fallback: no masking
        #     if expl_noise > 0:
        #         noise = torch.randn_like(action_logits) * expl_noise
        #         action_logits = action_logits + noise
        #     action_idx = torch.argmax(action_logits).item()
        
        # Extract charging duration (scalar value)
        charge_dur = charging_duration.squeeze().item()
        
        # Translate action_idx to (node_id, is_charging_action) using action_to_node_map
        if action_idx >= len(state.action_to_node_map):
            raise ValueError(f"action_idx {action_idx} out of range for action_to_node_map (size {len(state.action_to_node_map)})")
        
        node_id, is_charging = state.action_to_node_map[action_idx]
        return node_id, charge_dur, is_charging

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
            # Select next action with target policy (respect feasibility mask)
            next_action_logits, next_charging_duration = self.actor_target(next_state, apply_mask=True)
            
            # Add clipped noise for target policy smoothing
            noise = (torch.randn_like(next_action_logits) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip)
            next_action_logits = next_action_logits + noise
            
            # Also add noise to charging duration
            charge_noise = (torch.randn_like(next_charging_duration) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip)
            next_charging_duration = (next_charging_duration + charge_noise).clamp(
                self.min_charging_duration, self.max_charging_duration)
            
            infeasible_fill = torch.tensor(-1e9, device=self.device)
            next_action_logits = torch.where(next_state.feasible_action_mask, next_action_logits, infeasible_fill)

            next_counts, next_offsets = self._get_action_layout(next_state)
            next_action_idx = self._sample_actions(
                next_action_logits,
                next_counts,
                next_offsets,
                temperature=self.target_action_temperature
            )

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

            actor_logits, actor_charging = self.actor(state, apply_mask=True)
            state_counts, state_offsets = self._get_action_layout(state)
            actor_action_idx = self._select_best_actions(actor_logits, state_counts, state_offsets)
            
            # Compute Q-value for selected actions
            actor_Q = self.critic.Q1(state, actor_action_idx, actor_charging)
            actor_loss = -actor_Q.mean()

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

    def _get_action_layout(self, data):
        """Return per-graph action counts and cumulative offsets."""
        if not hasattr(data, 'num_actions'):
            raise ValueError("State is missing num_actions metadata")
        counts = data.num_actions
        if isinstance(counts, torch.Tensor):
            if counts.dim() == 0:
                counts = counts.unsqueeze(0)
            counts = counts.to(torch.long).to(self.device)
        else:
            counts = torch.tensor([int(counts)], dtype=torch.long, device=self.device)
        if counts.numel() == 0:
            return counts, counts
        offsets = torch.cumsum(
            torch.cat([counts.new_zeros(1), counts[:-1]]), dim=0
        )
        return counts, offsets

    def _select_best_actions(self, logits, counts, offsets):
        """Pick argmax action index for each graph given concatenated logits."""
        best_actions = []
        num_graphs = counts.shape[0]
        for i in range(num_graphs):
            count = int(counts[i].item())
            if count <= 0:
                best_actions.append(torch.tensor(0, dtype=torch.long, device=logits.device))
                continue
            start = int(offsets[i].item())
            end = start + count
            graph_logits = logits[start:end]
            if graph_logits.numel() == 0:
                best_actions.append(torch.tensor(0, dtype=torch.long, device=logits.device))
            else:
                best_actions.append(torch.argmax(graph_logits))
        if best_actions:
            return torch.stack(best_actions)
        return torch.empty((0,), dtype=torch.long, device=logits.device)

    def _sample_actions(self, logits, counts, offsets, temperature=1.0):
        """Sample action indices per graph using softmax distribution."""
        sampled = []
        num_graphs = counts.shape[0]
        temperature = max(temperature, 1e-3)
        for i in range(num_graphs):
            count = int(counts[i].item())
            start = int(offsets[i].item()) if counts.numel() > 0 else 0
            if count <= 0:
                sampled.append(torch.tensor(0, dtype=torch.long, device=logits.device))
                continue
            end = start + count
            graph_logits = logits[start:end]
            if graph_logits.numel() == 0:
                sampled.append(torch.tensor(0, dtype=torch.long, device=logits.device))
                continue
            probs = torch.softmax(graph_logits / temperature, dim=0)
            action_idx = torch.multinomial(probs, 1).squeeze(0)
            sampled.append(action_idx)
        if sampled:
            return torch.stack(sampled)
        return torch.empty((0,), dtype=torch.long, device=logits.device)

    def get_action_diagnostics(self, state):
        """Return diagnostic metrics for current state (logits/Q stats)."""
        state = state.to(self.device)
        with torch.no_grad():
            logits, charging = self.actor(state, apply_mask=True)
            counts, offsets = self._get_action_layout(state)
            diag = {
                'diag/top1_logit': 0.0,
                'diag/top_gap': 0.0,
                'diag/action_entropy': 0.0,
                'diag/q_best': 0.0
            }
            if logits.numel() == 0 or counts.sum() == 0:
                return diag

            top1_vals = []
            gaps = []
            entropies = []
            num_graphs = counts.shape[0]
            for i in range(num_graphs):
                count = int(counts[i].item())
                if count <= 0:
                    continue
                start = int(offsets[i].item())
                end = start + count
                graph_logits = logits[start:end]
                probs = torch.softmax(graph_logits, dim=0)
                entropies.append(-(probs * torch.log(probs + 1e-8)).sum())
                topk = torch.topk(graph_logits, k=min(3, count)).values
                top1_vals.append(topk[0])
                if topk.numel() > 1:
                    gaps.append(topk[0] - topk[1])

            if top1_vals:
                diag['diag/top1_logit'] = torch.stack(top1_vals).mean().item()
            if gaps:
                diag['diag/top_gap'] = torch.stack(gaps).mean().item()
            if entropies:
                diag['diag/action_entropy'] = torch.stack(entropies).mean().item()

            best_idx = self._select_best_actions(logits, counts, offsets)
            if best_idx.numel() > 0:
                q_best = self.critic.Q1(state, best_idx, charging)
                diag['diag/q_best'] = q_best.mean().item()

            return diag

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
