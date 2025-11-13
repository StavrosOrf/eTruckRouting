"""
Replay Buffer for TD3 training.
"""

import numpy as np
import torch
from typing import List, Tuple, Any
from torch_geometric.data import Batch


class ReplayBuffer:
    """
    Replay buffer for storing and sampling experience tuples.
    Supports both standard transitions and GNN state representations.
    """
    
    def __init__(self, max_size: int = 1000000):
        """
        Initialize replay buffer.
        
        Args:
            max_size: Maximum number of transitions to store
        """
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, state: Any, action: Any, next_state: Any, 
            reward: float, done: bool):
        """
        Add a transition to the buffer.
        
        Args:
            state: Current state (GNN state object)
            action: Action taken - can be (node_index, charging_duration) tuple or just node_index
            next_state: Next state (GNN state object)
            reward: Reward received
            done: Whether episode terminated
        """
        # Move states to CPU for storage to save GPU memory
        if hasattr(state, 'to'):
            state = state.to('cpu')
        if hasattr(next_state, 'to'):
            next_state = next_state.to('cpu')
        
        # Handle action format - can be (node_idx, charging_duration) or just node_idx
        if isinstance(action, tuple):
            action_idx, charging_duration = action
        else:
            action_idx = action
            charging_duration = 0.0  # Default charging duration
        
        # Convert action index to numpy if needed
        if isinstance(action_idx, (int, np.integer)):
            action_idx = np.array(action_idx, dtype=np.int64)
        elif not isinstance(action_idx, np.ndarray):
            action_idx = np.array(action_idx)
        
        # Ensure charging duration is scalar
        if isinstance(charging_duration, np.ndarray):
            charging_duration = float(charging_duration.item())
        else:
            charging_duration = float(charging_duration)
            
        data = (state, action_idx, charging_duration, next_state, reward, done)
        
        if len(self.storage) == self.max_size:
            self.storage[self.ptr] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size: int, device: torch.device = None):
        """
        Sample a batch of transitions from the buffer.
        Handles variable-sized action vectors from HeteroData graphs.
        
        Args:
            batch_size: Number of transitions to sample
            device: Device to place tensors on
            
        Returns:
            Tuple of (states, actions, next_states, rewards, not_dones)
            Note: actions will be a single tensor with all actions concatenated
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Random sampling
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        
        states = []
        actions = []
        charging_durations = []
        next_states = []
        rewards = []
        not_dones = []
        
        for i in ind:
            s, a, cd, ns, r, d = self.storage[i]
            states.append(s)
            actions.append(a)
            charging_durations.append(cd)
            next_states.append(ns)
            rewards.append(r)
            not_dones.append(1.0 - float(d))
        
        # Convert to tensors
        actions_array = np.array(actions, dtype=np.int64)
        actions = torch.LongTensor(actions_array).to(device).squeeze()
        
        charging_durations_array = np.array(charging_durations, dtype=np.float32)
        charging_durations = torch.FloatTensor(charging_durations_array).unsqueeze(1).to(device)
        
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        not_dones = torch.FloatTensor(not_dones).unsqueeze(1).to(device)
        
        # Batch the GNN states (HeteroData)
        # Exclude metadata attributes that have variable sizes and need special handling
        exclude_keys = [
            'action_to_node_map',
            'node_id_to_type',
            'can_charge_here',
            'node_type_offsets',
            'num_actions',
            'feasible_action_mask',  # Variable length - handle manually below
        ]
        state_batch = Batch.from_data_list(states, exclude_keys=exclude_keys)
        next_state_batch = Batch.from_data_list(next_states, exclude_keys=exclude_keys)
        
        # Manually concatenate feasible_action_mask for batched graphs
        # Each graph has a different number of actions, so we concat them
        state_masks = [s.feasible_action_mask for s in states if hasattr(s, 'feasible_action_mask')]
        next_state_masks = [s.feasible_action_mask for s in next_states if hasattr(s, 'feasible_action_mask')]
        
        if state_masks:
            state_batch.feasible_action_mask = torch.cat(state_masks, dim=0)
        if next_state_masks:
            next_state_batch.feasible_action_mask = torch.cat(next_state_masks, dim=0)
        
        # Move everything to device
        state_batch = state_batch.to(device)
        next_state_batch = next_state_batch.to(device)
        
        return state_batch, actions, charging_durations, next_state_batch, rewards, not_dones

    def __len__(self) -> int:
        """Return current size of buffer."""
        return len(self.storage)
    
    def clear(self):
        """Clear the buffer."""
        self.storage = []
        self.ptr = 0
