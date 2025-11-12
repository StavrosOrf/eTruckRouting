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

    def add(self, state: Any, action: np.ndarray, next_state: Any, 
            reward: float, done: bool):
        """
        Add a transition to the buffer.
        
        Args:
            state: Current state (GNN state object)
            action: Action taken
            next_state: Next state (GNN state object)
            reward: Reward received
            done: Whether episode terminated
        """
        # Move states to CPU for storage to save GPU memory
        if hasattr(state, 'to'):
            state = state.to('cpu')
        if hasattr(next_state, 'to'):
            next_state = next_state.to('cpu')
            
        data = (state, action, next_state, reward, done)
        
        if len(self.storage) == self.max_size:
            self.storage[self.ptr] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size: int, device: torch.device = None):
        """
        Sample a batch of transitions from the buffer.
        
        Args:
            batch_size: Number of transitions to sample
            device: Device to place tensors on
            
        Returns:
            Tuple of (states, actions, next_states, rewards, not_dones)
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Random sampling
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        
        states = []
        actions = []
        next_states = []
        rewards = []
        not_dones = []
        
        for i in ind:
            s, a, ns, r, d = self.storage[i]
            states.append(s)
            actions.append(a)
            next_states.append(ns)
            rewards.append(r)
            not_dones.append(1.0 - float(d))
        
        # Convert actions to tensor
        actions = torch.FloatTensor(actions).to(device)
        if actions.dim() == 0:
            actions = actions.unsqueeze(0)
        
        # Convert rewards and not_dones to tensors
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        not_dones = torch.FloatTensor(not_dones).unsqueeze(1).to(device)
        
        # Batch the GNN states
        state_batch = Batch.from_data_list(states)
        next_state_batch = Batch.from_data_list(next_states)
        
        # Move everything to device
        state_batch = state_batch.to(device)
        next_state_batch = next_state_batch.to(device)
        
        return state_batch, actions, next_state_batch, rewards, not_dones

    def __len__(self) -> int:
        """Return current size of buffer."""
        return len(self.storage)
    
    def clear(self):
        """Clear the buffer."""
        self.storage = []
        self.ptr = 0
