import numpy as np
import random
from collections import namedtuple, deque
import torch
from typing import Tuple, List, Union


class ReplayBuffer:
    """
    A replay buffer to store and sample experience tuples for reinforcement learning.

    Args:
        buffer_size (int): Maximum size of the replay buffer.
        batch_size (int): Size of each sample batch.
        device (str): Device to use for PyTorch tensors ('cpu' or 'cuda').
    """

    def __init__(self, buffer_size: int, batch_size: int, device: str) -> None:
        """
        Initialize the replay buffer.

        Args:
            buffer_size (int): Maximum size of the replay buffer.
            batch_size (int): Size of each sample batch.
            device (str): Device to use for PyTorch tensors ('cpu' or 'cuda').
        """
        self.buffer = deque(maxlen=buffer_size)  # Initialize buffer as a deque with maximum size
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "terminated", "truncated"])
        self.device = device

    def push(self, state: np.ndarray, action: Union[int, np.ndarray], reward: float,
             next_state: np.ndarray, terminated: bool, truncated: bool) -> None:
        """
        Add a new experience tuple to the replay buffer.

        Args:
            state (np.ndarray): Current state.
            action (Union[int, np.ndarray]): Action taken.
            reward (float): Reward received after taking the action.
            next_state (np.ndarray): Next state after taking the action.
            terminated (bool): Whether the episode terminated after taking the action.
            truncated (bool): Whether the episode was truncated after taking the action.
        """
        e = self.experience(state, action, reward, next_state, terminated, truncated)
        self.buffer.append(e)  # Append the new experience tuple to the buffer

    def sample(self) -> Tuple[torch.Tensor, ...]:
        """
        Randomly sample a batch of experiences from the replay buffer.

        Returns:
            Tuple[torch.Tensor, ...]: Tuple containing batches of states, actions, rewards,
                                      next states, termination flags, and truncation flags.
        """
        if len(self.buffer) < self.batch_size:
            # Return None if there are not enough experiences in the buffer
            return None

        experiences = random.sample(self.buffer, k=self.batch_size)  # Randomly sample experiences from the buffer

        # Convert experiences to PyTorch tensors
        states = torch.from_numpy(np.stack([e.state for e in experiences])).float().to(self.device)
        actions = torch.from_numpy(np.stack([e.action for e in experiences])).long().to(self.device)
        rewards = torch.from_numpy(np.stack([e.reward for e in experiences])).float().to(self.device)
        next_states = torch.from_numpy(np.stack([e.next_state for e in experiences])).float().to(self.device)
        terminations = torch.from_numpy(np.stack([e.terminated for e in experiences]).astype(np.uint8)).float().to(self.device)
        truncations = torch.from_numpy(np.stack([e.truncated for e in experiences]).astype(np.uint8)).float().to(self.device)

        return states, actions, rewards, next_states, terminations, truncations

    def __len__(self) -> int:
        """
        Return the current number of experiences in the replay buffer.

        Returns:
            int: Number of experiences in the replay buffer.
        """
        return len(self.buffer)