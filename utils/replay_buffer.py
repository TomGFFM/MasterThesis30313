import numpy as np
import random
from collections import namedtuple, deque
import torch
from typing import Tuple, List, Union, Any


class SumTree:
    """A sum tree data structure for efficient priority-based sampling.

    This implementation is designed for use in Prioritized Experience Replay
    in reinforcement learning.

    Attributes:
        capacity: The maximum number of elements that can be stored.
        tree: The sum tree array.
        data: The data array corresponding to priorities in the tree.
        write: The index where the next element will be written.
    """

    def __init__(self, capacity: int):
        """Initialize the SumTree.

        Args:
            capacity: The maximum number of elements that can be stored.
        """
        # Set the maximum capacity of the tree
        self.capacity = capacity
        # Initialize the tree with zeros (2*capacity - 1 for a complete binary tree)
        self.tree = np.zeros(2 * capacity - 1)
        # Initialize the data array to store actual experiences
        self.data = np.zeros(capacity, dtype=object)
        # Initialize the write pointer
        self.write = 0

    def add(self, priority: float, data: Any) -> None:
        """Add a new element with its priority.

        Args:
            priority: The priority of the element.
            data: The data to be stored.

        Raises:
            ValueError: If the priority is negative.
        """
        # Check if the priority is non-negative
        if priority < 0:
            raise ValueError("Priority must be non-negative")

        # Calculate the index in the tree array
        tree_index = self.write + self.capacity - 1
        # Store the data in the data array
        self.data[self.write] = data
        # Update the tree with the new priority
        self.update(tree_index, priority)

        # Move the write pointer
        self.write += 1
        # Reset write pointer if it exceeds capacity
        if self.write >= self.capacity:
            self.write = 0

    def update(self, tree_index: int, priority: float) -> None:
        """Update the priority of an existing element.

        Args:
            tree_index: The index of the element in the tree.
            priority: The new priority value.

        Raises:
            ValueError: If the priority is negative.
        """
        # Check if the priority is non-negative
        if priority < 0:
            raise ValueError("Priority must be non-negative")

        # Calculate the change in priority
        change = priority - self.tree[tree_index]
        # Update the priority at the given index
        self.tree[tree_index] = priority

        # Propagate the change up the tree
        while tree_index != 0:
            # Move to the parent node
            tree_index = (tree_index - 1) // 2
            # Update the parent's value
            self.tree[tree_index] += change

    def get_leaf(self, value: float) -> Tuple[int, float, Any]:
        """Retrieve a leaf node based on a value.

        Args:
            value: The value used to traverse the tree.

        Returns:
            A tuple containing:
            - leaf_index: The index of the chosen leaf.
            - priority: The priority of the chosen leaf.
            - data: The data associated with the chosen leaf.
        """
        # Start from the root
        parent_index = 0

        while True:
            # Calculate indices of left and right children
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            # If we reach beyond the tree, we've found our leaf
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break

            # Decide whether to go left or right
            if value <= self.tree[left_child_index]:
                parent_index = left_child_index
            else:
                value -= self.tree[left_child_index]
                parent_index = right_child_index

        # Calculate the index in the data array
        data_index = leaf_index - self.capacity + 1
        # Return leaf index, priority, and associated data
        return leaf_index, self.tree[leaf_index], self.data[data_index]

    def total_priority(self) -> float:
        """Get the total priority of all elements.

        Returns:
            The sum of all priorities, which is the root node value.
        """
        # The root (index 0) contains the sum of all priorities
        return self.tree[0]


class PrioritizedReplayBuffer:
    """A prioritized replay buffer for reinforcement learning.

    This buffer uses a SumTree to efficiently sample experiences based on their priorities.

    Attributes:
        alpha (float): The exponent determining how much prioritization is used.
        tree (SumTree): The SumTree data structure for storing priorities.
        buffer_size (int): The maximum number of experiences that can be stored.
        batch_size (int): The number of experiences to sample in each batch.
        device (str): The device (e.g., 'cpu' or 'cuda') to use for tensor operations.
        experience (namedtuple): A named tuple for storing individual experiences.
        epsilon (float): A small value added to priorities to ensure non-zero sampling probability.
    """

    def __init__(self, buffer_size: int, batch_size: int, device: str, alpha=0.6):
        """Initialize the PrioritizedReplayBuffer.

        Args:
            buffer_size (int): The maximum number of experiences that can be stored.
            batch_size (int): The number of experiences to sample in each batch.
            device (str): The device to use for tensor operations.
            alpha (float, optional): The exponent determining how much prioritization is used. Defaults to 0.6.
        """
        # Set the prioritization exponent
        self.alpha = alpha
        # Initialize the SumTree with the given buffer size
        self.tree = SumTree(buffer_size)
        # Set the maximum buffer size
        self.buffer_size = buffer_size
        # Set the batch size for sampling
        self.batch_size = batch_size
        # Set the device for tensor operations
        self.device = device
        # Create a named tuple for storing experiences
        self.experience = namedtuple("Experience",
                                     field_names=["state", "action", "reward", "next_state", "terminated", "truncated"])
        # Set a small epsilon to ensure non-zero priorities
        self.epsilon = 0.01

    def push(self, state: np.ndarray, action: Union[int, np.ndarray], reward: float, next_state: np.ndarray,
             terminated: bool, truncated: bool) -> None:
        """Add a new experience to the buffer.

        Args:
            state (np.ndarray): The current state.
            action (Union[int, np.ndarray]): The action taken.
            reward (float): The reward received.
            next_state (np.ndarray): The resulting next state.
            terminated (bool): Whether the episode terminated.
            truncated (bool): Whether the episode was truncated.
        """
        # Create an experience tuple
        experience = self.experience(state, action, reward, next_state, terminated, truncated)
        # Get the maximum priority in the tree
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])
        # If all priorities are zero, set max_priority to 1
        if max_priority == 0:
            max_priority = 1.0
        # Add the experience to the tree with max_priority
        self.tree.add(max_priority, experience)

    def sample(self, beta=0.4):
        """Sample a batch of experiences from the buffer.

        Args:
            beta (float, optional): The importance sampling exponent. Defaults to 0.4.

        Returns:
            tuple: A tuple containing batches of states, actions, rewards, next_states,
                   terminations, truncations, importance sampling weights, and indices.
        """
        # Initialize lists to store sampled experiences, indices, and priorities
        experiences = []
        indices = []
        priorities = []
        # Calculate the priority segment
        segment = self.tree.total_priority() / self.batch_size

        # Sample experiences based on priorities
        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            value = random.uniform(a, b)
            index, priority, experience = self.tree.get_leaf(value)
            experiences.append(experience)
            indices.append(index)
            priorities.append(priority)

        # Calculate sampling probabilities and importance sampling weights
        sampling_probabilities = priorities / self.tree.total_priority()
        is_weights = np.power(self.tree.capacity * sampling_probabilities, -beta)
        is_weights /= is_weights.max()

        # Convert experiences to tensors
        states = torch.from_numpy(np.stack([e.state for e in experiences])).float().to(self.device)
        actions = torch.from_numpy(np.stack([e.action for e in experiences])).long().to(self.device)
        rewards = torch.from_numpy(np.stack([e.reward for e in experiences])).float().to(self.device)
        next_states = torch.from_numpy(np.stack([e.next_state for e in experiences])).float().to(self.device)
        terminations = torch.from_numpy(np.stack([e.terminated for e in experiences]).astype(np.uint8)).float().to(
            self.device)
        truncations = torch.from_numpy(np.stack([e.truncated for e in experiences]).astype(np.uint8)).float().to(
            self.device)
        is_weights = torch.from_numpy(is_weights).float().to(self.device)

        return states, actions, rewards, next_states, terminations, truncations, is_weights, indices

    def update_priorities(self, indices, priorities):
        """Update the priorities of the experiences in the buffer.

        Args:
            indices (list): The indices of the experiences to update.
            priorities (list): The new priorities for the experiences.
        """
        # Update the priority for each experience
        for idx, priority in zip(indices, priorities):
            self.tree.update(idx, priority)

    def __len__(self):
        """Return the current size of the buffer.

        Returns:
            int: The number of experiences currently in the buffer.
        """
        # Return the number of non-zero elements in the data array
        return len(self.tree.data)


