import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import random
import logging
import yaml

from utils import ReplayBuffer
from typing import Dict, Tuple, Union, Any


class DeepQNetworkAgent:
    """A DQN agent that interacts with and learns from the environment."""

    def __init__(self, model: torch.nn.Module, action_size: int, device: torch.device, agent_hyper_params: Dict,
                 network_hyper_params: Dict, model_name: str = None):

        """
        Initialize an Agent object.

        Args:
            model (torch.nn.Module): Pytorch Model
            action_size (int): dimension of each action
            device (torch.device): device create from torch
            agent_hyper_params (Dict): hyperparameters dictionary containing relevant settings for agent training etc.
            network_hyper_params (Dict): hyperparameters dictionary containing network specific settings
        """
        # make hyperparam dicts available for property usage
        self._agent_hyper_params = agent_hyper_params
        self._network_hyper_params = network_hyper_params

        # set relevant hyperparameter and model vars
        self.DQN = model
        self.action_size = action_size
        self.device = device
        self.batch_size = agent_hyper_params['batch_size']
        self.gamma = agent_hyper_params['gamma']
        # self.input_shape = agent_hyper_params['input_shape']
        self.learning_rate = agent_hyper_params['learning_rate']
        self.replay_buffer_size = agent_hyper_params['replay_buffer_size']
        self.tau = agent_hyper_params['tau']
        self.update_every = agent_hyper_params['update_every']
        self.update_target = agent_hyper_params['update_target']

        # Q-Network
        self.policy_net = self.DQN(**network_hyper_params).to(self.device)
        self.target_net = self.DQN(**network_hyper_params).to(self.device)
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.learning_rate)

        # Update model names if set from outside
        if model_name:
            self.policy_net.model_name = model_name
            self.target_net.model_name = model_name

        # Replay memory
        self.memory = ReplayBuffer(self.replay_buffer_size, self.batch_size, self.device)

        self.t_step = 0

    @property
    def agent_model(self):
        return self.policy_net.model_name

    @property
    def agent_hyper_params(self):
        return self._agent_hyper_params

    @property
    def network_hyper_params(self):
        return self._network_hyper_params

    def step(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, terminated: bool,
             truncated: bool) -> float:
        """
        Perform a single step of the agent's interaction with the environment.

        Args:
            state (np.ndarray): The current state of the environment.
            action (int): The action taken by the agent.
            reward (float): The reward received by the agent.
            next_state (np.ndarray): The next state of the environment.
            terminated (bool): Whether the episode has terminated.
            truncated (bool): Whether the episode has been truncated.
            previous_reward (float): The previous reward received by the agent.

        Returns:
            loss(float): loss value from q values
        """

        # Save experience in replay memory
        self.memory.push(state, action, reward, next_state, terminated, truncated)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every

        # Init loss
        loss = 0

        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.update_target:
                experiences = self.memory.sample()
                if experiences is not None:
                    loss = self.learn(experiences)
        return loss

    def act(self, state: np.ndarray, eps: float = 0.) -> Tuple[torch.Tensor, str]:
        """
        Choose an action to take based on the current state.

        Args:
            state (np.ndarray): The current state of the environment.
            eps (float): The epsilon value for epsilon-greedy action selection.

        Returns:
            int: The action to take.
        """
        state = torch.from_numpy(state).unsqueeze(0).to(self.device)
        self.policy_net.eval()
        with torch.no_grad():
            action_values = self.policy_net(state)
        self.policy_net.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            predicted_action = np.argmax(action_values.cpu().data.numpy())
            logging.debug(f"Predicted action was taken: {predicted_action}")
            return predicted_action, 'predicted'
        else:
            random_action = random.choice(np.arange(self.action_size))
            logging.debug(f"Random action was taken: {random_action}")
            return random_action, 'randomized'

    def learn(self, experiences: Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> float:
        """
        Update the Q-network using a batch of experiences.

        Args:
            experiences (Tuple[torch.Tensor]): A tuple of (states, actions, rewards, next_states, terminations, truncations) tensors.

        Returns:
            loss(float): loss value from q values
        """
        states, actions, rewards, next_states, terminations, truncations = experiences

        # Get expected Q values from policy model
        q_expected_current = self.policy_net(states)
        q_expected = q_expected_current.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Get max predicted Q values (for next states) from target model
        q_targets_next = self.target_net(next_states).detach().max(1)[0]

        # Compute Q targets for current states
        q_targets = rewards + (self.gamma * q_targets_next * (1 - terminations) * (1 - truncations))

        # Compute loss
        # loss = F.mse_loss(q_expected, q_targets)
        loss = F.smooth_l1_loss(q_expected, q_targets)
        logging.debug(f"Current loss: {loss}")

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.soft_update(self.policy_net, self.target_net)

        return loss.item()

    def soft_update(self, policy_model: torch.nn.Module, target_model: torch.nn.Module) -> None:

        """
        Perform a soft update of the target Q-network.

        Args:
            policy_model (torch.nn.Module): The policy Q-network.
            target_model (torch.nn.Module): The target Q-network.
        """
        # tau = self.tau * (1 - self.t_step / self.update_every)  # Decrease tau over time
        tau = self.tau
        for target_param, policy_param in zip(target_model.parameters(), policy_model.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)

    def save(self, filepath: str) -> None:
        """
        Save the current policy network and optimizer state to a file.

        Args:
            filepath (str): The path to save the state to.
        """

        # Save torch networks in filepath destination
        state = {
            'policy_net': self.policy_net.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(state, filepath)

    def load(self, filepath: str, map_location: torch.device = None) -> None:
        """
        Load the policy network and optimizer state from a file.

        Args:
            :param filepath: The path to load the state from.
            :param map_location: If set this parameter is used to set a new device location while loading the model
        """
        if map_location:
            state = torch.load(filepath, map_location=map_location)
        else:
            state = torch.load(filepath)

        self.policy_net.load_state_dict(state['policy_net'])
        self.optimizer.load_state_dict(state['optimizer'])


class DeepQNetworkAgentv2:
    """A DQN agent that interacts with and learns from the environment."""

    def __init__(self, model: torch.nn.Module, action_size: int, device: torch.device, agent_hyper_params: Dict,
                 network_hyper_params: Dict, model_name: str = None):

        """
        Initialize an Agent object.

        Args:
            model (torch.nn.Module): Pytorch Model
            action_size (int): dimension of each action
            device (torch.device): device create from torch
            agent_hyper_params (Dict): hyperparameters dictionary containing relevant settings for agent training etc.
            network_hyper_params (Dict): hyperparameters dictionary containing network specific settings
        """
        # make hyperparam dicts available for property usage
        self._agent_hyper_params = agent_hyper_params
        self._network_hyper_params = network_hyper_params

        # set relevant hyperparameter and model vars
        self.DQN = model
        self.action_size = action_size
        self.device = device
        self.batch_size = agent_hyper_params['batch_size']
        self.gamma = agent_hyper_params['gamma']
        # self.input_shape = agent_hyper_params['input_shape']
        self.learning_rate = agent_hyper_params['learning_rate']
        self.replay_buffer_size = agent_hyper_params['replay_buffer_size']
        self.tau = agent_hyper_params['tau']
        self.update_every = agent_hyper_params['update_every']
        self.update_target = agent_hyper_params['update_target']

        # Q-Network
        self.policy_net = self.DQN(**network_hyper_params).to(self.device)
        self.target_net = self.DQN(**network_hyper_params).to(self.device)
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.learning_rate)

        # Update model names if set from outside
        if model_name:
            self.policy_net.model_name = model_name
            self.target_net.model_name = model_name

        # Replay memory
        self.memory = ReplayBuffer(self.replay_buffer_size, self.batch_size, self.device)

        self.t_step = 0

    @property
    def agent_model(self):
        return self.policy_net.model_name

    @property
    def agent_hyper_params(self):
        return self._agent_hyper_params

    @property
    def network_hyper_params(self):
        return self._network_hyper_params

    def step(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, terminated: bool,
             truncated: bool, previous_reward: float, previous_action: int) -> float:
        """
        Perform a single step of the agent's interaction with the environment.

        Args:
            state (np.ndarray): The current state of the environment.
            action (int): The action taken by the agent.
            reward (float): The reward received by the agent.
            next_state (np.ndarray): The next state of the environment.
            terminated (bool): Whether the episode has terminated.
            truncated (bool): Whether the episode has been truncated.
            previous_reward (float): The previous reward received by the agent.
            previous_action (int): The previous action taken by the agent.

        Returns:
            loss(float): loss value from q values
        """
        # Shape reward if the next score was higher or if the action changed
        if reward > previous_reward or previous_action != action:
            reward = self.reward_update(reward)

        # Punish if action is 0 or if the same action is repeated
        if action == 0 or previous_action == action:
            reward = self.reward_update(reward, punish=True)

        # Additional punishment if the episode has terminated
        if terminated:
            reward = self.reward_update(reward, punish=True)

        # Save experience in replay memory
        self.memory.push(state, action, reward, next_state, terminated, truncated)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every

        # Init loss
        loss = 0

        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.update_target:
                experiences = self.memory.sample()
                if experiences is not None:
                    loss = self.learn(experiences)
        return loss

    def act(self, state: np.ndarray, eps: float = 0.) -> Tuple[torch.Tensor, str]:
        """
        Choose an action to take based on the current state.

        Args:
            state (np.ndarray): The current state of the environment.
            eps (float): The epsilon value for epsilon-greedy action selection.

        Returns:
            int: The action to take.
        """
        state = torch.from_numpy(state).unsqueeze(0).to(self.device)
        self.policy_net.eval()
        with torch.no_grad():
            action_values = self.policy_net(state)
        self.policy_net.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            predicted_action = np.argmax(action_values.cpu().data.numpy())
            logging.debug(f"Predicted action was taken: {predicted_action}")
            return predicted_action, 'predicted'
        else:
            random_action = random.choice(np.arange(self.action_size))
            logging.debug(f"Random action was taken: {random_action}")
            return random_action, 'randomized'

    def learn(self, experiences: Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> float:
        """
        Update the Q-network using a batch of experiences.

        Args:
            experiences (Tuple[torch.Tensor]): A tuple of (states, actions, rewards, next_states, terminations, truncations) tensors.

        Returns:
            loss(float): loss value from q values
        """
        states, actions, rewards, next_states, terminations, truncations = experiences

        # Get expected Q values from policy model
        q_expected_current = self.policy_net(states)
        q_expected = q_expected_current.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Get max predicted Q values (for next states) from target model
        q_targets_next = self.target_net(next_states).detach().max(1)[0]

        # Compute Q targets for current states
        q_targets = rewards + (self.gamma * q_targets_next * (1 - terminations) * (1 - truncations))

        # Compute loss
        # loss = F.mse_loss(q_expected, q_targets)
        loss = F.smooth_l1_loss(q_expected, q_targets)
        logging.debug(f"Current loss: {loss}")

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.soft_update(self.policy_net, self.target_net)

        return loss.item()

    def soft_update(self, policy_model: torch.nn.Module, target_model: torch.nn.Module) -> None:

        """
        Perform a soft update of the target Q-network.

        Args:
            policy_model (torch.nn.Module): The policy Q-network.
            target_model (torch.nn.Module): The target Q-network.
        """
        # tau = self.tau * (1 - self.t_step / self.update_every)  # Decrease tau over time
        tau = self.tau
        for target_param, policy_param in zip(target_model.parameters(), policy_model.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)

    def save(self, filepath: str) -> None:
        """
        Save the current policy network and optimizer state to a file.

        Args:
            filepath (str): The path to save the state to.
        """

        # Save torch networks in filepath destination
        state = {
            'policy_net': self.policy_net.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(state, filepath)

    def load(self, filepath: str, map_location: torch.device = None) -> None:
        """
        Load the policy network and optimizer state from a file.

        Args:
            :param filepath: The path to load the state from.
            :param map_location: If set this parameter is used to set a new device location while loading the model
        """
        if map_location:
            state = torch.load(filepath, map_location=map_location)
        else:
            state = torch.load(filepath)

        self.policy_net.load_state_dict(state['policy_net'])
        self.optimizer.load_state_dict(state['optimizer'])

    def reward_update(self, reward: float, punish: bool = False) -> float:
        """
        Adjust the reward based on an exponentially increasing or decreasing factor of the score.

        Args:
            reward (float): The current reward.
            punish (bool): Whether to apply a punishment (decrease the reward). Default is False.

        Returns:
            float: The adjusted reward.
        """
        # Define an exponential base for the factor
        base_factor = 1.5

        # Calculate the factor exponentially based on the reward
        factor = base_factor ** (abs(reward) / 100)

        # Adjust the reward based on whether it's a punishment or reward
        if punish:
            adjusted_reward = reward / factor
        else:
            adjusted_reward = reward * factor

        return adjusted_reward
