import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
import random
import logging
import yaml

from utils import ReplayBuffer, ReplayBufferWithFrames
from typing import Dict, Tuple, Union, Any


class DeepQNetworkAgentv4:
    """A DQN agent that interacts with and learns from the environment."""

    def __init__(self,
                 policy_net: torch.nn.Module,
                 target_net: torch.nn.Module,
                 action_size: int,
                 device: torch.device,
                 agent_hyper_params: Dict,
                 network_hyper_params: Dict,
                 optimizer: torch.optim.Optimizer,
                 lr_scheduler: torch.optim.lr_scheduler = None,
                 loss_function: torch.nn.functional = None,
                 reward_shaping: bool = False,
                 reward_factor: float = 1.4,
                 punish_factor: float = 1.8):

        """
        Initialize an Agent object.

        Args:
            action_size (int): Dimension of each action.
            device (torch.device): The device on which the model will be run (e.g., CPU or GPU).
            agent_hyper_params (Dict): Dictionary containing hyperparameters for the agent, such as 'batch_size', 'gamma', 'learning_rate', 'replay_buffer_size', 'tau', 'final_tau', 'max_steps', 'update_every', and 'update_target'.
            network_hyper_params (Dict): Dictionary containing network-specific settings, such as the model architecture and any required parameters for initializing the model.
            optimizer (torch.optim.Optimizer): The optimizer class used for training the policy network.
            lr_scheduler (torch.optim.lr_scheduler, optional): Learning rate scheduler to adjust the learning rate during training. Default is None.
            loss_function (torch.nn.functional, optional): Loss function used for training. Default is None.
            reward_shaping (bool, optional): Whether to apply reward shaping during training. Default is False.
            reward_factor (float, optional): Factor by which rewards are multiplied during reward shaping. Default is 1.4.
            punish_factor (float, optional): Factor by which rewards are multiplied when applying punishment during reward shaping. Default is 1.8.
            model_name (str, optional): Optional name for the model. This name can be used to identify and save models during training. Default is None.
        """

        # make hyperparam dicts available for property usage
        self._agent_hyper_params = agent_hyper_params
        self._network_hyper_params = network_hyper_params

        # set relevant hyperparameter and model vars
        self.action_size = action_size
        self.device = device
        self.batch_size = agent_hyper_params['batch_size']
        self.gamma = agent_hyper_params['gamma']

        # learning related parameters
        self.replay_buffer_size = agent_hyper_params['replay_buffer_size']
        self.tau = agent_hyper_params['tau']
        self.final_tau = agent_hyper_params['final_tau']
        self.max_steps = agent_hyper_params['max_steps_episode'] * agent_hyper_params['n_episodes']
        self.update_every = agent_hyper_params['update_every']
        self.update_target = agent_hyper_params['update_target']
        self.learn_start = agent_hyper_params['learn_start']

        # reward shaping settings
        self.reward_shaping = reward_shaping
        self.reward_factor = reward_factor
        self.punish_factor = punish_factor

        # Q-Network
        self.policy_net = policy_net
        self.target_net = target_net

        # Optimizer, LR and loss function
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.lr_scheduler = lr_scheduler

        # Replay memory
        self.memory = ReplayBuffer(self.replay_buffer_size, self.batch_size, self.device)

        # Init internal step counter
        self.step_count = 0

        # Init metric collection
        self.tau_step_value = self.tau
        self.lr_step_value = agent_hyper_params['learning_rate']
        self.df_final_q_metrics = pd.DataFrame()
        self.episode = 0

    @property
    def agent_model(self):
        return self.policy_net.model_name

    @property
    def q_value_metrics(self):
        return self.df_final_q_metrics

    @property
    def agent_hyper_params(self):
        return self._agent_hyper_params

    @property
    def network_hyper_params(self):
        return self._network_hyper_params

    def step(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, terminated: bool,
             truncated: bool, previous_reward: float, previous_action: int, episode: int) -> float:
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
            episode (int): The current episode number.

        Returns:
            loss(float): loss value from q values
        """

        self.episode = episode

        if self.reward_shaping:
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

        # Init loss
        loss = 0

        if episode > self.learn_start and len(self.memory) > self.update_target:

            # Learn every UPDATE_EVERY time steps.
            step_modulo = self.step_count % self.update_every

            # If the modulo condition was met and the experiences buffer is not empty the learning process is performed
            if step_modulo == 0:
                experiences = self.memory.sample()
                if experiences is not None:
                    loss = self.learn(experiences)

        # Increment step_count
        self.step_count += 1

        return loss, reward, self.tau_step_value, self.lr_step_value

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

        # Compute Q metrics
        metrics = {}

        # Calculate average Q-value
        metrics['avg_q_value'] = q_expected_current.mean().item()

        # Calculate max and min Q-value
        metrics['max_q_value'] = q_expected_current.max().item()
        metrics['min_q_value'] = q_expected_current.min().item()

        # Calculate Q-value variance
        metrics['q_value_variance'] = q_expected_current.var().item()

        # Calculate TD-Error (absolute difference between q_targets and q_expected)
        metrics['td_error'] = (q_targets - q_expected).abs().mean().item()

        # Calculate the ratio of positive Q-values
        metrics['positive_q_ratio'] = (q_expected_current > 0).float().mean().item()

        # Calculate Q-value for the chosen actions
        metrics['q_value_for_action'] = q_expected_current.gather(1, actions.unsqueeze(1)).mean().item()

        # Create final q metrics dataframe
        df_q_metrics = pd.DataFrame([metrics])
        df_q_metrics['episode'] = self.episode
        self.df_final_q_metrics = pd.concat([self.df_final_q_metrics, df_q_metrics], ignore_index=True)

        # Compute loss
        if self.loss_function_delta is None:
            loss = self.loss_function(q_expected, q_targets)
        else:
            loss = self.loss_function(q_expected, q_targets, delta=self.loss_function_delta)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            self.lr_step_value = self.optimizer.param_groups[0]['lr']

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
        # Dynamically decrease tau over time but keep above 0
        tau = max(self.tau * (1 - self.step_count / self.max_steps), self.final_tau)
        self.tau_step_value = tau

        with torch.no_grad():
            for target_param, policy_param in zip(target_model.parameters(), policy_model.parameters()):
                target_param.copy_(tau * policy_param + (1.0 - tau) * target_param)

        logging.debug(f'Model was updated via soft update. tau: {tau}; step count: {self.step_count}')

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

        # Set base factor according to configuration
        if punish:
            base_factor = self.punish_factor
        else:
            base_factor = self.reward_factor

        # Apply factor to reward
        if punish:
            adjusted_reward = reward / base_factor
        else:
            adjusted_reward = reward * base_factor

        return adjusted_reward


# class DeepQNetworkAgentv5:
#     """A DQN agent that interacts with and learns from the environment."""
#
#     def __init__(self,
#                  policy_net: torch.nn.Module,
#                  target_net: torch.nn.Module,
#                  action_size: int,
#                  device: torch.device,
#                  agent_hyper_params: Dict,
#                  network_hyper_params: Dict,
#                  optimizer: torch.optim.Optimizer,
#                  lr_scheduler: torch.optim.lr_scheduler = None,
#                  loss_function: torch.nn.functional = None,
#                  loss_function_delta = None,
#                  reward_shaping: bool = False,
#                  reward_factor: float = 1.4,
#                  punish_factor: float = 1.8):
#
#         """
#         Initialize an Agent object.
#
#         Args:
#             action_size (int): Dimension of each action.
#             device (torch.device): The device on which the model will be run (e.g., CPU or GPU).
#             agent_hyper_params (Dict): Dictionary containing hyperparameters for the agent, such as 'batch_size', 'gamma', 'learning_rate', 'replay_buffer_size', 'tau', 'final_tau', 'max_steps', 'update_every', and 'update_target'.
#             network_hyper_params (Dict): Dictionary containing network-specific settings, such as the model architecture and any required parameters for initializing the model.
#             optimizer (torch.optim.Optimizer): The optimizer class used for training the policy network.
#             lr_scheduler (torch.optim.lr_scheduler, optional): Learning rate scheduler to adjust the learning rate during training. Default is None.
#             loss_function (torch.nn.functional, optional): Loss function used for training. Default is None.
#             loss_function_delta (float, optional): Loss function delta used for training. Default is None.
#             reward_shaping (bool, optional): Whether to apply reward shaping during training. Default is False.
#             reward_factor (float, optional): Factor by which rewards are multiplied during reward shaping. Default is 1.4.
#             punish_factor (float, optional): Factor by which rewards are multiplied when applying punishment during reward shaping. Default is 1.8.
#             model_name (str, optional): Optional name for the model. This name can be used to identify and save models during training. Default is None.
#         """
#
#         # make hyperparam dicts available for property usage
#         self._agent_hyper_params = agent_hyper_params
#         self._network_hyper_params = network_hyper_params
#
#         # set relevant hyperparameter and model vars
#         self.action_size = action_size
#         self.device = device
#         self.batch_size = agent_hyper_params['batch_size']
#         self.gamma = agent_hyper_params['gamma']
#
#         # learning related parameters
#         self.replay_buffer_size = agent_hyper_params['replay_buffer_size']
#         self.tau = agent_hyper_params['tau']
#         self.final_tau = agent_hyper_params['final_tau']
#         self.max_steps = agent_hyper_params['max_steps_episode'] * agent_hyper_params['n_episodes']
#         self.update_every = agent_hyper_params['update_every']
#         self.update_target = agent_hyper_params['update_target']
#
#         # reward shaping settings
#         self.reward_shaping = reward_shaping
#         self.reward_factor = reward_factor
#         self.punish_factor = punish_factor
#
#         # Q-Network
#         self.policy_net = policy_net
#         self.target_net = target_net
#
#         # Optimizer, LR and loss function
#         self.optimizer = optimizer
#         self.loss_function = loss_function
#         self.loss_function_delta = loss_function_delta
#         self.lr_scheduler = lr_scheduler
#
#         # Replay memory
#         self.memory = ReplayBufferWithFrames(self.replay_buffer_size, self.batch_size, self.device)
#
#         # Init internal step counter
#         self.step_count = 0
#
#         # Init metric collection
#         self.tau_step_value = self.tau
#         self.lr_step_value = agent_hyper_params['learning_rate']
#         self.df_final_q_metrics = pd.DataFrame()
#         self.episode = 0
#
#     @property
#     def agent_model(self):
#         return self.policy_net.model_name
#
#     @property
#     def q_value_metrics(self):
#         return self.df_final_q_metrics
#
#     @property
#     def agent_hyper_params(self):
#         return self._agent_hyper_params
#
#     @property
#     def network_hyper_params(self):
#         return self._network_hyper_params
#
#     def step(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, terminated: bool,
#              truncated: bool, previous_reward: float, previous_action: int, episode: int) -> float:
#         """
#         Perform a single step of the agent's interaction with the environment.
#
#         Args:
#             state (np.ndarray): The current state of the environment.
#             action (int): The action taken by the agent.
#             reward (float): The reward received by the agent.
#             next_state (np.ndarray): The next state of the environment.
#             terminated (bool): Whether the episode has terminated.
#             truncated (bool): Whether the episode has been truncated.
#             previous_reward (float): The previous reward received by the agent.
#             previous_action (int): The previous action taken by the agent.
#             episode (int): The current episode number.
#
#         Returns:
#             loss(float): loss value from q values
#         """
#
#         self.episode = episode
#
#         if self.reward_shaping:
#             # Shape reward if the next score was higher or if the action changed
#             if reward > previous_reward or previous_action != action:
#                 reward = self.reward_update(reward)
#
#             # Punish if action is 0 or if the same action is repeated
#             if action == 0 or previous_action == action:
#                 reward = self.reward_update(reward, punish=True)
#
#             # Additional punishment if the episode has terminated
#             if terminated:
#                 reward = self.reward_update(reward, punish=True)
#
#         # Save experience in replay memory
#         self.memory.push(state, action, reward, next_state, terminated, truncated)
#
#         # Learn every UPDATE_EVERY time steps.
#         step_modulo = self.step_count % self.update_every
#
#         # Init loss
#         loss = 0
#
#         if step_modulo == 0:
#             # If enough samples are available in memory, get random subset and learn
#             if len(self.memory) > self.update_target:
#                 experiences = self.memory.sample()
#                 if experiences is not None:
#                     loss = self.learn(experiences)
#
#         # Increment step_count
#         self.step_count += 1
#
#         return loss, reward, self.tau_step_value, self.lr_step_value
#
#     def act(self, state: np.ndarray, eps: float = 0.) -> Tuple[torch.Tensor, str]:
#         """
#         Choose an action to take based on the current state.
#
#         Args:
#             state (np.ndarray): The current state of the environment.
#             eps (float): The epsilon value for epsilon-greedy action selection.
#
#         Returns:
#             int: The action to take.
#         """
#         state = torch.from_numpy(state).unsqueeze(0).to(self.device)
#         self.policy_net.eval()
#         with torch.no_grad():
#             action_values = self.policy_net(state)
#         self.policy_net.train()
#
#         # Epsilon-greedy action selection
#         if random.random() > eps:
#             predicted_action = np.argmax(action_values.cpu().data.numpy())
#             logging.debug(f"Predicted action was taken: {predicted_action}")
#             return predicted_action, 'predicted'
#         else:
#             random_action = random.choice(np.arange(self.action_size))
#             logging.debug(f"Random action was taken: {random_action}")
#             return random_action, 'randomized'
#
#     def learn(self, experiences: Tuple[
#         torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> float:
#         """
#         Update the Q-network using a batch of experiences.
#
#         Args:
#             experiences (Tuple[torch.Tensor]): A tuple of (states, actions, rewards, next_states, terminations, truncations) tensors.
#
#         Returns:
#             loss(float): loss value from q values
#         """
#         states, actions, rewards, next_states, terminations, truncations, frame_indices = experiences
#
#         # Get expected Q values from policy model
#         q_expected_current = self.policy_net(states, frame_indices)
#         q_expected = q_expected_current.gather(1, actions.unsqueeze(1)).squeeze(1)
#
#         # Get max predicted Q values (for next states) from target model
#         q_targets_next = self.target_net(next_states, frame_indices).detach().max(1)[0]
#
#         # Compute Q targets for current states
#         q_targets = rewards + (self.gamma * q_targets_next * (1 - terminations) * (1 - truncations))
#
#         # Compute Q metrics
#         metrics = {}
#
#         # Calculate average Q-value
#         metrics['avg_q_value'] = q_expected_current.mean().item()
#
#         # Calculate max and min Q-value
#         metrics['max_q_value'] = q_expected_current.max().item()
#         metrics['min_q_value'] = q_expected_current.min().item()
#
#         # Calculate Q-value variance
#         metrics['q_value_variance'] = q_expected_current.var().item()
#
#         # Calculate TD-Error (absolute difference between q_targets and q_expected)
#         metrics['td_error'] = (q_targets - q_expected).abs().mean().item()
#
#         # Calculate the ratio of positive Q-values
#         metrics['positive_q_ratio'] = (q_expected_current > 0).float().mean().item()
#
#         # Calculate Q-value for the chosen actions
#         metrics['q_value_for_action'] = q_expected_current.gather(1, actions.unsqueeze(1)).mean().item()
#
#         # Create final q metrics dataframe
#         df_q_metrics = pd.DataFrame([metrics])
#         df_q_metrics['episode'] = self.episode
#         self.df_final_q_metrics = pd.concat([self.df_final_q_metrics, df_q_metrics], ignore_index=True)
#
#         # Compute loss
#         if self.loss_function_delta is None:
#             loss = self.loss_function(q_expected, q_targets)
#         else:
#             loss = self.loss_function(q_expected, q_targets, delta=self.loss_function_delta)
#
#         logging.debug(f"Current loss: {loss}")
#
#         # Minimize the loss
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()
#
#         if self.lr_scheduler is not None:
#             self.lr_scheduler.step()
#             self.lr_step_value = self.optimizer.param_groups[0]['lr']
#
#         # Update target network
#         self.soft_update(self.policy_net, self.target_net)
#
#         return loss.item()
#
#     def soft_update(self, policy_model: torch.nn.Module, target_model: torch.nn.Module) -> None:
#
#         """
#         Perform a soft update of the target Q-network.
#
#         Args:
#             policy_model (torch.nn.Module): The policy Q-network.
#             target_model (torch.nn.Module): The target Q-network.
#         """
#         # Dynamically decrease tau over time but keep above 0
#         tau = max(self.tau * (1 - self.step_count / self.max_steps), self.final_tau)
#         self.tau_step_value = tau
#
#         with torch.no_grad():
#             for target_param, policy_param in zip(target_model.parameters(), policy_model.parameters()):
#                 target_param.copy_(tau * policy_param + (1.0 - tau) * target_param)
#
#         logging.debug(f'Model was updated via soft update. tau: {tau}; step count: {self.step_count}')
#
#     def save(self, filepath: str) -> None:
#         """
#         Save the current policy network and optimizer state to a file.
#
#         Args:
#             filepath (str): The path to save the state to.
#         """
#
#         # Save torch networks in filepath destination
#         state = {
#             'policy_net': self.policy_net.state_dict(),
#             'optimizer': self.optimizer.state_dict()
#         }
#         torch.save(state, filepath)
#
#     def load(self, filepath: str, map_location: torch.device = None) -> None:
#         """
#         Load the policy network and optimizer state from a file.
#
#         Args:
#             :param filepath: The path to load the state from.
#             :param map_location: If set this parameter is used to set a new device location while loading the model
#         """
#         if map_location:
#             state = torch.load(filepath, map_location=map_location)
#         else:
#             state = torch.load(filepath)
#
#         self.policy_net.load_state_dict(state['policy_net'])
#         self.optimizer.load_state_dict(state['optimizer'])
#
#     def reward_update(self, reward: float, punish: bool = False) -> float:
#         """
#         Adjust the reward based on an exponentially increasing or decreasing factor of the score.
#
#         Args:
#             reward (float): The current reward.
#             punish (bool): Whether to apply a punishment (decrease the reward). Default is False.
#
#         Returns:
#             float: The adjusted reward.
#         """
#
#         # Set base factor according to configuration
#         if punish:
#             base_factor = self.punish_factor
#         else:
#             base_factor = self.reward_factor
#
#         # Apply factor to reward
#         if punish:
#             adjusted_reward = reward / base_factor
#         else:
#             adjusted_reward = reward * base_factor
#
#         return adjusted_reward

