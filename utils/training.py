import os
from datetime import datetime
import math
import numpy as np
import pandas as pd
from utils import FrameProcessor
import torch
import logging
from typing import List, Dict
import yaml
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler


class AgentOptimizer:
    """
    A class for training the Deep Q-Network (DQN) agent.
    """

    def __init__(self, agent, env, hyperparameter: Dict, device: torch.device):
        """
        Initialize the AgentTrainer.

        Args:
            agent (DeepQNetworkAgent): The DQN agent to train.
            env (gym.Env): The environment to train the agent on.
            hyperparameter (Dict): The hyperparameters for training.
            device (torch.device): The device on which the agent gets training for clearing memory.
        """
        self.agent = agent
        self.env = env
        self.fp = FrameProcessor()
        self.hyper_params = hyperparameter
        self.scores: List[float] = []  # Type hint for scores
        self.losses: List[float] = []  # Type hint for losses
        self.device = device
        self.n_episodes: int = self.hyper_params["n_episodes"]  # Type hint for n_episodes

    def epsilon_decay(self, episode: int) -> float:
        """
        Calculate the epsilon value for a given episode based on exponential decay.

        Args:
            episode (int): The current epsilon.

        Returns:
            float: The epsilon value for the given episode.
        """
        return self.hyper_params['epsilon_end'] + (self.hyper_params['epsilon_start'] - self.hyper_params['epsilon_end']) * math.exp(
            -self.hyper_params['epsilon_decay'] * episode)

    def train(self, output_dir: str = './output') -> None:
        """
        Train the agent for a specified number of episodes.

        Args:
            output_dir (str): The directory to save the output files. Default is 'output'.
        """
        metrics_data: List[Dict] = []  # Initialize an empty list to store metrics data
        eps = self.hyper_params['epsilon_start']

        # init pre-episode scores
        best_score = 0
        mvg_avg_score = 0
        mvg_avg_loss = 0

        # Save agent hyperparameter configuration in filepath destination
        with open(output_dir + f'/metrics/{self.agent.agent_model}_agent_hyper_params.yaml', 'w') as yaml_file:
            yaml.dump(self.agent.agent_hyper_params, yaml_file, default_flow_style=False)

        # Save network hyperparameter configuration in filepath destination
        with open(output_dir + f'/metrics/{self.agent.agent_model}_network_hyper_params.yaml', 'w') as yaml_file:
            yaml.dump(self.agent.network_hyper_params, yaml_file, default_flow_style=False)

        for episode in range(1, self.n_episodes + 1):
            # Init relevant variables for episode
            score = 0
            loss = 0
            eps = self.epsilon_decay(episode=episode)
            count_predicted_actions = 0
            count_random_actions = 0

            # Preprocess the initial state
            state = self.fp.preprocess(stacked_frames=None,
                                       env_state=self.env.reset()[0],
                                       exclude=(8, -12, -12, 4),
                                       output=84,
                                       is_new=True)

            while True:
                # Select an action based on the current state
                action, action_type = self.agent.act(state, eps)
                if action_type == 'predicted':
                    count_predicted_actions += 1
                elif action_type == 'randomized':
                    count_random_actions += 1

                # Take the action and observe the next state, reward, and termination flags
                next_state, reward, terminated, truncated, info = self.env.step(action)

                # Preprocess the next state
                next_state = self.fp.preprocess(stacked_frames=state,
                                                env_state=next_state,
                                                exclude=(8, -12, -12, 4),
                                                output=84,
                                                is_new=False)

                # Update the agent with the observed transition
                loss += self.agent.step(state=state,
                                        action=action,
                                        reward=reward,
                                        next_state=next_state,
                                        terminated=terminated,
                                        truncated=truncated)

                logging.debug(f'STEP LOSS FOR DEBUG REVIEW: {loss:.2f}')

                state = next_state
                score += reward

                if terminated:
                    break

            # Update scores and losses arrays
            self.scores.append(score)
            self.losses.append(loss)

            # calc moving average score
            if len(self.scores) >= 20:
                mvg_avg_score = np.mean(self.scores[-20:])

            # calc moving average loss
            if len(self.losses) >= 20:
                mvg_avg_loss = np.mean(self.losses[-20:])

            # average_score = np.mean(self.scores_window)
            logging.info(f'Episode: {episode}, '
                         f'Average Score: {np.mean(self.scores):.2f}, '
                         f'Average Loss: {np.mean(self.losses):.2f}, '
                         f'Moving Average Score: {mvg_avg_score:.2f}, '
                         f'Moving Average Loss: {mvg_avg_loss:.2f}, '
                         f'Episode Score: {score:.2f}, '
                         f'Episode Loss: {loss:.2f}, '
                         f'Epsilon: {eps:.2f}, '
                         f'Predicted: {count_predicted_actions}, '
                         f'Randomized: {count_random_actions}')

            # Append current scores to collect metrics data
            metrics_data.append({'episode': episode,
                                 'epsilon': eps,
                                 'avg_episode_score': np.mean(self.scores),
                                 'avg_episode_loss': np.mean(self.losses),
                                 'mvg_avg_score': mvg_avg_score,
                                 'mvg_avg_loss': mvg_avg_loss,
                                 'episode_score': score,
                                 'episode_loss': loss,
                                 'count_predicted_actions': count_predicted_actions,
                                 'count_random_action': count_random_actions,
                                 'model_saved': mvg_avg_score > best_score})

            if mvg_avg_score > best_score:
                best_score = mvg_avg_score
                model_path = self.get_file_path(output_dir + '/models', f'best_model_score.pth')
                self.agent.save(model_path)  # Save the best model
                logging.info(f'New best model saved with score: {best_score:.2f}')

            # Save metrics to Parquet file
            df_metrics = pd.DataFrame(metrics_data)
            df_metrics.to_parquet(self.get_file_path(output_dir + '/metrics', 'metrics.pq'), index=False)

        # Clear cache based on the device
        if self.device == torch.device("cuda"):
            torch.cuda.empty_cache()
            logging.info(f"Cleared cache for: {self.device}")
        elif self.device == torch.device("mps"):
            torch.mps.empty_cache()
            logging.info(f"Cleared cache for: {self.device}")

    def get_file_path(self, output_dir: str, filename: str) -> str:
        """
        Get the file path with the current date and the model name.

        Args:
            output_dir (str): The directory to save the file.
            filename (str): The base filename.

        Returns:
            str: The file path with the current date and the model name.
        """
        current_date = datetime.now().strftime('%Y%m%d')
        file_path = os.path.join(output_dir, f'{current_date}_{self.agent.agent_model}_{filename}')
        return file_path


class AgentOptimizerv2:
    """
    A class for training the Deep Q-Network (DQN) agent.
    """

    def __init__(self, agent, env, hyperparameter: Dict, device: torch.device, network_hyper_params: Dict):
        """
        Initialize the AgentTrainer.

        Args:
            agent (DeepQNetworkAgent): The DQN agent to train.
            env (gym.Env): The environment to train the agent on.
            hyperparameter (Dict): The hyperparameters for training.
            device (torch.device): The device on which the agent gets training for clearing memory.
            network_hyper_params: (Dict): The hyperparameters defined for the network to ensure correct data shaping
            in preprocessing.
        """
        self.agent = agent
        self.env = env
        self.fp = FrameProcessor()
        self.hyper_params = hyperparameter
        self.scores: List[float] = []  # Type hint for scores
        self.losses: List[float] = []  # Type hint for losses
        self.device = device
        self.n_episodes: int = self.hyper_params["n_episodes"]  # Type hint for n_episodes
        self.network_hyper_params = network_hyper_params

    def epsilon_decay(self, episode: int) -> float:
        """
        Calculate the epsilon value for a given episode based on exponential decay.

        Args:
            episode (int): The current epsilon.

        Returns:
            float: The epsilon value for the given episode.
        """
        return self.hyper_params['epsilon_end'] + (self.hyper_params['epsilon_start'] - self.hyper_params['epsilon_end']) * math.exp(
            -self.hyper_params['epsilon_decay'] / 5 * episode)

    def epsilon_decay_linear(self, episode: int) -> float:
        """
        Calculate the epsilon value for a given episode based on linear decay.

        Args:
            episode (int): The current episode.

        Returns:
            float: The epsilon value for the given episode.
        """
        decay_rate = (self.hyper_params['epsilon_start'] - self.hyper_params['epsilon_end']) / self.n_episodes
        return max(self.hyper_params['epsilon_end'], self.hyper_params['epsilon_start'] - decay_rate * episode)

    def train(self, output_dir: str = './output') -> None:
        """
        Train the agent for a specified number of episodes.

        Args:
            output_dir (str): The directory to save the output files. Default is 'output'.
        """
        metrics_data: List[Dict] = []  # Initialize an empty list to store metrics data
        eps = self.hyper_params['epsilon_start']

        # initalize scaler for reward scaling to achieve consistent training situation (scaling from -1 to 1)
        mmscaler = MinMaxScaler(feature_range=(0, 1))
        mmscaler.fit(np.array([0, 1000]).reshape(-1, 1))

        # get expected picture shape for preprocessing correctly
        output_size = self.network_hyper_params['input_shape'][1]

        # init pre-episode scores
        best_score = 0
        mvg_avg_score = 0
        mvg_avg_loss = 0

        # Save agent hyperparameter configuration in filepath destination
        with open(output_dir + f'/metrics/{self.agent.agent_model}_agent_hyper_params.yaml', 'w') as yaml_file:
            yaml.dump(self.agent.agent_hyper_params, yaml_file, default_flow_style=False)

        # Save network hyperparameter configuration in filepath destination
        with open(output_dir + f'/metrics/{self.agent.agent_model}_network_hyper_params.yaml', 'w') as yaml_file:
            yaml.dump(self.agent.network_hyper_params, yaml_file, default_flow_style=False)

        for episode in range(1, self.n_episodes + 1):
            # Init relevant variables for episode
            previous_action = 0
            score = 0
            loss = 0
            eps = self.epsilon_decay(episode=episode)
            count_predicted_actions = 0
            count_random_actions = 0

            # Preprocess the initial state
            state = self.fp.preprocess(stacked_frames=None,
                                       env_state=self.env.reset()[0],
                                       exclude=(8, -12, -12, 4),
                                       output=output_size,
                                       is_new=True)

            while True:
                # Select an action based on the current state
                action, action_type = self.agent.act(state, eps)
                if action_type == 'predicted':
                    count_predicted_actions += 1
                elif action_type == 'randomized':
                    count_random_actions += 1

                # Take the action and observe the next state, reward, and termination flags
                next_state, reward, terminated, truncated, info = self.env.step(action)
                logging.debug(f"reward before scaling: {reward}")

                # Normalize the reward
                reward = mmscaler.transform([[reward]])[0, 0]
                logging.debug(f"reward after scaling: {reward}")

                # Preprocess the next state
                next_state = self.fp.preprocess(stacked_frames=state,
                                                env_state=next_state,
                                                exclude=(8, -12, -12, 4),
                                                output=output_size,
                                                is_new=False)

                # Update the agent with the observed transition
                loss += self.agent.step(state=state,
                                        action=action,
                                        reward=reward,
                                        next_state=next_state,
                                        terminated=terminated,
                                        truncated=truncated,
                                        previous_reward=score,
                                        previous_action=previous_action,)

                logging.debug(f'STEP LOSS FOR DEBUG REVIEW: {loss:.2f}')

                state = next_state
                score += reward
                previous_action = action

                if terminated:
                    break

            # Update scores and losses arrays
            self.scores.append(score)
            self.losses.append(loss)

            # calc moving average score
            if len(self.scores) >= 20:
                mvg_avg_score = np.mean(self.scores[-20:])

            # calc moving average loss
            if len(self.losses) >= 20:
                mvg_avg_loss = np.mean(self.losses[-20:])

            # average_score = np.mean(self.scores_window)
            logging.info(f'Episode: {episode}, '
                         f'Average Score: {np.mean(self.scores):.5f}, '
                         f'Average Loss: {np.mean(self.losses):.5f}, '
                         f'Moving Average Score: {mvg_avg_score:.5f}, '
                         f'Moving Average Loss: {mvg_avg_loss:.5f}, '
                         f'Episode Score: {score:.5f}, '
                         f'Episode Loss: {loss:.5f}, '
                         f'Epsilon: {eps:.5f}, '
                         f'Predicted: {count_predicted_actions}, '
                         f'Randomized: {count_random_actions}')

            # Append current scores to collect metrics data
            metrics_data.append({'episode': episode,
                                 'epsilon': eps,
                                 'avg_episode_score': np.mean(self.scores),
                                 'avg_episode_loss': np.mean(self.losses),
                                 'mvg_avg_score': mvg_avg_score,
                                 'mvg_avg_loss': mvg_avg_loss,
                                 'episode_score': score,
                                 'episode_loss': loss,
                                 'count_predicted_actions': count_predicted_actions,
                                 'count_random_action': count_random_actions,
                                 'model_saved': mvg_avg_score > best_score})

            if mvg_avg_score > best_score:
                best_score = mvg_avg_score
                model_path = self.get_file_path(output_dir + '/models', f'best_model_episode_{episode}_score_{round(best_score,5)}.pth')
                self.agent.save(model_path)  # Save the best model
                logging.info(f'New best model saved with score: {best_score:.2f}')

            # Save metrics to Parquet file
            df_metrics = pd.DataFrame(metrics_data)
            df_metrics.to_parquet(self.get_file_path(output_dir + '/metrics', 'metrics.pq'), index=False)

        # Clear cache based on the device
        if self.device == torch.device("cuda"):
            torch.cuda.empty_cache()
            logging.info(f"Cleared cache for: {self.device}")
        elif self.device == torch.device("mps"):
            torch.mps.empty_cache()
            logging.info(f"Cleared cache for: {self.device}")

    def get_file_path(self, output_dir: str, filename: str) -> str:
        """
        Get the file path with the current date and the model name.

        Args:
            output_dir (str): The directory to save the file.
            filename (str): The base filename.

        Returns:
            str: The file path with the current date and the model name.
        """
        current_date = datetime.now().strftime('%Y%m%d')
        file_path = os.path.join(output_dir, f'{current_date}_{self.agent.agent_model}_{filename}')
        return file_path


class AgentOptimizerv3:
    """
    A class for training the Deep Q-Network (DQN) agent.
    """

    def __init__(self, agent, env, hyperparameter: Dict, device: torch.device, network_hyper_params: Dict):
        """
        Initialize the AgentTrainer.

        Args:
            agent (DeepQNetworkAgent): The DQN agent to train.
            env (gym.Env): The environment to train the agent on.
            hyperparameter (Dict): The hyperparameters for training.
            device (torch.device): The device on which the agent gets training for clearing memory.
            network_hyper_params: (Dict): The hyperparameters defined for the network to ensure correct data shaping
            in preprocessing.
        """
        self.agent = agent
        self.env = env
        self.fp = FrameProcessor()
        self.hyper_params = hyperparameter
        self.scores: List[float] = []  # Type hint for scores
        self.losses: List[float] = []  # Type hint for losses
        self.device = device
        self.n_episodes: int = self.hyper_params["n_episodes"]  # Type hint for n_episodes
        self.network_hyper_params = network_hyper_params

    def epsilon_decay(self, episode: int) -> float:
        """
        Calculate the epsilon value for a given episode based on exponential decay.

        Args:
            episode (int): The current epsilon.

        Returns:
            float: The epsilon value for the given episode.
        """
        return self.hyper_params['epsilon_end'] + (self.hyper_params['epsilon_start'] - self.hyper_params['epsilon_end']) * math.exp(
            -self.hyper_params['epsilon_decay'] / 5 * episode)

    def epsilon_decay_linear(self, episode: int) -> float:
        """
        Calculate the epsilon value for a given episode based on linear decay.

        Args:
            episode (int): The current episode.

        Returns:
            float: The epsilon value for the given episode.
        """
        decay_rate = (self.hyper_params['epsilon_start'] - self.hyper_params['epsilon_end']) / self.n_episodes
        return max(self.hyper_params['epsilon_end'], self.hyper_params['epsilon_start'] - decay_rate * episode)

    def train(self, output_dir: str = './output') -> None:
        """
        Train the agent for a specified number of episodes.

        Args:
            output_dir (str): The directory to save the output files. Default is 'output'.
        """
        metrics_data: List[Dict] = []  # Initialize an empty list to store metrics data
        eps = self.hyper_params['epsilon_start']

        # initalize scaler for reward scaling to achieve consistent training situation (scaling from -1 to 1)
        mscaler = MaxAbsScaler()
        mscaler.fit(np.array([0, 1000]).reshape(-1, 1))

        # get expected picture shape for preprocessing correctly
        output_size = self.network_hyper_params['input_shape'][1]

        # init pre-episode scores
        best_score = 0
        mvg_avg_score = 0
        mvg_avg_loss = 0

        # Save agent hyperparameter configuration in filepath destination
        with open(output_dir + f'/metrics/{self.agent.agent_model}_agent_hyper_params.yaml', 'w') as yaml_file:
            yaml.dump(self.agent.agent_hyper_params, yaml_file, default_flow_style=False)

        # Save network hyperparameter configuration in filepath destination
        with open(output_dir + f'/metrics/{self.agent.agent_model}_network_hyper_params.yaml', 'w') as yaml_file:
            yaml.dump(self.agent.network_hyper_params, yaml_file, default_flow_style=False)

        for episode in range(1, self.n_episodes + 1):
            # Init relevant variables for episode
            previous_action = 0
            score = 0
            loss = 0
            eps = self.epsilon_decay(episode=episode)
            count_predicted_actions = 0
            count_random_actions = 0

            # Preprocess the initial state
            state = self.fp.preprocess(stacked_frames=None,
                                       env_state=self.env.reset()[0],
                                       exclude=(8, -12, -12, 4),
                                       output=output_size,
                                       is_new=True)

            while True:
                # Select an action based on the current state
                action, action_type = self.agent.act(state, eps)
                if action_type == 'predicted':
                    count_predicted_actions += 1
                elif action_type == 'randomized':
                    count_random_actions += 1

                # Take the action and observe the next state, reward, and termination flags
                next_state, reward, terminated, truncated, info = self.env.step(action)
                logging.debug(f"reward before scaling: {reward}")
                # if reward > 0:
                #     print(f"reward before scaling: {reward}")

                # Normalize the reward
                reward = mscaler.fit_transform([[reward]])[0, 0]
                # if reward > 0:
                #     print(f"reward after scaling: {reward}")
                logging.debug(f"reward after scaling: {reward}")

                # Preprocess the next state
                next_state = self.fp.preprocess(stacked_frames=state,
                                                env_state=next_state,
                                                exclude=(8, -12, -12, 4),
                                                output=output_size,
                                                is_new=False)

                # Update the agent with the observed transition
                updated_loss, updated_reward = self.agent.step(state=state,
                                                               action=action,
                                                               reward=reward,
                                                               next_state=next_state,
                                                               terminated=terminated,
                                                               truncated=truncated,
                                                               previous_reward=score,
                                                               previous_action=previous_action,)

                loss += updated_loss
                logging.debug(f'STEP LOSS FOR DEBUG REVIEW: {loss:.2f}')

                state = next_state
                score += updated_reward
                previous_action = action

                if terminated:
                    break

            # Update scores and losses arrays
            self.scores.append(score)
            self.losses.append(loss)

            # calc moving average score
            if len(self.scores) >= 20:
                mvg_avg_score = np.mean(self.scores[-20:])

            # calc moving average loss
            if len(self.losses) >= 20:
                mvg_avg_loss = np.mean(self.losses[-20:])

            # average_score = np.mean(self.scores_window)
            logging.info(f'Episode: {episode}, '
                         f'Average Score: {np.mean(self.scores):.5f}, '
                         f'Average Loss: {np.mean(self.losses):.5f}, '
                         f'Moving Average Score: {mvg_avg_score:.5f}, '
                         f'Moving Average Loss: {mvg_avg_loss:.5f}, '
                         f'Episode Score: {score:.5f}, '
                         f'Episode Loss: {loss:.5f}, '
                         f'Epsilon: {eps:.5f}, '
                         f'Predicted: {count_predicted_actions}, '
                         f'Randomized: {count_random_actions}')

            # Append current scores to collect metrics data
            metrics_data.append({'episode': episode,
                                 'epsilon': eps,
                                 'avg_episode_score': np.mean(self.scores),
                                 'avg_episode_loss': np.mean(self.losses),
                                 'mvg_avg_score': mvg_avg_score,
                                 'mvg_avg_loss': mvg_avg_loss,
                                 'episode_score': score,
                                 'episode_loss': loss,
                                 'count_predicted_actions': count_predicted_actions,
                                 'count_random_action': count_random_actions,
                                 'model_saved': mvg_avg_score > best_score})

            if mvg_avg_score > best_score:
                best_score = mvg_avg_score
                model_path = self.get_file_path(output_dir + '/models', f'best_model_episode_{episode}_score_{round(best_score,5)}.pth')
                self.agent.save(model_path)  # Save the best model
                logging.info(f'New best model saved with score: {best_score:.2f}')

            # Save metrics to Parquet file
            df_metrics = pd.DataFrame(metrics_data)
            df_metrics.to_parquet(self.get_file_path(output_dir + '/metrics', 'metrics.pq'), index=False)

        # Clear cache based on the device
        if self.device == torch.device("cuda"):
            torch.cuda.empty_cache()
            logging.info(f"Cleared cache for: {self.device}")
        elif self.device == torch.device("mps"):
            torch.mps.empty_cache()
            logging.info(f"Cleared cache for: {self.device}")

    def get_file_path(self, output_dir: str, filename: str) -> str:
        """
        Get the file path with the current date and the model name.

        Args:
            output_dir (str): The directory to save the file.
            filename (str): The base filename.

        Returns:
            str: The file path with the current date and the model name.
        """
        current_date = datetime.now().strftime('%Y%m%d')
        file_path = os.path.join(output_dir, f'{current_date}_{self.agent.agent_model}_{filename}')
        return file_path


class AgentOptimizerv4:
    """
    A class for training the Deep Q-Network (DQN) agent.
    """

    def __init__(self, agent, env, hyperparameter: Dict, device: torch.device, network_hyper_params: Dict):
        """
        Initialize the AgentTrainer.

        Args:
            agent (DeepQNetworkAgent): The DQN agent to train.
            env (gym.Env): The environment to train the agent on.
            hyperparameter (Dict): The hyperparameters for training.
            device (torch.device): The device on which the agent gets training for clearing memory.
            network_hyper_params: (Dict): The hyperparameters defined for the network to ensure correct data shaping
            in preprocessing.
        """
        self.agent = agent
        self.env = env
        self.fp = FrameProcessor()
        self.hyper_params = hyperparameter
        self.scores: List[float] = []  # Type hint for scores
        self.losses: List[float] = []  # Type hint for losses
        self.device = device
        self.n_episodes: int = self.hyper_params["n_episodes"]  # Type hint for n_episodes
        self.network_hyper_params = network_hyper_params

    def epsilon_decay(self, episode: int) -> float:
        """
        Calculate the epsilon value for a given episode based on exponential decay.

        Args:
            episode (int): The current epsilon.

        Returns:
            float: The epsilon value for the given episode.
        """
        return self.hyper_params['epsilon_end'] + (self.hyper_params['epsilon_start'] - self.hyper_params['epsilon_end']) * math.exp(
            -self.hyper_params['epsilon_decay'] / 5 * episode)

    def epsilon_decay_linear(self, episode: int) -> float:
        """
        Calculate the epsilon value for a given episode based on linear decay.

        Args:
            episode (int): The current episode.

        Returns:
            float: The epsilon value for the given episode.
        """
        decay_rate = (self.hyper_params['epsilon_start'] - self.hyper_params['epsilon_end']) / self.n_episodes
        return max(self.hyper_params['epsilon_end'], self.hyper_params['epsilon_start'] - decay_rate * episode)

    def train(self, output_dir: str = './output') -> None:
        """
        Train the agent for a specified number of episodes.

        Args:
            output_dir (str): The directory to save the output files. Default is 'output'.
        """
        metrics_data: List[Dict] = []  # Initialize an empty list to store metrics data
        eps = self.hyper_params['epsilon_start']

        # initalize scaler for reward scaling to achieve consistent training situation (scaling from -1 to 1)
        mscaler = MaxAbsScaler()
        mscaler.fit(np.array([0, 1000]).reshape(-1, 1))

        # get expected picture shape for preprocessing correctly
        output_size = self.network_hyper_params['input_shape'][1]

        # init pre-episode scores
        best_score = 0
        mvg_avg_score = 0
        mvg_avg_loss = 0

        # Save agent hyperparameter configuration in filepath destination
        with open(output_dir + f'/metrics/{self.agent.agent_model}_agent_hyper_params.yaml', 'w') as yaml_file:
            yaml.dump(self.agent.agent_hyper_params, yaml_file, default_flow_style=False)

        # Save network hyperparameter configuration in filepath destination
        with open(output_dir + f'/metrics/{self.agent.agent_model}_network_hyper_params.yaml', 'w') as yaml_file:
            yaml.dump(self.agent.network_hyper_params, yaml_file, default_flow_style=False)

        for episode in range(1, self.n_episodes + 1):
            # Init relevant variables for episode
            previous_action = 0
            score = 0
            loss = 0
            eps = self.epsilon_decay(episode=episode)
            count_predicted_actions = 0
            count_random_actions = 0

            # Preprocess the initial state
            state = self.fp.preprocess(stacked_frames=None,
                                       env_state=self.env.reset()[0],
                                       exclude=(8, -12, -12, 4),
                                       output=output_size,
                                       is_new=True)

            while True:
                # Select an action based on the current state
                action, action_type = self.agent.act(state, eps)
                if action_type == 'predicted':
                    count_predicted_actions += 1
                elif action_type == 'randomized':
                    count_random_actions += 1

                # Take the action and observe the next state, reward, and termination flags
                next_state, reward, terminated, truncated, info = self.env.step(action)
                logging.debug(f"reward before scaling: {reward}")
                # if reward > 0:
                #     print(f"reward before scaling: {reward}")

                # Normalize the reward
                reward = mscaler.fit_transform([[reward]])[0, 0]
                # if reward > 0:
                #     print(f"reward after scaling: {reward}")
                logging.debug(f"reward after scaling: {reward}")

                # Preprocess the next state
                next_state = self.fp.preprocess(stacked_frames=state,
                                                env_state=next_state,
                                                exclude=(8, -12, -12, 4),
                                                output=output_size,
                                                is_new=False)

                # Update the agent with the observed transition
                updated_loss, updated_reward = self.agent.step(state=state,
                                                               action=action,
                                                               reward=reward,
                                                               next_state=next_state,
                                                               terminated=terminated,
                                                               truncated=truncated,
                                                               previous_reward=score,
                                                               previous_action=previous_action,)

                loss += updated_loss
                logging.debug(f'STEP LOSS FOR DEBUG REVIEW: {loss:.2f}')

                state = next_state
                score += updated_reward
                previous_action = action

                if terminated:
                    break

            # Update scores and losses arrays
            self.scores.append(score)
            self.losses.append(loss)

            # calc moving average score
            if len(self.scores) >= 20:
                mvg_avg_score = np.mean(self.scores[-20:])

            # calc moving average loss
            if len(self.losses) >= 20:
                mvg_avg_loss = np.mean(self.losses[-20:])

            # average_score = np.mean(self.scores_window)
            logging.info(f'Episode: {episode}, '
                         f'Average Score: {np.mean(self.scores):.5f}, '
                         f'Average Loss: {np.mean(self.losses):.5f}, '
                         f'Moving Average Score: {mvg_avg_score:.5f}, '
                         f'Moving Average Loss: {mvg_avg_loss:.5f}, '
                         f'Episode Score: {score:.5f}, '
                         f'Episode Loss: {loss:.5f}, '
                         f'Epsilon: {eps:.5f}, '
                         f'Predicted: {count_predicted_actions}, '
                         f'Randomized: {count_random_actions}, '
                         f'Total No. Steps: {count_predicted_actions + count_random_actions}, ')

            # Append current scores to collect metrics data
            metrics_data.append({'episode': episode,
                                 'epsilon': eps,
                                 'avg_episode_score': np.mean(self.scores),
                                 'avg_episode_loss': np.mean(self.losses),
                                 'mvg_avg_score': mvg_avg_score,
                                 'mvg_avg_loss': mvg_avg_loss,
                                 'episode_score': score,
                                 'episode_loss': loss,
                                 'count_predicted_actions': count_predicted_actions,
                                 'count_random_actions': count_random_actions,
                                 'total_no_steps': count_predicted_actions + count_random_actions,
                                 'model_saved': mvg_avg_score > best_score})

            if mvg_avg_score > best_score:
                best_score = mvg_avg_score
                model_path = self.get_file_path(output_dir + '/models', f'best_model_episode_{episode}_score_{round(best_score,5)}.pth')
                self.agent.save(model_path)  # Save the best model
                logging.info(f'New best model saved with score: {best_score:.2f}')

            # Save metrics to Parquet file
            df_metrics = pd.DataFrame(metrics_data)
            df_metrics.to_parquet(self.get_file_path(output_dir + '/metrics', 'metrics.pq'), index=False)

        # Clear cache based on the device
        if self.device == torch.device("cuda"):
            torch.cuda.empty_cache()
            logging.info(f"Cleared cache for: {self.device}")
        elif self.device == torch.device("mps"):
            torch.mps.empty_cache()
            logging.info(f"Cleared cache for: {self.device}")

    def get_file_path(self, output_dir: str, filename: str) -> str:
        """
        Get the file path with the current date and the model name.

        Args:
            output_dir (str): The directory to save the file.
            filename (str): The base filename.

        Returns:
            str: The file path with the current date and the model name.
        """
        current_date = datetime.now().strftime('%Y%m%d')
        file_path = os.path.join(output_dir, f'{current_date}_{self.agent.agent_model}_{filename}')
        return file_path

