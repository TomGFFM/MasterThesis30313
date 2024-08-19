# import standards
from datetime import datetime
import gc
import logging
import numpy as np
import math
import os
import pandas as pd
from typing import List, Dict
import yaml

# torch and gym related, other ML
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler

# hyperparameter tuning
import optuna
from optuna import Trial

# custom
from utils import FrameProcessor


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
        # mscaler = MinMaxScaler(feature_range=(0, 1))
        # mscaler.fit(np.array([0, 1000]).reshape(-1, 1))

        # get expected picture shape for preprocessing correctly
        output_size = self.network_hyper_params['input_shape'][1]

        # init pre-episode scores and other metrics
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
            taus_in_episode = []
            lrs_in_episode = []
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
                updated_loss, updated_reward, step_tau, step_lr = self.agent.step(state=state,
                                                                                  action=action,
                                                                                  reward=reward,
                                                                                  next_state=next_state,
                                                                                  terminated=terminated,
                                                                                  truncated=truncated,
                                                                                  previous_reward=score,
                                                                                  previous_action=previous_action,
                                                                                  episode=episode,)

                loss += updated_loss
                taus_in_episode.append(step_tau)
                lrs_in_episode.append(step_lr)
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
                         f'Total No. Steps: {count_predicted_actions + count_random_actions}, '
                         f'Min. Tau in Episode: {min(taus_in_episode):.5f}, '
                         f'Min. LR in Episode: {min(lrs_in_episode):.5f}, ')

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
                                 'model_saved': mvg_avg_score > best_score,
                                 'minimum_tau_in_episode': min(taus_in_episode),
                                 'minimum_lr_in_episode': min(lrs_in_episode)})

            if mvg_avg_score > best_score:
                best_score = mvg_avg_score
                model_path = self.get_file_path(output_dir + '/models', f'best_model_episode_{episode}_score_{round(best_score,5)}.pth')
                self.agent.save(model_path)  # Save the best model
                logging.info(f'New best model saved with score: {best_score:.2f}')

            # Save non q-metrics to Parquet file
            df_metrics = pd.DataFrame(metrics_data)
            df_metrics.to_parquet(self.get_file_path(output_dir + '/metrics', 'metrics.pq'), index=False)

            # Save full set of q-metrics to Parquet file
            self.agent.q_value_metrics.to_parquet(self.get_file_path(output_dir + '/metrics', 'q_metrics.pq'), index=False)

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


class AgentOptimizerv5:
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
        # mscaler = MinMaxScaler(feature_range=(0, 1))
        # mscaler.fit(np.array([0, 1000]).reshape(-1, 1))

        # get expected picture shape for preprocessing correctly
        output_size = self.network_hyper_params['input_shape'][1]

        # init pre-episode scores and other metrics
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
            taus_in_episode = []
            lrs_in_episode = []
            eps = self.epsilon_decay(episode=episode)
            count_predicted_actions = 0
            count_random_actions = 0

            # create index per frame for positional encoding creation
            # works only with classes which can handle the index for the encoding creation
            frame_index = 0

            # Preprocess the initial state
            state = self.fp.preprocess(stacked_frames=None,
                                       env_state=self.env.reset()[0],
                                       exclude=(8, -12, -12, 4),
                                       output=output_size,
                                       is_new=True)

            while True:
                # Select an action based on the current state
                action, action_type = self.agent.act(state=state,
                                                     eps=eps,
                                                     frame_index=frame_index)
                if action_type == 'predicted':
                    count_predicted_actions += 1
                elif action_type == 'randomized':
                    count_random_actions += 1

                # Take the action and observe the next state, reward, and termination flags
                next_state, reward, terminated, truncated, info = self.env.step(action)
                logging.debug(f"reward before scaling: {reward}")

                # Normalize the reward
                reward = mscaler.fit_transform([[reward]])[0, 0]
                logging.debug(f"reward after scaling: {reward}")

                # Preprocess the next state
                next_state = self.fp.preprocess(stacked_frames=state,
                                                env_state=next_state,
                                                exclude=(8, -12, -12, 4),
                                                output=output_size,
                                                is_new=False)

                # Update the agent with the observed transition
                updated_loss, updated_reward, step_tau, step_lr = self.agent.step(state=state,
                                                                                  action=action,
                                                                                  reward=reward,
                                                                                  next_state=next_state,
                                                                                  terminated=terminated,
                                                                                  truncated=truncated,
                                                                                  previous_reward=score,
                                                                                  previous_action=previous_action,
                                                                                  episode=episode,
                                                                                  frame_index=frame_index)

                loss += updated_loss
                taus_in_episode.append(step_tau)
                lrs_in_episode.append(step_lr)
                state = next_state
                score += updated_reward
                previous_action = action
                frame_index += 1

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
                         f'Total No. Steps: {count_predicted_actions + count_random_actions}, '
                         f'Min. Tau in Episode: {min(taus_in_episode):.5f}, '
                         f'Min. LR in Episode: {min(lrs_in_episode):.5f}, ')

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
                                 'model_saved': mvg_avg_score > best_score and episode > self.hyper_params['learn_start'],
                                 'minimum_tau_in_episode': min(taus_in_episode),
                                 'minimum_lr_in_episode': min(lrs_in_episode)})

            if mvg_avg_score > best_score and episode > self.hyper_params['learn_start']:
                best_score = mvg_avg_score
                model_path = self.get_file_path(output_dir + '/models', f'best_model_episode_{episode}_score_{round(best_score,5)}.pth')
                self.agent.save(model_path)  # Save the best model
                logging.info(f'New best model saved with score: {best_score:.2f}')

            # Save non q-metrics to Parquet file
            df_metrics = pd.DataFrame(metrics_data)
            df_metrics.to_parquet(self.get_file_path(output_dir + '/metrics', 'metrics.pq'), index=False)

            # Save full set of q-metrics to Parquet file
            self.agent.q_value_metrics.to_parquet(self.get_file_path(output_dir + '/metrics', 'q_metrics.pq'), index=False)

        # Clear cache based on the device
        if self.device == torch.device("cuda"):
            torch.cuda.empty_cache()
            logging.info(f"Cleared cache for: {self.device}")
        elif self.device == torch.device("mps"):
            torch.mps.empty_cache()
            logging.info(f"Cleared cache for: {self.device}")

        gc.collect()

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


class AgentOptimizerv6:
    """
    A class for training the Deep Q-Network (DQN) agent.
    """

    def __init__(self, agent, network: torch.nn.Module, env, hyperparameter: object, device: torch.device, output_dir: str = './output'):
        """
        Initialize the AgentTrainer.

        Args:
            agent (DeepQNetworkAgent): The DQN agent to train.
            network (torch.nn.Module): The network to train.
            env (gym.Env): The environment to train the agent on.
            hyperparameter (object): Object containing full hyperparameter space for training.
            device (torch.device): The device on which the agent gets training for clearing memory.
            output_dir (str): The directory to save the output files. Default is 'output'.
        """
        self.agent_class = agent
        self.network_class = network
        self.env = env
        self.fp = FrameProcessor()
        self.hyperparameter = hyperparameter
        self.scores: List[float] = []  # Type hint for scores
        self.losses: List[float] = []  # Type hint for losses
        self.device = device
        self.output_dir = output_dir
        self.file_ref = ''

    def epsilon_decay(self, episode: int, hyper_params: Dict) -> float:
        """
        Calculate the epsilon value for a given episode based on exponential decay.

        Args:
            episode (int): The current epsilon.
            hyper_params (Dict): The currently set hyperparameters chosen in trial for training.

        Returns:
            float: The epsilon value for the given episode.
        """
        return hyper_params['epsilon_end'] + (hyper_params['epsilon_start'] - hyper_params['epsilon_end']) * math.exp(
            -hyper_params['epsilon_decay'] / 5 * episode)

    def init_agent(self, policy_net: torch.nn.Module, target_net: torch.nn.Module,
                   agent_hyper_params: Dict, network_hyper_params: Dict,) -> object:

        """
            Initializes and returns a Deep Q-Network (DQN) agent with specified configurations.

            Args:
                policy_net (torch.nn.Module): The neural network used as the policy network.
                target_net (torch.nn.Module): The neural network used as the target network.
                agent_hyper_params (Dict): Hyperparameters specific to the agent, including optimizer, learning rate, and other training settings.
                network_hyper_params (Dict): Hyperparameters specific to the network architecture, including input shapes, layer sizes, and more.

            Returns:
                object: An initialized agent instance.

            Raises:
                KeyError: If an unrecognized optimizer, learning rate scheduler, or loss function is specified in `agent_hyper_params`.

            Example:
                agent = init_agent(policy_net, target_net, agent_hyper_params, network_hyper_params)

            Hyperparameters:
                optimizer_name (str): The optimizer to use (options: "Adam", "NAdam", "SGD", "RMSprop").
                learning_rate (float): The learning rate for the optimizer.
                lr_scheduler_name (str): The learning rate scheduler (options: "cosine", "step", None).
                n_episodes (int): The number of episodes for the learning rate scheduler (required if `lr_scheduler_name` is "cosine").
                learning_rate_step_size (int): The step size for the learning rate scheduler (used if `lr_scheduler_name` is "step").
                learning_rate_gamma (float): The gamma value for the learning rate scheduler (used if `lr_scheduler_name` is "step").
                loss_name (str): The loss function to use (options: "huber", "mse", "l1").
            """

        # Init optimizer
        if agent_hyper_params['optimizer_name'] == "Adam":
            optimizer = optim.Adam(policy_net.parameters(), lr=agent_hyper_params['learning_rate'])
        elif agent_hyper_params['optimizer_name'] == "NAdam":
            optimizer = optim.NAdam(policy_net.parameters(), lr=agent_hyper_params['learning_rate'])
        elif agent_hyper_params['optimizer_name'] == "SGD":
            optimizer = optim.SGD(policy_net.parameters(), lr=agent_hyper_params['learning_rate'], momentum=0.9,
                                  nesterov=True)
        elif agent_hyper_params['optimizer_name'] == "Adagrad":
            optimizer = optim.Adagrad(policy_net.parameters(), lr=agent_hyper_params['learning_rate'])
        elif agent_hyper_params['optimizer_name'] == "Adadelta":
            optimizer = optim.Adadelta(policy_net.parameters(), lr=1.0)
        elif agent_hyper_params['optimizer_name'] == "RAdam":
            optimizer = optim.RAdam(policy_net.parameters(), lr=agent_hyper_params['learning_rate'])
        elif agent_hyper_params['optimizer_name'] == "RMSprop":
            optimizer = optim.RMSprop(policy_net.parameters(), lr=agent_hyper_params['learning_rate'])

        # Init lr scheduler
        if agent_hyper_params['lr_scheduler_name'] == "cosine":
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                T_max=agent_hyper_params['n_episodes'],
                                                                eta_min=0.000001)
        elif agent_hyper_params['lr_scheduler_name'] == "step":
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer,
                                                     step_size=agent_hyper_params['learning_rate_step_size'],
                                                     gamma=agent_hyper_params['learning_rate_gamma'])
        elif agent_hyper_params['lr_scheduler_name'] == "reduce_on_plateau":
            lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        else:
            lr_scheduler = None

        # Init loss function
        if agent_hyper_params['loss_name'] == "huber":
            loss_function = F.smooth_l1_loss
        elif agent_hyper_params['loss_name'] == "mse":
            loss_function = F.mse_loss
        else:
            loss_function = F.l1_loss

        # init agent
        agent = self.agent_class(policy_net=policy_net,
                                 target_net=target_net,
                                 action_size=self.env.action_space.n,
                                 device=self.device,
                                 agent_hyper_params=agent_hyper_params,
                                 network_hyper_params=network_hyper_params,
                                 optimizer=optimizer,
                                 lr_scheduler=lr_scheduler,
                                 reward_shaping=True,
                                 reward_factor=1.4,
                                 punish_factor=1.8,
                                 loss_function=loss_function)

        return agent

    def objective(self, trial: Trial) -> float:

        # obtain trial number
        trial_number = trial.number

        # obtain hyperparameters based on trial
        agent_hyper_params, network_hyper_params = self.hyperparameter().get_params(trial)

        # init policy and target Q-Network
        policy_net = self.network_class(**network_hyper_params).to(self.device)
        target_net = self.network_class(**network_hyper_params).to(self.device)

        # set file reference based on trial configuration and network base name
        model_name = policy_net.model_name
        trial_ref = f"trial_{trial_number}_{model_name}"
        file_ref = f"{trial_ref}_{agent_hyper_params['batch_size']}_{agent_hyper_params['learning_rate']:.5f}_{network_hyper_params['num_heads']}heads_{network_hyper_params['num_layers']}layers_tau{agent_hyper_params['tau']:.5f}_optimizer_{agent_hyper_params['optimizer_name']}_scheduler_{agent_hyper_params['lr_scheduler_name']}_lossf_{agent_hyper_params['loss_name']}"

        # init agent
        agent = self.init_agent(policy_net, target_net, agent_hyper_params, network_hyper_params)

        # Initialize an empty list to store metrics data
        metrics_data: List[Dict] = []

        # initalize scaler for reward scaling to achieve consistent training situation (scaling from -1 to 1)
        mscaler = MaxAbsScaler()
        mscaler.fit(np.array([0, 1000]).reshape(-1, 1))

        # get expected picture shape for preprocessing correctly
        output_size = network_hyper_params['input_shape'][1]

        # init pre-episode scores and other metrics
        best_score = 0
        mvg_avg_score = 0
        mvg_avg_loss = 0

        # Save agent hyperparameter configuration in filepath destination
        with open(self.output_dir + f'/metrics/{trial_ref}_agent_hyper_params.yaml', 'w') as yaml_file:
            yaml.dump(agent_hyper_params, yaml_file, default_flow_style=False)

        # Save network hyperparameter configuration in filepath destination
        with open(self.output_dir + f'/metrics/{trial_ref}_network_hyper_params.yaml', 'w') as yaml_file:
            yaml.dump(network_hyper_params, yaml_file, default_flow_style=False)

        for episode in range(1, agent_hyper_params['n_episodes'] + 1):
            # Init relevant variables for episode
            previous_action = 0
            score = 0
            loss = 0
            taus_in_episode = []
            lrs_in_episode = []
            eps = self.epsilon_decay(episode=episode, hyper_params=agent_hyper_params)
            count_predicted_actions = 0
            count_random_actions = 0

            # create index per frame for positional encoding creation
            # works only with classes which can handle the index for the encoding creation
            # frame_index = 0

            # Preprocess the initial state
            state = self.fp.preprocess(stacked_frames=None,
                                       env_state=self.env.reset()[0],
                                       exclude=(8, -12, -12, 4),
                                       output=output_size,
                                       is_new=True)

            while True:
                # Select an action based on the current state
                action, action_type = agent.act(state=state,
                                                eps=eps)
                if action_type == 'predicted':
                    count_predicted_actions += 1
                elif action_type == 'randomized':
                    count_random_actions += 1

                # Take the action and observe the next state, reward, and termination flags
                next_state, reward, terminated, truncated, info = self.env.step(action)
                logging.debug(f"reward before scaling: {reward}")

                # Normalize the reward
                reward = mscaler.fit_transform([[reward]])[0, 0]
                logging.debug(f"reward after scaling: {reward}")

                # Preprocess the next state
                next_state = self.fp.preprocess(stacked_frames=state,
                                                env_state=next_state,
                                                exclude=(8, -12, -12, 4),
                                                output=output_size,
                                                is_new=False)

                # Update the agent with the observed transition
                updated_loss, updated_reward, step_tau, step_lr = agent.step(state=state,
                                                                             action=action,
                                                                             reward=reward,
                                                                             next_state=next_state,
                                                                             terminated=terminated,
                                                                             truncated=truncated,
                                                                             previous_reward=score,
                                                                             previous_action=previous_action,
                                                                             episode=episode)

                loss += updated_loss
                taus_in_episode.append(step_tau)
                lrs_in_episode.append(step_lr)
                state = next_state
                score += updated_reward
                previous_action = action
                # frame_index += 1

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
            logging.info(f'Optuna trial no.: {trial_number}, '
                         f'Episode: {episode}, '
                         f'Average Score: {np.mean(self.scores):.5f}, '
                         f'Average Loss: {np.mean(self.losses):.5f}, '
                         f'Moving Average Score: {mvg_avg_score:.5f}, '
                         f'Moving Average Loss: {mvg_avg_loss:.5f}, '
                         f'Episode Score: {score:.5f}, '
                         f'Episode Loss: {loss:.5f}, '
                         f'Epsilon: {eps:.5f}, '
                         f'Predicted: {count_predicted_actions}, '
                         f'Randomized: {count_random_actions}, '
                         f'Total No. Steps: {count_predicted_actions + count_random_actions}, '
                         f'Min. Tau in Episode: {min(taus_in_episode):.5f}, '
                         f'Min. LR in Episode: {min(lrs_in_episode):.5f}, ')

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
                                 'model_saved': mvg_avg_score > best_score and episode > agent_hyper_params['learn_start'],
                                 'minimum_tau_in_episode': min(taus_in_episode),
                                 'minimum_lr_in_episode': min(lrs_in_episode)})

            if mvg_avg_score > best_score and episode > agent_hyper_params['learn_start']:
                best_score = mvg_avg_score
                model_path = self.get_file_path(self.output_dir + '/models',f'{file_ref}_best_model_episode_{episode}_mvgavgscore_{round(best_score, 5)}.pth')
                agent.save(model_path)  # Save the best model
                logging.info(f'New best model saved with score: {best_score:.2f}')

            # Save non q-metrics to Parquet file
            df_metrics = pd.DataFrame(metrics_data)
            df_metrics.to_parquet(self.get_file_path(self.output_dir + '/metrics', f'{file_ref}_metrics.pq'), index=False)

            # Save full set of q-metrics to Parquet file
            agent.q_value_metrics.to_parquet(self.get_file_path(self.output_dir + '/metrics', f'{file_ref}_q_metrics.pq'), index=False)

            # Report score to optuna
            trial.report(mvg_avg_score, episode)

            # Early stopping und Pruning für Optuna
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            # Clear cache based on the device
            if self.device == torch.device("cuda"):
                torch.cuda.empty_cache()
                logging.debug(f"Cleared cache for: {self.device}")
            elif self.device == torch.device("mps"):
                torch.mps.empty_cache()
                logging.debug(f"Cleared cache for: {self.device}")

            gc.collect()

        # Save final model of trial
        model_path = self.get_file_path(self.output_dir + '/models', f'{file_ref}_final_model.pth')
        agent.save(model_path)

        return mvg_avg_score

    def train(self, n_trials: int = 100, n_jobs: int = 2, warmup_steps: int = 500) -> None:
        """
        Search hyperparameter space with optuna and optimize the reward.

        Args:
            n_trials (int): The number of trials for optimizing the agent reward based on hyperparameters.
            n_jobs (int): The number of parallel jobs for optimizing the agent reward based on hyperparameters.
            warmup_steps (int): The number of warmup steps for optimizing the agent reward based on hyperparameters.
        """

        # set storage path to save study results
        study_storage_path = 'sqlite:///' + self.get_file_path(self.output_dir + '/optuna_study', f'{self.network_class.__name__}.db')

        # Define early stopping via pruning/ set n_warmup_steps to same as learn_start
        pruner = optuna.pruners.MedianPruner(n_startup_trials=1, n_warmup_steps=warmup_steps)

        # Create study
        study = optuna.create_study(study_name=f'{self.network_class.__name__}',
                                    storage=study_storage_path,
                                    direction="maximize",
                                    load_if_exists=True,
                                    pruner=pruner)

        # start optimization
        study.optimize(self.objective, n_trials=n_trials, n_jobs=n_jobs, gc_after_trial=True)

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
        file_path = os.path.join(output_dir, f'{current_date}_{filename}')
        return file_path