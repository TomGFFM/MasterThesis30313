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
from utils import FrameProcessor, FrameProcessorDynamic, OptimizerSelector, LossFunctionSelector, LRSchedulerSelector


class AgentOptimizerClassic:
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

        # initalize scaler for reward scaling to achieve consistent training situation (scaling from -1 to 1)
        mscaler = MaxAbsScaler()
        mscaler.fit(np.array([0, 1000]).reshape(-1, 1))

        # get expected picture shape for preprocessing correctly
        output_size = self.network_hyper_params['input_shape'][1]

        # init pre-episode scores and other metrics
        best_score = 0
        mvg_avg_score = 0
        mvg_avg_loss = 0
        scores: List[float] = []
        losses: List[float] = []
        metrics_data: List[Dict] = []
        action_data: List[Dict] = []

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

            while count_predicted_actions + count_random_actions <= self.hyper_params['max_steps_episode']:
                # Select an action based on the current state
                action, action_type, q_values_sum = self.agent.act(state, eps)

                # Collect action data
                action_data.append({'episode': episode,
                                    'action': action,
                                    'action_type': action_type,
                                    'q_values_sum': q_values_sum})

                # count predicted and randomized actions
                if action_type == 'predicted':
                    count_predicted_actions += 1
                elif action_type == 'randomized':
                    count_random_actions += 1

                # Take the action and observe the next state, reward, and termination flags
                next_state, reward, terminated, truncated, info = self.env.step(action)
                logging.debug(f"reward before scaling: {reward}")

                # Normalize the reward
                reward = mscaler.transform([[reward]])[0, 0]

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
            scores.append(score)
            losses.append(loss)

            # calc moving average score
            if len(scores) >= 20:
                mvg_avg_score = np.mean(scores[-20:])

            # calc moving average loss
            if len(losses) >= 20:
                mvg_avg_loss = np.mean(losses[-20:])

            # log metrics on console
            logging.info(f'Episode: {episode}, '
                         f'Average Score: {np.mean(scores):.5f}, '
                         f'Average Loss: {np.mean(losses):.5f}, '
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
                                 'avg_episode_score': np.mean(scores),
                                 'avg_episode_loss': np.mean(losses),
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

            # save model file if the relevant score has improved
            if mvg_avg_score > best_score and episode >= self.hyper_params['learn_start']:
                best_score = mvg_avg_score
                model_path = self.get_file_path(output_dir + '/models', f'best_model_score_{best_score:.3f}_episode_{episode}.pth')
                self.agent.save(model_path)  # Save the best model
                logging.info(f'New best model saved with score: {best_score:.2f}')

            # Save non q-metrics to Parquet file
            df_metrics = pd.DataFrame(metrics_data)
            df_metrics.to_parquet(self.get_file_path(output_dir + '/metrics', 'metrics.pq'), index=False)

            # Save full set of q-metrics to Parquet file
            self.agent.q_value_metrics.to_parquet(self.get_file_path(output_dir + '/metrics', 'q_metrics.pq'), index=False)

            # Save action log dataframe
            df_action_logs = pd.DataFrame(action_data)
            df_action_logs.to_parquet(self.get_file_path(output_dir + '/metrics', 'action_logs.pq'), index=False)

        # Clear cache based on the device
        if self.device == torch.device("cuda"):
            torch.cuda.empty_cache()
            logging.info(f"Cleared cache for: {self.device}")
        elif self.device == torch.device("mps"):
            torch.mps.empty_cache()
            logging.info(f"Cleared cache for: {self.device}")

        # Save final model of trial
        model_path = self.get_file_path(output_dir + '/models', f'final_model.pth')
        self.agent.save(model_path)

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


class AgentOptimizerClassicNoisy:
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
        self.device = device
        self.n_episodes: int = self.hyper_params["n_episodes"]  # Type hint for n_episodes
        self.network_hyper_params = network_hyper_params

    def train(self, output_dir: str = './output') -> None:
        """
        Train the agent for a specified number of episodes.

        Args:
            output_dir (str): The directory to save the output files. Default is 'output'.
        """

        # initalize scaler for reward scaling to achieve consistent training situation (scaling from -1 to 1)
        mscaler = MaxAbsScaler()
        mscaler.fit(np.array([0, 1000]).reshape(-1, 1))

        # get expected picture shape for preprocessing correctly
        output_size = self.network_hyper_params['input_shape'][1]

        # init pre-episode scores and other metrics
        best_score = 0
        mvg_avg_score = 0
        mvg_avg_loss = 0
        scores: List[float] = []
        losses: List[float] = []
        metrics_data: List[Dict] = []
        action_data: List[Dict] = []

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
            count_actions = 0

            # Preprocess the initial state
            state = self.fp.preprocess(stacked_frames=None,
                                       env_state=self.env.reset()[0],
                                       exclude=(8, -12, -12, 4),
                                       output=output_size,
                                       is_new=True)

            while count_actions <= self.hyper_params['max_steps_episode']:
                # Select an action based on the current state
                action, action_type, q_values_sum = self.agent.act(state)

                # Collect action data
                action_data.append({'episode': episode,
                                    'action': action,
                                    'action_type': action_type,
                                    'q_values_sum': q_values_sum})

                # count predicted and randomized actions
                if action_type == 'predicted':
                    count_actions += 1

                # Take the action and observe the next state, reward, and termination flags
                next_state, reward, terminated, truncated, info = self.env.step(action)
                logging.debug(f"reward before scaling: {reward}")

                # Normalize the reward
                reward = mscaler.transform([[reward]])[0, 0]

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
            scores.append(score)
            losses.append(loss)

            # calc moving average score
            if len(scores) >= 20:
                mvg_avg_score = np.mean(scores[-20:])

            # calc moving average loss
            if len(losses) >= 20:
                mvg_avg_loss = np.mean(losses[-20:])

            # log metrics on console
            logging.info(f'Episode: {episode}, '
                         f'Average Score: {np.mean(scores):.5f}, '
                         f'Average Loss: {np.mean(losses):.5f}, '
                         f'Moving Average Score: {mvg_avg_score:.5f}, '
                         f'Moving Average Loss: {mvg_avg_loss:.5f}, '
                         f'Episode Score: {score:.5f}, '
                         f'Episode Loss: {loss:.5f}, '
                         f'Total No. Steps: {count_actions}, '
                         f'Min. Tau in Episode: {min(taus_in_episode):.5f}, '
                         f'Min. LR in Episode: {min(lrs_in_episode):.5f}, ')

            # Append current scores to collect metrics data
            metrics_data.append({'episode': episode,
                                 'avg_episode_score': np.mean(scores),
                                 'avg_episode_loss': np.mean(losses),
                                 'mvg_avg_score': mvg_avg_score,
                                 'mvg_avg_loss': mvg_avg_loss,
                                 'episode_score': score,
                                 'episode_loss': loss,
                                 'total_no_steps': count_actions,
                                 'model_saved': mvg_avg_score > best_score,
                                 'minimum_tau_in_episode': min(taus_in_episode),
                                 'minimum_lr_in_episode': min(lrs_in_episode)})

            # save model file if the relevant score has improved
            if mvg_avg_score > best_score and episode >= self.hyper_params['learn_start']:
                best_score = mvg_avg_score
                model_path = self.get_file_path(output_dir + '/models', f'best_model_score_{best_score:.3f}_episode_{episode}.pth')
                self.agent.save(model_path)  # Save the best model
                logging.info(f'New best model saved with score: {best_score:.2f}')

            # Save non q-metrics to Parquet file
            df_metrics = pd.DataFrame(metrics_data)
            df_metrics.to_parquet(self.get_file_path(output_dir + '/metrics', 'metrics.pq'), index=False)

            # Save full set of q-metrics to Parquet file
            self.agent.q_value_metrics.to_parquet(self.get_file_path(output_dir + '/metrics', 'q_metrics.pq'), index=False)

            # Save action log dataframe
            df_action_logs = pd.DataFrame(action_data)
            df_action_logs.to_parquet(self.get_file_path(output_dir + '/metrics', 'action_logs.pq'), index=False)

        # Clear cache based on the device
        if self.device == torch.device("cuda"):
            torch.cuda.empty_cache()
            logging.info(f"Cleared cache for: {self.device}")
        elif self.device == torch.device("mps"):
            torch.mps.empty_cache()
            logging.info(f"Cleared cache for: {self.device}")

        # Save final model of trial
        model_path = self.get_file_path(output_dir + '/models', f'final_model.pth')
        self.agent.save(model_path)

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


class AgentOptimizerOptuna:
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
        oselector = OptimizerSelector()
        optimizer = oselector(agent_hyper_params=agent_hyper_params, network=policy_net)

        # Init lr scheduler (optional)
        lrselector = LRSchedulerSelector()
        lr_scheduler = lrselector(agent_hyper_params=agent_hyper_params, optimizer=optimizer)

        # Init loss function
        lfselector = LossFunctionSelector()
        loss_function = lfselector(agent_hyper_params=agent_hyper_params)

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
                                 reward_factor=1.5,
                                 punish_factor=1.6,
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

        # initalize scaler for reward scaling to achieve consistent training situation (scaling from -1 to 1)
        mscaler = MaxAbsScaler()
        mscaler.fit(np.array([0, 1000]).reshape(-1, 1))

        # get expected picture shape for preprocessing correctly
        output_size = network_hyper_params['input_shape'][1]

        # init pre-episode scores and other metrics
        best_score = 0
        mvg_avg_score = 0
        mvg_avg_loss = 0
        scores: List[float] = []
        losses: List[float] = []
        metrics_data: List[Dict] = []
        action_data: List[Dict] = []

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

            # Preprocess the initial state
            state = self.fp.preprocess(stacked_frames=None,
                                       env_state=self.env.reset()[0],
                                       exclude=(8, -12, -12, 4),
                                       output=output_size,
                                       is_new=True)

            while count_predicted_actions + count_random_actions <= agent_hyper_params['max_steps_episode']:
                # Select an action based on the current state
                action, action_type, q_values_sum = agent.act(state, eps)

                # Collect action data
                action_data.append({'episode': episode,
                                    'action': action,
                                    'action_type': action_type,
                                    'q_values_sum': q_values_sum})

                if action_type == 'predicted':
                    count_predicted_actions += 1
                elif action_type == 'randomized':
                    count_random_actions += 1

                # Take the action and observe the next state, reward, and termination flags
                next_state, reward, terminated, truncated, info = self.env.step(action)
                logging.debug(f"reward before scaling: {reward}")

                # Normalize the reward
                reward = mscaler.transform([[reward]])[0, 0]
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
            scores.append(score)
            losses.append(loss)

            # calc moving average score
            if len(scores) >= 20:
                mvg_avg_score = np.mean(scores[-20:])

            # calc moving average loss
            if len(losses) >= 20:
                mvg_avg_loss = np.mean(losses[-20:])

            # log metrics on console
            logging.info(f'Optuna trial no.: {trial_number}, '
                         f'Episode: {episode}, '
                         f'Average Score: {np.mean(scores):.5f}, '
                         f'Average Loss: {np.mean(losses):.5f}, '
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
                                 'avg_episode_score': np.mean(scores),
                                 'avg_episode_loss': np.mean(losses),
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

            # save model file if the relevant score has improved
            if mvg_avg_score > best_score and episode >= agent_hyper_params['learn_start']:
                best_score = mvg_avg_score
                model_path = self.get_file_path(self.output_dir + '/models', f'{file_ref}.pth')
                agent.save(model_path)  # Save the best model
                logging.info(f'New best model saved with score: {best_score:.2f}')

            # Save non q-metrics to Parquet file
            df_metrics = pd.DataFrame(metrics_data)
            df_metrics.to_parquet(self.get_file_path(self.output_dir + '/metrics', f'{file_ref}_metrics.pq'), index=False)

            # Save full set of q-metrics to Parquet file
            agent.q_value_metrics.to_parquet(self.get_file_path(self.output_dir + '/metrics', f'{file_ref}_q_metrics.pq'), index=False)

            # Save action log dataframe
            df_action_logs = pd.DataFrame(action_data)
            df_action_logs.to_parquet(self.get_file_path(self.output_dir + '/metrics', f'{file_ref}_action_logs.pq'), index=False)

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

    def train(self, n_trials: int = 100, n_jobs: int = 1, warmup_steps: int = 500) -> None:
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


class AgentOptimizerOptunaNoisy:
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
        self.hyperparameter = hyperparameter
        self.device = device
        self.output_dir = output_dir
        self.file_ref = ''

    def init_agent(self,
                   policy_net: torch.nn.Module,
                   target_net: torch.nn.Module,
                   agent_hyper_params: Dict,
                   network_hyper_params: Dict,) -> object:

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
                reward_factor (float): The reward factor to use.
                punish_factor (float): The punishment factor to use.
            """

        # Init optimizer
        oselector = OptimizerSelector()
        optimizer = oselector(agent_hyper_params=agent_hyper_params, network=policy_net)

        # Init lr scheduler (optional)
        lrselector = LRSchedulerSelector()
        lr_scheduler = lrselector(agent_hyper_params=agent_hyper_params, optimizer=optimizer)

        # Init loss function
        lfselector = LossFunctionSelector()
        loss_function = lfselector(agent_hyper_params=agent_hyper_params)

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
                                 reward_factor=agent_hyper_params['reward_factor'],
                                 punish_factor=agent_hyper_params['punish_factor'],
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
        file_ref = f"trial_{trial_number}_{model_name}"

        # init agent
        agent = self.init_agent(policy_net, target_net, agent_hyper_params, network_hyper_params)

        # initalize scaler for reward scaling to achieve consistent training situation (scaling from -1 to 1)
        mscaler = MaxAbsScaler()
        mscaler.fit(np.array([0, 1000]).reshape(-1, 1))

        # get expected picture shape for preprocessing correctly
        output_size = network_hyper_params['input_shape'][1]

        # init dynamic frameprocessor for pre-processing
        fp = FrameProcessorDynamic(num_stacked_frames=network_hyper_params['input_shape'][0])

        # init pre-episode scores and other metrics
        best_score = 0
        mvg_avg_score = 0
        mvg_avg_loss = 0
        scores: List[float] = []
        losses: List[float] = []
        metrics_data: List[Dict] = []
        action_data: List[Dict] = []

        # Save agent hyperparameter configuration in filepath destination
        with open(self.output_dir + f'/metrics/{file_ref}_agent_hyper_params.yaml', 'w') as yaml_file:
            yaml.dump(agent_hyper_params, yaml_file, default_flow_style=False)

        # Save network hyperparameter configuration in filepath destination
        with open(self.output_dir + f'/metrics/{file_ref}_network_hyper_params.yaml', 'w') as yaml_file:
            yaml.dump(network_hyper_params, yaml_file, default_flow_style=False)

        for episode in range(1, agent_hyper_params['n_episodes'] + 1):
            # Init relevant variables for episode
            previous_action = 0
            score = 0
            loss = 0
            taus_in_episode = []
            lrs_in_episode = []
            count_actions = 0

            # Preprocess the initial state
            state = fp.preprocess(stacked_frames=None,
                                  env_state=self.env.reset()[0],
                                  exclude=(8, -12, -12, 4),
                                  output=output_size,
                                  is_new=True)

            while count_actions <= agent_hyper_params['max_steps_episode']:
                # Select an action based on the current state
                action, action_type, q_values_sum = agent.act(state)

                # Collect action data
                action_data.append({'episode': episode,
                                    'action': action,
                                    'action_type': action_type,
                                    'q_values_sum': q_values_sum})

                if action_type == 'predicted':
                    count_actions += 1

                # Take the action and observe the next state, reward, and termination flags
                next_state, reward, terminated, truncated, info = self.env.step(action)
                logging.debug(f"reward before scaling: {reward}")

                # Normalize the reward
                reward = mscaler.transform([[reward]])[0, 0]
                logging.debug(f"reward after scaling: {reward}")

                # Preprocess the next state
                next_state = fp.preprocess(stacked_frames=state,
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
            scores.append(score)
            losses.append(loss)

            # calc moving average score
            if len(scores) >= 20:
                mvg_avg_score = np.mean(scores[-20:])

            # calc moving average loss
            if len(losses) >= 20:
                mvg_avg_loss = np.mean(losses[-20:])

            # log metrics on console
            logging.info(f'Optuna trial no.: {trial_number}, '
                         f'Episode: {episode}, '
                         f'Average Score: {np.mean(scores):.5f}, '
                         f'Average Loss: {np.mean(losses):.5f}, '
                         f'Moving Average Score: {mvg_avg_score:.5f}, '
                         f'Moving Average Loss: {mvg_avg_loss:.5f}, '
                         f'Episode Score: {score:.5f}, '
                         f'Episode Loss: {loss:.5f}, '
                         f'Total No. Steps: {count_actions}, '
                         f'Min. Tau in Episode: {min(taus_in_episode):.5f}, '
                         f'Min. LR in Episode: {min(lrs_in_episode):.5f}, ')

            # Append current scores to collect metrics data
            metrics_data.append({'episode': episode,
                                 'avg_episode_score': np.mean(scores),
                                 'avg_episode_loss': np.mean(losses),
                                 'mvg_avg_score': mvg_avg_score,
                                 'mvg_avg_loss': mvg_avg_loss,
                                 'episode_score': score,
                                 'episode_loss': loss,
                                 'total_no_steps': count_actions,
                                 'model_saved': mvg_avg_score > best_score and episode > agent_hyper_params['learn_start'],
                                 'minimum_tau_in_episode': min(taus_in_episode),
                                 'minimum_lr_in_episode': min(lrs_in_episode)})

            # save model file if the relevant score has improved
            if mvg_avg_score > best_score and episode >= agent_hyper_params['learn_start']:
                best_score = mvg_avg_score
                model_path = self.get_file_path(self.output_dir + '/models', f'{file_ref}_score_{best_score:.5f}.pth')
                agent.save(model_path)  # Save the best model
                logging.info(f'New best model saved with score: {best_score:.2f}')

            # Save non q-metrics to Parquet file
            df_metrics = pd.DataFrame(metrics_data)
            df_metrics.to_parquet(self.get_file_path(self.output_dir + '/metrics', f'{file_ref}_metrics.pq'), index=False)

            # Save full set of q-metrics to Parquet file
            agent.q_value_metrics.to_parquet(self.get_file_path(self.output_dir + '/metrics', f'{file_ref}_q_metrics.pq'), index=False)

            # Save action log dataframe
            df_action_logs = pd.DataFrame(action_data)
            df_action_logs.to_parquet(self.get_file_path(self.output_dir + '/metrics', f'{file_ref}_action_logs.pq'), index=False)

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

        return best_score

    def train(self, n_trials: int = 100, n_jobs: int = 1, warmup_steps: int = 500) -> None:
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