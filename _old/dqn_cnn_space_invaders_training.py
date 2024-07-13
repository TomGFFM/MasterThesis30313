# import standards
import logging
import os
import pandas as pd

# import torch and gym
import torch
import gym

# import custom
from agents import DeepQNetworkAgent
from networks import DQNetworkCNN
from utils import FrameProcessor, AgentOptimizer

# #####################################################
# ################ init logging #######################
# #####################################################
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# #####################################################
# ################ set device  ########################
# #####################################################
# set cuda env variables
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Device for PyTorch (GPU or CPU)
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# log hyperparameter settings
logging.info(f"Device was set to: {device}")

# #####################################################
# ################ init hyperparameter ################
# #####################################################
hyper_params = {
    "batch_size": 64,                   # size of each batch pushed through the network
    "episodes": 10000,                  # total number of gaming sessions for the agent
    "epsilon_start": 0.99,              # start value for epsilon
    "epsilon_end": 0.01,                # lowest possible epsilon value
    "epsilon_decay": 0.001,             # factor by which epsilon gets reduced
    "gamma": 0.99,                      # tbd.
    "input_shape": (4, 84, 84),         # desired shape of state pictures
    "learn_start": 128,                 # tbd.
    "learning_rate": 0.0001,            # learning rate
    "max_steps": 1200,                  # maximum actions to be taken within an episode
    "replay_buffer_size": 100000,       # size of the replay buffer
    "tau": 0.001,                       # tbd.
    "update_every": 1,                  # after how many steps gets the network updated
    "update_target": 200,               # threshold to start the replay
    "n_episodes": 5000                  # number of episodes to play for the agent
}

# #####################################################
# ################ init gym environment ###############
# #####################################################
# initialize the gym environment
env = gym.make("ALE/SpaceInvaders-v5", render_mode="human")

# initialize frame processor for preprocess the game images and for stacking the frames
fp = FrameProcessor()

# init agent
agent = DeepQNetworkAgent(model=DQNetworkCNN, action_size=env.action_space.n, device=device, hyperparameter=hyper_params)

# #####################################################
# ################ train agent ########################
# #####################################################
ao = AgentOptimizer(agent=agent, env=env, hyperparameter=hyper_params, device=device)
ao.train(output_dir='../output')


# :TODO additional networks transformers and lstms, augmented versions
# :TODO test all networks
# :TODO seperate jupyter notebook for cloud execution
# :TODO quellen und beschreibungen für exponentielle zerfallsformel, warum macht diese sinn bei epsilon decay

"""
Eine gute und sinnvolle Formel für die Epsilon-Decay-Strategie sollte folgende Eigenschaften aufweisen:

Hoher Anfangswert: Epsilon sollte anfangs einen hohen Wert haben, um ausreichend Exploration zu ermöglichen und verschiedene Aktionen auszuprobieren.
Monotone Abnahme: Der Wert von Epsilon sollte im Laufe der Zeit monoton abnehmen, um schrittweise von der Exploration zur Exploitation überzugehen.
Asymptotisches Verhalten: Die Abnahme von Epsilon sollte asymptotisch erfolgen, sodass immer noch eine gewisse Exploration stattfindet, auch wenn der Wert sehr klein wird.
Anpassbarkeit: Die Formel sollte parametrisierbar sein, um sie an verschiedene Problemstellungen und Lernraten anpassen zu können.

"""

#
