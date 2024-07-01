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
    "n_episodes": 10000                 # number of episodes to play for the agent
}

# #####################################################
# ############### run trained agent ###################
# #####################################################
# initialize the gym environment
env = gym.make("ALE/SpaceInvaders-v5", render_mode="human")

# initialize frame processor for preprocess the game images and for stacking the frames
fp = FrameProcessor()

# init agent object
trained_agent = DeepQNetworkAgent(model=DQNetworkCNN,
                                  action_size=env.action_space.n,
                                  device=device,
                                  hyperparameter=hyper_params)

# load pre-trained model into agent
trained_agent.load('./output/20240523_QNetworkCNN_best_model.pth')

score = 0
state = fp.preprocess(stacked_frames=None,
                      env_state=env.reset()[0],
                      exclude=(8, -12, -12, 4),
                      output=84,
                      is_new=True)

while True:
    env.render()
    action, _ = trained_agent.act(state)
    next_state, reward, terminated, truncated, info = env.step(action)
    score += reward
    state = fp.preprocess(stacked_frames=state,
                               env_state=next_state,
                               exclude=(8, -12, -12, 4),
                               output=84,
                               is_new=False)

    if terminated:
        print("You Final score is:", score)
        break
env.close()



