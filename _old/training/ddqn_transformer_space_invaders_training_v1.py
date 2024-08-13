# import standards
import logging
import os
import sys

# add project folder to path dynamically
project_dir = os.path.dirname(os.getcwd())
sys.path.append(project_dir)

# import torch and gym
import torch
import gym

# import custom
from agents import DeepQNetworkAgent
from networks import DDQAugmentedTransformerNN
from utils import FrameProcessor, AgentOptimizer

# #####################################################
# ################ output directory ###################
# #####################################################
# target output directory
output_dir = '../../output'

# #####################################################
# ################ init logging #######################
# #####################################################
# logging.basicConfig(filename=output_dir + '/log_files/DDQAugmentedTransformerNN_training.log',level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
# ################ init gym environment ###############
# #####################################################
# target output directory
output_dir = '../../output'

# initialize the gym environment
env = gym.make("ALE/SpaceInvaders-v5", render_mode="rgb_array")

# #####################################################
# ################ init hyperparameter ################
# #####################################################
agent_hyper_params = {
    "batch_size": 128,                      # size of each batch pushed through the network
    "action_size": env.action_space.n,      # number of allowed actions in game
    "epsilon_start": 0.99,                  # start value for epsilon
    "epsilon_end": 0.01,                    # lowest possible epsilon value
    "epsilon_decay": 0.0005,                # factor by which epsilon gets reduced
    "gamma": 0.99,                          # how much are future rewards valued
    "learn_start": 128,                     # number of rounds before the training starts
    "learning_rate": 0.0001,                # learning rate
    "max_steps": 1500,                      # maximum actions to be taken within an episode
    "replay_buffer_size": 100000,           # size of the replay buffer
    "tau": 0.001,                           # defines how fast the target network gets adjusted to the policy netw.
    "update_every": 1,                      # after how many steps gets the network updated
    "update_target": 200,                   # threshold to start the replay
    "n_episodes": 15000                     # number of episodes to play for the agent
}

network_hyper_params = {
    "input_shape": (4, 84, 84),             # desired shape of state pictures
    "num_actions": env.action_space.n,      # number of allowed actions in game
    "num_heads": 8,                         # number of attention heads in transformer layers
    "num_layers": 6,                        # number of transformer encoding layers
    "conv_channels": [32, 64, 128, 256],    # convolutional channels for CNN picture extraction
    "save_images": False,                   # save images from CNN layer (for testing only, keep false for normal training)
    "output_dir": output_dir                # output directory for saving images (directory has to contain subfolder images)
}

# #####################################################
# ################ init agent #########################
# #####################################################

# initialize frame processor for preprocess the game images and for stacking the frames
fp = FrameProcessor()

# init agent
agent = DeepQNetworkAgent(model=DDQAugmentedTransformerNN,
                          action_size=env.action_space.n,
                          device=device,
                          agent_hyper_params=agent_hyper_params,
                          network_hyper_params=network_hyper_params,
                          model_name='DDQAugmentedTransformerNNv1')

# #####################################################
# ################ train agent ########################
# #####################################################
ao = AgentOptimizer(agent=agent, env=env, hyperparameter=agent_hyper_params, device=device)
ao.train(output_dir=output_dir)
