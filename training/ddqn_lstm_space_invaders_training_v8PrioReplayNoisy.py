# import standards
import logging
import os
import sys

# torch and gym related
import torch
import gym
import torch.nn.functional as F
import torch.optim as optim

# import custom
from agents import DeepQNetworkAgentPrioritizedNoisy
from networks import DDQAugmentedNoisyLSTMNN
from utils import AgentOptimizerClassicNoisy

# add project folder to path dynamically
project_dir = os.path.dirname(os.getcwd())
sys.path.append(project_dir)

# #####################################################
# ################ output directory ###################
# #####################################################
# target output directory
output_dir = '../output'

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
# ################ init gym environment ###############
# #####################################################
# target output directory
output_dir = '../output'

# initialize the gym environment
env = gym.make("ALE/SpaceInvaders-v5", render_mode="rgb_array", frameskip=1)

# #####################################################
# ################ init hyperparameter ################
# #####################################################
agent_hyper_params = {
    "batch_size": 32,                      # size of each batch pushed through the network
    "action_size": env.action_space.n,      # number of allowed actions in game
    "gamma": 0.99,                          # how much are future rewards valued
    "learning_rate": 2.05837e-05,           # learning rate
    "learning_rate_step_size": 172,         # decrease learning rate by lr gamma after so many steps (works only if lr_scheduler object was passed to agent)
    "learning_rate_gamma": 0.2797,          # factor by which the lr is reduced after lr steps (works only if lr_scheduler object was passed to agent)
    "max_steps_episode": 3000,              # maximum actions to be expected within an episode
    "replay_buffer_size": 100000,           # size of the replay buffer (max_steps_episode x n_episodes) / 20
    "tau": 0.00146,                            # defines how fast the target network gets adjusted to the policy netw.
    "final_tau": 0.0001,                    # defines the lowest possible tau value
    "learn_start": 20,                      # number of episodes which have to be played before the training starts
    "update_every": 197,                    # number of steps after each the network gets updated once all other conditions were met
    "soft_update_target": 263,              # threshold of steps(actions) to start the soft update of the target network
    "n_episodes": 2000                      # number of episodes to play for the agent
}

network_hyper_params = {
    "input_shape": (4, 90, 90),             # desired shape of state pictures
    "num_actions": env.action_space.n,      # number of allowed actions in game
    "hidden_size": 64,                      # size of hidden/cell state
    "num_layers": 2,                        # number of lstm layers
    "size_linear_layers": 64,               # size of the fully connect linear layers in the transformer encoder setup
    # "conv_channels": [64, 128, 192, 256],   # convolutional channels for CNN picture extraction
    "conv_channels": [8, 0, 0, 32],
    "dropout_linear": 0.0086,               # dropout rate in linear layer
    "sigma_init": 0.0006038,                # sigma value for the noisy network; higher sigma increases noise in network
    "lean_cnn": True,                      # Inits a lean version of the CNN layer which only has the first and the last conv layer but less abstraction (so careful usage)
    "save_images": False,                   # save images from CNN layer (for testing only, keep false for normal training)
    "output_dir": output_dir                # output directory for saving images (directory has to contain subfolder images)
}

# {'batch_size': 32, 'learning_rate': 2.058375402654718e-05, 'learning_rate_step_size': 172, 'learning_rate_gamma': 0.27971591073013, 'replay_buffer_size': 100000, 'tau': 0.001460655225593748, 'update_every': 197,
# 'soft_update_target': 263, 'optimizer': 'SGD', 'lr_scheduler': 'cosine', 'loss_function': 'huber', 'hidden_size': 64, 'num_layers': 2, 'size_linear_layers': 64, 'dropout_linear': 0.008611372068533154, 'sigma_init': 0.0006038341278307473}

# #####################################################
# ##### init networks, optimizers and co. #############
# #####################################################
# inital network and optimizer setup
model_name = 'DDQAugmentedLSTMNNv8PrioReplayNoisy'

# Q-Network
policy_net = DDQAugmentedNoisyLSTMNN(**network_hyper_params).to(device)
target_net = DDQAugmentedNoisyLSTMNN(**network_hyper_params).to(device)

# Set model name parameter in networks for logging purposes
if model_name:
    policy_net.model_name = model_name
    target_net.model_name = model_name

# Init optimizer
optimizer = optim.SGD(policy_net.parameters(), lr=agent_hyper_params['learning_rate'])

# Init lr scheduler (optional)
lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=agent_hyper_params['n_episodes'], eta_min=0.000001)
# lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.25, patience=100)
# lr_scheduler = None

# #####################################################
# ################ init agent #########################
# #####################################################
# init agent
agent = DeepQNetworkAgentPrioritizedNoisy(policy_net=policy_net,
                                          target_net=target_net,
                                          action_size=env.action_space.n,
                                          device=device,
                                          agent_hyper_params=agent_hyper_params,
                                          network_hyper_params=network_hyper_params,
                                          optimizer=optimizer,
                                          lr_scheduler=lr_scheduler,
                                          reward_shaping=True,
                                          reward_factor=1.4,
                                          punish_factor=1.2,
                                          loss_function=F.smooth_l1_loss)

# #####################################################
# ################ train agent ########################
# #####################################################
ao = AgentOptimizerClassicNoisy(agent=agent,
                                env=env,
                                hyperparameter=agent_hyper_params,
                                network_hyper_params=network_hyper_params,
                                device=device)

ao.train(output_dir=output_dir)
