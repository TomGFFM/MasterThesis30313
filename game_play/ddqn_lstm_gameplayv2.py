# import standards
import logging
import os
import gym

# import torch and gym
import torch

# import custom
from agents import DeepQNetworkAgent
from networks import DDQAugmentedLSTMNN
from utils import FrameProcessor

# #####################################################
# ################ output directory ###################
# #####################################################
# target output directory
output_dir = '../output'

# #####################################################
# ################ init logging #######################
# #####################################################
# logging.basicConfig(filename=output_dir + '/log_files/DDQAugmentedLSTMNN_training.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# #####################################################
# ################ set device  ########################
# #####################################################
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
# initialize the gym environment
env = gym.make("ALE/SpaceInvaders-v5", render_mode="human")

# #####################################################
# ################ init hyperparameter ################
# #####################################################
agent_hyper_params = {
    "batch_size": 64,                       # size of each batch pushed through the network
    "action_size": env.action_space.n,      # number of allowed actions in game
    "epsilon_start": 0.99,                  # start value for epsilon
    "epsilon_end": 0.01,                    # lowest possible epsilon value
    "epsilon_decay": 0.0005,                # factor by which epsilon gets reduced
    "gamma": 0.99,                          # how much are future rewards valued
    "learn_start": 128,                     # number of rounds before the training starts
    "learning_rate": 0.0001,                # learning rate
    "max_steps": 2500,                      # maximum actions to be taken within an episode
    "replay_buffer_size": 100000,           # size of the replay buffer
    "tau": 0.001,                           # defines how fast the target network gets adjusted to the policy netw.
    "update_every": 1,                      # after how many steps gets the network updated
    "update_target": 256,                   # threshold to start the replay
    "n_episodes": 20000                     # number of episodes to play for the agent
}

network_hyper_params = {
    "input_shape": (4, 84, 84),             # desired shape of state pictures
    "num_actions": env.action_space.n,      # number of allowed actions in game
    "hidden_size": 1024,                    # size of hidden/cell state
    "num_layers": 12,                       # number of transformer encoding layers
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
trained_agent = DeepQNetworkAgent(model=DDQAugmentedLSTMNN,
                                  action_size=env.action_space.n,
                                  device=device,
                                  agent_hyper_params=agent_hyper_params,
                                  network_hyper_params=network_hyper_params)

# load pre-trained model into agent
trained_agent.load('/Users/thomas/Repositories/MasterThesis30313/output/models/20240720_DDQAugmentedLSTMNNv2_best_model_score.pth', map_location=device)

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