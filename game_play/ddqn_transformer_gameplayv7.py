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
from agents import DeepQNetworkAgentv4
from networks import DDQAugmentedTransformerNN
from utils import FrameProcessor, AgentOptimizerv4

# #####################################################
# ################ output directory ###################
# #####################################################
# target output directory
output_dir = '../output'

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
output_dir = '../output'

# initialize the gym environment
env = gym.make("ALE/SpaceInvaders-v5", render_mode="human")

# #####################################################
# ################ init hyperparameter ################
# #####################################################
agent_hyper_params = {
    "batch_size": 256,                      # size of each batch pushed through the network
    "action_size": env.action_space.n,      # number of allowed actions in game
    "epsilon_start": 0.99,                  # start value for epsilon
    "epsilon_end": 0.4,                     # lowest possible epsilon value
    "epsilon_decay": 0.01,                  # factor by which epsilon gets reduced
    "gamma": 0.99,                          # how much are future rewards valued
    "learn_start": 250,                     # number of rounds before the training starts
    "learning_rate": 0.001,                 # learning rate
    "learning_rate_step_size": 250,         # decrease learning rate by lr gamma after so many steps (works only if lr_scheduler object was passed to agent)
    "learning_rate_gamma": 0.5,             # factor by which the lr is reduced after lr steps (works only if lr_scheduler object was passed to agent)
    "max_steps_episode": 3000,              # maximum actions to be expected within an episode
    "replay_buffer_size": 100000,           # size of the replay buffer
    "tau": 0.01,                            # defines how fast the target network gets adjusted to the policy netw.
    "final_tau": 0.0001,                    # defines the lowest possible tau value
    "update_every": 100,                    # after how many steps gets the network updated
    "update_target": 5000,                  # threshold of steps to start the replay
    "n_episodes": 3000                      # number of episodes to play for the agent
}

network_hyper_params = {
    "input_shape": (4, 90, 90),             # desired shape of state pictures
    "num_actions": env.action_space.n,      # number of allowed actions in game
    "num_heads": 64,                        # number of attention heads in transformer layers
    "num_layers": 32,                       # number of transformer encoding layers
    "size_linear_layers": 4096,             # size of the fully connect linear layers in the transformer encoder setup
    "conv_channels": [64, 128, 192, 256],   # convolutional channels for CNN picture extraction
    "save_images": False,                   # save images from CNN layer (for testing only, keep false for normal training)
    "output_dir": output_dir                # output directory for saving images (directory has to contain subfolder images)
}


# #####################################################
# ################ init agent #########################
# #####################################################

# initialize frame processor for preprocess the game images and for stacking the frames
fp = FrameProcessor()

# init agent
trained_agent = DeepQNetworkAgentv4(model=DDQAugmentedTransformerNN,
                                    action_size=env.action_space.n,
                                    device=device,
                                    agent_hyper_params=agent_hyper_params,
                                    network_hyper_params=network_hyper_params,
                                    optimizer=optim.RAdam,
                                    lr_scheduler=None,
                                    reward_shaping=True,
                                    reward_factor=1.4,
                                    punish_factor=1.8,
                                    loss_function=F.smooth_l1_loss,
                                    model_name='DDQAugmentedTransformerNNv7')

# load pre-trained model into agent
trained_agent.load('/Users/thomas/Repositories/MasterThesis30313/output/models/20240813_DDQAugmentedTransformerNNv7_best_model_episode_590_score_17.74667.pth', map_location=device)

output_size = network_hyper_params['input_shape'][1]

score = 0
state = fp.preprocess(stacked_frames=None,
                      env_state=env.reset()[0],
                      exclude=(8, -12, -12, 4),
                      output=output_size,
                      is_new=True)

while True:
    env.render()
    action, _ = trained_agent.act(state)
    next_state, reward, terminated, truncated, info = env.step(action)
    score += reward
    state = fp.preprocess(stacked_frames=state,
                               env_state=next_state,
                               exclude=(8, -12, -12, 4),
                               output=output_size,
                               is_new=False)

    if terminated:
        print("You Final score is:", score)
        break
env.close()
