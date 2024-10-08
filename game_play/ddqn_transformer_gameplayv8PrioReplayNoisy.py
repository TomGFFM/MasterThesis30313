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
from agents import DeepQNetworkAgentPrioritized
from networks import DDQAugmentedNoisyTransformerNN
from utils import FrameProcessor, AgentOptimizerClassicNoisy

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
env = gym.make("ALE/SpaceInvaders-v5", render_mode="human", frameskip=1)

# #####################################################
# ################ init hyperparameter ################
# #####################################################
agent_hyper_params = {
    "batch_size": 256,                      # size of each batch pushed through the network
    "action_size": env.action_space.n,      # number of allowed actions in game
    "gamma": 0.99,                          # how much are future rewards valued
    "learning_rate": 0.005,                 # learning rate
    "learning_rate_step_size": 250,         # decrease learning rate by lr gamma after so many steps (works only if lr_scheduler object was passed to agent)
    "learning_rate_gamma": 0.25,            # factor by which the lr is reduced after lr steps (works only if lr_scheduler object was passed to agent)
    "max_steps_episode": 3000,              # maximum actions to be expected within an episode
    "replay_buffer_size": 300000,           # size of the replay buffer (max_steps_episode x n_episodes) / 20
    "tau": 0.01,                            # defines how fast the target network gets adjusted to the policy netw.
    "final_tau": 0.0001,                    # defines the lowest possible tau value
    "learn_start": 50,                      # number of episodes which have to be played before the training starts
    "update_every": 100,                    # number of steps after each the network gets updated once all other conditions were met
    "soft_update_target": 200,              # threshold of steps(actions) to start the soft update of the target network
    "n_episodes": 2000                      # number of episodes to play for the agent
}

network_hyper_params = {
    "input_shape": (4, 90, 90),             # desired shape of state pictures
    "num_actions": env.action_space.n,      # number of allowed actions in game
    "num_heads": 8,                         # number of attention heads in transformer layers
    "num_layers": 16,                       # number of transformer encoding layers
    "size_linear_layers": 1024,             # size of the fully connect linear layers in the transformer encoder setup
    "conv_channels": [64, 128, 192, 256],   # convolutional channels for CNN picture extraction
    "dropout_linear": 0.017,                # sigma value for the noisy network; higher sigma increases noise in network
    "sigma_init": 0.3,                      # dropout rate in linear layer
    "save_images": False,                   # save images from CNN layer (for testing only, keep false for normal training)
    "output_dir": output_dir                # output directory for saving images (directory has to contain subfolder images)
}


# #####################################################
# ##### init networks, optimizers and co. #############
# #####################################################
# uncomment if cnn extractions from gameplay should be extracted.
network_hyper_params['save_images'] = True

# inital network and optimizer setup
model_name = 'DDQAugmentedTransformerNNv8PrioReplayNoisy'

# Q-Network
policy_net = DDQAugmentedNoisyTransformerNN(**network_hyper_params).to(device)

# Set model name parameter in networks for logging purposes
if model_name:
    policy_net.model_name = model_name

# Init optimizer
optimizer = optim.RAdam(policy_net.parameters(), lr=agent_hyper_params['learning_rate'])

# Init lr scheduler (optional)
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.25, patience=100)
# lr_scheduler = None

# #####################################################
# ################ init agent #########################
# #####################################################

# initialize frame processor for preprocess the game images and for stacking the frames
fp = FrameProcessor()

# init agent
trained_agent = DeepQNetworkAgentPrioritized(policy_net=policy_net,
                                             target_net=None,
                                             action_size=env.action_space.n,
                                             device=device,
                                             agent_hyper_params=agent_hyper_params,
                                             network_hyper_params=network_hyper_params,
                                             optimizer=optimizer,
                                             lr_scheduler=lr_scheduler,
                                             reward_shaping=False,
                                             reward_factor=1.5,
                                             punish_factor=1.6,
                                             loss_function=F.huber_loss)

# load pre-trained model into agent
trained_agent.load('/Users/thomas/Repositories/MasterThesis30313/output/models/20240825_DDQAugmentedTransformerNNv8PrioReplayNoisy_best_model.pth', map_location=device)

output_size = network_hyper_params['input_shape'][1]

score = 0
state = fp.preprocess(stacked_frames=None,
                      env_state=env.reset()[0],
                      exclude=(8, -12, -12, 4),
                      output=output_size,
                      is_new=True)

while True:
    env.render()
    action, _, _ = trained_agent.act(state, eps=0.0, eval_mode=True)
    print(f'action: {action}')
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
