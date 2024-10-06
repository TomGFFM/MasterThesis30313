# import standards
import logging
import os
import sys
import yaml

# torch and gym related
import torch
import gym

# import custom
from agents import DeepQNetworkAgentPrioritized, DeepQNetworkAgentPrioritizedNoisy
from networks import DDQAugmentedNoisyTransformerNN
from utils import FrameProcessor, LRSchedulerSelector, OptimizerSelector, LossFunctionSelector, FrameProcessorDynamic

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
# initialize the gym environment
env = gym.make("ALE/SpaceInvaders-v5", render_mode="human", frameskip=1)

# #####################################################
# ################ init hyperparameter ################
# #####################################################
agent_params_file = open('/Users/thomas/Repositories/MasterThesis30313/output_remote_20240917/metrics/trial_19_DDQAugmentedNoisyTransformerNN_agent_hyper_params.yaml', 'r')
network_params_file = open('/Users/thomas/Repositories/MasterThesis30313/output_remote_20240917/metrics/trial_19_DDQAugmentedNoisyTransformerNN_network_hyper_params.yaml', 'r')
agent_hyper_params = yaml.load(agent_params_file, Loader=yaml.FullLoader)
network_hyper_params = yaml.load(network_params_file, Loader=yaml.FullLoader)

# #####################################################
# ##### init networks, optimizers and co. #############
# #####################################################
# uncomment if cnn extractions from gameplay should be extracted.
network_hyper_params['save_images'] = True

# inital network and optimizer setup
model_name = 'DDQAugmentedTransformerNNv9OptunaPrioReplayNoisy'

# Q-Network
policy_net = DDQAugmentedNoisyTransformerNN(**network_hyper_params).to(device)

# Set model name parameter in networks for logging purposes
if model_name:
    policy_net.model_name = model_name

# Init optimizer
oselector = OptimizerSelector()
optimizer = oselector(agent_hyper_params=agent_hyper_params, network=policy_net)

# Init lr scheduler (optional)
lrselector = LRSchedulerSelector()
lr_scheduler = lrselector(agent_hyper_params=agent_hyper_params, optimizer=optimizer)

# Init loss function
lfselector = LossFunctionSelector()
loss_function = lfselector(agent_hyper_params=agent_hyper_params)

# #####################################################
# ################ init agent #########################
# #####################################################

# initialize frame processor for preprocess the game images and for stacking the frames
fp = FrameProcessorDynamic(num_stacked_frames=network_hyper_params['input_shape'][0])

# init agent
trained_agent = DeepQNetworkAgentPrioritizedNoisy(policy_net=policy_net,
                                                  target_net=None,
                                                  action_size=env.action_space.n,
                                                  device=device,
                                                  agent_hyper_params=agent_hyper_params,
                                                  network_hyper_params=network_hyper_params,
                                                  optimizer=optimizer,
                                                  lr_scheduler=lr_scheduler,
                                                  reward_factor=agent_hyper_params['reward_factor'],
                                                  punish_factor=agent_hyper_params['punish_factor'],
                                                  loss_function=loss_function)

# load pre-trained model into agent
trained_agent.load('/Users/thomas/Repositories/MasterThesis30313/output_remote_20240917/models/20240914_trial_19_DDQAugmentedNoisyTransformerNN.pth', map_location=device)

output_size = network_hyper_params['input_shape'][1]

score = 0
state = fp.preprocess(stacked_frames=None,
                      env_state=env.reset()[0],
                      exclude=(8, -12, -12, 4),
                      output=output_size,
                      is_new=True)

while True:
    env.render()
    action, _, _ = trained_agent.act(state, eval_mode=True)
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
