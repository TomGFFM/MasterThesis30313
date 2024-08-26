# import standards
import logging
import os
import sys

# torch and gym related
import torch
import gym

# add project folder to path dynamically
project_dir = os.path.dirname(os.getcwd())
sys.path.append(project_dir)

# import custom
from agents import DeepQNetworkAgentPrioritizedNoisy
from networks import DDQAugmentedNoisyTransformerNN
from utils import AgentOptimizerOptunaNoisy

# #####################################################
# ################ output directory ###################
# #####################################################
# target output directory
output_dir = '../output'

# #####################################################
# ################ init logging #######################
# #####################################################
# logging.basicConfig(filename=output_dir + '/log_files/DDQAugmentedTransformerNN_training.log',level=logging.INFO,
# format='%(asctime)s - %(levelname)s - %(message)s')
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
env = gym.make("ALE/SpaceInvaders-v5", render_mode="rgb_array")


# #####################################################
# ############# init hyperparameter class #############
# #####################################################
class Hyperparameters(object):
    """Hyperparameters setup class build for optuna usage"""
    def __init__(self):
        pass

    @staticmethod
    def get_params(trial):
        agent_hyper_params = {
            "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),                         # size of each batch pushed through the network
            "action_size": env.action_space.n,                                                              # number of allowed actions in game
            "gamma": 0.99,                                                                                  # how much are future rewards valued
            "learning_rate": trial.suggest_float("learning_rate", 0.00001, 0.001, log=True),                # learning rate
            "learning_rate_step_size": trial.suggest_int("learning_rate_step_size", 100, 300),              # decrease learning rate by lr gamma after so many steps (works only if lr_scheduler object was passed to agent)
            "learning_rate_gamma": trial.suggest_float("learning_rate_gamma", 0.2, 0.4),                    # factor by which the lr is reduced after lr steps (works only if lr_scheduler object was passed to agent)
            "max_steps_episode": 2000,                                                                      # maximum actions to be expected within an episode
            "replay_buffer_size": trial.suggest_categorical("batch_size", [1000, 5000, 10000,
                                                                           50000, 100000]),                 # size of the replay buffer
            "tau": trial.suggest_float("tau", 1e-4, 1e-2, log=True),                                        # defines how fast the target network gets adjusted to the policy netw.
            "final_tau": 0.0001,                                                                            # defines the lowest possible tau value
            "learn_start": 15,                                                                              # number of episodes which have to be played before the training starts (10% of n_episodes)
            "update_every": trial.suggest_int("update_every", 50, 200),                                     # number of steps after each the network gets updated once all other conditions were met
            "soft_update_target": trial.suggest_int("soft_update_target", 100, 500),                        # threshold of steps(actions) to start the soft update of the target network
            "n_episodes": 2000,                                                                             # number of episodes to play for the agent
            "optimizer_name": trial.suggest_categorical("optimizer", ["Adam", "NAdam", "SGD", "RMSprop",
                                                                      "Adagrad", "Adadelta", "RAdam"]),     # name of the optimizer to be used for loss optimization
            "lr_scheduler_name": trial.suggest_categorical("lr_scheduler", ["cosine", "step",
                                                                            "reduce_on_plateau", "none"]),  # name of learning rate scheduler to be used
            "loss_name": trial.suggest_categorical("loss_function", ["huber", "mse", "l1"]),                # name of loss function to be used
            "reward_factor": trial.suggest_float("reward_factor", 1.0, 1.9),                                # factor which improves the reward in reward shaping
            "punish_factor": trial.suggest_float("punish_factor", 1.0, 1.9),                                # factor which decreases the reward in reward shaping
        }

        network_hyper_params = {
            "input_shape": (4, 90, 90),                                                                  # desired shape of state pictures
            "num_actions": env.action_space.n,                                                           # number of allowed actions in game
            "num_heads": trial.suggest_categorical("num_heads", [2, 4, 8, 16]),                          # number of attention heads in transformer layers
            "num_layers": trial.suggest_int("num_layers", 2, 32, step=2),                                # number of transformer encoding layers
            "size_linear_layers": trial.suggest_categorical("size_linear_layers", [256, 512, 1024]),     # size of the fully connect linear layers in the transformer encoder setup
            "dropout_linear": trial.suggest_float("dropout_linear", 0.05, 0.5),                          # dropout rate in linear layer
            "sigma_init": trial.suggest_float('sigma_init', 0.01, 0.1, log=True),                        # sigma value for the noisy network; higher sigma increases noise in network
            "conv_channels": [64, 128, 192, 256],                                                        # convolutional channels for CNN picture extraction
            "save_images": False,                                                                        # save images from CNN layer (for testing only, keep false for normal training)
            "output_dir": output_dir                                                                     # output directory for saving images (directory has to contain subfolder images)
        }

        return agent_hyper_params, network_hyper_params


# #####################################################
# ################ train agent ########################
# #####################################################
ao = AgentOptimizerOptunaNoisy(agent=DeepQNetworkAgentPrioritizedNoisy,
                               network=DDQAugmentedNoisyTransformerNN,
                               env=env,
                               hyperparameter=Hyperparameters,
                               device=device,
                               output_dir=output_dir)

ao.train(n_trials=100, n_jobs=1, warmup_steps=50)
