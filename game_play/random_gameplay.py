# import standards
import logging
import os
import sys
import yaml

# torch and gym related
import torch
import gym


# #####################################################
# ################ init gym environment ###############
# #####################################################
# initialize the gym environment
env = gym.make("ALE/Carnival-v5", render_mode="human", frameskip=1)
# env = gym.make("ALE/Galaxian-v5", render_mode="human", frameskip=1)

obs, info = env.reset()
score = 0

while True:
    env.render()
    action = env.action_space.sample()
    # print(f'sampled action: {action}')
    next_state, reward, terminated, truncated, info = env.step(action)
    score += reward

    if terminated:
        print("You Final score is:", score)
        break
env.close()