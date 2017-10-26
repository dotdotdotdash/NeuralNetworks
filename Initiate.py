import matplotlib.pyplot as plt
import tensorflow as tf
import gym
import numpy as np
import math

import reinforcement_learning as rl

tf.__version__
gym.__version__

# choose the game that you want to play
# Also if you want to play breakout-v0, use another trained model from here: 
# https://drive.google.com/a/asu.edu/file/d/0B1N26drdz1d7RVdNckpDQW83aWs/view?usp=sharing

env_name = 'Breakout-v0'
#env_name = 'SpaceInvaders-v0'
rl.checkpoint_base_dir = 'checkpoints_tutorial16/'

rl.update_paths(env_name=env_name)

# uncomment this code, if you want the check points to be downloaded
#rl.maybe_download_checkpoint(env_name=env_name)

agent = rl.Agent(env_name=env_name,
                 training=True,
                 render=False,
                 use_logging=False)

model = agent.model
replay_memory = agent.replay_memory
agent.run(num_episodes=1)

agent.epsilon_greedy.epsilon_testing
agent.training = False
agent.reset_episode_rewards()
agent.render = True
agent.run(num_episodes=1)
