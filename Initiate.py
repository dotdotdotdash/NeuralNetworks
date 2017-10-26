import matplotlib.pyplot as plt
import tensorflow as tf
import gym
import numpy as np
import math

import reinforcement_learning as rl

tf.__version__
gym.__version__

env_name = 'Breakout-v0'
#env_name = 'SpaceInvaders-v0'
rl.checkpoint_base_dir = 'checkpoints_tutorial16/'


rl.update_paths(env_name=env_name)


#rl.maybe_download_checkpoint(env_name=env_name)


agent = rl.Agent(env_name=env_name,
                 training=True,
                 render=False,
                 use_logging=False)

model = agent.model
replay_memory = agent.replay_memory
agent.run(num_episodes=1)
#log_q_values = rl.LogQValues()
#log_reward = rl.LogReward()
#
#log_q_values.read()
#log_reward.read()
#
#plt.plot(log_reward.count_states, log_reward.episode, label='Episode Reward')
#plt.plot(log_reward.count_states, log_reward.mean, label='Mean of 30 episodes')
#plt.xlabel('State-Count for Game Environment')
#plt.legend()
#plt.show()
#
#plt.plot(log_q_values.count_states, log_q_values.mean, label='Q-Value Mean')
#plt.xlabel('State-Count for Game Environment')
#plt.legend()
#plt.show()
agent.epsilon_greedy.epsilon_testing
agent.training = False
agent.reset_episode_rewards()
agent.render = True
agent.run(num_episodes=1)
