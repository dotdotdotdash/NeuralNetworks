import gym
import numpy as np
import random
import math
from time import sleep
import json

## Initialize the "Cart-Pole" environment
env = gym.make('CartPole-v0')
env._max_episode_steps=60001
## Defining the environment related constants

# Number of discrete states (bucket) per state dimension
NUM_BUCKETS = (1, 1, 8, 3)  # (x, x', theta, theta')
# Number of discrete actions
NUM_ACTIONS = env.action_space.n # (left, right)
# Bounds for each discrete state
STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))
STATE_BOUNDS[0] = [-2.4, 2.4]
STATE_BOUNDS[1] = [-0.8, 0.8]
STATE_BOUNDS[2] = [-math.radians(12), math.radians(12)]
STATE_BOUNDS[3] = [-math.radians(30), math.radians(30)]
# Index of the action
ACTION_INDEX = len(NUM_BUCKETS)

## Creating a Q-Table for each state-action pair
q_table = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,))
## Learning related constants
# MIN_EXPLORE_RATE = 0.08
MIN_EXPLORE_RATE = 0.001
MIN_LEARNING_RATE = 0.01

## Defining the simulation related constants
NUM_EPISODES = 10000
MAX_T = 60001
MAX_RUNS = 1
STREAK_TO_END = 120
SOLVED_T = 60000
DEBUG_MODE = False
episode_duration=[1]
randomnum=0

def simulate(i_n):
	episode_solved = []
	global q_table,episode_duration
	for run in range(MAX_RUNS):
       		## Instantiating the learning related parameters
		print("Run - {} with noise category {}".format(run+1,i_n+1))
		learning_rate = get_learning_rate(0)
		explore_rate = get_explore_rate(0)
		discount_factor = 1  # since the world is unchanging
		prev_max = 0
		run_over = False
		max_time = 0
		episode = 0
		q_table = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,))
		#print(q_table)
		for episode in range(NUM_EPISODES):
		        # Reset the environment
			obv = env.reset()
		                # the initial state
			state_0 = state_to_bucket(obv)
		
			for t in range(MAX_T):
				noise = design_noise(i_n)
		                        # Select an action
				action = select_action(state_0, explore_rate)
				
				if noise == 1:
					actuator = action + 2*noise
					noise = 0
				elif noise == 2:
					actuator = action + 3*noise
					noise = 0
				else:
					actuator = action

				obv, reward, done, _ = env.step(actuator)
				
				actuator = 0
		                        # Observe the result
				obv = obv+obv*noise

				state = state_to_bucket(obv)

		                        # Calculate Reward
				thetad=np.degrees(obv[2])
				if (thetad>-5.0 and thetad<5.0):
					reward=reward+1.0
					# Update the Q based on the results
				best_q = np.amax(q_table[state])
				q_table[state_0 + (action,)] += learning_rate*(reward + discount_factor*(best_q) - q_table[state_0 + (action,)])

		                        # Setting up for the next iteration
				state_0 = state
				if done:
					episode_duration.append(t)
					max_time = np.amax(episode_duration)
					if t == SOLVED_T:
						episode_solved.append(episode)
						run_over = True
						episode_duration = [1]
						#q_table = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,))
					if max_time > prev_max:
						print("Achieved {} time steps in {} episodes".format(max_time,episode))
						prev_max = max_time
					break
					
			if run_over:
				break

		                # Update parameters
			explore_rate = get_explore_rate(episode)
			learning_rate = get_learning_rate(episode)
		
	average = round(sum(episode_solved)/len(episode_solved))
	success_rate = (len(episode_solved)/MAX_RUNS)*100
	episode_solved = []
	return average,success_rate

def design_noise(i):
	if i == 0:
		noise = 0
	elif i == 1:
		noise = 1
	elif i == 2:
		noise = 2
	elif i == 3:
		noise = random.uniform(0,0.05)
	elif i == 4:
		noise = random.uniform(0,0.1)
	elif i == 5:
		noise = random.gauss(0,0.1)
	else:
		noise = random.gauss(0,0.2)
	return noise	                

def select_action(state, explore_rate):
        # Select a random action
        if random.random() < explore_rate:
                action = env.action_space.sample()
        # Select the action with the highest q
        else:
                action = np.argmax(q_table[state])
        return action


def get_explore_rate(t):
        # return max(MIN_EXPLORE_RATE, min(1, 1.0 - (math.log(t+1)/math.log(2000))-.05))
        return max(MIN_EXPLORE_RATE, min(1, 1.0 - (math.log(t+.2)/math.log(300))))

def get_learning_rate(t):
        return max(MIN_LEARNING_RATE, min(1, 1- (math.log(t+.2)/math.log(800))))
        #return max(MIN_LEARNING_RATE, min(1.0, 1.0 - math.log((t/10)+1)))


def state_to_bucket(state):
        bucket_indice = []
        for i in range(len(state)):
                if state[i] <= STATE_BOUNDS[i][0]:
                        bucket_index = 0
                elif state[i] >= STATE_BOUNDS[i][1]:
                        bucket_index = NUM_BUCKETS[i] - 1
                else:
                        # Mapping the state bounds to the bucket array
                        bound_width = STATE_BOUNDS[i][1] - STATE_BOUNDS[i][0]
                        offset = (NUM_BUCKETS[i]-1)*STATE_BOUNDS[i][0]/bound_width
                        scaling = (NUM_BUCKETS[i]-1)/bound_width
                        bucket_index = int(round(scaling*state[i] - offset))
                bucket_indice.append(bucket_index)
        return tuple(bucket_indice)


if __name__ == "__main__":
	avg = []
	sr = []
	for i in range(7):
		average, success_rate = simulate(i)
		avg.append(average)
		sr.append(success_rate)
	print(avg)
	print(sr)
	#print("On average over {} runs, 60000 time steps were reached in {} episodes".format(MAX_RUNS,average))
        #print("Success rate over {} is {}".format(MAX_RUNS,success_rate))
