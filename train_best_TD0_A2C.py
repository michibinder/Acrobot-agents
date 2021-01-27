#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 16:03:09 2021

@author: tennismichel
"""
# import os
import numpy as np
import matplotlib.pyplot as plt
import gym
import torch

# from networks import NNetwork, NeuralNetworkPolicy
from agents import Q_Agent, Q_DQN_Agent, SARSA_Agent, SARSA_DQN_Agent
from agents_nnp import A2C_Agent, MC_PolGrad_Agent, TD_A2C_Agent

#%% ENVIRONMENT
env = gym.make('Acrobot-v1')
# env = gym.make('MountainCar-v0')

#%% SEED
random_seed = 1234
np.random.seed(random_seed)
torch.manual_seed(random_seed)
env.seed(random_seed)

#%% TRAINING
training_results = list() # A list for storing the hyperparameters and the corresponding results
MAX_EPISODES = 4000
MAX_STEPS = 500
LOG_INTERVAL = 100

LR_POL = 0.001
GAMMA = 0.99
DROPOUT = 0
HIDDEN_DIM = 32


#%% Train Q_DQN-Agent (semi-gradient) ###
agent_results = list()
hyperparam_dict = {'name': 'A2C (DROPOUT:0.4, NN$_{dim}$:512)'}
agent = TD_A2C_Agent(env, num_episodes=MAX_EPISODES, num_steps=MAX_STEPS, learning_rate=LR_POL,
                      gamma=GAMMA, hidden_dim=HIDDEN_DIM, dropout=DROPOUT, log_interval=LOG_INTERVAL)
ep_rewards, running_rewards = agent.train()

# agent_results.append(running_rewards)
# agent_results = np.array(agent_results)
# rewards_mu = agent_results.mean(axis=0)
# rewards_sigma = agent_results.std(axis=0)

training_results.append((hyperparam_dict, ep_rewards, running_rewards))


#%% VISUALIZATION
plt.rcParams.update({'font.size': 9})
width = 170/25.4 # 6.7in
# FIGSIZE = (width,width*1/3)
FIGSIZE = (width,width*3/8)

# Plot the results
fig, ax = plt.subplots(figsize=FIGSIZE)
i=1
for result in training_results:
    # i += 1
    # if not i%2==0:
    #     continue
    hp = result[0]
    ep_reward = result[1]
    running_reward = result[2]
     
    plt.plot(range(len(ep_reward)), ep_reward, lw=1.2, label='TD(0) A2C (DROPOUT: 0, NN$_{dim}$: 32)')
    plt.plot(range(len(running_reward)), running_reward, lw=1.2, label='EMA (stopped @ -75)')
    # ax.fill_between(range(len(mu)), mu+sigma, mu-sigma, alpha=0.5)
    
    # plt.plot(range(len(ep_rewards)), ep_rewards, lw=2, color="red", label=hp['name'])
    # title_str = "Acrobot-v1 ($hiddenDim_{qnet}$: " + str(HIDDEN_DIM_QNET) + ", $hiddenDim_{pol}$: " + str(HIDDEN_DIM_POL) + ")"
    # plt.title(title_str)

ax.set_xlim(0,2500)
plt.grid()
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.legend(loc='lower right', ncol=1) # ncol=1
fig.tight_layout()
plt.show()
fig.savefig('images/TD0_A2C_agent.pdf')

#%% Save neural network of agent
agent.save_pol_network(save_dir="models", file_name="TD0_A2C_v2.pt")