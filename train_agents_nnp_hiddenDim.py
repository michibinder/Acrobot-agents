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
from agents import Base_Agent, Q_Agent, Q_DQN_Agent, SARSA_Agent, SARSA_DQN_Agent
from agents_nnp import MC_PolGrad_Agent, AAC_Agent, AAC_TD_Agent

#%% ENVIRONMENT
env = gym.make('Acrobot-v1')

#%% SEED
random_seed = 1234
np.random.seed(random_seed)
torch.manual_seed(random_seed)
env.seed(random_seed)

#%% TRAINING
training_results = list() # A list for storing the hyperparameters and the corresponding results
MAX_EPISODES = 800
DROPOUT = 0.4 # 0.5
LR_POL = 0.001
# LR_QNET = 0.0001
GAMMA = 0.99
HIDDEN_DIM_POL = 128
HIDDEN_DIM_POL_2 = 32
LOG_INTERVAL = 100

#%% Train AAC-Agent (neural network policy - agent) ###
hyperparam_dict = {'name': 'AAC (' + str(16) + ')'}
aac_agent = AAC_Agent(env, num_episodes=MAX_EPISODES, num_steps=500, learning_rate=LR_POL,
                      gamma=GAMMA, hidden_dim=16, dropout=DROPOUT, log_interval=LOG_INTERVAL)
ep_rewards, running_rewards = aac_agent.train()
training_results.append((hyperparam_dict, ep_rewards, running_rewards))

#%% Train AAC-Agent (neural network policy - agent) ###
hyperparam_dict = {'name': 'AAC (' + str(32) + ')'}
aac_agent = AAC_Agent(env, num_episodes=MAX_EPISODES, num_steps=500, learning_rate=LR_POL,
                      gamma=GAMMA, hidden_dim=32, dropout=DROPOUT, log_interval=LOG_INTERVAL)
ep_rewards, running_rewards = aac_agent.train()
training_results.append((hyperparam_dict, ep_rewards, running_rewards))


#%% Train AAC-Agent (neural network policy - agent) ###
hyperparam_dict = {'name': 'AAC (' + str(64) + ')'}
aac_agent = AAC_Agent(env, num_episodes=MAX_EPISODES, num_steps=500, learning_rate=LR_POL,
                      gamma=GAMMA, hidden_dim=64, dropout=DROPOUT, log_interval=LOG_INTERVAL)
ep_rewards, running_rewards = aac_agent.train()
training_results.append((hyperparam_dict, ep_rewards, running_rewards))


#%% Train AAC-Agent (neural network policy - agent) ###
hyperparam_dict = {'name': 'AAC (' + str(HIDDEN_DIM_POL) + ')'}
aac_agent = AAC_Agent(env, num_episodes=MAX_EPISODES, num_steps=500, learning_rate=LR_POL,
                      gamma=GAMMA, hidden_dim=HIDDEN_DIM_POL, dropout=DROPOUT, log_interval=LOG_INTERVAL)
ep_rewards, running_rewards = aac_agent.train()
training_results.append((hyperparam_dict, ep_rewards, running_rewards))


#%% Train AAC-Agent (neural network policy - agent) ###
hyperparam_dict = {'name': 'AAC (' + str(256) + ')'}
aac_agent = AAC_Agent(env, num_episodes=MAX_EPISODES, num_steps=500, learning_rate=LR_POL,
                      gamma=GAMMA, hidden_dim=HIDDEN_DIM_POL, dropout=DROPOUT, log_interval=LOG_INTERVAL)
ep_rewards, running_rewards = aac_agent.train()
training_results.append((hyperparam_dict, ep_rewards, running_rewards))


#%% Train AAC-Agent (neural network policy - agent) ###
hyperparam_dict = {'name': 'AAC (' + str(512) + ')'}
aac_agent = AAC_Agent(env, num_episodes=MAX_EPISODES, num_steps=500, learning_rate=LR_POL,
                      gamma=GAMMA, hidden_dim=512, dropout=DROPOUT, log_interval=LOG_INTERVAL)
ep_rewards, running_rewards = aac_agent.train()
training_results.append((hyperparam_dict, ep_rewards, running_rewards))




#%% VISUALIZATION
plt.rcParams.update({'font.size': 8})
width = 185/25.4
# FIGSIZE = (width,width*1/3)
FIGSIZE = (width,width*1/3)

# Plot the results
fig = plt.figure(1, figsize=FIGSIZE)

for result in training_results:
    hp = result[0]
    ep_rewards = result[1]
    running_rewards = result[2]
    # plt.plot(range(len(ep_rewards)), ep_rewards, lw=2, color="red", label=hp['name'])
    plt.plot(range(len(running_rewards)), running_rewards, lw=1.2, label=hp['name'])
    
    # title_str = 'Acrobot-v1'
    # title_str = "Acrobot-v1 ($hiddenDim_{qnet}$: " + str(HIDDEN_DIM_QNET) + ", $hiddenDim_{pol}$: " + str(HIDDEN_DIM_POL) + ")"
    # plt.title(title_str)

plt.grid()
plt.xlabel('Episodes')
plt.ylabel('Running average of Rewards')
plt.legend(loc='lower right', ncol=1) # ncol=1
fig.tight_layout()
plt.show()
fig.savefig('images/AAC_hidden_Dim_comp.pdf')
#%% Save neural network of agent
# aac_agent.save_pol_network(save_dir="models", file_name="aac_polNet.pt")

