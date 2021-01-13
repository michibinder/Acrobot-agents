#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 16:35:25 2021

@author: tennismichel
"""
import numpy as np
import matplotlib.pyplot as plt
import gym
import torch

# from networks import NNetwork, NeuralNetworkPolicy
from agents import Q_Agent, Q_DQN_Agent, SARSA_Agent, SARSA_DQN_Agent
from agents_nnp import AAC_Agent, MC_PolGrad_Agent
from networks import QNetwork, PolicyNetwork


#%% TRAINING
training_results = list() # A list for storing the hyperparameters and the corresponding results
MAX_EPISODES = 300
EPS = 0.2
DROPOUT = 0.4 # 0.5
LR_POL = 0.001
LR_QNET = 0.0001
GAMMA = 0.99
HIDDEN_DIM_POL = 128
HIDDEN_DIM_QNET = 16
LOG_INTERVAL = 100


# #%% EVALUATION & VIDEO (AAC)
# env_to_wrap = gym.make('Acrobot-v1')
# env = gym.wrappers.Monitor(env_to_wrap, 'videos/AAC', force = True)

# aac_agent = AAC_Agent(env, num_episodes=MAX_EPISODES, num_steps=500, learning_rate=LR_POL,
#                 gamma=GAMMA, hidden_dim=HIDDEN_DIM_POL, dropout=DROPOUT, log_interval=LOG_INTERVAL)
# aac_agent.load_pol_network(state_file='models/aac_polNet.pt')

# ep_rewards = aac_agent.polGrad_evaluation(n_episodes=10, vid_env=env)

# env.close()
# env_to_wrap.close()


#%% EVALUATION & VIDEO (SARSA)
env_to_wrap = gym.make('Acrobot-v1')
env = gym.wrappers.Monitor(env_to_wrap, 'videos/SARSA_DQN', force = True)

agent = Q_Agent(env, num_episodes=MAX_EPISODES, num_steps=500, learning_rate=LR_QNET,
                  gamma=GAMMA, epsilon=EPS, hidden_dim=HIDDEN_DIM_QNET, log_interval=LOG_INTERVAL)
agent.load_q_network(state_file='models/sarsa_dqn_qNet.pt')

ep_rewards = agent.evaluation(n_episodes=10, vid_env=env)

env.close()
env_to_wrap.close()