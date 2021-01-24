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
from agents_nnp import A2C_Agent, MC_PolGrad_Agent
from networks import QNetwork, PolicyNetwork


#%% TRAINING
training_results = list() # A list for storing the hyperparameters and the corresponding results
MAX_EPISODES = 3000
MAX_STEPS = 500
LOG_INTERVAL = 100

LR_QNET = 0.0001
GAMMA = 0.99
EPS = 0.3
ACTION_SELECTION = 'eps_decay'
HIDDEN_DIM_QNET = 128

# DQN
MINI_BATCH_SIZE = 32 # for experience replay
# MEM_MAX_SIZE = 10000 # memory size for experience replay
# C_TARGET_NET_UPDATE = 100 # steps with constant target q-network
TARGET_NET = True

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
env = gym.wrappers.Monitor(env_to_wrap, 'videos/DQN', force = True)

agent = Q_DQN_Agent(env, num_episodes=MAX_EPISODES, num_steps=MAX_STEPS, learning_rate=LR_QNET,
                  gamma=GAMMA, epsilon=EPS, hidden_dim=HIDDEN_DIM_QNET, const_target=TARGET_NET, act_sel=ACTION_SELECTION, batch_size=MINI_BATCH_SIZE, log_interval=LOG_INTERVAL)
agent.load_q_network(state_file='models/DQN.pt')

ep_rewards = agent.evaluation(n_episodes=10, vid_env=env)

env.close()
env_to_wrap.close()