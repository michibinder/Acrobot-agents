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
import os

# os.environ['KMP_DUPLICATE_LIB_OK']='True'

#%% TRAINING
training_results = list() # A list for storing the hyperparameters and the corresponding results
MAX_EPISODES = 3000
MAX_STEPS = 500
LOG_INTERVAL = 100

LR_QNET = 0.0001
GAMMA = 0.99
EPS = 0.3
ACTION_SELECTION = 'eps_decay'
HIDDEN_DIM_QNET = 256

# DQN
MINI_BATCH_SIZE = 32 # for experience replay
# MEM_MAX_SIZE = 10000 # memory size for experience replay
# C_TARGET_NET_UPDATE = 100 # steps with constant target q-network
TARGET_NET = True

# #%% EVALUATION & VIDEO (A2C)
# env_to_wrap = gym.make('Acrobot-v1')
# env = gym.wrappers.Monitor(env_to_wrap, 'videos/A2C', force = True)

# agent = A2C_Agent(env, num_episodes=MAX_EPISODES, num_steps=500, learning_rate=LR_POL,
#                 gamma=GAMMA, hidden_dim=HIDDEN_DIM_POL, dropout=DROPOUT, log_interval=LOG_INTERVAL)
# agent.load_pol_network(state_file='models/aac_polNet.pt')

# ep_rewards = gent.polGrad_evaluation(n_episodes=10, vid_env=env)

# env.close()
# env_to_wrap.close()


#%% EVALUATION & VIDEO (DQN)
env_to_wrap = gym.make('Acrobot-v1')
env = gym.wrappers.Monitor(env_to_wrap, 'videos/DQN', force = True)

agent = Q_DQN_Agent(env, num_episodes=MAX_EPISODES, num_steps=MAX_STEPS, learning_rate=LR_QNET,
                  gamma=GAMMA, epsilon=EPS, hidden_dim=HIDDEN_DIM_QNET, const_target=TARGET_NET, act_sel=ACTION_SELECTION, batch_size=MINI_BATCH_SIZE, log_interval=LOG_INTERVAL)
agent.load_q_network(state_file='models/DQN.pt')

ep_rewards = agent.evaluation(n_episodes=10, vid_env=env)

env.close()
env_to_wrap.close()