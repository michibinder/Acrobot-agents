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
from agents_nnp import A2C_Agent, MC_PolGrad_Agent

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
MAX_EPISODES = 1000
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


#%% Train Q_DQN-Agent (semi-gradient) ###
TARGET_NET = False
agent_results = list()
# hyperparam_dict = {'name': 'Q-DQN (' + str(HIDDEN_DIM_QNET) + ')'}
hyperparam_dict = {'name': 'Q-DQN (eps decay, C: 1)'}
agent = Q_DQN_Agent(env, num_episodes=MAX_EPISODES, num_steps=MAX_STEPS, learning_rate=LR_QNET,
                  gamma=GAMMA, epsilon=EPS, hidden_dim=HIDDEN_DIM_QNET, const_target=TARGET_NET, act_sel=ACTION_SELECTION, batch_size=MINI_BATCH_SIZE, log_interval=LOG_INTERVAL)
ep_rewards, running_rewards = agent.train()
agent_results.append(running_rewards)

agent = Q_DQN_Agent(env, num_episodes=MAX_EPISODES, num_steps=MAX_STEPS, learning_rate=LR_QNET,
                  gamma=GAMMA, epsilon=EPS, hidden_dim=HIDDEN_DIM_QNET, const_target=TARGET_NET, act_sel=ACTION_SELECTION, batch_size=MINI_BATCH_SIZE, log_interval=LOG_INTERVAL)
ep_rewards, running_rewards = agent.train()
agent_results.append(running_rewards)

agent = Q_DQN_Agent(env, num_episodes=MAX_EPISODES, num_steps=MAX_STEPS, learning_rate=LR_QNET,
                  gamma=GAMMA, epsilon=EPS, hidden_dim=HIDDEN_DIM_QNET, const_target=TARGET_NET, act_sel=ACTION_SELECTION, batch_size=MINI_BATCH_SIZE, log_interval=LOG_INTERVAL)
ep_rewards, running_rewards = agent.train()
agent_results.append(running_rewards)

agent = Q_DQN_Agent(env, num_episodes=MAX_EPISODES, num_steps=MAX_STEPS, learning_rate=LR_QNET,
                  gamma=GAMMA, epsilon=EPS, hidden_dim=HIDDEN_DIM_QNET, const_target=TARGET_NET, act_sel=ACTION_SELECTION, batch_size=MINI_BATCH_SIZE, log_interval=LOG_INTERVAL)
ep_rewards, running_rewards = agent.train()
agent_results.append(running_rewards)
agent_results = np.array(agent_results)
rewards_mu = agent_results.mean(axis=0)
rewards_sigma = agent_results.std(axis=0)
training_results.append((hyperparam_dict, rewards_mu, rewards_sigma))


#%% Train Q_DQN-Agent (semi-gradient) ###
TARGET_NET = True
agent_results = list()
hyperparam_dict = {'name': 'Q-DQN (eps decay, C: 200)'}
agent = Q_DQN_Agent(env, num_episodes=MAX_EPISODES, num_steps=MAX_STEPS, learning_rate=LR_QNET,
                  gamma=GAMMA, epsilon=EPS, hidden_dim=HIDDEN_DIM_QNET, const_target=TARGET_NET, act_sel=ACTION_SELECTION, batch_size=MINI_BATCH_SIZE, log_interval=LOG_INTERVAL)
ep_rewards, running_rewards = agent.train()
agent_results.append(running_rewards)

agent = Q_DQN_Agent(env, num_episodes=MAX_EPISODES, num_steps=MAX_STEPS, learning_rate=LR_QNET,
                  gamma=GAMMA, epsilon=EPS, hidden_dim=HIDDEN_DIM_QNET, const_target=TARGET_NET, act_sel=ACTION_SELECTION, batch_size=MINI_BATCH_SIZE, log_interval=LOG_INTERVAL)
ep_rewards, running_rewards = agent.train()
agent_results.append(running_rewards)

agent = Q_DQN_Agent(env, num_episodes=MAX_EPISODES, num_steps=MAX_STEPS, learning_rate=LR_QNET,
                  gamma=GAMMA, epsilon=EPS, hidden_dim=HIDDEN_DIM_QNET, const_target=TARGET_NET, act_sel=ACTION_SELECTION, batch_size=MINI_BATCH_SIZE, log_interval=LOG_INTERVAL)
ep_rewards, running_rewards = agent.train()
agent_results.append(running_rewards)

agent = Q_DQN_Agent(env, num_episodes=MAX_EPISODES, num_steps=MAX_STEPS, learning_rate=LR_QNET,
                  gamma=GAMMA, epsilon=EPS, hidden_dim=HIDDEN_DIM_QNET, const_target=TARGET_NET, act_sel=ACTION_SELECTION, batch_size=MINI_BATCH_SIZE, log_interval=LOG_INTERVAL)
ep_rewards, running_rewards = agent.train()
agent_results.append(running_rewards)
agent_results = np.array(agent_results)
rewards_mu = agent_results.mean(axis=0)
rewards_sigma = agent_results.std(axis=0)
training_results.append((hyperparam_dict, rewards_mu, rewards_sigma))


#%% Train Q_DQN-Agent (semi-gradient) ###
TARGET_NET = True
ACTION_SELECTION = 'softmax'
agent_results = list()
hyperparam_dict = {'name': 'Q-DQN (softmax, C: 200)'}
agent = Q_DQN_Agent(env, num_episodes=MAX_EPISODES, num_steps=MAX_STEPS, learning_rate=LR_QNET,
                  gamma=GAMMA, epsilon=EPS, hidden_dim=HIDDEN_DIM_QNET, const_target=TARGET_NET, act_sel=ACTION_SELECTION, batch_size=MINI_BATCH_SIZE, log_interval=LOG_INTERVAL)
ep_rewards, running_rewards = agent.train()
agent_results.append(running_rewards)

agent = Q_DQN_Agent(env, num_episodes=MAX_EPISODES, num_steps=MAX_STEPS, learning_rate=LR_QNET,
                  gamma=GAMMA, epsilon=EPS, hidden_dim=HIDDEN_DIM_QNET, const_target=TARGET_NET, act_sel=ACTION_SELECTION, batch_size=MINI_BATCH_SIZE, log_interval=LOG_INTERVAL)
ep_rewards, running_rewards = agent.train()
agent_results.append(running_rewards)

agent = Q_DQN_Agent(env, num_episodes=MAX_EPISODES, num_steps=MAX_STEPS, learning_rate=LR_QNET,
                  gamma=GAMMA, epsilon=EPS, hidden_dim=HIDDEN_DIM_QNET, const_target=TARGET_NET, act_sel=ACTION_SELECTION, batch_size=MINI_BATCH_SIZE, log_interval=LOG_INTERVAL)
ep_rewards, running_rewards = agent.train()
agent_results.append(running_rewards)

agent = Q_DQN_Agent(env, num_episodes=MAX_EPISODES, num_steps=MAX_STEPS, learning_rate=LR_QNET,
                  gamma=GAMMA, epsilon=EPS, hidden_dim=HIDDEN_DIM_QNET, const_target=TARGET_NET, act_sel=ACTION_SELECTION, batch_size=MINI_BATCH_SIZE, log_interval=LOG_INTERVAL)
ep_rewards, running_rewards = agent.train()
agent_results.append(running_rewards)
agent_results = np.array(agent_results)
rewards_mu = agent_results.mean(axis=0)
rewards_sigma = agent_results.std(axis=0)
training_results.append((hyperparam_dict, rewards_mu, rewards_sigma))



# #%% Train SARSA_DQN-Agent (semi-gradient) ###
# agent_results = list()
# HIDDEN_DIM_QNET = 32 # 32 SARSA top
# hyperparam_dict = {'name': 'SARSA DQN (' + str(HIDDEN_DIM_QNET) + ')'}
# agent = SARSA_DQN_Agent(env, num_episodes=MAX_EPISODES, num_steps=MAX_STEPS, learning_rate=LR_QNET,
#                   gamma=GAMMA, epsilon=EPS, hidden_dim=HIDDEN_DIM_QNET, const_target=True, act_sel='softmax', batch_size=MINI_BATCH_SIZE, log_interval=LOG_INTERVAL)
# ep_rewards, running_rewards = agent.train()
# agent_results.append(running_rewards)

# agent = SARSA_DQN_Agent(env, num_episodes=MAX_EPISODES, num_steps=MAX_STEPS, learning_rate=LR_QNET,
#                   gamma=GAMMA, epsilon=EPS, hidden_dim=HIDDEN_DIM_QNET, const_target=True, act_sel='softmax', batch_size=MINI_BATCH_SIZE, log_interval=LOG_INTERVAL)
# ep_rewards, running_rewards = agent.train()
# agent_results.append(running_rewards)

# agent = SARSA_DQN_Agent(env, num_episodes=MAX_EPISODES, num_steps=MAX_STEPS, learning_rate=LR_QNET,
#                   gamma=GAMMA, epsilon=EPS, hidden_dim=HIDDEN_DIM_QNET, const_target=True, act_sel='softmax', batch_size=MINI_BATCH_SIZE, log_interval=LOG_INTERVAL)
# ep_rewards, running_rewards = agent.train()
# agent_results.append(running_rewards)

# agent = SARSA_DQN_Agent(env, num_episodes=MAX_EPISODES, num_steps=MAX_STEPS, learning_rate=LR_QNET,
#                   gamma=GAMMA, epsilon=EPS, hidden_dim=HIDDEN_DIM_QNET, const_target=True, act_sel='softmax', batch_size=MINI_BATCH_SIZE, log_interval=LOG_INTERVAL)
# ep_rewards, running_rewards = agent.train()
# agent_results.append(running_rewards)
# agent_results = np.array(agent_results)
# rewards_mu = agent_results.mean(axis=0)
# rewards_sigma = agent_results.std(axis=0)
# training_results.append((hyperparam_dict, rewards_mu, rewards_sigma))


#%% VISUALIZATION
plt.rcParams.update({'font.size': 9})
width = 170/25.4 # 6.7in
# FIGSIZE = (width,width*1/3)
FIGSIZE = (width,width*3/8)

# Plot the results
fig, ax = plt.subplots(figsize=FIGSIZE)
i=0
for result in training_results:
    i += 1
    if i==3:
        continue
    hp = result[0]
    mu = result[1]
    sigma = result[2]
     
    plt.plot(range(len(mu)), mu, lw=1.2, label=hp['name'])
    ax.fill_between(range(len(mu)), mu+sigma, mu-sigma, alpha=0.5)
    
    # plt.plot(range(len(ep_rewards)), ep_rewards, lw=2, color="red", label=hp['name'])
    # title_str = "Acrobot-v1 ($hiddenDim_{qnet}$: " + str(HIDDEN_DIM_QNET) + ", $hiddenDim_{pol}$: " + str(HIDDEN_DIM_POL) + ")"
    # plt.title(title_str)

plt.grid()
plt.xlabel('Episodes')
plt.ylabel('Rewards$_{EMA}$')
plt.legend(loc='lower right', ncol=1) # ncol=1
fig.tight_layout()
plt.show()
fig.savefig('images/DQN_agents.pdf')

#%% Save neural network of agent
#sarsa_dqn_agent.save_q_network(save_dir="models", file_name="sarsa_dqn_qNet.pt")