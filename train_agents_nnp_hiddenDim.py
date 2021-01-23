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
# env = gym.make('MountainCar-v0').env

#%% SEED
random_seed = 1234
np.random.seed(random_seed)
torch.manual_seed(random_seed)
env.seed(random_seed)

#%% TRAINING
training_results = list() # A list for storing the hyperparameters and the corresponding results
MAX_EPISODES = 1000
MAX_STEPS = 500
DROPOUT = 0.4 # 0.5
LR_POL = 0.001
GAMMA = 0.99
LOG_INTERVAL = 100

#%% Train AAC-Agent (neural network policy - agent) ###
agent_results = list()
HIDDEN_DIM = 32
hyperparam_dict = {'name': 'A2C (' + str(HIDDEN_DIM) + ')'}
aac_agent = AAC_Agent(env, num_episodes=MAX_EPISODES, num_steps=MAX_STEPS, learning_rate=LR_POL,
                      gamma=GAMMA, hidden_dim=HIDDEN_DIM, dropout=DROPOUT, log_interval=LOG_INTERVAL)
ep_rewards, running_rewards = aac_agent.train()
agent_results.append(running_rewards)

aac_agent = AAC_Agent(env, num_episodes=MAX_EPISODES, num_steps=MAX_STEPS, learning_rate=LR_POL,
                      gamma=GAMMA, hidden_dim=HIDDEN_DIM, dropout=DROPOUT, log_interval=LOG_INTERVAL)
ep_rewards, running_rewards = aac_agent.train()
agent_results.append(running_rewards)

aac_agent = AAC_Agent(env, num_episodes=MAX_EPISODES, num_steps=MAX_STEPS, learning_rate=LR_POL,
                      gamma=GAMMA, hidden_dim=HIDDEN_DIM, dropout=DROPOUT, log_interval=LOG_INTERVAL)
ep_rewards, running_rewards = aac_agent.train()
agent_results.append(running_rewards)
agent_results = np.array(agent_results)
rewards_mu = agent_results.mean(axis=0)
rewards_sigma = agent_results.std(axis=0)
training_results.append((hyperparam_dict, rewards_mu, rewards_sigma))



#%% Train AAC-Agent (neural network policy - agent) ###
agent_results = list()
HIDDEN_DIM = 64
hyperparam_dict = {'name': 'A2C (' + str(HIDDEN_DIM) + ')'}
aac_agent = AAC_Agent(env, num_episodes=MAX_EPISODES, num_steps=MAX_STEPS, learning_rate=LR_POL,
                      gamma=GAMMA, hidden_dim=HIDDEN_DIM, dropout=DROPOUT, log_interval=LOG_INTERVAL)
ep_rewards, running_rewards = aac_agent.train()
agent_results.append(running_rewards)

aac_agent = AAC_Agent(env, num_episodes=MAX_EPISODES, num_steps=MAX_STEPS, learning_rate=LR_POL,
                      gamma=GAMMA, hidden_dim=HIDDEN_DIM, dropout=DROPOUT, log_interval=LOG_INTERVAL)
ep_rewards, running_rewards = aac_agent.train()
agent_results.append(running_rewards)

aac_agent = AAC_Agent(env, num_episodes=MAX_EPISODES, num_steps=MAX_STEPS, learning_rate=LR_POL,
                      gamma=GAMMA, hidden_dim=HIDDEN_DIM, dropout=DROPOUT, log_interval=LOG_INTERVAL)
ep_rewards, running_rewards = aac_agent.train()
agent_results.append(running_rewards)
agent_results = np.array(agent_results)
rewards_mu = agent_results.mean(axis=0)
rewards_sigma = agent_results.std(axis=0)
training_results.append((hyperparam_dict, rewards_mu, rewards_sigma))


#%% Train AAC-Agent (neural network policy - agent) ###
agent_results = list()
HIDDEN_DIM = 128
hyperparam_dict = {'name': 'A2C (' + str(HIDDEN_DIM) + ')'}
aac_agent = AAC_Agent(env, num_episodes=MAX_EPISODES, num_steps=MAX_STEPS, learning_rate=LR_POL,
                      gamma=GAMMA, hidden_dim=HIDDEN_DIM, dropout=DROPOUT, log_interval=LOG_INTERVAL)
ep_rewards, running_rewards = aac_agent.train()
agent_results.append(running_rewards)

aac_agent = AAC_Agent(env, num_episodes=MAX_EPISODES, num_steps=MAX_STEPS, learning_rate=LR_POL,
                      gamma=GAMMA, hidden_dim=HIDDEN_DIM, dropout=DROPOUT, log_interval=LOG_INTERVAL)
ep_rewards, running_rewards = aac_agent.train()
agent_results.append(running_rewards)

aac_agent = AAC_Agent(env, num_episodes=MAX_EPISODES, num_steps=MAX_STEPS, learning_rate=LR_POL,
                      gamma=GAMMA, hidden_dim=HIDDEN_DIM, dropout=DROPOUT, log_interval=LOG_INTERVAL)
ep_rewards, running_rewards = aac_agent.train()
agent_results.append(running_rewards)
agent_results = np.array(agent_results)
rewards_mu = agent_results.mean(axis=0)
rewards_sigma = agent_results.std(axis=0)
training_results.append((hyperparam_dict, rewards_mu, rewards_sigma))


#%% Train AAC-Agent (neural network policy - agent) ###
agent_results = list()
HIDDEN_DIM = 256
hyperparam_dict = {'name': 'A2C (' + str(HIDDEN_DIM) + ')'}
aac_agent = AAC_Agent(env, num_episodes=MAX_EPISODES, num_steps=MAX_STEPS, learning_rate=LR_POL,
                      gamma=GAMMA, hidden_dim=HIDDEN_DIM, dropout=DROPOUT, log_interval=LOG_INTERVAL)
ep_rewards, running_rewards = aac_agent.train()
agent_results.append(running_rewards)

aac_agent = AAC_Agent(env, num_episodes=MAX_EPISODES, num_steps=MAX_STEPS, learning_rate=LR_POL,
                      gamma=GAMMA, hidden_dim=HIDDEN_DIM, dropout=DROPOUT, log_interval=LOG_INTERVAL)
ep_rewards, running_rewards = aac_agent.train()
agent_results.append(running_rewards)

aac_agent = AAC_Agent(env, num_episodes=MAX_EPISODES, num_steps=MAX_STEPS, learning_rate=LR_POL,
                      gamma=GAMMA, hidden_dim=HIDDEN_DIM, dropout=DROPOUT, log_interval=LOG_INTERVAL)
ep_rewards, running_rewards = aac_agent.train()
agent_results.append(running_rewards)
agent_results = np.array(agent_results)
rewards_mu = agent_results.mean(axis=0)
rewards_sigma = agent_results.std(axis=0)
training_results.append((hyperparam_dict, rewards_mu, rewards_sigma))


#%% Train AAC-Agent (neural network policy - agent) ###
agent_results = list()
HIDDEN_DIM = 512
hyperparam_dict = {'name': 'A2C (' + str(HIDDEN_DIM) + ')'}
aac_agent = AAC_Agent(env, num_episodes=MAX_EPISODES, num_steps=MAX_STEPS, learning_rate=LR_POL,
                      gamma=GAMMA, hidden_dim=HIDDEN_DIM, dropout=DROPOUT, log_interval=LOG_INTERVAL)
ep_rewards, running_rewards = aac_agent.train()
agent_results.append(running_rewards)

aac_agent = AAC_Agent(env, num_episodes=MAX_EPISODES, num_steps=MAX_STEPS, learning_rate=LR_POL,
                      gamma=GAMMA, hidden_dim=HIDDEN_DIM, dropout=DROPOUT, log_interval=LOG_INTERVAL)
ep_rewards, running_rewards = aac_agent.train()
agent_results.append(running_rewards)

aac_agent = AAC_Agent(env, num_episodes=MAX_EPISODES, num_steps=MAX_STEPS, learning_rate=LR_POL,
                      gamma=GAMMA, hidden_dim=HIDDEN_DIM, dropout=DROPOUT, log_interval=LOG_INTERVAL)
ep_rewards, running_rewards = aac_agent.train()
agent_results.append(running_rewards)
agent_results = np.array(agent_results)
rewards_mu = agent_results.mean(axis=0)
rewards_sigma = agent_results.std(axis=0)
training_results.append((hyperparam_dict, rewards_mu, rewards_sigma))


#%% Train AAC-Agent (neural network policy - agent) ###
agent_results = list()
HIDDEN_DIM = 1028
hyperparam_dict = {'name': 'A2C (' + str(HIDDEN_DIM) + ')'}
aac_agent = AAC_Agent(env, num_episodes=MAX_EPISODES, num_steps=MAX_STEPS, learning_rate=LR_POL,
                      gamma=GAMMA, hidden_dim=HIDDEN_DIM, dropout=DROPOUT, log_interval=LOG_INTERVAL)
ep_rewards, running_rewards = aac_agent.train()
agent_results.append(running_rewards)

aac_agent = AAC_Agent(env, num_episodes=MAX_EPISODES, num_steps=MAX_STEPS, learning_rate=LR_POL,
                      gamma=GAMMA, hidden_dim=HIDDEN_DIM, dropout=DROPOUT, log_interval=LOG_INTERVAL)
ep_rewards, running_rewards = aac_agent.train()
agent_results.append(running_rewards)

aac_agent = AAC_Agent(env, num_episodes=MAX_EPISODES, num_steps=MAX_STEPS, learning_rate=LR_POL,
                      gamma=GAMMA, hidden_dim=HIDDEN_DIM, dropout=DROPOUT, log_interval=LOG_INTERVAL)
ep_rewards, running_rewards = aac_agent.train()
agent_results.append(running_rewards)
agent_results = np.array(agent_results)
rewards_mu = agent_results.mean(axis=0)
rewards_sigma = agent_results.std(axis=0)
training_results.append((hyperparam_dict, rewards_mu, rewards_sigma))



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
fig.savefig('images/AAC_hidden_Dim_comp.pdf')

#%% Save neural network of agent
# aac_agent.save_pol_network(save_dir="models", file_name="aac_polNet.pt")

