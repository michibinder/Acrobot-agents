#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 16:22:08 2020

@author: tennismichel
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import copy

from networks import QNetwork, DQN_Network, PolicyNetwork

# Re-Initialization constants for weights (Xavier-nornal)
RE_INIT_VAL = -490
RE_INIT_EP = 150

LR_DECAY_RATE = 0.999

REWARD_THRESHOLD = -70  # low enough to not stop traning
START_EPS_DECAY = 400   # episode for MountainCar but this shouldn't be too important
EPS_DECAY_RATE = 0.995  # 0.9995 for MountainCar

# DQN
MEM_MAX_SIZE = 10000        # memory size for experience replay
C_TARGET_NET_UPDATE = 200   # steps with constant target q-network

class Base_Agent:
    """
    A basic agent with functions for evaluation and 
    """

    def __init__(self, env, num_episodes, num_steps, learning_rate, gamma, log_interval=100):
        """
        Constructor
        """
        self.env = env
        self.num_actions = env.action_space.n
        # self.num_actions = env.action_space.shape[0]
        self.reward_threshold = REWARD_THRESHOLD
        # self.reward_threshold = self.env.spec.reward_threshold

        self.num_episodes = num_episodes
        self.num_steps = num_steps

        self.learning_rate = learning_rate
        self.gamma = gamma
        
        self.log_interval = log_interval
        
       
    def random_action(self):
        """
        Select a random action.
        """
        return self.env.action_space.sample() 
    
    
    def eps_greedy_action(self, q):
        """
        Select an action based on epsilon-greedy policy
        """     
        if (np.random.rand(1) < self.epsilon):
            action = self.random_action()
        else:
            _, action = torch.max(q, -1)
            # Convert from tensor to float/int
            action = action.item()
            
        return action
    
    
    def softmax_action(self, q):
        """
        Select an action based on softmax policy
        """
        # _, amax = torch.max(q, -1)
        
        probs = F.softmax(q, dim=0)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        
        # if amax==action:
        #     print('TRUE')
        # else:
        #     print('FALSE')
            
        return action  
    
    
    def save_q_network(self, save_dir="models", file_name="network.pt"):
        """
        Save NN of agent.
        """
        self.q_net.save(save_dir=save_dir, file_name=file_name)
        return
    
    
    def load_q_network(self, state_file = "models/network.pt"):
        """
        Load NN of agent.
        """
        self.q_net.load_state_dict(torch.load(state_file))
        
        # self.q_network = QNetwork.load(save_dir=save_dir, file_name=file_name)
        return
    
    
    def save_pol_network(self, save_dir="models", file_name="network.pt"):
        """
        Save NN of agent.
        """
        self.actor.save(save_dir=save_dir, file_name=file_name)
        return
    
    
    def load_pol_network(self, state_file = "models/network.pt"):
        """
        Load NN of agent.
        """
        self.policy.actor.load_state_dict(torch.load(state_file))
        return
    

    def evaluation(self, n_episodes=10, vid_env=None):
        """
        Evaluates the agent (its trained policy)
        """
        
        # Load saved network
        # saved_q_network = NeuralNetwork.load(self.env)
        # aved_q_network.eval()
        
        # Evaluation mode
        self.q_net.eval()
        
        if vid_env != None:
            self.env = vid_env

        # Array to store cumulative rewards per episode
        ep_rewards = []
       
        print('Evaluation...')
        # For each episode
        for i_episode in range(1,n_episodes+1):
            
            # Cumulative reward
            ep_reward = 0
                
            # Initialize the environment and get the first state
            state = self.env.reset()
            
            # Set the done variable to False
            done = False

            # Variable for tracking the time
            t = 0
            
            # For each step of the episode
            while (not done):
                
                q = self.q_net(torch.autograd.Variable(torch.from_numpy(state).type(torch.FloatTensor)))
                                
                # Choose greedy action based on q
                _, action = torch.max(q, -1)
                # Convert from tensor to float/int
                action = action.item()
                    
                # Perform the action and observe the next_state and reward
                state, reward, done, _ = self.env.step(action)
                
                # Record history
                ep_reward += reward
                
                # Increment the timestep
                t += 1
                
                # Exit if the max number of steps has been exceeded
                if t>=self.num_steps:
                    done = True
                    
            print('Episode {}\tReward: {:.2f}'.format(
                  i_episode, ep_reward))
                
            # Store the cumulative reward for this episode        
            ep_rewards.append(ep_reward)
            
        return ep_rewards
    
    
    def polGrad_evaluation(self, n_episodes=10, vid_env=None):
        """
        Evaluates the agent (its trained policy)
        """
        
        # Load saved network
        # saved_q_network = NeuralNetwork.load(self.env)
        # aved_q_network.eval()
        
        # Evaluation mode
        self.policy.eval()
        
        if vid_env != None:
            self.env = vid_env
            
        # Array to store cumulative rewards per episode
        ep_rewards = []
       
        print('Evaluation...')
        # For each episode
        for i_episode in range(1,n_episodes+1):
            
            # Cumulative reward
            ep_reward = 0
                
            # Initialize the environment and get the first state
            state = self.env.reset()
            
            # Set the done variable to False
            done = False
    
            # Variable for tracking the time
            t = 0
            
            # For each step of the episode
            while (not done):
                # Convert the state from a numpy array to a torch tensor
                state = torch.from_numpy(state).float().unsqueeze(0)
                
                action, _ = self.policy(state)
                    
                # Perform the action and observe the next_state and reward
                state, reward, done, _ = self.env.step(action)

                # Record history
                ep_reward += reward
                
                # Increment the timestep
                t += 1
                
                # Exit if the max number of steps has been exceeded
                if t>=self.num_steps:
                    done = True
                    
            print('Episode {}\tReward: {:.2f}'.format(
                  i_episode, ep_reward))
                
            # Store the cumulative reward for this episode        
            ep_rewards.append(ep_reward)
            
        return ep_rewards
    
    
    def random_policy(self):
        """
        Evaluates the agent (its trained policy)
        """
        
        # Array to store cumulative rewards per episode
        ep_rewards = np.zeros((self.num_episodes, 1))
        
        # Lists to store the episodic and running rewards for plotting
        ep_rewards = list()
        running_rewards = list()
        
        # Track training
        print('Training...')
        
        # For each episode
        for i_episode in range(1,self.num_episodes+1):
            # Cumulative reward
            ep_reward = 0
            ep_loss = 0
                
            # Initialize the environment and get the first state
            state = self.env.reset()
            
            # Set the done variable to False
            done = False

            # Variable for tracking the time
            t = 0
                    
            # For each step of the episode
            while (not done):                               
                action = self.random_action()
                    
                # Perform the action and observe the next_state and reward
                next_state, reward, done, _ = self.env.step(action)

                ep_reward += reward
                
                # Update the state
                state = next_state
                
                # Increment the timestep
                t += 1
                
                # Exit if the max number of steps has been exceeded
                if t>=self.num_steps:
                    done = True
            
            # Update the running reward
            if i_episode==1:
                running_reward = ep_reward
            else:
                running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
            
            # Store the rewards for plotting
            ep_rewards.append(ep_reward)
            running_rewards.append(running_reward)
            
            if i_episode % self.log_interval == 0:
                print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                      i_episode, ep_reward, running_reward))
            # Stopping criteria
            if running_reward > self.env.spec.reward_threshold:
                print('Running reward is now {} and the last episode ran for {} steps!'.format(running_reward, t))
                break
            if i_episode >= self.num_episodes:
                print('Max episodes exceeded, quitting.')
                break

        return ep_rewards, running_rewards
    
    
    @staticmethod
    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_normal_(m.weight)
            try:
                m.bias.data.fill_(0)
            except:
                print('CATCH: Xavier-normal init only applied to weights, bias not available.')
                

class Q_Agent(Base_Agent):
    """
    A Q-Learning agent.
    """

    def __init__(self, env, num_episodes, num_steps, learning_rate, gamma, epsilon=0.3, hidden_dim=100, act_sel='eps_decay', log_interval=100):
        """
        Constructor
        """
        super().__init__(env, num_episodes, num_steps, learning_rate, gamma, log_interval=log_interval)
        
        self.epsilon = epsilon
        
        self.q_net = QNetwork(env=self.env, hidden_dim=hidden_dim)
        self.q_net.apply(Q_Agent.init_weights)
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.learning_rate)
        # self.optimizer = optim.SGD(self.q_network.parameters(), lr=self.learning_rate)
        self.criteria = nn.MSELoss()
        
        # Learning rate schedule
        self.lr_decayRate = LR_DECAY_RATE
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=self.lr_decayRate)
        
        self.act_sel = act_sel
       
       
    def train(self):
        """
        Implementation of the Q-Learning algorithm using a neural network as the function approximator
        for the action-value function.
        
        returns: 
            ep_rewards: Array containg the cumulative reward in each episode
            running_rewards
        """
        # Array to store cumulative rewards per episode
        ep_rewards = np.zeros((self.num_episodes, 1))
        
        # Lists to store the episodic and running rewards for plotting
        ep_rewards = list()
        running_rewards = list()
        
        # Track training
        print('Training...')
        
        # For each episode
        for i_episode in range(1,self.num_episodes+1):
            # Cumulative reward
            ep_reward = 0
            ep_loss = 0
                
            # Initialize the environment and get the first state
            state = self.env.reset()
            
            # Set the done variable to False
            done = False
            
            if self.act_sel=='eps_decay' and i_episode>START_EPS_DECAY:
                self.epsilon = EPS_DECAY_RATE*self.epsilon
                
            # Variable for tracking the time
            t = 0
                    
            # For each step of the episode
            while (not done):
                
                # Use the state as input to compute the q-values (for all actions in 1 forward pass)
                q = self.q_net(torch.autograd.Variable(torch.from_numpy(state).type(torch.FloatTensor)))
                                
                if self.act_sel == 'eps_decay':
                    action = self.eps_greedy_action(q)
                else:
                    action = self.softmax_action(q)
                    
                # Perform the action and observe the next_state and reward
                next_state, reward, done, _ = self.env.step(action)
                                
                # Find max q for next state
                with torch.no_grad():
                    q_next = self.q_net(torch.autograd.Variable(torch.from_numpy(next_state).type(torch.FloatTensor)))
                    q_next = q_next.detach()
                max_q_next, _ = torch.max(q_next, -1)
                
                # Create target q value for training
                q_target = q.clone()
                q_target = torch.autograd.Variable(q_target.data)
                q_target[action] = reward + torch.mul(max_q_next.detach(), self.gamma)

                # Calculate loss
                loss = self.criteria(q, q_target)

                # Update policy
                self.q_net.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Record history
                ep_loss += loss.item()
                ep_reward += reward
                
                # Update the state
                state = next_state
                
                # Increment the timestep
                t += 1
                
                # Exit if the max number of steps has been exceeded
                if t>=self.num_steps:
                    done = True
            
            # Update the running reward
            if i_episode==1:
                running_reward = ep_reward
            else:
                running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
            
            # Store the rewards for plotting
            ep_rewards.append(ep_reward)
            running_rewards.append(running_reward)
            
            if i_episode % self.log_interval == 0:
                print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                      i_episode, ep_reward, running_reward))
            # Stopping criteria
            if running_reward > self.reward_threshold:
                print('Running reward is now {} and the last episode ran for {} steps!'.format(running_reward, t))
                break
            if i_episode >= self.num_episodes:
                print('Max episodes exceeded, quitting.')
                break
            
            if i_episode % RE_INIT_EP == 0 and running_reward < RE_INIT_VAL:
                self.q_net.apply(Q_Agent.init_weights)
                

        return ep_rewards, running_rewards    


#%%
class SARSA_Agent(Base_Agent):
    """
    A Sarsa agent.
    """

    def __init__(self, env, num_episodes, num_steps, learning_rate, gamma, epsilon=0.3, hidden_dim=100, act_sel='eps_decay', log_interval=100):
        """
        Constructor
        """
        super().__init__(env, num_episodes, num_steps, learning_rate, gamma, log_interval=log_interval)
        
        self.epsilon = epsilon
        
        self.q_net = QNetwork(env=self.env, hidden_dim=hidden_dim)
        self.q_net.apply(Base_Agent.init_weights)
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.learning_rate)
        self.criteria = nn.MSELoss()
        
        # Learning rate schedule
        self.lr_decayRate = LR_DECAY_RATE
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=self.lr_decayRate)
        
        self.act_sel = act_sel
        
        
    def train(self):
        """
        Implementation of the SARSA-Learning algorithm using a neural network as the function approximator
        for the action-value function.
        
        returns: 
            ep_rewards: Array containg the cumulative reward in each episode
            running_rewars
        """
        
        # Lists to store the episodic and running rewards for plotting
        ep_rewards = list()
        running_rewards = list()
        
        # Track training
        print('Training...')
        
        # For each episode
        for i_episode in range(1,self.num_episodes+1):
            
            # Cumulative reward
            ep_reward = 0
            ep_loss = 0
                
            # Initialize the environment and get the first state
            state = self.env.reset()
            
            # Set the done variable to False
            done = False

            # Variable for tracking the time
            t = 0
            
            # Use the state as input to compute the q-values (for all actions in 1 forward pass)
            q = self.q_net(torch.autograd.Variable(torch.from_numpy(state).type(torch.FloatTensor)))
            
            if self.act_sel=='eps_decay' and i_episode>START_EPS_DECAY:
                self.epsilon = EPS_DECAY_RATE*self.epsilon
                
            if self.act_sel == 'eps_decay':
                action = self.eps_greedy_action(q)
            else:
                action = self.softmax_action(q)
            
            # For each step of the episode
            while (not done):
                # Use the state as input to compute the q-values (for all actions in 1 forward pass)
                q = self.q_net(torch.autograd.Variable(torch.from_numpy(state).type(torch.FloatTensor)))
                
                # Perform the action and observe the next_state and reward
                next_state, reward, done, _ = self.env.step(action)
                
                #### Perform next action
                # Use the state as input to compute the q-values (for all actions in 1 forward pass)
                qprime = self.q_net(torch.autograd.Variable(torch.from_numpy(next_state).type(torch.FloatTensor)))
                
                if self.act_sel == 'eps_decay':
                    aprime = self.eps_greedy_action(qprime)
                else:
                    aprime = self.softmax_action(qprime)
                
                # Find q for next state and action
                with torch.no_grad():
                    q_next = self.q_net(torch.autograd.Variable(torch.from_numpy(next_state).type(torch.FloatTensor)))
                    q_next = q_next.detach()
                    q_next_a = q_next[aprime]
                    
                # Create target q value for training
                q_target = q.clone()
                q_target = torch.autograd.Variable(q_target.data)
                q_target[action] = reward + torch.mul(q_next_a.detach(), self.gamma)
                
                # Calculate loss
                loss = self.criteria(q, q_target)

                # Update policy
                self.q_net.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Record history
                ep_loss += loss.item()
                ep_reward += reward
                
                # Update the state
                state = next_state
                
                #### update action 
                action = aprime
                
                # Increment the timestep
                t += 1
                
                # Exit if the max number of steps has been exceeded
                if t>=self.num_steps:
                    done = True
            
            # Update the running reward
            if i_episode==1:
                running_reward = ep_reward
            else:
                running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
            
            # Store the rewards for plotting
            ep_rewards.append(ep_reward)
            running_rewards.append(running_reward)
            
            
            if i_episode % self.log_interval == 0:
                print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                      i_episode, ep_reward, running_reward))
            # Stopping criteria
            if running_reward > self.reward_threshold:
                print('Running reward is now {} and the last episode ran for {} steps!'.format(running_reward, t))
                break
            if i_episode >= self.num_episodes:
                print('Max episodes exceeded, quitting.')
                break
            
            if i_episode % RE_INIT_EP == 0 and running_reward < RE_INIT_VAL:
                self.q_net.apply(SARSA_Agent.init_weights)
                
        return ep_rewards, running_rewards
    

#%%    
class Q_DQN_Agent(Base_Agent):
    """
    A DQN Q-Learning agent.
    """

    def __init__(self, env, num_episodes, num_steps, learning_rate, gamma, epsilon=0.3, hidden_dim=100, const_target=True, act_sel = 'softmax', batch_size=8, log_interval=100):
        """
        Constructor
        """
        super().__init__(env, num_episodes, num_steps, learning_rate, gamma, log_interval=log_interval)
        
        self.epsilon = epsilon
        
        # Networks
        # self.q_net = QNetwork(env=self.env, hidden_dim=hidden_dim)
        # self.target_q_net = QNetwork(env=self.env, hidden_dim=hidden_dim)
        self.q_net = DQN_Network(env=self.env, hidden_dim=hidden_dim)
        self.target_q_net = DQN_Network(env=self.env, hidden_dim=hidden_dim)
        self.q_net.apply(Base_Agent.init_weights)
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.learning_rate)
        # self.optimizer = optim.SGD(self.q_network.parameters(), lr=self.learning_rate)
        
        # Learning rate schedule
        # self.lr_decayRate = LR_DECAY_RATE
        # self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=self.lr_decayRate)
        
        # Loss function
        self.criteria = nn.MSELoss()
        
        # DQN
        self.minibatch_size=batch_size
        self.mem_max_size = MEM_MAX_SIZE
        self.C = C_TARGET_NET_UPDATE
        
        self.const_target = const_target        
        self.act_sel = act_sel
        
        
    def replay(self, replay_memory):
        # choose <s,a,r,s',done> experiences randomly from the memory
        minibatch = np.random.choice(replay_memory, self.minibatch_size, replace=True)
        
        # create one list containing s, one list containing a, etc
        s_l =      np.array(list(map(lambda x: x['s'], minibatch)))
        a_l =      np.array(list(map(lambda x: x['a'], minibatch)))
        r_l =      np.array(list(map(lambda x: x['r'], minibatch)))
        sprime_l = np.array(list(map(lambda x: x['sprime'], minibatch)))
        done_l   = np.array(list(map(lambda x: x['done'], minibatch)))
        
        # Find q(s,a) for all possible actions a. Store in list
        s_l = Variable(torch.from_numpy(s_l).type(torch.FloatTensor))
        q = self.q_net(s_l)
        q_target = q.clone()
        
        # Find q(s', a') for all possible actions a'. Store in list
        # We'll use the maximum of these values for q-update
        sprime_l = Variable(torch.from_numpy(sprime_l).type(torch.FloatTensor))
        with torch.no_grad():
            qvals_sprime_l = self.target_q_net(sprime_l)
        
        # q_target
        # For the action we took, use the q-update value  
        # For other actions, use the current nnet predicted value
        for i,(s,a,r,qvals_sprime, done) in enumerate(zip(s_l,a_l,r_l,qvals_sprime_l, done_l)): 
            if not done:  target = r + self.gamma * torch.max(qvals_sprime.detach())
            else:         target = r
            q_target[i][a] = target
            
        # Loss function is 0 for actions we didn't take        
        loss = self.criteria(q, q_target)
        
        return loss


    def train(self):
        """
        Implementation of the Q-Learning algorithm using a neural network as the function approximator
        for the action-value function.
        
        returns: 
            ep_rewards: Array containg the cumulative reward in each episode
            running_rewards
        """
        # Array to store cumulative rewards per episode
        ep_rewards = np.zeros((self.num_episodes, 1))
        
        # Lists to store the episodic and running rewards for plotting
        ep_rewards = list()
        running_rewards = list()
        
        replay_memory = [] # replay memory holds s, a, r, s'
        step = 0
        
        # Track training
        print('Training...')
        
        # For each episode
        for i_episode in range(1,self.num_episodes+1):
            # Cumulative reward
            ep_reward = 0
            ep_loss = 0
                
            # Initialize the environment and get the first state
            state = self.env.reset()
            
            # Set the done variable to False
            done = False

            # Variable for tracking the time
            t = 0
            
            # Decrease epsilon
            if self.act_sel=='eps_decay' and i_episode>START_EPS_DECAY:
                self.epsilon = EPS_DECAY_RATE*self.epsilon
                
            # For each step of the episode
            while (not done):
                
                # Use the state as input to compute the q-values (for all actions in 1 forward pass)
                q = self.q_net(torch.autograd.Variable(torch.from_numpy(state).type(torch.FloatTensor)))
                
                if self.act_sel == 'eps_decay':
                    action = self.eps_greedy_action(q)
                else:
                    action = self.softmax_action(q)
                    
                # Perform the action and observe the next_state and reward
                sprime, reward, done, _ = self.env.step(action)
                
                # add to memory, respecting memory buffer limit 
                if len(replay_memory) > self.mem_max_size:
                    replay_memory.pop(0)
                replay_memory.append({"s":state,"a":action,"r":reward,"sprime":sprime,"done":done})
        
                # Calculate loss
                loss = self.replay(replay_memory)
                
                # Update policy
                self.q_net.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Record history
                ep_loss += loss.item()
                ep_reward += reward
                
                # Update the state
                state = sprime          
                
                # Increment the timestep
                t += 1
                step += 1
                
                # Exit if the max number of steps has been exceeded
                if t>=self.num_steps:
                    done = True
                
                # Exchange target network
                if self.const_target and (step % self.C == 0):
                    self.target_q_net.load_state_dict(self.q_net.state_dict())
            
            # Update the running reward
            if i_episode==1:
                running_reward = ep_reward
            else:
                running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
            
            # Store the rewards for plotting
            ep_rewards.append(ep_reward)
            running_rewards.append(running_reward)
            
            if i_episode % self.log_interval == 0:
                print('Episode {}\tEpsilon: {:.2f}\tAverage reward: {:.2f}'.format(
                      i_episode, self.epsilon, running_reward))
            # Stopping criteria
            if running_reward > self.reward_threshold:
                print('Running reward is now {} and the last episode ran for {} steps!'.format(running_reward, t))
                break
            if i_episode >= self.num_episodes:
                print('Max episodes exceeded, quitting.')
                break
            
            if i_episode % RE_INIT_EP == 0 and running_reward < RE_INIT_VAL:
                self.q_net.apply(Q_DQN_Agent.init_weights)

        return ep_rewards, running_rewards
            
            
#%%
class SARSA_DQN_Agent(Base_Agent):
    """
    A DQN SARSA-Learning agent.
    """

    def __init__(self, env, num_episodes, num_steps, learning_rate, gamma, epsilon=0.1, hidden_dim=100, const_target=True, act_sel = 'softmax', batch_size = 8, log_interval=100):
        """
        Constructor
        """
        super().__init__(env, num_episodes, num_steps, learning_rate, gamma, log_interval=log_interval)
        
        self.epsilon = epsilon
        
        self.q_net = QNetwork(env=self.env, hidden_dim=hidden_dim)
        self.target_q_net = QNetwork(env=self.env, hidden_dim=hidden_dim)
        self.q_net.apply(Base_Agent.init_weights)
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.learning_rate)
        
        # Learning rate schedule
        # self.lr_decayRate = LR_DECAY_RATE
        # self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=self.lr_decayRate)
        
        # Loss function
        self.criteria = nn.MSELoss()
        
        # DQN
        self.minibatch_size=batch_size # for experience replay
        self.mem_max_size = MEM_MAX_SIZE # 100000
        self.C = C_TARGET_NET_UPDATE # steps with constant target q-network
        
        self.const_target = const_target        
        self.act_sel = act_sel # softmax or eps_decay
        
        
    def replay(self, replay_memory):
        # choose <s,a,r,s',done> experiences randomly from the memory
        minibatch = np.random.choice(replay_memory, self.minibatch_size, replace=True)
        
        # create one list containing s, one list containing a, etc
        s_l =      np.array(list(map(lambda x: x['s'], minibatch)))
        a_l =      np.array(list(map(lambda x: x['a'], minibatch)))
        r_l =      np.array(list(map(lambda x: x['r'], minibatch)))
        sprime_l = np.array(list(map(lambda x: x['sprime'], minibatch)))
        aprime_l = np.array(list(map(lambda x: x['aprime'], minibatch)))
        done_l   = np.array(list(map(lambda x: x['done'], minibatch)))
        
        # Find q(s,a) for all possible actions a. Store in list
        s_l = Variable(torch.from_numpy(s_l).type(torch.FloatTensor))
        q = self.q_net.forward(s_l)
        q_target = q.clone()
        
        # Find q(s', a') for all possible actions a'. Store in list
        # We'll use the maximum of these values for q-update
        with torch.no_grad(): 
            sprime_l = Variable(torch.from_numpy(sprime_l).type(torch.FloatTensor))
            if self.const_target:
                qvals_sprime_l = self.target_q_net(sprime_l)
            else:
                qvals_sprime_l = self.q_net(sprime_l)
             
        # q-update target
        # For the action we took, use the q-update value  
        # For other actions, use the current nnet predicted value
        for i,(s,a,r,qvals_sprime, aprime, done) in enumerate(zip(s_l,a_l,r_l,qvals_sprime_l, aprime_l, done_l)): 
            if not done:  target = r + self.gamma * qvals_sprime.detach()[aprime]
            else:         target = r
            q_target[i][a] = target
            
        # Update weights of neural network 
        # Loss function is 0 for actions we didn't take        
        loss = self.criteria(q, q_target)
        
        return loss
    
    
    def train(self):
        """
        Implementation of the SARSA-Learning algorithm using a neural network as the function approximator
        for the action-value function.
        
        returns: 
            ep_rewards: Array containg the cumulative reward in each episode
            running_rewars
        """
        
        # Lists to store the episodic and running rewards for plotting
        ep_rewards = list()
        running_rewards = list()
        
        replay_memory = [] # replay memory holds s, a, r, s'
        step = 0
        
        # Track training
        print('Training...')
        
        # For each episode
        for i_episode in range(1,self.num_episodes+1):
            
            # Cumulative reward
            ep_reward = 0
            ep_loss = 0
                
            # Initialize the environment and get the first state
            state = self.env.reset()
            
            # Set the done variable to False
            done = False

            # Variable for tracking the time
            t = 0
            
            # Use the state as input to compute the q-values (for all actions in 1 forward pass)
            q = self.q_net(torch.autograd.Variable(torch.from_numpy(state).type(torch.FloatTensor)))
            
            # Decrease epsilon
            if self.act_sel=='eps_decay' and i_episode>START_EPS_DECAY:
                self.epsilon = EPS_DECAY_RATE*self.epsilon
            
            if self.act_sel=='eps_decay':    
                action = self.eps_greedy_action(q)
            else:
                action = self.softmax_action(q)
            
            # For each step of the episode
            while (not done):
                # Perform the action and observe the next_state and reward
                sprime, reward, done, _ = self.env.step(action)
                
                # Next action
                # Use the state as input to compute the q-values (for all actions in 1 forward pass)
                with torch.no_grad():           
                    qprime = self.q_net(torch.autograd.Variable(torch.from_numpy(sprime).type(torch.FloatTensor)))
                
                if self.act_sel=='eps_decay':    
                    aprime = self.eps_greedy_action(qprime)
                else:
                    aprime = self.softmax_action(qprime)
                
                # add to memory, respecting memory buffer limit 
                if len(replay_memory) > self.mem_max_size:
                    replay_memory.pop(0)
                replay_memory.append({"s":state,"a":action,"r":reward,"sprime":sprime, "aprime":aprime, "done":done})
        
                # Calculate loss
                loss = self.replay(replay_memory)
                
                # Update policy
                self.q_net.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Record history
                ep_loss += loss.item()
                ep_reward += reward
                
                # Update the state and action
                state = sprime
                action = aprime
                
                # Increment the timestep
                t += 1
                step += 1
                    
                # Exit if the max number of steps has been exceeded
                if t>=self.num_steps:
                    done = True
            
                if self.const_target and (step % self.C == 0):
                    self.target_q_net.load_state_dict(self.q_net.state_dict())
                    
            
            # Update the running reward
            if i_episode==1:
                running_reward = ep_reward
            else:
                running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
            
            # Store the rewards for plotting
            ep_rewards.append(ep_reward)
            running_rewards.append(running_reward)
            
            
            if i_episode % self.log_interval == 0:
                print('Episode {}\tEpsilon: {:.2f}\tAverage reward: {:.2f}'.format(
                      i_episode, self.epsilon, running_reward))
            # Stopping criteria
            if running_reward > self.reward_threshold:
                print('Running reward is now {} and the last episode ran for {} steps!'.format(running_reward, t))
                break
            if i_episode >= self.num_episodes:
                print('Max episodes exceeded, quitting.')
                break
            
            if i_episode % RE_INIT_EP == 0 and running_reward < RE_INIT_VAL:
                self.q_net.apply(SARSA_DQN_Agent.init_weights)

        return ep_rewards, running_rewards
    