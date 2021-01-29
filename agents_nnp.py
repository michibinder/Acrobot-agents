#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 00:08:25 2020

@author: tennismichel
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import copy
from itertools import chain

from networks import PolicyNetwork, CriticNetwork, ActorCriticNetworks
from agents import Base_Agent

RE_INIT_VAL = -490
RE_INIT_EP = 150

class MC_PolGrad_Agent(Base_Agent):
    """
    A Monte Carlo policy gradient (with network) agent. (REINFORCE algorithm)
    """

    def __init__(self, env, num_episodes, num_steps, learning_rate, gamma, hidden_dim=100, dropout=0.6, log_interval=100):
        """
        Constructor
        """
        super().__init__(env, num_episodes, num_steps, learning_rate, gamma, log_interval=log_interval)
        
        self.policy = PolicyNetwork(self.env, hidden_dim=hidden_dim, dropout=dropout)
        self.policy.apply(Base_Agent.init_weights)
        
        # Define the optimizer and set the learning rate
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        
        # Learning rate schedule
        # self.lr_decayRate = 0.999
        # self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=self.lr_decayRate)
        
        # self.criteria = nn.MSELoss()
        # loss_function = nn.CrossEntropyLoss()
        
        self.rewards = []
    
   
    def compute_returns(self):
        """
        Function for computing returns from the list of rewards observed during an episode.
        
        Hint: Take special care of the fact that in the algorithm, the list of rewards are for 
        steps 1,2,...,T. Since Python follows a zero-based indexing system, the kth reward is 
        accessed by: rewards[k-1].
        
        Args:
            rewards (list): Rewards observed in an episode [r1, r1, ..., rT] for T steps.
        Returns:
            (list): List of returns.
        """
        
        rewards = self.rewards
        returns = list()
        len_rewards = len(rewards)
        for i, reward in enumerate(rewards):
            return_i = 0
            k = 0 # k is exponent of gamma - j iterates through relevant rewards
            for j in range(i, len_rewards):
                return_i += self.gamma**k * rewards[j]
                k += 1
            returns.append(return_i) 
        
        return returns
    
    
    def update_policy(self):
        """
        Performs the parameter updates of the policy network after an episode is completed.
        """
        
        policy_loss = []
    
        # Define a small float which is used to avoid divison by zero
        eps = np.finfo(np.float32).eps.item()
    
        # Go through the list of observed rewards and calculate the returns
        returns = self.compute_returns()
    
        # Convert the list of returns into a torch tensor
        returns = torch.tensor(returns)
    
        # Here we normalize the returns by subtracting the mean and dividing
        # by the standard deviation. Normalization is a standard technique in
        # deep learning and it improves performance, as discussed in 
        # http://karpathy.github.io/2016/05/31/rl/
        returns = (returns - returns.mean()) / (returns.std() + eps)
    
        # Here, we deviate slightly from the standard REINFORCE algorithm
        # `-log_prob * G` instead of `log_prob * G` for gradient ASCENT
        for log_prob, G in zip(self.policy.saved_log_probs, returns):
            policy_loss.append(-log_prob * G)
    
        # Reset the gradients of the parameters
        self.optimizer.zero_grad()
    
        # Compute the cumulative loss
        policy_loss = torch.cat(policy_loss).mean()
    
        # Backpropagate the loss through the network
        policy_loss.backward()
    
        # Perform a parameter update step
        self.optimizer.step()
    
        # Reset the saved rewards and log probabilities
        del self.rewards[:]
        del self.policy.saved_log_probs[:]
    
    
    def train(self, save=False):
        """
        Implementation of the main body of the REINFORCE algorithm.
        
        Args:
            save (bool): Whether to save the trained network. Defaults to False.
            
        Returns:
            ep_rewards (list): List of actual cumulative rewards in each episode. 
            running_rewards (numpy array): List of smoothed cumulative rewards in each episode. 
        """
    
        # Set the training mode
        self.policy.train()
    
        # Lists to store the episodic and running rewards for plotting
        ep_rewards = list()
        running_rewards = list()
    
        # Track training
        print('Training...')
        # Start executing an episode
        for i_episode in range(1, self.num_episodes+1):
            # Reset the environment
            state = self.env.reset()
            t = 0
            
            # Initialize `ep_reward` (the total reward for this episode)
            ep_reward = 0
            
            # 3. For each step of the episode
            while (True):
                # Convert the state from a numpy array to a torch tensor
                state = torch.from_numpy(state).float().unsqueeze(0)
                
                # Select an action using the policy network
                action = self.policy.select_action(state)
                # action = self.select_action(state)
                
                # 3.2 Perform the action and note the next state and reward and if the episode is done
                state, reward, done, info = self.env.step(action)
                
                # 3.3 Store the current reward in `policy.rewards`
                self.rewards.append(reward)
                
                # 3.4 Increment the total reward in this episode
                ep_reward += reward
                
                # Steps in episode for output required
                t += 1
                
                if t>=self.num_steps:
                    done=True
                    
                # 3.5 Check if the episode is finished using the `done` variable and break if yes
                if (done):
                    break
    
            # Update the running reward
            if i_episode==1:
                running_reward = ep_reward
            else:
                running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
    
            # Store the rewards for plotting
            ep_rewards.append(ep_reward)
            running_rewards.append(running_reward)
            
            # Perform the parameter update according to REINFORCE
            self.update_policy()
    
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
                self.policy.apply(Base_Agent.init_weights)
            
            
        # Save the trained policy network
        if save:
            self.policy.save()
    
        return ep_rewards, running_rewards
    
    
class A2C_Agent(Base_Agent):
    """
    A Monte-Carlo advantage actor-critic policy agent.
    """

    def __init__(self, env, num_episodes, num_steps, learning_rate, gamma, hidden_dim=100, dropout=0.6, log_interval=100):
        """
        Constructor
        """
        super().__init__(env, num_episodes, num_steps, learning_rate, gamma, log_interval=log_interval)
        
        ### Networks
        self.actor = PolicyNetwork(self.env, hidden_dim=hidden_dim, dropout=dropout)
        self.critic = CriticNetwork(env, hidden_dim=hidden_dim, dropout=dropout)
        
        self.policy = ActorCriticNetworks(self.actor, self.critic)
        self.policy.apply(Base_Agent.init_weights)
        
        ### Optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate, weight_decay=0)
        # self.optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        
        # all_params = chain(self.actor.parameters(), self.critic.parameters())
        # self.optimizer = optim.Adam(all_params, lr=self.learning_rate)
        
        ### Learning rate decay
        # self.lr_decayRate = 0.999
        # self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=self.lr_decayRate)
        
        # self.criteria = nn.MSELoss()
        self.rewards = []
        self.values = []
        
        
    def compute_returns(self):
        """
        Function for computing returns from the list of rewards observed during an episode.
        
        Hint: Take special care of the fact that in the algorithm, the list of rewards are for 
        steps 1,2,...,T. Since Python follows a zero-based indexing system, the kth reward is 
        accessed by: rewards[k-1].
        
        Args:
            rewards (list): Rewards observed in an episode [r1, r1, ..., rT] for T steps.
        Returns:
            (list): List of returns.
        """
        
        rewards = self.rewards
        returns = list()
        len_rewards = len(rewards)
        for i, reward in enumerate(rewards):
            return_i = 0
            k = 0 # k is exponent of gamma - j iterates through relevant rewards
            for j in range(i, len_rewards):
                return_i += self.gamma**k * rewards[j]
                k += 1
            returns.append(return_i) 
        
        return returns
    
    
    def update_policy(self):
        """
        Performs the parameter updates of the policy network after an episode is completed.
        """
        
        policy_loss = []
    
        # Define a small float which is used to avoid divison by zero
        eps = np.finfo(np.float32).eps.item()
    
        # Go through the list of observed rewards and calculate the returns
        returns = self.compute_returns()
    
        # Convert the list of returns into a torch tensor
        returns = torch.tensor(returns)
    
        # Here we normalize the returns by subtracting the mean and dividing
        # by the standard deviation. Normalization is a standard technique in
        # deep learning and it improves performance, as discussed in 
        # http://karpathy.github.io/2016/05/31/rl/
        returns = (returns - returns.mean()) / (returns.std() + eps)
        
        values = torch.cat(self.values).squeeze(-1)
        advantages = returns - values
        advantages = advantages.detach()
        
        # `-log_prob for gradient ASCENT
        # for log_prob, delta in zip(self.policy.actor.saved_log_probs, advantages):
        #     policy_loss.append(-log_prob * delta)
        # policy_loss = torch.cat(policy_loss).mean()
        
        # Reset the gradients of the parameters
        self.optimizer.zero_grad()
        
        # Actor loss
        log_probs = torch.cat(self.policy.actor.saved_log_probs)
        policy_loss = -(log_probs * advantages).mean()
        
        # Critic loss
        value_loss = F.smooth_l1_loss(returns, values).sum()
        # value_loss = (returns - values).pow(2).mean()
        
        
        # Backpropagate the loss through the network
        policy_loss.backward()
        value_loss.backward()
        
        # Perform a parameter update step
        self.optimizer.step()
    
        # Reset the saved rewards and log probabilities
        del self.rewards[:]
        del self.values[:]
        del self.policy.actor.saved_log_probs[:]
    
    
    def train(self, save=False):
        """
        Implementation of the main for training the A2C agent.
        
        Args:
            save (bool): Whether to save the trained network. Defaults to False.
            
        Returns:
            ep_rewards (list): List of actual cumulative rewards in each episode. 
            running_rewards (numpy array): List of smoothed cumulative rewards in each episode. 
        """
        
        # state_size = self.env.observation_space.shape[0]
    
        # Set the training mode
        self.actor.train()
    
        # Lists to store the episodic and running rewards for plotting
        ep_rewards = list()
        running_rewards = list()
    
        # Track training
        print('Training...')
        # Start executing an episode
        for i_episode in range(1, self.num_episodes+1):
            # Reset the environment
            state = self.env.reset()
            t = 0
            
            # Initialize `ep_reward` (the total reward for this episode)
            ep_reward = 0
            
            # 3. For each step of the episode
            while (True):
                # Convert the state from a numpy array to a torch tensor
                state = torch.from_numpy(state).float().unsqueeze(0)
                
                # Select an action using the policy network
                action, value = self.policy(state)
                # action = self.actor.select_action(state)
                # value = self.critic(state)
                
                # Perform the action and note the next state and reward and if the episode is done
                state, reward, done, info = self.env.step(action)
                
                # Store the current reward in `policy.rewards`
                self.rewards.append(reward)
            
                # Store the current value in `policy.values'
                self.values.append(value)
                
                # Increment the total reward in this episode
                ep_reward += reward
                
                # Steps in episode for output required
                t += 1
                
                if t>=self.num_steps:
                    done = True
                    
                # 3.5 Check if the episode is finished using the `done` variable and break if yes
                if (done):
                    break
    
            # Update the running reward
            if i_episode==1:
                running_reward = ep_reward
            else:
                running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
            
            # Store the rewards for plotting
            ep_rewards.append(ep_reward)
            running_rewards.append(running_reward)
            
            # Perform the parameter update according to REINFORCE
            self.update_policy()
    
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
                self.policy.apply(Base_Agent.init_weights)
        
        # Save the trained policy network
        if save:
            self.policy.save()
    
        return ep_rewards, running_rewards
            
    
class TD_A2C_Agent(Base_Agent):
    """
    A TD(0) advantage actor-critic policy agent (TD(0)-A2C)
    """

    def __init__(self, env, num_episodes, num_steps, learning_rate, gamma, hidden_dim=100, dropout=0.6, log_interval=100):
        """
        Constructor
        """
        super().__init__(env, num_episodes, num_steps, learning_rate, gamma, log_interval=log_interval)
        
        ### Networks
        self.actor = PolicyNetwork(self.env, hidden_dim=hidden_dim, dropout=dropout)
        # self.critic = CriticNetwork(env, hidden_dim=hidden_dim, dropout=dropout)
        self.critic = CriticNetwork(env, hidden_dim=hidden_dim, dropout=dropout)
        
        self.policy = ActorCriticNetworks(self.actor, self.critic)
        self.policy.apply(Base_Agent.init_weights)
        
        ### Optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        # self.optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        
        # all_params = chain(self.actor.parameters(), self.critic.parameters())
        # self.optimizer = optim.Adam(all_params, lr=self.learning_rate)
        
        ### Learning rate decay
        # self.lr_decayRate = 0.999
        # self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=self.lr_decayRate)
    
    
    def update_policy(self):
        """
        Performs the parameter updates of the policy network after an episode is completed.
        
        """
        
        sprime = torch.from_numpy(self.sprime).float().unsqueeze(0)
        _, sprime_value = self.policy(sprime)
        
        # Calculate target based on reward and value of next step
        if self.done:
            target = self.reward +self.gamma*sprime_value - self.gamma*sprime_value
        else:
            target = self.reward +self.gamma*sprime_value
        
        # Advantage function delta
        # value = torch.cat(self.value).squeeze(-1)
        # value = torch.tensor(self.values)
        value = self.value
        advantage = target - value
        advantage = advantage.detach()
        
        # Actor loss
        log_prob = self.policy.actor.saved_log_probs[0][0]
        policy_loss = -self.l * advantage * log_prob
        
        # Critic loss
        value_loss = F.smooth_l1_loss(target, value)
        
        # Reset the gradients of the parameters
        self.optimizer.zero_grad()
        
        # Backpropagate the loss through the network
        policy_loss.backward()
        value_loss.backward()
        
        # Perform a parameter update step
        self.optimizer.step()
    
        # Reset the saved rewards and log probabilities
        del self.policy.actor.saved_log_probs[:]
    
    
    def train(self, save=False):
        """
        Implementation of the main body of the TD(0)-A2C algorithm. Update
        happens every step!
        
        Args:
            save (bool): Whether to save the trained network. Defaults to False.
            
        Returns:
            ep_rewards (list): List of actual cumulative rewards in each episode. 
            running_rewards (numpy array): List of smoothed cumulative rewards in each episode. 
        """
        
        # state_size = self.env.observation_space.shape[0]
    
        # Set the training mode
        self.actor.train()
    
        # Lists to store the episodic and running rewards for plotting
        ep_rewards = list()
        running_rewards = list()
    
        # Track training
        print('Training...')
        # Start executing an episode
        for i_episode in range(1, self.num_episodes+1):
            # Reset the environment
            state = self.env.reset()
            t = 0
            
            # Initialize `ep_reward` (the total reward for this episode)
            ep_reward = 0
            
            self.l=1
            
            # 3. For each step of the episode
            while (True):
                # Convert the state from a numpy array to a torch tensor
                state = torch.from_numpy(state).float().unsqueeze(0)
                
                # Select an action using the policy network
                action, self.value = self.policy(state)
                # action = self.actor.select_action(state)
                # value = self.critic(state)
                
                # Perform the action and note the next state and reward and if the episode is done
                self.sprime, self.reward, self.done, info = self.env.step(action)
                        
                self.update_policy()
                
                state = self.sprime 
                self.l = self.gamma*self.l
                
                # Increment the total reward in this episode
                ep_reward += self.reward
                
                # Steps in episode for output required
                t += 1
                
                if t>=self.num_steps:
                    done=True
                    
                # 3.5 Check if the episode is finished using the `done` variable and break if yes
                if (self.done):
                    break
    
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
                self.policy.apply(Base_Agent.init_weights)
        
        # Save the trained policy network
        if save:
            self.policy.save()
    
        return ep_rewards, running_rewards
            
            