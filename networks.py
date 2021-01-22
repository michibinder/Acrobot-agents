#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 16:10:50 2020

@author: tennismichel
"""

import os
# import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """
    A fully connected neural network with 1 hidden layer.
    """
    def __init__(self, env, hidden_dim=100):
        super(QNetwork, self).__init__()
        self.env = env
        self.state_space = env.observation_space.shape[0]
        self.action_space = env.action_space.n
        self.hidden_dim = hidden_dim # smaller makes more stable
        
        # Layers
        self.l1 = nn.Linear(self.state_space, self.hidden_dim, bias=False)
        self.l2 = nn.Linear(self.hidden_dim, self.action_space, bias=False)

    
    def forward(self, x):    
        model = torch.nn.Sequential(
            self.l1,
            nn.ReLU(),
            self.l2
        )
        return model(x)

    def save(self, save_dir="models", file_name="q_network.pt"):
        # Save the model state
        if not (os.path.exists(save_dir) and os.path.isdir(save_dir)):
            os.makedirs(save_dir)
        torch.save(self.state_dict(), os.path.join(save_dir, file_name))

    @staticmethod
    def load(env, save_dir="models", file_name="q_network.pt"):
        # Create a network object with the constructor parameters
        network = QNetwork(env)
        # Load the weights
        network.load_state_dict(torch.load(os.path.join(save_dir, file_name)))
        # Set the network to evaluation mode
        network.eval()
        return network
    

class QNetwork2(nn.Module):
    """
    A fully connected neural network with 1 hidden layer.
    """
    def __init__(self, env, hidden_dim=100, dropout=0.4):
        super(QNetwork2, self).__init__()
        self.env = env
        self.state_space = env.observation_space.shape[0] # 6
        self.action_space = env.action_space.n # 3
        self.hidden_dim = hidden_dim
        
        # Layer definitions
        self.affine1 = nn.Linear(self.state_space, self.hidden_dim)
        self.dropout1 = nn.Dropout(p=dropout)        
        self.affine2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.dropout2 = nn.Dropout(p=dropout)
        self.affine3 = nn.Linear(self.hidden_dim, self.action_space)
        
        
    def forward(self, x):
        """
        Defines the forward pass of the policy network.
        
        Args:
            x (Tensor): The current state as observed by the agent.
        Returns:
            (Tensor): Action probabilities.
        """        
        model = torch.nn.Sequential(
            self.affine1,
            self.dropout1,
            nn.ReLU(),
            self.affine2,
            self.dropout2,
            nn.ReLU(),
            self.affine3
        )
        
        out = model(x)
        # out = F.softmax(numerical_prefs, dim=1)
        return out
    

    def save(self, save_dir="models", file_name="q_network.pt"):
        # Save the model state
        if not (os.path.exists(save_dir) and os.path.isdir(save_dir)):
            os.makedirs(save_dir)
        torch.save(self.state_dict(), os.path.join(save_dir, file_name))

    @staticmethod
    def load(env, save_dir="models", file_name="q_network.pt"):
        # Create a network object with the constructor parameters
        network = QNetwork(env)
        # Load the weights
        network.load_state_dict(torch.load(os.path.join(save_dir, file_name)))
        # Set the network to evaluation mode
        network.eval()
        return network

    
    
class PolicyNetwork(nn.Module):
    """
    Neural network policy.
    """
    def __init__(self, env, hidden_dim=100, dropout=0.6):
        super(PolicyNetwork, self).__init__()
        self.env = env
        self.state_space = env.observation_space.shape[0] # 6
        self.action_space = env.action_space.n # 3
        self.hidden_dim = hidden_dim
        
        # Layer definitions
        self.affine1 = nn.Linear(self.state_space, self.hidden_dim)
        self.dropout1 = nn.Dropout(p=dropout)        
        self.affine2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.dropout2 = nn.Dropout(p=dropout)
        self.affine3 = nn.Linear(self.hidden_dim, self.action_space)

        # Used for storing the log probabilities of the actions
        # which is required to compute the loss (and hence needed for the parameter update step)
        self.saved_log_probs = []
        
        # Used for tracking the rewards the agent recieves in an episode.
        self.rewards = []

    def forward(self, x):
        """
        Defines the forward pass of the policy network.
        
        Args:
            x (Tensor): The current state as observed by the agent.
        Returns:
            (Tensor): Action probabilities.
        """        
        model = torch.nn.Sequential(
            self.affine1,
            self.dropout1,
            nn.ReLU(),
            self.affine2,
            self.dropout2,
            nn.ReLU(),
            self.affine3
        )
        
        numerical_prefs = model(x)
        return F.softmax(numerical_prefs, dim=1)
    
    def select_action(self,state):
        """
        Selects an action for the agent, by sampling from the action probabilities
        produced by the network, based on the current state. Also stores the 
        log probability of the actions.
        
        Args:
            state (numpy array): The current state as observed by the agent.
            
        Returns:
            (int): Action to perform.
        """
        # Convert the state from a numpy array to a torch tensor
        # state = torch.from_numpy(state).float().unsqueeze(0)
        
        # Get the predicted probabilities from the policy network
        probs = self.forward(state)
        
        # Sample the actions according to their respective probabilities
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        
        # Also calculate the log of the probability for the selected action
        self.saved_log_probs.append(m.log_prob(action))
        
        # Return the chosen action
        return action.item()

    def save(self, save_dir="models", file_name="network.pt"):
        """
        Saves a trained policy network.
        """
        # Save the model state
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        torch.save(self.state_dict(), os.path.join(save_dir, file_name))
    

class CriticNetwork(nn.Module):
    """
    Network for Actor-Critic value function approximation
    --> Represents the critic part
    """
    def __init__(self, env, hidden_dim=100, dropout=0.6):
        super().__init__()
        
        self.state_space = env.observation_space.shape[0] # 6
        self.output_dim=1
        self.affine1 = nn.Linear(self.state_space, hidden_dim)
        self.affine2 = nn.Linear(hidden_dim, hidden_dim)
        self.affine3 = nn.Linear(hidden_dim, self.output_dim)
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x):
        model1 = torch.nn.Sequential(
            self.affine1,
            self.dropout,
            nn.ReLU(),
            self.affine3
        )
        model2 = torch.nn.Sequential(
            self.affine1,
            self.dropout,
            nn.ReLU(),
            self.affine2,
            self.dropout2,
            nn.ReLU(),
            self.affine3
        )
        # Convert the state from a numpy array to a torch tensor
        # x = torch.from_numpy(x).float().unsqueeze(0)
        
        x = model1(x)
        return x
    
    
class ActorCriticNetworks(nn.Module):
    def __init__(self, actor, critic):
        super().__init__()
        
        self.actor = actor
        self.critic = critic
        
    def forward(self, state):
        
        action = self.actor.select_action(state)
        value = self.critic(state)
        
        return action, value