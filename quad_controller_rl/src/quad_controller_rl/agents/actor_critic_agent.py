"""AC Policy Search ."""

import numpy as np

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.layers.merge import Add, Multiply
from keras.optimizers import Adam
import keras.backend as K

import tensorflow as tf

import random
from collections import deque

from quad_controller_rl.agents.base_agent import BaseAgent
from .OUNoise import OUNoise
from .critic import Critic
from .actor import Actor
from .replay_buffer import ReplayBuffer

class ActorCriticAgent(BaseAgent):
    """Agent that uses Actor Critic method to search for optimal policy."""

    def __init__(self, task):
        # Task (environment) information
        self.task = task  # should contain observation_space and action_space
        self.state_size = np.prod(self.task.observation_space.shape)
        self.state_range = self.task.observation_space.high - self.task.observation_space.low
        self.action_size = np.prod(self.task.action_space.shape)
        self.action_range = self.task.action_space.high - self.task.action_space.low
        self.action_low = self.task.action_space.low
        self.action_high = self.task.action_space.high
        
        self.batch_size = 64
        
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = .995
        self.gamma = .95
        self.tau   = .125
        
        self.buffer_size = 200000
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)         
        
        # Noise process
        self.exploration_mu = 0
        self.exploration_theta = 0.15
        self.exploration_sigma = 0.2
        self.noise = OUNoise(self.action_size, self.exploration_mu,
                             self.exploration_theta, self.exploration_sigma)
        
        #setup the Actor network
        self.actor_model = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        self.target_actor_model = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        

        #set up the critic network
        self.critic_model = Critic(self.state_size, self.action_size)
        self.target_critic_model = Critic(self.state_size, self.action_size)
        
        self.target_critic_model.model.set_weights(self.critic_model.model.get_weights())
        self.target_actor_model.model.set_weights(self.actor_model.model.get_weights())

        # Score tracker
        self.best_reward = -np.inf 
            
        # Episode variables
        self.reset_episode_vars()
        
    def train(self, experiences):
        #Reshape data so that it is stacked to feed into our network.
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(
            np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(
            np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(
            np.uint8).reshape(-1, 1)
        next_states = np.vstack(
            [e.next_state for e in experiences if e is not None])
        
        
        # Get predicted next-state actions and Q values from target models
        #     Q_targets_next = critic_target(next_state, actor_target(next_state))
        actions_next = self.target_actor_model.model.predict_on_batch(next_states)
        Q_targets_next = self.target_critic_model.model.predict_on_batch(
            [next_states, actions_next])
        
        # Compute Q targets for current states and train critic model (local)
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        self.critic_model.model.train_on_batch(x=[states, actions], y=Q_targets)
        
        # Train actor model (local)
        action_gradients = np.reshape(self.critic_model.get_action_gradients(
            [states, actions, 0]), (-1, self.action_size))
        # custom training function
        self.actor_model.train_fn([states, action_gradients, 1])
        
        # Soft-update target models
        self.soft_update(self.critic_model.model, self.target_critic_model.model)
        self.soft_update(self.actor_model.model, self.target_actor_model.model)
        
        self.reset_episode_vars()
    
    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())
        assert len(local_weights) == len(target_weights), "Local and target model parameters must have the same size"
        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)
                                                                                                  
                                             
    def reset_episode_vars(self):
        self.total_reward = 0.0
        self.noise.reset()
        state = self.task.reset()
        self.last_state = None

    def step(self, state, reward, done):
        
        # Choose an action
        action = self.act(state)
        
        if self.last_state is not None:
            self.memory.add(self.last_state, action, reward, state, done)    
        
        # Learn, if at end of episode
        if done:
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.train(experiences)
            self.reset_episode_vars()
        
        self.last_state = state
        return action
                                             
    def act(self, state):
        # Choose action based on given state and policy
        state = np.reshape(state, [-1, self.state_size])
        action = self.actor_model.model.predict(state)[0]
        # add some noise 
        return np.array(list(action + self.noise.sample()))
