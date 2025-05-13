import collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from src.envs.agents import BaseAgent
from src.game.template import (
    getActionGreedyMasked2,
    parse_state,
    ENTITY_TOKENS,
)
from src.envs.models.reinforce_model import REINFORCEModel

# Hyperparameters
GAMMA = 0.99
LEARNING_RATE = 1e-4
EPSILON_START = 1.0
EPSILON_FINAL = 0.02
EPSILON_DECAY_LAST_FRAME = 10**5

Episode = collections.namedtuple('Episode', field_names=['states', 'actions', 'rewards'])


class EpisodeBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.current_episode = None
        
    def start_episode(self):
        self.states = []
        self.actions = []
        self.rewards = []
        
    def add_step(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        
    def end_episode(self):
        if not self.states:
            return None
        
        self.current_episode = Episode(
            states=self.states.copy(),
            actions=self.actions.copy(),
            rewards=self.rewards.copy()
        )
        self.states = []
        self.actions = []
        self.rewards = []
        
        return self.current_episode
    
    def encode_states(self, states, return_tensors=True):
        data = dict(
            entity_kind=[],
            entity_features=[],
            entity_dir=[],
        )
        for state in states:
            kind_list, features_list, dir_list = parse_state(state)
            data['entity_kind'].append(kind_list)
            data['entity_features'].append(features_list)
            data['entity_dir'].append(dir_list)
        if return_tensors:
            data = {k: torch.tensor(v) for k,v in data.items()}
        return data


class REINFORCEAgent(BaseAgent):
    def __init__(self, state_type, action_space_n,
                 lr=LEARNING_RATE,
                 epsilon_start=EPSILON_START, epsilon_final=EPSILON_FINAL,
                 epsilon_decay_last_frame=EPSILON_DECAY_LAST_FRAME,
                 gamma=GAMMA,
                 train=False, verbose=False):
        self.state_type = state_type
        self.action_space_n = action_space_n
        self.eps = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_start = epsilon_start
        self.epsilon_decay_last_frame = epsilon_decay_last_frame
        self.gamma = gamma
        self.train = train
        self.verbose = verbose
        self.state_actions = None
        self.frame_idx = 0
        self.episode_idx = 0
        self.episode_buffer = EpisodeBuffer()
        
        # Initialize the policy network
        self.model = REINFORCEModel(
            vocab_size=len(ENTITY_TOKENS) + 1,
            features_dim=7,
            embed_dim=32,
            hidden_dim=32,
            inner_dim=16,
            num_classes=action_space_n
        )
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.last_loss = np.inf
        
        # Start a new episode
        self.episode_buffer.start_episode()
    
    def _calculate_returns(self, rewards):
        """Calculate discounted returns for all steps in an episode"""
        returns = []
        G = 0
        
        # Calculate returns from the end of the episode
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
            
        # Normalize returns for stability
        returns = torch.tensor(returns)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)
            
        return returns
    
    def generate_state_and_step(self, observer, player_id):
        self.frame_idx += 1
        self.eps = max(
            self.epsilon_final, self.epsilon_start - self.frame_idx / self.epsilon_decay_last_frame)
        
        state = observer.get_state(player_id, self.state_type)
        valid_actions = observer.env.get_valid_action_mask()[player_id]
        player_mask = ~np.array(valid_actions)
        
        # During training, sometimes choose random action for exploration
        if self.train and np.random.random() < self.eps:
            action = getActionGreedyMasked2(state, {}, self.action_space_n, player_mask)
        else:
            # Use policy network to get action probabilities
            data = self.episode_buffer.encode_states([state])
            action_probs = self.model(data)[0].detach().cpu().numpy()
            
            # Apply mask to invalid actions
            for i in range(len(action_probs)):
                if player_mask[i]:
                    action_probs[i] = 0
                    
            # Renormalize probabilities
            if np.sum(action_probs) > 0:
                action_probs = action_probs / np.sum(action_probs)
            else:
                # If all actions are masked, choose randomly from valid actions
                action = getActionGreedyMasked2(state, {}, self.action_space_n, player_mask)
                self.state_actions = (state, action)
                return state, action
            
            # Sample action from probability distribution
            action = np.random.choice(self.action_space_n, p=action_probs)
        
        self.state_actions = (state, action)
        return state, action
    
    def train_step(self, reward, game_over, new_state=None):
        if not self.train:
            return
        
        state, action = self.state_actions
        # Add this step to our episode
        self.episode_buffer.add_step(state, action, reward / 100.0)
        
        # print(f"game_over={game_over} or reward={reward}")
        # If the episode has ended, train on it
        if game_over or reward is None:
            self.train_model()
            # Start a new episode
            self.episode_buffer.start_episode()
            self.episode_idx += 1
    
    def check_policy(self):
        return self.last_loss
    
    def train_model(self):
        # print("train_model")
        # End the current episode and get the episode data
        episode = self.episode_buffer.end_episode()
        if episode is None:
            return
        
        # Prepare for training
        self.model.train()
        self.optimizer.zero_grad()
        
        # Encode all states from this episode
        states_data = self.episode_buffer.encode_states(episode.states)
        
        # Get log probabilities for all actions
        log_probs = self.model.get_log_probs(states_data)
        
        # Create tensor of selected actions
        actions_tensor = torch.tensor(episode.actions)
        
        # Get log probability of each taken action
        selected_log_probs = log_probs[range(len(actions_tensor)), actions_tensor]
        
        # Calculate returns
        returns = self._calculate_returns(episode.rewards)
        
        # Calculate policy loss (negative because we want to maximize expected return)
        loss = -(selected_log_probs * returns).mean()
        
        # Backpropagate and update
        loss.backward()
        self.optimizer.step()
        
        self.last_loss = loss.item()
        
        if self.verbose:
            print(f"Episode {self.episode_idx}, Loss: {loss.item():.4f}, Return: {np.sum(episode.rewards):.4f}")