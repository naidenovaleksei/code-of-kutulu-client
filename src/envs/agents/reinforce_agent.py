import collections
import numpy as np
import torch

from src.envs.agents.nn_agent import(
    NNAgent,
    GAMMA,
    LEARNING_RATE,
    EPSILON_START,
    EPSILON_FINAL,
    EPSILON_DECAY_LAST_FRAME,
)
from src.envs.buffers import (
    Experience,
)
from src.envs.agents.dqn_agent_ext import (
    DQNStateEncoderExt,
)
from src.envs.agents.dqn_agent_conv import (
    DQNStateEncoderConv,
)
from src.game.template import (
    ENTITY_TOKENS,
)
from src.envs.models.ext_state_model import ExtStateModel
from src.envs.models.conv_state_model import ConvStateModel


class EpisodeBuffer:
    def __init__(self, state_encoder):
        self.buffer = []
        self.state_encoder = state_encoder
        
    def start_episode(self):
        self.buffer = []

    def append(self, experience):
        self.buffer.append(experience)
        
    def end_episode(self):
        assert len(self.buffer) > 0
        states, actions, rewards, _, _, _ = zip(*self.buffer)
        return states, actions, rewards

    def encode_states(self, states):
        return self.state_encoder.encode_states(states)


class REINFORCEAgent(NNAgent):
    def __init__(self, state_type, action_space_n,
                 epsilon_params,
                 lr=LEARNING_RATE,
                 gamma=GAMMA, model_params={},
                 train=False, verbose=False):
        if state_type == 'closest_ext':
            model = ExtStateModel(
                num_classes=action_space_n,
                vocab_size=len(ENTITY_TOKENS) + 1,
                return_softmax=True,
                **model_params
            )
            state_encoder = DQNStateEncoderExt()
        elif state_type == 'conv':
            self.size = model_params.pop('size')
            model = ConvStateModel(
                num_classes=action_space_n,
                size=self.size,
                return_softmax=True,
                **model_params)
            state_encoder = DQNStateEncoderConv()
        else:
            ValueError('unknown state_type for reinforce agent: {self.state_type}')

        super(REINFORCEAgent, self).__init__(
            state_type=state_type,
            action_space_n=action_space_n,
            lr=lr,
            gamma=gamma,
            epsilon_start=epsilon_params.get('start', EPSILON_START),
            epsilon_final=epsilon_params.get('final', EPSILON_FINAL),
            epsilon_decay_last_frame=epsilon_params.get('decay', EPSILON_DECAY_LAST_FRAME),
            epsilon_reset=epsilon_params.get('reset'),
            epsilon_reset_coef=epsilon_params.get('reset_coef'),
            train=train,
            verbose=verbose,
            model=model,
            episode_buffer=EpisodeBuffer(state_encoder),
        )
        self.episode_idx = 0
        self.episode_buffer.start_episode()

    def generate_random_step(self, actions_masked, player_mask):
        ps = actions_masked.filled(0)
        if ps.sum() == 0:
            return np.random.randint(self.action_space_n)
        return np.random.choice(np.arange(self.action_space_n), p=ps / ps.sum())

    def train_step(self, reward, game_over, new_state):
        if not self.train:
            return
        assert reward is not None
        
        state, action, observation = self.state_actions
        exp = Experience(state, action, reward, game_over, None, observation)
        self.episode_buffer.append(exp)

        # If the episode has ended, train on it
        if game_over or reward is None:
            self._train_model()
            # Start a new episode
            self.episode_buffer.start_episode()
            self.episode_idx += 1
    
    def _train_model(self):
        # End the current episode and get the episode data
        states, actions, rewards = self.episode_buffer.end_episode()

        self.model.train()
        self.optimizer.zero_grad()
        
        states = self.episode_buffer.encode_states(states)
        actions = torch.tensor(actions)
        # Get log probabilities for all actions
        log_probs = self.model.get_log_probs(states)
        
        # Get log probability of each taken action
        selected_log_probs = log_probs[range(actions.shape[0]), actions]
        # Calculate returns
        returns = self._calculate_returns(rewards)
        
        # Calculate policy loss (negative because we want to maximize expected return)
        loss = -(selected_log_probs * returns).mean()
        
        # Backpropagate and update
        loss.backward()
        self.optimizer.step()
        
        self.last_loss = loss.item()
        
        if self.verbose:
            print(f"Episode {self.episode_idx}, Loss: {loss.item():.4f}, Return: {np.sum(rewards):.4f}")

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
