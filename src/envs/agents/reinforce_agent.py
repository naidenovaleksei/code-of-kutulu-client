import numpy as np
import torch

from src.envs.agents.nn_agent import(
    GAMMA,
    LEARNING_RATE,
)
from src.envs.agents.actor_agent import (
    ActorAgent,
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


class REINFORCEAgent(ActorAgent):
    def __init__(self, state_type, action_space_n,
                 lr=LEARNING_RATE,
                 gamma=GAMMA, model_params=None,
                 train=False, verbose=False, entropy_coeff=0, n_step=10):
        if model_params is None:
            model_params = {}
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
            train=train,
            verbose=verbose,
            model=model,
            state_encoder=state_encoder,
        )
        self.entropy_coeff = entropy_coeff
        self.n_step = n_step
    
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

        if self.entropy_coeff > 0:
            entropy = -(log_probs * log_probs.exp()).sum(dim=1)  # энтропия для каждого шага
            loss = -(selected_log_probs * returns).mean() - self.entropy_coeff * entropy.mean()
        else:
            # Calculate policy loss (negative because we want to maximize expected return)
            loss = -(selected_log_probs * returns).mean()

        # Backpropagate and update
        loss.backward()
        self.optimizer.step()

        self.last_loss = loss.item()

        if self.verbose:
            print(f"Episode {self.episode_idx}, Loss: {loss.item():.4f}, Return: {np.sum(rewards):.4f}")

    def _calculate_returns(self, rewards):
        """Calculate n-step discounted returns for all steps in an episode"""
        returns = []
        T = len(rewards)

        for t in range(T):
            G = 0
            for k in range(self.n_step):
                if t + k < T:
                    G += (self.gamma ** k) * rewards[t + k]
            returns.append(G)

        # Normalize returns for stability
        returns = torch.tensor(returns)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        return returns
