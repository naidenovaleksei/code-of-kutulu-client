import numpy as np
import torch
import torch.nn.functional as F

from src.envs.agents.nn_agent import(
    GAMMA,
    LEARNING_RATE,
)
from src.envs.agents.actor_agent import (
    ActorAgent,
)
from src.envs.agents.dqn_agent_conv import (
    DQNStateEncoderConv,
)
from src.envs.kutulu_observer import (
    KutuluConvObserver,
)
from src.envs.models.conv_a2c_model import ConvA2CModel


class A2CAgent(ActorAgent):
    def __init__(self, state_type, action_space_n,
                 lr=LEARNING_RATE,
                 gamma=GAMMA, model_params={},
                 train=False, verbose=False, 
                 entropy_coef=0.01, value_loss_coef=0.5, n_step=10, batch_size=10):
        """
        A2C (Advantage Actor-Critic) Agent

        Args:
            state_type: Type of state representation ('conv' for convolutional)
            action_space_n: Number of possible actions
            epsilon_params: Parameters for epsilon-greedy exploration
            lr: Learning rate
            gamma: Discount factor
            model_params: Parameters for the model
            train: Whether the agent is in training mode
            verbose: Whether to print training information
            entropy_coef: Coefficient for entropy regularization
            value_loss_coef: Coefficient for value loss
        """
        assert state_type == 'conv', "A2C agent only supports 'conv' state type"

        self.size = model_params.pop('size', 3)
        model = ConvA2CModel(
            num_classes=action_space_n,
            size=self.size,
            **model_params
        )

        super(A2CAgent, self).__init__(
            state_type=state_type,
            action_space_n=action_space_n,
            lr=lr,
            gamma=gamma,
            train=train,
            verbose=verbose,
            model=model,
            state_encoder=DQNStateEncoderConv(),
        )
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.n_step = n_step
        self.batch_size = batch_size

    def set_env(self, env):
        assert self.state_type == 'conv'
        self.observer = KutuluConvObserver(env, self.size)

    
    def _train_model(self):
        """Train the A2C model on the collected episode"""
        # End the current episode and get the episode data
        states, actions, rewards = self.episode_buffer.end_episode()

        if len(states) == 0:
            return
    
        self.model.train()
        self.optimizer.zero_grad()

        # Encode states
        states_tensor = self.episode_buffer.encode_states(states)
        actions_tensor = torch.tensor(actions)

        # Forward pass to get policy and values
        policy, values = self.model(states_tensor)
        values = values.view(-1)

        # Get log probabilities of actions
        log_probs = torch.log(policy + 1e-8)
        selected_log_probs = log_probs[range(len(actions)), actions_tensor]

        # Calculate entropy for regularization
        entropy = -(log_probs * policy).sum(dim=1).mean()

        # Calculate returns and advantages
        returns, advantages = self._calculate_returns_and_advantages(rewards, values.detach())

        # Calculate policy loss (actor)
        policy_loss = -(selected_log_probs * advantages.detach()).mean()

        # Calculate value loss (critic)
        value_loss = F.mse_loss(values, returns)

        # Total loss with entropy regularization
        loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

        # Backpropagate and update
        loss.backward()
        self.optimizer.step()

        self.last_loss = loss.item()

        if self.verbose:
            print(f"Episode {self.episode_idx}, Loss: {loss.item():.4f}, "
                  f"Policy Loss: {policy_loss.item():.4f}, Value Loss: {value_loss.item():.4f}, "
                  f"Entropy: {entropy.item():.4f}, Return: {np.sum(rewards):.4f}")

    def _calculate_returns_and_advantages(self, rewards, values):
        returns = []
        T = len(rewards)

        for t in range(T):
            G = 0
            for k in range(self.n_step):
                if t + k < T:
                    G += (self.gamma ** k) * rewards[t + k]
            # Add bootstrapped value estimate if within bounds
            if t + self.n_step < T:
                G += (self.gamma ** self.n_step) * values[t + self.n_step].item()
            returns.append(G)

        # Convert to tensor
        returns = torch.tensor(returns)

        # Calculate advantages
        advantages = returns - values

        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return returns, advantages
