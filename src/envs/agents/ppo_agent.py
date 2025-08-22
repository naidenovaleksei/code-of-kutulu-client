from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from src.game.template import (
    EXTENDED_KUTULU_ACTIONS,
)
from src.envs.agents.nn_agent import(
    GAMMA,
    LEARNING_RATE,
)
from src.envs.agents.actor_agent import (
    ActorAgent,
    BaseStateEncoder,
    Experience,
)
from src.envs.agents.dqn_agent_conv import (
    DQNStateEncoderConv,
)
from src.envs.kutulu_observer import (
    KutuluConvObserver,
)
from src.envs.models.conv_a2c_model import ConvA2CModel
from src.envs.reward_shaper import PotentialRewardShaper, RewardShaper


class PPOBuffer:
    """Enhanced buffer for PPO that stores additional information needed for clipped objective"""
    
    def __init__(self, state_encoder: BaseStateEncoder, actions,
                 reward_params=None, need_aug=False, use_potential=True, verbose=False):
        self.buffer = []
        self.state_encoder = state_encoder
        self.need_aug = need_aug
        self.latest_bonus_averages = None
        if reward_params is None:
            reward_params = {}
        if use_potential:
            try:
                self.reward_shaper = PotentialRewardShaper(actions, verbose, **reward_params)
            except Exception:
                self.reward_shaper = RewardShaper(actions, verbose, **reward_params)
        else:
            self.reward_shaper = RewardShaper(actions, verbose, **reward_params)
        
    def start_episode(self):
        self.buffer = []

    def append(self, experience, log_prob, value):
        """Append experience with additional PPO-specific data"""
        # Store the experience along with log probability and value
        ppo_experience = {
            'experience': experience,
            'log_prob': log_prob,
            'value': value
        }
        self.buffer.append(ppo_experience)
        
    def end_episode(self):
        """End episode and return all collected data"""
        assert len(self.buffer) > 0
        
        if self.need_aug:
            # Apply augmentation to experiences
            augmented_data = [self.rotation_augment_ppo(item) for item in self.buffer]
        else:
            augmented_data = self.buffer
            
        # Extract components
        experiences = [item['experience'] for item in augmented_data]
        log_probs = [item['log_prob'] for item in augmented_data]
        values = [item['value'] for item in augmented_data]
        
        states, actions, rewards, dones, other_rewards, observations = zip(*experiences)
        
        # Get recalculated rewards and bonus averages
        result = self.reward_shaper.recalculate_rewards(
            rewards, actions, states, dones, other_rewards, observations
        )
        
        # Check if the result is a tuple (rewards, avg_bonuses) or just rewards
        if isinstance(result, tuple) and len(result) == 2:
            rewards, avg_bonuses = result
            # Store the average bonuses in the buffer
            self.latest_bonus_averages = avg_bonuses
        else:
            rewards = result
        
        return states, actions, rewards, dones, log_probs, values

    def encode_states(self, states):
        return self.state_encoder.encode_states(states)
    
    def rotation_augment_ppo(self, ppo_item):
        """Apply rotation augmentation to PPO experience"""
        clockwise_dir = np.random.randint(0, 4)
        exp = ppo_item['experience']
        
        augmented_exp = Experience(
            self.state_encoder.state_rotation_augment(exp.state, clockwise_dir),
            self.state_encoder.action_rotation_augment(exp.action, clockwise_dir),
            exp.reward,
            exp.done,
            exp.new_state,
            exp.observation,
        )

        return {
            'experience': augmented_exp,
            'log_prob': ppo_item['log_prob'],
            'value': ppo_item['value']
        }


class PPOAgent(ActorAgent):
    def __init__(self, state_type, actions=EXTENDED_KUTULU_ACTIONS,
                 lr=LEARNING_RATE,
                 gamma=GAMMA, model_params={},
                 train=False, verbose=False, 
                 entropy_coef=0.01, value_loss_coef=0.5, 
                 clip_ratio=0.2, ppo_epochs=4, mini_batch_size=64,
                 target_kl=0.01, max_grad_norm=0.5,
                 gae_lambda=0.95, reward_params=None, need_aug=False, **kw):
        """
        PPO (Proximal Policy Optimization) Agent with Clipped Objective

        Args:
            state_type: Type of state representation ('conv' for convolutional)
            action_space_n: Number of possible actions
            lr: Learning rate
            gamma: Discount factor
            model_params: Parameters for the model
            train: Whether the agent is in training mode
            verbose: Whether to print training information
            entropy_coef: Coefficient for entropy regularization
            value_loss_coef: Coefficient for value loss
            clip_ratio: Clipping parameter for PPO objective (epsilon)
            ppo_epochs: Number of optimization epochs per batch of data
            mini_batch_size: Size of mini-batches for SGD updates
            target_kl: Target KL divergence for early stopping
            max_grad_norm: Maximum gradient norm for clipping
            gae_lambda: Lambda parameter for GAE (Generalized Advantage Estimation)
        """
        assert state_type == 'conv', "PPO agent only supports 'conv' state type"
        action_space_n = len(actions)
        self.size = model_params.pop('size', 3)
        model = ConvA2CModel(
            num_classes=action_space_n,
            size=self.size,
            **model_params
        )
        
        super(PPOAgent, self).__init__(
            state_type=state_type,
            action_space_n=action_space_n,
            lr=lr,
            gamma=gamma,
            train=train,
            verbose=verbose,
            model=model,
            state_encoder=DQNStateEncoderConv(),
            **kw,
        )
        
        self.actions = actions
        if reward_params is None:
            reward_params = {}
        self.reward_params = reward_params
        self.need_aug = need_aug
        # Initialize with PPO-specific buffer
        ppo_buffer = PPOBuffer(DQNStateEncoderConv(), self.actions, need_aug=self.need_aug,
                               verbose=self.verbose, reward_params=self.reward_params)
        # Replace the episode buffer with PPO buffer
        self.episode_buffer = ppo_buffer
        self.episode_buffer.start_episode()
        
        # PPO-specific parameters
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.clip_ratio = clip_ratio
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.target_kl = target_kl
        self.max_grad_norm = max_grad_norm
        self.gae_lambda = gae_lambda
        
        # Initialize latest_bonus_averages to store reward shaper metrics
        self.latest_bonus_averages = None
        
        # Multi-environment support
        self.num_envs = 1
        self.env_buffers = None

    def init_multi_env(self, num_envs):
        """Initialize multi-environment support"""
        self.num_envs = num_envs
        if num_envs > 1:
            # Create separate buffers for each environment
            self.env_buffers = [
                PPOBuffer(DQNStateEncoderConv(), self.actions, need_aug=self.need_aug,
                          verbose=self.verbose, reward_params=self.reward_params)
                for _ in range(num_envs)
            ]
            for buffer in self.env_buffers:
                buffer.start_episode()
            if self.verbose:
                print(f"PPO Agent initialized with {num_envs} environments")

    def set_env(self, env):
        assert self.state_type == 'conv'
        self.observer = KutuluConvObserver(env, self.size)

    def generate_state_and_step(self, player_id, need_update=True):
        output = super().generate_state_and_step(player_id, need_update, True)
        action = output['action']
        policy = output['policy']
        value = output['value']

        # PPO-specific: calculate log probability and store value
        action_dist = torch.distributions.Categorical(policy)
        log_prob = action_dist.log_prob(torch.tensor(action)).item()
        value_estimate = value.item()
        
        # Store for later use in append_observation
        self.current_log_prob = log_prob
        self.current_value = value_estimate

        return output

    def append_observation(self, player_id, reward, game_over, env_idx=None, other_rewards=None):
        """Append observation with PPO-specific data"""
        if not self.train or reward is None:
            return

        state, action, observation = self.state_actions
        assert reward is not None
        exp = Experience(state, action, reward, game_over, other_rewards, observation)

        # Choose the appropriate buffer
        if self.env_buffers is not None:
            # Multi-environment mode
            buffer = self.env_buffers[env_idx]
        else:
            # Single environment mode
            buffer = self.episode_buffer
        
        # Append with log probability and value
        buffer.append(
            exp, 
            self.current_log_prob, 
            self.current_value
        )

    def collect_all_data_from_buffers(self):
        # Collect data from all environment buffers
        all_states, all_actions, all_rewards, all_dones = [], [], [], []
        all_log_probs, all_values = [], []
        
        # Collect bonus averages from all buffers
        bonus_sums = {}
        bonus_counts = {}

        for env_buffer in self.env_buffers:
            if len(env_buffer.buffer) > 0:
                states, actions, rewards, dones, log_probs, values = env_buffer.end_episode()
                all_states.extend(states)
                all_actions.extend(actions)
                all_rewards.extend(rewards)
                all_dones.extend(dones)
                all_log_probs.extend(log_probs)
                all_values.extend(values)
                
                # Collect bonus averages if available
                if env_buffer.latest_bonus_averages is not None:
                    for bonus_type, value in env_buffer.latest_bonus_averages.items():
                        if bonus_type not in bonus_sums:
                            bonus_sums[bonus_type] = 0
                            bonus_counts[bonus_type] = 0
                        bonus_sums[bonus_type] += value
                        bonus_counts[bonus_type] += 1
                
                env_buffer.start_episode()  # Reset for next episode
        
        # Calculate average bonuses across all environments
        if bonus_sums:
            self.latest_bonus_averages = {
                bonus_type: total / max(1, bonus_counts[bonus_type])
                for bonus_type, total in bonus_sums.items()
            }
        return all_states, all_actions, all_rewards, all_dones, all_log_probs, all_values

    def train_step(self):
        if self.train and len(self.episode_buffer.buffer) > 0:
            super().train_step()

    def _train_model(self):
        """Train the PPO model with clipped objective"""
        # End the current episode and get the episode data
        states, actions, rewards, dones, old_log_probs, values = self.episode_buffer.end_episode()
        # Get bonus averages if available
        if self.episode_buffer.latest_bonus_averages is not None:
            self.latest_bonus_averages = self.episode_buffer.latest_bonus_averages
        self._train_model_with_data(states, actions, rewards, dones, old_log_probs, values)
    
    def train_multi_env_step(self):
        """Train using data from all environments"""
        if not self.train or self.env_buffers is None:
            return self.train_step()  # Fall back to single-env
        
        all_data = self.collect_all_data_from_buffers()
        # Use the combined data for training
        self._train_model_with_data(*all_data)
        
        if self.scheduler:
            self.scheduler.step()
    
    def _train_model_with_data(self, states, actions, rewards, dones, old_log_probs, values):
        """Train the PPO model with provided data"""
        if len(states) == 0:
            return
    
        self.model.train()

        # Encode states
        states_tensor = self.episode_buffer.encode_states(states)
        actions_tensor = torch.tensor(actions, dtype=torch.long)
        old_log_probs_tensor = torch.tensor(old_log_probs, dtype=torch.float32)

        # Calculate returns and advantages using GAE
        returns_tensor, advantages_tensor = self._calculate_gae(rewards, values, dones)

        # Normalize advantages
        if len(advantages_tensor) > 1:
            advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

        # PPO training loop
        dataset_size = len(states)
        early_stop = False
        
        for epoch in range(self.ppo_epochs):
            if early_stop:
                break
            # Create mini-batches
            indices = torch.randperm(dataset_size)
            
            for start_idx in range(0, dataset_size, self.mini_batch_size):
                end_idx = min(start_idx + self.mini_batch_size, dataset_size)
                batch_indices = indices[start_idx:end_idx]
                
                # Get mini-batch data
                batch_states = states_tensor[batch_indices]
                batch_actions = actions_tensor[batch_indices]
                batch_old_log_probs = old_log_probs_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                
                # Forward pass
                policy, current_values = self.model(batch_states)
                current_values = current_values.view(-1)
                
                # Get current log probabilities
                log_probs = torch.log(policy + 1e-8)
                current_log_probs = log_probs[range(len(batch_actions)), batch_actions]
                
                # Calculate probability ratio
                ratio = torch.exp(current_log_probs - batch_old_log_probs)
                
                # Calculate surrogate losses
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                
                # PPO clipped objective (we want to maximize, so minimize negative)
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(current_values, batch_returns)
                
                # Entropy for exploration
                entropy = -(log_probs * policy).sum(dim=1).mean()
                
                # Total loss
                loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
                
                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                self.optimizer.step()
                
                # Calculate KL divergence for early stopping
                with torch.no_grad():
                    kl_div = (batch_old_log_probs - current_log_probs).mean()
                
                self.metrics_aggregator.add_metrics({
                    'policy_loss': policy_loss.item(),
                    'value_loss': value_loss.item(),
                    'entropy': entropy.item(),
                    'loss': loss.item(),
                    'kl_div': kl_div.item(),
                })

                # Early stopping if KL divergence is too high
                if abs(kl_div) > self.target_kl:
                    if self.verbose:
                        print(f"Early stopping at epoch {epoch + 1} due to high KL divergence: {kl_div:.6f}")
                    early_stop = True
                    break

        self.metrics_aggregator.save_metrics(self.episode_idx)

    def _calculate_gae(self, rewards: List[float], values: List[float], dones: List[bool]):
        """
        Generalized Advantage Estimation (GAE)
        
        Args:
            rewards: list of rewards (length T)
            values: list of state values, shape [T+1] or [T]
            dones: list of done flags (length T) — 1 if episode ends at step t
            gamma: discount factor
            lam: GAE lambda

        Returns:
            returns: discounted return (target for value function)
            advantages: advantage estimates
        """
        T = len(rewards)
        values = torch.tensor(values, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        
        # Если values не включает последний state (T+1), добавим его как 0
        if len(values) == T:
            values = torch.cat([values, torch.zeros(1)])

        advantages = torch.zeros(T)
        last_gae = 0

        for t in reversed(range(T)):
            non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * values[t + 1] * non_terminal - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + values[:-1]
        return returns.detach(), advantages.detach()
