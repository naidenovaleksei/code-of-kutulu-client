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
from src.envs.agents.dqn_agent_ext import (
    DQNStateEncoderExtv2,
)
from src.envs.agents import AgentObservation
from src.envs.models.conv_a2c_model import ConvA2CModel, ConvA2CDeepModel, ConvA2CGRUModel
from src.envs.models.ext_state_v2_model import ExtStatev2A2AModel
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

    def append(self, experience, log_prob, value, hidden_state=None):
        """Append experience with additional PPO-specific data"""
        # Store the experience along with log probability and value
        ppo_experience = {
            'experience': experience,
            'log_prob': log_prob,
            'value': value,
            'hidden_state': hidden_state,
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
        hidden_states = [item['hidden_state'] for item in augmented_data]
        values = [item['value'] for item in augmented_data]
        turns_to_death = np.arange(len(augmented_data))[::-1] + 1
        
        states, actions, rewards, dones, other_rewards, observations = zip(*experiences)
        is_occupied = np.zeros(len(states), dtype=np.bool)
        # size = len(states[0]['WANDERER_COUNT'])
        # sanity_diffs = np.diff(np.array([obs.obs.entities[0].param0 for obs in observations]))
        # is_occupied = [
        #     int(
        #         max(
        #             state['WANDERER_COUNT'][size // 2][size // 2],
        #             state['SLASHER_COUNT'][size // 2][size // 2],
        #         ) > 0 or \
        #         sanity_diff < -20
        #     )
        #     for state, sanity_diff in zip(states[1:], sanity_diffs)
        # ] + [-1]

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
        
        return states, actions, rewards, dones, log_probs, values, turns_to_death, is_occupied, hidden_states

    def encode_states(self, states):
        return self.state_encoder.encode_states(states)
    
    def rotation_augment_ppo(self, ppo_item):
        """Apply rotation augmentation to PPO experience"""
        clockwise_dir = np.random.randint(0, 4)
        exp = ppo_item['experience']
        
        if exp.action < 4:
            clockwise_dir = (clockwise_dir - exp.action + 4) % 4
        
        augmented_exp = Experience(
            self.state_encoder.state_rotation_augment(exp.state, clockwise_dir),
            self.state_encoder.action_rotation_augment(exp.action, clockwise_dir),
            exp.reward,
            exp.done,
            exp.new_state,
            exp.observation,
        )
        
        ppo_item = dict(ppo_item)
        ppo_item['experience'] = augmented_exp

        return ppo_item


class PPOAgent(ActorAgent):
    def __init__(self, state_type, actions=EXTENDED_KUTULU_ACTIONS,
                 lr=LEARNING_RATE,
                 gamma=GAMMA, model_params={},
                 train=False, verbose=False, 
                 entropy_coef=0.01, value_loss_coef=0.5,
                 terminate_loss_coef=0, occupation_loss_coef=0,
                 clip_ratio=0.2, ppo_epochs=4, mini_batch_size=64,
                 target_kl=0.01, max_grad_norm=0.5, use_deep=False,
                 gae_lambda=0.95, reward_params=None, need_aug=False, use_gru=False,
                 segment_length=20,
                 encode_states_first=True, **kw):
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
        action_space_n = len(actions)
        if state_type in ['conv', 'conv_ext']:
            if state_type == 'conv':
                self.state_encoder = DQNStateEncoderConv(is_ext=False)
            elif state_type == 'conv_ext':
                self.state_encoder = DQNStateEncoderConv(is_ext=True)
            self.size = model_params.pop('size', 3)
            if use_deep:
                model = ConvA2CDeepModel(
                    num_classes=action_space_n,
                    size=self.size,
                    in_channels=self.state_encoder.layers_count(),
                    **model_params
                )
            elif use_gru:
                model = ConvA2CGRUModel(
                    num_classes=action_space_n,
                    size=self.size,
                    in_channels=self.state_encoder.layers_count(),
                    **model_params
                )
            else:
                model = ConvA2CModel(
                    num_classes=action_space_n,
                    size=self.size,
                    in_channels=self.state_encoder.layers_count(),
                    **model_params
                )
        elif state_type == 'closest_ext_v2':
            self.state_encoder = DQNStateEncoderExtv2()
            model = ExtStatev2A2AModel(
                num_classes=action_space_n,
                **model_params
            )
        else:
            raise ValueError("PPO agent only supports 'conv', 'conv_ext' and 'closest_ext_v2' state type")

        super(PPOAgent, self).__init__(
            state_type=state_type,
            action_space_n=action_space_n,
            lr=lr,
            gamma=gamma,
            train=train,
            verbose=verbose,
            model=model,
            state_encoder=self.state_encoder,
            use_gru=use_gru,
            segment_length=segment_length,
            **kw,
        )
        
        self.actions = actions
        if reward_params is None:
            reward_params = {}
        self.reward_params = reward_params
        self.need_aug = need_aug
        # Initialize with PPO-specific buffer
        ppo_buffer = PPOBuffer(self.state_encoder, self.actions, need_aug=self.need_aug,
                               verbose=self.verbose, reward_params=self.reward_params)
        # Replace the episode buffer with PPO buffer
        self.episode_buffer = ppo_buffer
        self.episode_buffer.start_episode()
        self.encode_states_first = encode_states_first
        
        # PPO-specific parameters
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.terminate_loss_coef = terminate_loss_coef
        self.occupation_loss_coef = occupation_loss_coef
        self.clip_ratio = clip_ratio
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.target_kl = target_kl
        self.max_grad_norm = max_grad_norm
        self.gae_lambda = gae_lambda
        self.occupation_criterion = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([10], dtype=torch.float32),
        )
        # self.occupation_criterion = torch.nn.CrossEntropyLoss(
        #     weight=torch.tensor([1, 10], dtype=torch.float32),
        #     ignore_index=-1,
        #     label_smoothing=0.1,
        # )
        
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
                PPOBuffer(self.state_encoder, self.actions, need_aug=self.need_aug,
                          verbose=self.verbose, reward_params=self.reward_params)
                for _ in range(num_envs)
            ]
            for buffer in self.env_buffers:
                buffer.start_episode()
            if self.verbose:
                print(f"PPO Agent initialized with {num_envs} environments")

    # def set_env(self, env):
    #     assert self.state_type == 'conv'
    #     self.observer = KutuluConvObserver(env, self.size)

    def generate_state_and_step(self, player_id, need_update=True):
        output = super().generate_state_and_step(player_id, need_update, True)
        action = output['action']
        policy = output['policy']
        value = output['value']

        # PPO-specific: calculate log probability and store value
        action_dist = torch.distributions.Categorical(policy)
        log_prob = action_dist.log_prob(torch.tensor([action])).item()
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
        if self.use_gru:
            buffer.append(
                exp, 
                self.current_log_prob, 
                self.current_value,
                self.hidden_state.cpu().tolist(),
            )
        else:
            buffer.append(
                exp, 
                self.current_log_prob, 
                self.current_value,
            )

    def collect_all_data_from_buffers(self):
        # Collect data from all environment buffers
        all_states, all_actions, all_rewards, all_dones = [], [], [], []
        all_log_probs, all_values = [], []
        all_turns_to_death = []
        all_is_occupied = []
        all_hidden_states = []
        
        # Collect bonus averages from all buffers
        bonus_sums = {}
        bonus_counts = {}

        for env_buffer in self.env_buffers:
            if len(env_buffer.buffer) > 0:
                states, actions, rewards, dones, log_probs, values, turns_to_death, is_occupied, hidden_states = env_buffer.end_episode()
                all_states.extend(states)
                all_actions.extend(actions)
                all_rewards.extend(rewards)
                all_dones.extend(dones)
                all_log_probs.extend(log_probs)
                all_values.extend(values)
                all_turns_to_death.extend(turns_to_death)
                all_is_occupied.extend(is_occupied)
                all_hidden_states.extend(hidden_states)
                
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
        return (
            all_states, all_actions, all_rewards, all_dones, all_log_probs, all_values,
            all_turns_to_death, all_is_occupied, all_hidden_states
        )

    def train_step(self):
        if self.train and len(self.episode_buffer.buffer) > 0:
            super().train_step()

    def _train_model(self):
        """Train the PPO model with clipped objective"""
        # End the current episode and get the episode data
        states, actions, rewards, dones, old_log_probs, values, turns_to_death, is_occupied = self.episode_buffer.end_episode()
        # Get bonus averages if available
        if self.episode_buffer.latest_bonus_averages is not None:
            self.latest_bonus_averages = self.episode_buffer.latest_bonus_averages
        self._train_model_with_data(states, actions, rewards, dones, old_log_probs, values, turns_to_death, is_occupied)
    
    def train_multi_env_step(self):
        """Train using data from all environments"""
        if not self.train or self.env_buffers is None:
            return self.train_step()  # Fall back to single-env
        
        all_data = self.collect_all_data_from_buffers()
        # Use the combined data for training
        if self.use_gru:
            self._train_model_with_data_reccurent(*all_data)
        else:
            self._train_model_with_data(*all_data)
        
        if self.scheduler:
            self.scheduler.step()
    
    def _train_model_with_data(self, states, actions, rewards, dones, old_log_probs, values, turns_to_death, is_occupied, hidden_states=None):
        """Train the PPO model with provided data"""
        if len(states) == 0:
            return
    
        self.model.train()

        # Encode states
        if self.encode_states_first:
            states_tensor = self.episode_buffer.encode_states(states)
        else:
            states = np.array(states)
        actions_tensor = torch.tensor(actions, dtype=torch.long)
        old_log_probs_tensor = torch.tensor(old_log_probs, dtype=torch.float32)
        turns_to_death_tensor = torch.tensor(turns_to_death, dtype=torch.float32)
        is_occupied_tensor = torch.tensor(is_occupied, dtype=torch.float32)

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
                if len(batch_indices) < self.mini_batch_size:
                    continue
                
                # Get mini-batch data
                if self.encode_states_first:
                    batch_states = states_tensor[batch_indices]
                else:
                    batch_states = self.episode_buffer.encode_states(states[batch_indices.tolist()])
                batch_actions = actions_tensor[batch_indices]
                batch_old_log_probs = old_log_probs_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_turns_to_death = turns_to_death_tensor[batch_indices]
                batch_is_occupied = is_occupied_tensor[batch_indices]
                
                # Forward pass
                output = self.model(batch_states)
                policy = output['policy']
                current_values = output['value'].view(-1)
                pred_turns_to_death = output['turns_to_death'].view(-1)
                pred_is_occupied = output['is_occupied'].view(-1)
                
                early_stop = self._backward_propagation(
                    policy, current_values, pred_turns_to_death, pred_is_occupied,
                    batch_actions, batch_advantages, batch_old_log_probs, batch_returns,
                    batch_turns_to_death, batch_is_occupied, epoch
                )

        self.metrics_aggregator.save_metrics(self.episode_idx)
    
    def _backward_propagation(self, policy, current_values, pred_turns_to_death, pred_is_occupied,
                              batch_actions, batch_advantages, batch_old_log_probs, batch_returns,
                              batch_turns_to_death, batch_is_occupied, epoch):
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

        # Aux terminate loss
        terminate_loss = F.l1_loss(
            pred_turns_to_death,
            torch.log1p(batch_turns_to_death)
        )

        occupation_loss = self.occupation_criterion(
            pred_is_occupied[batch_is_occupied >= 0],
            batch_is_occupied[batch_is_occupied >= 0],
        )

        # Entropy for exploration
        entropy = -(log_probs * policy).sum(dim=1).mean()
        
        # Total loss
        loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
        loss += self.terminate_loss_coef * terminate_loss
        loss += self.occupation_loss_coef * occupation_loss
        
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
            'terminate_loss': terminate_loss.item(),
            'occupation_loss': occupation_loss.item(),
            'entropy': entropy.item(),
            'loss': loss.item(),
            'kl_div': kl_div.item(),
        })

        # Early stopping if KL divergence is too high
        if abs(kl_div) > self.target_kl:
            if self.verbose:
                print(f"Early stopping at epoch {epoch + 1} due to high KL divergence: {kl_div:.6f}")
            return True

        return False

    
    def _train_model_with_data_reccurent(self, states, actions, rewards, dones,
                                         old_log_probs, values,
                                         turns_to_death, is_occupied, hidden_states):
        """Train the PPO model with TBPTT (Truncated Backpropagation Through Time)"""
        if len(states) == 0:
            return
    
        self.model.train()

        # Step 1: Split all lists by dones to create sequences
        sequences = self._split_by_dones(
            states, actions, rewards, dones,
            old_log_probs, values,
            turns_to_death, is_occupied, hidden_states
        )
        
        if len(sequences) == 0:
            return
        
        # Step 2: Pad sequences to have same length
        padded_sequences = self._pad_sequences(sequences)
        
        # Step 3: Stack and transform to tensors with shape (batch_size, max_seq_length)
        batch_data = self._stack_sequences_to_tensors(padded_sequences)
        
        # Step 4: Calculate masks for all sequences
        masks = batch_data['masks']
        
        # Step 5: Encode states if needed
        if self.encode_states_first:
            batch_data['states'] = self._encode_padded_states(batch_data['states'], masks)
        
        # Calculate returns and advantages using GAE for each sequence
        batch_data['returns'], batch_data['advantages'] = self._calculate_gae_sequences(
            batch_data['rewards'], batch_data['values'], batch_data['dones'], masks
        )
        
        # Normalize advantages across all sequences using masks
        all_advantages = batch_data['advantages'][masks]
        if len(all_advantages) > 1:
            mean_adv = all_advantages.mean()
            std_adv = all_advantages.std()
            batch_data['advantages'] = (batch_data['advantages'] - mean_adv) / (std_adv + 1e-8)

        # Step 6: Each epoch runs along all calculated data once
        early_stop = False

        for epoch in range(self.ppo_epochs):
            if early_stop:
                break
                
            # Get sequence length (number of valid timesteps)
            seq_len = batch_data['states'].shape[1]
            
            # Process all sequences in segments for TBPTT
            for segment_start in range(0, seq_len, self.segment_length):
                segment_end = min(segment_start + self.segment_length, seq_len)
                
                # Extract segment data from all sequences
                segment_data = self._extract_batch_segment_data(batch_data, masks, segment_start, segment_end)
                
                # Step 8: Calculate loss in segment and process backpropagation
                early_stop = self._train_batch_segment(segment_data, epoch)
            
            if early_stop:
                break

        self.metrics_aggregator.save_metrics(self.episode_idx)
    
    def _split_by_dones(self, states, actions, rewards, dones,
                        old_log_probs, values,
                        turns_to_death, is_occupied, hidden_state):
        """Split all data by done flags to create individual sequences"""
        sequences = []
        current_seq = {
            'states': [], 'actions': [], 'rewards': [], 'dones': [],
            'old_log_probs': [], 'values': [],
            'turns_to_death': [], 'is_occupied': [], 'hidden_state': [],
        }
        
        for i in range(len(states)):
            current_seq['states'].append(states[i])
            current_seq['actions'].append(actions[i])
            current_seq['rewards'].append(rewards[i])
            current_seq['dones'].append(dones[i])
            current_seq['old_log_probs'].append(old_log_probs[i])
            current_seq['values'].append(values[i])
            current_seq['turns_to_death'].append(turns_to_death[i])
            current_seq['is_occupied'].append(is_occupied[i])
            current_seq['hidden_state'].append(hidden_state[i])
            
            # If episode is done, finish current sequence and start new one
            if dones[i]:
                if len(current_seq['states']) > 0:
                    sequences.append(current_seq)
                current_seq = {
                    'states': [], 'actions': [], 'rewards': [], 'dones': [],
                    'old_log_probs': [], 'values': [],
                    'turns_to_death': [], 'is_occupied': [], 'hidden_state': [],
                }
        
        # Add remaining sequence if not empty
        if len(current_seq['states']) > 0:
            sequences.append(current_seq)
        
        return sequences
    
    def _pad_sequences(self, sequences):
        """Pad all sequences to have the same length"""
        if not sequences:
            return sequences
        
        # Find maximum sequence length
        max_length = max(len(seq['states']) for seq in sequences)
        
        padded_sequences = []
        for seq in sequences:
            padded_seq = {}
            seq_len = len(seq['states'])
            
            for key in seq.keys():
                padded_list = seq[key].copy()
                
                # Pad with appropriate values
                if key in ['states', 'hidden_state']:
                    # Pad states with zeros or last state
                    pad_value = seq[key][-1] if seq[key] else None
                    for _ in range(max_length - seq_len):
                        padded_list.append(pad_value)
                elif key in ['actions', 'turns_to_death']:
                    # Pad with zeros
                    for _ in range(max_length - seq_len):
                        padded_list.append(0)
                elif key in ['rewards', 'old_log_probs', 'values']:
                    # Pad with zeros
                    for _ in range(max_length - seq_len):
                        padded_list.append(0.0)
                elif key == 'dones':
                    # Pad with True (episode done)
                    for _ in range(max_length - seq_len):
                        padded_list.append(True)
                elif key == 'is_occupied':
                    # Pad with -1 (ignore index)
                    for _ in range(max_length - seq_len):
                        padded_list.append(-1)
                
                padded_seq[key] = padded_list
            
            # Create mask for valid timesteps
            mask = [True] * seq_len + [False] * (max_length - seq_len)
            padded_seq['mask'] = mask
            
            padded_sequences.append(padded_seq)
        
        return padded_sequences
    
    def _stack_sequences_to_tensors(self, padded_sequences):
        """Stack padded sequences into tensors with shape (batch_size, max_seq_length)"""
        if not padded_sequences:
            return {}
        
        # Stack all sequences
        batch_data = {}
        
        # Handle states separately (they might not be tensors yet)
        batch_data['states'] = [seq['states'] for seq in padded_sequences]
        
        # Convert other data to tensors
        batch_data['actions'] = torch.tensor([seq['actions'] for seq in padded_sequences], dtype=torch.long)
        batch_data['rewards'] = torch.tensor([seq['rewards'] for seq in padded_sequences], dtype=torch.float32)
        batch_data['dones'] = torch.tensor([seq['dones'] for seq in padded_sequences], dtype=torch.float32)
        batch_data['old_log_probs'] = torch.tensor([seq['old_log_probs'] for seq in padded_sequences], dtype=torch.float32)
        batch_data['values'] = torch.tensor([seq['values'] for seq in padded_sequences], dtype=torch.float32)
        batch_data['turns_to_death'] = torch.tensor([seq['turns_to_death'] for seq in padded_sequences], dtype=torch.float32)
        batch_data['is_occupied'] = torch.tensor([seq['is_occupied'] for seq in padded_sequences], dtype=torch.float32)
        batch_data['hidden_state'] = torch.tensor([seq['hidden_state'] for seq in padded_sequences], dtype=torch.float32)
        batch_data['masks'] = torch.tensor([seq['mask'] for seq in padded_sequences], dtype=torch.bool)
        
        return batch_data
    
    def _encode_padded_states(self, states_list, masks):
        """Encode padded states while respecting masks"""
        batch_size = len(states_list)
        max_seq_length = len(states_list[0])
        
        # Flatten all states for encoding
        all_states = []
        for seq_states in states_list:
            all_states.extend(seq_states)
        
        # Encode all states at once
        encoded_states = self.episode_buffer.encode_states(all_states)
        
        # Reshape back to (batch_size, max_seq_length, ...)
        state_shape = encoded_states.shape[1:]  # Get feature dimensions
        encoded_states = encoded_states.view(batch_size, max_seq_length, *state_shape)
        
        return encoded_states
    
    def _calculate_gae_sequences(self, rewards, values, dones, masks):
        """Calculate GAE for sequences with masks"""
        batch_size, max_seq_length = rewards.shape
        
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        
        for seq_idx in range(batch_size):
            if not masks[seq_idx].any():
                continue
                
            # Get valid length for this sequence
            seq_len = masks[seq_idx].sum().item()
            
            # Extract valid parts of the sequence
            seq_rewards = rewards[seq_idx, :seq_len]
            seq_values = values[seq_idx, :seq_len]
            seq_dones = dones[seq_idx, :seq_len]
            
            # Add final value (0 for terminal states)
            seq_values_extended = torch.cat([seq_values, torch.zeros(1)])
            
            # Calculate GAE for this sequence
            seq_advantages = torch.zeros(seq_len)
            last_gae = 0
            
            for t in reversed(range(seq_len)):
                non_terminal = 1.0 - seq_dones[t]
                delta = seq_rewards[t] + self.gamma * seq_values_extended[t + 1] * non_terminal - seq_values_extended[t]
                last_gae = delta + self.gamma * self.gae_lambda * non_terminal * last_gae
                seq_advantages[t] = last_gae
            
            seq_returns = seq_advantages + seq_values
            
            # Store back in the batch tensors
            returns[seq_idx, :seq_len] = seq_returns
            advantages[seq_idx, :seq_len] = seq_advantages
        
        return returns, advantages
    
    def _extract_batch_segment_data(self, batch_data, masks, segment_start, segment_end):
        """Extract segment data from all sequences for batch processing"""
        segment_data = {}
        
        # Extract segments from all sequences
        if self.encode_states_first:
            segment_data['states'] = batch_data['states'][:, segment_start:segment_end]
        else:
            # For non-encoded states, extract from each sequence
            segment_states = []
            for seq_idx in range(len(batch_data['states'])):
                segment_states.append(batch_data['states'][seq_idx][segment_start:segment_end])
            segment_data['states'] = segment_states
        
        segment_data['actions'] = batch_data['actions'][:, segment_start:segment_end]
        segment_data['old_log_probs'] = batch_data['old_log_probs'][:, segment_start:segment_end]
        segment_data['returns'] = batch_data['returns'][:, segment_start:segment_end]
        segment_data['advantages'] = batch_data['advantages'][:, segment_start:segment_end]
        segment_data['turns_to_death'] = batch_data['turns_to_death'][:, segment_start:segment_end]
        segment_data['is_occupied'] = batch_data['is_occupied'][:, segment_start:segment_end]
        segment_data['hidden_state'] = batch_data['hidden_state'][:, segment_start:segment_end]
        segment_data['masks'] = masks[:, segment_start:segment_end]
        
        return segment_data
    
    def _train_batch_segment(self, segment_data, epoch):
        """Train on a batch of segments with TBPTT"""
        assert self.use_gru
        
        batch_size = segment_data['actions'].shape[0]
        segment_length = segment_data['actions'].shape[1]

        # Filter out sequences that have no valid data in this segment
        segment_states = segment_data['states']
        hidden_states = segment_data['hidden_state']
        padded_mask = segment_data['masks'].flatten()
        batch_actions = segment_data['actions'].flatten()[padded_mask]
        batch_old_log_probs = segment_data['old_log_probs'].flatten()[padded_mask]
        batch_advantages = segment_data['advantages'].flatten()[padded_mask]
        batch_returns = segment_data['returns'].flatten()[padded_mask]
        batch_turns_to_death = segment_data['turns_to_death'].flatten()[padded_mask]
        batch_is_occupied = segment_data['is_occupied'].flatten()[padded_mask]
        
        # Encode states if needed
        if not self.encode_states_first:
            # Flatten states for encoding
            all_states = []
            for seq_idx in range(batch_size):
                all_states.extend(segment_data['states'][seq_idx])
            segment_states = self.episode_buffer.encode_states(all_states)
            # Reshape back to (batch_size, segment_length, ...)
            state_shape = segment_states.shape[1:]
            segment_states = segment_states.view(batch_size, segment_length, *state_shape)
        else:
            segment_states = segment_data['states']
        
        policy_list, v_list, turns_to_death_list, is_occupied_list = [], [], [], []
        hidden_state = hidden_states[:, 0, 0, 0].unsqueeze(0)
        for l in range(segment_data['states'].shape[1]):
            output = self.model(segment_states[:, l], hidden_state)
            _policy = output['policy']
            _current_values = output['value']
            _pred_turns_to_death = output['turns_to_death']
            _pred_is_occupied = output['is_occupied']
            hidden_state = output['hidden_state']
            hidden_state *= segment_data['masks'][:, l].unsqueeze(1)
            
            policy_list.append(_policy)
            v_list.append(_current_values)
            turns_to_death_list.append(_pred_turns_to_death)
            is_occupied_list.append(_pred_is_occupied)
        
        policy = torch.stack(policy_list, dim=1).reshape(-1, policy_list[0].shape[1])[padded_mask]
        current_values = torch.stack(v_list, dim=1).flatten()[padded_mask]
        pred_turns_to_death = torch.stack(turns_to_death_list, dim=1).flatten()[padded_mask]
        pred_is_occupied = torch.stack(is_occupied_list, dim=1).flatten()[padded_mask]

        return self._backward_propagation(
            policy, current_values, pred_turns_to_death, pred_is_occupied,
            batch_actions, batch_advantages, batch_old_log_probs, batch_returns,
            batch_turns_to_death, batch_is_occupied, epoch
        )

    def get_metric_names(self):
        return [
            'policy_loss',
            'value_loss',
            'terminate_loss',
            'occupation_loss',
            'entropy',
            'kl_div',
            'loss'
        ]

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
