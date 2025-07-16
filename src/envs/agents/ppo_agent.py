from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from collections import deque

from src.envs.kutulu_entities import (
    EntityKind,
    EffectType,
    KutuluEntity,
    MoveType,
)
from src.game.template import (
    EXTENDED_KUTULU_ACTIONS,
    REL_POSITIONS,
    get_all_distances,
)
from src.envs.agents import AgentObservation
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
from src.envs.distance import find_path, distance, UnreachedPositionError
from src.envs.reward_shaper import PotentialRewardShaper


METRICS_SMOOTH_COEF = 0.05
PLAN_DISTANCE = 2
LIGHT_DISTANCE = 5
YELL_DISTANCE = 1

class RewardShaper:
    def __init__(self, actions, verbose,
            good_plan_bonus=0.3,
            bad_plan_bonus=-0.2,
            good_light_bonus=0.3,
            bad_light_bonus=-0.1,
            other_reward_coef=0.01,
            good_explorers_nearby_bonus=0.02,
            bad_explorers_nearby_bonus=-0.04,
            yell_bonus_coef=0.5,
            bad_yell_bonus=-0.2,
            shelter_bonus=0.5,
            wait_reward_coef=1.0,
            bad_towards_enemy_bonus=-0.3,
            ):
        self.actions = actions
        self.verbose = verbose
        self.good_plan_bonus = good_plan_bonus
        self.bad_plan_bonus = bad_plan_bonus
        self.good_light_bonus = good_light_bonus
        self.bad_light_bonus = bad_light_bonus
        self.other_reward_coef = other_reward_coef
        self.good_explorers_nearby_bonus = good_explorers_nearby_bonus
        self.bad_explorers_nearby_bonus = bad_explorers_nearby_bonus
        self.bad_yell_bonus = bad_yell_bonus
        self.yell_bonus_coef = yell_bonus_coef
        self.shelter_bonus = shelter_bonus
        self.wait_reward_coef = wait_reward_coef
        self.bad_towards_enemy_bonus = bad_towards_enemy_bonus

    def _get_wanderers(self, observation):
        return [
            e for e in observation.obs.entities
            if e.kind in (EntityKind.WANDERER.value, EntityKind.SLASHER.value)
        ]
    
    def _get_explorers(self, observation, player_id) -> List[KutuluEntity]:
        return [
            e for e in observation.obs.entities
            if e.kind == EntityKind.EXPLORER.value and e.id != player_id
        ]
    
    def _get_shelters(self, observation) -> List[KutuluEntity]:
        return [
            e for e in observation.obs.entities
            if e.kind == EntityKind.EFFECT_SHELTER.value
        ]
    
    def _get_nearby_count(self,
                          player_pos: Tuple[int],
                          entities: List[KutuluEntity],
                          lines: List[List[str]],
                          limit: int):
        e_nearby_count = 0
        for e in entities:
            e_pos = (e.x, e.y)
            if distance(player_pos, e_pos) <= limit:
                path = find_path(player_pos, e_pos, lines)
                if len(path) <= limit:
                    e_nearby_count += 1
        return e_nearby_count
    
    def _get_min_dist(self,
                      player_pos: Tuple[int],
                      entities: List[KutuluEntity],
                      lines: List[List[str]],
                      limit: int):
        min_dist = 1000
        for e in entities:
            e_pos = (e.x, e.y)
            if distance(player_pos, e_pos) <= limit:
                path = find_path(player_pos, e_pos, lines)
                min_dist = min(min_dist, len(path))
        return min_dist

    def _score_moves_by_wanderers(self, player_pos, observation: AgentObservation, limit=4):
        wanderers = self._get_wanderers(observation)
        wanderers = [
            w.to_dict() for w in wanderers
            if distance(player_pos, (w.x, w.y)) <= limit
        ]
        all_distances = get_all_distances(wanderers, player_pos, observation.info.lines)
        all_distances = {
            # [1, 2, 2] -> 0.6385
            k: np.exp(-np.array(v)).sum()
            for k, v in all_distances.items()
        }
        move_scores = [all_distances.get(rel_pos, 0) for rel_pos in REL_POSITIONS[:4]]
        return move_scores

    def _compute_shaped_reward(self,
                               observation: AgentObservation,
                               action: int,
                               next_observations: AgentObservation,
                               original_reward: float,
                               other_rewards: List[float]):
        reward = 0
        if self.actions[action] == EffectType.PLAN.value:
            player = next_observations.obs.entities[0]
            assert player.id == next_observations.player_id
            player_pos = (player.x, player.y)
            explorers = self._get_explorers(next_observations, player.id)
            players_nearby_count = self._get_nearby_count(
                player_pos,
                explorers,
                next_observations.info.lines,
                PLAN_DISTANCE,
            )
            if players_nearby_count > 0:
                plan_bonus = self.good_plan_bonus * (players_nearby_count + 1)
            else:
                plan_bonus = self.bad_plan_bonus
            reward += plan_bonus
            if self.verbose:
                print(f"plan_bonus: {plan_bonus}")
        elif self.actions[action] == EffectType.LIGHT.value:
            player = observation.obs.entities[0]
            assert player.id == observation.player_id
            player_pos = (player.x, player.y)
            wanderers = self._get_wanderers(observation)
            is_bad_light = min([distance(player_pos, (w.x, w.y)) for w in wanderers], default=0) <= 1
            if is_bad_light:
                light_bonus = self.bad_light_bonus
            else:
                cur_enemies_min_dist = self._get_min_dist(
                    player_pos,
                    wanderers,
                    observation.info.lines,
                    LIGHT_DISTANCE,
                )
                next_enemies_min_dist = self._get_min_dist(
                    player_pos,
                    self._get_wanderers(next_observations),
                    next_observations.info.lines,
                    LIGHT_DISTANCE,
                )
                if cur_enemies_min_dist < next_enemies_min_dist:
                    light_bonus = self.good_light_bonus
                else:
                    light_bonus = self.bad_light_bonus
            reward += light_bonus
            if self.verbose:
                print(f"light_bonus: {light_bonus}")
        elif self.actions[action] == EffectType.YELL.value:
            yell_bonus = self.bad_yell_bonus
            other_rewards = [r for r in other_rewards if r is not None]
            if len(other_rewards) != 0:
                min_other_reward = min(other_rewards)
                if min_other_reward < 0:
                    player = observation.obs.entities[0]
                    assert player.id == observation.player_id
                    player_pos = (player.x, player.y)
                    explorers = self._get_explorers(observation, player.id)
                    players_nearby_count = self._get_nearby_count(
                        player_pos,
                        explorers,
                        observation.info.lines,
                        1,
                    )
                    if players_nearby_count > 0:
                        yell_bonus = - self.yell_bonus_coef * min_other_reward
            reward += yell_bonus
            if self.verbose:
                print(f"yell_bonus: {yell_bonus}")
        elif self.actions[action] == MoveType.WAIT.value:
            if original_reward < 0:
                wait_bonus = self.wait_reward_coef * original_reward
                reward += wait_bonus
                if self.verbose:
                    print(f"wait_bonus: {wait_bonus}")
        else:
            # MOVE
            assert self.actions[action] in (
                MoveType.UP.value,
                MoveType.RIGHT.value,
                MoveType.DOWN.value,
                MoveType.LEFT.value,
            )
            player = observation.obs.entities[0]
            assert player.id == observation.player_id
            player_pos = (player.x, player.y)
            move_scores = self._score_moves_by_wanderers(player_pos, observation)
            if move_scores[action] > 0 and move_scores[action] == max(move_scores):
                towards_enemy_bonus = self.bad_towards_enemy_bonus
                reward += towards_enemy_bonus
                if self.verbose:
                    print(f"towards_enemy_bonus: {towards_enemy_bonus}")
        if self.other_reward_coef is not None:
            other_rewards = [r for r in other_rewards if r is not None]
            if len(other_rewards) != 0:
                min_other_reward = min(other_rewards)
                min_other_reward = min(min_other_reward, 0)
                other_reward_bouns = - self.other_reward_coef * min_other_reward
                reward += other_reward_bouns
                if self.verbose:
                    print(f"other_reward_bouns: {other_reward_bouns}")
        if self.good_explorers_nearby_bonus != 0 or self.bad_explorers_nearby_bonus != 0:
            player = next_observations.obs.entities[0]
            assert player.id == next_observations.player_id
            player_pos = (player.x, player.y)
            explorers = self._get_explorers(next_observations, player.id)
            players_nearby_count = self._get_nearby_count(
                player_pos,
                explorers,
                next_observations.info.lines,
                PLAN_DISTANCE,
            )
            if players_nearby_count > 0:
                nearby_bonus = self.good_explorers_nearby_bonus * players_nearby_count
            else:
                nearby_bonus = self.bad_explorers_nearby_bonus
            reward += nearby_bonus
            if self.verbose:
                print(f"nearby_bonus: {nearby_bonus}")
        if self.shelter_bonus != 0:
            player = next_observations.obs.entities[0]
            assert player.id == next_observations.player_id
            lines = next_observations.info.lines
            player_pos = (player.x, player.y)
            shelters = self._get_shelters(next_observations)
            active_shelters = [sh for sh in shelters if sh.param0 > 0]
            shelters_underfoot_count = self._get_nearby_count(
                player_pos,
                active_shelters,
                next_observations.info.lines,
                0,
            )
            if shelters_underfoot_count > 0:
                reward += self.shelter_bonus
                if self.verbose:
                    print(f"shelter_bonus: {self.shelter_bonus}")

        return reward

    def recalculate_rewards(self,
                            rewards: List[float],
                            actions: List[int],
                            states: List,
                            dones: List[bool],
                            other_rewards: List[List[float]],
                            observations: List):
        assert dones[-1]
        assert sum(dones[:-1]) == 0
        rewards = list(rewards)
        T = len(rewards)
        for t in range(0, T - 1):
            rewards[t] += self._compute_shaped_reward(
                observations[t], actions[t], observations[t+1], rewards[t], other_rewards[t],
            )
        return rewards


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
            self.reward_shaper = PotentialRewardShaper(actions, verbose, **reward_params)
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
            None,
            None,
        )
        
        return {
            'experience': augmented_exp,
            'log_prob': ppo_item['log_prob'],
            'value': ppo_item['value']
        }


class PPOAgent(ActorAgent):
    def __init__(self, state_type, action_space_n, actions=EXTENDED_KUTULU_ACTIONS,
                 lr=LEARNING_RATE,
                 gamma=GAMMA, model_params={},
                 train=False, verbose=False, 
                 entropy_coef=0.01, value_loss_coef=0.5, 
                 clip_ratio=0.2, ppo_epochs=4, mini_batch_size=64,
                 target_kl=0.01, max_grad_norm=0.5,
                 gae_lambda=0.95, reward_params=None, **kw):
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
        # Initialize with PPO-specific buffer
        ppo_buffer = PPOBuffer(DQNStateEncoderConv(), self.actions,
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
                PPOBuffer(DQNStateEncoderConv(), self.actions,
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
        state, action, policy, value = super().generate_state_and_step(player_id, need_update, True)

        # PPO-specific: calculate log probability and store value
        action_dist = torch.distributions.Categorical(policy)
        log_prob = action_dist.log_prob(torch.tensor(action)).item()
        value_estimate = value.item()
        
        # Store for later use in append_observation
        self.current_log_prob = log_prob
        self.current_value = value_estimate

        return state, action

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

    def train_step(self):
        if self.train and len(self.episode_buffer.buffer) > 0:
            super().train_step()
    
    def train_multi_env_step(self):
        """Train using data from all environments"""
        if not self.train or self.env_buffers is None:
            return self.train_step()  # Fall back to single-env
        
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
        
        if len(all_states) == 0:
            return
        
        # Use the combined data for training
        self._train_model_with_data(all_states, all_actions, all_rewards, all_dones,
                                   all_log_probs, all_values)
        
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
            
            total_policy_loss = 0
            total_value_loss = 0
            total_entropy = 0
            total_kl_div = 0
            num_batches = 0
            
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
                
                # Accumulate metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                num_batches += 1
                
                # Calculate KL divergence for early stopping
                with torch.no_grad():
                    kl_div = (batch_old_log_probs - current_log_probs).mean()
                    total_kl_div += kl_div.item()
                
                # Early stopping if KL divergence is too high
                if abs(kl_div) > self.target_kl:
                    if self.verbose:
                        print(f"Early stopping at epoch {epoch + 1} due to high KL divergence: {kl_div:.6f}")
                    early_stop = True
                    break

        # Update loss tracking
        final_loss = total_policy_loss + total_value_loss - total_entropy
        if self.last_loss == np.inf:
            self.last_loss = final_loss / num_batches
            self.policy_loss = total_policy_loss / num_batches
            self.value_loss = total_value_loss / num_batches
            self.entropy = total_entropy / num_batches
            self.kl_div = total_kl_div / num_batches
        else:
            self.last_loss = METRICS_SMOOTH_COEF * (final_loss / num_batches) + (1 - METRICS_SMOOTH_COEF) * self.last_loss
            self.policy_loss = METRICS_SMOOTH_COEF * (total_policy_loss / num_batches) + (1 - METRICS_SMOOTH_COEF) * self.policy_loss
            self.value_loss = METRICS_SMOOTH_COEF * (total_value_loss / num_batches) + (1 - METRICS_SMOOTH_COEF) * self.value_loss
            self.entropy = METRICS_SMOOTH_COEF * (total_entropy / num_batches) + (1 - METRICS_SMOOTH_COEF) * self.entropy
            self.kl_div = METRICS_SMOOTH_COEF * (total_kl_div / num_batches) + (1 - METRICS_SMOOTH_COEF) * self.kl_div

        if self.verbose:
            print(f"Multi-Env Episode {self.episode_idx}, "
                  f"Samples: {dataset_size}, "
                  f"Policy Loss: {total_policy_loss/num_batches:.4f}, "
                  f"Value Loss: {total_value_loss/num_batches:.4f}, "
                  f"Entropy: {total_entropy/num_batches:.4f}, "
                  f"KL Div: {total_kl_div/num_batches:.6f}, "
                  f"Total Return: {np.sum(rewards):.4f}")

    def _train_model(self):
        """Train the PPO model with clipped objective"""
        # End the current episode and get the episode data
        states, actions, rewards, dones, old_log_probs, values = self.episode_buffer.end_episode()
        
        # Get bonus averages if available
        if self.episode_buffer.latest_bonus_averages is not None:
            self.latest_bonus_averages = self.episode_buffer.latest_bonus_averages

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
            
            total_policy_loss = 0
            total_value_loss = 0
            total_entropy = 0
            total_kl_div = 0
            num_batches = 0
            
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
                
                # Accumulate metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                num_batches += 1
                
                # Calculate KL divergence for early stopping
                with torch.no_grad():
                    kl_div = (batch_old_log_probs - current_log_probs).mean()
                    total_kl_div += kl_div.item()
                
                # Early stopping if KL divergence is too high
                if abs(kl_div) > self.target_kl:
                    if self.verbose:
                        print(f"Early stopping at epoch {epoch + 1} due to high KL divergence: {kl_div:.6f}")
                    early_stop = True
                    break

        # Update loss tracking
        final_loss = total_policy_loss + total_value_loss - total_entropy
        if self.last_loss == np.inf:
            self.last_loss = final_loss / num_batches
            self.policy_loss = total_policy_loss / num_batches
            self.value_loss = total_value_loss / num_batches
            self.entropy = total_entropy / num_batches
            self.kl_div = total_kl_div / num_batches
        else:
            self.last_loss = METRICS_SMOOTH_COEF * (final_loss / num_batches) + (1 - METRICS_SMOOTH_COEF) * self.last_loss
            self.policy_loss = METRICS_SMOOTH_COEF * (total_policy_loss / num_batches) + (1 - METRICS_SMOOTH_COEF) * self.policy_loss
            self.value_loss = METRICS_SMOOTH_COEF * (total_value_loss / num_batches) + (1 - METRICS_SMOOTH_COEF) * self.value_loss
            self.entropy = METRICS_SMOOTH_COEF * (total_entropy / num_batches) + (1 - METRICS_SMOOTH_COEF) * self.entropy
            self.kl_div = METRICS_SMOOTH_COEF * kl_div + (1 - METRICS_SMOOTH_COEF) * self.kl_div

        if self.verbose:
            print(f"Episode {self.episode_idx}, "
                  f"Policy Loss: {total_policy_loss/num_batches:.4f}, "
                  f"Value Loss: {total_value_loss/num_batches:.4f}, "
                  f"Entropy: {total_entropy/num_batches:.4f}, "
                  f"KL Div: {total_kl_div/num_batches:.6f}, "
                  f"Return: {np.sum(rewards):.4f}")

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
