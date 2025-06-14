import numpy as np
import torch
import torch.nn as nn
import copy

from src.envs.agents.nn_agent import(
    NNAgent,
    GAMMA,
    LEARNING_RATE,
    EPSILON_START,
    EPSILON_FINAL,
    EPSILON_DECAY_LAST_FRAME,
)
from src.envs.models.dqn_model import DQN
from src.envs.buffers import (
    Experience,
    ExperienceBuffer,
    PrioritizedExperienceBuffer,
    BaseStateEncoder,
)

BATCH_SIZE = 32
REPLAY_SIZE = 10000
SYNC_TARGET_FRAMES = 1000
REPLAY_START_SIZE = 10000


def parse_dir(encoded_dir):
    if encoded_dir is None:
        return [0, 0, 0, 0, 1]
    result = [0, 0, 0, 0, 0]
    for _dir in encoded_dir:
        result[_dir] = 1
    return result

def parse_dist(encoded_dist):
    if encoded_dist is None:
        return [-1]
    return [encoded_dist]


class DQNStateEncoder(BaseStateEncoder):
    def encode_states(self, states, return_tensors=True):
        data = dict(
            closest_explorer_dir=[parse_dir(state[0]) for state in states],
            closest_explorer_dist=[parse_dist(state[1]) for state in states],
            closest_wanderer_dir=[parse_dir(state[2]) for state in states],
            closest_wanderer_dist=[parse_dist(state[3]) for state in states],
        )
        data = {k: torch.tensor(v) for k,v in data.items()}
        return data


class DQNAgentBase(NNAgent):
    def __init__(self, state_type, action_space_n,
                 buffer_params,
                 epsilon_params,
                 model,
                 state_encoder,
                 lr=LEARNING_RATE,
                 replay_start_size=REPLAY_START_SIZE,
                 batch_size=BATCH_SIZE,
                 sync_target_frames=SYNC_TARGET_FRAMES,
                 gamma=GAMMA,
                 train=False, verbose=False,
                 prioritized_replay=False, loss='mse'):
        if prioritized_replay:
            episode_buffer = PrioritizedExperienceBuffer(
                state_encoder, **buffer_params,
            )
        else:
            episode_buffer = ExperienceBuffer(
                state_encoder, **buffer_params,
            )
        assert len(set(epsilon_params) - set([
            'start', 'final', 'decay', 'reset', 'reset_coef',
        ])) == 0
        super(DQNAgentBase, self).__init__(
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
            episode_buffer=episode_buffer,
        )
        self.replay_start_size = replay_start_size
        self.sync_target_frames = sync_target_frames
        self.batch_size = batch_size
        self.prioritized_replay = prioritized_replay

        self.tgt_net = copy.deepcopy(self.model)
        if loss == 'mse':
            self.criterion = nn.MSELoss(reduction='none')
        elif loss == 'huber':
            self.criterion = nn.HuberLoss(reduction='none')
        else:
            raise ValueError(f'wrong loss: {loss}')


    def generate_random_step(self, actions_masked, player_mask):
        ps = np.ma.array(np.ones(self.action_space_n), mask=player_mask).filled(0)
        if ps.sum() == 0:
            return np.random.randint(self.action_space_n)
        return np.random.choice(np.arange(self.action_space_n), p=ps / ps.sum())

    def train_step(self, reward, game_over, new_state):
        if reward is None or not self.train:
            return

        state, action, observation = self.state_actions
        if new_state is None:
            new_state = state
        exp = Experience(state, action, reward, game_over, new_state, observation)
        self.episode_buffer.append(exp)

        if len(self.episode_buffer) >= self.replay_start_size:
            self._train_model()

    def _train_model(self):
        if self.frame_idx % self.sync_target_frames == 0:
            self.tgt_net.load_state_dict(self.model.state_dict())

        self.model.train()
        self.optimizer.zero_grad()
        
        # Sample from buffer - different behavior for prioritized vs. regular buffer
        if self.prioritized_replay:
            states, actions, rewards, dones, next_states, indices, weights = self.episode_buffer.sample(self.batch_size)
            # Convert weights to tensor if not already
            weights = weights.to(actions.device)
        else:
            states, actions, rewards, dones, next_states = self.episode_buffer.sample(self.batch_size)
            weights = None
        
        # Get current state-action values
        state_action_values = self.model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        
        # Get next state values
        next_state_values = self.tgt_net(next_states).max(1)[0]
        next_state_values[dones] = 0.0
        next_state_values = next_state_values.detach()

        # Calculate expected state-action values
        expected_state_action_values = next_state_values * self.gamma + rewards
        
        # Calculate loss - per sample if using prioritized replay
        losses = self.criterion(state_action_values, expected_state_action_values)
        
        # Apply importance sampling weights if using prioritized replay
        if weights is not None:
            losses = losses * weights
        
        # Calculate mean loss for backpropagation
        loss = losses.mean()
        
        # Backpropagate
        loss.backward()
        self.optimizer.step()
        
        # Update priorities if using prioritized replay
        if self.prioritized_replay:
            # Get absolute TD errors as priorities
            priorities = losses.detach().cpu().numpy()
            self.episode_buffer.update_priorities(indices, priorities)
        
        self.last_loss = loss.item()

        if self.verbose:
            print(f"Loss: {loss.item():.4f}")


class DQNAgent(DQNAgentBase):
    def __init__(self, state_type, action_space_n, model_params=None, loss='mse', **kw):  
        if model_params is None:
            model_params = {}
        super(DQNAgent, self).__init__(
            state_type=state_type,
            action_space_n=action_space_n,
            model=DQN(
                num_classes=action_space_n,
                **model_params,
            ),
            state_encoder=DQNStateEncoder(),
            **kw
        )
