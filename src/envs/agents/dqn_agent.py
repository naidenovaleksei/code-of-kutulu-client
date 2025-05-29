import collections

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


Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])


class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def encode_states(self, states):
        data = dict(
            closest_explorer_dir=[parse_dir(state[0]) for state in states],
            closest_explorer_dist=[parse_dist(state[1]) for state in states],
            closest_wanderer_dir=[parse_dir(state[2]) for state in states],
            closest_wanderer_dist=[parse_dist(state[3]) for state in states],
        )
        data = {k: torch.tensor(v) for k,v in data.items()}
        return data

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])

        states = self.encode_states(states)
        next_states = self.encode_states(next_states)
        actions = torch.tensor(actions)
        rewards = torch.tensor(np.array(rewards, dtype=np.float32))
        dones = torch.ByteTensor(np.array(dones, dtype=np.uint8))
        return states, actions, rewards, dones, next_states


class DQNAgent(NNAgent):
    def __init__(self, state_type, action_space_n,
                 lr=LEARNING_RATE, replay_size=REPLAY_SIZE,
                 replay_start_size=REPLAY_START_SIZE,
                 batch_size=BATCH_SIZE,
                 sync_target_frames=SYNC_TARGET_FRAMES,
                 epsilon_start=EPSILON_START, epsilon_final=EPSILON_FINAL,
                 epsilon_decay_last_frame=EPSILON_DECAY_LAST_FRAME,
                 gamma=GAMMA,
                 model=None, model_params={}, episode_buffer=None,
                 train=False, verbose=False):
        if model is None:
            model = DQN(action_space_n, **model_params)
        if episode_buffer is None:
            episode_buffer = ExperienceBuffer(replay_size)
        super(DQNAgent, self).__init__(
            state_type=state_type,
            action_space_n=action_space_n,
            lr=lr,
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_final=epsilon_final,
            epsilon_decay_last_frame=epsilon_decay_last_frame,
            train=train,
            verbose=verbose,
            model=model,
            episode_buffer=episode_buffer,
        )
        self.replay_start_size = replay_start_size
        self.sync_target_frames = sync_target_frames
        self.batch_size = batch_size

        self.tgt_net = copy.deepcopy(self.model)
        self.criterion = nn.MSELoss()

    def generate_random_step(self, actions_masked, player_mask):
        ps = np.ma.array(np.ones(self.action_space_n), mask=player_mask).filled(0)
        if ps.sum() == 0:
            return np.random.randint(self.action_space_n)
        return np.random.choice(np.arange(self.action_space_n), p=ps / ps.sum())

    def train_step(self, reward, game_over, new_state=None):
        if reward is None or not self.train:
            return

        state, action = self.state_actions
        exp = Experience(state, action, reward / 100, game_over, new_state)
        self.episode_buffer.append(exp)

        if len(self.episode_buffer) >= self.replay_start_size:
            self._train_model()

    def _train_model(self):
        if self.frame_idx % self.sync_target_frames == 0:
            self.tgt_net.load_state_dict(self.model.state_dict())

        self.model.train()
        self.optimizer.zero_grad()
        batch = self.episode_buffer.sample(self.batch_size)
        states, actions, rewards, dones, next_states = batch
        state_action_values = self.model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        next_state_values = self.tgt_net(next_states).max(1)[0]
        # next_state_values[dones] = 0.0
        next_state_values = next_state_values.detach()

        normalized_rewards = torch.sign(rewards) * torch.log1p(torch.abs(rewards))

        expected_state_action_values = next_state_values * self.gamma + normalized_rewards
        loss = self.criterion(state_action_values, expected_state_action_values)
        loss.backward()
        self.optimizer.step()
        
        self.last_loss = loss.item()

        if self.verbose:
            print(f"Loss: {loss.item():.4f}")
