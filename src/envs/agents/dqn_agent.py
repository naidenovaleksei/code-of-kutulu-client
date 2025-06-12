import collections

import numpy as np
import torch
import torch.nn as nn
import copy
import pickle as pkl

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


Experience = collections.namedtuple('Experience', field_names=[
    'state',
    'action',
    'reward',
    'done',
    'new_state',
    'observation',
])


class ExperienceBuffer:
    def __init__(self, capacity, need_aug=False):
        self.buffer = collections.deque(maxlen=capacity)
        self.need_aug = need_aug

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
        states, actions, rewards, dones, next_states, _ = zip(*[self.buffer[idx] for idx in indices])
        if self.need_aug:
            states = list(states)
            next_states = list(next_states)
            actions = list(actions)
            assert len(next_states) == batch_size
            for i in range(batch_size):
                # [+1, +2, +3]
                dir_shift = np.random.randint(1, 4)
                states[i] = self._aug_rotation_state(states[i], dir_shift)
                next_states[i] = self._aug_rotation_state(next_states[i], dir_shift)
                actions[i] = self._aug_rotation_dir(actions[i], dir_shift, ignore_errors=True)
        states = self.encode_states(states)
        next_states = self.encode_states(next_states)
        actions = torch.tensor(actions)
        rewards = torch.tensor(np.array(rewards, dtype=np.float32))
        dones = torch.BoolTensor(np.array(dones))
        return states, actions, rewards, dones, next_states

    def save_buffer(self, fname):
        with open(fname, "wb") as f:
            pkl.dump(self.buffer, f)
    
    def load_buffer(self, fname):
        with open(fname, "rb") as f:
            self.buffer = pkl.load(f)
    
    def _aug_rotation_state(self, state, dir_shift):
        return [
            self._aug_rotation_entity(e, dir_shift)
            for e in state
        ]

    def _aug_rotation_entity(self, e, dir_shift):
        new_e = dict(e)
        if e['dir'] is not None:
            new_e['dir'] = tuple(
                self._aug_rotation_dir(d, dir_shift) for d in e['dir']
            )
        theta = np.pi / 2 * dir_shift
        new_e['rel_x'] = int(np.round(e['rel_x'] * np.cos(theta) - e['rel_y'] * np.sin(theta)))
        new_e['rel_y'] = int(np.round(e['rel_x'] * np.sin(theta) + e['rel_y'] * np.cos(theta)))
        return new_e

    def _aug_rotation_dir(self, _dir, dir_shift, ignore_errors=False):
        if _dir >= 4:
            if ignore_errors or _dir == 4:
                return _dir
            raise ValueError(f'wrong dir: {_dir}')
        return (_dir + dir_shift) % 4


class DQNAgent(NNAgent):
    def __init__(self, state_type, action_space_n,
                 lr=LEARNING_RATE, replay_size=REPLAY_SIZE,
                 replay_start_size=REPLAY_START_SIZE,
                 batch_size=BATCH_SIZE,
                 need_aug=False,
                 sync_target_frames=SYNC_TARGET_FRAMES,
                 epsilon_start=EPSILON_START, epsilon_final=EPSILON_FINAL,
                 epsilon_decay_last_frame=EPSILON_DECAY_LAST_FRAME,
                 gamma=GAMMA,
                 model=None, model_params={}, episode_buffer=None,
                 train=False, verbose=False):
        if model is None:
            model = DQN(action_space_n, **model_params)
        if episode_buffer is None:
            episode_buffer = ExperienceBuffer(replay_size, need_aug)
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
        batch = self.episode_buffer.sample(self.batch_size)
        states, actions, rewards, dones, next_states = batch
        state_action_values = self.model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        next_state_values = self.tgt_net(next_states).max(1)[0]
        next_state_values[dones] = 0.0
        next_state_values = next_state_values.detach()

        # normalized_rewards = torch.sign(rewards) * torch.log1p(torch.abs(rewards))

        expected_state_action_values = next_state_values * self.gamma + rewards
        loss = self.criterion(state_action_values, expected_state_action_values)
        loss.backward()
        self.optimizer.step()
        
        self.last_loss = loss.item()

        if self.verbose:
            print(f"Loss: {loss.item():.4f}")
