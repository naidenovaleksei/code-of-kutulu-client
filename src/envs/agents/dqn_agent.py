import collections

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.envs.agents import BaseAgent
from src.game.template import (
    getActionGreedyMasked2,
)
# from src.envs.agents.cross_entropy_agent import (
#     encode_states,
# )
from src.envs.models.dqn_model import DQN

GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000
REPLAY_START_SIZE = 10000

EPSILON_DECAY_LAST_FRAME = 10**5
EPSILON_START = 1.0
EPSILON_FINAL = 0.02


def parse_dir(encoded_dir):
    if encoded_dir is None:
        return [0, 0, 0, 0, 1]
    result = [0, 0, 0, 0, 0]
    for _dir in encoded_dir:
        result[_dir] = 1
    return result


# def parse_dir(encoded_dir):
#     if encoded_dir is None:
#         return [0, 0, 0, 0, 0, 0]
#     # return list(encoded_dir) + [0] *
#     return [x + 1 for x in encoded_dir] + [0] * (6 - len(encoded_dir))

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

        states = self.encode_states(np.array(states))
        next_states = self.encode_states(np.array(next_states))
        actions = torch.tensor(actions)
        rewards = torch.tensor(np.array(rewards, dtype=np.float32))
        dones = torch.ByteTensor(np.array(dones, dtype=np.uint8))
        return states, actions, rewards, dones, next_states


class DQNAgent(BaseAgent):
    def __init__(self, state_type, action_space_n,
                 lr=LEARNING_RATE, replay_size=REPLAY_SIZE,
                 replay_start_size=REPLAY_START_SIZE,
                 batch_size=BATCH_SIZE,
                 sync_target_frames=SYNC_TARGET_FRAMES,
                 epsilon_start=EPSILON_START, epsilon_final=EPSILON_FINAL,
                 epsilon_decay_last_frame=EPSILON_DECAY_LAST_FRAME,
                 alpha=None, gamma=GAMMA,
                 train=False, verbose=False):
        # super(DQNAgent, self).__init__()
        self.state_type = state_type
        self.action_space_n = action_space_n
        self.eps = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_start = epsilon_start
        self.epsilon_decay_last_frame = epsilon_decay_last_frame
        self.replay_start_size = replay_start_size
        self.sync_target_frames = sync_target_frames
        self.batch_size = batch_size
        self.alpha = alpha
        self.gamma = gamma
        self.train = train
        self.verbose = verbose
        self.state_actions = None
        self.exp_buffer = ExperienceBuffer(replay_size)

        self.model = DQN(
            vocab_size=6 + 1,
            embed_dim=32,
            hidden_dim=32,
            num_classes=action_space_n
        )
        self.tgt_net = DQN(
            vocab_size=6 + 1,
            embed_dim=32,
            hidden_dim=32,
            num_classes=action_space_n
        )
    
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.last_loss = np.inf
        self.frame_idx = 0

    def generate_state_and_step(self, observer, player_id):
        self.frame_idx += 1
        self.eps = max(
            self.epsilon_final, self.epsilon_start - self.frame_idx / self.epsilon_decay_last_frame)

        state = observer.get_state(player_id, self.state_type)
        valid_actions = observer.env.get_valid_action_mask()[player_id]
        player_mask = ~np.array(valid_actions)

        if self.train and np.random.random() < self.eps:
            action = getActionGreedyMasked2(state, {}, self.action_space_n, player_mask)
        else:
            data = self.exp_buffer.encode_states([state])
            # data['mask'] = torch.tensor([player_mask])
            model_output = self.model(data)[0].detach().cpu().numpy()
            q_vals_v = np.ma.array(model_output, mask=player_mask)
            action = q_vals_v.argmax()

        self.state_actions = (state, action)
        return state, action

    def train_step(self, reward, game_over, new_state=None):
        # if reward is None or game_over or not self.train:
        if reward is None or not self.train:
            return

        state, action = self.state_actions
        exp = Experience(state, action, reward / 100, game_over, new_state)
        self.exp_buffer.append(exp)

        if len(self.exp_buffer) >= self.replay_start_size:
            self.train_model()

    def check_policy(self):
#         return dict(model.named_parameters())['output.weight'].detach().cpu().numpy().std()
        return self.last_loss

    def train_model(self):
        if self.frame_idx % self.sync_target_frames == 0:
            self.tgt_net.load_state_dict(self.model.state_dict())

        self.model.train()
        self.optimizer.zero_grad()
        batch = self.exp_buffer.sample(self.batch_size)
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
