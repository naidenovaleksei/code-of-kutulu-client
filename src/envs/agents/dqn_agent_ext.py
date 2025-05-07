import collections

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.envs.agents.dqn_agent import DQNAgent
from src.game.template import (
    getActionGreedyMasked2,
)
from src.envs.agents.dqn_agent import (
    ExperienceBuffer,
    Experience,
)
from src.envs.models.dqn_model_ext import DQNExt

GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000
REPLAY_START_SIZE = 10000

EPSILON_DECAY_LAST_FRAME = 10**5
EPSILON_START = 1.0
EPSILON_FINAL = 0.02


WANDERER_STATES = ["SPAWNING", "WANDERING", "STALKING", "RUSHING", "STUNNED"]
ENTITY_TOKENS = [
    "EXPLORER",
    "WANDERER_SPAWNING",
    "WANDERER_WANDERING",
    "SLASHER_SPAWNING",
    "SLASHER_WANDERING",
    "SLASHER_STALKING",
    "SLASHER_RUSHING",
    "SLASHER_STUNNED",
    "EFFECT_PLAN",
    "EFFECT_LIGHT",
    "EFFECT_SHELTER",
    "EFFECT_YELL",
]
ENTITY_TOKENS_MAP = {k: v for v, k in enumerate(ENTITY_TOKENS)}
MAX_ENTITY_COUNT = 10

def parse_dist_dir(e):
    encoded_dir = e['dir']
    if encoded_dir is None:
        return [0., 0., 0., 0., 0.]
    encoded_dist = e['dist']
    result = [0., 0., 0., 0., 0.]
    for _dir in encoded_dir:
        result[_dir] = encoded_dist
    return result

def parse_kind(e):
    kind = e['kind']
    if kind in ('WANDERER', 'SLASHER'):
        kind = f"{e['kind']}_{WANDERER_STATES[e['param1']]}"
    return ENTITY_TOKENS_MAP[kind] + 1

def parse_features(e):
    result = [
        e['param0'],
        e['param2'],
        e['rel_x'],
        e['rel_y'],
        # e['dist'] or -1,
        e['raw_dist'],
        e['on_los'],
        e['param0'],
    ]
    result = [float(x) for x in result]
    return result

def parse_state(state):
    state = state[:MAX_ENTITY_COUNT]
    kind_list = [
        parse_kind(e) for e in state
    ]
    features_list = [parse_features(e) for e in state]
    dir_list = [parse_dist_dir(e) for e in state]
    for _ in range(MAX_ENTITY_COUNT - len(state)):
        kind_list.append(0)
        features_list.append([0., 0., 0., 0., 0., 0., 0.])
        dir_list.append([0., 0., 0., 0., 0.])
    return kind_list, features_list, dir_list


class ExperienceBufferExt(ExperienceBuffer):
    def __init__(self, capacity):
        super(ExperienceBufferExt, self).__init__(capacity)

    def encode_states(self, states):
        data = dict(
            entity_kind=[],
            entity_features=[],
            entity_dir=[],
        )
        for state in states:
            kind_list, features_list, dir_list = parse_state(state)
            data['entity_kind'].append(kind_list)
            data['entity_features'].append(features_list)
            data['entity_dir'].append(dir_list)
        data = {k: torch.tensor(v) for k,v in data.items()}
        return data


class DQNAgentExt(DQNAgent):
    def __init__(self, state_type, action_space_n,
                 lr=LEARNING_RATE, replay_size=REPLAY_SIZE,
                 replay_start_size=REPLAY_START_SIZE,
                 batch_size=BATCH_SIZE,
                 sync_target_frames=SYNC_TARGET_FRAMES,
                 epsilon_start=EPSILON_START, epsilon_final=EPSILON_FINAL,
                 epsilon_decay_last_frame=EPSILON_DECAY_LAST_FRAME,
                 alpha=None, gamma=GAMMA,
                 train=False, verbose=False):
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
        self.exp_buffer = ExperienceBufferExt(replay_size)

        self.model = DQNExt(
            vocab_size=len(ENTITY_TOKENS) + 1,
            features_dim=7,
            embed_dim=32,
            hidden_dim=32,
            inner_dim=16,
            num_classes=action_space_n
        )
        self.tgt_net = DQNExt(
            vocab_size=len(ENTITY_TOKENS) + 1,
            features_dim=7,
            embed_dim=32,
            hidden_dim=32,
            inner_dim=16,
            num_classes=action_space_n
        )
    
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.last_loss = np.inf
        self.frame_idx = 0
