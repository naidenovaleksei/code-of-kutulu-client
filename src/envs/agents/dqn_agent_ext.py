import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.envs.agents.dqn_agent import DQNAgent
from src.envs.agents.dqn_agent import (
    ExperienceBuffer,
)
from src.envs.models.dqn_model_ext import DQNExt
from src.game.template import (
    parse_state,
    ENTITY_TOKENS,
)

GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000
REPLAY_START_SIZE = 10000

EPSILON_DECAY_LAST_FRAME = 10**5
EPSILON_START = 1.0
EPSILON_FINAL = 0.02

class ExperienceBufferExt(ExperienceBuffer):
    def __init__(self, capacity):
        super(ExperienceBufferExt, self).__init__(capacity)

    def encode_states(self, states, return_tensors=True):
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
        if return_tensors:
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
