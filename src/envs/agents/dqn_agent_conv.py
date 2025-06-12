import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict

from src.envs.kutulu_observer import (
    KutuluConvObserver,
)


from src.envs.agents.nn_agent import(
    GAMMA,
    LEARNING_RATE,
    EPSILON_START,
    EPSILON_FINAL,
    EPSILON_DECAY_LAST_FRAME,
)
from src.envs.agents.dqn_agent import (
    DQNAgent,
    BATCH_SIZE,
    REPLAY_SIZE,
    SYNC_TARGET_FRAMES,
    REPLAY_START_SIZE,
)
from src.envs.agents.dqn_agent import (
    ExperienceBuffer,
)
from src.envs.models.dqn_model_conv import DQNConv
from src.game.template import (
    parse_state,
    ENTITY_TOKENS,
)


class ExperienceBufferConv(ExperienceBuffer):
    def __init__(self, capacity, need_aug=False):
        super(ExperienceBufferConv, self).__init__(capacity, need_aug)

    def encode_states(self, states, return_tensors=True):
        data = []
        for state in states:
            features = []
            for k in [
                'map',
                'EXPLORER_param0', 'EXPLORER_param1', 'EXPLORER_param2',
                'WANDERER_param0', 'WANDERER_param1',
                'SLASHER_param0', 'SLASHER_param1',
                'EFFECT_PLAN_param0',
                'EFFECT_LIGHT_param0',
                'EFFECT_SHELTER_param0',
                'EFFECT_YELL_param0'
                ]:
                features.append(state[k])
            data.append(features)
        if return_tensors:
            data = torch.FloatTensor(data)
        return data


class DQNAgentConv(DQNAgent):
    def __init__(self, state_type, action_space_n,
                 lr=LEARNING_RATE, replay_size=REPLAY_SIZE,
                 replay_start_size=REPLAY_START_SIZE,
                 batch_size=BATCH_SIZE,
                 sync_target_frames=SYNC_TARGET_FRAMES,
                 epsilon_start=EPSILON_START, epsilon_final=EPSILON_FINAL,
                 epsilon_decay_last_frame=EPSILON_DECAY_LAST_FRAME,
                 gamma=GAMMA, model_params={}, size=3,
                 train=False, verbose=False,
                 loss='mse'):
        super(DQNAgentConv, self).__init__(
            state_type=state_type,
            action_space_n=action_space_n,
            lr=lr,
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_final=epsilon_final,
            epsilon_decay_last_frame=epsilon_decay_last_frame,
            train=train,
            verbose=verbose,
            replay_start_size=replay_start_size,
            sync_target_frames=sync_target_frames,
            batch_size=batch_size,
            model=DQNConv(
                num_classes=action_space_n,
                size=size,
                **model_params,
            ),
            episode_buffer=ExperienceBufferConv(replay_size),
        )
        self.size = size
        if loss == 'mse':
            self.criterion = nn.MSELoss()
        elif loss == 'huber':
            self.criterion = nn.HuberLoss()
        else:
            raise ValueError(f'wrong loss: {loss}')

    def set_env(self, env):
        assert self.state_type == 'conv'
        self.observer = KutuluConvObserver(env, self.size)
