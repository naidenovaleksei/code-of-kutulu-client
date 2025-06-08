import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


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
from src.envs.models.dqn_model_ext import DQNExt
from src.game.template import (
    parse_state,
    ENTITY_TOKENS,
)


class ExperienceBufferExt(ExperienceBuffer):
    def __init__(self, capacity, need_aug=False):
        super(ExperienceBufferExt, self).__init__(capacity, need_aug)

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
                 gamma=GAMMA, model_params={},
                 train=False, verbose=False):
        super(DQNAgentExt, self).__init__(
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
            model=DQNExt(
                vocab_size=len(ENTITY_TOKENS) + 1,
                num_classes=action_space_n,
                **model_params,
            ),
            episode_buffer=ExperienceBufferExt(replay_size),
        )
