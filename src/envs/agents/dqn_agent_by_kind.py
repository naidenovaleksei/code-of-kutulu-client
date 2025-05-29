import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy

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
    Experience,
    ExperienceBuffer,
)
from src.game.template import (
    parse_state_by_kind,
    ENTITY_TOKENS,
)
from src.envs.models.dqn_model_by_kind import DQNExtByKind


class ExperienceBufferByKind(ExperienceBuffer):
    def __init__(self, capacity):
        super(ExperienceBufferByKind, self).__init__(capacity)

    def encode_states(self, states, return_tensors=True):
        data_by_kind = {}

        for state in states:
            state_by_kind = parse_state_by_kind(state)

            for kind, (kind_list, features_list, dir_list) in state_by_kind.items():
                if kind not in data_by_kind:
                    data_by_kind[kind] = {
                        'entity_kind': [],
                        'entity_features': [],
                        'entity_dir': [],
                    }
                data_by_kind[kind]['entity_kind'].append(kind_list)
                data_by_kind[kind]['entity_features'].append(features_list)
                data_by_kind[kind]['entity_dir'].append(dir_list)

        if return_tensors:
            for kind in data_by_kind:
                for key in data_by_kind[kind]:
                    data_by_kind[kind][key] = torch.tensor(data_by_kind[kind][key])
        
        return data_by_kind


class DQNAgentByKind(DQNAgent):
    def __init__(self, state_type, action_space_n,
                 lr=LEARNING_RATE, replay_size=REPLAY_SIZE,
                 replay_start_size=REPLAY_START_SIZE,
                 batch_size=BATCH_SIZE,
                 sync_target_frames=SYNC_TARGET_FRAMES,
                 epsilon_start=EPSILON_START, epsilon_final=EPSILON_FINAL,
                 epsilon_decay_last_frame=EPSILON_DECAY_LAST_FRAME,
                 gamma=GAMMA,
                 train=False, verbose=False):
        entity_kinds = ["EXPLORER", "WANDERER", "SLASHER"]
        
        super(DQNAgentByKind, self).__init__(
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
            model=DQNExtByKind(
                vocab_size=len(ENTITY_TOKENS) + 1,
                num_dirs=5,
                features_dim=8,
                embed_dim=32,
                hidden_dim=32,
                inner_dim=16,
                num_classes=action_space_n,
                entity_kinds=entity_kinds,
            ),
            episode_buffer=ExperienceBufferByKind(replay_size),
        )
