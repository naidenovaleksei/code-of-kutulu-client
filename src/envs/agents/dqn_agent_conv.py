import numpy as np
import torch
import torch.nn as nn

from src.envs.kutulu_observer import (
    KutuluConvObserver,
)
from src.envs.buffers import (
    BaseStateEncoder,
)
from src.envs.agents.dqn_agent import (
    DQNAgentBase,
)
from src.envs.models.conv_state_model import ConvStateModel


class DQNStateEncoderConv(BaseStateEncoder):
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
            data = torch.FloatTensor(np.array(data))
        return data

    def state_rotation_augment(self, state, clockwise_dir):
        return {
            k: np.rot90(v, k=-clockwise_dir)
            for k, v in state.items()
        }


class DQNAgentConv(DQNAgentBase):
    def __init__(self, state_type, action_space_n, model_params=None, size=3, **kw):
        if model_params is None:
            model_params = {}
        super(DQNAgentConv, self).__init__(
            state_type=state_type,
            action_space_n=action_space_n,
            model=ConvStateModel(
                num_classes=action_space_n,
                size=size,
                **model_params,
            ),
            state_encoder=DQNStateEncoderConv(),
            **kw
        )
        self.size = size
