import logging

import numpy as np
import torch
import torch.nn as nn

from src.envs.buffers import (
    BaseStateEncoder,
)
from src.envs.agents.dqn_agent import (
    DQNAgentBase,
)
from src.envs.models.conv_state_model import ConvStateModel, DuelingConvStateModel
from src.envs.models.conv_state_by_kind_model import ConvStateByKindModel, DuelingConvStateByKindModel

logger = logging.getLogger(__name__)


class DQNStateEncoderConv(BaseStateEncoder):
    def __init__(self, is_ext=False):
        self.layer_keys = [
            'map',
            'EXPLORER_param0', 'EXPLORER_param1', 'EXPLORER_param2',
            'WANDERER_param0', 'WANDERER_param1',
            'SLASHER_param0', 'SLASHER_param1',
            'EFFECT_PLAN_param0',
            'EFFECT_LIGHT_param0',
            'EFFECT_SHELTER_param0',
            'EFFECT_YELL_param0'
        ]
        if is_ext:
            self.layer_keys = [
                'map',
                'EXPLORER_param0', 'EXPLORER_param1', 'EXPLORER_param2',
                'WANDERER_param0', #'WANDERER_param1',
                # 'SLASHER_param0', 'SLASHER_param1',
                # 'EFFECT_PLAN_param0',
                # 'EFFECT_LIGHT_param0',
                'EFFECT_SHELTER_param0',
                # 'EFFECT_YELL_param0',
                'EXPLORER_COUNT',
                'EXPLORER_MIN_DIST',
                'WANDERER_COUNT',
                'WANDERER_MIN_DIST',
                'WANDERER_SPAWNING',
                'SLASHER_COUNT',
                'SLASHER_STALKING',
                'SLASHER_WANDERING',
                'SLASHER_SPAWNING',
                'SLASHER_STUNNED',
                'EFFECT_LIGHT',
                'EFFECT_PLAN',
                'EXPLORER_param0_border',
                'WANDERER_param0_border',
                'SLASHER_param0_border',
                'EFFECT_SHELTER_param0_border',
                # 'SLASHER_TIME_TO_LAND'
            ]
    
    def layers_count(self):
        return len(self.layer_keys)

    def encode_states(self, states, return_tensors=True):
        data = []
        for state in states:
            features = []
            for k in self.layer_keys:
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
    def __init__(self, state_type, action_space_n, model_params=None, size=3, dueling=False, **kw):
        if model_params is None:
            model_params = {}
        
        # Choose model based on state_type and dueling parameter
        if state_type == 'conv_by_kind':
            if dueling:
                model = DuelingConvStateByKindModel(
                    num_classes=action_space_n,
                    size=size,
                    **model_params,
                )
            else:
                model = ConvStateByKindModel(
                    num_classes=action_space_n,
                    size=size,
                    **model_params,
                )
        elif state_type == 'conv':
            if dueling:
                model = DuelingConvStateModel(
                    num_classes=action_space_n,
                    size=size,
                    **model_params,
                )
            else:
                model = ConvStateModel(
                    num_classes=action_space_n,
                    size=size,
                    **model_params,
                )
        else:
            raise ValueError(f'unknown state_type for DQNAgentConv: {state_type}')
        
        super(DQNAgentConv, self).__init__(
            state_type=state_type,
            action_space_n=action_space_n,
            model=model,
            state_encoder=DQNStateEncoderConv(),
            **kw
        )
        self.size = size
        self.dueling = dueling
        
        model_type = "Dueling " if dueling else "Standard "
        model_arch = "ConvStateByKindModel" if state_type == 'conv_by_kind' else "ConvStateModel"
        logger.info("Initialized %s%s agent", model_type, model_arch)
