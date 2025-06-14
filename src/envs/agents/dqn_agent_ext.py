import torch
from src.envs.agents.dqn_agent import (
    DQNAgentBase,
)
from src.envs.buffers import (
    BaseStateEncoder,
)
from src.envs.models.dqn_model_ext import DQNExt
from src.game.template import (
    parse_state,
    ENTITY_TOKENS,
)


class DQNStateEncoderExt(BaseStateEncoder):
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


class DQNAgentExt(DQNAgentBase):
    def __init__(self, state_type, action_space_n, model_params=None, **kw):  
        if model_params is None:
            model_params = {}
        super(DQNAgentExt, self).__init__(
            state_type=state_type,
            action_space_n=action_space_n,
            model=DQNExt(
                vocab_size=len(ENTITY_TOKENS) + 1,
                num_classes=action_space_n,
                **model_params,
            ),
            state_encoder=DQNStateEncoderExt(),
            **kw
        )
