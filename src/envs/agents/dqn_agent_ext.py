from copy import deepcopy
import torch
import numpy as np
from src.envs.agents.dqn_agent import (
    DQNAgentBase,
)
from src.envs.buffers import (
    BaseStateEncoder,
)
from src.envs.models.ext_state_model import ExtStateModel
from src.game.template import (
    parse_state,
    parse_state_v2,
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


def listdict_to_dictnp(list_of_dicts, return_tensors):
    if not list_of_dicts:
        return {}

    def recurse(values):
        # all values are from the same key across items
        first = values[0]
        if isinstance(first, dict):
            # nested dict -> recurse per key
            return {k: recurse([v[k] for v in values]) for k in first.keys()}
        else:
            # assume scalar, list, or np.array -> convert to np.array and stack
            if return_tensors:
                return torch.stack([torch.tensor(v, dtype=torch.float32) for v in values])
            return np.stack([np.array(v) for v in values])

    return {k: recurse([d[k] for d in list_of_dicts]) for k in list_of_dicts[0].keys()}


class DQNStateEncoderExtv2(BaseStateEncoder):
    def encode_states(self, states, return_tensors=True):
        data = listdict_to_dictnp([parse_state_v2(state) for state in states], return_tensors)
        return data

    def _replace_dir(self, state, clockwise_dir):
        for k in state:
            if 'dir' in k and len(state[k]) > 0 and state[k][0] < 4:
                state[k] = tuple(sorted([
                    self.action_rotation_augment(x, clockwise_dir) for x in state[k]
                ]))
            elif isinstance(state[k], dict):
                state[k] = self._replace_dir(state[k], clockwise_dir)
            elif isinstance(state[k], list):
                state[k] = [self._replace_dir(x, clockwise_dir) for x in state[k]]
        return state

    def state_rotation_augment(self, state, clockwise_dir):
        state2 = deepcopy(state)
        return self._replace_dir(state2, clockwise_dir)


class DQNAgentExt(DQNAgentBase):
    def __init__(self, state_type, action_space_n, model_params=None, **kw):  
        if model_params is None:
            model_params = {}
        super(DQNAgentExt, self).__init__(
            state_type=state_type,
            action_space_n=action_space_n,
            model=ExtStateModel(
                vocab_size=len(ENTITY_TOKENS) + 1,
                num_classes=action_space_n,
                **model_params,
            ),
            state_encoder=DQNStateEncoderExt(),
            **kw
        )
