import torch
from src.envs.buffers import (
    BaseStateEncoder,
)
from src.envs.agents.dqn_agent import (
    DQNAgent,
)
from src.game.template import (
    parse_state_by_kind,
    ENTITY_TOKENS,
)
from src.envs.models.dqn_model_by_kind import DQNExtByKind


class DQNStateEncoderByKind(BaseStateEncoder):
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
    def __init__(self, state_type, action_space_n, model_params=None, **kw):
        if model_params is None:
            model_params = {}
        super(DQNAgentByKind, self).__init__(
            state_type=state_type,
            action_space_n=action_space_n,
            model=DQNExtByKind(
                vocab_size=len(ENTITY_TOKENS) + 1,
                num_classes=action_space_n,
                **model_params,
            ),
            state_encoder=DQNStateEncoderByKind(),
            **kw
        )
