import pytest
import numpy as np

from tests.utils import calculate_entities
from src.envs.agents.reinforce_agent import REINFORCEAgent
from src.game.template import (
    calculate_output_np,
)


@pytest.fixture
def explorers():
    return [(5, 3), (1, 3)]

@pytest.fixture
def wanderers():
    return [(3, 1, 1), (3, 5, 1), (3, 2, 0)]


@pytest.mark.parametrize("model_params", [
    {'out_linear_bias': True},
    {'out_linear_bias': False},
])
def test_calculate_output_np(model_params, explorers, wanderers, mock_env):
    agent = REINFORCEAgent(
        state_type='closest_ext',
        action_space_n=5,
        model_params=model_params,
    )
    player_pos = (3, 3)
    entities = calculate_entities(player_pos, explorers, wanderers)
    agent.set_env(mock_env)
    agent.observer.env._set_entities(entities)
    agent.observer.env._set_players(entities, set_ids=True)
    state = agent.observer.get_state(0)
    test_data = agent.episode_buffer.state_encoder.encode_states([state], return_tensors=False)
    tensor_data = agent.episode_buffer.state_encoder.encode_states([state], return_tensors=True)

    weights = {}
    for k,v in agent.model.named_parameters():
        weights[k] = v.detach().cpu().numpy()
    np_output = calculate_output_np(test_data, weights, agent.action_space_n, softmax=True)

    agent.model.eval()
    model_output = agent.model(tensor_data)[0].detach().cpu().numpy()

    assert np.allclose(np_output, model_output, atol=1e-3)
