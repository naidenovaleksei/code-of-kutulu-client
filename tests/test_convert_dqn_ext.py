import pytest
import numpy as np

from tests.utils import calculate_entities
from src.envs.agents.dqn_agent_ext import DQNAgentExt
from src.envs.agents.dqn_agent_by_kind import DQNAgentByKind
from src.game.template import (
    calculate_output_np,
    DEFAULT_KUTULU_ACTIONS,
    DQNSolver,
    DQNByKindSolver,
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
    agent = DQNAgentExt(
        state_type='closest_ext',
        action_space_n=5,
        model_params=model_params,
        buffer_params={'capacity': 10000},
        epsilon_params={'start': 1.0, 'final': 0.01, 'decay': 10000},
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
    np_output = calculate_output_np(test_data, weights, agent.action_space_n)

    model_output = agent.model(tensor_data)[0].detach().cpu().numpy()

    assert np.allclose(np_output, model_output)


class TestDQNAgentExt:
    @pytest.mark.parametrize("model_params", [
        {'out_linear_bias': True},
        {'out_linear_bias': False},
    ])
    def test_convert_qdn_ext(self, model_params, explorers, wanderers, mock_env):
        agent = DQNAgentExt(
            state_type='closest_ext',
            action_space_n=5,
            model_params=model_params,
            buffer_params={'capacity': 10000},
            epsilon_params={'start': 1.0, 'final': 0.01, 'decay': 10000},
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

        info = {
            'width': agent.observer.env.width,
            'height': agent.observer.env.height,
            'lines': agent.observer.env.map,
        }

        solver = DQNSolver(info, DEFAULT_KUTULU_ACTIONS, weights)
        np_output = solver.calculate_output(agent.observer.env._get_entites(0), player_pos)

        agent.model.eval()
        model_output = agent.model(tensor_data)[0].detach().cpu().numpy()

        assert np.allclose(np_output, model_output, atol=1e-3)


class TestDQNAgentByKind:
    @pytest.mark.parametrize("model_params", [
        {'out_linear_bias': True},
        {'out_linear_bias': False},
    ])
    def test_convert_qdn_by_kind(self, model_params, explorers, wanderers, mock_env):
        agent = DQNAgentByKind(
            state_type='closest_ext',
            action_space_n=5,
            model_params=model_params,
            buffer_params={'capacity': 10000},
            epsilon_params={'start': 1.0, 'final': 0.01, 'decay': 10000},
        )
        player_pos = (3, 3)
        entities = calculate_entities(player_pos, explorers, wanderers)
        agent.set_env(mock_env)
        agent.observer.env._set_entities(entities)
        agent.observer.env._set_players(entities, set_ids=True)
        state = agent.observer.get_state(0)
        tensor_data = agent.episode_buffer.state_encoder.encode_states([state], return_tensors=True)

        weights = {}
        for k,v in agent.model.named_parameters():
            weights[k] = v.detach().cpu().numpy()
        
        info = {
            'width': agent.observer.env.width,
            'height': agent.observer.env.height,
            'lines': agent.observer.env.map,
        }

        solver = DQNByKindSolver(info, weights)
        np_output = solver.calculate_output(agent.observer.env._get_entites(0), player_pos)

        model_output = agent.model(tensor_data)[0].detach().cpu().numpy()

        assert np.allclose(np_output, model_output)
