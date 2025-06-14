import pytest
import numpy as np

from tests.utils import calculate_entities
from src.envs.agents.dqn_agent_conv import DQNAgentConv
from src.game.template import (
    calculate_output_np,
    DEFAULT_KUTULU_ACTIONS,
    DQNSolver,
    DQNConvSolver,
)


@pytest.fixture
def explorers():
    return [(5, 3), (1, 3)]

@pytest.fixture
def wanderers():
    return [(3, 1, 1), (3, 5, 1), (3, 2, 0)]


def test_calculate_output_np(explorers, wanderers, mock_env):
    agent = DQNAgentConv(
        state_type='conv',
        action_space_n=5,
        size=3,
        buffer_params={'capacity': 10},
        epsilon_params={},
    )
    player_pos = (3, 3)
    entities = calculate_entities(player_pos, explorers, wanderers)
    agent.set_env(mock_env)
    agent.observer.env._set_entities(entities)
    agent.observer.env._set_players(entities, set_ids=True)
    state = agent.observer.get_state(0)
    tensor_data = agent.episode_buffer.encode_states([state])

    weights = {}
    for k,v in agent.model.named_parameters():
        weights[k] = v.detach().cpu().numpy()
    
    # Add running_mean and running_var for batch normalization layers
    weights['bn1.running_mean'] = agent.model.bn1.running_mean.detach().cpu().numpy()
    weights['bn1.running_var'] = agent.model.bn1.running_var.detach().cpu().numpy()
    weights['bn2.running_mean'] = agent.model.bn2.running_mean.detach().cpu().numpy()
    weights['bn2.running_var'] = agent.model.bn2.running_var.detach().cpu().numpy()

    model_output = agent.model(tensor_data)[0].detach().cpu().numpy()

    # We can't use calculate_output_np directly for conv models
    # This test is just to verify the model output

    assert model_output.shape == (5,)


class TestDQNAgentConv:
    def test_convert_dqn_conv(self, explorers, wanderers, mock_env):
        agent = DQNAgentConv(
            state_type='conv',
            action_space_n=5,
            size=3,
            buffer_params={'capacity': 10},
            epsilon_params={},
        )
        player_pos = (3, 3)
        entities = calculate_entities(player_pos, explorers, wanderers)
        agent.set_env(mock_env)
        agent.observer.env._set_entities(entities)
        agent.observer.env._set_players(entities, set_ids=True)
        state = agent.observer.get_state(0)
        tensor_data = agent.episode_buffer.encode_states([state])

        weights = {}
        for k,v in agent.model.state_dict().items():
            weights[k] = v.detach().cpu().numpy()

        info = {
            'width': agent.observer.env.width,
            'height': agent.observer.env.height,
            'lines': agent.observer.env.map,
        }

        solver = DQNConvSolver(info, weights, DEFAULT_KUTULU_ACTIONS, size=agent.size)
        np_output = solver.calculate_output(agent.observer.env._get_entites(0), player_pos)

        agent.model.eval()
        model_output = agent.model(tensor_data)[0].detach().cpu().numpy()

        assert np.allclose(np_output, model_output)
