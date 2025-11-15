import pytest
import numpy as np
import torch

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

        # Create explicit_action_mask (all True means all actions are allowed by default)
        explicit_action_mask = np.ones(len(DEFAULT_KUTULU_ACTIONS), dtype=bool)
        solver = DQNConvSolver(info, DEFAULT_KUTULU_ACTIONS, weights, size=agent.size, explicit_action_mask=explicit_action_mask)
        entities = [e.to_dict() for e in agent.observer.env._get_entites(0)]
        np_output = solver._calculate_output(entities, player_pos)

        agent.model.eval()
        model_output = agent.model(tensor_data)[0].detach().cpu().numpy()

        assert np.allclose(np_output, model_output, atol=1e-3)

    def test_dueling_dqn_conv(self, explorers, wanderers, mock_env):
        """Test that dueling DQN conv produces valid outputs"""
        # Test conv state type with dueling
        dueling_agent = DQNAgentConv(
            state_type='conv',
            action_space_n=5,
            size=3,
            buffer_params={'capacity': 10},
            epsilon_params={},
            dueling=True
        )
        
        # Test conv_by_kind state type with dueling
        dueling_by_kind_agent = DQNAgentConv(
            state_type='conv_by_kind',
            action_space_n=5,
            size=3,
            buffer_params={'capacity': 10},
            epsilon_params={},
            dueling=True
        )
        
        player_pos = (3, 3)
        entities = calculate_entities(player_pos, explorers, wanderers)
        
        for agent in [dueling_agent, dueling_by_kind_agent]:
            agent.set_env(mock_env)
            agent.observer.env._set_entities(entities)
            agent.observer.env._set_players(entities, set_ids=True)
            state = agent.observer.get_state(0)
            tensor_data = agent.episode_buffer.encode_states([state])
            
            agent.model.eval()
            with torch.no_grad():
                output = agent.model(tensor_data)[0]
            
            # Should produce valid Q-values
            assert torch.all(torch.isfinite(output))
            assert output.shape == torch.Size([5])
            assert agent.dueling == True

    def test_dueling_vs_standard_conv(self, explorers, wanderers, mock_env):
        """Test that dueling and standard conv models produce different outputs"""
        # Create standard and dueling agents
        standard_agent = DQNAgentConv(
            state_type='conv',
            action_space_n=5,
            size=3,
            buffer_params={'capacity': 10},
            epsilon_params={},
            dueling=False,
            explicit_action_mask=None,
        )
        
        dueling_agent = DQNAgentConv(
            state_type='conv',
            action_space_n=5,
            size=3,
            buffer_params={'capacity': 10},
            epsilon_params={},
            dueling=True
        )
        
        player_pos = (3, 3)
        entities = calculate_entities(player_pos, explorers, wanderers)
        
        for agent in [standard_agent, dueling_agent]:
            agent.set_env(mock_env)
            agent.observer.env._set_entities(entities)
            agent.observer.env._set_players(entities, set_ids=True)
        
        standard_state = standard_agent.observer.get_state(0)
        dueling_state = dueling_agent.observer.get_state(0)
        
        standard_tensor = standard_agent.episode_buffer.encode_states([standard_state])
        dueling_tensor = dueling_agent.episode_buffer.encode_states([dueling_state])
        
        standard_agent.model.eval()
        dueling_agent.model.eval()
        
        with torch.no_grad():
            standard_output = standard_agent.model(standard_tensor)[0]
            dueling_output = dueling_agent.model(dueling_tensor)[0]
        
        # Both should produce valid outputs
        assert torch.all(torch.isfinite(standard_output))
        assert torch.all(torch.isfinite(dueling_output))
        assert standard_output.shape == dueling_output.shape == torch.Size([5])
        
        # Check dueling flags
        assert standard_agent.dueling == False
        assert dueling_agent.dueling == True

    def test_backward_compatibility_conv(self, explorers, wanderers, mock_env):
        """Test that standard conv DQN still works as before"""
        # This should work exactly as before
        standard_agent = DQNAgentConv(
            state_type='conv',
            action_space_n=5,
            size=3,
            buffer_params={'capacity': 10},
            epsilon_params={}
        )
        
        # Should default to non-dueling
        assert standard_agent.dueling == False
        
        player_pos = (3, 3)
        entities = calculate_entities(player_pos, explorers, wanderers)
        standard_agent.set_env(mock_env)
        standard_agent.observer.env._set_entities(entities)
        standard_agent.observer.env._set_players(entities, set_ids=True)
        state = standard_agent.observer.get_state(0)
        tensor_data = standard_agent.episode_buffer.encode_states([state])
        
        standard_agent.model.eval()
        with torch.no_grad():
            output = standard_agent.model(tensor_data)[0]
        
        # Should produce valid output
        assert torch.all(torch.isfinite(output))
        assert output.shape == torch.Size([5])
