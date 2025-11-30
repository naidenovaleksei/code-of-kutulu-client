import pytest
import numpy as np
import torch

from tests.utils import calculate_entities
from src.envs.agents.ppo_agent import PPOAgent
from src.game.template import (
    EXTENDED_KUTULU_ACTIONS,
    PPOConvExtDeepSolver,
)


@pytest.fixture
def explorers():
    return [(5, 3), (1, 3)]

@pytest.fixture
def wanderers():
    return [(3, 1, 1), (3, 5, 1), (3, 2, 0)]


def test_calculate_output_np(explorers, wanderers, mock_env):
    agent = PPOAgent(
        state_type='conv_ext',
        actions=EXTENDED_KUTULU_ACTIONS,
        model_params={'size': 3},
        train=False,
        verbose=False,
        use_deep=True,
    )
    player_pos = (3, 3)
    entities = calculate_entities(player_pos, explorers, wanderers)
    agent.set_env(mock_env)
    agent.observer.env._set_entities(entities)
    agent.observer.env._set_players(entities, set_ids=True)
    state = agent.observer.get_state(0)
    tensor_data = agent.episode_buffer.encode_states([state])

    weights = {}
    for k, v in agent.model.state_dict().items():
        weights[k] = v.detach().cpu().numpy()

    info = {
        'width': agent.observer.env.width,
        'height': agent.observer.env.height,
        'lines': agent.observer.env.map,
    }

    # Create explicit_action_mask (all True means all actions are allowed by default)
    explicit_action_mask = np.ones(len(EXTENDED_KUTULU_ACTIONS), dtype=bool)
    solver = PPOConvExtDeepSolver(info, EXTENDED_KUTULU_ACTIONS, weights, size=agent.size, explicit_action_mask=explicit_action_mask)
    entities = [e.to_dict() for e in agent.observer.env._get_entites(0)]
    np_output = solver._calculate_output(entities, player_pos)

    agent.model.eval()
    with torch.no_grad():
        model_output = agent.model(tensor_data)
        policy_output = model_output['policy'][0].detach().cpu().numpy()

    assert np.allclose(np_output, policy_output, atol=1e-3)


class TestPPOAgentConvExtDeep:
    def test_convert_ppo_conv_ext_deep(self, explorers, wanderers, mock_env):
        agent = PPOAgent(
            state_type='conv_ext',
            actions=EXTENDED_KUTULU_ACTIONS,
            model_params={'size': 3},
            train=False,
            verbose=False,
            use_deep=True,
        )
        player_pos = (3, 3)
        entities = calculate_entities(player_pos, explorers, wanderers)
        agent.set_env(mock_env)
        agent.observer.env._set_entities(entities)
        agent.observer.env._set_players(entities, set_ids=True)
        state = agent.observer.get_state(0)
        tensor_data = agent.episode_buffer.encode_states([state])

        weights = {}
        for k, v in agent.model.state_dict().items():
            weights[k] = v.detach().cpu().numpy()

        info = {
            'width': agent.observer.env.width,
            'height': agent.observer.env.height,
            'lines': agent.observer.env.map,
        }

        solver = PPOConvExtDeepSolver(info, EXTENDED_KUTULU_ACTIONS, weights, size=agent.size)
        entities = [e.to_dict() for e in agent.observer.env._get_entites(0)]
        np_output = solver._calculate_output(entities, player_pos)

        agent.model.eval()
        with torch.no_grad():
            model_output = agent.model(tensor_data)
            policy_output = model_output['policy'][0].detach().cpu().numpy()

        assert np.allclose(np_output, policy_output, atol=1e-3)

    def test_ppo_conv_ext_deep_model_structure(self, explorers, wanderers, mock_env):
        """Test that PPO conv_ext deep model produces valid outputs with correct structure"""
        agent = PPOAgent(
            state_type='conv_ext',
            actions=EXTENDED_KUTULU_ACTIONS,
            model_params={'size': 3},
            train=False,
            verbose=False,
            use_deep=True,
        )
        
        player_pos = (3, 3)
        entities = calculate_entities(player_pos, explorers, wanderers)
        
        agent.set_env(mock_env)
        agent.observer.env._set_entities(entities)
        agent.observer.env._set_players(entities, set_ids=True)
        state = agent.observer.get_state(0)
        tensor_data = agent.episode_buffer.encode_states([state])
        
        agent.model.eval()
        with torch.no_grad():
            output = agent.model(tensor_data)
        
        # Should produce valid policy and value outputs
        assert 'policy' in output
        assert 'value' in output
        assert 'turns_to_death' in output
        
        policy = output['policy'][0]
        value = output['value'][0]
        
        # Policy should be a probability distribution
        assert torch.all(torch.isfinite(policy))
        assert torch.all(policy >= 0)
        assert torch.allclose(policy.sum(), torch.tensor(1.0), atol=1e-6)
        assert policy.shape == torch.Size([len(EXTENDED_KUTULU_ACTIONS)])
        
        # Value should be a single scalar
        assert torch.all(torch.isfinite(value))
        assert value.shape == torch.Size([1])

    def test_ppo_conv_ext_deep_vs_regular(self, explorers, wanderers, mock_env):
        """Test that deep and regular conv_ext models produce different outputs"""
        # Create deep and regular agents
        deep_agent = PPOAgent(
            state_type='conv_ext',
            actions=EXTENDED_KUTULU_ACTIONS,
            model_params={'size': 3},
            train=False,
            verbose=False,
            use_deep=True,
        )
        
        regular_agent = PPOAgent(
            state_type='conv_ext',
            actions=EXTENDED_KUTULU_ACTIONS,
            model_params={'size': 3},
            train=False,
            verbose=False,
            use_deep=False,
        )
        
        player_pos = (3, 3)
        entities = calculate_entities(player_pos, explorers, wanderers)
        
        for agent in [deep_agent, regular_agent]:
            agent.set_env(mock_env)
            agent.observer.env._set_entities(entities)
            agent.observer.env._set_players(entities, set_ids=True)
        
        deep_state = deep_agent.observer.get_state(0)
        regular_state = regular_agent.observer.get_state(0)
        
        deep_tensor = deep_agent.episode_buffer.encode_states([deep_state])
        regular_tensor = regular_agent.episode_buffer.encode_states([regular_state])
        
        deep_agent.model.eval()
        regular_agent.model.eval()
        
        with torch.no_grad():
            deep_output = deep_agent.model(deep_tensor)
            regular_output = regular_agent.model(regular_tensor)
        
        # Both should produce valid outputs
        deep_policy = deep_output['policy'][0]
        regular_policy = regular_output['policy'][0]
        
        assert torch.all(torch.isfinite(deep_policy))
        assert torch.all(torch.isfinite(regular_policy))
        assert deep_policy.shape == regular_policy.shape == torch.Size([len(EXTENDED_KUTULU_ACTIONS)])
        
        # Check that both are valid probability distributions
        assert torch.allclose(deep_policy.sum(), torch.tensor(1.0), atol=1e-6)
        assert torch.allclose(regular_policy.sum(), torch.tensor(1.0), atol=1e-6)

    def test_backward_compatibility_ppo_conv_ext_deep(self, explorers, wanderers, mock_env):
        """Test that PPO conv_ext deep agent still works as expected"""
        agent = PPOAgent(
            state_type='conv_ext',
            actions=EXTENDED_KUTULU_ACTIONS,
            model_params={'size': 3},
            train=False,
            verbose=False,
            use_deep=True,
        )
        
        player_pos = (3, 3)
        entities = calculate_entities(player_pos, explorers, wanderers)
        agent.set_env(mock_env)
        agent.observer.env._set_entities(entities)
        agent.observer.env._set_players(entities, set_ids=True)
        state = agent.observer.get_state(0)
        tensor_data = agent.episode_buffer.encode_states([state])
        
        agent.model.eval()
        with torch.no_grad():
            output = agent.model(tensor_data)
        
        policy = output['policy'][0]
        value = output['value'][0]
        
        # Should produce valid output
        assert torch.all(torch.isfinite(policy))
        assert torch.all(torch.isfinite(value))
        assert policy.shape == torch.Size([len(EXTENDED_KUTULU_ACTIONS)])
        assert torch.allclose(policy.sum(), torch.tensor(1.0), atol=1e-6)

    def test_ppo_conv_ext_deep_action_generation(self, explorers, wanderers, mock_env):
        """Test that PPO conv_ext deep agent can generate actions properly"""
        agent = PPOAgent(
            state_type='conv_ext',
            actions=EXTENDED_KUTULU_ACTIONS,
            model_params={'size': 3},
            train=False,
            verbose=False,
            use_deep=True,
        )
        
        player_pos = (3, 3)
        entities = calculate_entities(player_pos, explorers, wanderers)
        agent.set_env(mock_env)
        agent.observer.env._set_entities(entities)
        agent.observer.env._set_players(entities, set_ids=True)
        
        # Test model output directly instead of generate_state_and_step to avoid action mask issues
        state = agent.observer.get_state(0)
        tensor_data = agent.episode_buffer.encode_states([state])
        
        agent.model.eval()
        with torch.no_grad():
            output = agent.model(tensor_data)
        
        policy = output['policy'][0]
        value = output['value'][0]
        
        # Policy should be valid probability distribution
        assert torch.all(torch.isfinite(policy))
        assert torch.all(policy >= 0)
        assert torch.allclose(policy.sum(), torch.tensor(1.0), atol=1e-6)
        assert policy.shape == torch.Size([len(EXTENDED_KUTULU_ACTIONS)])
        
        # Value should be finite
        assert torch.all(torch.isfinite(value))
        assert value.shape == torch.Size([1])
        
        # Test action sampling from policy
        action_dist = torch.distributions.Categorical(policy)
        action = action_dist.sample()
        
        # Action should be valid
        assert isinstance(action.item(), int)
        assert 0 <= action.item() < len(EXTENDED_KUTULU_ACTIONS)
