import pytest
import numpy as np
import torch

from src.envs.agents.reinforce_agent import REINFORCEAgent
from src.envs.models.ext_state_model import ExtStateModel
from src.game.template import ENTITY_TOKENS


def test_reinforce_model_output():
    """Test that the REINFORCE model produces valid probability distributions"""
    model = ExtStateModel(
        vocab_size=len(ENTITY_TOKENS) + 1,
        num_dirs=5,
        features_dim=7,
        embed_dim=32,
        hidden_dim=32,
        inner_dim=16,
        num_classes=5,
        return_softmax=True,
    )
    
    # Create dummy input data
    batch_size = 2
    entity_dim = 3  # Number of entities
    
    data = {
        'entity_kind': torch.randint(0, len(ENTITY_TOKENS), (batch_size, entity_dim)),
        'entity_features': torch.rand(batch_size, entity_dim, 7),
        'entity_dir': torch.zeros(batch_size, entity_dim, 5)
    }
    
    # Add some valid directions
    data['entity_dir'][0, 0, 0] = 1
    data['entity_dir'][0, 1, 2] = 1
    data['entity_dir'][1, 0, 1] = 1
    
    # Forward pass
    output = model(data)
    
    # Check output dimensions
    assert output.shape == (batch_size, 5)
    
    # Check that output is a valid probability distribution
    for i in range(batch_size):
        # Sum of probabilities should be close to 1
        assert abs(output[i].sum().item() - 1.0) < 1e-6
        # All probabilities should be >= 0
        assert (output[i] >= 0).all().item()


def test_reinforce_agent_creation():
    """Test that the REINFORCE agent can be created and initialized properly"""
    agent = REINFORCEAgent(
        state_type='closest_ext',
        action_space_n=5,
        train=True,
    )
    
    # Check that model was initialized
    assert isinstance(agent.model, ExtStateModel)
    
    # Check that episode buffer was initialized
    assert agent.episode_buffer is not None
    

def test_reinforce_agent_returns_calculation():
    """Test that returns are calculated correctly"""
    agent = REINFORCEAgent(
        state_type='closest_ext',
        action_space_n=5,
        gamma=0.9,
        train=True,
    )
    
    # Test with simple rewards sequence
    rewards = [1, 0, 2]
    returns = agent._calculate_returns(rewards)
    
    # Expected discounted returns: 
    # G_0 = 1 + 0.9 * 0 + 0.9^2 * 2 = 1 + 0 + 1.62 = 2.62
    # G_1 = 0 + 0.9 * 2 = 1.8
    # G_2 = 2
    expected = torch.tensor([2.62, 1.8, 2.0])
    
    # Normalize expected
    if len(expected) > 1:
        expected = (expected - expected.mean()) / (expected.std() + 1e-9)
    
    # Check calculated returns
    assert torch.allclose(returns, expected, atol=1e-2)