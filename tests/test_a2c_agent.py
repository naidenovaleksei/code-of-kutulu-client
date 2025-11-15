import pytest
import numpy as np
import torch

from src.envs.agents.a2c_agent import A2CAgent
from src.envs.models.conv_a2c_model import ConvA2CModel


def test_conv_a2c_model_output():
    """Test that the ConvA2CModel produces valid probability distributions and value estimates"""
    # Create model
    size = 3
    model = ConvA2CModel(
        size=size,
        in_channels=12,
        num_classes=5,
        fc_dim=64
    )
    
    # Create dummy input data
    batch_size = 2
    in_channels = 12
    
    # The model applies a MaxPool2d which reduces the size by half
    # After pooling, the size is size/2, so we need to adjust our input
    # to ensure the dimensions match after flattening
    x = torch.rand(batch_size, in_channels, size*2, size*2)
    
    # Forward pass
    output = model(x)
    policy = output['policy']
    value = output['value']
    
    # Check output dimensions
    assert policy.shape == (batch_size, 5)
    assert value.shape == (batch_size, 1)
    
    # Check that policy output is a valid probability distribution
    for i in range(batch_size):
        # Sum of probabilities should be close to 1
        assert abs(policy[i].sum().item() - 1.0) < 1e-6
        # All probabilities should be >= 0
        assert (policy[i] >= 0).all().item()
    
    # Test individual methods (these return the specific values from the dict)
    policy_only = model.get_policy(x)
    assert torch.allclose(policy_only, policy)
    
    value_only = model.get_value(x)
    assert torch.allclose(value_only, value)
    
    log_probs = model.get_log_probs(x)
    assert log_probs.shape == (batch_size, 5)


def test_a2c_agent_creation():
    """Test that the A2C agent can be created and initialized properly"""
    agent = A2CAgent(
        state_type='conv',
        action_space_n=5,
        train=True,
    )
    
    # Check that model was initialized
    assert isinstance(agent.model, ConvA2CModel)
    
    # Check that episode buffer was initialized
    assert agent.episode_buffer is not None
    
    # Check default parameters
    assert agent.entropy_coef == 0.01
    assert agent.value_loss_coef == 0.5
    assert agent.n_step == 10


def test_a2c_agent_returns_and_advantages_calculation():
    """Test that returns and advantages are calculated correctly"""
    agent = A2CAgent(
        state_type='conv',
        action_space_n=5,
        gamma=0.9,
        n_step=3,
        train=True,
    )
    
    # Test with simple rewards sequence
    rewards = [1, 0, 2, 1, 0]
    values = torch.tensor([0.5, 0.6, 0.7, 0.8, 0.9])
    
    returns, advantages = agent._calculate_returns_and_advantages(rewards, values)
    
    # Expected returns for n_step=3:
    # G_0 = 1 + 0.9*0 + 0.9^2*2 + 0.9^3*0.8 = 1 + 0 + 1.62 + 0.5832 = 3.2032
    # G_1 = 0 + 0.9*2 + 0.9^2*1 + 0.9^3*0.9 = 0 + 1.8 + 0.81 + 0.6561 = 3.2661
    # G_2 = 2 + 0.9*1 + 0.9^2*0 = 2 + 0.9 + 0 = 2.9
    # G_3 = 1 + 0.9*0 = 1
    # G_4 = 0
    expected_returns = torch.tensor([3.2032, 3.2661, 2.9, 1.0, 0.0])
    
    # Expected advantages = returns - values
    expected_advantages = expected_returns - values
    
    # Normalize expected advantages
    expected_advantages = (expected_advantages - expected_advantages.mean()) / (expected_advantages.std() + 1e-8)
    
    # Check calculated returns and advantages
    assert torch.allclose(returns, expected_returns, atol=1e-4)
    assert torch.allclose(advantages, expected_advantages, atol=1e-4)


def test_a2c_agent_n_step_parameter():
    """Test that the n-step parameter affects the return calculation"""
    # Create agents with different n-step values
    agent1 = A2CAgent(
        state_type='conv',
        action_space_n=5,
        gamma=0.9,
        n_step=1,
        train=True,
    )
    
    agent2 = A2CAgent(
        state_type='conv',
        action_space_n=5,
        gamma=0.9,
        n_step=3,
        train=True,
    )
    
    # Test with the same rewards sequence
    rewards = [1, 0, 2, 1, 0]
    values = torch.tensor([0.5, 0.6, 0.7, 0.8, 0.9])
    
    returns1, _ = agent1._calculate_returns_and_advantages(rewards, values)
    returns2, _ = agent2._calculate_returns_and_advantages(rewards, values)
    
    # Verify that the returns are different based on n-step value
    assert not torch.allclose(returns1, returns2)
    
    # Expected returns for n_step=1:
    # G_0 = 1 + 0.9*0.6 = 1.54
    # G_1 = 0 + 0.9*0.7 = 0.63
    # G_2 = 2 + 0.9*0.8 = 2.72
    # G_3 = 1 + 0.9*0.9 = 1.81
    # G_4 = 0
    expected_returns1 = torch.tensor([1.54, 0.63, 2.72, 1.81, 0.0])
    
    # Expected returns for n_step=3 (calculated in previous test)
    expected_returns2 = torch.tensor([3.2032, 3.2661, 2.9, 1.0, 0.0])
    
    # Check calculated returns for both agents
    assert torch.allclose(returns1, expected_returns1, atol=1e-2)
    assert torch.allclose(returns2, expected_returns2, atol=1e-4)
