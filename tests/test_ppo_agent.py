import pytest
import numpy as np
import torch

from src.envs.agents.ppo_agent import PPOAgent, PPOBuffer
from src.envs.models.conv_a2c_model import ConvA2CModel
from src.envs.agents.dqn_agent_conv import DQNStateEncoderConv
from src.envs.buffers import Experience


def test_ppo_buffer_creation():
    """Test that PPOBuffer can be created and initialized properly"""
    state_encoder = DQNStateEncoderConv()
    buffer = PPOBuffer(state_encoder)
    
    assert buffer.state_encoder == state_encoder
    assert buffer.need_aug == False
    assert len(buffer.buffer) == 0


def test_ppo_buffer_append_and_end_episode():
    """Test PPOBuffer append and end_episode functionality"""
    state_encoder = DQNStateEncoderConv()
    buffer = PPOBuffer(state_encoder)
    buffer.start_episode()
    
    # Create dummy experience
    dummy_state = np.random.rand(12, 6, 6)
    exp = Experience(
        state=dummy_state,
        action=1,
        reward=1.0,
        done=False,
        new_state=None,
        observation=None
    )
    
    # Append experience with PPO data
    log_prob = -1.5
    value = 0.8
    buffer.append(exp, log_prob, value)
    
    assert len(buffer.buffer) == 1
    
    # End episode and check returned data
    states, actions, rewards, log_probs, values = buffer.end_episode()
    
    assert len(states) == 1
    assert len(actions) == 1
    assert len(rewards) == 1
    assert len(log_probs) == 1
    assert len(values) == 1
    
    assert actions[0] == 1
    assert rewards[0] == 1.0
    assert log_probs[0] == -1.5
    assert values[0] == 0.8


def test_ppo_agent_creation():
    """Test that the PPO agent can be created and initialized properly"""
    agent = PPOAgent(
        state_type='conv',
        action_space_n=5,
        train=True,
    )
    
    # Check that model was initialized
    assert isinstance(agent.model, ConvA2CModel)
    
    # Check that PPO buffer was initialized
    assert isinstance(agent.episode_buffer, PPOBuffer)
    
    # Check default PPO parameters
    assert agent.entropy_coef == 0.01
    assert agent.value_loss_coef == 0.5
    assert agent.clip_ratio == 0.2
    assert agent.ppo_epochs == 4
    assert agent.mini_batch_size == 64
    assert agent.target_kl == 0.01
    assert agent.max_grad_norm == 0.5
    assert agent.gae_lambda == 0.95


def test_ppo_agent_custom_parameters():
    """Test PPO agent with custom parameters"""
    agent = PPOAgent(
        state_type='conv',
        action_space_n=5,
        clip_ratio=0.3,
        ppo_epochs=8,
        mini_batch_size=32,
        target_kl=0.02,
        max_grad_norm=1.0,
        gae_lambda=0.9,
        train=True,
    )
    
    assert agent.clip_ratio == 0.3
    assert agent.ppo_epochs == 8
    assert agent.mini_batch_size == 32
    assert agent.target_kl == 0.02
    assert agent.max_grad_norm == 1.0
    assert agent.gae_lambda == 0.9


def test_ppo_agent_gae_calculation():
    """Test that GAE (Generalized Advantage Estimation) is calculated correctly"""
    agent = PPOAgent(
        state_type='conv',
        action_space_n=5,
        gamma=0.9,
        gae_lambda=0.95,
        train=True,
    )
    
    # Test with simple rewards sequence
    rewards = [1, 0, 2, 1, 0]
    values = torch.tensor([0.5, 0.6, 0.7, 0.8, 0.9])
    
    returns, advantages = agent._calculate_gae(rewards, values)
    
    # Check that returns and advantages have correct length
    assert len(returns) == len(rewards)
    assert len(advantages) == len(rewards)
    
    # Check that returns are reasonable (should be >= rewards in most cases due to future rewards)
    assert isinstance(returns, np.ndarray)
    assert isinstance(advantages, np.ndarray)
    
    # Verify that returns = advantages + values (approximately)
    expected_returns = advantages + values.numpy()
    np.testing.assert_allclose(returns, expected_returns, rtol=1e-5)


def test_ppo_agent_gae_vs_different_lambda():
    """Test that different GAE lambda values produce different advantages"""
    agent1 = PPOAgent(
        state_type='conv',
        action_space_n=5,
        gamma=0.9,
        gae_lambda=0.0,  # No GAE, just TD error
        train=True,
    )
    
    agent2 = PPOAgent(
        state_type='conv',
        action_space_n=5,
        gamma=0.9,
        gae_lambda=1.0,  # Full Monte Carlo
        train=True,
    )
    
    # Test with the same rewards sequence
    rewards = [1, 0, 2, 1, 0]
    values = torch.tensor([0.5, 0.6, 0.7, 0.8, 0.9])
    
    returns1, advantages1 = agent1._calculate_gae(rewards, values)
    returns2, advantages2 = agent2._calculate_gae(rewards, values)
    
    # Verify that different lambda values produce different results
    assert not np.allclose(advantages1, advantages2)
    assert not np.allclose(returns1, returns2)


def test_ppo_clipped_objective_components():
    """Test that PPO clipped objective components work correctly"""
    agent = PPOAgent(
        state_type='conv',
        action_space_n=5,
        clip_ratio=0.2,
        train=True,
    )
    
    # Test clipping behavior
    clip_ratio = agent.clip_ratio
    
    # Test ratios that should be clipped
    ratio_high = torch.tensor([1.5, 2.0, 0.5])  # Some above 1+clip_ratio, some below 1-clip_ratio
    advantages = torch.tensor([1.0, -1.0, 1.0])
    
    # Calculate surrogate losses
    surr1 = ratio_high * advantages
    surr2 = torch.clamp(ratio_high, 1 - clip_ratio, 1 + clip_ratio) * advantages
    
    # Check that clipping works as expected
    expected_clipped_ratios = torch.tensor([1.2, 1.2, 0.8])  # Clipped to [0.8, 1.2]
    expected_surr2 = expected_clipped_ratios * advantages
    
    torch.testing.assert_close(surr2, expected_surr2)
    
    # The minimum should prefer the clipped version when ratio is too high/low
    min_surr = torch.min(surr1, surr2)
    
    # For ratio=1.5, advantage=1.0: min(1.5, 1.2) = 1.2
    # For ratio=2.0, advantage=-1.0: min(-2.0, -1.2) = -2.0 (less negative is better)
    # For ratio=0.5, advantage=1.0: min(0.5, 0.8) = 0.5
    expected_min = torch.tensor([1.2, -2.0, 0.5])
    torch.testing.assert_close(min_surr, expected_min)


def test_ppo_agent_state_type_assertion():
    """Test that PPO agent only accepts 'conv' state type"""
    # This should work
    agent = PPOAgent(
        state_type='conv',
        action_space_n=5,
        train=True,
    )
    assert agent.state_type == 'conv'
    
    # This should raise an assertion error
    with pytest.raises(AssertionError):
        PPOAgent(
            state_type='closest',
            action_space_n=5,
            train=True,
        )


def test_ppo_agent_model_compatibility():
    """Test that PPO agent uses the same model as A2C (ConvA2CModel)"""
    agent = PPOAgent(
        state_type='conv',
        action_space_n=5,
        model_params={'size': 3, 'fc_dim': 128},
        train=True,
    )
    
    # Check that the model is ConvA2CModel with correct parameters
    assert isinstance(agent.model, ConvA2CModel)
    
    # Test model forward pass
    batch_size = 2
    in_channels = 12
    size = 3
    
    # Create dummy input (note: model applies MaxPool2d, so input size should be size*2)
    x = torch.rand(batch_size, in_channels, size*2, size*2)
    
    policy, value = agent.model(x)
    
    # Check output shapes
    assert policy.shape == (batch_size, 5)
    assert value.shape == (batch_size, 1)
    
    # Check that policy is a valid probability distribution
    for i in range(batch_size):
        assert abs(policy[i].sum().item() - 1.0) < 1e-6
        assert (policy[i] >= 0).all().item()


def test_ppo_buffer_with_augmentation():
    """Test PPOBuffer with data augmentation enabled"""
    state_encoder = DQNStateEncoderConv()
    buffer = PPOBuffer(state_encoder, need_aug=True)
    buffer.start_episode()
    
    # Create dummy experience with a proper state dictionary format
    dummy_state = {
        'map': np.zeros((6, 6)),
        'EXPLORER_param0': np.zeros((6, 6)),
        'EXPLORER_param1': np.zeros((6, 6)),
        'EXPLORER_param2': np.zeros((6, 6)),
        'WANDERER_param0': np.zeros((6, 6)),
        'WANDERER_param1': np.zeros((6, 6)),
        'SLASHER_param0': np.zeros((6, 6)),
        'SLASHER_param1': np.zeros((6, 6)),
        'EFFECT_PLAN_param0': np.zeros((6, 6)),
        'EFFECT_LIGHT_param0': np.zeros((6, 6)),
        'EFFECT_SHELTER_param0': np.zeros((6, 6)),
        'EFFECT_YELL_param0': np.zeros((6, 6)),
    }
    # Mark a specific position in the map
    dummy_state['map'][0, 0] = 1.0
    
    exp = Experience(
        state=dummy_state,
        action=0,  # UP action
        reward=1.0,
        done=False,
        new_state=None,
        observation=None
    )
    
    buffer.append(exp, -1.5, 0.8)
    
    # End episode with augmentation
    states, actions, rewards, log_probs, values = buffer.end_episode()
    
    # The action might be rotated (0->1, 1->2, 2->3, 3->0 for clockwise rotations)
    # But log_probs and values should remain the same
    assert len(states) == 1
    assert actions[0] in [0, 1, 2, 3]  # Action should be rotated to one of the directional actions
    assert rewards[0] == 1.0
    assert log_probs[0] == -1.5
    assert values[0] == 0.8


def test_ppo_agent_training_components():
    """Test that PPO agent has all necessary components for training"""
    agent = PPOAgent(
        state_type='conv',
        action_space_n=5,
        train=True,
        verbose=True,
    )
    
    # Check that agent has optimizer
    assert hasattr(agent, 'optimizer')
    assert agent.optimizer is not None
    
    # Check that agent has loss tracking
    assert hasattr(agent, 'last_loss')
    
    # Check that agent has episode tracking
    assert hasattr(agent, 'episode_idx')
    
    # Check that training methods exist
    assert hasattr(agent, '_train_model')
    assert hasattr(agent, '_calculate_gae')
    assert hasattr(agent, 'train_step')


def test_ppo_mini_batch_size_handling():
    """Test that PPO handles different mini-batch sizes correctly"""
    # Test with mini-batch size larger than episode length
    agent = PPOAgent(
        state_type='conv',
        action_space_n=5,
        mini_batch_size=100,  # Larger than typical episode
        train=True,
    )
    
    assert agent.mini_batch_size == 100
    
    # Test with very small mini-batch size
    agent_small = PPOAgent(
        state_type='conv',
        action_space_n=5,
        mini_batch_size=1,
        train=True,
    )
    
    assert agent_small.mini_batch_size == 1
