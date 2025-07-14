import pytest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.envs.trainer import Trainer
from src.envs.agents.ppo_agent import PPOAgent, AgentObservation
from src.game.template import DEFAULT_KUTULU_ACTIONS
from src.envs.kutulu_world import (
    KutuluObservation,
    KutuluEnvInfo,
    KutuluEntity,
)


def test_single_env_backward_compatibility():
    """Test that single environment training still works (backward compatibility)"""
    actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT']
    
    agents_info = [
        {
            'type': 'ppo',
            'state_type': 'conv',
            'action_space_n': len(actions),
            'actions': actions,
            'train': True,
            'verbose': False,
            'model_params': {'size': 3},
            'explicit_action_mask': None,
        },
        {
            'type': 'epsilon_wait',
            'state_type': 'closest',
            'action_space_n': len(actions),
            'epsilon_params': {'start': 0.1, 'final': 0.1, 'decay': 1000},
            'action': 'WAIT',
        }
    ]
    
    # Test single environment (default behavior)
    trainer = Trainer(
        num_experiments=2,
        agents_info=agents_info,
        league_level=3,
        silent=True,
    )
    
    assert trainer.num_envs == 1
    
    # Test that PPO agent is not initialized for multi-env
    ppo_agent = trainer.agents[0]
    assert isinstance(ppo_agent, PPOAgent)
    assert ppo_agent.num_envs == 1
    assert ppo_agent.env_buffers is None
    
    # Run a short training
    rewards, _, _ = trainer.train()
    assert len(rewards) == 2
    trainer.close()


def test_multi_env_initialization():
    """Test that multi-environment setup initializes correctly"""
    actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT']
    
    agents_info = [
        {
            'type': 'ppo',
            'state_type': 'conv',
            'action_space_n': len(actions),
            'actions': actions,
            'train': True,
            'verbose': False,
            'model_params': {'size': 3},
            'explicit_action_mask': None,
        },
        {
            'type': 'epsilon_wait',
            'state_type': 'closest',
            'action_space_n': len(actions),
            'epsilon_params': {'start': 0.1, 'final': 0.1, 'decay': 1000},
            'action': 'WAIT',
        }
    ]
    
    # Test multi-environment setup
    trainer = Trainer(
        num_experiments=1,
        agents_info=agents_info,
        league_level=3,
        num_envs=4,  # Use 4 environments for faster testing
        silent=True,
    )
    
    assert trainer.num_envs == 4
    
    # Test that PPO agent is initialized for multi-env
    ppo_agent = trainer.agents[0]
    assert isinstance(ppo_agent, PPOAgent)
    assert ppo_agent.num_envs == 4
    assert ppo_agent.env_buffers is not None
    assert len(ppo_agent.env_buffers) == 4
    
    # Test that each buffer is properly initialized
    for buffer in ppo_agent.env_buffers:
        assert len(buffer.buffer) == 0
    
    trainer.close()


def test_ppo_buffer_functionality():
    """Test PPO buffer operations"""
    from src.envs.agents.ppo_agent import PPOBuffer
    from src.envs.agents.dqn_agent_conv import DQNStateEncoderConv
    from src.envs.agents.actor_agent import Experience
    
    # Create buffer
    buffer = PPOBuffer(DQNStateEncoderConv(), DEFAULT_KUTULU_ACTIONS)
    buffer.start_episode()
    
    # Test empty buffer
    assert len(buffer.buffer) == 0
    
    # Add some mock experiences
    mock_state = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # Simple 3x3 state
    mock_observation = AgentObservation(
        {},
        KutuluEnvInfo([[]], 0, 0, ''),
        KutuluObservation(2, [KutuluEntity('', 0, 0, 0, 0, 0, 0)]),
        0,
    )
    
    buffer.append(Experience(mock_state, 4, 1.0, False, [], mock_observation), 0.5, 0.8)  # log_prob=0.5, value=0.8
    buffer.append(Experience(mock_state, 4, 1.0, True, [], mock_observation), 0.3, 0.6)  # log_prob=0.3, value=0.6
    
    assert len(buffer.buffer) == 2
    
    # Test end_episode
    states, actions, rewards, dones, log_probs, values = buffer.end_episode()
    
    assert len(states) == 2
    assert len(actions) == 2
    assert len(rewards) == 2
    assert len(log_probs) == 2
    assert len(values) == 2
    
    assert log_probs == [0.5, 0.3]
    assert values == [0.8, 0.6]


if __name__ == "__main__":
    # Run tests manually if executed directly
    test_single_env_backward_compatibility()
    print("✓ Single environment backward compatibility test passed")
    
    test_multi_env_initialization()
    print("✓ Multi-environment initialization test passed")
    
    test_ppo_buffer_functionality()
    print("✓ PPO buffer functionality test passed")
    
    test_environment_diversity()
    print("✓ Environment diversity test passed")
    
    test_multi_env_training()
    print("✓ Multi-environment training test passed")
    
    print("\nAll tests passed! 🎉")
