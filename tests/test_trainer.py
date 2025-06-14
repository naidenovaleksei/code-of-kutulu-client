import pytest
import numpy as np

from src.envs.trainer import Trainer
from src.game.template import EXTENDED_KUTULU_ACTIONS

@pytest.fixture
def agents_info():
    return [
        {
            'train': True,
            'type': 'qdn_by_kind',
            'action_space_n': 8,
            'state_type': 'closest_ext',
            'buffer_params': {'capacity': 100},
            'epsilon_params': {},
        },
        {
            'train': False,
            'type': 'qlearning',
            'action_space_n': 8,
            'state_type': 'closest',
            'strategy': 'random',
        },
        {
            'train': False,
            'type': 'qlearning',
            'action_space_n': 8,
            'state_type': 'closest',
            'strategy': 'random',
        },
        {
            'train': False,
            'type': 'qlearning',
            'action_space_n': 8,
            'state_type': 'closest',
            'strategy': 'random',
        },
    ]

@pytest.fixture
def trainer(agents_info):
    return Trainer(
        num_experiments=5000, agents_info=agents_info, shuffle=False,
        league_level=3, actions=EXTENDED_KUTULU_ACTIONS, log_dir='../runs'
    )


def test_play_rollout_rewards_and_dones(trainer):
    """Test get_state with player and one wanderer."""
    rollout_rewards = trainer.play_rollout(verbose=False)

    agent_id = 0
    buffer = trainer.agents[agent_id].episode_buffer.buffer
    buffer_agent_rewards = [e.reward for e in buffer]
    rollout_agent_rewards = rollout_rewards[:len(buffer), agent_id]
    assert np.allclose(buffer_agent_rewards, rollout_agent_rewards)

    buffer_agent_dones = [e.done for e in buffer]
    assert np.allclose(buffer_agent_dones[:-1], 0)
    assert buffer_agent_dones[-1] == 1

    buffer_agent_states = [e.state for e in buffer]
    buffer_agent_new_states = [e.new_state for e in buffer]
    for i, (state, new_state) in enumerate(
        zip(buffer_agent_states[:-1], buffer_agent_new_states[:-1])
    ):
        assert state != new_state
    assert buffer_agent_states[-1] == buffer_agent_new_states[-1]
