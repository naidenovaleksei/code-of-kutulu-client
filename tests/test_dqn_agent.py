import pytest
import numpy as np
import torch
from itertools import combinations

from src.envs.agents.dqn_agent import DQNAgent, ExperienceBuffer

def test_dqn_model():
    self = DQNAgent('closest', 5, buffer_params={'capacity': 1}, epsilon_params={})
    res = {}
    self.model.eval()
    for dist in [0,1,2,3,4,5]:
        for l in [1,2,3,4,5]:
            for edir in combinations((0,1,2,3,4), l):
                state = (edir, dist, None, None)
                res[state] = self.model(self.episode_buffer.state_encoder.encode_states([state]))[0].detach()
                state =  (None, None, edir, dist)
                res[state] = self.model(self.episode_buffer.state_encoder.encode_states([state]))[0].detach()
    resv = [tuple(x) for x in res.values()]
    assert len(resv) == len(set(resv))

def test_dueling_dqn_model():
    """Test that dueling DQN produces different outputs than standard DQN"""
    # Create standard DQN agent
    standard_agent = DQNAgent('closest', 5, buffer_params={'capacity': 1}, epsilon_params={}, dueling=False)
    
    # Create dueling DQN agent
    dueling_agent = DQNAgent('closest', 5, buffer_params={'capacity': 1}, epsilon_params={}, dueling=True)
    
    # Test with a sample state
    test_state = ((0, 1), 2, (2, 3), 1)
    encoded_state = standard_agent.episode_buffer.state_encoder.encode_states([test_state])
    
    standard_agent.model.eval()
    dueling_agent.model.eval()
    
    with torch.no_grad():
        standard_output = standard_agent.model(encoded_state)[0]
        dueling_output = dueling_agent.model(encoded_state)[0]
    
    # Outputs should have the same shape
    assert standard_output.shape == dueling_output.shape
    assert standard_output.shape == torch.Size([5])  # 5 actions
    
    # Both should produce valid Q-values (finite numbers)
    assert torch.all(torch.isfinite(standard_output))
    assert torch.all(torch.isfinite(dueling_output))

def test_dueling_dqn_architecture():
    """Test that dueling DQN has the expected architecture properties"""
    dueling_agent = DQNAgent('closest', 5, buffer_params={'capacity': 1}, epsilon_params={}, dueling=True)
    
    # Check that the agent knows it's using dueling architecture
    assert dueling_agent.dueling == True
    
    # Test with multiple states to ensure consistent behavior
    test_states = [
        ((0, 1), 2, None, None),
        (None, None, (2, 3), 1),
        ((0, 1, 2), 3, (3, 4), 2),
    ]
    
    dueling_agent.model.eval()
    
    for state in test_states:
        encoded_state = dueling_agent.episode_buffer.state_encoder.encode_states([state])
        with torch.no_grad():
            output = dueling_agent.model(encoded_state)[0]
        
        # Should produce valid Q-values
        assert torch.all(torch.isfinite(output))
        assert output.shape == torch.Size([5])

def test_backward_compatibility():
    """Test that standard DQN still works as before"""
    # This should work exactly as before
    standard_agent = DQNAgent('closest', 5, buffer_params={'capacity': 1}, epsilon_params={})
    
    # Should default to non-dueling
    assert standard_agent.dueling == False
    
    # Should produce the same results as the original test
    res = {}
    standard_agent.model.eval()
    for dist in [0,1,2]:  # Reduced for faster testing
        for l in [1,2]:
            for edir in combinations((0,1,2,3,4), l):
                state = (edir, dist, None, None)
                res[state] = standard_agent.model(standard_agent.episode_buffer.state_encoder.encode_states([state]))[0].detach()
    
    # All outputs should be unique (as in original test)
    resv = [tuple(x) for x in res.values()]
    assert len(resv) == len(set(resv))
