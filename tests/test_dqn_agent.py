import pytest
import numpy as np
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
