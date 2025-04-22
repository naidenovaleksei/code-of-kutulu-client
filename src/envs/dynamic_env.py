import gym
from gym.spaces import Discrete
import numpy as np

from typing import List


class DynamicActionEnv(gym.Env):
    def __init__(self, actions: List[str], players_count: int = 4):
        super(DynamicActionEnv, self).__init__()
        self._actions = actions
        self.action_count = len(actions)
        self.action_space = Discrete(self.action_count)