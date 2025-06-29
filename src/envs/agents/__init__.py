from dataclasses import dataclass
from typing import Dict

import numpy as np

from src.envs.kutulu_observer import (
    KutuluClosestObserver,
    KutuluClosestExtObserver,
    KutuluConvObserver,
)
from src.envs.kutulu_world import (
    KutuluObservation,
    KutuluEnvInfo,
)


@dataclass
class AgentObservation:
    const: Dict
    info: KutuluEnvInfo
    obs: KutuluObservation
    player_id: int


class BaseAgent:
    def __init__(self, state_type, action_space_n, train):
        self.state_type = state_type
        self.action_space_n = action_space_n
        self.train = train
        self.observer = None
        self.state_actions = None
        self.last_action = np.zeros(action_space_n)

    def set_env(self, env):
        if self.state_type == 'closest':
            self.observer = KutuluClosestObserver(env)
        elif self.state_type == 'closest_ext':
            self.observer = KutuluClosestExtObserver(env)
        elif self.state_type == 'conv':
            self.observer = KutuluConvObserver(env, self.size)
        elif self.state_type == 'conv_by_kind':
            self.observer = KutuluConvObserver(env, self.size)
        else:
            ValueError('unknown state_type: {self.state_type}')

    def get_state(self, player_id):
        return self.observer.get_state(player_id)
    
    def get_raw_observation(self, player_id) -> AgentObservation:
        env_obs: KutuluObservation = self.observer.env.get_obs(player_id)
        assert env_obs.entities[0].id == player_id
        return AgentObservation(
            self.observer.env.constants,
            self.observer.env.get_info(),
            env_obs,
            player_id,
        )
    
    def get_valid_actions(self, player_id):
        return self.observer.env.get_valid_action_mask()[player_id]

    def generate_state_and_step(self, player_id, need_update):
        raise NotImplementedError

    def get_eps(self):
        raise NotImplementedError
    
    def get_last_action(self):
        return self.last_action

    def get_lr(self):
        return None

    def train_step(self, reward, game_over, new_state):
        raise NotImplementedError

    def check_policy(self):
        raise NotImplementedError

    def save_agent(self, checkpoint_dir):
        raise NotImplementedError

    def load_agent(self, checkpoint_dir):
        raise NotImplementedError
