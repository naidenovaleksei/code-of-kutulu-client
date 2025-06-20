import numpy as np

from src.envs.kutulu_observer import (
    KutuluClosestObserver,
    KutuluClosestExtObserver,
    KutuluConvObserver,
)

class BaseAgent:
    def __init__(self, state_type, action_space_n, train):
        self.state_type = state_type
        self.action_space_n = action_space_n
        self.train = train
        self.observer = None
        self.state_actions = None
        self.output_std = np.inf

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
    
    def get_raw_observation(self, player_id):
        return {
            'const': self.observer.env.constants,
            'info': self.observer.env._get_info(),
            'obs': self.observer.env._get_obs(),
            'player_id': player_id,
        }
    
    def get_valid_actions(self, player_id):
        return self.observer.env.get_valid_action_mask()[player_id]

    def generate_state_and_step(self, player_id, need_update):
        raise NotImplementedError

    def get_eps(self):
        raise NotImplementedError
    
    def get_output_std(self):
        return self.output_std

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
