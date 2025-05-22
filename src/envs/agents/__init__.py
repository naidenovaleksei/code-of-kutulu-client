
from src.envs.kutulu_observer import (
    KutuluClosestObserver,
    KutuluClosestExtObserver
)

class BaseAgent:
    def __init__(self, state_type, action_space_n, train):
        self.state_type = state_type
        self.action_space_n = action_space_n
        self.train = train
        self.observer = None
        self.state_actions = None
    
    def set_env(self, env):
        if self.state_type == 'closest':
            self.observer = KutuluClosestObserver(env)
        elif self.state_type == 'closest_ext':
            self.observer = KutuluClosestExtObserver(env)
        else:
            ValueError('unknown state_type: {self.state_type}')

    def get_state(self, player_id):
        return self.observer.get_state(player_id)
    
    def get_valid_actions(self, player_id):
        return self.observer.env.get_valid_action_mask()[player_id]

    def generate_state_and_step(self, player_id):
        raise NotImplementedError

    def train_step(self, reward, game_over, new_state):
        raise NotImplementedError

    def check_policy(self):
        raise NotImplementedError
