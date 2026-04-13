import numpy as np
import pickle as pkl
from src.envs.agents import BaseAgent

from src.envs.agents.nn_agent import(
    EPSILON_START,
    EPSILON_FINAL,
    EPSILON_DECAY_LAST_FRAME,
)
from src.game.template import (
    EXTENDED_KUTULU_ACTIONS
)


class EpsilonConstAgent(BaseAgent):
    def __init__(self, state_type, action_space_n, epsilon_params, action, seed=None):
        super(EpsilonConstAgent, self).__init__(
            state_type=state_type,
            action_space_n=action_space_n,
            train=False,
        )
        
        assert len(set(epsilon_params) - set([
            'start', 'final', 'decay', 'reset', 'reset_coef',
        ])) == 0
        
        if type(action) == str:
            action = {
                v: i for i, v in enumerate(EXTENDED_KUTULU_ACTIONS)
            }[action]
        self.action = action
        self.epsilon_start = epsilon_params.get('start', EPSILON_START)
        self.epsilon_final = epsilon_params.get('final', EPSILON_FINAL)
        self.eps = self.epsilon_start
        self.epsilon_decay_last_frame = epsilon_params.get('decay', EPSILON_DECAY_LAST_FRAME)
        self.epsilon_reset = epsilon_params.get('reset')
        self.epsilon_reset_coef = epsilon_params.get('reset_coef')
        self.frame_idx = 0
        if seed:
            np.random.seed(seed)

    def get_eps(self):
        return max(
            self.epsilon_final,
            self.eps,
        )

    def generate_random_step(self, actions_masked, player_mask):
        ps = np.ma.array(np.ones(self.action_space_n), mask=player_mask).filled(0)
        if ps.sum() == 0:
            return np.random.randint(self.action_space_n)
        return np.random.choice(np.arange(self.action_space_n), p=ps / ps.sum())

    def inference_step(self, player_id):
        return self.generate_state_and_step(player_id, need_update=False)

    def generate_state_and_step(self, player_id, need_update=True):
        if need_update:
            self.frame_idx += 1
            self._update_eps()
        if np.random.random() < self.get_eps():
            action = self.action
        else:
            valid_actions = self.get_valid_actions(player_id)
            player_mask = ~np.array(valid_actions)
            player_mask = player_mask[:self.action_space_n]
            action = self.generate_random_step(None, player_mask)
        action = int(action) if not isinstance(action, int) else action
        return {
            'state': None,
            'action': action,
            'model_output': None,
            'valid_actions': self.get_valid_actions(player_id),
        }

    def _update_eps(self):
        self.eps -= self.epsilon_start / self.epsilon_decay_last_frame
        if self.epsilon_reset is not None:
            if self.frame_idx >= self.epsilon_reset:
                self.eps = min(self.epsilon_start, self.eps * 2)
                self.epsilon_reset *= self.epsilon_reset_coef

    def train_step(self):
        pass
    
    def check_policy(self):
        return 0

    def save_agent(self, checkpoint_dir):
        pass

    def load_agent(self, checkpoint_dir):
        pass
