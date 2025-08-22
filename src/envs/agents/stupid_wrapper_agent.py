import numpy as np
import scipy.special as sp

from src.envs.agents import BaseAgent
from src.game.template import (
    EXTENDED_KUTULU_ACTIONS
)


class StupidAgentWrapper:
    def __init__(self, agent: BaseAgent, epsilon, mode, actions_mask=None,
                 explicit_change=False, softmax=False, seed=None):
        self.agent = agent
        self.agent.train = False
        self.train = False
        self.epsilon = epsilon
        self.mode = mode
        self.actions_mask = None
        self.softmax = softmax
        self.explicit_change = explicit_change

        if actions_mask is not None:
            self.actions_mask = np.array([
                i for i, v in enumerate(EXTENDED_KUTULU_ACTIONS)
                if v in actions_mask
            ])[:self.agent.action_space_n]
        if seed:
            np.random.seed(seed)

    def generate_random_step(self, actions_masked, player_mask):
        return self.agent.generate_random_step(actions_masked, player_mask)

    def inference_step(self, player_id):
        output = self.agent.inference_step(player_id)
        
        if np.random.random() < self.epsilon:
            model_output = output['model_output']
            valid_actions = output['valid_actions']

            action_space_n = self.agent.action_space_n
            valid_actions = np.array(valid_actions)[:action_space_n]

            if self.actions_mask is not None:
                valid_actions &= self.actions_mask
            
            if self.explicit_change:
                valid_actions[output['action']] = False
                
            if self.mode == 'random':
                ps = valid_actions
            else:
                assert self.mode == 'sample'
                if self.softmax:
                    model_output = sp.softmax(model_output)
                ps = np.ma.array(model_output, mask=~valid_actions).filled(0)

            if ps.sum() == 0:
                action = np.random.randint(action_space_n)
            else:
                action = np.random.choice(np.arange(action_space_n), p=ps / ps.sum())
        else:
            action = output['action']
        return {
            'action': action,
        }

    def generate_state_and_step(self, player_id, need_update=True):
        return self.inference_step(player_id)
    
    def get_eps(self):
        return 0

    def check_policy(self):
        return 0
    
    def get_last_action(self):
        return self.agent.last_action

    def set_env(self, env):
        self.agent.set_env(env)

    def train_step(self):
        pass

    def save_agent(self, checkpoint_dir):
        pass

    def load_agent(self, checkpoint_dir):
        pass
