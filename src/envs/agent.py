import numpy as np
from src.envs.strategy import BaseQStrategy


TEST_STATES = [
    ((4,), 0, None, None),
    ((0,), 1, None, None),
    ((1,), 1, None, None),
    ((2,), 1, None, None),
    ((3,), 1, None, None),
    ((0,), 2, None, None),
    ((1,), 2, None, None),
    ((2,), 2, None, None),
    ((3,), 2, None, None),
    ((0,), 3, None, None),
    ((1,), 3, None, None),
    ((2,), 3, None, None),
    ((3,), 3, None, None),
    ((0,), 4, None, None),
    ((1,), 4, None, None),
    ((2,), 4, None, None),
    ((3,), 4, None, None),
]

def check_policy(Q):
    result_list = []
    for state in TEST_STATES:
        _dir = state[0][0]
        result = np.argmax(Q.get(state, [0])) == _dir
        result_list.append(result)
    return np.mean(result_list)


class Agent:
    def __init__(self, strategy: BaseQStrategy, state_type, action_space_n, eps=0., train=False, alpha=None, gamma=None):
        self.strategy = strategy
        self.state_type = state_type
        self.action_space_n = action_space_n
        self.eps = eps
        self.train = train
        self.alpha = alpha
        self.gamma = gamma
        self.state_actions = None

    def generate_state_and_step(self, observer, player_id):
        state = observer.get_state(player_id, self.state_type)
        valid_actions = observer.env.get_valid_action_mask()[player_id]
        player_mask = ~np.array(valid_actions)
        action = self.strategy.getActionEpsGreedyMasked(
            state,
            self.action_space_n,
            self.eps,
            player_mask
        )
        self.state_actions = (state, action)
        return state, action

    def train_step(self, reward, game_over):
        if reward is None or not self.train:
            return
        state, At = self.state_actions
        Q = self.strategy.Q
        if game_over:
            Q[state][At] += self.alpha * (reward - Q[state][At])
        else:
            Q[state][At] += self.alpha * (
                reward + self.gamma * Q[state].max() - Q[state][At]
            )

    def check_policy(self):
        return check_policy(self.strategy.Q)
