import numpy as np
import pickle as pkl
from src.envs.strategy import BaseQStrategy
from src.envs.agents import BaseAgent


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


class QlearningAgent(BaseAgent):
    def __init__(self, strategy: BaseQStrategy, state_type, action_space_n, eps=0., train=False, alpha=None, gamma=None):
        super(QlearningAgent, self).__init__(
            state_type=state_type,
            action_space_n=action_space_n,
            train=train,
        )
        self.strategy = strategy
        self.eps = eps
        self.alpha = alpha
        self.gamma = gamma

    def generate_state_and_step(self, player_id, need_update=True):
        state = self.get_state(player_id)
        valid_actions = self.get_valid_actions(player_id)
        player_mask = ~np.array(valid_actions)
        if self.train:
            action = self.strategy.getActionEpsGreedyMasked(
                state,
                self.action_space_n,
                self.eps,
                player_mask
            )
        else:
            action = self.strategy.getActionGreedyMasked(
                state,
                self.action_space_n,
                player_mask
            )
        self.state_actions = (state, action)
        return state, action

    def train_step(self):
        if reward is None or not self.train:
            return
        state, At = self.state_actions
        Q = self.strategy.Q
        if game_over:
            Q[state][At] += self.alpha * (reward - Q[state][At])
        else:
            assert self.strategy.check_state(new_state, self.action_space_n)
            Q[state][At] += self.alpha * (
                reward + self.gamma * Q[new_state].max() - Q[state][At]
            )

    def check_policy(self):
        result_list = []
        for state in TEST_STATES:
            _dir = state[0][0]
            result = np.argmax(self.strategy.Q.get(state, [0])) == _dir
            result_list.append(result)
        return np.mean(result_list)

    def get_eps(self):
        return self.eps

    def save_agent(self, checkpoint_dir):
        Q = self.strategy.Q
        new_keys = list(Q.keys())
        data1 = pkl.dumps(np.array(list(Q[k] for k in new_keys), dtype='float16'))
        data2 = pkl.dumps(list(new_keys))
        with open(f"{checkpoint_dir}/data1.pkl", "wb") as f:
            f.write(data1)
        with open(f"{checkpoint_dir}/data2.pkl", "wb") as f:
            f.write(data2)

    def load_agent(self, checkpoint_dir):
        with open(f"{checkpoint_dir}/data1.pkl", "rb") as f:
            vals = pkl.loads(f.read())
        with open(f"{checkpoint_dir}/data2.pkl", "rb") as f:
            keys = pkl.loads(f.read())
        self.strategy.Q = dict(zip(keys, vals))
