import numpy as np
from src.game.template import getActionGreedyMasked


class BaseQStrategy:
    def __init__(self, hashes_map={}):
        self.Q = {
            state: np.random.normal(size=len(empty_spaces))
            for state, empty_spaces in hashes_map.items()
        }

    def getActionGreedy(self, state, actions):
        raise NotImplementedError


class RandomStrategy(BaseQStrategy):
    def getActionGreedy(self, state, action_space_n):
        return np.random.randint(action_space_n)
    
    def getActionEpsGreedyMasked(self, state, action_space_n, eps, mask):
        actions_idx = np.arange(action_space_n)[~mask]
        if len(actions_idx) == 0:
            return np.random.randint(action_space_n)
        return np.random.choice(actions_idx)


class GreedyStrategy(BaseQStrategy):
    def getActionGreedy(self, state, actions):
        a_star = self.Q[state].argmax()
        return a_star

    def check_state(self, state, action_space_n):
        return state in self.Q and len(self.Q[state]) == action_space_n

    def getActionEpsGreedy(self, state, action_space_n, eps):
        assert self.check_state(state, action_space_n)
        assert len(self.Q[state]) == action_space_n
        a_star = self.Q[state].argmax()
        ps = np.ones(action_space_n) * eps / action_space_n
        ps[a_star] = 1 - eps + eps / action_space_n
        At = np.random.choice(np.arange(action_space_n), p=ps / ps.sum())
        return At

    def getActionEpsGreedyMasked(self, state, action_space_n, eps, mask):
        assert self.check_state(state, action_space_n)
        assert len(self.Q[state]) == action_space_n
        a = np.ma.array(self.Q[state], mask=mask)
        a_star = a.argmax()
        ps = np.ones(action_space_n) * eps / action_space_n
        ps[a_star] = 1 - eps + eps / action_space_n
        At = np.random.choice(np.arange(action_space_n), p=ps / ps.sum())
        return At


class LazyGreedyStrategy(GreedyStrategy):
    def __init__(self):
        super(LazyGreedyStrategy, self).__init__()

    def getActionGreedy(self, state, action_space_n):
        if state not in self.Q:
            return np.random.randint(action_space_n)
        assert len(self.Q[state]) == action_space_n
        a_star = self.Q[state].argmax()
        return a_star

    def getActionGreedyMasked(self, state, action_space_n, mask):
        return getActionGreedyMasked(state, self.Q, action_space_n, mask)

    def check_state(self, state, action_space_n):
        if state not in self.Q:
            actions = np.random.normal(size=action_space_n)
            self.Q[state] = actions
        return True
