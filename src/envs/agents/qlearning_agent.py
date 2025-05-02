import numpy as np
from src.envs.strategy import BaseQStrategy
from src.envs.cross_entropy import CrossEntropyModel

import torch
import torch.nn as nn
import torch.optim as optim


class BaseAgent:
    def __init__(self):
        pass
    def generate_state_and_step(self, observer, player_id):
        raise NotImplementedError

    def train_step(self, reward, game_over, new_state):
        raise NotImplementedError

    def check_policy(self):
        raise NotImplementedError


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


class Agent(BaseAgent):
    def __init__(self, strategy: BaseQStrategy, state_type, action_space_n, eps=0., train=False, alpha=None, gamma=None):
        # super(Agent, self).__init__()
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

    def train_step(self, reward, game_over, new_state):
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
        return check_policy(self.strategy.Q)

def parse_dir(encoded_dir):
    result = [0, 0]
    if encoded_dir is None:
        return result
    for _dir, (index, amp) in enumerate([(1, +1), (0, +1), (1, -1), (0, -1)]):
        if _dir in encoded_dir:
            result[index] += amp
    return result

def parse_dist(encoded_dist):
    if encoded_dist is None:
        return [100]
    return [encoded_dist]

def encode_states(states):
    data = dict(
        closest_explorer_dir=[parse_dir(state[0]) for state in states],
        closest_explorer_dist=[parse_dist(state[1]) for state in states],
        closest_wanderer_dir=[parse_dir(state[2]) for state in states],
        closest_wanderer_dist=[parse_dist(state[3]) for state in states],
    )
    data = {k: torch.tensor(v) for k,v in data.items()}
    return data


class CrossEntropyAgent(BaseAgent):
    def __init__(self, state_type, action_space_n, train=False, verbose=False):
        # super(CrossEntropyAgent, self).__init__()
        self.state_type = state_type
        self.action_space_n = action_space_n
        self.train = train
        self.verbose = verbose
        self.state_actions = []
        self.rewards = []

        self.model = CrossEntropyModel(
            vocab_size=3,
            embed_dim=16,
            hidden_dim=8,
            num_classes=action_space_n
        )
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.last_loss = None

    def generate_state_and_step(self, observer, player_id):
        state = observer.get_state(player_id, self.state_type)        
        valid_actions = observer.env.get_valid_action_mask()[player_id]
        player_mask = ~np.array(valid_actions)
        
        data = encode_states([state])
        model_output = torch.softmax(self.model(data)[0], -1).detach().cpu().numpy()

        ps = np.ma.array(model_output, mask=player_mask).filled(0)
        action = np.random.choice(np.arange(len(ps)), p=ps / ps.sum())

        if self.train:
            self.state_actions.append((state, action))
        return state, action

    def train_step(self, reward, game_over, new_state=None):
        if not self.train:
            return

        if reward is None:
            while len(self.rewards) < len(self.state_actions):
                self.state_actions.pop()
        else:
            self.rewards.append(reward)

        assert len(self.rewards) == len(self.state_actions)

        if game_over:
            winner = reward is not None
            if winner:
                assert reward > 0
            # threshold = np.percentile(self.rewards, 50)
            threshold = -2
            state_actions = [sa for sa, r in zip(self.state_actions, self.rewards) if r >= threshold]
            rewards = [r for sa, r in zip(self.state_actions, self.rewards) if r >= threshold]
            self.state_actions = state_actions
            self.rewards = rewards
            self.train_model()

    def check_policy(self):
#         return dict(model.named_parameters())['output.weight'].detach().cpu().numpy().std()
        return self.last_loss

    def train_model(self):
        assert len(self.rewards) == len(self.state_actions)

        states = [s for s, a in self.state_actions]
        data = encode_states(states)
        
        actions = [a for s, a in self.state_actions]
        target = torch.tensor(actions)
        
        # old_weight = self.model.output.weight.clone().detach()
        self.model.train()
        self.optimizer.zero_grad()
        
        preds = self.model(data)
        loss = self.criterion(preds, target)
        loss.backward()
        self.optimizer.step()
        
        self.last_loss = loss.item()

        self.state_actions = []
        self.rewards = []
        
        # new_weight = self.model.output.weight.detach()
        # print("Weight changed:", not torch.allclose(old_weight, new_weight))
        if self.verbose:
            print(f"Loss: {loss.item():.4f}")