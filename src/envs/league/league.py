from typing import List, Dict
from enum import Enum
import numpy as np
from tqdm import tqdm

from src.envs.trainer import Trainer, WOOD_MAZES, BRONZE_MAZES
from src.game.template import DEFAULT_KUTULU_ACTIONS, EXTENDED_KUTULU_ACTIONS
from src.envs.league.elo_league import EloLeague
from src.envs.league.agent_description import AgentDescription
from src.envs.agents import BaseAgent
from src.envs.agents.agent_factory import get_agent

RANDOM_AGENT_INFO = {
    'train': False,
    'type': 'epsilon_wait',
    'action_space_n': 5,
    'epsilon_params': {'start': 0, 'final': 0, 'decay': int(4 * 10**5)},
    'state_type': 'closest',
    'action': 'WAIT',
}


class LeagueAgentKind(Enum):
    ACTIVE = 1
    RETIRED = 2
    FIXED = 3

class LeagueAgent:
    def __init__(self, agent_desc: AgentDescription, kind: LeagueAgentKind):
        agent_info = agent_desc.agent_info
        agent_info['train'] = kind == LeagueAgentKind.ACTIVE
        self.type = agent_info['type']
        self.agent_info = agent_info
        self.agent = get_agent(agent_info)
        self.kind = kind
        self.agent_desc = agent_desc
        self.version = 0
        self.key = (self.agent_desc.key, self.type, self.version)


class League:
    def __init__(self,
                 active_population: List[AgentDescription],
                 fixed_population: List[AgentDescription],
                 elo_league: EloLeague = None, use_tqdm=False):
        if elo_league is None:
            self.elo_league = EloLeague()
        else:
            self.elo_league = elo_league
        self.use_tqdm = use_tqdm
        self.active_population: List[LeagueAgent] = [
            LeagueAgent(a, LeagueAgentKind.ACTIVE)
            for a in active_population
        ]
        self.retired_population: List[LeagueAgent] = []
        self.fixed_population: List[LeagueAgent] = [
            LeagueAgent(a, LeagueAgentKind.FIXED)
            for a in fixed_population
        ]
        self.all_agents = self.active_population + self.fixed_population
        self.agents_map = {
            a.key: a
            for a in self.all_agents
        }

    def calculate_ratings(self, num_rounds=50):
        self._calculate_ratings(self.all_agents, num_rounds)

    def find_best(self):
        leader, rating = self.elo_league.leaderboard(top_n=1)[0]
        return self.all_agents[leader]

    def sample_opponent(self, agent):
        pass

    def play_match(self, agent, opponent):
        pass

    def update_winrate(self, agent, opponent):
        pass

    def is_ready_to_retire(self, agent):
        pass

    def is_stagnating(self, agent):
        pass

    def train_exploiter(self, agent, exploiter):
        pass

    def _load_agent(self, agent_info):
        pass

    def _calculate_ratings(self, agents: List[LeagueAgent], num_rounds=10):
        for _ in range(num_rounds):
            idx = np.arange(len(agents))
            np.random.shuffle(idx)
            idx = idx[:idx.shape[0] // 4 * 4]
            assert idx.shape[0] % 4 == 0
            agent_ids_all = np.split(idx, len(agents) // 4)
            if self.use_tqdm:
                agent_ids_all = tqdm(agent_ids_all)
            for agent_ids in agent_ids_all:
                cur_agents: List[BaseAgent] = [agents[i].agent for i in agent_ids]
                scores = self._play_round(cur_agents)
                match_result = [
                    (agents[i].key, score.item())
                    for i, score in zip(agent_ids, scores)
                ]
                self.elo_league.record_match(match_result)

    def _play_round(self, agents: List[BaseAgent],
                   league_level=3, num_envs=1):
        assert len(agents) == 4
        mazes = BRONZE_MAZES if league_level >= 3 else WOOD_MAZES
        actions = EXTENDED_KUTULU_ACTIONS if league_level >= 3 else DEFAULT_KUTULU_ACTIONS
        trainer = Trainer(
            num_experiments=1, agents_info=None, agents=agents, shuffle=True,
            league_level=league_level, mazes=mazes, actions=actions, log_dir='../runs', verbose=False,
            silent=True, num_envs=num_envs, only_train=False, use_tqdm=False,
        )
        result = trainer.play_single_rollout(only_eval=True)
        scores = np.array([np.argwhere(~np.isnan(result[:,i])).max().item() for i in range(4)])
        return scores
