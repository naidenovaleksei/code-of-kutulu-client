from typing import List, Dict
from enum import Enum
from copy import deepcopy
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import scipy.stats as st

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
NUM_EXPERIMENTS = 10


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
        self.version = '0'
    
    @property
    def key(self):
        return (self.agent_desc.key, self.type, self.version)


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
        self.retired_population = []
        self.all_agents = self.active_population + self.fixed_population
        self.agents_map = {
            a.key: a
            for a in self.all_agents
        }

    def calculate_ratings(self, num_rounds=50):
        self._calculate_ratings(self.all_agents, num_rounds)

    def find_best(self):
        leader, rating = self.elo_league.leaderboard(top_n=1)[0]
        return self.agents_map[leader]

    def sample_opponents(self, agent: LeagueAgent, k: int, max_diff=100):
        print(f"Sample opponents for agent: {agent.key} "
              f"with rating {self.elo_league.ratings[agent.key]}")
        ratings = np.array(list(self.elo_league.ratings.values()))
        std = np.std(ratings)
        potential_opponents = self.active_population + self.fixed_population + self.retired_population

        proba_opponents = []
        probas = []

        seen = set()
        for a in potential_opponents:
            if a.key == agent.key:
                continue
            if a.key in seen:
                continue
            seen.add(a.key)
            diff = self.elo_league.ratings[a.key] - self.elo_league.ratings[agent.key]
            p = st.norm.pdf(diff, scale=std)
            probas.append(p)
            proba_opponents.append(a)

        probas = np.array(probas) / np.sum(probas)
        opponents = []
        for op in np.random.choice(proba_opponents, k, replace=False, p=probas):
            print(f"Sample opponent: {op.key} ({str(op.kind)}) " 
                  f"with rating {self.elo_league.ratings[op.key]}")
            a = deepcopy(op.agent)
            a.train = False
            opponents.append(a)
        return opponents

    def train_new_agent(self, old_agent: LeagueAgent, round_id: int, silent=True, num_experiments=NUM_EXPERIMENTS):
        new_agent = deepcopy(old_agent)
        new_agent.agent.train = True
        opponents = self.sample_opponents(new_agent, 3)
        agents = [new_agent.agent] + opponents
        trainer, result = self.play_match(agents, num_experiments, silent)
        new_agent.agent = trainer.agents[0]
        new_agent.version += str(round_id)
        return new_agent, result

    def play_match(self, agents, num_experiments, silent):
        LEAGUE_LEVEL = 3
        NUM_ENVS = 8
        env_kwargs={
            # 'reward_params': {
            #     'sanity_coef': 0.047,
            #     'reward_for_win': 0,
            #     'reward_for_lose': 0,
            # }
            'reward_params': {
                'sanity_coef': 0.047,
                'reward_for_win': 0,
                'reward_for_lose': -1.,
                # 'step_bonus': 0.01,
                'step_bonus': 0.0,
            }
        }
        trainer = Trainer(
            num_experiments=num_experiments, agents_info=None, agents=agents,
            league_level=LEAGUE_LEVEL,
            num_envs=NUM_ENVS, env_kwargs=env_kwargs, silent=silent, use_tqdm=False,
        )
        result = trainer.train(metrics_int=10)
        return trainer, result
    
    def show_leaderboard(self, top_n=10, normalize=True):
        print("🏆 Leaderboard:")
        ratings = np.array(list(self.elo_league.ratings.values()))
        mean = np.mean(ratings)
        std = np.std(ratings)
        seen = set()
        for player, rating in self.elo_league.leaderboard(top_n):
            is_train = True
            if player in self.agents_map:
                league_agent = self.agents_map[player]
                is_train = league_agent.kind == LeagueAgentKind.ACTIVE
            if normalize:
                rating = (rating - mean) / std
            if player[0] not in seen:
                seen.add(player[0])
                print('T', player, is_train, round(rating, 2))
            else:
                print('B', player, is_train, round(rating, 2))
           
    def show_average_rating_by_version(self, top_n=100):     
        scores = defaultdict(list)
        for player, rating in self.elo_league.leaderboard(top_n):
            if player in self.agents_map:
                league_agent = self.agents_map[player]
                if league_agent.kind != LeagueAgentKind.ACTIVE:
                    continue
            scores[player[2]].append(rating)
        print({k: np.mean(v) for k,v in scores.items()})

    def update_winrate(self, new_population):
        self._calculate_ratings(self.all_agents + new_population, 50)

    def is_ready_to_retire(self, agent):
        pass

    def is_stagnating(self, agent):
        pass

    def train_exploiter(self, agent, exploiter):
        pass

    def _calculate_ratings(self, agents: List[LeagueAgent], num_rounds=10):
        round_ids = range(num_rounds)
        if self.use_tqdm:
            round_ids = tqdm(round_ids)
        for _ in round_ids:
            idx = np.arange(len(agents))
            np.random.shuffle(idx)
            idx = idx[:idx.shape[0] // 4 * 4]
            assert idx.shape[0] % 4 == 0
            agent_ids_all = np.split(idx, len(agents) // 4)
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
        trainer = Trainer(
            num_experiments=1, agents_info=None, agents=agents, shuffle=True,
            league_level=league_level, verbose=False,
            silent=True, num_envs=num_envs, only_train=False, use_tqdm=False,
        )
        result = trainer.play_single_rollout(only_eval=True)
        scores = np.array([np.argwhere(~np.isnan(result[:,i])).max().item() for i in range(4)])
        return scores
