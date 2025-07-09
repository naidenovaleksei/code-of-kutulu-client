from typing import List, Dict
import numpy as np
from tqdm import tqdm

from src.envs.trainer import Trainer, WOOD_MAZES, BRONZE_MAZES
from src.game.template import DEFAULT_KUTULU_ACTIONS, EXTENDED_KUTULU_ACTIONS
from src.envs.league.elo_league import EloLeague
from src.envs.league.agent_description import AgentDescription

RANDOM_AGENT_INFO = {
    'train': False,
    'type': 'epsilon_wait',
    'action_space_n': 5,
    'epsilon_params': {'start': 0, 'final': 0, 'decay': int(4 * 10**5)},
    'state_type': 'closest',
    'action': 'WAIT',
}

class League:
    def __init__(self,
                 active_population: List[AgentDescription],
                 fixed_population: List[AgentDescription],
                 elo_league: EloLeague = None, use_tqdm=False):
        if elo_league is None:
            elo_league = EloLeague()
        self.elo_league = elo_league
        self.use_tqdm = use_tqdm
        self.active_population: Dict[str, AgentDescription] = {
            a.key: a for a in active_population
        }
        assert len(self.active_population) == len(active_population)
        self.fixed_population: Dict[str, AgentDescription] = {
            a.key: a for a in fixed_population
        }
        assert len(self.fixed_population) == len(fixed_population)

    def calculate_ratings(self, num_rounds=50):
        agent_descs = list(self.active_population.values()) + list(self.fixed_population.values())
        self._calculate_ratings(agent_descs, num_rounds)

    def find_best(self):
        agent_descs_dict = dict(
            **self.active_population,
            **self.fixed_population
        )
        leader_desc, rating = self.elo_league.leaderboard(top_n=1)[0]
        return agent_descs_dict[leader_desc]

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

    def _calculate_ratings(self, agent_descs: List[AgentDescription], num_rounds=10):
        for _ in range(num_rounds):
            idx = np.arange(len(agent_descs))
            np.random.shuffle(idx)
            idx = idx[:idx.shape[0] // 4 * 4]
            assert idx.shape[0] % 4 == 0
            agent_ids_all = np.split(idx, len(agent_descs) // 4)
            if self.use_tqdm:
                agent_ids_all = tqdm(agent_ids_all)
            for agent_ids in agent_ids_all:
                cur_agent_descs: List[AgentDescription] = [agent_descs[i] for i in agent_ids]
                scores = self.play_round(cur_agent_descs)
                assert len(scores) == 1
                match_result = [
                    (agent_desc.key, score.item())
                    for agent_desc, score in zip(cur_agent_descs, scores[0])
                ]
                self.elo_league.record_match(match_result)

    def play_round(self, agent_descs: List[AgentDescription],
                   num_experiments=1, league_level=3, num_envs=1):
        assert len(agent_descs) == 4
        agents_info = [
            a.agent_info
            for a in agent_descs
        ]
        mazes = BRONZE_MAZES if league_level >= 3 else WOOD_MAZES
        actions = EXTENDED_KUTULU_ACTIONS if league_level >= 3 else DEFAULT_KUTULU_ACTIONS
        trainer = Trainer(
            num_experiments=num_experiments, agents_info=agents_info, shuffle=True,
            league_level=league_level, mazes=mazes, actions=actions, log_dir='../runs', verbose=False,
            silent=True, num_envs=num_envs, only_train=False, use_tqdm=False,
        )
        result = trainer.train()
        scores = np.array([[np.argwhere(~np.isnan(r[:,i])).max().item() for i in range(4)] for r in result[0]])
        return scores
