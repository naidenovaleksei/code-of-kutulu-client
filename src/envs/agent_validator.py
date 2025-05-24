import numpy as np

from src.envs.kutulu_world import KutuluWorldEnv
from src.game.template import MOVE_REL_POS, REL_POSITIONS

class AgentValidator:
    def __init__(self, actions):
        self.env = KutuluWorldEnv('', '', 1, actions=actions)
        self.env.map = [
            #0123456
            '#######', # 0
            '#.....#', # 1
            '#.#.#.#', # 2
            '#.....#', # 3
            '#.#.#.#', # 4
            '#.....#', # 5
            '#######', # 6
        ]
        self.env.width = len(self.env.map[0])
        self.env.height = len(self.env.map)
    
    def check_entity_nearby(self, agent, entity_kind, n_min=1, n_max=3):
        result = []
        output_stds = []
        params = []
        train = agent.train
        agent.train = False
        for i in range(n_min, n_max + 1):
            if entity_kind == 'EXPLORER':
                next_params = self._generate_exporer_nearby(i)
            elif entity_kind == 'WANDERER':
                next_params = self._generate_wanderer_nearby(i)
            else:
                raise ValueError(f'unknown entity_kind: {entity_kind}')
            params += next_params
        for answer, explorers, wanderers in params:
            self._set_env(agent, explorers, wanderers)
            state, action = agent.generate_state_and_step(0)
            output_stds.append(agent.get_output_std())
            result.append(action in answer)
        agent.observer = None
        agent.train = train
        return np.mean(result), np.mean(output_stds)

    def _set_env(self, agent, explorers, wanderers):
        player_pos = (3, 3)
        obs = [
            None,
            f'EXPLORER 0 {player_pos[0]} {player_pos[1]} 100 0 0'
        ] + [
            f'EXPLORER {i + 1} {player_pos[0] + x} {player_pos[1] + y} 100 0 0'
            for i, (x, y) in enumerate(explorers) 
        ] + [
            f'WANDERER {i + 10} {player_pos[0] + x} {player_pos[1] + y} 10 1 0'
            for i, (x, y) in enumerate(wanderers) 
        ]
        self.env._set_entities(obs)
        self.env._set_players(obs, set_ids=True)
        agent.set_env(self.env)

    def _generate_exporer_nearby(self, n_step):
        return [
            (set([answer]), [(tuple(x * n_step for x in rel_pos))], [])
            for answer, rel_pos in enumerate(REL_POSITIONS[:-1])
        ]

    def _generate_wanderer_nearby(self, n_step):
        all_answers = set(range(4))
        return [
            (all_answers - set([answer]), [], [(tuple(x * n_step for x in rel_pos))])
            for answer, rel_pos in enumerate(REL_POSITIONS[:-1])
        ]
