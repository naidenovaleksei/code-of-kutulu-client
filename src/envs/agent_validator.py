from collections import Counter
import numpy as np

from src.envs.kutulu_world import KutuluWorldEnv
from src.game.template import MOVE_REL_POS, REL_POSITIONS, EXTENDED_KUTULU_ACTIONS

def get_normal_env(actions):
        env = KutuluWorldEnv('', '', 1, actions=actions)
        env.map = [
            #012345678
            '#########', # 0
            '#..#.#..#', # 1
            '#.......#', # 2
            '#..#.#..#', # 3
            '#.......#', # 4
            '#..#.#..#', # 5
            '#.......#', # 6
            '#..#.#..#', # 7
            '#########', # 8
        ]
        env.width = len(env.map[0])
        env.height = len(env.map)
        return env

def get_coridor_env(actions):
        env = KutuluWorldEnv('', '', 1, actions=actions)
        env.map = [
            #012345678
            '#########', # 0
            '#..#.#..#', # 1
            '#.......#', # 2
            '#.#####.#', # 3
            '#.......#', # 4
            '#.#####.#', # 5
            '#.......#', # 6
            '#..#.#..#', # 7
            '#########', # 8
        ]
        env.width = len(env.map[0])
        env.height = len(env.map)
        return env

def get_corner_env(actions):
        env = KutuluWorldEnv('', '', 1, actions=actions)
        env.map = [
            #012345678
            '#########', # 0
            '#..#.#..#', # 1
            '#.......#', # 2
            '#.#####.#', # 3
            '#.##....#', # 4
            '#.##.##.#', # 5
            '#..#.#..#', # 6
            '#.......#', # 7
            '#########', # 8
        ]
        env.width = len(env.map[0])
        env.height = len(env.map)
        return env


class AgentValidator:
    def __init__(self, actions, player_pos=(4, 4), player_params=(100,0,0),
                 explorers_params=(100,0,0), wanderers_params=(10,1,0)):
        self.normal_env = get_normal_env(actions)
        self.horizontal_coridor_env = self._rotate_env(get_coridor_env(actions), 0) 
        self.vertical_coridor_env = self._rotate_env(get_coridor_env(actions), 1)  
        self.top_left_coridor_env = self._rotate_env(get_corner_env(actions), 0) 
        self.top_right_coridor_env = self._rotate_env(get_corner_env(actions), 3) 
        self.down_right_coridor_env = self._rotate_env(get_corner_env(actions), 2) 
        self.down_left_coridor_env = self._rotate_env(get_corner_env(actions), 1)     
        self.player_pos = player_pos
        self.player_params = player_params
        self.explorers_params = explorers_params
        self.wanderers_params = wanderers_params
        self.plan_action = EXTENDED_KUTULU_ACTIONS.index('PLAN')
    
    def check_entity_nearby(self, agent, entity_kind, n_min=1, n_max=3, env_type='normal', verbose=False):
        result = []
        output_stds = []
        output_maxs = []
        output_means = []
        actions = []
        train = agent.train
        agent.train = False
        if env_type == 'normal':
            env_list = [
                self.normal_env
            ]
            params_list = [
                self._get_params(entity_kind, n_min, n_max, ['UP', 'RIGHT', 'DOWN', 'LEFT'])
            ]
        elif env_type == 'coridor':
            env_list = [
                self.horizontal_coridor_env,
                self.vertical_coridor_env,
            ]
            params_list = [
                self._get_params(entity_kind, n_min, n_max, ['RIGHT', 'LEFT']),
                self._get_params(entity_kind, n_min, n_max, ['UP', 'DOWN']),
            ]
        elif env_type == 'corner':
            env_list = [
                self.top_left_coridor_env,
                self.top_right_coridor_env,
                self.down_right_coridor_env,
                self.down_left_coridor_env,
            ]
            params_list = [
                self._get_params(entity_kind, n_min, n_max, ['DOWN', 'RIGHT']),
                self._get_params(entity_kind, n_min, n_max, ['LEFT', 'DOWN']),
                self._get_params(entity_kind, n_min, n_max, ['UP', 'LEFT']),
                self._get_params(entity_kind, n_min, n_max, ['RIGHT', 'UP']),
            ]
        else:
            raise ValueError(f'wrong type: {env_type}')
        for env, params in zip(env_list, params_list):
            for answer, explorers, wanderers in params:
                self._set_env(env, agent, explorers, wanderers)
                _, action = agent.generate_state_and_step(0, need_update=False)
                last_action = agent.get_last_action()
                output_stds.append(last_action.std())
                output_maxs.append(last_action.max())
                output_means.append(last_action.mean())
                result.append(action in answer)
                actions.append(action.item())
                if verbose:
                    print(f'answer: {answer}, action: {action}, explorers: {explorers}, wanderers: {wanderers}')
                    print(f'action: {result[-1]}, std: {output_stds[-1]}')
                    env.viz_map(action=action, agent_id=0)
                    print()
        agent.observer = None
        agent.train = train
        if verbose:
            return result, output_stds, actions
        ad = dict(Counter(actions))
        max_v = max(ad.values())
        max_ad = {k for k,v in ad.items() if v == max_v}
        top_action = len(max_ad)
        return np.mean(result), np.mean(output_stds), top_action, np.max(output_maxs), np.mean(output_means)
    
    def _get_params(self, entity_kind, n_min, n_max, actions):
        params = []
        for i in range(n_min, n_max + 1):
            if entity_kind == 'EXPLORER':
                next_params = self._generate_exporer_nearby(i, actions)
            elif entity_kind == 'WANDERER':
                next_params = self._generate_wanderer_nearby(i, actions)
            else:
                raise ValueError(f'unknown entity_kind: {entity_kind}')
            params += next_params
        return params

    def _rotate_env(self, env, k):
        if k == 0:
            return env
        env.map = [''.join(s) for s in np.rot90([list(s) for s in env.map], k=k)]
        env.width = len(env.map[0])
        env.height = len(env.map)
        return env

    def _set_env(self, env, agent, explorers, wanderers):
        player_pos = self.player_pos
        assert len(self.player_params) == 3
        assert len(self.explorers_params) == 3
        assert len(self.wanderers_params) == 3
        player_params = " ".join(map(str, self.player_params))
        explorers_params = " ".join(map(str, self.explorers_params))
        wanderers_params = " ".join(map(str, self.wanderers_params))
        obs = [
            None,
            f'EXPLORER 0 {player_pos[0]} {player_pos[1]} {player_params}'
        ] + [
            f'EXPLORER {i + 1} {player_pos[0] + x} {player_pos[1] + y} {explorers_params}'
            for i, (x, y) in enumerate(explorers) 
        ] + [
            f'WANDERER {i + 10} {player_pos[0] + x} {player_pos[1] + y} {wanderers_params}'
            for i, (x, y) in enumerate(wanderers) 
        ]
        env._set_entities(obs)
        env._set_players(obs, set_ids=True)
        agent.set_env(env)

    def _generate_exporer_nearby(self, n_step, moves):
        moves = set([MOVE_REL_POS[move] for move in moves])
        has_plan = self.player_params[1] > 0
        plan_actions = [self.plan_action] if has_plan and n_step <= 2 else []
        return [
            (set([answer] + plan_actions), [(tuple(x * n_step for x in rel_pos))], [])
            for answer, rel_pos in enumerate(REL_POSITIONS)
            if rel_pos in moves
        ]

    def _generate_wanderer_nearby(self, n_step, moves):
        moves = set([MOVE_REL_POS[move] for move in moves])
        all_answers = set(range(4))
        return [
            (all_answers - set([answer]), [], [(tuple(x * n_step for x in rel_pos))])
            for answer, rel_pos in enumerate(REL_POSITIONS)
            if rel_pos in moves
        ]
