import gym
from gym import spaces

# from numba import int32, float32    # import the types
# from numba.experimental import jitclass

import numpy as np
import requests
from functools import lru_cache

from src.envs.distance import find_path
from src.envs.kutulu_reward_manager import KutuluRewardManager
from src.game.template import (
    get_valid_action_mask_by_coords,
    REL_POSITIONS,
)


class KutuluWorldEnv(gym.Env):
    def __init__(self, server_host, maze_name, league_level, actions, reward_params={}, players_count=4):
        super(KutuluWorldEnv, self).__init__()

        self.host = f"http://{server_host}/game"

        self.maze_name = maze_name
        self.league_level = league_level
        self.players_count = players_count
        self.reward_params = reward_params

        self._actions = actions
        self.map = []
        self.entities = []
        self.players = dict()
        self.players_ids = []
        self.turn = 0
        
        self.seed = None
        self.constants = {}
        self.reward_manager = None
        self.width = 0
        self.height = 0

    def reset(self, seed=None):
        if seed is None:
            seed = np.random.randint(999999)
        super(KutuluWorldEnv, self).reset(seed=seed)
        response = requests.post(
            f'{self.host}/create',
            json={
                "playerCount": self.players_count,
                "mazeName": self.maze_name,
                "seed": seed,
                "leagueLevel": self.league_level
            }
        )
        self.seed = seed
        data = response.json()
        self.constants = data['constants']
        self.reward_manager = KutuluRewardManager(
            spread_madness_per_turn=self.constants['SPREAD_MADNESS_PER_TURN_AMOUNT'],
            **self.reward_params
        )
        self.map = data['map']
        self.width = len(self.map[0])
        self.height = len(self.map)

        self._set_entities(data['state'])
        self._set_players(data['state'], set_ids=True)
        self.reward_manager.update_players(self.players)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, actions):
        player_actions = []
        for i, action in enumerate(actions):
            action_state = {
                'playerId': i,
                'action': self._convert_player_action(action)
            }
            player_actions.append(action_state)
            if self._actions[action] == 'PLAN':
                self.players[i]['effect_left'] = 4
            elif self._actions[action] == 'LIGHT':
                self.players[i]['effect_left'] = 2

        response = requests.post(
            f'{self.host}/turn',
            json={'playerActions': player_actions}
        )
        data = response.json()
        self.turn = data['turn']
        game_over = data['gameOver']
        self._set_entities(data['state'])

        reward = self.reward_manager.calculate_rewards(self.entities, self.turn, game_over)

        self._set_players(data['state'])
        self.reward_manager.update_players(self.players)

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, game_over, info

    def viz_map(self, action=None, agent_id=None):
        curr_map = [list(line) for line in self.map]
        for e in self.entities:
            curr_map[e['y']][e['x']] = e['kind'][0]
            if e['kind'] == 'EXPLORER':
                curr_map[e['y']][e['x']] = str(e['id'])
                if e['id'] == agent_id:
                    agent_pos = (e['x'], e['y'])
        if action is not None:
            rel_pos = REL_POSITIONS[action]
            x = agent_pos[0] + rel_pos[0]
            y = agent_pos[1] + rel_pos[1]
            curr_map[y][x] = '^'
        for line in curr_map:
            print(''.join(line))
        print()

    def _get_info(self):
        return {
            # 'map': self.map,
            # 'constants': self.constants,
            'lines': self.map,
            'width': self.width, 
            'height': self.height,
            'maze_name': self.maze_name,
        }

    def _get_obs(self, player_id=None):
        return {
            'active_player_count': len(self.players),
            'entities': self._get_entites(player_id)
        }
    
    def _get_entites(self, player_id=None):
        if player_id is None:
            return self.entities
        else:
            return [
                next(e for e in self.entities if e['kind'] == 'EXPLORER' and e['id'] == player_id)
            ] + [
               e for e in self.entities if not (e['kind'] == 'EXPLORER' and e['id'] == player_id)
            ]

    def _convert_player_action(self, action):
        action_name = self._actions[action]
        return f"{action_name} {action_name}"

    def _parse_entity(self, line):
        ekind, eid, ex, ey, eparam0, eparam1, eparam2 = line.split()
        result = {
            'kind': ekind,
            'id': int(eid),
            'x': int(ex),
            'y': int(ey),
            "param0": int(eparam0),
            "param1": int(eparam1),
            "param2": int(eparam2),
        }

        if ekind == 'EXPLORER':
            result['sanity'] = int(eparam0)
        elif ekind == 'WANDERER':
            result['wandering'] = int(eparam1)
            result['target'] = int(eparam2)
            if result['wandering']:
                # time before being recalled
                result['recall_time_left'] = int(eparam0)
            else:
                # time before spawn
                result['spawn_time_left'] = int(eparam0)
        elif ekind == 'SLASHER':
            # time before changing state
            result['change_time_left'] = int(eparam0)
        elif ekind in ('EFFECT_PLAN', 'EFFECT_LIGHT', 'EFFECT_YELL', ):
            # time before changing state
            result['out_time_left'] = int(eparam0)
        elif ekind == 'EFFECT_SHELTER':
            # time before changing state
            result['energy_left'] = int(eparam0)
        else:
            raise ValueError(f'unknown kind: {ekind}')
        return result

    def _parse_player(self, line):
        player = self._parse_entity(line)
        player['active'] = True
        player['effect_left'] = 0
        return player

    def _set_entities(self, state):
        self.entities = [
            self._parse_entity(e) for e in state[1:]
        ]

    def _set_players(self, state, set_ids=False):
        if set_ids:
            self.players_ids = []
        for player in self.players.values():
            player['active'] = False
        for e in state[1:]:
            if e.startswith('EXPLORER'):
                player = self._parse_player(e)
                if player['id'] in self.players:
                    player['effect_left'] = max(0, self.players[player['id']]['effect_left'] - 1)
                self.players[player['id']] = player
                if set_ids:
                    self.players_ids.append(player['id'])

    def get_valid_action_mask(self):
        return {
            player_id: get_valid_action_mask_by_coords(
                player, self._get_info(),
            )[:len(self._actions)]
            for player_id, player in self.players.items()
        }

    def sample_valid_action(self, seed=None):
        mask_dict = self.get_valid_action_mask()
        mask = [mask_dict[player_id] for player_id in self.players_ids]
        if seed is not None:
            prng = np.random.RandomState(seed)
            return [
                prng.choice(np.where(np.array(player_mask))[0])
                for player_mask in mask
            ]
        return [
            np.random.choice(np.where(np.array(player_mask))[0])
            for player_mask in mask
        ]

    @lru_cache(maxsize=10000)
    def find_path_cached(self, start_point, finish_point):
        return find_path(start_point, finish_point, self.map)

    def active_players(self):
        # return [player for player in self.players.values() if player['active']]
        return {player_id for player_id, player in self.players.items() if player['active']}

    def is_game_over_for_player(self, player_id):
        return self.reward_manager.death_turns.get(player_id) == self.turn