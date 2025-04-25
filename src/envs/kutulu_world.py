import gym
from gym import spaces

# from numba import int32, float32    # import the types
# from numba.experimental import jitclass

import numpy as np
import requests
from functools import lru_cache

from src.envs.distance import find_path

DEFAULT_KUTULU_ACTIONS = [
    'UP',
    'RIGHT',
    'DOWN',
    'LEFT',
    'WAIT',
]

CELL_WALL = '#'

class KutuluWorldEnv(gym.Env):
    def __init__(self, server_host, maze_name, league_level, players_count=4,
                 actions=DEFAULT_KUTULU_ACTIONS):
        super(KutuluWorldEnv, self).__init__()

        self.host = f"http://{server_host}/game"

        self.maze_name = maze_name
        self.league_level = league_level
        self.players_count = players_count
        
        self._actions = actions
        self.map = []

        self.action_space = spaces.Dict({
            i: spaces.Discrete(len(actions))
            for i in range(self.players_count)
        }) 
        
        # # Observation space: agent position (x, y)
        # self.observation_space = spaces.Box(
        #     low=0, high=self.grid_size-1, 
        #     shape=(2,), dtype=np.int32
        # )

    def _get_obs(self, player_id=None):
        return {
            'active_player_count': len(self.players),
            'map': self.map,
            'entities': self._get_entites(player_id)
        }
    
    def _get_entites(self, player_id=None):
        if player_id is None:
            return self.entities
        else:
            return [
                next(e for e in self.entities if e['type'] == 'EXPLORER' and e['id'] == player_id)
            ] + [
               e for e in self.entities if not (e['type'] == 'EXPLORER' and e['id'] == player_id)
            ]

    def _get_info(self):
        return {
            'constants': self.constants,
        }

    def _get_valid_action_mask_by_coords(self, x, y, can_wait=True):
        # 'UP',
        # 'RIGHT',
        # 'DOWN',
        # 'LEFT',
        # 'WAIT',
        return [
            y > 0 and self.map[y - 1][x] != CELL_WALL,
            x < self.width - 1 and self.map[y][x + 1] != CELL_WALL,
            y < self.height - 1 and self.map[y + 1][x] != CELL_WALL,
            x > 0 and self.map[y][x - 1] != CELL_WALL,
            can_wait
        ]

    def _get_valid_action_mask(self, can_wait=True):
        return [
            self._get_valid_action_mask_by_coords(player['x'], player['y'], can_wait)
            for player in self.players
        ]

    def _convert_player_action(self, action):
        action_name = self._actions[action]
        return f"{action_name} {action_name}"
    
    def _parse_entity(self, line):
        etype, eid, ex, ey, eparam0, eparam1, eparam2 = line.split()
        result = {
            'type': etype,
            'id': int(eid),
            'x': int(ex),
            'y': int(ey),
        }
        
        if etype == 'EXPLORER':
            result['sanity'] = int(eparam0)
        elif etype == 'WANDERER':
            result['wandering'] = int(eparam1)
            result['target'] = int(eparam2)
            if result['wandering']:
                result['wandering_left'] = int(eparam0)
            else:
                result['spawn_left'] = int(eparam0)
        else:
            raise ValueError()
        return result
    
    def _parse_player(self, line):
        etype, eid, ex, ey, esanity, eplans, elight = line.split()
        return {
            'id': eid,
            'x': int(ex),
            'y': int(ey),
            'sanity': float(esanity),
            'remainingPlans': int(eplans),
            'remainingLights': int(elight),
            'active': True,
        #     'score': 0
        }

    def sample_valid_action(self, seed=None, can_wait=True):
        mask = self._get_valid_action_mask(can_wait)
        if seed:
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
    def _find_path_cached(self, start_point, finish_point):
        return find_path(start_point, finish_point, self.map)

    def reset(self, seed=0):
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
        data = response.json()
        # self.config = data['config']
        self.constants = data['constants']
        self.map = data['map']
        self.width = len(self.map[0])
        self.height = len(self.map)
        self.game_id = data['gameId']
        
        self.entities = [
            self._parse_entity(e) for e in data['state'][1:]
        ]
        self.players = [
            self._parse_player(e) for e in data['state'][1:]
            if e.startswith('EXPLORER')
        ]
        self.death_turns = {}
        self.turn = 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, actions):
        player_actions = [
            {
                'playerId': i,
                'action': self._convert_player_action(action)
            }
            for i, action in enumerate(actions)
        ]
        response = requests.post(
            f'{self.host}/turn',
            json={'playerActions': player_actions}
        )
        data = response.json()
        # self.players = data['players']
        self.turn = data['turn']
        game_over = data['gameOver']

        self.entities = [
            self._parse_entity(e) for e in data['state'][1:]
        ]
        players = [
            self._parse_player(e) for e in data['state'][1:]
            if e.startswith('EXPLORER')
        ]
        active_sanity = {
            player['id']: player['sanity'] for player in players
        }
        active_pos = {
            player['id']: (player['x'], player['y']) for player in players
        }
        rewards = {}
        for player in self.players:
            player['active'] = player['id'] in active_sanity
            if player['active']:
                rewards[player['id']] = active_sanity[player['id']] - player['sanity'] + 1
                player['sanity'] = active_sanity[player['id']]
                player['x'], player['y'] = active_pos[player['id']]
            else:
                if player['id'] not in self.death_turns:
                    self.death_turns[player['id']] = self.turn

        if game_over:
            for player in self.players:
                if player['active']:
                    rewards[player['id']] += self.turn
            winner = max(self.players, key=lambda x: x['sanity'])['id']
            rewards[winner] = rewards.get(winner, 0) + 1000
                
        
        reward = [rewards.get(player['id']) for player in self.players]
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, reward, game_over, info

    def active_players(self):
        return [player for player in self.players if player['active']]
