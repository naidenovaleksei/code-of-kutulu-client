import gym
from gym import spaces

# from numba import int32, float32    # import the types
# from numba.experimental import jitclass

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Set, Any
import numpy as np
import requests
from functools import lru_cache
import warnings

from src.envs.distance import find_path, distance
from src.envs.kutulu_reward_manager import KutuluRewardManager
from src.envs.kutulu_entities import (
    KutuluEntity, KutuluPlayer, KutuluWanderer, KutuluSlasher, 
    EffectType, EntityKind, KutuluObservation,
)
from src.game.template import (
    get_valid_action_mask_by_coords,
    REL_POSITIONS,
    EXTENDED_KUTULU_ACTIONS,
    DEFAULT_KUTULU_ACTIONS,
)


@dataclass
class KutuluObservation:
    active_player_count: int
    entities: List[KutuluEntity]


@dataclass
class KutuluEnvInfo:
    lines: List[List[str]]
    width: int
    height: int
    maze_name: str


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
        self.entities: List[KutuluEntity] = []
        self.players: Dict[int, KutuluPlayer] = dict()
        self.players_ids: List[int] = []
        self.turn: int = 0
        self.yelled = defaultdict(set)
        
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
        assert data['mazeName'] == self.maze_name, f"{data['mazeName']} != {self.maze_name}"
        self.game_id = data['gameId']
        self.constants = data['constants']
        madness_per_turn_coef = self.reward_params.get('madness_per_turn_coef', 0)
        spread_madness_per_turn = self.constants['SPREAD_MADNESS_PER_TURN_AMOUNT'] * madness_per_turn_coef
        self.reward_manager = KutuluRewardManager(
            spread_madness_per_turn=spread_madness_per_turn,
            **self.reward_params
        )
        self.map = data['map']
        self.width = len(self.map[0])
        self.height = len(self.map)

        self._set_entities(data['state'])
        self._set_players(data['state'], set_ids=True)
        self.reward_manager.update_players(self._get_players_dict_format())

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
            
            # Apply effects using KutuluPlayer methods
            if i in self.players:
                player = self.players[i]
                if self._actions[action] == 'PLAN':
                    player.apply_effect(EffectType.PLAN, 4)
                elif self._actions[action] == 'LIGHT':
                    player.apply_effect(EffectType.LIGHT, 2)
                elif self._actions[action] == 'YELL':
                    is_yelled = False
                    for p_id, p in self.players.items():
                        if p_id != i and distance((player.x, player.y), (p.x, p.y)) <= 1:
                            self.yelled[i].add(p_id)
                            is_yelled = True
                    if not is_yelled:
                        warnings.warn("not is_yelled")

        response = requests.post(
            f'{self.host}/turn',
            json={
                'playerActions': player_actions,
                'gameId': self.game_id
            }
        )
        data = response.json()
        self.turn = data['turn']
        game_over = data['gameOver']
        self._set_entities(data['state'])

        reward = self.reward_manager.calculate_rewards(
            [e.to_dict() for e in self.entities], self.turn, game_over
        )

        self._set_players(data['state'])
        self.reward_manager.update_players(self._get_players_dict_format())

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, game_over, info

    def viz_map(self, action=None, agent_id=None):
        curr_map = [list(line) for line in self.map]
        for e in self.entities:
            curr_map[e.y][e.x] = e.kind[0]
            if e.kind == 'EXPLORER':
                curr_map[e.y][e.x] = str(e.id)
                if e.id == agent_id:
                    agent_pos = (e.x, e.y)
            else:
                curr_map[e.y][e.x] = e.kind[0]
        if action is not None:
            rel_pos = REL_POSITIONS[action]
            x = agent_pos[0] + rel_pos[0]
            y = agent_pos[1] + rel_pos[1]
            curr_map[y][x] = '^'
        for line in curr_map:
            print(''.join(line))
        print()

    def _get_info(self) -> Dict[str, Any]:
        return {
            # 'map': self.map,
            # 'constants': self.constants,
            'lines': self.map,
            'width': self.width, 
            'height': self.height,
            'maze_name': self.maze_name,
        }

    def _get_obs(self, player_id: int = None) -> KutuluObservation:
        return KutuluObservation(
            len(self.players),
            self._get_entites(player_id)
        )
    
    def _get_entites(self, player_id: int = None) -> List[KutuluEntity]:
        if player_id is None:
            return self.entities
        else:
            player_entity = None
            for e in self.entities:
                if e.kind == 'EXPLORER' and e.id == player_id:
                    player_entity = e
                    break
            assert player_entity is not None
            # Put specific player first, then other entities
            other_entities = [e for e in self.entities if not (e.kind == 'EXPLORER' and e.id == player_id)]
            return [player_entity] + other_entities

    def _convert_player_action(self, action) -> str:
        action_name = self._actions[action]
        return f"{action_name} {action_name}"

    def _parse_entity(self, line: str) -> KutuluEntity:
        """Parse entity string and return appropriate KutuluEntity object."""
        entity = KutuluEntity.from_string(line)
        
        # Create specialized entity types
        if entity.kind == EntityKind.EXPLORER.value:
            return KutuluPlayer.from_entity_string(line)
        elif entity.kind == EntityKind.WANDERER.value:
            return KutuluWanderer(
                entity.id, entity.x, entity.y,
                entity.param0, entity.param1, entity.param2
            )
        elif entity.kind == EntityKind.SLASHER.value:
            return KutuluSlasher(
                entity.id, entity.x, entity.y,
                entity.param0, entity.param1, entity.param2
            )
        else:
            # Other entity types (effects, etc.) - return base KutuluEntity
            return entity


    def _set_entities(self, state) -> None:
        self.entities = [
            self._parse_entity(e) for e in state[1:]
        ]
    
    def _check_yell(self, player):
        for p_id, p in self.players.items():
            if p_id != player.id and \
                p_id not in self.yelled[player.id] and \
                distance(
                    (player.x, player.y), (p.x, p.y)
                ) <= 1:
                return True
        return False

    def _set_players(self, state, set_ids=False) -> None:
        if set_ids:
            self.players_ids = []
        
        # Mark existing players as inactive
        for player in self.players.values():
            player.active = False
        
        for e in state[1:]:
            if e.startswith('EXPLORER'):
                # Create KutuluPlayer object from entity string
                kutulu_player = KutuluPlayer.from_entity_string(e)
                
                # Preserve effect_left from previous state if player exists
                if kutulu_player.id in self.players:
                    existing_player = self.players[kutulu_player.id]
                    kutulu_player.effect_left = max(0, existing_player.effect_left - 1)
                
                self.players[kutulu_player.id] = kutulu_player
                
                if set_ids:
                    self.players_ids.append(kutulu_player.id)

        for player in self.players.values():
            player.can_yell = self._check_yell(player)

    def get_valid_action_mask(self) -> Dict[int, List[bool]]:
        return {
            player_id: get_valid_action_mask_by_coords(
                player.to_dict(), self._get_info(),
            )[:len(self._actions)]
            for player_id, player in self.players.items()
        }

    def sample_valid_action(self, seed=None) -> List[int]:
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

    def get_obs(self, player_id: int = None) -> KutuluObservation:
        return KutuluObservation(
            len(self.players),
            self._get_entites(player_id)
        )

    def get_info(self) -> KutuluEnvInfo:
        return KutuluEnvInfo(
            self.map,
            self.width,
            self.height,
            self.maze_name,
        )

    @lru_cache(maxsize=10000)
    def find_path_cached(self, start_point, finish_point):
        return find_path(start_point, finish_point, self.map)

    def active_players(self) -> Set[int]:
        return {player_id for player_id, player in self.players.items() if player.active}

    def _get_players_dict_format(self) -> Dict[int, Dict]:
        return {player_id: player.to_dict() for player_id, player in self.players.items()}

    def is_game_over_for_player(self, player_id) -> bool:
        return self.reward_manager.death_turns.get(player_id) == self.turn
