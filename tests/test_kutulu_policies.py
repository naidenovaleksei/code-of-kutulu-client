import pytest
import pickle as pkl
import numpy as np
from unittest.mock import MagicMock, patch
from src.envs.distance import find_path
from src.envs.kutulu_observer import KutuluClosestObserver
from src.envs.kutulu_world import KutuluWorldEnv
from src.envs.strategy import LazyGreedyStrategy
from src.game.template import CELL_WALL, DEFAULT_KUTULU_ACTIONS

MODEL_DIR = "output/2025-04-30/20:47:29.344655"

class TestKutuluPolicies:
    @pytest.fixture
    def env(self):
        """Create a mock KutuluWorldEnv instance with a predefined map."""
        env = KutuluWorldEnv('', '', 1, actions=DEFAULT_KUTULU_ACTIONS)
        env.map = [
            '###########',
            '#.........#',
            '#.#.#.#.#.#',
            '#.........#',
            '#.#.#.#.#.#',
            '#.........#',
            '###########',
        ]
        env.width = len(env.map[0])
        env.height = len(env.map)
        env.find_path_cached = lambda a, b: find_path(a, b, env.map)
        return env

    @pytest.fixture
    def observer(self, env):
        """Create a KutuluClosestObserver instance with the mock environment."""
        return KutuluClosestObserver(env)

    @pytest.fixture
    def strategy(self):
        with open(f"{MODEL_DIR}/data1.pkl", "rb") as f:
            data1 = f.read()
        with open(f"{MODEL_DIR}/data2.pkl", "rb") as f:
            data2 = f.read()
        
        Q = dict(zip(pkl.loads(data2), pkl.loads(data1)))
        
        pi = LazyGreedyStrategy()
        pi.Q = Q
        
        return pi
    
    @pytest.mark.parametrize("player_pos, explorers", [
        [(3, 3), [(4, 3)]],
    ])
    def test_get_state_with_one_explorer(self, player_pos, explorers, strategy, observer):
        """Test get_state with player and one other explorer."""
        # Setup mock _get_obs to return player and another explorer
        
        player_id = 0
        obs = [
            None,
            f'EXPLORER 0 {player_pos[0]} {player_pos[1]} 0 0 0'
        ] + [
            f'EXPLORER {i + 1} {x} {y} 0 0 0'
            for i, (x, y) in enumerate(explorers) 
        ]
        observer.env._set_entities(obs)
        observer.env._set_players(obs, set_ids=True)
        
        # Call get_state
        state = observer.get_state(player_id)
        valid_actions = observer.env.get_valid_action_mask()[player_id]
        player_mask = ~np.array(valid_actions)
        At = strategy.getActionGreedyMasked(state, 5, player_mask)
        
        assert At == 1
