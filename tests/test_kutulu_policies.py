import pytest
import pickle as pkl
import numpy as np
from unittest.mock import MagicMock, patch
from src.envs.distance import find_path
from src.envs.kutulu_player import KutuluPlayer
from src.envs.kutulu_world import KutuluWorldEnv, CELL_WALL, DEFAULT_KUTULU_ACTIONS
from src.envs.strategy import LazyGreedyStrategy

MODEL_DIR = "output/2025-04-30/20:47:29.344655"

class TestKutuluPolicies:
    @pytest.fixture
    def mock_env(self):
        """Create a mock KutuluWorldEnv instance with a predefined map."""
        mock_env = MagicMock(spec=KutuluWorldEnv)
        
        # Sample map for testing
        mock_env.map = [
            '###########',
            '#.........#',
            '#.#.#.#.#.#',
            '#.........#',
            '#.#.#.#.#.#',
            '#.........#',
            '###########',
        ]
        mock_env.width = len(mock_env.map[0])
        mock_env.height = len(mock_env.map)
        
        def get_valid_action_mask_by_coords(x, y, can_wait=True):
            return [
                y > 0 and mock_env.map[y - 1][x] != CELL_WALL,
                x < mock_env.width - 1 and mock_env.map[y][x + 1] != CELL_WALL,
                y < mock_env.height - 1 and mock_env.map[y + 1][x] != CELL_WALL,
                x > 0 and mock_env.map[y][x - 1] != CELL_WALL,
                can_wait
            ]
        mock_env._get_valid_action_mask_by_coords = get_valid_action_mask_by_coords
        
        # Mock find_path_cached function to return predefined paths
        mock_env.find_path_cached = lambda a, b: find_path(a, b, mock_env.map)
        
        return mock_env
    
    @pytest.fixture
    def player(self, mock_env):
        """Create a KutuluPlayer instance with the mock environment."""
        return KutuluPlayer(mock_env)

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
    def test_get_state_with_one_explorer(self, player_pos, explorers, strategy, player, mock_env):
        """Test get_state with player and one other explorer."""
        # Setup mock _get_obs to return player and another explorer
        
        player_id = 0
        mock_env._get_obs.return_value = {
            'entities': [
                {'kind': 'EXPLORER', 'id': player_id, 'x': player_pos[0], 'y': player_pos[1]},
            ] + [
                {'kind': 'EXPLORER', 'id': i + 1, 'x': x, 'y': y}
                for i, (x, y) in enumerate(explorers) 
            ]
        }
        
        # Call get_state
        state = player.get_state(player_id)
        print(state)
        player_mask = ~np.array(mock_env._get_valid_action_mask_by_coords(player_pos[0], player_pos[1]))
        print(player_mask)
        At = strategy.getActionGreedyMasked(state, 5, player_mask)
        
        assert At == 1
