import pytest
from unittest.mock import MagicMock, patch
from src.envs.distance import find_path
from src.envs.kutulu_player import KutuluPlayer
from src.envs.kutulu_world import KutuluWorldEnv

class TestKutuluPlayer:
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
        
        # Mock find_path_cached function to return predefined paths
        mock_env.find_path_cached = lambda a, b: find_path(a, b, mock_env.map)
        
        return mock_env
    
    @pytest.fixture
    def player(self, mock_env):
        """Create a KutuluPlayer instance with the mock environment."""
        return KutuluPlayer(mock_env)
    
    def test_get_state_with_only_player(self, player, mock_env):
        """Test get_state with only the player entity present."""
        # Setup mock _get_obs to return only the player
        mock_env._get_obs.return_value = {
            'entities': [
                {'type': 'EXPLORER', 'id': 0, 'x': 3, 'y': 3}
            ]
        }
        
        # Call get_state
        result = player.get_state(0)
        
        # Since there are no other entities, all directions and distances should be None
        assert result == (None, None, None, None)
        mock_env._get_obs.assert_called_once_with(0)
    
    def test_get_state_with_explorer(self, player, mock_env):
        """Test get_state with player and one other explorer."""
        # Setup mock _get_obs to return player and another explorer
        mock_env._get_obs.return_value = {
            'entities': [
                {'type': 'EXPLORER', 'id': 0, 'x': 3, 'y': 3},
                {'type': 'EXPLORER', 'id': 1, 'x': 5, 'y': 3}
            ]
        }
        
        # Call get_state
        result = player.get_state(0)
        
        # Explorer is to the right (index 1)
        assert result[0] == (1,)  # Explorer direction
        assert result[1] == 2     # Explorer distance
        assert result[2] is None  # No wanderer
        assert result[3] is None  # No wanderer distance
    
    def test_get_state_with_wanderer(self, player, mock_env):
        """Test get_state with player and one wanderer."""
        # Setup mock _get_obs to return player and a wanderer
        mock_env._get_obs.return_value = {
            'entities': [
                {'type': 'EXPLORER', 'id': 0, 'x': 3, 'y': 3},
                {'type': 'WANDERER', 'id': 2, 'x': 3, 'y': 1, 'wandering': 1}
            ]
        }

        # Call get_state
        result = player.get_state(0)

        # Wanderer is above (index 0)
        assert result[0] is None  # No explorer
        assert result[1] is None  # No explorer distance
        assert result[2] == (0,)  # Wanderer direction
        assert result[3] == 2     # Wanderer distance
    
    def test_get_state_the_same_position(self, player, mock_env):
        """Test get_state with player and one other explorer."""
        # Setup mock _get_obs to return player and another explorer
        mock_env._get_obs.return_value = {
            'entities': [
                {'type': 'EXPLORER', 'id': 0, 'x': 3, 'y': 3},
                {'type': 'EXPLORER', 'id': 1, 'x': 3, 'y': 3},
                {'type': 'WANDERER', 'id': 2, 'x': 3, 'y': 3, 'wandering': 1}
            ]
        }
        
        # Call get_state
        result = player.get_state(0)
        
        # Explorer is to the right (index 1)
        assert result[0] == (4,)  # Explorer direction
        assert result[1] == 0     # Explorer distance
        assert result[2] == (4,)  # No wanderer
        assert result[3] == 0     # No wanderer distance
    
    def test_get_state_wanderers_around_1(self, player, mock_env):
        """Test get_state with player and one other explorer."""
        # Setup mock _get_obs to return player and another explorer
        mock_env._get_obs.return_value = {
            'entities': [
                {'type': 'EXPLORER', 'id': 0, 'x': 3, 'y': 3},
                {'type': 'WANDERER', 'id': 0, 'x': 2, 'y': 3, 'wandering': 1},
                {'type': 'WANDERER', 'id': 1, 'x': 3, 'y': 2, 'wandering': 1},
                {'type': 'WANDERER', 'id': 2, 'x': 4, 'y': 3, 'wandering': 1},
                {'type': 'WANDERER', 'id': 3, 'x': 3, 'y': 4, 'wandering': 1},
            ]
        }
        
        # Call get_state
        result = player.get_state(0)
        
        # Explorer is to the right (index 1)
        assert result[0] is None  # Explorer direction
        assert result[1] is None  # Explorer distance
        assert result[2] == (0, 1, 2, 3,)  # No wanderer
        assert result[3] == 1     # No wanderer distance
    
    def test_get_state_wanderers_around_2(self, player, mock_env):
        """Test get_state with player and one other explorer."""
        # Setup mock _get_obs to return player and another explorer
        mock_env._get_obs.return_value = {
            'entities': [
                {'type': 'EXPLORER', 'id': 0, 'x': 3, 'y': 3},
                {'type': 'WANDERER', 'id': 0, 'x': 1, 'y': 3, 'wandering': 1},
                {'type': 'WANDERER', 'id': 1, 'x': 3, 'y': 1, 'wandering': 1},
                {'type': 'WANDERER', 'id': 2, 'x': 5, 'y': 3, 'wandering': 1},
                {'type': 'WANDERER', 'id': 3, 'x': 3, 'y': 5, 'wandering': 1},
            ]
        }
        
        # Call get_state
        result = player.get_state(0)
        
        # Explorer is to the right (index 1)
        assert result[0] is None  # Explorer direction
        assert result[1] is None  # Explorer distance
        assert result[2] == (0, 1, 2, 3,)  # No wanderer
        assert result[3] == 2     # No wanderer distance
    
    def test_get_state_with_multiple_entities(self, player, mock_env):
        """Test get_state with player, explorers, and wanderers."""
        # Setup mock _get_obs to return player, explorers, and wanderers
        mock_env._get_obs.return_value = {
            'entities': [
                {'type': 'EXPLORER', 'id': 0, 'x': 3, 'y': 3},
                {'type': 'EXPLORER', 'id': 1, 'x': 5, 'y': 3},
                {'type': 'EXPLORER', 'id': 2, 'x': 1, 'y': 3},
                {'type': 'WANDERER', 'id': 3, 'x': 3, 'y': 1, 'wandering': 1},
                {'type': 'WANDERER', 'id': 4, 'x': 3, 'y': 5, 'wandering': 1},
                {'type': 'WANDERER', 'id': 5, 'x': 3, 'y': 2, 'wandering': 0}  # Not wandering
            ]
        }

        # Call get_state
        result = player.get_state(0)
        
        # Explorer is to the left (index 3) with distance 1
        # Wanderer is up (index 0) with distance 2
        assert result[0] == (1, 3)         # Either left or right explorer is closest
        assert result[1] == 2              # Distance to closest explorer
        assert result[2] == (0, 2)         # Closest wanderer is up
        assert result[3] == 2              # Distance to closest wanderer
    
    def test_get_state_with_max_distance_limits(self, player, mock_env):
        """Test get_state with max distance limits for explorers and wanderers."""
        # Setup mock _get_obs to return player, explorers, and wanderers
        mock_env._get_obs.return_value = {
            'entities': [
                {'type': 'EXPLORER', 'id': 0, 'x': 3, 'y': 3},
                {'type': 'EXPLORER', 'id': 1, 'x': 5, 'y': 3},
                {'type': 'WANDERER', 'id': 2, 'x': 3, 'y': 1, 'wandering': 1}
            ]
        }
        
        # Call get_state with max distance limits
        result = player.get_state(0, max_explorer_dist=0, max_wanderer_dist=1)
        
        # Check that distances are capped
        assert result[0] == (1,)  # Explorer is to the right
        assert result[1] == 0     # Explorer distance capped at 0
        assert result[2] == (0,)  # Wanderer is up
        assert result[3] == 1     # Wanderer distance capped at 1
