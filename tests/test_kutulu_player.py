import pytest
from tests.utils import calculate_entities
from src.envs.distance import find_path
from src.envs.kutulu_observer import KutuluClosestObserver
from src.envs.kutulu_world import KutuluWorldEnv
from src.game.template import DEFAULT_KUTULU_ACTIONS

    
class TestKutuluClosestObserver:
    @pytest.fixture
    def observer(self, mock_env):
        """Create a KutuluClosestObserver instance with the mock environment."""
        return KutuluClosestObserver(mock_env)
    
    @pytest.mark.parametrize("player_pos, explorers, state", [
        [(3, 3), [], (None, None, None, None)],
    ])
    def test_get_state_with_only_player(self, player_pos, explorers, state, observer):
        """Test get_state with only the player entity present."""
        entities = calculate_entities(player_pos, explorers=explorers)
        observer.env._set_entities(entities)
        observer.env._set_players(entities, set_ids=True)
        # Call get_state
        result = observer.get_state(0)
        # Since there are no other entities, all directions and distances should be None
        assert result == state

    @pytest.mark.parametrize("player_pos, explorers, state", [
        [(3, 3), [(3, 2)], ((0,), 1, None, None)],
        [(3, 3), [(5, 3)], ((1,), 2, None, None)],
        [(3, 3), [(4, 3)], ((1,), 1, None, None)],
    ])
    def test_get_state_with_explorer(self, player_pos, explorers, state, observer):
        """Test get_state with player and one other explorer."""
        entities = calculate_entities(player_pos, explorers=explorers)
        observer.env._set_entities(entities)
        observer.env._set_players(entities, set_ids=True)
        # Call get_state
        result = observer.get_state(0)
        assert result == state

    @pytest.mark.parametrize("player_pos, wanderers", [
        [(3, 3), [(3, 1, 1)]],
    ])
    def test_get_state_with_wanderer(self, player_pos, wanderers, observer):
        """Test get_state with player and one wanderer."""
        entities = calculate_entities(player_pos, wanderers=wanderers)
        observer.env._set_entities(entities)
        observer.env._set_players(entities, set_ids=True)
        # Call get_state
        result = observer.get_state(0)

        # Wanderer is above (index 0)
        assert result[0] is None  # No explorer
        assert result[1] is None  # No explorer distance
        assert result[2] == (0,)  # Wanderer direction
        assert result[3] == 2     # Wanderer distance
    
    @pytest.mark.parametrize("player_pos, explorers, wanderers", [
        [(3, 3), [(3, 3)], [(3, 3, 1)]],
    ])
    def test_get_state_the_same_position(self, player_pos, explorers, wanderers, observer):
        """Test get_state with player and one other explorer."""
        entities = calculate_entities(player_pos, explorers, wanderers)
        observer.env._set_entities(entities)
        observer.env._set_players(entities, set_ids=True)
        
        # Call get_state
        result = observer.get_state(0)
        
        # Explorer is to the right (index 1)
        assert result[0] == (4,)  # Explorer direction
        assert result[1] == 0     # Explorer distance
        assert result[2] == (4,)  # No wanderer
        assert result[3] == 0     # No wanderer distance

    @pytest.mark.parametrize("player_pos, wanderers", [
        [(3, 3), [(2, 3, 1), (3, 2, 1), (4, 3, 1), (3, 4, 1)]],
    ])
    def test_get_state_wanderers_around_1(self, player_pos, wanderers, observer):
        """Test get_state with player and one wanderer."""
        entities = calculate_entities(player_pos, wanderers=wanderers)
        observer.env._set_entities(entities)
        observer.env._set_players(entities, set_ids=True)
        # Call get_state
        result = observer.get_state(0)
        # Explorer is to the right (index 1)
        assert result[0] is None  # Explorer direction
        assert result[1] is None  # Explorer distance
        assert result[2] == (0, 1, 2, 3,)  # No wanderer
        assert result[3] == 1     # No wanderer distance

    @pytest.mark.parametrize("player_pos, wanderers", [
        [(3, 3), [(1, 3, 1), (3, 1, 1), (5, 3, 1), (3, 5, 1)]],
    ])
    def test_get_state_wanderers_around_2(self, player_pos, wanderers, observer):
        """Test get_state with player and one wanderer."""
        entities = calculate_entities(player_pos, wanderers=wanderers)
        observer.env._set_entities(entities)
        observer.env._set_players(entities, set_ids=True)
        # Call get_state
        result = observer.get_state(0)
        # Explorer is to the right (index 1)
        assert result[0] is None  # Explorer direction
        assert result[1] is None  # Explorer distance
        assert result[2] == (0, 1, 2, 3,)  # No wanderer
        assert result[3] == 2     # No wanderer distance

    @pytest.mark.parametrize("player_pos, explorers, wanderers", [
        [(3, 3), [(5, 3), (1, 3)], [(3, 1, 1), (3, 5, 1), (3, 2, 0)]],
    ])
    def test_get_state_wanderers_around_2(self, player_pos, explorers, wanderers, observer):
        """Test get_state with player and one wanderer."""
        entities = calculate_entities(player_pos, explorers, wanderers)
        observer.env._set_entities(entities)
        observer.env._set_players(entities, set_ids=True)
        # Call get_state
        result = observer.get_state(0)
        # Explorer is to the left (index 3) with distance 1
        # Wanderer is up (index 0) with distance 2
        assert result[0] == (1, 3)         # Either left or right explorer is closest
        assert result[1] == 2              # Distance to closest explorer
        assert result[2] == (0, 2)         # Closest wanderer is up
        assert result[3] == 2              # Distance to closest wanderer
