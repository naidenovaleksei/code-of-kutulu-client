"""
Tests for Kutulu entities and state management classes.
"""

import pytest
from src.envs.kutulu_entities import (
    KutuluEntity, KutuluPlayer, KutuluWanderer, KutuluSlasher, 
    KutuluObservation, EntityKind, EffectType
)


class TestKutuluEntity:
    """Test the base KutuluEntity class."""
    
    def test_entity_creation(self):
        """Test basic entity creation."""
        entity = KutuluEntity("EXPLORER", 1, 5, 3, 100, 2, 1)
        assert entity.kind == "EXPLORER"
        assert entity.id == 1
        assert entity.x == 5
        assert entity.y == 3
        assert entity.param0 == 100
        assert entity.param1 == 2
        assert entity.param2 == 1
    
    def test_position_property(self):
        """Test position property returns correct tuple."""
        entity = KutuluEntity("EXPLORER", 1, 5, 3)
        assert entity.position == (5, 3)
    
    def test_distance_to_entity(self):
        """Test distance calculation to another entity."""
        entity1 = KutuluEntity("EXPLORER", 1, 0, 0)
        entity2 = KutuluEntity("WANDERER", 2, 3, 4)
        assert entity1.distance_to(entity2) == 7  # Manhattan distance
    
    def test_distance_to_position(self):
        """Test distance calculation to a position tuple."""
        entity = KutuluEntity("EXPLORER", 1, 0, 0)
        assert entity.distance_to((3, 4)) == 7
    
    def test_to_dict(self):
        """Test conversion to dictionary format."""
        entity = KutuluEntity("EXPLORER", 1, 5, 3, 100, 2, 1)
        expected = {
            'kind': 'EXPLORER',
            'id': 1,
            'x': 5,
            'y': 3,
            'param0': 100,
            'param1': 2,
            'param2': 1,
        }
        assert entity.to_dict() == expected
    
    def test_from_dict(self):
        """Test creation from dictionary format."""
        data = {
            'kind': 'EXPLORER',
            'id': 1,
            'x': 5,
            'y': 3,
            'param0': 100,
            'param1': 2,
            'param2': 1,
        }
        entity = KutuluEntity.from_dict(data)
        assert entity.kind == "EXPLORER"
        assert entity.id == 1
        assert entity.position == (5, 3)
        assert entity.param0 == 100
    
    def test_from_string(self):
        """Test creation from server string format."""
        entity_string = "EXPLORER 1 5 3 100 2 1"
        entity = KutuluEntity.from_string(entity_string)
        assert entity.kind == "EXPLORER"
        assert entity.id == 1
        assert entity.position == (5, 3)
        assert entity.param0 == 100
        assert entity.param1 == 2
        assert entity.param2 == 1
    
    def test_from_string_invalid_format(self):
        """Test that invalid string format raises ValueError."""
        with pytest.raises(ValueError):
            KutuluEntity.from_string("EXPLORER 1 5")  # Too few parts


class TestKutuluPlayer:
    """Test the KutuluPlayer class."""
    
    def test_player_creation(self):
        """Test basic player creation."""
        player = KutuluPlayer(1, 5, 3, sanity=100, active=True, effect_left=2)
        assert player.id == 1
        assert player.position == (5, 3)
        assert player.sanity == 100
        assert player.active is True
        assert player.effect_left == 2
        assert player.kind == "EXPLORER"
    
    def test_sanity_property(self):
        """Test sanity property getter and setter."""
        player = KutuluPlayer(1, 5, 3, sanity=100)
        assert player.sanity == 100
        assert player.param0 == 100
        
        player.sanity = 80
        assert player.sanity == 80
        assert player.param0 == 80
    
    def test_is_alive(self):
        """Test is_alive method."""
        # Alive player
        player = KutuluPlayer(1, 5, 3, sanity=100, active=True)
        assert player.is_alive() is True
        
        # Dead player (no sanity)
        player.sanity = 0
        assert player.is_alive() is False
        
        # Inactive player
        player.sanity = 100
        player.active = False
        assert player.is_alive() is False
    
    def test_apply_effect(self):
        """Test applying effects to player."""
        player = KutuluPlayer(1, 5, 3)
        
        player.apply_effect(EffectType.PLAN, 4)
        assert player.effect_left == 4
        
        player.apply_effect(EffectType.LIGHT, 2)
        assert player.effect_left == 2
    
    def test_update_effect(self):
        """Test updating effect duration."""
        player = KutuluPlayer(1, 5, 3, effect_left=3)
        
        player.update_effect()
        assert player.effect_left == 2
        
        player.update_effect()
        assert player.effect_left == 1
        
        player.update_effect()
        assert player.effect_left == 0
        
        # Should not go below 0
        player.update_effect()
        assert player.effect_left == 0
    
    def test_can_use_effect(self):
        """Test checking if player can use effects."""
        player = KutuluPlayer(1, 5, 3)
        player.param1 = 2  # PLAN charges
        player.param2 = 1  # LIGHT charges
        
        # Can use effects when no effect is active
        assert player.can_use_effect(EffectType.PLAN) is True
        assert player.can_use_effect(EffectType.LIGHT) is True
        assert player.can_use_effect(EffectType.YELL) is True
        
        # Cannot use effects when one is active
        player.effect_left = 2
        assert player.can_use_effect(EffectType.PLAN) is False
        assert player.can_use_effect(EffectType.LIGHT) is False
        assert player.can_use_effect(EffectType.YELL) is False
        
        # Cannot use effects without charges
        player.effect_left = 0
        player.param1 = 0  # No PLAN charges
        player.param2 = 0  # No LIGHT charges
        assert player.can_use_effect(EffectType.PLAN) is False
        assert player.can_use_effect(EffectType.LIGHT) is False
        assert player.can_use_effect(EffectType.YELL) is True  # YELL always available
    
    def test_from_entity_string(self):
        """Test creating player from entity string."""
        entity_string = "EXPLORER 1 5 3 100 2 1"
        player = KutuluPlayer.from_entity_string(entity_string)
        assert player.id == 1
        assert player.position == (5, 3)
        assert player.sanity == 100
        assert player.active is True
        assert player.effect_left == 0
    
    def test_from_entity_string_invalid_kind(self):
        """Test that non-EXPLORER string raises ValueError."""
        with pytest.raises(ValueError):
            KutuluPlayer.from_entity_string("WANDERER 1 5 3 100 2 1")
    
    def test_to_dict_includes_player_fields(self):
        """Test that to_dict includes player-specific fields."""
        player = KutuluPlayer(1, 5, 3, sanity=100, active=True, effect_left=2)
        result = player.to_dict()
        
        assert result['active'] is True
        assert result['effect_left'] == 2
        assert result['sanity'] == 100
        assert result['kind'] == 'EXPLORER'
    
    def test_from_dict(self):
        """Test creating player from dictionary."""
        data = {
            'id': 1,
            'x': 5,
            'y': 3,
            'sanity': 100,
            'active': True,
            'effect_left': 2,
            'param1': 3,
            'param2': 1
        }
        player = KutuluPlayer.from_dict(data)
        assert player.id == 1
        assert player.position == (5, 3)
        assert player.sanity == 100
        assert player.active is True
        assert player.effect_left == 2
        assert player.param1 == 3
        assert player.param2 == 1


class TestKutuluWanderer:
    """Test the KutuluWanderer class."""
    
    def test_wanderer_creation(self):
        """Test basic wanderer creation."""
        wanderer = KutuluWanderer(1, 5, 3, 10, 1, 2)
        assert wanderer.id == 1
        assert wanderer.position == (5, 3)
        assert wanderer.kind == "WANDERER"
    
    def test_wandering_property(self):
        """Test wandering property."""
        # Wandering wanderer
        wanderer = KutuluWanderer(1, 5, 3, 10, 1, 2)
        assert wanderer.wandering is True
        
        # Spawning wanderer
        wanderer = KutuluWanderer(1, 5, 3, 10, 0, 2)
        assert wanderer.wandering is False
    
    def test_target_player_id(self):
        """Test target player ID property."""
        wanderer = KutuluWanderer(1, 5, 3, 10, 1, 2)
        assert wanderer.target_player_id == 2
    
    def test_time_properties(self):
        """Test time-related properties."""
        # Wandering wanderer (recall time)
        wanderer = KutuluWanderer(1, 5, 3, 5, 1, 2)
        assert wanderer.time_left == 5
        assert wanderer.recall_time_left == 5
        assert wanderer.spawn_time_left is None
        
        # Spawning wanderer (spawn time)
        wanderer = KutuluWanderer(1, 5, 3, 3, 0, 2)
        assert wanderer.time_left == 3
        assert wanderer.spawn_time_left == 3
        assert wanderer.recall_time_left is None


class TestKutuluSlasher:
    """Test the KutuluSlasher class."""
    
    def test_slasher_creation(self):
        """Test basic slasher creation."""
        slasher = KutuluSlasher(1, 5, 3, 8, 1, 0)
        assert slasher.id == 1
        assert slasher.position == (5, 3)
        assert slasher.kind == "SLASHER"
    
    def test_change_time_left(self):
        """Test change time left property."""
        slasher = KutuluSlasher(1, 5, 3, 8, 1, 0)
        assert slasher.change_time_left == 8


class TestKutuluObservation:
    """Test the KutuluObservation class."""
    
    def test_game_state_creation(self):
        """Test basic game state creation."""
        map_data = ["###", "#.#", "###"]
        state = KutuluObservation(3, 3, map_data, turn=5)
        
        assert state.width == 3
        assert state.height == 3
        assert state.map == map_data
        assert state.turn == 5
        assert len(state.players) == 0
        assert len(state.entities) == 0
        assert len(state.active_player_ids) == 0
    
    def test_add_remove_player(self):
        """Test adding and removing players."""
        state = KutuluObservation(3, 3, ["###", "#.#", "###"])
        player = KutuluPlayer(1, 1, 1, sanity=100, active=True)
        
        # Add player
        state.add_player(player)
        assert len(state.players) == 1
        assert 1 in state.players
        assert 1 in state.active_player_ids
        
        # Remove player
        state.remove_player(1)
        assert len(state.players) == 0
        assert 1 not in state.active_player_ids
    
    def test_get_player(self):
        """Test getting player by ID."""
        state = KutuluObservation(3, 3, ["###", "#.#", "###"])
        player = KutuluPlayer(1, 1, 1, sanity=100)
        state.add_player(player)
        
        retrieved = state.get_player(1)
        assert retrieved is player
        
        assert state.get_player(999) is None
    
    def test_get_active_players(self):
        """Test getting active players."""
        state = KutuluObservation(3, 3, ["###", "#.#", "###"])
        
        player1 = KutuluPlayer(1, 1, 1, active=True)
        player2 = KutuluPlayer(2, 1, 2, active=False)
        player3 = KutuluPlayer(3, 2, 1, active=True)
        
        state.add_player(player1)
        state.add_player(player2)
        state.add_player(player3)
        
        active = state.get_active_players()
        assert len(active) == 2
        assert player1 in active
        assert player3 in active
        assert player2 not in active
    
    def test_update_entities(self):
        """Test updating entities from server strings."""
        state = KutuluObservation(10, 10, ["." * 10] * 10)
        
        entity_strings = [
            None,  # First element is always None
            "EXPLORER 1 5 3 100 2 1",
            "WANDERER 2 7 8 10 1 1",
            "SLASHER 3 2 4 5 2 0"
        ]
        
        state.update_entities(entity_strings)
        
        assert len(state.entities) == 3
        assert len(state.players) == 1
        
        # Check player was created and added
        player = state.get_player(1)
        assert player is not None
        assert player.position == (5, 3)
        assert player.sanity == 100
        
        # Check entities include all types
        kinds = [e.kind for e in state.entities]
        assert "EXPLORER" in kinds
        assert "WANDERER" in kinds
        assert "SLASHER" in kinds
    
    def test_mark_inactive_players(self):
        """Test marking players as inactive."""
        state = KutuluObservation(3, 3, ["###", "#.#", "###"])
        
        player1 = KutuluPlayer(1, 1, 1, active=True)
        player2 = KutuluPlayer(2, 1, 2, active=True)
        
        state.add_player(player1)
        state.add_player(player2)
        
        # Add only player1 to entities (simulating player2 being eliminated)
        state.entities = [player1]
        
        state.mark_inactive_players()
        
        assert player1.active is True
        assert player2.active is False
    
    def test_get_player_observation(self):
        """Test getting player-specific observation."""
        state = KutuluObservation(3, 3, ["###", "#.#", "###"])
        
        player1 = KutuluPlayer(1, 1, 1, active=True)
        player2 = KutuluPlayer(2, 1, 2, active=True)
        wanderer = KutuluWanderer(3, 2, 1, 10, 1, 1)
        
        state.add_player(player1)
        state.add_player(player2)
        state.entities = [player1, player2, wanderer]
        
        obs = state.get_player_observation(1)
        
        assert obs['active_player_count'] == 2
        assert len(obs['entities']) == 3
        # Player 1 should be first in the entities list
        assert obs['entities'][0]['id'] == 1
    
    def test_get_entities_dict_format(self):
        """Test getting entities in dictionary format."""
        state = KutuluObservation(3, 3, ["###", "#.#", "###"])
        
        player = KutuluPlayer(1, 1, 1)
        wanderer = KutuluWanderer(2, 2, 1, 10, 1, 1)
        
        state.entities = [player, wanderer]
        
        # Test without specific player
        entities_dict = state.get_entities_dict_format()
        assert len(entities_dict) == 2
        assert all(isinstance(e, dict) for e in entities_dict)
        
        # Test with specific player (should be first)
        entities_dict = state.get_entities_dict_format(player_id=1)
        assert entities_dict[0]['id'] == 1
    
    def test_to_dict(self):
        """Test converting game state to dictionary."""
        map_data = ["###", "#.#", "###"]
        state = KutuluObservation(3, 3, map_data, turn=5)
        
        player = KutuluPlayer(1, 1, 1, sanity=100)
        state.add_player(player)
        
        result = state.to_dict()
        
        assert result['width'] == 3
        assert result['height'] == 3
        assert result['map'] == map_data
        assert result['turn'] == 5
        assert len(result['players']) == 1
        assert 1 in result['players']
        assert result['active_player_ids'] == [1]
