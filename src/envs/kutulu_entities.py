"""
Kutulu game entities and state management classes.

This module provides structured classes for managing players, entities, and game state
in the Kutulu game environment, replacing the dictionary-based approach with proper
object-oriented design.
"""

from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum


class EntityKind(Enum):
    """Enumeration of all entity types in the Kutulu game."""
    EXPLORER = "EXPLORER"
    WANDERER = "WANDERER"
    SLASHER = "SLASHER"
    EFFECT_PLAN = "EFFECT_PLAN"
    EFFECT_LIGHT = "EFFECT_LIGHT"
    EFFECT_SHELTER = "EFFECT_SHELTER"
    EFFECT_YELL = "EFFECT_YELL"


class EffectType(Enum):
    """Types of effects that can be applied to players."""
    PLAN = "PLAN"
    LIGHT = "LIGHT"
    YELL = "YELL"


class KutuluEntity:
    """Base class for all entities in the Kutulu game."""
    
    def __init__(self, kind: str, id: int, x: int, y: int, 
                 param0: int = 0, param1: int = 0, param2: int = 0):
        self.kind = kind
        self.id = id
        self.x = x
        self.y = y
        self.param0 = param0
        self.param1 = param1
        self.param2 = param2
    
    @property
    def position(self) -> Tuple[int, int]:
        """Get the entity's position as a tuple."""
        return (self.x, self.y)
    
    def distance_to(self, other: Union['KutuluEntity', Tuple[int, int]]) -> int:
        """Calculate Manhattan distance to another entity or position."""
        if isinstance(other, tuple):
            other_x, other_y = other
        else:
            other_x, other_y = other.x, other.y
        return abs(self.x - other_x) + abs(self.y - other_y)
    
    def to_dict(self) -> Dict:
        """Convert entity to dictionary format for backward compatibility."""
        return {
            'kind': self.kind,
            'id': self.id,
            'x': self.x,
            'y': self.y,
            'param0': self.param0,
            'param1': self.param1,
            'param2': self.param2,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'KutuluEntity':
        """Create entity from dictionary format."""
        return cls(
            kind=data['kind'],
            id=data['id'],
            x=data['x'],
            y=data['y'],
            param0=data.get('param0', 0),
            param1=data.get('param1', 0),
            param2=data.get('param2', 0)
        )
    
    @classmethod
    def from_string(cls, entity_string: str) -> 'KutuluEntity':
        """Parse entity from server string format."""
        parts = entity_string.split()
        if len(parts) != 7:
            raise ValueError(f"Invalid entity string format: {entity_string}")
        
        kind, id_str, x_str, y_str, param0_str, param1_str, param2_str = parts
        return cls(
            kind=kind,
            id=int(id_str),
            x=int(x_str),
            y=int(y_str),
            param0=int(param0_str),
            param1=int(param1_str),
            param2=int(param2_str)
        )


class KutuluPlayer(KutuluEntity):
    """Represents a player (explorer) in the Kutulu game."""
    
    def __init__(self, id: int, x: int, y: int, sanity: int = 0, 
                 active: bool = True, effect_left: int = 0):
        super().__init__(EntityKind.EXPLORER.value, id, x, y, sanity, 0, 0)
        self.active = active
        self.effect_left = effect_left
    
    @property
    def sanity(self) -> int:
        """Get player's sanity level (param0 for explorers)."""
        return self.param0
    
    @sanity.setter
    def sanity(self, value: int):
        """Set player's sanity level."""
        self.param0 = value
    
    def is_alive(self) -> bool:
        """Check if player is alive (active and has sanity)."""
        return self.active and self.sanity > 0
    
    def apply_effect(self, effect_type: EffectType, duration: int):
        """Apply an effect to the player."""
        if effect_type == EffectType.PLAN:
            self.effect_left = 4
        elif effect_type == EffectType.LIGHT:
            self.effect_left = 2
        elif effect_type == EffectType.YELL:
            # YELL is instantaneous, no duration
            pass
    
    def update_effect(self):
        """Update effect duration (called each turn)."""
        if self.effect_left > 0:
            self.effect_left -= 1
    
    def can_use_effect(self, effect_type: EffectType) -> bool:
        """Check if player can use a specific effect."""
        if self.effect_left > 0:
            return False
        
        if effect_type == EffectType.PLAN:
            return self.param1 > 0  # Has PLAN charges
        elif effect_type == EffectType.LIGHT:
            return self.param2 > 0  # Has LIGHT charges
        elif effect_type == EffectType.YELL:
            return True  # YELL is always available
        
        return False
    
    def to_dict(self) -> Dict:
        """Convert player to dictionary format for backward compatibility."""
        base_dict = super().to_dict()
        base_dict.update({
            'active': self.active,
            'effect_left': self.effect_left,
            'sanity': self.sanity,
        })
        return base_dict
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'KutuluPlayer':
        """Create player from dictionary format."""
        player = cls(
            id=data['id'],
            x=data['x'],
            y=data['y'],
            sanity=data.get('sanity', data.get('param0', 0)),
            active=data.get('active', True),
            effect_left=data.get('effect_left', 0)
        )
        player.param1 = data.get('param1', 0)
        player.param2 = data.get('param2', 0)
        return player
    
    @classmethod
    def from_entity_string(cls, entity_string: str) -> 'KutuluPlayer':
        """Parse player from server string format."""
        entity = KutuluEntity.from_string(entity_string)
        if entity.kind != EntityKind.EXPLORER.value:
            raise ValueError(f"Expected EXPLORER, got {entity.kind}")
        
        player = cls(
            id=entity.id,
            x=entity.x,
            y=entity.y,
            sanity=entity.param0,
            active=True,
            effect_left=0
        )
        # Set param1 and param2 for effect charges
        player.param1 = entity.param1
        player.param2 = entity.param2
        return player


class KutuluWanderer(KutuluEntity):
    """Represents a wanderer enemy in the Kutulu game."""
    
    def __init__(self, id: int, x: int, y: int, param0: int, param1: int, param2: int):
        super().__init__(EntityKind.WANDERER.value, id, x, y, param0, param1, param2)
    
    @property
    def wandering(self) -> bool:
        """Check if wanderer is in wandering state."""
        return self.param1 == 1
    
    @property
    def target_player_id(self) -> int:
        """Get the ID of the player this wanderer is targeting."""
        return self.param2
    
    @property
    def time_left(self) -> int:
        """Get time left for current state (spawn or recall)."""
        return self.param0
    
    @property
    def spawn_time_left(self) -> Optional[int]:
        """Get spawn time left if not wandering."""
        return self.param0 if not self.wandering else None
    
    @property
    def recall_time_left(self) -> Optional[int]:
        """Get recall time left if wandering."""
        return self.param0 if self.wandering else None


class KutuluSlasher(KutuluEntity):
    """Represents a slasher enemy in the Kutulu game."""
    
    def __init__(self, id: int, x: int, y: int, param0: int, param1: int, param2: int):
        super().__init__(EntityKind.SLASHER.value, id, x, y, param0, param1, param2)
    
    @property
    def change_time_left(self) -> int:
        """Get time before changing state."""
        return self.param0


class KutuluObservation:
    """Represents the complete state of a Kutulu game."""
    
    def __init__(self, width: int, height: int, map_data: List[str], turn: int = 0):
        self.width = width
        self.height = height
        self.map = map_data
        self.turn = turn
        self.players: Dict[int, KutuluPlayer] = {}
        self.entities: List[KutuluEntity] = []
        self.active_player_ids: List[int] = []
    
    def add_player(self, player: KutuluPlayer):
        """Add a player to the game state."""
        self.players[player.id] = player
        if player.active and player.id not in self.active_player_ids:
            self.active_player_ids.append(player.id)
    
    def remove_player(self, player_id: int):
        """Remove a player from the game state."""
        if player_id in self.players:
            del self.players[player_id]
        if player_id in self.active_player_ids:
            self.active_player_ids.remove(player_id)
    
    def get_player(self, player_id: int) -> Optional[KutuluPlayer]:
        """Get a player by ID."""
        return self.players.get(player_id)
    
    def get_active_players(self) -> List[KutuluPlayer]:
        """Get all active players."""
        return [
            self.players[pid] for pid in self.active_player_ids 
            if pid in self.players and self.players[pid].active
        ]
    
    def get_active_player_ids(self) -> List[int]:
        """Get IDs of all active players."""
        return [
            pid for pid in self.active_player_ids 
            if pid in self.players and self.players[pid].active
        ]
    
    def update_entities(self, entity_strings: List[str]):
        """Update entities from server response strings."""
        self.entities.clear()
        
        for entity_string in entity_strings[1:]:  # Skip first element (None)
            if entity_string is None:
                continue
                
            entity = KutuluEntity.from_string(entity_string)
            
            # Create specialized entity types
            if entity.kind == EntityKind.EXPLORER.value:
                player = KutuluPlayer.from_entity_string(entity_string)
                # Update existing player or add new one
                if player.id in self.players:
                    existing_player = self.players[player.id]
                    existing_player.x = player.x
                    existing_player.y = player.y
                    existing_player.sanity = player.sanity
                    existing_player.active = True
                    # Preserve effect_left from previous state
                    existing_player.effect_left = max(0, existing_player.effect_left - 1)
                else:
                    self.add_player(player)
                
                self.entities.append(self.players[player.id])
            
            elif entity.kind == EntityKind.WANDERER.value:
                wanderer = KutuluWanderer(
                    entity.id, entity.x, entity.y,
                    entity.param0, entity.param1, entity.param2
                )
                self.entities.append(wanderer)
            
            elif entity.kind == EntityKind.SLASHER.value:
                slasher = KutuluSlasher(
                    entity.id, entity.x, entity.y,
                    entity.param0, entity.param1, entity.param2
                )
                self.entities.append(slasher)
            
            else:
                # Other entity types (effects, etc.)
                self.entities.append(entity)
    
    def mark_inactive_players(self):
        """Mark players as inactive if they're not in current entities."""
        current_player_ids = {
            e.id for e in self.entities 
            if isinstance(e, KutuluPlayer) or e.kind == EntityKind.EXPLORER.value
        }
        
        for player_id, player in self.players.items():
            if player_id not in current_player_ids:
                player.active = False
    
    def get_player_observation(self, player_id: int) -> Dict:
        """Get observation data for a specific player."""
        player = self.get_player(player_id)
        if not player:
            return {}
        
        # Put the requesting player first in entities list
        player_entities = [player]
        other_entities = [e for e in self.entities if e.id != player_id]
        
        return {
            'active_player_count': len(self.get_active_players()),
            'entities': [e.to_dict() for e in player_entities + other_entities]
        }
    
    def get_entities_dict_format(self, player_id: Optional[int] = None) -> List[Dict]:
        """Get entities in dictionary format for backward compatibility."""
        if player_id is None:
            return [e.to_dict() for e in self.entities]
        else:
            # Put specific player first
            player_entities = [e for e in self.entities if e.id == player_id]
            other_entities = [e for e in self.entities if e.id != player_id]
            return [e.to_dict() for e in player_entities + other_entities]
    
    def to_dict(self) -> Dict:
        """Convert game state to dictionary format for backward compatibility."""
        return {
            'width': self.width,
            'height': self.height,
            'map': self.map,
            'turn': self.turn,
            'players': {pid: player.to_dict() for pid, player in self.players.items()},
            'entities': [e.to_dict() for e in self.entities],
            'active_player_ids': self.active_player_ids,
        }
