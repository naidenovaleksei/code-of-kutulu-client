import numpy as np
from src.envs.agents import BaseAgent


class RawEuristicAgent(BaseAgent):
    """
    Heuristic-based agent that uses raw entity observations.
    
    This agent implements the same policy as DummyAgent but works with KutuluRawObserver
    which provides raw entity data instead of pre-processed closest entity information.
    
    Policy:
    - Avoid wanderers when they are within distance <= 3
    - Otherwise, move towards the closest explorer
    - Falls back to random valid action if no clear direction
    """
    
    def __init__(self, state_type, action_space_n,
                 train=False, verbose=False):
        """
        Initialize RawEuristicAgent.
        
        Args:
            state_type: Must be 'raw' to use KutuluRawObserver
            action_space_n: Number of available actions (typically 5: UP, RIGHT, DOWN, LEFT, WAIT)
            train: Whether agent is in training mode (not used for heuristic agent)
            verbose: Whether to print debug information
        """
        super(RawEuristicAgent, self).__init__(
            state_type=state_type,
            action_space_n=action_space_n,
            train=train,
        )
        assert state_type == 'raw'

    def get_metric_names(self):
        """
        Get names of metrics tracked by this agent.
        
        Returns:
            List of metric names (empty for heuristic agent)
        """
        return []

    def inference_step(self, player_id):
        """
        Perform one inference step to select an action.
        
        Args:
            player_id: ID of the player to generate action for
            
        Returns:
            Dict containing:
                - 'state': Raw entity list from observer
                - 'action': Selected action index (0-4)
                - 'valid_actions': Boolean list of valid actions
        """
        # get_valid_actions returns: List[bool] - boolean mask where True means action is valid
        # Example: [False, True, True, False, True] means actions 1, 2, 4 are valid
        valid_actions = self.get_valid_actions(player_id)
        player_mask = np.array(valid_actions)
        player_mask = player_mask[:self.action_space_n]
        player_mask = ~player_mask

        # get_state returns: List[Dict] - list of entity dictionaries with keys:
        # 'id', 'kind', 'x', 'y', 'param0', 'param1', 'param2'
        # First entity is always the player, followed by other entities (explorers, wanderers, slashers, effects)
        state = self.get_state(player_id)
        
        # Parse raw state to extract closest explorer and wanderer info
        closest_explorer_dir, closest_explorer_dist, closest_wanderer_dir, closest_wanderer_dist = self._parse_raw_state(state, player_id)
        
        # Decision logic: move away from wanderer if nearby, otherwise move towards explorer
        action = self._calculate_avoidance_action(
            closest_explorer_dir, closest_explorer_dist,
            closest_wanderer_dir, closest_wanderer_dist,
            player_mask
        )

        return {
            'state': state,
            'action': action,
            'valid_actions': valid_actions,
        }

    def _parse_raw_state(self, entities, player_id):
        """
        Parse raw entity list to find closest explorer and wanderer.
        
        Args:
            entities: List[Dict] - raw entity data from observer
            player_id: ID of the current player
            
        Returns:
            Tuple of (closest_explorer_dir, closest_explorer_dist, closest_wanderer_dir, closest_wanderer_dist)
            - closest_explorer_dir: Tuple[int] or None - direction to closest explorer (0-3)
            - closest_explorer_dist: int or None - Manhattan distance to closest explorer
            - closest_wanderer_dir: Tuple[int] or None - direction to closest wanderer (0-3)
            - closest_wanderer_dist: int or None - Manhattan distance to closest wanderer
        """
        # First entity should be the player
        player = entities[0]
        assert player['id'] == player_id
        player_pos = (player['x'], player['y'])
        
        closest_explorer = None
        closest_explorer_dist = float('inf')
        closest_wanderer = None
        closest_wanderer_dist = float('inf')
        
        # Find closest explorer and wanderer
        for entity in entities[1:]:
            entity_pos = (entity['x'], entity['y'])
            distance = self._manhattan_distance(player_pos, entity_pos)
            
            if entity['kind'] == 'EXPLORER':
                if distance < closest_explorer_dist:
                    closest_explorer_dist = distance
                    closest_explorer = entity_pos
            elif entity['kind'] == 'WANDERER':
                if distance < closest_wanderer_dist:
                    closest_wanderer_dist = distance
                    closest_wanderer = entity_pos
        
        # Calculate directions
        closest_explorer_dir = self._get_direction(player_pos, closest_explorer) if closest_explorer else None
        closest_wanderer_dir = self._get_direction(player_pos, closest_wanderer) if closest_wanderer else None
        
        # Convert distances to None if no entity found
        closest_explorer_dist = closest_explorer_dist if closest_explorer else None
        closest_wanderer_dist = closest_wanderer_dist if closest_wanderer else None
        
        return closest_explorer_dir, closest_explorer_dist, closest_wanderer_dir, closest_wanderer_dist
    
    def _manhattan_distance(self, pos1, pos2):
        """
        Calculate Manhattan distance between two positions.
        
        Args:
            pos1: Tuple[int, int] - (x, y) coordinates of first position
            pos2: Tuple[int, int] - (x, y) coordinates of second position
            
        Returns:
            int - Manhattan distance (|x1-x2| + |y1-y2|)
        """
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def _get_direction(self, from_pos, to_pos):
        """
        Get direction from from_pos to to_pos.
        
        Args:
            from_pos: Tuple[int, int] - starting position (x, y)
            to_pos: Tuple[int, int] or None - target position (x, y)
            
        Returns:
            Tuple[int] or None - tuple with primary direction (0-3), or None if to_pos is None
            Direction mapping: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
        """
        if to_pos is None:
            return None
        
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        
        # Prioritize vertical movement if dy is larger, otherwise horizontal
        if abs(dy) > abs(dx):
            if dy < 0:
                return (0,)  # UP
            else:
                return (2,)  # DOWN
        else:
            if dx > 0:
                return (1,)  # RIGHT
            else:
                return (3,)  # LEFT

    def _calculate_avoidance_action(self, closest_explorer_dir, closest_explorer_dist, 
                                   closest_wanderer_dir, closest_wanderer_dist, player_mask):
        """
        Calculate action based on avoidance logic.
        
        Args:
            closest_explorer_dir: Tuple[int] or None - direction to closest explorer
            closest_explorer_dist: int or None - distance to closest explorer
            closest_wanderer_dir: Tuple[int] or None - direction to closest wanderer
            closest_wanderer_dist: int or None - distance to closest wanderer
            player_mask: np.ndarray - boolean mask where True means action is INVALID
            
        Returns:
            int - selected action index (0-4)
            
        Logic:
        1. If wanderer is close (distance <= 3), move away from it
        2. Otherwise, move towards closest explorer
        3. If no clear direction, choose random valid action
        """
        action = None
        WANDERER_AVOIDANCE_DISTANCE = 3
        
        # Priority 1: Avoid nearby wanderers
        if closest_wanderer_dist is not None and closest_wanderer_dist <= WANDERER_AVOIDANCE_DISTANCE:
            # Move away from wanderer - choose opposite direction
            action = self._get_opposite_direction(closest_wanderer_dir)
        # Priority 2: Move towards closest explorer
        elif closest_explorer_dir is not None:
            # Move towards closest explorer
            action = closest_explorer_dir[0]  # Take first direction from tuple
        
        # Apply action masking and fallback to random if needed
        if action is None or action >= len(player_mask) or player_mask[action]:
            # Choose random valid action
            valid_indices = np.where(~player_mask)[0]
            if len(valid_indices) > 0:
                action = np.random.choice(valid_indices)
            else:
                action = 4  # WAIT as last resort
        
        return action
    
    def _get_opposite_direction(self, direction_tuple):
        """
        Get the opposite direction for avoidance.
        
        Args:
            direction_tuple: Tuple[int] or None - direction to reverse (0-4)
            
        Returns:
            int or None - opposite direction, or None if input is None
            
        Direction mapping: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT, 4=WAIT
        Opposites: UP<->DOWN (0<->2), RIGHT<->LEFT (1<->3)
        """
        if direction_tuple is None or len(direction_tuple) == 0:
            return None
        
        # Direction opposites mapping
        opposite_map = {0: 2, 1: 3, 2: 0, 3: 1, 4: 4}
        return opposite_map.get(direction_tuple[0], 4)  # Default to WAIT

    def generate_state_and_step(self, player_id, need_update=True):
        """
        Generate state and perform inference step (wrapper for inference_step).
        
        Args:
            player_id: ID of the player
            need_update: Whether to update internal state (unused for heuristic agent)
            
        Returns:
            Dict from inference_step containing state, action, and valid_actions
        """
        return self.inference_step(player_id)

    def train_step(self):
        """
        Perform training step (no-op for heuristic agent).
        Heuristic agents don't learn, so this method does nothing.
        """
        pass

    def save_agent(self, checkpoint_dir):
        """
        Save agent state to checkpoint directory (no-op for heuristic agent).
        Heuristic agents have no learnable parameters to save.
        
        Args:
            checkpoint_dir: Directory path to save checkpoint
        """
        pass

    def load_agent(self, checkpoint_dir):
        """
        Load agent state from checkpoint directory (no-op for heuristic agent).
        Heuristic agents have no learnable parameters to load.
        
        Args:
            checkpoint_dir: Directory path to load checkpoint from
        """
        pass
