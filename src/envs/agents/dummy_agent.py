import numpy as np
import pickle as pkl
from src.envs.agents import BaseAgent


TEST_STATES = [
    ((4,), 0, None, None),
    ((0,), 1, None, None),
    ((1,), 1, None, None),
    ((2,), 1, None, None),
    ((3,), 1, None, None),
    ((0,), 2, None, None),
    ((1,), 2, None, None),
    ((2,), 2, None, None),
    ((3,), 2, None, None),
    ((0,), 3, None, None),
    ((1,), 3, None, None),
    ((2,), 3, None, None),
    ((3,), 3, None, None),
    ((0,), 4, None, None),
    ((1,), 4, None, None),
    ((2,), 4, None, None),
    ((3,), 4, None, None),
]


class DummyAgent(BaseAgent):
    def __init__(self, state_type, action_space_n,
                 train=False, verbose=False, seed=0):
        super(DummyAgent, self).__init__(
            state_type=state_type,
            action_space_n=action_space_n,
            train=train,
        )
        assert state_type == 'closest'
        self.rng = np.random.RandomState(seed)

    def get_metric_names(self):
        return []

    def inference_step(self, player_id):
        valid_actions = self.get_valid_actions(player_id)
        player_mask = np.array(valid_actions)
        player_mask = player_mask[:self.action_space_n]
        player_mask = ~player_mask

        state = self.get_state(player_id)
        closest_explorer_dir, closest_explorer_dist, closest_wanderer_dir, closest_wanderer_dist = state
        
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

    def _calculate_avoidance_action(self, closest_explorer_dir, closest_explorer_dist, 
                                   closest_wanderer_dir, closest_wanderer_dist, player_mask):
        """
        Calculate action based on avoidance logic:
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
                action = self.rng.choice(valid_indices)
            else:
                action = 4  # WAIT as last resort
        
        return action
    
    def _get_opposite_direction(self, direction_tuple):
        """
        Get the opposite direction for avoidance.
        Direction mapping: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT, 4=WAIT
        Opposites: UP<->DOWN, RIGHT<->LEFT
        """
        if direction_tuple is None or len(direction_tuple) == 0:
            return None
        
        # Direction opposites mapping
        opposite_map = {0: 2, 1: 3, 2: 0, 3: 1, 4: 4}
        return opposite_map.get(direction_tuple[0], 4)  # Default to WAIT

    def generate_state_and_step(self, player_id, need_update=True):
        return self.inference_step(player_id)

    def train_step(self):
        pass

    def save_agent(self, checkpoint_dir):
        pass

    def load_agent(self, checkpoint_dir):
        pass
