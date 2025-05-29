import unittest
import numpy as np
import torch

from src.game.template import (
    parse_state_by_kind,
    ENTITY_TOKENS,
)
from src.envs.agents.dqn_agent_by_kind import ExperienceBufferByKind, DQNAgentByKind
from src.envs.models.dqn_model_by_kind import DQNExtByKind


class TestDQNAgentByKind(unittest.TestCase):
    def test_parse_state_by_kind(self):
        # Create a sample state
        state = [
            {'kind': 'EXPLORER', 'id': 0, 'x': 1, 'y': 1, 'param0': 100, 'param1': 0, 'param2': 0, 
             'rel_x': 0, 'rel_y': 0, 'dist': 0, 'raw_dist': 0, 'on_los': 1, 'dir': (0, 1, 2)},
            {'kind': 'WANDERER', 'id': 1, 'x': 5, 'y': 5, 'param0': 0, 'param1': 1, 'param2': 0, 
             'rel_x': 4, 'rel_y': 4, 'dist': 8, 'raw_dist': 8, 'on_los': 1, 'dir': (3, 4)},
            {'kind': 'EXPLORER', 'id': 2, 'x': 3, 'y': 3, 'param0': 90, 'param1': 0, 'param2': 0, 
             'rel_x': 2, 'rel_y': 2, 'dist': 4, 'raw_dist': 4, 'on_los': 1, 'dir': (0, 2)},
        ]
        
        # Parse the state
        result = parse_state_by_kind(state)
        
        # Check the result
        self.assertIn('EXPLORER', result)
        self.assertIn('WANDERER', result)
        self.assertEqual(len(result['EXPLORER'][0]), 3)  # 2 explorers + 1 padding
        self.assertEqual(len(result['WANDERER'][0]), 10)  # 1 wanderer + 9 padding (MAX_ENTITY_COUNT_BY_KIND["WANDERER"] = 10)
    
    def test_experience_buffer_by_kind(self):
        # Create a sample state
        state = [
            {'kind': 'EXPLORER', 'id': 0, 'x': 1, 'y': 1, 'param0': 100, 'param1': 0, 'param2': 0, 
             'rel_x': 0, 'rel_y': 0, 'dist': 0, 'raw_dist': 0, 'on_los': 1, 'dir': (0, 1, 2)},
            {'kind': 'WANDERER', 'id': 1, 'x': 5, 'y': 5, 'param0': 0, 'param1': 1, 'param2': 0, 
             'rel_x': 4, 'rel_y': 4, 'dist': 8, 'raw_dist': 8, 'on_los': 1, 'dir': (3, 4)},
        ]
        
        # Create a buffer
        buffer = ExperienceBufferByKind(100)
        
        # Encode the state
        result = buffer.encode_states([state])
        
        # Check the result
        self.assertIn('EXPLORER', result)
        self.assertIn('WANDERER', result)
        self.assertIn('entity_features', result['EXPLORER'])
        self.assertIn('entity_dir', result['EXPLORER'])
    
    def test_dqn_ext_by_kind(self):
        # Create a model
        model = DQNExtByKind(
            vocab_size=len(ENTITY_TOKENS) + 1,
            num_dirs=5,
            features_dim=8,
            embed_dim=32,
            hidden_dim=32,
            inner_dim=16,
            num_classes=5,
            entity_kinds=["EXPLORER", "WANDERER"]
        )
        
        # Create sample data
        data = {
            'EXPLORER': {
                'entity_kind': torch.ones(1, 2).long(),
                'entity_features': torch.randn(1, 2, 8),
                'entity_dir': torch.randn(1, 2, 5),
            },
            'WANDERER': {
                'entity_kind': torch.ones(1, 1).long(),
                'entity_features': torch.randn(1, 1, 8),
                'entity_dir': torch.randn(1, 1, 5),
            },
        }
        
        # Forward pass
        output = model(data)
        
        # Check the output shape
        self.assertEqual(output.shape, (1, 5))


if __name__ == '__main__':
    unittest.main()
