import pytest
import torch
import numpy as np

from src.envs.agents.reinforce_agent import REINFORCEAgent
from src.game.template import calculate_output_np


data = {
    'entity_kind': [[1, 1, 3, 2, 0, 0, 0, 0, 0, 0]],
    'entity_features': [[
        [218.0, 3.0, 3.0, -3.0, 6.0, 0.0, 218.0],
        [221.0, 3.0, 1.0, -9.0, 10.0, 0.0, 221.0],
        [27.0, 0.0, 6.0, 6.0, 12.0, 0.0, 27.0],
        [2.0, -1.0, 7.0, 4.0, 11.0, 0.0, 2.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ]],
    'entity_dir': [[
        [0.0, 6, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 12, 0.0],
        [0.0, 12, 0.0, 0.0, 0.0],
        [0.0, 15, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0]
    ]],
}

def test_convert_qdn_ext():
    tensor_data = {
        'entity_kind': torch.IntTensor(data['entity_kind']),
        'entity_features': torch.FloatTensor(data['entity_features']),
        'entity_dir': torch.FloatTensor(data['entity_dir']),
    }
    info = {
        'train': True,
        'state_type': 'closest_ext',
        'gamma': 0.9,
        'action_space_n': 8,
    }
    agent = REINFORCEAgent(**info)
    weights = {}
    for k,v in agent.model.named_parameters():
        weights[k] = v.detach().cpu().numpy()
    np_output = calculate_output_np(data, weights, num_classes=agent.action_space_n, softmax=True)

    model_output = agent.model(tensor_data)[0].detach().cpu().numpy()

    assert np.allclose(np_output, model_output)
