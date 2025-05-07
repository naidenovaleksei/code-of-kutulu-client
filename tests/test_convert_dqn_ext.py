import pytest
import torch
import numpy as np

from src.envs.agents.dqn_agent_ext import DQNAgentExt
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
        'state_type': 'closest_ext',
        'gamma': 0.9,
        'replay_size': 10000,
        'replay_start_size': 100,
        'sync_target_frames': 1000,
        'batch_size': 64,
        'action_space_n': 5,
    }
    agent = DQNAgentExt(**info)
    agent.model.load_state_dict(torch.load(f"./output/2025-05-07/18:38:17.436318/agent0/model.pt"))

    weights = {}
    for k,v in agent.model.named_parameters():
        weights[k] = v.detach().cpu().numpy()
    np_output = calculate_output_np(data, weights, agent.action_space_n)

    model_output = agent.model(tensor_data)[0].detach().cpu().numpy()

    assert np.allclose(np_output, model_output)
