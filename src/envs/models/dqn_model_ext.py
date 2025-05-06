import torch
import torch.nn as nn
# from src.envs.models.dqn_model import DQN

class DQNExt(nn.Module):
    def __init__(self, vocab_size, embed_dim, features_dim, hidden_dim, inner_dim, num_classes):
        super(DQNExt, self).__init__()
        
        self.kind_embs = nn.Embedding(vocab_size, embed_dim)
        self.features_linear = nn.Linear(features_dim, hidden_dim)
        self.entity_linear = nn.Linear(embed_dim + hidden_dim, inner_dim)
        self.out_linear = nn.Linear(inner_dim, 1)
        self.num_classes = num_classes

    def forward(self, data):
        assert data['entity_dir'].shape[-1] == self.num_classes
        
        x_kind_embs = self.kind_embs(data['entity_kind'])
        
        entity_features = data['entity_features']
        x_features = self.features_linear(entity_features)
        
        mask = data['entity_dir']
        
        x = torch.cat((x_kind_embs, x_features), dim=-1)
        x = self.entity_linear(x)

        output = torch.stack((
            (self.out_linear(x).squeeze(-1) * mask[...,0]).min(-1)[0],
            (self.out_linear(x).squeeze(-1) * mask[...,1]).min(-1)[0],
            (self.out_linear(x).squeeze(-1) * mask[...,2]).min(-1)[0],
            (self.out_linear(x).squeeze(-1) * mask[...,3]).min(-1)[0],
            (self.out_linear(x).squeeze(-1) * mask[...,4]).min(-1)[0],
        ), dim=1)

        return output
