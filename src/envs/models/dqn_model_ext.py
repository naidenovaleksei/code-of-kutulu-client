import torch
import torch.nn as nn
# from src.envs.models.dqn_model import DQN

class DQNExt(nn.Module):
    def __init__(self, num_classes, vocab_size,
                 num_dirs=5, embed_dim=32, features_dim=8,
                 hidden_dim=32, inner_dim=16, out_linear_bias=False):
        super(DQNExt, self).__init__()
        
        self.kind_embs = nn.Embedding(vocab_size, embed_dim)
        self.features_linear = nn.Linear(features_dim, hidden_dim)
        self.dir_linear = nn.Linear(num_dirs, inner_dim)
        self.entity_linear = nn.Linear(embed_dim + hidden_dim + inner_dim, inner_dim)
        self.entity_impact = nn.Linear(embed_dim + hidden_dim + inner_dim, num_classes)
        self.out_linear = nn.Linear(inner_dim, 1, bias=out_linear_bias)
        # self.out_linear = nn.Linear(inner_dim * num_classes, num_classes)
        self.num_classes = num_classes
        self.inner_dim = inner_dim
        self.num_dirs = num_dirs
        
    def forward(self, data):
        assert data['entity_dir'].shape[-1] == self.num_dirs
        
        x_kind_embs = self.kind_embs(data['entity_kind'])
        
        entity_features = data['entity_features']
        x_features = self.features_linear(entity_features)

        entity_dir = data['entity_dir']
        x_dir = self.dir_linear(entity_dir)
        entities_mask = (entity_dir > 0).max(dim=-1, keepdim=True)[0]
        
        # [batch_size, entity_dim, embed_dim + hidden_dim + inner_dim]
        x_entitity = torch.cat((x_kind_embs, x_features, x_dir), dim=-1)
        # [batch_size, entity_dim, inner_dim]
        x = self.entity_linear(x_entitity)

        entity_weights = self.entity_impact(x_entitity) * entities_mask
        # entity_weights = torch.softmax(self.entity_impact(x_entitity), dim=-1) * entities_mask
        # [batch_size, inner_dim, entity_dim]
        x_transposed = x.transpose(1, 2)
        # [batch_size, inner_dim, num_classes]
        x = torch.bmm(x_transposed, entity_weights)
        # [batch_size, num_classes, inner_dim]
        x = x.transpose(2, 1)
        # [batch_size, num_classes]
        # output = self.out_linear(x.reshape(-1, self.num_classes * self.inner_dim))
        output = self.out_linear(x).squeeze(-1)
        
        # # [batch_size, num_classes]
        # ind = (self.entity_impact(x_entitity) * entities_mask).max(1)[1].squeeze(-1)
        # # [batch_size, num_classes, inner_dim]
        # ind_expanded = ind.unsqueeze(-1).expand(-1, -1, 16)
        # x_selected = torch.gather(x, dim=1, index=ind_expanded)
        # output = self.out_linear(x_selected).squeeze(-1)
        
        # # [batch_size, inner_dim]
        # # x = (x * entities_mask).sum(dim=1) / entities_mask.sum(1)
        # x = (x * entities_mask).max(dim=1)[0]
        # # [batch_size, num_classes]
        # output = self.out_linear(x)
        return output

        # mask = data['entity_dir']
        # output = torch.stack((
        #     (self.out_linear(x).squeeze(-1) * mask[...,0]).min(-1)[0],
        #     (self.out_linear(x).squeeze(-1) * mask[...,1]).min(-1)[0],
        #     (self.out_linear(x).squeeze(-1) * mask[...,2]).min(-1)[0],
        #     (self.out_linear(x).squeeze(-1) * mask[...,3]).min(-1)[0],
        #     (self.out_linear(x).squeeze(-1) * mask[...,4]).min(-1)[0],
        # ), dim=1)
        # return output
