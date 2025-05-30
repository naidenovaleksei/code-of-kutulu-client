import torch
import torch.nn as nn

from src.envs.models.dqn_model_ext import DQNExt

class DQNExtByKind(nn.Module):
    def __init__(self,
                vocab_size,
                num_classes,
                entity_kinds = ["EXPLORER", "WANDERER", "SLASHER"],
                num_dirs=5,
                features_dim=8,
                embed_dim=32,
                hidden_dim=32,
                inner_dim=16,
                out_linear_bias=False):
        super(DQNExtByKind, self).__init__()
        
        # # Default entity kinds if not provided
        # if entity_kinds is None:
        #     entity_kinds = ["EXPLORER", "WANDERER", "SLASHER", "EFFECT_PLAN", "EFFECT_LIGHT", "EFFECT_SHELTER", "EFFECT_YELL"]
        
        self.entity_kinds = entity_kinds
        
        self.model_by_kind = nn.ModuleDict({
            kind: DQNExt(
                    vocab_size=vocab_size,
                    num_dirs=num_dirs,
                    embed_dim=embed_dim,
                    features_dim=features_dim,
                    hidden_dim=hidden_dim,
                    inner_dim=inner_dim,
                    num_classes=num_classes,
                    out_linear_bias=out_linear_bias,
            )
            for kind in self.entity_kinds
        })

    def forward(self, data_by_kind):
        output_by_kind = [
            self.model_by_kind[kind](data_by_kind[kind])
            for kind in self.entity_kinds
        ]
        output = torch.stack(output_by_kind, dim=-1).sum(-1)
        return output
