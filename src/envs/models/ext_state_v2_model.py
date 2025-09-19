import torch
import torch.nn as nn
import torch.nn.functional as F


class DummyPositionEncoder(nn.Module):
    def __init__(self,
                 outer_dim=32,
                 features_dim=6,
                ):
        super(DummyPositionEncoder, self).__init__()
        self.features_linear = nn.Linear(features_dim, outer_dim)

    def forward(self, data):
        return self.features_linear(data)

class EntityEncoder(nn.Module):
    def __init__(self,
                 pos_encoder,
                 dir_fields,
                 state_field=None,
                 n_states=10,
                 features_field='features',
                 features_dim=8,
                 hidden_dim=32,
                 outer_dim=16,
                ):
        super(EntityEncoder, self).__init__()

        cat_hidden_dim = 0
        self.state_field = state_field
        if state_field:
            self.state_embs = nn.Embedding(n_states, hidden_dim)
            cat_hidden_dim += hidden_dim

        self.features_field = features_field
        self.features_linear = nn.Linear(features_dim, hidden_dim)
        cat_hidden_dim += hidden_dim

        self.pos_encoder = pos_encoder
        self.dir_fields = dir_fields
        for _ in self.dir_fields:
            cat_hidden_dim += hidden_dim

        self.entity_linear = nn.Sequential(
            nn.Linear(cat_hidden_dim, outer_dim),
            nn.ReLU()
        )

    def forward(self, data):
        x_features_list = []
        
        if self.state_field:
            kind_state = data[self.state_field].squeeze(-1).long()
            x_kind_state = self.state_embs(kind_state)
            x_features_list.append(x_kind_state)
        
        kind_state = data[self.features_field]
        x_features = self.features_linear(kind_state)
        x_features_list.append(x_features)
    
        for dir_field in self.dir_fields:
            dirs = data[dir_field]
            x_pos = self.pos_encoder(dirs)
            x_features_list.append(x_pos)

        mask = data['mask']
        x_entitity = torch.cat(x_features_list, dim=-1)
        x = self.entity_linear(x_entitity)
        output = x

        return output, mask

class ExtStatev2EncoderModel(nn.Module):
    def __init__(self, outer_dim, hidden_dim=16, inner_dim=8, out_linear_bias=False):
        super(ExtStatev2EncoderModel, self).__init__()

        self.pos_encoder = DummyPositionEncoder(hidden_dim)

        self.minion_encoder = EntityEncoder(
            self.pos_encoder,
            dir_fields=['dirs', 'target_dirs'],
            state_field='kind_states',
            n_states=10,
            features_field='features',
            features_dim=1,
            hidden_dim=hidden_dim,
            outer_dim=inner_dim,
        )
        self.explorer_encoder = EntityEncoder(
            self.pos_encoder,
            dir_fields=['dirs'],
            state_field='kind_states',
            n_states=2,
            features_field='features',
            features_dim=4,
            hidden_dim=hidden_dim,
            outer_dim=inner_dim,
        )
        self.effects_encoder = EntityEncoder(
            self.pos_encoder,
            dir_fields=['dirs'],
            state_field='kind_states',
            n_states=2,
            features_field='features',
            features_dim=2,
            hidden_dim=hidden_dim,
            outer_dim=inner_dim,
        )
        self.entity_linear = nn.Sequential(
            nn.Linear(inner_dim, inner_dim),
            nn.ReLU(),
        )
        self.entity_impact = nn.Sequential(
            nn.Linear(inner_dim, inner_dim),
            nn.Sigmoid(),
        )
        self.out_linear = nn.Linear(inner_dim, outer_dim, bias=out_linear_bias)

    def forward(self, data):
        x_m, mask_m = self.minion_encoder(data['minions'])
        x_ex, mask_ex = self.explorer_encoder(data['explorers'])
        x_ef, mask_ef = self.effects_encoder(data['effects'])
        
        x = torch.concatenate((x_m, x_ex, x_ef), axis=1)
        mask = torch.concatenate((mask_m, mask_ex, mask_ef), axis=1)
        x = self.entity_linear(x)
        
        x = self.entity_impact(x)

        x = (x * mask.unsqueeze(-1)).sum(dim=1)
        output = self.out_linear(x)

        return output


class ExtStatev2A2AModel(ExtStatev2EncoderModel):
    def __init__(self, num_classes, fc_dim=8, hidden_dim=16, inner_dim=8, out_linear_bias=False):
        super(ExtStatev2A2AModel, self).__init__(
            fc_dim,
            hidden_dim,
            inner_dim,
            out_linear_bias,
        )
        # Actor (policy) head
        self.actor = nn.Linear(fc_dim, num_classes)
        # Critic (value) head
        self.critic = nn.Linear(fc_dim, 1)
        # Aux turns_to_death head
        self.terminator = nn.Linear(fc_dim, 1)
        # Aux occupation prediction head
        self.occupation_head = nn.Linear(fc_dim, 1)

    def forward(self, x):
        # Shared feature extraction
        x = super().forward(x)
        shared_features = F.relu(x)

        # Actor: output action probabilities
        policy_logits = self.actor(shared_features)
        policy = F.softmax(policy_logits, dim=1)

        with torch.no_grad():
            sf_det = shared_features.detach()
        # Critic: output state value
        value = self.critic(shared_features)
        
        turns_to_death = self.terminator(shared_features)
        is_occupied = self.occupation_head(shared_features)

        return {
            'policy': policy,
            'value': value,
            'turns_to_death': turns_to_death,
            'is_occupied': is_occupied,
        }
    
    def get_policy(self, x):
        """Return action probabilities"""
        return self.forward(x)['policy']
    
    def get_value(self, x):
        """Return state value"""
        return self.forward(x)['value']
    
    def get_log_probs(self, x):
        """Return log probabilities of actions for policy gradient update"""
        probs = self.forward(x)['policy']
        # Ensure probabilities are valid and add larger epsilon for numerical stability
        probs = torch.clamp(probs, min=1e-8, max=1.0)
        # Renormalize to ensure they sum to 1
        probs = probs / probs.sum(dim=1, keepdim=True)
        return torch.log(probs)