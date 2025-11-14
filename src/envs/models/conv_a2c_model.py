import torch
import torch.nn as nn
import torch.nn.functional as F

from src.envs.models.conv_state_model import (
    ConvStateEncoderModel,
    ConvStateDeepEncoderModel,
)


class ConvA2CModel(nn.Module):
    def __init__(self, size, in_channels=12, num_classes=5, conv_dim=32, fc_dim=64,
                 use_deep=False,
                 use_gru=False, hidden_size=None, num_gru_layers=None, gru_dropout=None,
                 detach_actor=False,
                 return_softmax=False):
        # super(ConvA2CModel, self).__init__(
        #     size,
        #     in_channels,
        #     conv_dim,
        #     fc_dim,
        # )
        super().__init__()
        self.detach_actor = detach_actor
        self.use_deep = use_deep
        self.use_gru = use_gru
        if self.use_deep:
            self.encoder = ConvStateDeepEncoderModel(
                size,
                in_channels,
                conv_dim,
                fc_dim,
            )
        else:
            self.encoder = ConvStateEncoderModel(
                size,
                in_channels,
                conv_dim,
                fc_dim,
            )
        # Actor (policy) head
        if self.use_gru:
            actor_inner_dim = hidden_size
        else:
            actor_inner_dim = fc_dim
        self.actor = nn.Linear(actor_inner_dim, num_classes)
        # Critic (value) head
        self.critic = nn.Linear(fc_dim, 1)
        # Aux turns_to_death head
        self.terminator = nn.Linear(fc_dim, 1)
        # Aux occupation prediction head
        self.occupation_head = nn.Linear(fc_dim, 1)
        
        if self.use_gru:
            self.hidden_size = hidden_size
            self.num_gru_layers = num_gru_layers
            self.gru_dropout = gru_dropout
            self.gru = nn.GRU(
                input_size=fc_dim,
                hidden_size=hidden_size,
                num_layers=num_gru_layers,
                dropout=gru_dropout if num_gru_layers > 1 else 0.0,
                batch_first=True
            )
            # self.shared_dropout = nn.Dropout(p=0.1)
            self.hidden_dropout = nn.Dropout(p=self.gru_dropout)
            

    def forward(self, x, hidden_state=None):
        # Shared feature extraction
        x = self.encoder(x)
        shared_features = F.relu(x)

        if self.detach_actor:
            actor_features = shared_features.detach()
        else:
            actor_features = shared_features

        if self.use_gru:
            gru_input = actor_features.unsqueeze(1)
            if hidden_state is not None:
                hidden_state = self.hidden_dropout(hidden_state)
            # GRU forward pass
            gru_output, new_hidden_state = self.gru(gru_input, hidden_state)
            # Remove sequence dimension: [batch_size, hidden_size]
            actor_features = gru_output.squeeze(1)

        # Actor: output action probabilities
        policy_logits = self.actor(actor_features)
        policy = F.softmax(policy_logits, dim=1)

        with torch.no_grad():
            sf_det = shared_features.detach()
        # Critic: output state value
        value = self.critic(shared_features)
        
        turns_to_death = self.terminator(shared_features)
        is_occupied = self.occupation_head(shared_features)

        outputs = {
            'policy': policy,
            'value': value,
            'turns_to_death': turns_to_death,
            'is_occupied': is_occupied,
        }
        if self.use_gru:
            outputs['hidden_state'] = new_hidden_state
        return outputs
    
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


# class ConvA2CGRUModel(ConvStateEncoderModel):
#     def __init__(self, size, in_channels=12, num_classes=5, conv_dim=32, fc_dim=64, 
#                  hidden_size=64, num_gru_layers=1, gru_dropout=0.0, return_softmax=False):
#         super(ConvA2CGRUModel, self).__init__(
#             size,
#             in_channels,
#             conv_dim,
#             fc_dim,
#         )
        
#         # GRU layer
#         self.hidden_size = hidden_size
#         self.num_gru_layers = num_gru_layers
#         self.gru = nn.GRU(
#             input_size=fc_dim,
#             hidden_size=hidden_size,
#             num_layers=num_gru_layers,
#             dropout=gru_dropout if num_gru_layers > 1 else 0.0,
#             batch_first=True
#         )
        
#         # Actor (policy) head
#         self.actor = nn.Linear(hidden_size, num_classes)
#         # Critic (value) head
#         self.critic = nn.Linear(fc_dim, 1)
#         # Aux turns_to_death head
#         self.terminator = nn.Linear(fc_dim, 1)
#         # Aux occupation prediction head
#         self.occupation_head = nn.Linear(fc_dim, 1)
        
#         self.shared_dropout = nn.Dropout(p=0.1)
#         self.hidden_dropout = nn.Dropout(p=0.5, inplace=True)

#     def forward(self, x, hidden_state=None):
#         # Shared feature extraction from conv layers
#         conv_features = super().forward(x)  # [batch_size, fc_dim]
#         conv_features = F.relu(conv_features)
        
#         # conv_features = self.shared_dropout(conv_features)
#         # hidden_state = self.shared_dropout(hidden_state)
        
#         # Reshape for GRU: [batch_size, seq_len=1, fc_dim]
#         # gru_input = self.shared_dropout(conv_features).unsqueeze(1)
#         gru_input = conv_features.unsqueeze(1)
        
#         if hidden_state is not None:
#             hidden_state = self.hidden_dropout(hidden_state)
        
#         # GRU forward pass
#         gru_output, new_hidden_state = self.gru(gru_input, hidden_state)
        
#         # Remove sequence dimension: [batch_size, hidden_size]
#         gru_features = gru_output.squeeze(1)
#         # hidden_features = new_hidden_state.clone().squeeze(1)
        
#         # Actor: output action probabilities
#         policy_logits = self.actor(gru_features)
#         policy = F.softmax(policy_logits, dim=1)
        
#         # Critic: output state value
#         value = self.critic(conv_features)
        
#         turns_to_death = self.terminator(conv_features)
#         is_occupied = self.occupation_head(conv_features)

#         return {
#             'policy': policy,
#             'value': value,
#             'turns_to_death': turns_to_death,
#             'is_occupied': is_occupied,
#             'hidden_state': new_hidden_state,
#         }
    
#     def get_policy(self, x, hidden_state=None):
#         """Return action probabilities"""
#         return self.forward(x, hidden_state)['policy']
    
#     def get_value(self, x, hidden_state=None):
#         """Return state value"""
#         return self.forward(x, hidden_state)['value']
    
#     def get_log_probs(self, x, hidden_state=None):
#         """Return log probabilities of actions for policy gradient update"""
#         probs = self.forward(x, hidden_state)['policy']
#         # Ensure probabilities are valid and add larger epsilon for numerical stability
#         probs = torch.clamp(probs, min=1e-8, max=1.0)
#         # Renormalize to ensure they sum to 1
#         probs = probs / probs.sum(dim=1, keepdim=True)
#         return torch.log(probs)


# class ConvA2CDeepModel(ConvStateDeepEncoderModel):
#     def __init__(self, size, in_channels=12, num_classes=5, conv_dim=32, fc_dim=64, return_softmax=False):
#         super(ConvA2CDeepModel, self).__init__(
#             size,
#             in_channels,
#             conv_dim,
#             fc_dim,
#         )
#         # Actor (policy) head
#         self.actor = nn.Linear(fc_dim, num_classes)
#         # Critic (value) head
#         self.critic = nn.Linear(fc_dim, 1)
#         # Aux turns_to_death head
#         self.terminator = nn.Linear(fc_dim, 1)
#         # Aux occupation prediction head
#         self.occupation_head = nn.Linear(fc_dim, 1)

#     def forward(self, x):
#         # Shared feature extraction
#         x = super().forward(x)
#         shared_features = F.relu(x)

#         # Actor: output action probabilities
#         policy_logits = self.actor(shared_features)
#         policy = F.softmax(policy_logits, dim=1)

#         # Critic: output state value
#         value = self.critic(shared_features)
        
#         turns_to_death = self.terminator(shared_features)
#         is_occupied = self.occupation_head(shared_features)

#         return {
#             'policy': policy,
#             'value': value,
#             'turns_to_death': turns_to_death,
#             'is_occupied': is_occupied,
#         }
    
#     def get_policy(self, x):
#         """Return action probabilities"""
#         return self.forward(x)['policy']
    
#     def get_value(self, x):
#         """Return state value"""
#         return self.forward(x)['value']
    
#     def get_log_probs(self, x):
#         """Return log probabilities of actions for policy gradient update"""
#         probs = self.forward(x)['policy']
#         # Ensure probabilities are valid and add larger epsilon for numerical stability
#         probs = torch.clamp(probs, min=1e-8, max=1.0)
#         # Renormalize to ensure they sum to 1
#         probs = probs / probs.sum(dim=1, keepdim=True)
#         return torch.log(probs)
