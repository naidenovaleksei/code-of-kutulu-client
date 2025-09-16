import torch
import torch.nn as nn
import torch.nn.functional as F

from src.envs.models.conv_state_model import (
    ConvStateEncoderModel,
    ConvStateDeepEncoderModel,
)


class ConvA2CModel(ConvStateEncoderModel):
    def __init__(self, size, in_channels=12, num_classes=5, conv_dim=32, fc_dim=64, return_softmax=False):
        super(ConvA2CModel, self).__init__(
            size,
            in_channels,
            conv_dim,
            fc_dim,
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


class ConvA2CDeepModel(ConvStateDeepEncoderModel):
    def __init__(self, size, in_channels=12, num_classes=5, conv_dim=32, fc_dim=64, return_softmax=False):
        super(ConvA2CDeepModel, self).__init__(
            size,
            in_channels,
            conv_dim,
            fc_dim,
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
