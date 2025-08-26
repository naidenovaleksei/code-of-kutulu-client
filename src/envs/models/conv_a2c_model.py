import torch
import torch.nn as nn
import torch.nn.functional as F

from src.envs.models.conv_state_model import ConvStateEncoderModel


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

        return {
            'policy': policy,
            'value': value,
            'turns_to_death': turns_to_death,
        }
    
    def get_policy(self, x):
        """Return action probabilities"""
        return self.forward(x)['policy']
    
    def get_value(self, x):
        """Return state value"""
        return self.forward(x)['value']
    
    def get_log_probs(self, x):
        """Return log probabilities of actions for policy gradient update"""
        return torch.log(self.forward(x)['policy'] + 1e-8)  # Add small epsilon to avoid log(0)
