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

    def forward(self, x):
        # Shared feature extraction
        x = super().forward(x)
        shared_features = F.relu(x)

        # Actor: output action probabilities
        policy_logits = self.actor(shared_features)
        policy = F.softmax(policy_logits, dim=1)

        # Critic: output state value
        value = self.critic(shared_features)

        return policy, value
    
    def get_policy(self, x):
        """Return action probabilities"""
        policy, _ = self.forward(x)
        return policy
    
    def get_value(self, x):
        """Return state value"""
        _, value = self.forward(x)
        return value
    
    def get_log_probs(self, x):
        """Return log probabilities of actions for policy gradient update"""
        policy, _ = self.forward(x)
        return torch.log(policy + 1e-8)  # Add small epsilon to avoid log(0)
