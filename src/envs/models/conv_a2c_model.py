import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvA2CModel(nn.Module):
    def __init__(self, size, in_channels=12, num_classes=5, fc_dim=64):
        super(ConvA2CModel, self).__init__()
        # Shared feature extractor
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)  # reduces size by half
        self.shared_fc = nn.Linear(32 * size * size, fc_dim)

        # Actor (policy) head
        self.actor = nn.Linear(fc_dim, num_classes)

        # Critic (value) head
        self.critic = nn.Linear(fc_dim, 1)

    def forward(self, x):
        # Shared feature extraction
        x = self.bn1(self.conv1(x))
        x = F.relu(x)
        x = self.pool(self.bn2(self.conv2(x)))
        x = F.relu(x)
        x = torch.flatten(x, 1)
        shared_features = F.relu(self.shared_fc(x))

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
