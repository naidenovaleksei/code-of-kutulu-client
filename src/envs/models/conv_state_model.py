import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvStateModel(nn.Module):
    def __init__(self, size, in_channels=12, num_classes=5, conv_dim=32, fc_dim=64, return_softmax=False):
        super(ConvStateModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, conv_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(conv_dim)
        self.conv2 = nn.Conv2d(conv_dim, conv_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(conv_dim)
        self.pool = nn.MaxPool2d(2, 2)  # уменьшает размер в 2 раза
        self.fc1 = nn.Linear(conv_dim * size * size, fc_dim)  # предполагаем вход размером 10x10
        self.fc2 = nn.Linear(fc_dim, num_classes)
        self.return_softmax = return_softmax

    def forward(self, x):
        # [bs, 16, H, W]
        x = self.bn1(self.conv1(x))     # Conv -> BatchNorm -> ReLU
        x = F.relu(x)
        # [bs, 32, H/2, W/2]
        x = self.pool(self.bn2(self.conv2(x)))
        x = F.relu(x)
        # [bs, 32*H/2*W/2]
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        if self.return_softmax:
            x = F.softmax(x, dim=1)

        return x
    
    def get_policy(self, x):
        return self.forward(x)

    def get_log_probs(self, data):
        """Return log probabilities of actions for policy gradient update"""
        assert self.return_softmax
        return torch.log(self.forward(data) + 1e-8)  # Add small epsilon to avoid log(0)
