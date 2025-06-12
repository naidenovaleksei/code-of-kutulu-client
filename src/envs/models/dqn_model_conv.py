import torch
import torch.nn as nn
import torch.nn.functional as F

class DQNConv(nn.Module):
    def __init__(self, size, in_channels=12, num_classes=5):
        super(DQNConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # уменьшает размер в 2 раза
        self.fc1 = nn.Linear(32 * size * size, 64)  # предполагаем вход размером 10x10
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # [bs, 16, H, W]
        x = F.relu(self.conv1(x))
        # [bs, 32, H/2, W/2]
        x = self.pool(F.relu(self.conv2(x)))
        # [bs, 32*H/2*W/2]
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
