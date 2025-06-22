import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvStateEncoderModel(nn.Module):
    def __init__(self, size, in_channels=12, conv_dim=32, emb_dim=64):
        super(ConvStateEncoderModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, conv_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(conv_dim)
        self.conv2 = nn.Conv2d(conv_dim, conv_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(conv_dim)
        self.pool = nn.MaxPool2d(2, 2)  # уменьшает размер в 2 раза
        self.fc = nn.Linear(conv_dim * size * size, emb_dim)

    def forward(self, x):
        # [bs, 16, H, W]
        x = self.bn1(self.conv1(x))     # Conv -> BatchNorm -> ReLU
        x = F.relu(x)
        # [bs, 32, H/2, W/2]
        x = self.pool(self.bn2(self.conv2(x)))
        x = F.relu(x)
        # [bs, 32*H/2*W/2]
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class ConvStateModel(ConvStateEncoderModel):
    def __init__(self, size, in_channels=12, num_classes=5, conv_dim=32, fc_dim=64, return_softmax=False):
        super(ConvStateModel, self).__init__(
            size,
            in_channels,
            conv_dim,
            fc_dim,
        )
        self.fc2 = nn.Linear(fc_dim, num_classes)
        self.return_softmax = return_softmax

    def forward(self, x):
        x = super().forward(x)
        x = self.fc2(F.relu(x))
        if self.return_softmax:
            x = F.softmax(x, dim=1)
        return x
    
    def get_policy(self, x):
        return self.forward(x)

    def get_log_probs(self, data):
        assert self.return_softmax
        return torch.log(self.forward(data) + 1e-8)


class DuelingConvStateModel(ConvStateEncoderModel):
    def __init__(self, size, in_channels=12, num_classes=5, conv_dim=32, fc_dim=64, return_softmax=False):
        super(DuelingConvStateModel, self).__init__(
            size,
            in_channels,
            conv_dim,
            fc_dim,
        )
        
        # Value stream - outputs single scalar V(s)
        self.value_fc = nn.Linear(fc_dim, 1)
        
        # Advantage stream - outputs advantage for each action A(s,a)
        self.advantage_fc = nn.Linear(fc_dim, num_classes)
        
        self.return_softmax = return_softmax
        self.num_classes = num_classes

    def forward(self, x):
        # Shared feature extraction
        x = super().forward(x)
        shared_features = F.relu(x)
        
        # Value stream - single value per state
        value = self.value_fc(shared_features)  # [batch_size, 1]
        
        # Advantage stream - advantage per action
        advantage = self.advantage_fc(shared_features)  # [batch_size, num_classes]
        
        # Dueling architecture: Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
        advantage_mean = advantage.mean(dim=1, keepdim=True)  # [batch_size, 1]
        q_values = value + advantage - advantage_mean  # [batch_size, num_classes]
        
        if self.return_softmax:
            q_values = F.softmax(q_values, dim=1)
        
        return q_values
    
    def get_policy(self, x):
        return self.forward(x)

    def get_log_probs(self, data):
        assert self.return_softmax
        return torch.log(self.forward(data) + 1e-8)
