import torch
import torch.nn as nn
import torch.nn.functional as F

from src.envs.models.conv_state_model import ConvStateModel, DuelingConvStateModel


class ConvStateByKindModel(nn.Module):
    def __init__(self, size, in_channels=12, num_classes=5, fc_dim=64, return_softmax=False):
        super(ConvStateByKindModel, self).__init__()
        
        # Create separate ConvStateModel instances for each channel
        self.models_by_channel = nn.ModuleDict({
            f"channel_{i}": ConvStateModel(
                size=size,
                in_channels=1,  # Each processes only 1 channel
                num_classes=num_classes,
                fc_dim=fc_dim,
                return_softmax=False  # We'll handle softmax at the end
            )
            for i in range(in_channels)
        })
        
        self.return_softmax = return_softmax
        self.in_channels = in_channels

    def forward(self, x):
        # Input: [bs, in_channels, H, W]
        
        # Split into individual channels: list of [bs, 1, H, W] tensors
        channels = torch.split(x, 1, dim=1)
        
        # Process each channel through its dedicated ConvStateModel
        channel_outputs = [
            self.models_by_channel[f"channel_{i}"](channels[i])
            for i in range(self.in_channels)
        ]
        
        # Sum all outputs: [bs, num_classes]
        output = torch.stack(channel_outputs, dim=0).sum(dim=0)
        
        if self.return_softmax:
            output = F.softmax(output, dim=1)
        
        return output
    
    def get_policy(self, x):
        return self.forward(x)

    def get_log_probs(self, data):
        """Return log probabilities of actions for policy gradient update"""
        assert self.return_softmax
        return torch.log(self.forward(data) + 1e-8)  # Add small epsilon to avoid log(0)


class DuelingConvStateByKindModel(nn.Module):
    def __init__(self, size, in_channels=12, num_classes=5, fc_dim=64, return_softmax=False):
        super(DuelingConvStateByKindModel, self).__init__()
        
        # Create separate DuelingConvStateModel instances for each channel
        self.models_by_channel = nn.ModuleDict({
            f"channel_{i}": DuelingConvStateModel(
                size=size,
                in_channels=1,  # Each processes only 1 channel
                num_classes=num_classes,
                fc_dim=fc_dim,
                return_softmax=False  # We'll handle softmax at the end
            )
            for i in range(in_channels)
        })
        
        self.return_softmax = return_softmax
        self.in_channels = in_channels

    def forward(self, x):
        # Input: [bs, in_channels, H, W]
        
        # Split into individual channels: list of [bs, 1, H, W] tensors
        channels = torch.split(x, 1, dim=1)
        
        # Process each channel through its dedicated DuelingConvStateModel
        channel_outputs = [
            self.models_by_channel[f"channel_{i}"](channels[i])
            for i in range(self.in_channels)
        ]
        
        # Sum all outputs: [bs, num_classes]
        output = torch.stack(channel_outputs, dim=0).sum(dim=0)
        
        if self.return_softmax:
            output = F.softmax(output, dim=1)
        
        return output
    
    def get_policy(self, x):
        return self.forward(x)

    def get_log_probs(self, data):
        """Return log probabilities of actions for policy gradient update"""
        assert self.return_softmax
        return torch.log(self.forward(data) + 1e-8)  # Add small epsilon to avoid log(0)
