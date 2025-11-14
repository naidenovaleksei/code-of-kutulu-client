import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)  # Use BatchNorm instead of LayerNorm
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)  # Use BatchNorm instead of LayerNorm
        self.dropout = nn.Dropout2d(0.1)  # Add dropout for regularization

    def forward(self, x):
        # Check for NaN/inf in input
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("Warning: NaN/inf detected in ResBlock input")
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        
        identity = x
        
        # First conv block
        h = self.conv1(x)
        h = self.bn1(h)
        h = F.relu(h)
        h = self.dropout(h)
        
        # Second conv block
        h = self.conv2(h)
        h = self.bn2(h)
        
        # Residual connection with gradient clipping
        h = h + identity
        
        # Check for NaN/inf before final activation
        if torch.isnan(h).any() or torch.isinf(h).any():
            print("Warning: NaN/inf detected in ResBlock before final activation")
            h = torch.nan_to_num(h, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return F.relu(h)


class MaskedConv2d(nn.Module):
    """
    Маскированная свёртка с нормализацией по количеству валидных пикселей
    и прокидыванием новой маски дальше по сети.
    """
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=groups, bias=bias)
        kH, kW = self.conv.kernel_size if isinstance(self.conv.kernel_size, tuple) else (self.conv.kernel_size, self.conv.kernel_size)
        # Ядро из единиц для подсчёта количества валидных пикселей в окне
        # Считаем по одной карте (=1 канал), затем расширяем по каналам выхода
        self.register_buffer("ones_kernel", torch.ones(1, 1, kH, kW))
        self.kernel_area = kH * kW
        self.eps = 1e-8

    def forward(self, x, mask):
        """
        x:    [B, C_in, H, W]
        mask: [B, 1,     H, W]  (бинарная {0,1} или [0..1])
        Возвращает:
          y:       [B, C_out, H', W']
          newmask: [B, 1,     H', W'] (где в окне был хоть один валидный пиксель)
        """
        # 1) Применяем маску к входу
        x_masked = x * mask

        # 2) Считаем обычную свёртку по замаскированному входу
        y = self.conv(x_masked)

        # 3) Счётчик валидных пикселей в каждом окне (геометрия та же, что у conv)
        cnt = F.conv2d(mask, self.ones_kernel,
                       stride=self.conv.stride, padding=self.conv.padding,
                       dilation=self.conv.dilation)

        # 4) Нормализация:
        # Вариант A (стабильная шкала): умножаем на (kernel_area / cnt)
        # чтобы сохранять порядок величин как у немаскированной свёртки
        scale = self.kernel_area / torch.clamp(cnt, min=self.eps)
        y = y * scale

        # Если где-то cnt==0 (ни одного валидного пикселя), зануляем выход
        zero_mask = (cnt <= 0)
        if zero_mask.any():
            y = y.masked_fill(zero_mask.expand_as(y), 0.0)

        # 5) Новая маска для следующего слоя: там, где было хоть что-то валидное
        newmask = (cnt > 0).float()

        return y, newmask


class ConvStateDeepEncoderModel(nn.Module):
    def __init__(self, size, in_channels=12, conv_dim=32, emb_dim=64, num_groups=8):
        super(ConvStateDeepEncoderModel, self).__init__()
        self.max_size = size
        self.mconv_list = nn.ModuleList([
            MaskedConv2d(in_channels - 1, conv_dim, kernel_size=2 * size + 1, padding=0, stride=2 * size + 1, bias=True)
            for size in range(1, self.max_size + 1)
        ])
        self.ln_list = nn.ModuleList([
            nn.LayerNorm(conv_dim, eps=1e-5, elementwise_affine=True)
            for size in range(1, self.max_size + 1)
        ])
        self.act_list = nn.ModuleList([
            nn.ReLU(inplace=True)
            for size in range(1, self.max_size + 1)
        ])
        self.fc_list = nn.ModuleList([
            nn.Linear(conv_dim, emb_dim)
            for size in range(1, self.max_size + 1)
        ])
        self.drop_list = nn.ModuleList([
            nn.Dropout(0.05)
            for size in range(1, self.max_size + 1)
        ])

    def forward(self, data):
        n = data.shape[-1]
        x_by_size = []
        for size, mconv, ln, act, fc, drop in zip(
            range(1, self.max_size + 1),
            self.mconv_list,
            self.ln_list,
            self.act_list,
            self.fc_list,
            self.drop_list,
        ):
            left = n // 2 - size
            right = n // 2 + size + 1
            x = data[:, 1:, left:right, left:right]
            mask = data[:, :1, left:right, left:right]
            x, mask = mconv(x, mask)
            x = ln(x.squeeze(-1).squeeze(-1)).unsqueeze(-1).unsqueeze(-1)
            x = act(x)
            x = x * mask
#             print(x.shape)
            x = torch.flatten(x, 1)
#             print(x.shape)
            x = fc(x)
            x = drop(x)
            x = x / (2 * size - 1) ** 2
            x_by_size.append(x.unsqueeze(1))
        x = torch.concatenate(x_by_size, axis=1)
        x = x.sum(dim=1)
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
        probs = self.forward(data)
        # Ensure probabilities are valid and add larger epsilon for numerical stability
        probs = torch.clamp(probs, min=1e-8, max=1.0)
        # Renormalize to ensure they sum to 1
        probs = probs / probs.sum(dim=1, keepdim=True)
        return torch.log(probs)


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
        probs = self.forward(data)
        # Ensure probabilities are valid and add larger epsilon for numerical stability
        probs = torch.clamp(probs, min=1e-8, max=1.0)
        # Renormalize to ensure they sum to 1
        probs = probs / probs.sum(dim=1, keepdim=True)
        return torch.log(probs)
