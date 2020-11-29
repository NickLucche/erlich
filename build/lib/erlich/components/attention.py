from erlich.components.base import get_activation
from matplotlib.pyplot import sca
import torch.nn as nn
import torch.nn.quantized
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, input_channels, hidden_size, activation="relu"):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_channels, hidden_size),
            get_activation(activation),
            nn.Linear(hidden_size, input_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        mean = torch.mean(x, dim=(2, 3))

        w = self.layers(mean).unsqueeze(-1).unsqueeze(-1)
        return x * w


class AttentionPool(nn.Module):
    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = scale

    def forward(self, x, attn_raw):
        batch = x.size(0)
        attention_channels = attn_raw.size(1)

        attn_raw = attn_raw * self.scale

        attn = F.softmax(attn_raw.view(batch, attention_channels, -1), dim=2)
        attn = attn.view(batch, attention_channels, x.size(2), x.size(3))

        return torch.einsum("bcwh,bawh->bac", x, attn).reshape(x.size(0), -1), attn