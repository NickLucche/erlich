import torch
import torch.nn as nn

from .base import Identity
from .convolutions import conv


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, normalization=None):
        super().__init__()

        self.layers = nn.Sequential(
            conv(in_channels, out_channels, 3, stride=stride, normalization=normalization),
            conv(out_channels, out_channels, 3, stride=1, normalization=normalization, activation=None)
        )

        if stride != 1 or in_channels != out_channels:
            if stride == 2:
                self.identity = nn.Sequential(
                    nn.AvgPool2d(2, stride=2),
                    conv(in_channels, out_channels, 1, stride=1, normalization=None, activation=None)
                )
            else:
                self.identity = conv(in_channels, out_channels, 1, stride=stride, normalization=None,
                                     activation=None)
        else:
            self.identity = Identity()

        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        return self.relu(self.layers(x)) + self.identity(x)


class DenseBlock(nn.Module):
    def __init__(self, input_channels, growth_rate, n_layers, activation="relu", normalization=None):
        super().__init__()

        ch = input_channels
        layers = []
        for i in range(n_layers):
            layers.append(conv(ch, growth_rate, 3, activation=activation, normalization=normalization))
            ch += growth_rate

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = torch.cat((x, layer(x)), dim=1)

        return x
