from typing import Union
import torch.nn as nn
import torch.nn.functional as F
from .base import get_activation


def upscale(x, s=2):
    return F.interpolate(x, scale_factor=s, mode="bilinear", align_corners=True)


class Upscale(nn.Module):
    def __init__(self, s=2):
        super().__init__()
        self.s = s

    def forward(self, x):
        return upscale(x, self.s)


def separable_conv(in_ch, out_ch=0, size=3, stride=1, normalization=None, activation="relu", pad: Union(int, str) = "same"):
    if pad == "same":
        padding = size // 2
    else:
        padding = int(pad)

    if out_ch <= 0:
        out_ch = in_ch

    if normalization is not None:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            normalization(out_ch),
            nn.Conv2d(out_ch, out_ch, size, stride=stride, padding=padding, bias=False, groups=out_ch),
            normalization(out_ch),
            get_activation(activation)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.Conv2d(out_ch, out_ch, size, stride=stride, padding=padding, bias=True, groups=out_ch),
            get_activation(activation)
        )


def conv(in_ch, out_ch=0, size=3, stride=1, normalization=None, activation="relu", pad: Union(int, str) = "same"):
    if pad == "same":
        padding = size // 2
    else:
        padding = int(pad)

    if out_ch <= 0:
        out_ch = in_ch

    if normalization is not None:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, size, stride=stride, padding=padding, bias=False),
            normalization(out_ch),
            get_activation(activation)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, size, stride=stride, padding=padding, bias=True),
            get_activation(activation)
        )
