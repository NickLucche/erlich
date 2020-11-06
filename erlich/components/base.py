import torch.nn as nn


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


def get_activation(name, inplace=True):
    if name is None:
        return Identity()
    if name == "relu":
        return nn.ReLU(inplace=inplace)
    elif name == "prelu":
        return nn.PReLU()
    elif name == "leaky relu":
        return nn.LeakyReLU()
    elif name == "hard swish":
        return nn.Hardswish()
    else:
        raise Exception(f"Unknown activation '{name}'")
