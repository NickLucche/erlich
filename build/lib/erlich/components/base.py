import torch
import torch.nn as nn


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class SwishF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish(nn.Module):
    def forward(self, input_tensor):
        return SwishF.apply(input_tensor)


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
    elif name == "swish":
        return Swish()
    else:
        raise Exception(f"Unknown activation '{name}'")
