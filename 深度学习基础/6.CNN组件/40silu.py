import torch


def swish(x):
    return x * torch.sigmoid(x)