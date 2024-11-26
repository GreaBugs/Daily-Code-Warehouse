import torch.nn as nn
import torch


# 搭建神经网络
class ShenYue(nn.Module):
    def __init__(self):
        super(ShenYue, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2, stride=1),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2, stride=1),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2, stride=1),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, input):
        output = self.model1(input)
        return output