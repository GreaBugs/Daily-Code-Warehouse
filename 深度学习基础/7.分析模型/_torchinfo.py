import torch
import torch.nn as nn
from torchinfo import summary

# 定义RMSNorm层
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # 计算均方根
        norm = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.scale * x / norm

# 自定义模型类
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.rms = RMSNorm(256)  # RMSNorm

    def forward(self, x):
        x_rms = self.rms(x)
        return x_rms

# 实例化模型并移动到 GPU
model = MyModel().to('cuda')

# 打印模型总结
summary(model, input_size=(1, 64, 64, 256))  # 注意 input_size 包含 batch size
