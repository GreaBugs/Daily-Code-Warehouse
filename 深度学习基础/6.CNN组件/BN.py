import torch
from torch import nn
from torch.nn import Module


class BatchNorm(Module):
    def __init__(self, channels: int, *,
                 eps: float = 1e-5, momentum: float = 0.1,
                 affine: bool = True, track_running_stats: bool = True):
        super().__init__()

        self.channels = channels  # 3
        self.eps = eps  # 1e-05
        self.momentum = momentum  # 0.1
        self.affine = affine  # True
        self.track_running_stats = track_running_stats  # True

        if self.affine:
            self.scale = nn.Parameter(torch.ones(channels))  # Parameter containing: tensor([1., 1., 1.], requires_grad=True)
            self.shift = nn.Parameter(torch.zeros(channels))  # Parameter containing: tensor([0., 0., 0.], requires_grad=True)

        if self.track_running_stats:
            self.register_buffer('exp_mean', torch.zeros(channels))  # tensor([0., 0., 0.])
            self.register_buffer('exp_var', torch.ones(channels))  # tensor([1., 1., 1.])

    def forward(self, x: torch.Tensor):
        x_shape = x.shape  # torch.Size([2, 3, 2, 4])
        batch_size = x_shape[0]  # 2
        assert self.channels == x.shape[1]
        x = x.view(batch_size, self.channels, -1)  # (2, 3, 8)

        if self.training or not self.track_running_stats:
            mean = x.mean(dim=[0, 2])
            mean_x2 = (x ** 2).mean(dim=[0, 2])
            var = mean_x2 - mean ** 2  # 方差公式： σ2=(x2).mean − mean2

            if self.training and self.track_running_stats:
                self.exp_mean = (1 - self.momentum) * self.exp_mean + self.momentum * mean
                self.exp_var = (1 - self.momentum) * self.exp_var + self.momentum * var

        else:
            mean = self.exp_mean
            var = self.exp_var
        x_norm = (x - mean.view(1, -1, 1)) / torch.sqrt(var + self.eps).view(1, -1, 1)

        if self.affine:
            x_norm = self.scale.view(1, -1, 1) * x_norm + self.shift.view(1, -1, 1)
        return x_norm.view(x_shape)


if __name__ == '__main__':
    x = torch.rand([2, 3, 2, 4])
    bn = BatchNorm(3)
    x = bn(x)
    print(x)