import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.in1 = nn.InstanceNorm2d(in_channels, affine=True)
        self.in2 = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x):
        x = self.depthwise(x)  # (1, 32, 16, 16) -> (1, 32, 16, 16)
        x = self.in1(x)
        x = F.relu(x)
        x = self.pointwise(x)  # (1, 64, 16, 16)
        x = self.in2(x)
        return F.relu(x)

class MobileNet(nn.Module):
    def __init__(self, num_classes_10=10, num_classes_24=24):
        super(MobileNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(32, 64, stride=1),
            DepthwiseSeparableConv(64, 128, stride=2),
            DepthwiseSeparableConv(128, 128, stride=1),
            DepthwiseSeparableConv(128, 256, stride=2),
            DepthwiseSeparableConv(256, 256, stride=1),
            DepthwiseSeparableConv(256, 512, stride=2),
            *[DepthwiseSeparableConv(512, 512, stride=1) for _ in range(5)],
            DepthwiseSeparableConv(512, 1024, stride=2),
            DepthwiseSeparableConv(1024, 1024, stride=1),
        )
        self.fc_10 = nn.Linear(1024, num_classes_10)
        self.fc_24 = nn.Linear(1024, num_classes_24)

    def forward(self, x):
        x = self.model(x)
        x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
        out_10 = self.fc_10(x)
        out_24 = self.fc_24(x)
        return out_10, out_24

# 示例：创建 MobileNet 模型并打印结构
model = MobileNet(num_classes_10=10, num_classes_24=24)
print(model)

# 测试输入
input_tensor = torch.randn(1, 3, 32, 32)
output_10, output_24 = model(input_tensor)
print(output_10.shape, output_24.shape)  # 应输出 (1, 10) 和 (1, 24)
