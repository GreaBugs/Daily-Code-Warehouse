import torch
from torch import nn
import netron
import onnx
from onnx import shape_inference


class AlexNet(nn.Module):
    def __init__(self, in_channel, num_classes):
        super(AlexNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channel, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten(),
            nn.Linear(6400, 4096), nn.ReLU(), nn.Dropout(),
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(),
            nn.Linear(4096, num_classes))

    def forward(self, x):
        out = self.net(x)
        return out


model = AlexNet(3, 10)  # 模型实例化
img = torch.randn((1, 3, 224, 224))

# 将模型导出，命名为：model.onnx
torch.onnx.export(model=model, args=img, f='model.onnx', input_names=['images'], output_names=['feature_map'])

# 可以推理一遍，保存模型的中间shape并可视化 图（右）；否则为图（中）
onnx.save(onnx.shape_inference.infer_shapes(onnx.load("model.onnx")), "model.onnx")
netron.start('model.onnx')  