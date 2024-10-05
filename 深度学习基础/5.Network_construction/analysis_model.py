import torch
from torchvision import models
import thop


if __name__ == '__main__':
    # resnet18 模型
    model = models.resnet18(weights=None)

    # 输入数据
    inputs = torch.randn(1, 3, 224, 224)

    # 使用 thop 计算 FLOPs 和参数量
    MACs, Params = thop.profile(model, inputs=(inputs,), verbose=False)
    FLOPs = MACs * 2
    MACs, FLOPs,  Params = thop.clever_format([MACs, FLOPs, Params], "%.3f")

    print(f"MACs: {MACs}")   # MACs: 1.824G
    print(f"FLOPs: {FLOPs}")  # FLOPs: 3.648G
    print(f"Params: {Params}")   # Params: 11.690M