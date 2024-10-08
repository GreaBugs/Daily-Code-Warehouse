import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear


# 定义P-tuning v2 风格的 ShenYue 网络
class ShenYuePTuning(nn.Module):
    def __init__(self):
        super(ShenYuePTuning, self).__init__()
        
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2, stride=1),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2, stride=1),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2, stride=1),
            MaxPool2d(2),
            Flatten()
        )

        # 引入可学习的提示层 (prompt) 插入全连接层之前
        self.prompt = nn.Parameter(torch.randn(64))  # 可学习的prompt向量
        
        # 原网络的全连接层部分可微调
        self.fc1 = Linear(1024 + 64, 64)  # 将prompt向量与Flatten层的输出拼接
        self.fc2 = Linear(64, 10)

    def forward(self, input):
        x = self.model1(input)

        # 拼接可学习的prompt向量
        x = torch.cat((x, self.prompt.expand(x.size(0), -1)), dim=1)

        # 可微调的全连接层
        x = self.fc1(x)
        output = self.fc2(x)

        return output


model = ShenYuePTuning()

# 优化器，只更新部分参数 (fc层和prompt)
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), 
    lr=1e-4
)

criterion = nn.CrossEntropyLoss()

for param in model.model1.parameters():
    param.requires_grad = False

# 打印模型结构，验证冻结状态
for name, param in model.named_parameters():
    print(f"{name}: requires_grad={param.requires_grad}")
