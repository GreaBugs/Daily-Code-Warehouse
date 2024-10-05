import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils import data
import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision.models as models
import logging
from copy import deepcopy
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ================================= 预处理图片 =================================

transform = transforms.Compose([transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.2435, 0.2616])
                                ])

# ================================= 获取数据集，生成 dataset ===========================

cifar10_dir = './cifar10'
train_set = torchvision.datasets.CIFAR10(cifar10_dir, train=True, download=True, transform=transform)
test_set = torchvision.datasets.CIFAR10(cifar10_dir, train=False, download=True, transform=transform)

# ================================= 定义 dataloader =================================

train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=True)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# ================================= 定义模型 =================================


class My_model(nn.Module):
    def __init__(self, resnet50):
        super(My_model, self).__init__()
        self.resnet50 = resnet50
        self.mlp = nn.Sequential(nn.Linear(in_features=2048, out_features=1024, bias=True),
                                 nn.ReLU(),
                                 nn.Linear(in_features=1024, out_features=256, bias=True),
                                 nn.ReLU(),
                                 nn.Linear(in_features=256, out_features=10, bias=True))

    def forward(self, x):
        for name, layer in self.resnet50.named_children():
            if 'fc' not in name:
                x = layer(x)    
        x = x.view(-1, 2048)
        x = self.mlp(x)
        return x

resnet50 = models.resnet50(weights="DEFAULT")
model = My_model(resnet50)
model.to(device)

# ================================= EMA =================================

class ModelEMA:
    def __init__(self, model, decay=0.9998, tau=2000, updates=0):
        # make a copy of the model for accumulating moving average of weights
        self.ema = deepcopy(model).eval() 
        self.updates = updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  
        
        for p in self.ema.parameters():
            p.requires_grad = False

    def update(self, model):
        self.updates += 1
        d = self.decay(self.updates)

        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:  
                v *= d
                v += (1 - d) * msd[k].detach()

    def update_attr(self, model):
        for k in model.__dict__.keys():
            if not k.startswith('_'):
                setattr(model, k, getattr(model, k))

ema = ModelEMA(model, 0.98)
    
# ================================= 训练模型 =================================

logging.basicConfig(format="%(message)s", level=logging.INFO)
console = logging.StreamHandler()
logging.getLogger().addHandler(console)

epochs = 20
lr = 0.01
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

eval_losses = []
eval_acces = []

for epoch in range(epochs):
    print('epoch : {}'.format(epoch))
    if (epoch + 1) % 5 == 0:
        optimizer.param_groups[0]['lr'] *= 0.1

    model.train()
    for steps, (imgs, labels) in enumerate(train_loader):
        imgs, labels = imgs.to(device), labels.to(device)
        predict = model(imgs)
        loss = criterion(predict, labels)

        # back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 更新 EMA
        ema.update(model)
    print('train loss: {}'.format(loss))
    
    eval_loss = 0
    eval_acc = 0
    ema.ema.eval()
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        predict = ema.ema(imgs)
        loss = criterion(predict, labels)

        # record loss
        eval_loss += loss.item()

        # record accurate rate
        result = torch.argmax(predict, dim=1)
        acc_num = (result == labels).sum().item()
        acc_rate = acc_num / imgs.shape[0]
        eval_acc += acc_rate

    eval_losses.append(eval_loss / len(test_loader))
    eval_acces.append(eval_acc / len(test_loader))

    print('eval loss： {}'.format(eval_loss / len(test_loader)))
    print('eval accurate rate: {}'.format(eval_acc / len(test_loader)))
    print('\n')


plt.title('evaluation loss')
plt.plot(np.arange(len(eval_losses)), eval_losses)
plt.show()
