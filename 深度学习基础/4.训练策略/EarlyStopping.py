# 理论笔记：https://cariclpajpr.feishu.cn/wiki/Eh4dwbXIXipnSIkKZHAc1fK7njd
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

train_loader = torch.utils.data.DataLoader(train_set, batch_size=512, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=512, shuffle=True)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# ================================= 定义模型 =================================

model = models.alexnet(weights="DEFAULT")
model.classifier[4] = nn.Linear(in_features=4096, out_features=1024, bias=True)
model.classifier[6] = nn.Linear(in_features=1024, out_features=10, bias=True)
model.to(device)
   
# ================================= EarlyStopping =================================

class EarlyStopping:
    def __init__(self, patience=3):
        self.best_accuracy = 0.0
        self.best_epoch = 0
        self.patience = patience or float("inf")  

    def __call__(self, epoch, accuracy):
        if accuracy >= self.best_accuracy:
            self.best_epoch = epoch
            self.best_accuracy = accuracy
            
        delta = epoch - self.best_epoch  
        stop = delta >= self.patience  
        
        if stop:
            logging.info(
                f"Stopping training early as no improvement observed in last {self.patience} epochs. "
                f"Best results observed at epoch {self.best_epoch}, best model saved as best.pt.\n"
            )
            
        return stop
    
stopper = EarlyStopping()
stop = False
    
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
    print('train loss: {}'.format(loss))
    
    eval_loss = 0
    eval_acc = 0
    model.eval()
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        predict = model(imgs)
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
    
    stop = stopper(epoch=epoch, accuracy=eval_acc / len(test_loader))
    
    if stop:
        break 

plt.title('evaluation loss')
plt.plot(np.arange(len(eval_losses)), eval_losses)
plt.show()
