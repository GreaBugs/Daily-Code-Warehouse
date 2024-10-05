import torch
import torchvision
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import time

# 定义训练设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 准备数据集
train_dataset = torchvision.datasets.CIFAR10("../dataset", train=True, transform=torchvision.transforms.ToTensor(),
                                             download=True)
test_dataset = torchvision.datasets.CIFAR10("../dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                            download=True)

# length长度
train_dataset_size = len(train_dataset)
test_dataset_size = len(test_dataset)
print("训练数据集的长度为：{}".format(train_dataset_size))
print("测试数据集的长度为：{}".format(test_dataset_size))

# 利用DataLoader加载数据集
train_dataloader = DataLoader(train_dataset, batch_size=64)
test_dataloader = DataLoader(test_dataset, batch_size=64)

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

# 创建网络模型


shenyue = ShenYue()
# 网络模型转移到cuda上面
shenyue = shenyue.to(device)
# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

# 优化器
learning_rate = 0.01
optimizer = torch.optim.SGD(shenyue.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练次数
total_train_step = 0
# 记录测试次数
total_test_step = 0
# 训练轮数
epoch = 10
# 添加tensorboard
writer = SummaryWriter("../logs_train")
start_time = time.time()
for i in range(epoch):
    print("*"*25, end="")
    print("第{}轮训练开始了".format(i+1), end="")
    print("*"*25)
    # 训练开始
    shenyue.train()
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = shenyue(imgs)
        loss = loss_fn(outputs, targets)
        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if (total_train_step % 100) == 0:
            end_time = time.time()
            print("耗时：{}".format(end_time - start_time))
            print("训练次数：{}，Loss:{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始
    shenyue.eval()
    total_test_loss = 0
    total_accuracy = 0
    # 该语句中的内容无梯度，保证不会被调优
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = shenyue(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy
    print("整体测试集上的Loss为：{}".format(total_test_loss))
    print("整体测试集上的正确率为：{}".format(total_accuracy/test_dataset_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_dataset_size, total_test_step)
    total_test_step += 1

    torch.save(shenyue.state_dict(), "shenyue_{}.pth".format(i))
    print("模型已保存")

writer.close()