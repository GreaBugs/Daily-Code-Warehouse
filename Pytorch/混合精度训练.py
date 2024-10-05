import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.cuda.amp import autocast, GradScaler

# 检查是否有可用的 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. 数据加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# 使用CIFAR10数据集作为示例
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# 2. 定义简单的模型（使用预训练的ResNet18）
model = torchvision.models.resnet18(pretrained=False)
model = model.to(device)

# 3. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 4. 创建GradScaler对象，用于动态损失缩放
scaler = GradScaler()

# 5. 训练模型（带混合精度训练）
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        
        # 使用 autocast 进行前向传播（混合精度）
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        # 清空梯度
        optimizer.zero_grad()
        
        # 反向传播（使用混合精度）
        scaler.scale(loss).backward()
        
        # 更新优化器参数
        scaler.step(optimizer)
        
        # 更新缩放器
        scaler.update()
        
        running_loss += loss.item()
        if i % 100 == 99:  # 每100个batch打印一次
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}], Loss: {running_loss / 100:.4f}')
            running_loss = 0.0

print('训练完成')

# 6. 在测试集上评估模型
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        # 使用autocast进行推理
        with autocast():
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

print(f'测试集准确率: {100 * correct / total}%')
