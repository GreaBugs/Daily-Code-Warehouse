import torch.nn as nn
import torch.nn.functional as F


# ResNet的基本Bottleneck类
class Bottleneck(nn.Module):   # 瓶颈结构 (维度变化，则对应Conv Block    维度不变，则对应Identity Block)
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):  # inplanes是输入通道数  planes是输出通道数
        super(Bottleneck, self).__init__()
        # 压缩通道数  这个瓶颈结构可以更好的提取特征，加深网络
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # 特征提取
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # 扩张通道数
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x  # x=Tensor:(1,64,150,150)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:   # 如果残差边上有卷积，则执行
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# FNP的类，初始化需要一个list，代表RESNET的每一个阶段的Bottleneck的数量
class FPN(nn.Module):
    def __init__(self, block, layers):
        super(FPN, self).__init__()
        self.inplanes = 64
        # 处理输入的C1模块（C1代表了RestNet的前几个卷积与池化层）
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 搭建自下而上的C2，C3，C4，C5
        # 150,150,64 -> 150,150,256
        self.layer1 = self._make_layer(block, 64, layers[0])    # (Bottleneck, 64, 3)
        # 150,150,256 -> 75,75,512
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # 75,75,512 -> 38,38,1024 到这里可以获得一个38,38,1024的共享特征层
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # self.layer4被用在classifier模型中
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # 对C5减少通道数，得到P5
        self.toplayer = nn.Conv2d(in_channels=2048, out_channels=256, kernel_size=1, stride=1, padding=0)
        # 3x3卷积融合特征
        self.fpn1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.fpn2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.fpn3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        # 横向连接，保证通道数相同
        self.lateral_layer1 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.lateral_layer2 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.lateral_layer3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0)

    def _make_layer(self, block, planes, blocks, stride=1):  # (Bottleneck, in_channel, 卷积层循环次数)
        downsample = None   # 残差边
        #-------------------------------------------------------------------#
        #   当模型需要进行高和宽的压缩的时候，就需要用到残差边的downsample
        #-------------------------------------------------------------------#
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))  # block(64, planes)
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    # 自上而下的采样模块
    def _upsample_add(self, x, y):
        _, _, H, W = y.shape
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def forward(self, x):
        # 自下而上
        c1 = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        # 自上而下
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.lateral_layer1(c4))
        p3 = self._upsample_add(p4, self.lateral_layer2(c3))
        p2 = self._upsample_add(p3, self.lateral_layer3(c2))
        # 卷积的融合，平滑处理
        p4 = self.fpn1(p4)
        p3 = self.fpn2(p3)
        p2 = self.fpn3(p2)
        return p2, p3, p4, p5


if __name__ == '__main__':
    net = FPN(Bottleneck, [3, 4, 6, 3])

    # 遍历模型的参数并打印形状
    for name, param in net.named_parameters():
        print(name, param.shape)