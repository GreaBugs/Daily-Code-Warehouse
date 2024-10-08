import torch
import torch.nn as nn
from Python._logging import create_logger


# 定义RMSNorm层
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(tuple(dim)))

    def forward(self, x):
        # x: (bs, seq_n, dim)
        # 计算均方根
        _dim = [i for i in range(1, len(x.shape))]
        
        norm = torch.sqrt(torch.mean(x ** 2, dim=_dim, keepdim=True) + self.eps)
        return self.weight * x / norm


if __name__ == '__main__':
    logger = create_logger("RMSlogging")
    
    # 输入测试
    x = torch.rand([2, 3, 8])
    
    # Pytorch实现RMS 
    rms1 = nn.RMSNorm([3, 8])
    
    # 自实现RMS
    rms2 = RMSNorm([3, 8])
    
    # 输出对比
    logger.info(f'\n 自实现输出: {rms2(x)} \n Pytorch实现输出: {rms1(x)}')