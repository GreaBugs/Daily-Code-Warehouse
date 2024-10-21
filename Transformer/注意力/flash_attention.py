import torch
import torch.nn.functional as F
from xformers.ops import memory_efficient_attention
import time


# 设置设备为GPU或CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 定义随机输入
batch_size = 16
seq_len = 1024  # 长序列
d_model = 64   # 特征维度


# 生成Q, K, V矩阵
Q = torch.randn(batch_size, seq_len, d_model).to(device)
K = torch.randn(batch_size, seq_len, d_model).to(device)
V = torch.randn(batch_size, seq_len, d_model).to(device)


# 普通注意力函数实现
def normal_attention(Q, K, V):
    # Scaled Dot-Product Attention
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, V)
    return output


# Flash Attention的简单模拟实现
def flash_attention(Q, K, V, block_size=64):
    # 计算注意力分数，使用块内计算减少内存消耗
    d_k = Q.size(-1)
    output = torch.zeros_like(Q)

    # 按块计算，逐步更新结果
    for i in range(0, seq_len, block_size):
        for j in range(0, seq_len, block_size):
            # 提取块
            Q_block = Q[:, i:i+block_size]
            K_block = K[:, j:j+block_size]
            V_block = V[:, j:j+block_size]
            
            # 计算块间的注意力分数
            scores = torch.matmul(Q_block, K_block.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
            attention_weights = F.softmax(scores, dim=-1)
            
            # 更新结果
            output[:, i:i+block_size] += torch.matmul(attention_weights, V_block)
    
    return output


# 使用xformers的Flash Attention实现
def flash_attention_xformers(Q, K, V):
    return memory_efficient_attention(Q, K, V)


if __name__ == "__main__":
    
    # 测试普通注意力的时间
    start_time = time.time()
    output_normal = normal_attention(Q, K, V)
    normal_attention_time = time.time() - start_time
    print(f"普通注意力计算时间: {normal_attention_time:.4f} 秒")

    # 测试Flash Attention模拟的时间
    start_time = time.time()
    output_flash = flash_attention(Q, K, V)
    flash_attention_time = time.time() - start_time
    print(f"Flash Attention模拟计算时间: {flash_attention_time:.4f} 秒")

    # 对比输出结果
    print(f"输出结果差异 (L2 范数): {torch.norm(output_normal - output_flash):.4f}")


    # 测试Flash Attention模拟的时间
    start_time = time.time()
    # 计算Flash Attention
    output_flash = flash_attention_xformers(Q, K, V)
    flash_attention_time = time.time() - start_time
    print(f"Flash Attention模拟计算时间: {flash_attention_time:.4f} 秒")

