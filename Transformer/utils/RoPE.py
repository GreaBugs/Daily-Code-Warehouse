# https://blog.csdn.net/weixin_43646592/article/details/130924280
import math
import torch
import torch.nn.functional as F
from Python._logging import create_logger


logger = create_logger("RoPE")


def sinusoidal_position_embedding(batch_size, nums_head, max_len, output_dim, device):
    # (max_len, 1)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(-1)
    # (output_dim//2)
    ids = torch.arange(0, output_dim // 2, dtype=torch.float)  # 即公式里的i, i的范围是 [0, d/2]
    theta = torch.pow(10000, -2 * ids / output_dim)  # size=16

    # (max_len, output_dim//2)
    embeddings = position * theta  # 即公式里的：pos / (10000^(2i/d))  size=(10, 16)

    # (max_len, output_dim//2, 2)
    embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)  # (10, 16, 2)

    # (bs, head, max_len, output_dim//2, 2)
    embeddings = embeddings.repeat((batch_size, nums_head, *([1] * len(embeddings.shape))))  # 在bs维度重复，其他维度都是1不重复

    # (bs, head, max_len, output_dim)
    # reshape后就是：偶数sin, 奇数cos了
    embeddings = torch.reshape(embeddings, (batch_size, nums_head, max_len, output_dim))
    embeddings = embeddings.to(device)
    
    return embeddings


def RoPE(q, k):
    # q, k: (bs, head, max_len, output_dim)
    batch_size = q.shape[0]
    nums_head = q.shape[1]
    max_len = q.shape[2]
    output_dim = q.shape[-1]

    # (bs, head, max_len, output_dim)
    pos_emb = sinusoidal_position_embedding(batch_size, nums_head, max_len, output_dim, q.device)

    # cos_pos,sin_pos: (bs, head, max_len, output_dim)
    # 看rope公式可知，相邻cos，sin之间是相同的，所以复制一遍。如(1,2,3)变成(1,1,2,2,3,3)
    cos_pos = pos_emb[...,  1::2].repeat_interleave(2, dim=-1)  # 将奇数列信息抽取出来也就是cos 拿出来并复制
    sin_pos = pos_emb[..., ::2].repeat_interleave(2, dim=-1)  # 将偶数列信息抽取出来也就是sin 拿出来并复制

    # q, k: (bs, head, max_len, output_dim)
    q2 = torch.stack([-q[..., 1::2], q[..., ::2]], dim=-1)  # (bs, head, max_len, output_dim // 2, 2)
    q2 = q2.reshape(q.shape)  # reshape后就是正负交替了

    # 更新qw, *对应位置相乘
    q = q * cos_pos + q2 * sin_pos

    k2 = torch.stack([-k[..., 1::2], k[..., ::2]], dim=-1)
    k2 = k2.reshape(k.shape)
    # 更新kw, *对应位置相乘
    k = k * cos_pos + k2 * sin_pos

    return q, k


def attention(q, k, v, mask=None, dropout=None, use_RoPE=True):
    # q.shape: (bs, head, seq_len, dk)
    # k.shape: (bs, head, seq_len, dk)
    # v.shape: (bs, head, seq_len, dk)

    if use_RoPE:
        q, k = RoPE(q, k)

    d_k = k.size()[-1]

    att_logits = torch.matmul(q, k.transpose(-2, -1))  # (bs, head, seq_len, seq_len)
    att_logits /= math.sqrt(d_k)

    if mask is not None:
        att_logits = att_logits.masked_fill(mask == 0, -1e9)  # mask掉为0的部分，设为无穷大

    att_scores = F.softmax(att_logits, dim=-1)  # (bs, head, seq_len, seq_len)

    if dropout is not None:
        att_scores = dropout(att_scores)

    # (bs, head, seq_len, seq_len) * (bs, head, seq_len, dk) = (bs, head, seq_len, dk)
    return torch.matmul(att_scores, v), att_scores


if __name__ == "__main__":
    # (bs, head, seq_len, dk)
    q = torch.randn((8, 12, 10, 32))
    k = torch.randn((8, 12, 10, 32))
    v = torch.randn((8, 12, 10, 32))

    res, att_scores = attention(q, k, v, mask=None, dropout=None, use_RoPE=True)
    # (bs, head, seq_len, dk),  (bs, head, seq_len, seq_len)
    logger.info(f'\n res.shape: {res.shape}, \n att_scores.shape: {att_scores.shape}')