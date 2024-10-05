from math import sqrt
import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim_in, dim_k, dim_v, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        assert dim_k%num_heads == 0 and dim_v%num_heads==0,"dim_k and dim_v must be multiple of num_heads"
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.num_heads = num_heads
        self.linear_q = nn.Linear(dim_in,dim_k,bias=False)
        self.linear_k = nn.Linear(dim_in,dim_k,bias=False)
        self.linear_v = nn.Linear(dim_in,dim_v,bias=False)
        self._norm_fact = 1/sqrt(dim_k//num_heads)

    def forward(self,x):
        batch,n,dim_in = x.shape
        assert dim_in ==self.dim_in
        nh = self.num_heads
        dk = self.dim_k//nh
        dv = self.dim_v //nh
        q = self.linear_q(x).reshape(batch,n,nh,dk).transpose(1,2) # (batch, nh, n, dk)
        k = self.linear_k(x).reshape(batch,n,nh,dk).transpose(1,2) # (batch, nh, n, dk)
        v = self.linear_v(x).reshape(batch,n,nh,dv).transpose(1,2) # (batch, nh, n, dv)
        # (batch, nh, n, n)每个头一起算分数
        dist = torch.matmul(q,k.transpose(2,3)) * self._norm_fact
        dist = torch.softmax(dist, dim=-1) # (batch, nh, n, n)
        att = torch.matmul(dist,v)
        att = att.transpose(1,2).reshape(batch,n,self.dim_v)
        return att