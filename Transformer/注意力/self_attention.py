from math import sqrt
import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self,dim_in,dim_k,dim_v):
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.linear_q = nn.Linear(dim_in,dim_k,bias = False)
        self.linear_k = nn.Linear(dim_in,dim_k,bias = False)
        self.linear_v = nn.Linear(dim_in,dim_v,bias = False)
        self._norm_fact = 1/sqrt(dim_k)

    def forward(self,x):
        batch,n,dim_in = x.shape
        assert dim_in==self.dim_in
        q = self.linear_q(x) # batch,n,dim_k
        k = self.linear_k(x) # 得到维度 batch,n,dim_k
        v = self.linear_v(x) # 得到维度 batch,n,dim_v
        dist = torch.bmm(q,k.transpose(1,2)) #两个tensor矩阵之间的乘积，得到结果为batch,n,n
        dist = torch.softmax(dist,dim=-1) #batch,n,n
        att = torch.bmm(dist,v)
        return att