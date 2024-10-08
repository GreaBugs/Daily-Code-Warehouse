import torch
import torch.nn as nn
import numpy as np


class LayerNorm(nn.Module):
		def __init__(self, hidden_size, eps=1e-12):
				"""Construct a layernorm module in the TF style (epsilon inside the square root)."""
				super(LayerNorm, self).__init__()
				self.weight = nn.Parameter(torch.ones(hidden_size))
				self.bias = nn.Parameter(torch.zeros(hidden_size))
				self.variance_epsilon = eps

		def forward(self, x):
				u = x.mean(-1, keepdim=True)
				s = (x - u).pow(2).mean(-1, keepdim=True)
				x = (x - u) / torch.sqrt(s + self.variance_epsilon)
				return self.weight * x + self.bias
			

if __name__ == "__main__":
    # 自实现LN
	ln = LayerNorm(3)  
 
    # 输入测试
	feature_array = np.array([[[[1, 0],  [0, 2]],
                               [[3, 4],  [1, 2]],
                               [[2, 3],  [4, 2]]],

                              [[[1, 2],  [-1, 0]],
                               [[1, 2], [3, 5]],
                               [[1, 4], [1, 5]]]], dtype=np.float32)  # (2, 3, 2, 2)


	feature_array = feature_array.reshape((2, 3, -1)).transpose(0, 2, 1)  # (2, 4, 3)
	feature_tensor = torch.tensor(feature_array.copy(), dtype=torch.float32)  # (2, 4, 3)

    # Pytorch实现
	ln_out = nn.LayerNorm(normalized_shape=3)(feature_tensor)
 
    # 输出对比
	print(ln_out)
	print(ln(feature_tensor))