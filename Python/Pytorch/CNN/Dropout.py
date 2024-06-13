import torch
import torch.nn as nn


class Dropout(nn.Module):
    def __init__(self, p, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.p = p

    def forward(self, input_matrix):
        assert 0 <= self.p <= 1
        if self.p == 1:
            return torch.zeros_like(input_matrix)
        if self.p == 0:
            return input_matrix
        mask = (torch.rand(input_matrix.shape) > self.p).float()
        return mask * input_matrix / (1.0 - self.p)


if __name__ == '__main__':
    X = torch.tensor([[0.0, 1.0, 2.0],
                      [3.0, 4.0, 5.0],
                      [6.0, 7.0, 8.0]])
    dropout = Dropout(0.5)
    print(dropout(X))