import torch
import torch.nn as nn


class Pooling2d(nn.Module):
    def __init__(self, pooling_size, mode="avg", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pooling_size = pooling_size
        self.mode = mode

    def pooling2d(self, input_matrix):
        assert self.mode in ['max', 'avg']
        p_h, p_w = self.pooling_size
        input_matrix_h, input_matrix_w = input_matrix.shape
        output = torch.zeros((input_matrix_h - p_h + 1, input_matrix_w - p_w + 1))
        for row_i in range(output.shape[0]):
            for col_i in range(output.shape[0]):
                if self.mode == 'max':
                    output[row_i, col_i] = X[row_i:row_i + p_h, col_i:col_i + p_w].max()
                elif self.mode == 'avg':
                    output[row_i, col_i] = X[row_i:row_i + p_h, col_i:col_i + p_w].mean()
        return output

    def forward(self, input_matrix):
        return self.pooling2d(input_matrix)


if __name__ == '__main__':
    X = torch.tensor([[0.0, 1.0, 2.0],
                      [3.0, 4.0, 5.0],
                      [6.0, 7.0, 8.0]])
    pool2d = Pooling2d((2, 2), 'avg')
    print(pool2d(X), '\n', pool2d(X))