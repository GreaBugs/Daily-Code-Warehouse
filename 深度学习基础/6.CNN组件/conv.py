from typing import Union
import torch
import torch.nn as nn


class Conv2d(nn.Module):
    def __init__(self, kernel_size: Union[int, int] = 3, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    @staticmethod
    def corr2d(input_matrix: torch.tensor, kernel_matrix: torch.tensor) -> torch.tensor:
        '''
            input_matrix: shape is (h ,w)
            kernel_matrix: shape is (h, w)

            X = torch.tensor([
                [0.0, 1.0, 2.0],
                [3.0, 4.0, 5.0],
                [6.0, 7.0, 8.0]
                ])
            K = torch.tensor([
                [0.0, 1.0],
                [2.0, 3.0]
                ])
        '''
        input_h, input_w = input_matrix.shape
        kernel_h, kernel_w = kernel_matrix.shape
        output = torch.zeros((input_h - kernel_h + 1, input_w - kernel_w + 1))
        for row_i in range(output.shape[0]):
            for col_i in range(output.shape[1]):
                output[row_i][col_i] = (
                            input_matrix[row_i: row_i + kernel_h, col_i: col_i + kernel_w] * kernel_matrix).sum()
        return output

    def multi_to_one(self, input_matrix: torch.tensor, kernel_matrix: torch.tensor) -> torch.tensor:
        '''
            input_matrix: shape is (c, h ,w)
            kernel_matrix: shape is (c, h w)

            X = torch.tensor(
                [
                    [[0.0, 1.0, 2.0],
                    [3.0, 4.0, 5.0],
                    [6.0, 7.0, 8.0]],

                    [[0.0, 1.0, 2.0],
                    [3.0, 4.0, 5.0],
                    [6.0, 7.0, 8.0]]
                    ])

            K = torch.tensor([
                [[0.0, 1.0],
                [2.0, 3.0]],

                [[0.0, 1.0],
                [2.0, 3.0]]]
                )
        '''
        input_matrix_c, input_matrix_h, input_matrix_w = input_matrix.shape
        kernel_matrix_c, kernel_matrix_h, kernel_matrix_w = kernel_matrix.shape

        assert input_matrix_c == kernel_matrix_c
        output = torch.zeros((1, input_matrix_h - kernel_matrix_h + 1, input_matrix_w - kernel_matrix_w + 1))
        for input_one_channel, kernel_one_channel in zip(input_matrix, kernel_matrix):
            output += self.corr2d(input_one_channel, kernel_one_channel)
        return output

    def multi_to_multi(self, input_matrix: torch.tensor, kernel_matrix: torch.tensor) -> torch.tensor:
        '''
            input_matrix: shape is (cin, h ,w)
            kernel_matrix: shape is (cout, cin, h, w)

            X = torch.tensor([
                [[0.0, 1.0, 2.0],
                [3.0, 4.0, 5.0],
                [6.0, 7.0, 8.0]],

                [[1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0]]
                ])

            K = torch.tensor([
                [[0.0, 1.0],
                [2.0, 3.0]],

                [[1.0, 2.0],
                [3.0, 4.0]]]
                            )
            K = torch.stack((K, K + 1, K + 2), dim=0)
        '''
        input_matrix_cin, input_matrix_h, input_matrix_w = input_matrix.shape
        kernel_matrix_cout, kernel_matrix_cin, kernel_matrix_h, kernel_matrix_w = kernel_matrix.shape
        assert input_matrix_cin == kernel_matrix_cin, "input_matrix_cin:{}, kernel_matrix_cin:{}".format(
            input_matrix_cin, kernel_matrix_cin)

        output = torch.zeros(
            (kernel_matrix_cout, input_matrix_h - kernel_matrix_h + 1, input_matrix_w - kernel_matrix_w + 1))
        for cout_i, kernel_matrix_one_channel in enumerate(kernel_matrix):
            output[cout_i] = self.multi_to_one(input_matrix, kernel_matrix_one_channel)

        return output

    def forward(self, input_matrix: torch.tensor, kernel_matrix=None) -> torch.tensor:

        output = self.multi_to_multi(input_matrix, kernel_matrix)
        return output