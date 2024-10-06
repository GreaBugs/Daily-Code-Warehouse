import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
from Python._logging import create_logger


logger = create_logger("positional_encoding")


def positional_encoding(d_model, length=5000):
    """
    :param d_model: dimension of the token
    :param length: (maximum) token number
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe


if __name__ == "__main__":
    length = 10
    d_model = 256
    input_data = torch.randn((2, length, d_model))
    out = input_data + positional_encoding(d_model, length)
    logger.info(f'\n{out}')
