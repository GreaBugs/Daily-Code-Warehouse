import numpy as np
import torch


def bilinear_interpolate_numpy(image, x, y):
    """
    双线性插值函数。

    参数:
    image -- 二维数据数组，即图像。
    x -- 要插值的 x 坐标。
    y -- 要插值的 y 坐标。

    返回:
    插值后的数据点。
    """
    # 获取图像的尺寸
    height, width = image.shape

    # 将坐标转换为整数索引，并找到四个最近的整数坐标点
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    x1, y1 = min(x0 + 1, width - 1), min(y0 + 1, height - 1)

    # 计算插值权重
    dx = x - x0
    dy = y - y0

    # 四个最近点的值
    Ia = image[y0, x0]
    Ib = image[y0, x1]
    Ic = image[y1, x0]
    Id = image[y1, x1]

    # 计算插值结果
    It = (Ia * (1 - dx) * (1 - dy) +
          Ib * dx * (1 - dy) +
          Ic * (1 - dx) * dy +
          Id * dx * dy)

    return It


def bilinear_interpolate_torch(img, x, y):
    """
    使用 PyTorch 实现双线性插值。
    
    参数:
    img -- 四维张量，形状为 (B, C, H, W)，其中 B 是批量大小，C 是通道数，H 是高度，W 是宽度。
    x -- 要插值的 x 坐标，形状为 (B, 1)。
    y -- 要插值的 y 坐标，形状为 (B, 1)。
    
    返回:
    插值后的张量，形状为 (B, C)。
    """
    B, C, H, W = img.shape
    x = x.view(B, 1)
    y = y.view(B, 1)

    # 计算四个最近点的坐标
    x0 = torch.floor(x).long().clamp(0, W - 1)
    x1 = (x0 + 1).clamp(0, W - 1)
    y0 = torch.floor(y).long().clamp(0, H - 1)
    y1 = (y0 + 1).clamp(0, H - 1)

    # 获取四个最近点的值
    Ia = img[:, :, y0, x0]
    Ib = img[:, :, y0, x1]
    Ic = img[:, :, y1, x0]
    Id = img[:, :, y1, x1]

    # 计算插值权重
    wa = (x1.float() - x) * (y1.float() - y)
    wb = (x1.float() - x) * (y - y0.float())
    wc = (x - x0.float()) * (y1.float() - y)
    wd = (x - x0.float()) * (y - y0.float())

    # 计算插值结果
    return wa * Ia + wb * Ib + wc * Ic + wd * Id


if __name__ == "__main__":

    # 示例使用
    # 创建一个简单的二维数组作为图像
    image = np.array([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])

    # 插值点坐标
    x = 1.5
    y = 1.5

    # 调用双线性插值函数
    value = bilinear_interpolate_numpy(image, x, y)
    print(f"插值结果: {value}")

    # 示例使用
    # 创建一个简单的四维张量作为图像
    img = torch.tensor([[[[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]]]])

    # 插值点坐标
    x = torch.tensor([[1.5]])
    y = torch.tensor([[1.5]])

    # 调用双线性插值函数
    value = bilinear_interpolate_torch(img, x, y)
    print(f"插值结果: \n{value}")
