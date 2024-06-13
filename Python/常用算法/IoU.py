import torch


def box_iou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    计算两组框的交并比（IoU）。

    参数:
        box1 (Tensor[N, 4]): 第一组框的坐标，形状为[N, 4]，每行表示一个框的左上角和右下角坐标（x1, y1, x2, y2）。
        box2 (Tensor[M, 4]): 第二组框的坐标，形状为[M, 4]，每行表示一个框的左上角和右下角坐标（x1, y1, x2, y2）。
        eps (float): 防止除零错误的小值（默认为1e-7）。

    返回:
        iou (Tensor[N, M]): 包含每对框之间的交并比值的矩阵，形状为[N, M]。
    """
    # 计算交集面积
    (box1_left_top, box1_right_bottom), (box2_left_top, box2_right_bottom) = box1[:, None].chunk(2, dim=2), box2.chunk(
        2, dim=1)
    inter_left_top = torch.max(box1_left_top, box2_left_top)  # (2, 3, 2)
    inter_right_bottom = torch.min(box1_right_bottom, box2_right_bottom)  # (2, 3, 2)
    inter_area = (inter_right_bottom - inter_left_top).clamp(0).prod(dim=2)  # (2, 3)
    # prod(dim=2)表示在第二个维度上进行张量的乘积运算

    # 计算 IoU
    return inter_area / (box_area(box1.T)[:, None] + box_area(box2.T) - inter_area + eps)


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    """
    计算一组框的面积。

    参数:
        boxes (Tensor[N, 4]): 框的坐标，形状为[N, 4]，每行表示一个框的左上角和右下角坐标（x1, y1, x2, y2）。

    返回:
        areas (Tensor[N]): 包含每个框的面积值的张量，形状为[N]。
    """
    return (boxes[2] - boxes[0]) * (boxes[3] - boxes[1])


if __name__ == "__main__":
    bbox1 = torch.tensor([[1, 1, 2, 3],
                          [1, 1, 2, 4]])  # shape:[2, 4]
    bbox2 = torch.tensor([[1, 1, 3, 2],
                          [1, 1, 3, 2],
                          [1, 1, 3, 3]])  # shape:[3, 4]
    print(box_iou(bbox1, bbox2))  # # shape:[2, 3]
