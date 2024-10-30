import torch


def generate_test_cases():
    test_cases = []

    # Case 1: Simple overlapping boxes
    boxes1 = [[100, 100, 210, 210], [105, 105, 215, 215], [150, 150, 250, 250]]
    scores1 = [0.9, 0.75, 0.8]
    iou_threshold1 = 0.5
    test_cases.append((boxes1, scores1, iou_threshold1))

    # Case 2: Non-overlapping boxes
    boxes2 = [[10, 10, 20, 20], [30, 30, 40, 40], [50, 50, 60, 60]]
    scores2 = [0.6, 0.8, 0.75]
    iou_threshold2 = 0.5
    test_cases.append((boxes2, scores2, iou_threshold2))

    # Case 3: All overlapping boxes
    boxes3 = [[0, 0, 100, 100], [10, 10, 110, 110], [20, 20, 120, 120]]
    scores3 = [0.9, 0.85, 0.8]
    iou_threshold3 = 0.3
    test_cases.append((boxes3, scores3, iou_threshold3))

    # Case 4: Mixed overlapping and non-overlapping boxes
    boxes4 = [[100, 100, 200, 200], [150, 150, 250, 250], [300, 300, 400, 400]]
    scores4 = [0.9, 0.85, 0.8]
    iou_threshold4 = 0.4
    test_cases.append((boxes4, scores4, iou_threshold4))

    # Case 5: Edge overlapping boxes
    boxes5 = [[100, 100, 200, 200], [200, 200, 300, 300], [400, 400, 500, 500]]
    scores5 = [0.9, 0.85, 0.8]
    iou_threshold5 = 0.5
    test_cases.append((boxes5, scores5, iou_threshold5))

    return test_cases
        

def box_area(boxes: torch.Tensor) -> torch.Tensor:
    """
    计算一组框的面积。

    参数:
        boxes (Tensor[N, 4]): 框的坐标，形状为[N, 4]，每行表示一个框的左上角和右下角坐标（x1, y1, x2, y2）。

    返回:
        areas (Tensor[N]): 包含每个框的面积值的张量，形状为[N]。
    """
    return (boxes[2] - boxes[0]) * (boxes[3] - boxes[1])

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
    (box1_left_top, box1_right_bottom), (box2_left_top, box2_right_bottom) = box1[:, None].chunk(2, dim=2), box2.chunk(2, dim=1)
    inter_left_top = torch.max(box1_left_top, box2_left_top)  # (2, 3, 2)
    inter_right_bottom = torch.min(box1_right_bottom, box2_right_bottom)  # (2, 3, 2)
    inter_area = (inter_right_bottom - inter_left_top).clamp(0).prod(dim=2)  # (2, 3)
    # prod(dim=2)表示在第二个维度上进行张量的乘积运算

    # 计算 IoU
    return inter_area / (box_area(box1.T)[:, None] + box_area(box2.T) - inter_area + eps)


def nms(bbox:list, scores:list, threshold:float) -> list:
    """
    实现非极大值抑制（Non-Maximum Suppression，NMS）。

    参数:
        bbox (Tensor[N, 4]): 框的坐标，形状为[N, 4]，每行表示一个框的左上角和右下角坐标（x1, y1, x2, y2）。
        scores (Tensor[N]): 框的得分，形状为[N]。
        threshold (float): IoU 阈值。

    返回:
        keep (Tensor[K]): 保留的框的索引，形状为[K]。
    """
    if len(bbox) < 1:
        return []
    bbox = torch.tensor(bbox)
    scores = torch.tensor(scores)
    # 按得分降序排列
    order = torch.argsort(scores, descending=True)
    keep = []
    while order.numel() > 0:
        # 保留得分最高的框
        cur_max_idx = order[0].item()
        keep.append(cur_max_idx)
        order = order[1:]
        # 计算剩余的框和保留的框的 IoU
        iou = box_iou(bbox[cur_max_idx][None, :], bbox[order])
        # 去除 IoU 大于阈值的框
        order = order[iou[0] < threshold]
    return keep


def main():
    test_cases = generate_test_cases()
    
    for i, (boxes, scores, iou_threshold) in enumerate(test_cases):
        print(f"Test Case {i+1}:")
        print("Boxes:", boxes)
        print("Scores:", scores)
        print("IoU Threshold:", iou_threshold)
        keep_indices = nms(boxes, scores, iou_threshold)
        print("Boxes to keep:", keep_indices)
        print("Kept boxes:", [boxes[i] for i in keep_indices])
        print()


if __name__ == "__main__":
    main()
    