

def Giou(rec1, rec2):
    # 分别是第一个矩形左右上下的坐标
    x1, x2, y1, y2 = rec1
    x3, x4, y3, y4 = rec2
    iou = Iou(rec1, rec2)
    area_C = (max(x1, x2, x3, x4) - min(x1, x2, x3, x4)) * (max(y1, y2, y3, y4) - min(y1, y2, y3, y4))
    area_1 = (x2 - x1) * (y1 - y2)
    area_2 = (x4 - x3) * (y3 - y4)
    sum_area = area_1 + area_2

    W1 = x2 - x1  # 第一个矩形的宽
    W2 = x4 - x3  # 第二个矩形的宽
    h1 = y1 - y2
    h2 = y3 - y4

    W = min(x1, x2, x3, x4) + W1 + W2 - max(x1, x2, x3, x4)  # 交叉部分的宽
    H = min(y1, y2, y3, y4) + h1 + h2 - max(y1, y2, y3, y4)  # 交叉部分的高
    Area = W * H  # 交叉的面积

    add_area = sum_area - Area  # 两矩形并集的面积
    end_area = (area_C - add_area) / area_C  # 闭包区域中不属于两个框的区域占闭包区域的比重

    giou = iou - end_area
    return giou