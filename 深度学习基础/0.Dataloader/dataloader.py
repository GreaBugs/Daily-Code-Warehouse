import torch
from torch.utils.data import DataLoader, Dataset


# 定义一个自定义的数据集
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# 自定义的 collate_fn 函数，将一个 batch 中的样本长度对齐并拼接成一个张量
def my_collate_fn(batch):
    batch_tensor = []
    max_legth = 0
    for item in batch:
        if len(item) > max_legth:
            max_legth = len(item)
    for item in batch:
        if len(item) < max_legth:
            # 计算需要填充的数量
            padding_length = max(0, max_legth - len(item))
            # 使用列表切片和重复操作进行填充
            item = item + [0] * padding_length
        batch_tensor.append(torch.tensor(item))

    return torch.tensor(torch.stack(batch_tensor, dim=0))


# 创建一个示例数据集
data = [[1, 2, 3], [4], [6, 7], [8, 9]]
dataset = CustomDataset(data)

# 使用 DataLoader 加载数据集，并指定 collate_fn
dataloader = DataLoader(dataset, batch_size=2, collate_fn=my_collate_fn)

# 遍历 DataLoader，并打印每个 batch
for batch in dataloader:
    print("Batch:", batch)
