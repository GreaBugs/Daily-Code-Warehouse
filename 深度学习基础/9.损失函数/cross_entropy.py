import numpy as np

def softmax(x):
    # x.shape: (nums, dim)
    expx = np.exp(x-np.max(x, axis=-1, keepdims=True)) # keepdims=True是为了使相减时能广播
    sm = expx / np.sum(expx, axis=-1, keepdims=True)

    return sm # (nums, dim)

def to_onehot(label, n_classes):
    N = label.shape[0] # 样本数
    onehot = np.zeros((N, n_classes))
    for i in  range(N):
        onehot[i][label[i]] = 1 # 给标签维度设1
    return onehot

def cross_entropy(predictions, targets, eps=1e-12):
	# eps: 防止log(0)输出负无穷大，导致后续计算无法进行
    N = predictions.shape[0]
    # 因为targets是one-hot编码，所以相当于只有为1的那个维度会计算
    return -np.sum(targets * np.log(predictions + eps)) / N
