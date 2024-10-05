# 理论笔记：https://cariclpajpr.feishu.cn/wiki/AlbIwVYO3i7BF1kGjqEcZ0KdnEc
import matplotlib.pyplot as plt

max_steps = 15000   # 模拟训练15000步
warmup_steps = 1500   # 前1500步采用 warmup
init_lr = 0.1

lr = []
for train_steps in range(max_steps):
    if train_steps < warmup_steps:
        warmup_percent_done = train_steps / warmup_steps
        learning_rate = init_lr * warmup_percent_done  # gradual warmup_lr
    else:
        learning_rate = learning_rate**1.0001  # 预热学习率结束后, 学习率呈指数衰减
        
    lr.append(learning_rate)

plt.plot(range(max_steps), lr)
plt.xlabel('steps')
plt.ylabel('learning rate')
plt.show()
plt.savefig("output/warm-up.png")