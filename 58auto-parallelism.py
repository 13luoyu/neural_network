import torch
from d2l import torch as d2l

###  在一个多核GPU上可运行

devices = d2l.try_all_gpus()
def run(x):
    return [x.mm(x) for _ in range(50)]

x_gpu1 = torch.rand(size=(400, 400), device=devices[0])
# x_gpu2 = torch.rand(size=(4000, 4000), device=devices[1])
#
# run(x_gpu1)
# run(x_gpu2)  # 预热设备
# torch.cuda.synchronize(devices[0])
# torch.cuda.synchronize(devices[1])
#
# with d2l.Benchmark('GPU1 time'):
#     run(x_gpu1)
#     torch.cuda.synchronize(devices[0])
#
# with d2l.Benchmark('GPU2 time'):
#     run(x_gpu2)
#     torch.cuda.synchronize(devices[1])
#
# # GPU1 time: 0.5072 sec
# # GPU2 time: 0.5005 sec
#
# # 如果我们删除两个任务之间的synchronize语句，系统就可以在两个设备上自动实现并行计算。
# with d2l.Benchmark('GPU1 & GPU2'):
#     run(x_gpu1)
#     run(x_gpu2)
#     torch.cuda.synchronize()

# GPU1 & GPU2: 0.5044 sec


# 并行计算通信
def copy_to_cpu(x, non_blocking=False):
    return [y.to('cpu', non_blocking=non_blocking) for y in x]

with d2l.Benchmark('在GPU1上运行'):
    y = run(x_gpu1)
    torch.cuda.synchronize()

with d2l.Benchmark('复制到CPU'):
    y_cpu = copy_to_cpu(y)
    torch.cuda.synchronize()

# 在GPU1上运行: 0.5073 sec
# 复制到CPU: 2.4357 sec

# 这种方式效率不高。注意到当列表中的其余部分还在计算时，我们可能就已经开始将y的部分复制到CPU了。例如，当我们计算
# 一个小批量的（反传）梯度时。某些参数的梯度将比其他参数的梯度更早可用。因此，在GPU仍在运行时就开始使用
# PCI-Express总线带宽来移动数据对我们是有利的。在PyTorch中，to()和copy_()等函数都允许显式的non_blocking参数，
# 这允许在不需要同步时调用方可以绕过同步。设置non_blocking=True让我们模拟这个场景。
with d2l.Benchmark('在GPU1上运行并复制到CPU'):
    y = run(x_gpu1)
    y_cpu = copy_to_cpu(y, non_blocking=True)
    torch.cuda.synchronize()

# 在GPU1上运行并复制到CPU: 1.8815 sec

